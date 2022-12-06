#include <cuda.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <limits>
#include "BufferCompaction.h"
#include "ExtensionFunctions.hpp"
#include "GpuRtConstants.h"
#include "HyperLogLogRank.h"

#if CUDA_VERSION < 10000
static_assert(false, "CUDA v10.0 or later is required.");
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350)
static_assert(false, "CUDA Compute Capability of 3.5 or greater is required.");
#endif

extern "C" __device__ int64_t get_thread_index() {
  return threadIdx.x;
}

extern "C" __device__ int64_t get_block_index() {
  return blockIdx.x;
}

extern "C" __device__ int32_t pos_start_impl(const int32_t* row_index_resume) {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

extern "C" __device__ int32_t group_buff_idx_impl() {
  return pos_start_impl(NULL);
}

extern "C" __device__ int32_t pos_step_impl() {
  return blockDim.x * gridDim.x;
}

extern "C" __device__ int8_t thread_warp_idx(const int8_t warp_sz) {
  return threadIdx.x % warp_sz;
}

extern "C" __device__ const int64_t* init_shared_mem_nop(
    const int64_t* groups_buffer,
    const int32_t groups_buffer_size) {
  return groups_buffer;
}

extern "C" __device__ void write_back_nop(int64_t* dest, int64_t* src, const int32_t sz) {
}

/*
 * Just declares and returns a dynamic shared memory pointer. Total size should be
 * properly set during kernel launch
 */
extern "C" __device__ int64_t* declare_dynamic_shared_memory() {
  extern __shared__ int64_t shared_mem_buffer[];
  return shared_mem_buffer;
}

/**
 * Initializes the shared memory buffer for perfect hash group by.
 * In this function, we simply copy the global group by buffer (already initialized on the
 * host and transferred) to all shared memory group by buffers.
 */
extern "C" __device__ const int64_t* init_shared_mem(const int64_t* global_groups_buffer,
                                                     const int32_t groups_buffer_size) {
  // dynamic shared memory declaration
  extern __shared__ int64_t shared_groups_buffer[];

  // it is assumed that buffer size is aligned with 64-bit units
  // so it is safe to assign 64-bit to each thread
  const int32_t buffer_units = groups_buffer_size >> 3;

  for (int32_t pos = threadIdx.x; pos < buffer_units; pos += blockDim.x) {
    shared_groups_buffer[pos] = global_groups_buffer[pos];
  }
  __syncthreads();
  return shared_groups_buffer;
}

#define init_group_by_buffer_gpu_impl init_group_by_buffer_gpu

#include "GpuInitGroups.cu"

#undef init_group_by_buffer_gpu_impl

// Dynamic watchdog: monitoring up to 64 SMs. E.g. GP100 config may have 60:
// 6 Graphics Processing Clusters (GPCs) * 10 Streaming Multiprocessors
// TODO(Saman): move these into a kernel parameter, allocated and initialized through CUDA
__device__ int64_t dw_sm_cycle_start[128];  // Set from host before launching the kernel
// TODO(Saman): make this cycle budget something constant in codegen level
__device__ int64_t dw_cycle_budget = 0;  // Set from host before launching the kernel
__device__ int32_t dw_abort = 0;         // TBD: set from host (async)
__device__ int32_t runtime_interrupt_flag = 0;

__inline__ __device__ uint32_t get_smid(void) {
  uint32_t ret;
  asm("mov.u32 %0, %%smid;" : "=r"(ret));
  return ret;
}

/*
 * The main objective of this function is to return true, if any of the following two
 * scenarios happen:
 * 1. receives a host request for aborting the kernel execution
 * 2. kernel execution takes longer clock cycles than it was initially allowed
 * The assumption is that all (or none) threads within a block return true for the
 * watchdog, and the first thread within each block compares the recorded clock cycles for
 * its occupying SM with the allowed budget. It also assumes that all threads entering
 * this function are active (no critical edge exposure)
 * NOTE: dw_cycle_budget, dw_abort, and dw_sm_cycle_start[] are all variables in global
 * memory scope.
 */
extern "C" __device__ bool dynamic_watchdog() {
  // check for dynamic watchdog, if triggered all threads return true
  if (dw_cycle_budget == 0LL) {
    return false;  // Uninitialized watchdog can't check time
  }
  if (dw_abort == 1) {
    return true;  // Received host request to abort
  }
  uint32_t smid = get_smid();
  if (smid >= 128) {
    return false;
  }
  __shared__ volatile int64_t dw_block_cycle_start;  // Thread block shared cycle start
  __shared__ volatile bool
      dw_should_terminate;  // all threads within a block should return together if
                            // watchdog criteria is met

  // thread 0 either initializes or read the initial clock cycle, the result is stored
  // into shared memory. Since all threads wihtin a block shares the same SM, there's no
  // point in using more threads here.
  if (threadIdx.x == 0) {
    dw_block_cycle_start = 0LL;
    int64_t cycle_count = static_cast<int64_t>(clock64());
    // Make sure the block hasn't switched SMs
    if (smid == get_smid()) {
      dw_block_cycle_start = static_cast<int64_t>(
          atomicCAS(reinterpret_cast<unsigned long long*>(&dw_sm_cycle_start[smid]),
                    0ULL,
                    static_cast<unsigned long long>(cycle_count)));
    }

    int64_t cycles = cycle_count - dw_block_cycle_start;
    if ((smid == get_smid()) && (dw_block_cycle_start > 0LL) &&
        (cycles > dw_cycle_budget)) {
      // Check if we're out of time on this particular SM
      dw_should_terminate = true;
    } else {
      dw_should_terminate = false;
    }
  }
  __syncthreads();
  return dw_should_terminate;
}

extern "C" __device__ bool check_interrupt() {
  return (runtime_interrupt_flag == 1) ? true : false;
}

template <typename T = unsigned long long>
inline __device__ T get_empty_key() {
  return EMPTY_KEY_64;
}

template <>
inline __device__ unsigned int get_empty_key() {
  return EMPTY_KEY_32;
}

template <typename T>
inline __device__ int64_t* get_matching_group_value(int64_t* groups_buffer,
                                                    const uint32_t h,
                                                    const T* key,
                                                    const uint32_t key_count,
                                                    const uint32_t row_size_quad) {
  const T empty_key = get_empty_key<T>();
  uint32_t off = h * row_size_quad;
  auto row_ptr = reinterpret_cast<T*>(groups_buffer + off);
  {
    const T old = atomicCAS(row_ptr, empty_key, *key);
    if (empty_key == old && key_count > 1) {
      for (size_t i = 1; i <= key_count - 1; ++i) {
        atomicExch(row_ptr + i, key[i]);
      }
    }
  }
  if (key_count > 1) {
    while (atomicAdd(row_ptr + key_count - 1, 0) == empty_key) {
      // spin until the winning thread has finished writing the entire key and the init
      // value
    }
  }
  bool match = true;
  for (uint32_t i = 0; i < key_count; ++i) {
    if (row_ptr[i] != key[i]) {
      match = false;
      break;
    }
  }

  if (match) {
    auto row_ptr_i8 = reinterpret_cast<int8_t*>(row_ptr + key_count);
    return reinterpret_cast<int64_t*>(align_to_int64(row_ptr_i8));
  }
  return NULL;
}

extern "C" __device__ int64_t* get_matching_group_value(int64_t* groups_buffer,
                                                        const uint32_t h,
                                                        const int64_t* key,
                                                        const uint32_t key_count,
                                                        const uint32_t key_width,
                                                        const uint32_t row_size_quad) {
  switch (key_width) {
    case 4:
      return get_matching_group_value(groups_buffer,
                                      h,
                                      reinterpret_cast<const unsigned int*>(key),
                                      key_count,
                                      row_size_quad);
    case 8:
      return get_matching_group_value(groups_buffer,
                                      h,
                                      reinterpret_cast<const unsigned long long*>(key),
                                      key_count,
                                      row_size_quad);
    default:
      return NULL;
  }
}

template <typename T>
__device__ int32_t get_matching_group_value_columnar_slot(int64_t* groups_buffer,
                                                          const uint32_t entry_count,
                                                          const uint32_t h,
                                                          const T* key,
                                                          const uint32_t key_count) {
  const T empty_key = get_empty_key<T>();
  const uint64_t old =
      atomicCAS(reinterpret_cast<T*>(groups_buffer + h), empty_key, *key);
  // the winner thread proceeds with writing the rest fo the keys
  if (old == empty_key) {
    uint32_t offset = h + entry_count;
    for (size_t i = 1; i < key_count; ++i) {
      *reinterpret_cast<T*>(groups_buffer + offset) = key[i];
      offset += entry_count;
    }
  }

  __threadfence();
  // for all threads except the winning thread, memory content of the keys
  // related to the hash offset are checked again. In case of a complete match
  // the hash offset is returned, otherwise -1 is returned
  if (old != empty_key) {
    uint32_t offset = h;
    for (uint32_t i = 0; i < key_count; ++i) {
      if (*reinterpret_cast<T*>(groups_buffer + offset) != key[i]) {
        return -1;
      }
      offset += entry_count;
    }
  }
  return h;
}

extern "C" __device__ int32_t
get_matching_group_value_columnar_slot(int64_t* groups_buffer,
                                       const uint32_t entry_count,
                                       const uint32_t h,
                                       const int64_t* key,
                                       const uint32_t key_count,
                                       const uint32_t key_width) {
  switch (key_width) {
    case 4:
      return get_matching_group_value_columnar_slot(
          groups_buffer,
          entry_count,
          h,
          reinterpret_cast<const unsigned int*>(key),
          key_count);
    case 8:
      return get_matching_group_value_columnar_slot(
          groups_buffer,
          entry_count,
          h,
          reinterpret_cast<const unsigned long long*>(key),
          key_count);
    default:
      return -1;
  }
}

extern "C" __device__ int64_t* get_matching_group_value_columnar(
    int64_t* groups_buffer,
    const uint32_t h,
    const int64_t* key,
    const uint32_t key_qw_count,
    const size_t entry_count) {
  uint32_t off = h;
  {
    const uint64_t old = atomicCAS(
        reinterpret_cast<unsigned long long*>(groups_buffer + off), EMPTY_KEY_64, *key);
    if (EMPTY_KEY_64 == old) {
      for (size_t i = 0; i < key_qw_count; ++i) {
        groups_buffer[off] = key[i];
        off += entry_count;
      }
      return &groups_buffer[off];
    }
  }
  __syncthreads();
  off = h;
  for (size_t i = 0; i < key_qw_count; ++i) {
    if (groups_buffer[off] != key[i]) {
      return NULL;
    }
    off += entry_count;
  }
  return &groups_buffer[off];
}

#include "GroupByRuntime.cpp"
#include "JoinHashTable/Runtime/JoinHashTableQueryRuntime.cpp"
#include "MurmurHash.cpp"
#include "TopKRuntime.cpp"

__device__ int64_t atomicMax64(int64_t* address, int64_t val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, max((long long)val, (long long)assumed));
  } while (assumed != old);

  return old;
}

__device__ int64_t atomicMin64(int64_t* address, int64_t val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, min((long long)val, (long long)assumed));
  } while (assumed != old);

  return old;
}

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600)
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

__device__ double atomicMax(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(max(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ float atomicMax(float* address, float val) {
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;

  do {
    assumed = old;
    old = atomicCAS(
        address_as_int, assumed, __float_as_int(max(val, __int_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

__device__ double atomicMin(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(min(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ double atomicMin(float* address, float val) {
  int* address_as_ull = (int*)address;
  int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed, __float_as_int(min(val, __int_as_float(assumed))));
  } while (assumed != old);

  return __int_as_float(old);
}

extern "C" __device__ uint64_t agg_count_shared(uint64_t* agg, const int64_t val) {
  return static_cast<uint64_t>(atomicAdd(reinterpret_cast<uint32_t*>(agg), 1U));
}

extern "C" __device__ uint64_t agg_count_if_shared(uint64_t* agg, const int64_t cond) {
  return cond ? static_cast<uint64_t>(atomicAdd(reinterpret_cast<uint32_t*>(agg), 1U))
              : static_cast<uint64_t>(*(reinterpret_cast<uint32_t*>(agg)));
}

extern "C" __device__ uint32_t agg_count_int32_shared(uint32_t* agg, const int32_t val) {
  return atomicAdd(agg, 1U);
}

extern "C" __device__ uint32_t agg_count_if_int32_shared(uint32_t* agg,
                                                         const int32_t cond) {
  return cond ? atomicAdd(agg, 1U) : *agg;
}

extern "C" __device__ uint64_t agg_count_double_shared(uint64_t* agg, const double val) {
  return agg_count_shared(agg, val);
}

extern "C" __device__ uint32_t agg_count_float_shared(uint32_t* agg, const float val) {
  return agg_count_int32_shared(agg, val);
}

extern "C" __device__ int64_t agg_sum_shared(int64_t* agg, const int64_t val) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(agg), val);
}

extern "C" __device__ int32_t agg_sum_int32_shared(int32_t* agg, const int32_t val) {
  return atomicAdd(agg, val);
}

extern "C" __device__ void agg_sum_float_shared(int32_t* agg, const float val) {
  atomicAdd(reinterpret_cast<float*>(agg), val);
}

extern "C" __device__ void agg_sum_double_shared(int64_t* agg, const double val) {
  atomicAdd(reinterpret_cast<double*>(agg), val);
}

extern "C" __device__ void agg_max_shared(int64_t* agg, const int64_t val) {
  atomicMax64(agg, val);
}

extern "C" __device__ void agg_max_int32_shared(int32_t* agg, const int32_t val) {
  atomicMax(agg, val);
}

extern "C" __device__ void agg_max_double_shared(int64_t* agg, const double val) {
  atomicMax(reinterpret_cast<double*>(agg), val);
}

extern "C" __device__ void agg_max_float_shared(int32_t* agg, const float val) {
  atomicMax(reinterpret_cast<float*>(agg), val);
}

extern "C" __device__ void agg_min_shared(int64_t* agg, const int64_t val) {
  atomicMin64(agg, val);
}

extern "C" __device__ void agg_min_int32_shared(int32_t* agg, const int32_t val) {
  atomicMin(agg, val);
}

#if CUDA_VERSION > 10000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__device__ void atomicMax16(int16_t* agg, const int16_t val) {
  unsigned short int* address_as_us = reinterpret_cast<unsigned short int*>(agg);
  unsigned short int old = *address_as_us, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_us,
                    assumed,
                    static_cast<unsigned short>(max(static_cast<short int>(val),
                                                    static_cast<short int>(assumed))));
  } while (assumed != old);
}
#else
__device__ void atomicMax16(int16_t* agg, const int16_t val) {
  // properly align the input pointer:
  unsigned int* base_address_u32 =
      reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(agg) & ~0x3);

  unsigned int old_value = *base_address_u32;
  unsigned int swap_value, compare_value;
  do {
    compare_value = old_value;
    swap_value =
        (reinterpret_cast<size_t>(agg) & 0x2)
            ? static_cast<unsigned int>(max(static_cast<int16_t>(old_value >> 16), val))
                      << 16 |
                  (old_value & 0xFFFF)
            : (old_value & 0xFFFF0000) |
                  static_cast<unsigned int>(
                      max(static_cast<int16_t>(old_value & 0xFFFF), val));
    old_value = atomicCAS(base_address_u32, compare_value, swap_value);
  } while (old_value != compare_value);
}
#endif

__device__ void atomicMax8(int8_t* agg, const int8_t val) {
  // properly align the input pointer:
  unsigned int* base_address_u32 =
      reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(agg) & ~0x3);

  // __byte_perm(unsigned int A, unsigned int B, unsigned int s):
  // if s == 0x3214 returns {A[31..24], A[23..16], A[15..8], B[7..0]}
  // if s == 0x3240 returns {A[31..24], A[23..16], B[7...0], A[7..0]}
  // if s == 0x3410 returns {A[31..24], B[7....0], A[15..8], A[7..0]}
  // if s == 0x4210 returns {B[7....0], A[23..16], A[15..8], A[7..0]}
  constexpr unsigned int byte_permutations[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int old_value = *base_address_u32;
  unsigned int swap_value, compare_value;
  do {
    compare_value = old_value;
    auto max_value = static_cast<unsigned int>(
        // compare val with its corresponding bits in the compare_value
        max(val,
            static_cast<int8_t>(__byte_perm(
                compare_value, 0, (reinterpret_cast<size_t>(agg) & 0x3) | 0x4440))));
    swap_value = __byte_perm(
        compare_value, max_value, byte_permutations[reinterpret_cast<size_t>(agg) & 0x3]);
    old_value = atomicCAS(base_address_u32, compare_value, swap_value);
  } while (compare_value != old_value);
}

#if CUDA_VERSION > 10000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__device__ void atomicMin16(int16_t* agg, const int16_t val) {
  unsigned short int* address_as_us = reinterpret_cast<unsigned short int*>(agg);
  unsigned short int old = *address_as_us, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_us,
                    assumed,
                    static_cast<unsigned short>(min(static_cast<short int>(val),
                                                    static_cast<short int>(assumed))));
  } while (assumed != old);
}
#else
__device__ void atomicMin16(int16_t* agg, const int16_t val) {
  // properly align the input pointer:
  unsigned int* base_address_u32 =
      reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(agg) & ~0x3);

  unsigned int old_value = *base_address_u32;
  unsigned int swap_value, compare_value;
  do {
    compare_value = old_value;
    swap_value =
        (reinterpret_cast<size_t>(agg) & 0x2)
            ? static_cast<unsigned int>(min(static_cast<int16_t>(old_value >> 16), val))
                      << 16 |
                  (old_value & 0xFFFF)
            : (old_value & 0xFFFF0000) |
                  static_cast<unsigned int>(
                      min(static_cast<int16_t>(old_value & 0xFFFF), val));
    old_value = atomicCAS(base_address_u32, compare_value, swap_value);
  } while (old_value != compare_value);
}
#endif

__device__ void atomicMin16SkipVal(int16_t* agg,
                                   const int16_t val,
                                   const int16_t skip_val) {
  // properly align the input pointer:
  unsigned int* base_address_u32 =
      reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(agg) & ~0x3);

  unsigned int old_value = *base_address_u32;
  unsigned int swap_value, compare_value;
  do {
    compare_value = old_value;
    int16_t selected_old_val = (reinterpret_cast<size_t>(agg) & 0x2)
                                   ? static_cast<int16_t>(old_value >> 16)
                                   : static_cast<int16_t>(old_value & 0xFFFF);

    swap_value =
        (reinterpret_cast<size_t>(agg) & 0x2)
            ? static_cast<unsigned int>(
                  selected_old_val == skip_val ? val : min(selected_old_val, val))
                      << 16 |
                  (old_value & 0xFFFF)
            : (old_value & 0xFFFF0000) |
                  static_cast<unsigned int>(
                      selected_old_val == skip_val ? val : min(selected_old_val, val));
    old_value = atomicCAS(base_address_u32, compare_value, swap_value);
  } while (old_value != compare_value);
}

__device__ void atomicMin8(int8_t* agg, const int8_t val) {
  // properly align the input pointer:
  unsigned int* base_address_u32 =
      reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(agg) & ~0x3);

  constexpr unsigned int byte_permutations[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int old_value = *base_address_u32;
  unsigned int swap_value, compare_value;
  do {
    compare_value = old_value;
    auto min_value = static_cast<unsigned int>(
        min(val,
            static_cast<int8_t>(__byte_perm(
                compare_value, 0, (reinterpret_cast<size_t>(agg) & 0x3) | 0x4440))));
    swap_value = __byte_perm(
        compare_value, min_value, byte_permutations[reinterpret_cast<size_t>(agg) & 0x3]);
    old_value = atomicCAS(base_address_u32, compare_value, swap_value);
  } while (compare_value != old_value);
}

__device__ void atomicMin8SkipVal(int8_t* agg, const int8_t val, const int8_t skip_val) {
  // properly align the input pointer:
  unsigned int* base_address_u32 =
      reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(agg) & ~0x3);

  constexpr unsigned int byte_permutations[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int old_value = *base_address_u32;
  unsigned int swap_value, compare_value;
  do {
    compare_value = old_value;
    int8_t selected_old_val = static_cast<int8_t>(
        __byte_perm(compare_value, 0, (reinterpret_cast<size_t>(agg) & 0x3) | 0x4440));
    auto min_value = static_cast<unsigned int>(
        selected_old_val == skip_val ? val : min(val, selected_old_val));
    swap_value = __byte_perm(
        compare_value, min_value, byte_permutations[reinterpret_cast<size_t>(agg) & 0x3]);
    old_value = atomicCAS(base_address_u32, compare_value, swap_value);
  } while (compare_value != old_value);
}

extern "C" __device__ void agg_max_int16_shared(int16_t* agg, const int16_t val) {
  return atomicMax16(agg, val);
}

extern "C" __device__ void agg_max_int8_shared(int8_t* agg, const int8_t val) {
  return atomicMax8(agg, val);
}

extern "C" __device__ void agg_min_int16_shared(int16_t* agg, const int16_t val) {
  return atomicMin16(agg, val);
}

extern "C" __device__ void agg_min_int8_shared(int8_t* agg, const int8_t val) {
  return atomicMin8(agg, val);
}

extern "C" __device__ void agg_min_double_shared(int64_t* agg, const double val) {
  atomicMin(reinterpret_cast<double*>(agg), val);
}

extern "C" __device__ void agg_min_float_shared(int32_t* agg, const float val) {
  atomicMin(reinterpret_cast<float*>(agg), val);
}

extern "C" __device__ void agg_id_shared(int64_t* agg, const int64_t val) {
  *agg = val;
}

extern "C" __device__ int8_t* agg_id_varlen_shared(int8_t* varlen_buffer,
                                                   const int64_t offset,
                                                   const int8_t* value,
                                                   const int64_t size_bytes) {
  for (auto i = 0; i < size_bytes; i++) {
    varlen_buffer[offset + i] = value[i];
  }
  return &varlen_buffer[offset];
}

extern "C" __device__ int32_t checked_single_agg_id_shared(int64_t* agg,
                                                           const int64_t val,
                                                           const int64_t null_val) {
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(agg);
  unsigned long long int old = *address_as_ull, assumed;

  if (val == null_val) {
    return 0;
  }

  do {
    if (static_cast<int64_t>(old) != null_val) {
      if (static_cast<int64_t>(old) != val) {
        // see Execute::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES
        return 15;
      } else {
        break;
      }
    }

    assumed = old;
    old = atomicCAS(address_as_ull, assumed, val);
  } while (assumed != old);

  return 0;
}

#define DEF_AGG_ID_INT_SHARED(n)                                            \
  extern "C" __device__ void agg_id_int##n##_shared(int##n##_t* agg,        \
                                                    const int##n##_t val) { \
    *agg = val;                                                             \
  }

DEF_AGG_ID_INT_SHARED(32)
DEF_AGG_ID_INT_SHARED(16)
DEF_AGG_ID_INT_SHARED(8)

#undef DEF_AGG_ID_INT_SHARED

extern "C" __device__ void agg_id_double_shared(int64_t* agg, const double val) {
  *agg = *(reinterpret_cast<const int64_t*>(&val));
}

extern "C" __device__ int32_t checked_single_agg_id_double_shared(int64_t* agg,
                                                                  const double val,
                                                                  const double null_val) {
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(agg);
  unsigned long long int old = *address_as_ull, assumed;

  if (val == null_val) {
    return 0;
  }

  do {
    if (static_cast<int64_t>(old) != __double_as_longlong(null_val)) {
      if (static_cast<int64_t>(old) != __double_as_longlong(val)) {
        // see Execute::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES
        return 15;
      } else {
        break;
      }
    }

    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return 0;
}

extern "C" __device__ void agg_id_double_shared_slow(int64_t* agg, const double* val) {
  *agg = *(reinterpret_cast<const int64_t*>(val));
}

extern "C" __device__ int32_t
checked_single_agg_id_double_shared_slow(int64_t* agg,
                                         const double* valp,
                                         const double null_val) {
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(agg);
  unsigned long long int old = *address_as_ull, assumed;
  double val = *valp;

  if (val == null_val) {
    return 0;
  }

  do {
    if (static_cast<int64_t>(old) != __double_as_longlong(null_val)) {
      if (static_cast<int64_t>(old) != __double_as_longlong(val)) {
        // see Execute::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES
        return 15;
      } else {
        break;
      }
    }

    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return 0;
}

extern "C" __device__ void agg_id_float_shared(int32_t* agg, const float val) {
  *agg = __float_as_int(val);
}

extern "C" __device__ int32_t checked_single_agg_id_float_shared(int32_t* agg,
                                                                 const float val,
                                                                 const float null_val) {
  int* address_as_ull = reinterpret_cast<int*>(agg);
  int old = *address_as_ull, assumed;

  if (val == null_val) {
    return 0;
  }

  do {
    if (old != __float_as_int(null_val)) {
      if (old != __float_as_int(val)) {
        // see Execute::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES
        return 15;
      } else {
        break;
      }
    }

    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __float_as_int(val));
  } while (assumed != old);

  return 0;
}

#define DEF_SKIP_AGG(base_agg_func)                             \
  extern "C" __device__ ADDR_T base_agg_func##_skip_val_shared( \
      ADDR_T* agg, const DATA_T val, const DATA_T skip_val) {   \
    if (val != skip_val) {                                      \
      return base_agg_func##_shared(agg, val);                  \
    }                                                           \
    return 0;                                                   \
  }

#define DATA_T int64_t
#define ADDR_T uint64_t
DEF_SKIP_AGG(agg_count)
DEF_SKIP_AGG(agg_count_if)
#undef DATA_T
#undef ADDR_T

#define DATA_T int32_t
#define ADDR_T uint32_t
DEF_SKIP_AGG(agg_count_int32)
DEF_SKIP_AGG(agg_count_if_int32)
#undef DATA_T
#undef ADDR_T

// Initial value for nullable column is INT32_MIN
extern "C" __device__ void agg_max_int32_skip_val_shared(int32_t* agg,
                                                         const int32_t val,
                                                         const int32_t skip_val) {
  if (val != skip_val) {
    agg_max_int32_shared(agg, val);
  }
}

extern "C" __device__ void agg_max_int16_skip_val_shared(int16_t* agg,
                                                         const int16_t val,
                                                         const int16_t skip_val) {
  if (val != skip_val) {
    agg_max_int16_shared(agg, val);
  }
}

extern "C" __device__ void agg_min_int16_skip_val_shared(int16_t* agg,
                                                         const int16_t val,
                                                         const int16_t skip_val) {
  if (val != skip_val) {
    atomicMin16SkipVal(agg, val, skip_val);
  }
}

extern "C" __device__ void agg_max_int8_skip_val_shared(int8_t* agg,
                                                        const int8_t val,
                                                        const int8_t skip_val) {
  if (val != skip_val) {
    agg_max_int8_shared(agg, val);
  }
}

extern "C" __device__ void agg_min_int8_skip_val_shared(int8_t* agg,
                                                        const int8_t val,
                                                        const int8_t skip_val) {
  if (val != skip_val) {
    atomicMin8SkipVal(agg, val, skip_val);
  }
}

__device__ int32_t atomicMin32SkipVal(int32_t* address,
                                      int32_t val,
                                      const int32_t skip_val) {
  int32_t old = atomicExch(address, INT_MAX);
  return atomicMin(address, old == skip_val ? val : min(old, val));
}

extern "C" __device__ void agg_min_int32_skip_val_shared(int32_t* agg,
                                                         const int32_t val,
                                                         const int32_t skip_val) {
  if (val != skip_val) {
    atomicMin32SkipVal(agg, val, skip_val);
  }
}

__device__ int32_t atomicSum32SkipVal(int32_t* address,
                                      const int32_t val,
                                      const int32_t skip_val) {
  unsigned int* address_as_int = (unsigned int*)address;
  int32_t old = atomicExch(address_as_int, 0);
  int32_t old2 = atomicAdd(address_as_int, old == skip_val ? val : (val + old));
  return old == skip_val ? old2 : (old2 + old);
}

extern "C" __device__ int32_t agg_sum_int32_skip_val_shared(int32_t* agg,
                                                            const int32_t val,
                                                            const int32_t skip_val) {
  if (val != skip_val) {
    const int32_t old = atomicSum32SkipVal(agg, val, skip_val);
    return old;
  }
  return 0;
}

__device__ int64_t atomicSum64SkipVal(int64_t* address,
                                      const int64_t val,
                                      const int64_t skip_val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  int64_t old = atomicExch(address_as_ull, 0);
  int64_t old2 = atomicAdd(address_as_ull, old == skip_val ? val : (val + old));
  return old == skip_val ? old2 : (old2 + old);
}

extern "C" __device__ int64_t agg_sum_skip_val_shared(int64_t* agg,
                                                      const int64_t val,
                                                      const int64_t skip_val) {
  if (val != skip_val) {
    return atomicSum64SkipVal(agg, val, skip_val);
  }
  return 0;
}

__device__ int64_t atomicMin64SkipVal(int64_t* address,
                                      int64_t val,
                                      const int64_t skip_val) {
  unsigned long long int* address_as_ull =
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    assumed == skip_val ? val : min((long long)val, (long long)assumed));
  } while (assumed != old);

  return old;
}

extern "C" __device__ void agg_min_skip_val_shared(int64_t* agg,
                                                   const int64_t val,
                                                   const int64_t skip_val) {
  if (val != skip_val) {
    atomicMin64SkipVal(agg, val, skip_val);
  }
}

__device__ int64_t atomicMax64SkipVal(int64_t* address,
                                      int64_t val,
                                      const int64_t skip_val) {
  unsigned long long int* address_as_ull =
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    assumed == skip_val ? val : max((long long)val, (long long)assumed));
  } while (assumed != old);

  return old;
}

extern "C" __device__ void agg_max_skip_val_shared(int64_t* agg,
                                                   const int64_t val,
                                                   const int64_t skip_val) {
  if (val != skip_val) {
    atomicMax64SkipVal(agg, val, skip_val);
  }
}

#undef DEF_SKIP_AGG
#define DEF_SKIP_AGG(base_agg_func)                             \
  extern "C" __device__ ADDR_T base_agg_func##_skip_val_shared( \
      ADDR_T* agg, const DATA_T val, const DATA_T skip_val) {   \
    if (val != skip_val) {                                      \
      return base_agg_func##_shared(agg, val);                  \
    }                                                           \
    return *agg;                                                \
  }

#define DATA_T double
#define ADDR_T uint64_t
DEF_SKIP_AGG(agg_count_double)
#undef ADDR_T
#undef DATA_T

#define DATA_T float
#define ADDR_T uint32_t
DEF_SKIP_AGG(agg_count_float)
#undef ADDR_T
#undef DATA_T

// Initial value for nullable column is FLOAT_MIN
extern "C" __device__ void agg_max_float_skip_val_shared(int32_t* agg,
                                                         const float val,
                                                         const float skip_val) {
  if (__float_as_int(val) != __float_as_int(skip_val)) {
    float old = atomicExch(reinterpret_cast<float*>(agg), -FLT_MAX);
    atomicMax(reinterpret_cast<float*>(agg),
              __float_as_int(old) == __float_as_int(skip_val) ? val : fmaxf(old, val));
  }
}

__device__ float atomicMinFltSkipVal(int32_t* address, float val, const float skip_val) {
  float old = atomicExch(reinterpret_cast<float*>(address), FLT_MAX);
  return atomicMin(
      reinterpret_cast<float*>(address),
      __float_as_int(old) == __float_as_int(skip_val) ? val : fminf(old, val));
}

extern "C" __device__ void agg_min_float_skip_val_shared(int32_t* agg,
                                                         const float val,
                                                         const float skip_val) {
  if (__float_as_int(val) != __float_as_int(skip_val)) {
    atomicMinFltSkipVal(agg, val, skip_val);
  }
}

__device__ void atomicSumFltSkipVal(float* address,
                                    const float val,
                                    const float skip_val) {
  float old = atomicExch(address, 0.f);
  atomicAdd(address, __float_as_int(old) == __float_as_int(skip_val) ? val : (val + old));
}

extern "C" __device__ void agg_sum_float_skip_val_shared(int32_t* agg,
                                                         const float val,
                                                         const float skip_val) {
  if (__float_as_int(val) != __float_as_int(skip_val)) {
    atomicSumFltSkipVal(reinterpret_cast<float*>(agg), val, skip_val);
  }
}

__device__ void atomicSumDblSkipVal(double* address,
                                    const double val,
                                    const double skip_val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  double old = __longlong_as_double(atomicExch(address_as_ull, __double_as_longlong(0.)));
  atomicAdd(
      address,
      __double_as_longlong(old) == __double_as_longlong(skip_val) ? val : (val + old));
}

extern "C" __device__ void agg_sum_double_skip_val_shared(int64_t* agg,
                                                          const double val,
                                                          const double skip_val) {
  if (__double_as_longlong(val) != __double_as_longlong(skip_val)) {
    atomicSumDblSkipVal(reinterpret_cast<double*>(agg), val, skip_val);
  }
}

__device__ double atomicMinDblSkipVal(double* address,
                                      double val,
                                      const double skip_val) {
  unsigned long long int* address_as_ull =
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int skip_val_as_ull =
      *reinterpret_cast<const unsigned long long*>(&skip_val);
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    assumed == skip_val_as_ull
                        ? *reinterpret_cast<unsigned long long*>(&val)
                        : __double_as_longlong(min(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

extern "C" __device__ void agg_min_double_skip_val_shared(int64_t* agg,
                                                          const double val,
                                                          const double skip_val) {
  if (val != skip_val) {
    atomicMinDblSkipVal(reinterpret_cast<double*>(agg), val, skip_val);
  }
}

extern "C" __device__ void agg_max_double_skip_val_shared(int64_t* agg,
                                                          const double val,
                                                          const double skip_val) {
  if (__double_as_longlong(val) != __double_as_longlong(skip_val)) {
    double old = __longlong_as_double(atomicExch(
        reinterpret_cast<unsigned long long int*>(agg), __double_as_longlong(-DBL_MAX)));
    atomicMax(reinterpret_cast<double*>(agg),
              __double_as_longlong(old) == __double_as_longlong(skip_val)
                  ? val
                  : fmax(old, val));
  }
}

#undef DEF_SKIP_AGG

extern "C" __device__ bool slotEmptyKeyCAS(int64_t* slot,
                                           int64_t new_val,
                                           int64_t init_val) {
  auto slot_address = reinterpret_cast<unsigned long long int*>(slot);
  const auto empty_key =
      static_cast<unsigned long long int*>(static_cast<void*>(&init_val));
  const auto new_val_cast =
      static_cast<unsigned long long int*>(static_cast<void*>(&new_val));

  const auto old_val = atomicCAS(slot_address, *empty_key, *new_val_cast);
  if (old_val == *empty_key) {
    return true;
  } else {
    return false;
  }
}

extern "C" __device__ bool slotEmptyKeyCAS_int32(int32_t* slot,
                                                 int32_t new_val,
                                                 int32_t init_val) {
  unsigned int* slot_address = reinterpret_cast<unsigned int*>(slot);
  unsigned int compare_value = static_cast<unsigned int>(init_val);
  unsigned int swap_value = static_cast<unsigned int>(new_val);

  const unsigned int old_value = atomicCAS(slot_address, compare_value, swap_value);
  return old_value == compare_value;
}

extern "C" __device__ bool slotEmptyKeyCAS_int16(int16_t* slot,
                                                 int16_t new_val,
                                                 int16_t init_val) {
  unsigned int* base_slot_address =
      reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(slot) & ~0x3);
  unsigned int old_value = *base_slot_address;
  unsigned int swap_value, compare_value;
  do {
    compare_value = old_value;
    // exit criteria: if init_val does not exist in the slot (some other thread has
    // succeeded)
    if (static_cast<unsigned int>(init_val) !=
        __byte_perm(
            compare_value, 0, (reinterpret_cast<size_t>(slot) & 0x2 ? 0x3244 : 0x4410))) {
      return false;
    }
    swap_value = __byte_perm(compare_value,
                             static_cast<unsigned int>(new_val),
                             (reinterpret_cast<size_t>(slot) & 0x2) ? 0x5410 : 0x3254);
    old_value = atomicCAS(base_slot_address, compare_value, swap_value);
  } while (compare_value != old_value);
  return true;
}

extern "C" __device__ bool slotEmptyKeyCAS_int8(int8_t* slot,
                                                int8_t new_val,
                                                int8_t init_val) {
  // properly align the slot address:
  unsigned int* base_slot_address =
      reinterpret_cast<unsigned int*>(reinterpret_cast<size_t>(slot) & ~0x3);
  constexpr unsigned int byte_permutations[] = {0x3214, 0x3240, 0x3410, 0x4210};
  unsigned int old_value = *base_slot_address;
  unsigned int swap_value, compare_value;
  do {
    compare_value = old_value;
    // exit criteria: if init_val does not exist in the slot (some other thread has
    // succeeded)
    if (static_cast<unsigned int>(init_val) !=
        __byte_perm(compare_value, 0, (reinterpret_cast<size_t>(slot) & 0x3) | 0x4440)) {
      return false;
    }
    swap_value = __byte_perm(compare_value,
                             static_cast<unsigned int>(new_val),
                             byte_permutations[reinterpret_cast<size_t>(slot) & 0x3]);
    old_value = atomicCAS(base_slot_address, compare_value, swap_value);
  } while (compare_value != old_value);
  return true;
}

#include "../Utils/ChunkIter.cpp"
#include "DateTruncate.cpp"
#include "ExtractFromTime.cpp"
#define EXECUTE_INCLUDE
#include "ArrayOps.cpp"
#include "DateAdd.cpp"
#include "GeoOps.cpp"
#include "StringFunctions.cpp"
#undef EXECUTE_INCLUDE
#include "../Utils/Regexp.cpp"
#include "../Utils/StringLike.cpp"

extern "C" __device__ uint64_t string_decode(int8_t* chunk_iter_, int64_t pos) {
  // TODO(alex): de-dup, the x64 version is basically identical
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  VarlenDatum vd;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, pos, false, &vd, &is_end);
  return vd.is_null ? 0
                    : (reinterpret_cast<uint64_t>(vd.pointer) & 0xffffffffffff) |
                          (static_cast<uint64_t>(vd.length) << 48);
}

extern "C" __device__ void linear_probabilistic_count(uint8_t* bitmap,
                                                      const uint32_t bitmap_bytes,
                                                      const uint8_t* key_bytes,
                                                      const uint32_t key_len) {
  const uint32_t bit_pos = MurmurHash3(key_bytes, key_len, 0) % (bitmap_bytes * 8);
  const uint32_t word_idx = bit_pos / 32;
  const uint32_t bit_idx = bit_pos % 32;
  atomicOr(((uint32_t*)bitmap) + word_idx, 1 << bit_idx);
}

extern "C" __device__ void agg_count_distinct_bitmap_gpu(int64_t* agg,
                                                         const int64_t val,
                                                         const int64_t min_val,
                                                         const int64_t base_dev_addr,
                                                         const int64_t base_host_addr,
                                                         const uint64_t sub_bitmap_count,
                                                         const uint64_t bitmap_bytes) {
  const uint64_t bitmap_idx = val - min_val;
  const uint32_t byte_idx = bitmap_idx >> 3;
  const uint32_t word_idx = byte_idx >> 2;
  const uint32_t byte_word_idx = byte_idx & 3;
  const int64_t host_addr = *agg;
  uint32_t* bitmap = (uint32_t*)(base_dev_addr + host_addr - base_host_addr +
                                 (threadIdx.x & (sub_bitmap_count - 1)) * bitmap_bytes);
  switch (byte_word_idx) {
    case 0:
      atomicOr(&bitmap[word_idx], 1 << (bitmap_idx & 7));
      break;
    case 1:
      atomicOr(&bitmap[word_idx], 1 << ((bitmap_idx & 7) + 8));
      break;
    case 2:
      atomicOr(&bitmap[word_idx], 1 << ((bitmap_idx & 7) + 16));
      break;
    case 3:
      atomicOr(&bitmap[word_idx], 1 << ((bitmap_idx & 7) + 24));
      break;
    default:
      break;
  }
}

extern "C" __device__ void agg_count_distinct_bitmap_skip_val_gpu(
    int64_t* agg,
    const int64_t val,
    const int64_t min_val,
    const int64_t skip_val,
    const int64_t base_dev_addr,
    const int64_t base_host_addr,
    const uint64_t sub_bitmap_count,
    const uint64_t bitmap_bytes) {
  if (val != skip_val) {
    agg_count_distinct_bitmap_gpu(
        agg, val, min_val, base_dev_addr, base_host_addr, sub_bitmap_count, bitmap_bytes);
  }
}

extern "C" __device__ void agg_approximate_count_distinct_gpu(
    int64_t* agg,
    const int64_t key,
    const uint32_t b,
    const int64_t base_dev_addr,
    const int64_t base_host_addr) {
  const uint64_t hash = MurmurHash64A(&key, sizeof(key), 0);
  const uint32_t index = hash >> (64 - b);
  const int32_t rank = get_rank(hash << b, 64 - b);
  const int64_t host_addr = *agg;
  int32_t* M = (int32_t*)(base_dev_addr + host_addr - base_host_addr);
  atomicMax(&M[index], rank);
}

extern "C" __device__ void force_sync() {
  __threadfence_block();
}

extern "C" __device__ void sync_warp() {
  __syncwarp();
}

/**
 * Protected warp synchornization to make sure all (or none) threads within a warp go
 * through a synchronization barrier. thread_pos: the current thread position to be used
 * for a memory access row_count: maximum number of rows to be processed The function
 * performs warp sync iff all 32 threads within that warp will process valid data NOTE:
 * it currently assumes that warp size is 32.
 */
extern "C" __device__ void sync_warp_protected(int64_t thread_pos, int64_t row_count) {
  // only syncing if NOT within the same warp as those threads experiencing the critical
  // edge
  if ((((row_count - 1) | 0x1F) - thread_pos) >= 32) {
    __syncwarp();
  }
}

extern "C" __device__ void sync_threadblock() {
  __syncthreads();
}

/*
 * Currently, we just use this function for handling non-grouped aggregates
 * with COUNT queries (with GPU shared memory used). Later, we should generate code for
 * this depending on the type of aggregate functions.
 * TODO: we should use one contiguous global memory buffer, rather than current default
 * behaviour of multiple buffers, each for one aggregate. Once that's resolved, we can
 * do much cleaner than this function
 */
extern "C" __device__ void write_back_non_grouped_agg(int64_t* input_buffer,
                                                      int64_t* output_buffer,
                                                      const int32_t agg_idx) {
  if (threadIdx.x == agg_idx) {
    agg_sum_shared(output_buffer, input_buffer[agg_idx]);
  }
}
