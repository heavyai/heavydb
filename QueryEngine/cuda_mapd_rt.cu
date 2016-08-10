#include <stdint.h>
#include <cuda.h>
#include <limits>
#include "ExtensionFunctions.hpp"
#include "GpuRtConstants.h"

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

extern "C" __device__ const int64_t* init_shared_mem_nop(const int64_t* groups_buffer,
                                                         const int32_t groups_buffer_size) {
  return groups_buffer;
}

extern "C" __device__ void write_back_nop(int64_t* dest, int64_t* src, const int32_t sz) {
}

extern "C" __device__ const int64_t* init_shared_mem(const int64_t* groups_buffer, const int32_t groups_buffer_size) {
  extern __shared__ int64_t fast_bins[];
  if (threadIdx.x == 0) {
    memcpy(fast_bins, groups_buffer, groups_buffer_size);
  }
  __syncthreads();
  return fast_bins;
}

extern "C" __device__ void write_back(int64_t* dest, int64_t* src, const int32_t sz) {
  __syncthreads();
  if (threadIdx.x == 0) {
    memcpy(dest, src, sz);
  }
}

#define init_group_by_buffer_gpu_impl init_group_by_buffer_gpu

#include "GpuInitGroups.cu"

#undef init_group_by_buffer_gpu_impl

extern "C" __device__ int64_t* get_matching_group_value(int64_t* groups_buffer,
                                                        const uint32_t h,
                                                        const int64_t* key,
                                                        const uint32_t key_qw_count,
                                                        const uint32_t row_size_quad,
                                                        const int64_t* init_vals) {
  uint32_t off = h * row_size_quad;
  {
    const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(groups_buffer + off), EMPTY_KEY_64, *key);
    if (EMPTY_KEY_64 == old) {
      memcpy(groups_buffer + off, key, key_qw_count * sizeof(int64_t));
      memcpy(groups_buffer + off + key_qw_count, init_vals, (row_size_quad - key_qw_count) * sizeof(int64_t));
    }
  }
  __syncthreads();
  bool match = true;
  for (uint32_t i = 0; i < key_qw_count; ++i) {
    if (groups_buffer[off + i] != key[i]) {
      match = false;
      break;
    }
  }
  return match ? groups_buffer + off + key_qw_count : NULL;
}

extern "C" __device__ int64_t* get_matching_group_value_columnar(int64_t* groups_buffer,
                                                                 const uint32_t h,
                                                                 const int64_t* key,
                                                                 const uint32_t key_qw_count,
                                                                 const size_t entry_count) {
  uint32_t off = h;
  {
    const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(groups_buffer + off), EMPTY_KEY_64, *key);
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

// As of 20160418, CUDA 8.0EA only defines `atomicAdd(double*, double)` for compute capability >= 6.0.
#if CUDA_VERSION < 8000 || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600)
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

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
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(max(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ float atomicMax(float* address, float val) {
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(max(val, __int_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

__device__ double atomicMin(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(min(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ double atomicMin(float* address, float val) {
  int* address_as_ull = (int*)address;
  int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __float_as_int(min(val, __int_as_float(assumed))));
  } while (assumed != old);

  return __int_as_float(old);
}

extern "C" __device__ uint64_t agg_count_shared(uint64_t* agg, const int64_t val) {
  return static_cast<uint64_t>(atomicAdd(reinterpret_cast<uint32_t*>(agg), 1UL));
}

extern "C" __device__ uint32_t agg_count_int32_shared(uint32_t* agg, const int32_t val) {
  return atomicAdd(agg, 1UL);
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

extern "C" __device__ void agg_min_double_shared(int64_t* agg, const double val) {
  atomicMin(reinterpret_cast<double*>(agg), val);
}

extern "C" __device__ void agg_min_float_shared(int32_t* agg, const float val) {
  atomicMin(reinterpret_cast<float*>(agg), val);
}

extern "C" __device__ void agg_id_shared(int64_t* agg, const int64_t val) {
  *agg = val;
}

extern "C" __device__ void agg_id_int32_shared(int32_t* agg, const int32_t val) {
  *agg = val;
}

extern "C" __device__ void agg_id_double_shared(int64_t* agg, const double val) {
  *agg = *(reinterpret_cast<const int64_t*>(&val));
}

extern "C" __device__ void agg_id_double_shared_slow(int64_t* agg, const double* val) {
  *agg = *(reinterpret_cast<const int64_t*>(val));
}

extern "C" __device__ void agg_id_float_shared(int32_t* agg, const float val) {
  *agg = __float_as_int(val);
}

#define DEF_SKIP_AGG(base_agg_func)                                                                                    \
  extern "C" __device__ ADDR_T base_agg_func##_skip_val_shared(ADDR_T* agg, const DATA_T val, const DATA_T skip_val) { \
    if (val != skip_val) {                                                                                             \
      return base_agg_func##_shared(agg, val);                                                                         \
    }                                                                                                                  \
    return 0;                                                                                                          \
  }

#define DATA_T int64_t
#define ADDR_T uint64_t
DEF_SKIP_AGG(agg_count)
#undef DATA_T
#undef ADDR_T

#define DATA_T int32_t
#define ADDR_T uint32_t
DEF_SKIP_AGG(agg_count_int32)
#undef DATA_T
#undef ADDR_T

// Initial value for nullable column is INT32_MIN
extern "C" __device__ void agg_max_int32_skip_val_shared(int32_t* agg, const int32_t val, const int32_t skip_val) {
  if (val != skip_val) {
    agg_max_int32_shared(agg, val);
  }
}

__device__ int32_t atomicMin32SkipVal(int32_t* address, int32_t val, const int32_t skip_val) {
  int32_t old = *address, assumed;

  do {
    assumed = old;
    old = atomicCAS(address, assumed, assumed == skip_val ? val : min(val, assumed));
  } while (assumed != old);

  return old;
}

extern "C" __device__ void agg_min_int32_skip_val_shared(int32_t* agg, const int32_t val, const int32_t skip_val) {
  if (val != skip_val) {
    atomicMin32SkipVal(agg, val, skip_val);
  }
}

__device__ int32_t atomicSum32SkipVal(int32_t* address, const int32_t val, const int32_t skip_val) {
  unsigned int* address_as_int = (unsigned int*)address;
  int32_t old = atomicExch(address_as_int, 0);
  int32_t old2 = atomicAdd(address_as_int, old == skip_val ? val : (val + old));
  return old == skip_val ? old2 : (old2 + old);
}

extern "C" __device__ int32_t agg_sum_int32_skip_val_shared(int32_t* agg, const int32_t val, const int32_t skip_val) {
  if (val != skip_val) {
    const int32_t old = atomicSum32SkipVal(agg, val, skip_val);
    return old;
  }
  return 0;
}

__device__ int64_t atomicSum64SkipVal(int64_t* address, const int64_t val, const int64_t skip_val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  int64_t old = atomicExch(address_as_ull, 0);
  int64_t old2 = atomicAdd(address_as_ull, old == skip_val ? val : (val + old));
  return old == skip_val ? old2 : (old2 + old);
}

extern "C" __device__ int64_t agg_sum_skip_val_shared(int64_t* agg, const int64_t val, const int64_t skip_val) {
  if (val != skip_val) {
    return atomicSum64SkipVal(agg, val, skip_val);
  }
  return 0;
}

__device__ int64_t atomicMin64SkipVal(int64_t* address, int64_t val, const int64_t skip_val) {
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, assumed == skip_val ? val : min((long long)val, (long long)assumed));
  } while (assumed != old);

  return old;
}

extern "C" __device__ void agg_min_skip_val_shared(int64_t* agg, const int64_t val, const int64_t skip_val) {
  if (val != skip_val) {
    atomicMin64SkipVal(agg, val, skip_val);
  }
}

__device__ int64_t atomicMax64SkipVal(int64_t* address, int64_t val, const int64_t skip_val) {
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, assumed == skip_val ? val : max((long long)val, (long long)assumed));
  } while (assumed != old);

  return old;
}

extern "C" __device__ void agg_max_skip_val_shared(int64_t* agg, const int64_t val, const int64_t skip_val) {
  if (val != skip_val) {
    atomicMax64SkipVal(agg, val, skip_val);
  }
}

#undef DEF_SKIP_AGG
#define DEF_SKIP_AGG(base_agg_func)                                                                                    \
  extern "C" __device__ ADDR_T base_agg_func##_skip_val_shared(ADDR_T* agg, const DATA_T val, const DATA_T skip_val) { \
    if (val != skip_val) {                                                                                             \
      return base_agg_func##_shared(agg, val);                                                                         \
    }                                                                                                                  \
    return *agg;                                                                                                       \
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
extern "C" __device__ void agg_max_float_skip_val_shared(int32_t* agg, const float val, const float skip_val) {
  if (val != skip_val) {
    agg_max_float_shared(agg, val);
  }
}

__device__ double atomicMinFltSkipVal(int32_t* address, float val, const float skip_val) {
  int32_t old = *address;
  int32_t skip_val_as_int = __float_as_int(skip_val);
  int32_t assumed;

  do {
    assumed = old;
    old =
        atomicCAS(address,
                  assumed,
                  assumed == skip_val_as_int ? __float_as_int(val) : __float_as_int(min(val, __int_as_float(assumed))));
  } while (assumed != old);

  return __float_as_int(old);
}

extern "C" __device__ void agg_min_float_skip_val_shared(int32_t* agg, const float val, const float skip_val) {
  if (val != skip_val) {
    atomicMinFltSkipVal(agg, val, skip_val);
  }
}

__device__ void atomicSumFltSkipVal(float* address, const float val, const float skip_val) {
  unsigned int* address_as_int = (unsigned*)address;
  int32_t old = atomicExch(address_as_int, __float_as_int(0.));
  atomicAdd(address_as_int, __float_as_int(old == __float_as_int(skip_val) ? val : (val + __int_as_float(old))));
}

extern "C" __device__ void agg_sum_float_skip_val_shared(int32_t* agg, const float val, const float skip_val) {
  if (val != skip_val) {
    atomicSumFltSkipVal(reinterpret_cast<float*>(agg), val, skip_val);
  }
}

__device__ void atomicSumDblSkipVal(double* address, const double val, const double skip_val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  double old = __longlong_as_double(atomicExch(address_as_ull, __double_as_longlong(0.)));
  atomicAdd(address_as_ull, __double_as_longlong(old == skip_val ? val : (val + old)));
}

extern "C" __device__ void agg_sum_double_skip_val_shared(int64_t* agg, const double val, const double skip_val) {
  if (val != skip_val) {
    atomicSumDblSkipVal(reinterpret_cast<double*>(agg), val, skip_val);
  }
}

__device__ double atomicMinDblSkipVal(double* address, double val, const double skip_val) {
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int skip_val_as_ull = *reinterpret_cast<const unsigned long long*>(&skip_val);
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    assumed == skip_val_as_ull ? *reinterpret_cast<unsigned long long*>(&val)
                                               : __double_as_longlong(min(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

extern "C" __device__ void agg_min_double_skip_val_shared(int64_t* agg, const double val, const double skip_val) {
  if (val != skip_val) {
    atomicMinDblSkipVal(reinterpret_cast<double*>(agg), val, skip_val);
  }
}

__device__ double atomicMaxDblSkipVal(double* address, double val, const double skip_val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int skip_val_as_ull = *((unsigned long long int*)&skip_val);
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    assumed == skip_val_as_ull ? *((unsigned long long int*)&val)
                                               : __double_as_longlong(max(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

extern "C" __device__ void agg_max_double_skip_val_shared(int64_t* agg, const double val, const double skip_val) {
  if (val != skip_val) {
    atomicMaxDblSkipVal(reinterpret_cast<double*>(agg), val, skip_val);
  }
}

#undef DEF_SKIP_AGG

#include "ExtractFromTime.cpp"
#include "DateTruncate.cpp"
#include "../Utils/ChunkIter.cpp"
#define EXECUTE_INCLUDE
#include "ArrayOps.cpp"
#include "StringFunctions.cpp"
#undef EXECUTE_INCLUDE
#include "../Utils/StringLike.cpp"

extern "C" __device__ uint64_t string_decode(int8_t* chunk_iter_, int64_t pos) {
  // TODO(alex): de-dup, the x64 version is basically identical
  ChunkIter* chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  VarlenDatum vd;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, pos, false, &vd, &is_end);
  return vd.is_null ? 0 : (reinterpret_cast<uint64_t>(vd.pointer) & 0xffffffffffff) |
                              (static_cast<uint64_t>(vd.length) << 48);
}

extern "C" __device__ void force_sync() {
  __threadfence_block();
}
