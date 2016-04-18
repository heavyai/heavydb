#include <stdint.h>
#include <cuda.h>
#include <limits>
#include "GpuRtConstants.h"

extern "C" __device__ int32_t pos_start_impl(const int32_t* row_index_resume) {
  return blockIdx.x * blockDim.x + threadIdx.x + (row_index_resume ? row_index_resume[blockIdx.x] : 0);
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
                                                        const uint32_t agg_col_count,
                                                        const int64_t* init_vals) {
  uint32_t off = h * (key_qw_count + agg_col_count);
  {
    const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(groups_buffer + off), EMPTY_KEY, *key);
    if (EMPTY_KEY == old) {
      memcpy(groups_buffer + off, key, key_qw_count * sizeof(*key));
      memcpy(groups_buffer + off + key_qw_count, init_vals, agg_col_count * sizeof(*init_vals));
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

__device__ double atomicMin(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(min(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

extern "C" __device__ void agg_count_shared(int64_t* agg, const int64_t val) {
  atomicAdd(reinterpret_cast<int32_t*>(agg), 1L);
}

extern "C" __device__ void agg_count_double_shared(int64_t* agg, const double val) {
  agg_count_shared(agg, val);
}

extern "C" __device__ void agg_sum_shared(int64_t* agg, const int64_t val) {
  atomicAdd(reinterpret_cast<unsigned long long*>(agg), val);
}

extern "C" __device__ void agg_sum_double_shared(int64_t* agg, const double val) {
  atomicAdd(reinterpret_cast<double*>(agg), val);
}

extern "C" __device__ void agg_max_shared(int64_t* agg, const int64_t val) {
  atomicMax64(agg, val);
}

extern "C" __device__ void agg_max_double_shared(int64_t* agg, const double val) {
  atomicMax(reinterpret_cast<double*>(agg), val);
}

extern "C" __device__ void agg_min_shared(int64_t* agg, const int64_t val) {
  atomicMin64(agg, val);
}

extern "C" __device__ void agg_min_double_shared(int64_t* agg, const double val) {
  atomicMin(reinterpret_cast<double*>(agg), val);
}

extern "C" __device__ void agg_id_shared(int64_t* agg, const int64_t val) {
  *agg = val;
}

extern "C" __device__ void agg_id_double_shared(int64_t* agg, const double val) {
  *agg = *(reinterpret_cast<const int64_t*>(&val));
}

#define DEF_SKIP_AGG(base_agg_func)                              \
  extern "C" __device__ void base_agg_func##_skip_val_shared(    \
      int64_t* agg, const int64_t val, const int64_t skip_val) { \
    if (val != skip_val) {                                       \
      base_agg_func##_shared(agg, val);                          \
    }                                                            \
  }

DEF_SKIP_AGG(agg_count)
DEF_SKIP_AGG(agg_sum)

__device__ int64_t atomicMin64SkipVal(int64_t* address, int64_t val, const int64_t skip_val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
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
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
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

#define DEF_SKIP_AGG(base_agg_func)                                                                                   \
  extern "C" __device__ void base_agg_func##_skip_val_shared(int64_t* agg, const double val, const double skip_val) { \
    if (val != skip_val) {                                                                                            \
      base_agg_func##_shared(agg, val);                                                                               \
    }                                                                                                                 \
  }

DEF_SKIP_AGG(agg_count_double)
DEF_SKIP_AGG(agg_sum_double)

__device__ double atomicMinDblSkipVal(double* address, double val, const double skip_val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int skip_val_as_ull = *((unsigned long long int*)&skip_val);
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    assumed == skip_val_as_ull ? *((unsigned long long int*)&val)
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

extern "C" __device__ int32_t merge_error_code(const int32_t err_code, int32_t* merged_err_code) {
  if (err_code) {
    int32_t assumed = *merged_err_code;
    int32_t old;
    do {
      old = atomicCAS(merged_err_code, assumed, err_code);
    } while (old != assumed);
  }
  __syncthreads();
  return *merged_err_code;
}
