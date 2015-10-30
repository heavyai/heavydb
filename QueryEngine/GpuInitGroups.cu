#include "GpuInitGroups.h"
#include "GpuRtConstants.h"

extern "C" __device__ void init_group_by_buffer_gpu_impl(int64_t* groups_buffer,
                                                         const int64_t* init_vals,
                                                         const uint32_t groups_buffer_entry_count,
                                                         const uint32_t key_qw_count,
                                                         const uint32_t agg_col_count,
                                                         const bool keyless,
                                                         const int8_t warp_size) {
#ifdef EXECUTOR_RT
  const int32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
#else
  const int32_t start = threadIdx.x;
  const int32_t step = blockDim.x;
#endif
  if (keyless) {
    for (int32_t i = start; i < groups_buffer_entry_count * agg_col_count * static_cast<int32_t>(warp_size);
         i += step) {
      groups_buffer[i] = init_vals[i % agg_col_count];
    }
    __syncthreads();
    return;
  }
  int32_t groups_buffer_entry_qw_count = groups_buffer_entry_count * (key_qw_count + agg_col_count);
  for (int32_t i = start; i < groups_buffer_entry_qw_count; i += step) {
    if (i % (key_qw_count + agg_col_count) < key_qw_count) {
      groups_buffer[i] = EMPTY_KEY;
    } else {
      groups_buffer[i] = init_vals[(i - key_qw_count) % (key_qw_count + agg_col_count)];
    }
  }
  __syncthreads();
}

extern "C" __device__ void init_columnar_group_by_buffer_gpu_impl(int64_t* groups_buffer,
                                                                  const int64_t* init_vals,
                                                                  const uint32_t groups_buffer_entry_count,
                                                                  const uint32_t key_qw_count,
                                                                  const uint32_t agg_col_count,
                                                                  const bool keyless) {
#ifdef EXECUTOR_RT
  const int32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
#else
  const int32_t start = threadIdx.x;
  const int32_t step = blockDim.x;
#endif
  int32_t i = start;
  if (!keyless) {
    for (; i < groups_buffer_entry_count; i += step) {
      groups_buffer[i] = EMPTY_KEY;
    }
    i = groups_buffer_entry_count;
  }
  for (int32_t init_idx = 0; init_idx < agg_col_count; ++init_idx) {
    for (int32_t j = i + start; j < i + groups_buffer_entry_count; j += step) {
      groups_buffer[j] = init_vals[init_idx];
    }
    i += groups_buffer_entry_count;
  }
  __syncthreads();
}

__device__ void init_render_buffer(int64_t* render_buffer, const uint32_t qw_count) {
  const uint32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t step = blockDim.x * gridDim.x;
  for (uint32_t i = start; i < qw_count; i += step) {
    render_buffer[i] = EMPTY_KEY;
  }
}

__global__ void init_render_buffer_wrapper(int64_t* render_buffer, const uint32_t qw_count) {
  init_render_buffer(render_buffer, qw_count);
}

__global__ void init_group_by_buffer_gpu_wrapper(int64_t* groups_buffer,
                                                 const int64_t* init_vals,
                                                 const uint32_t groups_buffer_entry_count,
                                                 const uint32_t key_qw_count,
                                                 const uint32_t agg_col_count,
                                                 const bool keyless,
                                                 const int8_t warp_size) {
  init_group_by_buffer_gpu_impl(
      groups_buffer, init_vals, groups_buffer_entry_count, key_qw_count, agg_col_count, keyless, warp_size);
}

__global__ void init_columnar_group_by_buffer_gpu_wrapper(int64_t* groups_buffer,
                                                          const int64_t* init_vals,
                                                          const uint32_t groups_buffer_entry_count,
                                                          const uint32_t key_qw_count,
                                                          const uint32_t agg_col_count,
                                                          const bool keyless) {
  init_columnar_group_by_buffer_gpu_impl(
      groups_buffer, init_vals, groups_buffer_entry_count, key_qw_count, agg_col_count, keyless);
}

void init_group_by_buffer_on_device(int64_t* groups_buffer,
                                    const int64_t* init_vals,
                                    const uint32_t groups_buffer_entry_count,
                                    const uint32_t key_qw_count,
                                    const uint32_t agg_col_count,
                                    const bool keyless,
                                    const int8_t warp_size,
                                    const size_t block_size_x,
                                    const size_t grid_size_x) {
  init_group_by_buffer_gpu_wrapper<<<grid_size_x, block_size_x>>>
      (groups_buffer, init_vals, groups_buffer_entry_count, key_qw_count, agg_col_count, keyless, warp_size);
}

void init_columnar_group_by_buffer_on_device(int64_t* groups_buffer,
                                             const int64_t* init_vals,
                                             const uint32_t groups_buffer_entry_count,
                                             const uint32_t key_qw_count,
                                             const uint32_t agg_col_count,
                                             const bool keyless,
                                             const size_t block_size_x,
                                             const size_t grid_size_x) {
  init_columnar_group_by_buffer_gpu_wrapper<<<grid_size_x, block_size_x>>>
      (groups_buffer, init_vals, groups_buffer_entry_count, key_qw_count, agg_col_count, keyless);
}

void init_render_buffer_on_device(int64_t* render_buffer,
                                  const uint32_t qw_count,
                                  const size_t block_size_x,
                                  const size_t grid_size_x) {
  init_render_buffer_wrapper<<<grid_size_x, block_size_x>>>(render_buffer, qw_count);
}
