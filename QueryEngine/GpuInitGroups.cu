#include "BufferCompaction.h"
#include "GpuInitGroups.h"
#include "GpuRtConstants.h"

template <typename T>
__device__ int8_t* init_columnar_buffer(T* buffer_ptr,
                                        const T init_val,
                                        const uint32_t entry_count,
                                        const int32_t start,
                                        const int32_t step) {
  for (int32_t i = start; i < entry_count; i += step) {
    buffer_ptr[i] = init_val;
  }
  return reinterpret_cast<int8_t*>(buffer_ptr + entry_count);
}

extern "C" __device__ void init_columnar_group_by_buffer_gpu_impl(int64_t* groups_buffer,
                                                                  const int64_t* init_vals,
                                                                  const uint32_t groups_buffer_entry_count,
                                                                  const uint32_t key_qw_count,
                                                                  const uint32_t agg_col_count,
                                                                  const int8_t* col_sizes,
                                                                  const bool need_padding,
                                                                  const bool keyless,
                                                                  const int8_t key_size) {
  const int32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t step = blockDim.x * gridDim.x;

  int8_t* buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);
  if (!keyless) {
    switch (key_size) {
      case 1:
        buffer_ptr = init_columnar_buffer<int8_t>(buffer_ptr, EMPTY_KEY_8, groups_buffer_entry_count, start, step);
        break;
      case 2:
        buffer_ptr = init_columnar_buffer<int16_t>(
            reinterpret_cast<int16_t*>(buffer_ptr), EMPTY_KEY_16, groups_buffer_entry_count, start, step);
        break;
      case 4:
        buffer_ptr = init_columnar_buffer<int32_t>(
            reinterpret_cast<int32_t*>(buffer_ptr), EMPTY_KEY_32, groups_buffer_entry_count, start, step);
        break;
      case 8:
        buffer_ptr = init_columnar_buffer<int64_t>(
            reinterpret_cast<int64_t*>(buffer_ptr), EMPTY_KEY_64, groups_buffer_entry_count, start, step);
        break;
      default:
        // FIXME(miyu): CUDA linker doesn't accept assertion on GPU yet right now.
        break;
    }
    buffer_ptr = align_to_int64(buffer_ptr);
  }
  for (int32_t i = 0; i < agg_col_count; ++i) {
    if (need_padding) {
      buffer_ptr = align_to_int64(buffer_ptr);
    }
    switch (col_sizes[i]) {
      case 1:
        buffer_ptr = init_columnar_buffer<int8_t>(buffer_ptr, init_vals[i], groups_buffer_entry_count, start, step);
        break;
      case 2:
        buffer_ptr = init_columnar_buffer<int16_t>(
            reinterpret_cast<int16_t*>(buffer_ptr), init_vals[i], groups_buffer_entry_count, start, step);
        break;
      case 4:
        buffer_ptr = init_columnar_buffer<int32_t>(
            reinterpret_cast<int32_t*>(buffer_ptr), init_vals[i], groups_buffer_entry_count, start, step);
        break;
      case 8:
        buffer_ptr = init_columnar_buffer<int64_t>(
            reinterpret_cast<int64_t*>(buffer_ptr), init_vals[i], groups_buffer_entry_count, start, step);
        break;
      default:
        // FIXME(miyu): CUDA linker doesn't accept assertion on GPU yet now.
        break;
    }
  }
  __syncthreads();
}

__device__ void init_render_buffer(int64_t* render_buffer, const uint32_t qw_count) {
  const uint32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t step = blockDim.x * gridDim.x;
  for (uint32_t i = start; i < qw_count; i += step) {
    render_buffer[i] = EMPTY_KEY_64;
  }
}

__global__ void init_render_buffer_wrapper(int64_t* render_buffer, const uint32_t qw_count) {
  init_render_buffer(render_buffer, qw_count);
}

inline __device__ bool try_reset_keys(int64_t* key_ptr_i64,
                                      const uint32_t key_qw_idx,
                                      const uint32_t key_count,
                                      const uint32_t key_width,
                                      const uint32_t row_size_quad) {
  switch (key_width) {
    case 4: {
      auto key_ptr_i32 = reinterpret_cast<int32_t*>(key_ptr_i64);
      if (key_qw_idx * 2 < key_count) {
        key_ptr_i32[0] = EMPTY_KEY_32;
        key_ptr_i32[1] = EMPTY_KEY_32;
        return true;
      } else if (key_qw_idx * 2 == key_count + 1) {
        key_ptr_i32[0] = EMPTY_KEY_32;
        key_ptr_i32[1] = 0;  // padding
        return true;
      }
    } break;
    case 8:
      if (key_qw_idx < key_count) {
        *key_ptr_i64 = EMPTY_KEY_64;
        return true;
      }
      break;
    default:
      // FIXME(miyu): CUDA linker doesn't accept assertion on GPU yet now.
      break;
  }
  return false;
}

__global__ void init_group_by_buffer_gpu(int64_t* groups_buffer,
                                         const int64_t* init_vals,
                                         const uint32_t groups_buffer_entry_count,
                                         const uint32_t key_count,
                                         const uint32_t key_width,
                                         const uint32_t row_size_quad,
                                         const bool keyless,
                                         const int8_t warp_size) {
  const int32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
  if (keyless) {
    for (int32_t i = start; i < groups_buffer_entry_count * row_size_quad * static_cast<int32_t>(warp_size);
         i += step) {
      groups_buffer[i] = init_vals[i % row_size_quad];
    }
    __syncthreads();
    return;
  }

  const int32_t slot_off_quad = align_to_int64(key_count * key_width) / sizeof(int64_t);
  int32_t groups_buffer_entry_qw_count = groups_buffer_entry_count * row_size_quad;
  for (int32_t i = start; i < groups_buffer_entry_qw_count; i += step) {
    if (!try_reset_keys(&groups_buffer[i], (i % row_size_quad), key_count, key_width, row_size_quad)) {
      groups_buffer[i] = init_vals[(i % row_size_quad) - slot_off_quad];
    }
  }
  __syncthreads();
}

__global__ void init_columnar_group_by_buffer_gpu_wrapper(int64_t* groups_buffer,
                                                          const int64_t* init_vals,
                                                          const uint32_t groups_buffer_entry_count,
                                                          const uint32_t key_qw_count,
                                                          const uint32_t agg_col_count,
                                                          const int8_t* col_sizes,
                                                          const bool need_padding,
                                                          const bool keyless,
                                                          const int8_t key_size) {
  init_columnar_group_by_buffer_gpu_impl(groups_buffer,
                                         init_vals,
                                         groups_buffer_entry_count,
                                         key_qw_count,
                                         agg_col_count,
                                         col_sizes,
                                         need_padding,
                                         keyless,
                                         key_size);
}

void init_group_by_buffer_on_device(int64_t* groups_buffer,
                                    const int64_t* init_vals,
                                    const uint32_t groups_buffer_entry_count,
                                    const uint32_t key_count,
                                    const uint32_t key_width,
                                    const uint32_t row_size_quad,
                                    const bool keyless,
                                    const int8_t warp_size,
                                    const size_t block_size_x,
                                    const size_t grid_size_x) {
  init_group_by_buffer_gpu<<<grid_size_x, block_size_x>>>(
      groups_buffer, init_vals, groups_buffer_entry_count, key_count, key_width, row_size_quad, keyless, warp_size);
}

void init_columnar_group_by_buffer_on_device(int64_t* groups_buffer,
                                             const int64_t* init_vals,
                                             const uint32_t groups_buffer_entry_count,
                                             const uint32_t key_qw_count,
                                             const uint32_t agg_col_count,
                                             const int8_t* col_sizes,
                                             const bool need_padding,
                                             const bool keyless,
                                             const int8_t key_size,
                                             const size_t block_size_x,
                                             const size_t grid_size_x) {
  init_columnar_group_by_buffer_gpu_wrapper<<<grid_size_x, block_size_x>>>(groups_buffer,
                                                                           init_vals,
                                                                           groups_buffer_entry_count,
                                                                           key_qw_count,
                                                                           agg_col_count,
                                                                           col_sizes,
                                                                           need_padding,
                                                                           keyless,
                                                                           key_size);
}

void init_render_buffer_on_device(int64_t* render_buffer,
                                  const uint32_t qw_count,
                                  const size_t block_size_x,
                                  const size_t grid_size_x) {
  init_render_buffer_wrapper<<<grid_size_x, block_size_x>>>(render_buffer, qw_count);
}
