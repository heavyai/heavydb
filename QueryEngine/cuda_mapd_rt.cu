#include <stdint.h>
#include <limits>

extern "C"
__device__ int32_t pos_start_impl() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

extern "C"
__device__ int32_t pos_step_impl() {
  return blockDim.x * gridDim.x;
}

#define EMPTY_KEY -9223372036854775808L

extern "C" __attribute__((noinline))
__device__ int64_t* get_matching_group_value(int64_t* groups_buffer,
                                  const int32_t h,
                                  const int64_t* key,
                                  const int32_t key_qw_count,
                                  const int32_t agg_col_count) {
  int64_t off = h * (key_qw_count + agg_col_count);
  if (groups_buffer[off] == EMPTY_KEY) {
    memcpy(groups_buffer + off, key, key_qw_count * sizeof(*key));
    return groups_buffer + off + key_qw_count;
  }
  bool match = true;
  for (int64_t i = 0; i < key_qw_count; ++i) {
    if (groups_buffer[off + i] != key[i]) {
      match = false;
      break;
    }
  }
  return match ? groups_buffer + off + key_qw_count : NULL;
}

extern "C" __attribute__((noinline))
__device__ int32_t key_hash(const int64_t* key, const int32_t key_qw_count,
                            const int32_t groups_buffer_entry_count) {
  int32_t hash = 0;
  for (int32_t i = 0; i < key_qw_count; ++i) {
    hash = ((hash << 5) - hash + key[i]) % groups_buffer_entry_count;
  }
  return hash;
}

extern "C" __attribute__((noinline))
__device__ int64_t* get_group_value(int64_t* groups_buffer,
                                    const int32_t groups_buffer_entry_count,
                                    const int64_t* key,
                                    const int32_t key_qw_count,
                                    const int32_t agg_col_count) {
  int64_t h = key_hash(key, key_qw_count, groups_buffer_entry_count);
  int64_t* matching_group = get_matching_group_value(groups_buffer, h, key, key_qw_count, agg_col_count);
  if (matching_group) {
    return matching_group;
  }
  int64_t h_probe = h + 1;
  while (h_probe != h) {
    matching_group = get_matching_group_value(groups_buffer, h_probe, key, key_qw_count, agg_col_count);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  // TODO(alex): handle error by resizing?
  return NULL;
}
