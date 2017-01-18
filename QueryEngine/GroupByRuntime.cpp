#include "JoinHashImpl.h"
#include "MurmurHash.h"

extern "C" ALWAYS_INLINE DEVICE uint32_t key_hash(const int64_t* key, const uint32_t key_qw_count) {
  return MurmurHash1(key, 8 * key_qw_count, 0);
}

template <bool run_with_watchdog>
NEVER_INLINE DEVICE int64_t* get_group_value_impl(int64_t* groups_buffer,
                                                  const uint32_t groups_buffer_entry_count,
                                                  const int64_t* key,
                                                  const uint32_t key_qw_count,
                                                  const uint32_t row_size_quad,
                                                  const int64_t* init_vals) {
  uint32_t h = key_hash(key, key_qw_count) % groups_buffer_entry_count;
  int64_t* matching_group = get_matching_group_value(groups_buffer, h, key, key_qw_count, row_size_quad, init_vals);
  if (matching_group) {
    return matching_group;
  }
  uint32_t watchdog_countdown = 100;
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group = get_matching_group_value(groups_buffer, h_probe, key, key_qw_count, row_size_quad, init_vals);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
    if (run_with_watchdog) {
      if (--watchdog_countdown == 0) {
        if (dynamic_watchdog(0LL)) {
          return NULL;
        }
        watchdog_countdown = 100;
      }
    }
  }
  return NULL;
}

extern "C" ALWAYS_INLINE DEVICE int64_t* get_group_value(int64_t* groups_buffer,
                                                         const uint32_t groups_buffer_entry_count,
                                                         const int64_t* key,
                                                         const uint32_t key_qw_count,
                                                         const uint32_t row_size_quad,
                                                         const int64_t* init_vals,
                                                         const int8_t run_with_watchdog) {
  if (run_with_watchdog) {
    return get_group_value_impl<true>(
        groups_buffer, groups_buffer_entry_count, key, key_qw_count, row_size_quad, init_vals);
  }
  return get_group_value_impl<false>(
      groups_buffer, groups_buffer_entry_count, key, key_qw_count, row_size_quad, init_vals);
}

template <bool run_with_watchdog>
NEVER_INLINE DEVICE int64_t* get_group_value_columnar_impl(int64_t* groups_buffer,
                                                           const uint32_t groups_buffer_entry_count,
                                                           const int64_t* key,
                                                           const uint32_t key_qw_count) {
  uint32_t h = key_hash(key, key_qw_count) % groups_buffer_entry_count;
  int64_t* matching_group =
      get_matching_group_value_columnar(groups_buffer, h, key, key_qw_count, groups_buffer_entry_count);
  if (matching_group) {
    return matching_group;
  }
  uint32_t watchdog_countdown = 100;
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group =
        get_matching_group_value_columnar(groups_buffer, h_probe, key, key_qw_count, groups_buffer_entry_count);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
    if (run_with_watchdog) {
      if (--watchdog_countdown == 0) {
        if (dynamic_watchdog(0LL)) {
          return NULL;
        }
        watchdog_countdown = 100;
      }
    }
  }
  return NULL;
}

extern "C" ALWAYS_INLINE DEVICE int64_t* get_group_value_columnar(int64_t* groups_buffer,
                                                                  const uint32_t groups_buffer_entry_count,
                                                                  const int64_t* key,
                                                                  const uint32_t key_qw_count,
                                                                  const int8_t run_with_watchdog) {
  if (run_with_watchdog) {
    return get_group_value_columnar_impl<true>(groups_buffer, groups_buffer_entry_count, key, key_qw_count);
  }
  return get_group_value_columnar_impl<false>(groups_buffer, groups_buffer_entry_count, key, key_qw_count);
}

extern "C" ALWAYS_INLINE DEVICE int64_t* get_group_value_fast(int64_t* groups_buffer,
                                                              const int64_t key,
                                                              const int64_t min_key,
                                                              const int64_t bucket,
                                                              const uint32_t row_size_quad) {
  int64_t key_diff = key - min_key;
  if (bucket) {
    key_diff /= bucket;
  }
  int64_t off = key_diff * row_size_quad;
  if (groups_buffer[off] == EMPTY_KEY_64) {
    groups_buffer[off] = key;
  }
  return groups_buffer + off + 1;
}

extern "C" ALWAYS_INLINE DEVICE uint32_t get_columnar_group_bin_offset(int64_t* key_base_ptr,
                                                                       const int64_t key,
                                                                       const int64_t min_key,
                                                                       const int64_t bucket) {
  int64_t off = key - min_key;
  if (bucket) {
    off /= bucket;
  }
  if (key_base_ptr[off] == EMPTY_KEY_64) {
    key_base_ptr[off] = key;
  }
  return off;
}

extern "C" ALWAYS_INLINE DEVICE int64_t* get_group_value_one_key(int64_t* groups_buffer,
                                                                 const uint32_t groups_buffer_entry_count,
                                                                 int64_t* small_groups_buffer,
                                                                 const uint32_t small_groups_buffer_qw_count,
                                                                 const int64_t key,
                                                                 const int64_t min_key,
                                                                 const uint32_t row_size_quad,
                                                                 const int64_t* init_vals,
                                                                 const int8_t run_with_watchdog) {
  int64_t off = key - min_key;
  if (0 <= off && off < small_groups_buffer_qw_count) {
    return get_group_value_fast(small_groups_buffer, key, min_key, 0, row_size_quad);
  }
  return get_group_value(
      groups_buffer, groups_buffer_entry_count, &key, 1, row_size_quad, init_vals, run_with_watchdog);
}

extern "C" ALWAYS_INLINE DEVICE int64_t* get_scan_output_slot(int64_t* output_buffer,
                                                              const uint32_t output_buffer_entry_count,
                                                              const uint32_t pos,
                                                              const uint32_t row_size_quad) {
  uint64_t off = static_cast<uint64_t>(pos) * static_cast<uint64_t>(row_size_quad);
  if (pos < output_buffer_entry_count) {
    output_buffer[off] = pos;
    return output_buffer + off + 1;
  }
  return NULL;
}

extern "C" ALWAYS_INLINE DEVICE int64_t hash_join_idx(int64_t hash_buff,
                                                      const int64_t key,
                                                      const int64_t min_key,
                                                      const int64_t max_key) {
  if (key >= min_key && key <= max_key) {
    return *SUFFIX(get_hash_slot)(reinterpret_cast<int32_t*>(hash_buff), key, min_key);
  }
  return -1;
}

extern "C" ALWAYS_INLINE DEVICE int64_t hash_join_idx_nullable(int64_t hash_buff,
                                                               const int64_t key,
                                                               const int64_t min_key,
                                                               const int64_t max_key,
                                                               const int64_t null_val) {
  if (key != null_val) {
    return hash_join_idx(hash_buff, key, min_key, max_key);
  }
  const int64_t translated_key = max_key + 1;
  return hash_join_idx(hash_buff, translated_key, min_key, translated_key);
}
