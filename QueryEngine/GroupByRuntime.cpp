#include "../Shared/funcannotations.h"

extern "C" ALWAYS_INLINE DEVICE uint32_t key_hash(const int64_t* key, const uint32_t key_qw_count) {
  uint32_t hash = 0;
  for (uint32_t i = 0; i < key_qw_count; ++i) {
    hash = (hash << 5) - hash + key[i];
  }
  return hash;
}

extern "C" NEVER_INLINE DEVICE int64_t* get_group_value(int64_t* groups_buffer,
                                                        const uint32_t groups_buffer_entry_count,
                                                        const int64_t* key,
                                                        const uint32_t key_qw_count,
                                                        const uint32_t agg_col_count) {
  uint32_t h = key_hash(key, key_qw_count) % groups_buffer_entry_count;
  int64_t* matching_group = get_matching_group_value(groups_buffer, h, key, key_qw_count, agg_col_count);
  if (matching_group) {
    return matching_group;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group = get_matching_group_value(groups_buffer, h_probe, key, key_qw_count, agg_col_count);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return NULL;
}

extern "C" ALWAYS_INLINE DEVICE int64_t* get_group_value_fast(int64_t* groups_buffer,
                                                              const int64_t key,
                                                              const int64_t min_key,
                                                              const uint32_t agg_col_count) {
  int64_t off = (key - min_key) * (1 + agg_col_count);
  if (groups_buffer[off] == EMPTY_KEY) {
    groups_buffer[off] = key;
  }
  return groups_buffer + off + 1;
}

extern "C" ALWAYS_INLINE DEVICE int64_t* get_columnar_group_value_fast(int64_t* groups_buffer,
                                                                       const int64_t key,
                                                                       const int64_t min_key) {
  int64_t off = key - min_key;
  if (groups_buffer[off] == EMPTY_KEY) {
    groups_buffer[off] = key;
  }
  return groups_buffer + off;
}

extern "C" ALWAYS_INLINE DEVICE int64_t* get_group_value_one_key(int64_t* groups_buffer,
                                                                 const uint32_t groups_buffer_entry_count,
                                                                 int64_t* small_groups_buffer,
                                                                 const uint32_t small_groups_buffer_qw_count,
                                                                 const int64_t key,
                                                                 const int64_t min_key,
                                                                 const uint32_t agg_col_count) {
  int64_t off = key - min_key;
  if (0 <= off && off < small_groups_buffer_qw_count) {
    return get_group_value_fast(small_groups_buffer, key, min_key, agg_col_count);
  }
  return get_group_value(groups_buffer, groups_buffer_entry_count, &key, 1, agg_col_count);
}
