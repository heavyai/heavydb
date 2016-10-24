#include "JoinHashImpl.h"

extern "C" NEVER_INLINE DEVICE uint32_t MurmurHash1(const void* key, int len, const uint32_t seed) {
  const unsigned int m = 0xc6a4a793;

  const int r = 16;

  unsigned int h = seed ^ (len * m);

  //----------

  const unsigned char* data = (const unsigned char*)key;

  while (len >= 4) {
    unsigned int k = *(unsigned int*)data;

    h += k;
    h *= m;
    h ^= h >> 16;

    data += 4;
    len -= 4;
  }

  //----------

  switch (len) {
    case 3:
      h += data[2] << 16;
    case 2:
      h += data[1] << 8;
    case 1:
      h += data[0];
      h *= m;
      h ^= h >> r;
  };

  //----------

  h *= m;
  h ^= h >> 10;
  h *= m;
  h ^= h >> 17;

  return h;
}

extern "C" ALWAYS_INLINE DEVICE uint32_t key_hash(const int64_t* key, const uint32_t key_qw_count) {
  return MurmurHash1(key, 8 * key_qw_count, 0);
}

extern "C" NEVER_INLINE DEVICE int64_t* get_group_value(int64_t* groups_buffer,
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
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group = get_matching_group_value(groups_buffer, h_probe, key, key_qw_count, row_size_quad, init_vals);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return NULL;
}

extern "C" NEVER_INLINE DEVICE int64_t* get_group_value_columnar(int64_t* groups_buffer,
                                                                 const uint32_t groups_buffer_entry_count,
                                                                 const int64_t* key,
                                                                 const uint32_t key_qw_count) {
  uint32_t h = key_hash(key, key_qw_count) % groups_buffer_entry_count;
  int64_t* matching_group =
      get_matching_group_value_columnar(groups_buffer, h, key, key_qw_count, groups_buffer_entry_count);
  if (matching_group) {
    return matching_group;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group =
        get_matching_group_value_columnar(groups_buffer, h_probe, key, key_qw_count, groups_buffer_entry_count);
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
                                                                 const int64_t* init_vals) {
  int64_t off = key - min_key;
  if (0 <= off && off < small_groups_buffer_qw_count) {
    return get_group_value_fast(small_groups_buffer, key, min_key, 0, row_size_quad);
  }
  return get_group_value(groups_buffer, groups_buffer_entry_count, &key, 1, row_size_quad, init_vals);
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

extern "C" ALWAYS_INLINE DEVICE int64_t
    hash_join_idx(int64_t hash_buff, const int64_t key, const int64_t min_key, const int64_t max_key) {
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
