#include "HashJoinRuntime.h"
#include "RuntimeFunctions.h"

namespace {

void init_groups(int64_t* groups_buffer,
                 const int32_t groups_buffer_entry_count,
                 const int32_t key_qw_count,
                 const int64_t* init_vals) {
  int32_t groups_buffer_entry_qw_count = groups_buffer_entry_count * (key_qw_count + 1);
  for (int32_t i = 0; i < groups_buffer_entry_qw_count; ++i) {
    groups_buffer[i] =
        (i % (key_qw_count + 1) < key_qw_count) ? EMPTY_KEY : init_vals[(i - key_qw_count) % (key_qw_count + 1)];
  }
}

}  // namespace

void init_hash_join_buff(int64_t* buff,
                         const int32_t groups_buffer_entry_count,
                         const int8_t* col_buff,
                         const size_t num_elems,
                         const size_t elem_sz,
                         const int64_t min_val) {
  int64_t init_val = -1;
  init_groups(buff, groups_buffer_entry_count, 1, &init_val);
  for (size_t i = 0; i < num_elems; ++i) {
    int64_t* entry_ptr = get_group_value_fast(buff, fixed_width_int_decode_noinline(col_buff, elem_sz, i), min_val, 1);
    // TODO; must check if it's one to one
    *entry_ptr = i;
  }
}

extern "C" __attribute__((noinline)) int64_t
    hash_join_idx(int64_t hash_buff, const int64_t key, const int64_t min_key, const int64_t max_key) {
  if (key >= min_key && key <= max_key) {
    // TODO(alex): don't use get_group_value_fast, it's not read-only
    return *get_group_value_fast(reinterpret_cast<int64_t*>(hash_buff), key, min_key, 1);
  }
  return -1;
}
