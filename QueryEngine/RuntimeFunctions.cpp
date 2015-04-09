#include "RuntimeFunctions.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <tuple>


// decoder implementations

extern "C" __attribute__((always_inline))
int64_t fixed_width_int_decode(
    const int8_t* byte_stream,
    const int32_t byte_width,
    const int64_t pos) {
  switch (byte_width) {
  case 1:
    return static_cast<int64_t>(byte_stream[pos * byte_width]);
  case 2:
    return *(reinterpret_cast<const int16_t*>(&byte_stream[pos * byte_width]));
  case 4:
    return *(reinterpret_cast<const int32_t*>(&byte_stream[pos * byte_width]));
  case 8:
    return *(reinterpret_cast<const int64_t*>(&byte_stream[pos * byte_width]));
  default:
    // TODO(alex)
    return std::numeric_limits<int64_t>::min() + 1;
  }
}

extern "C"
int64_t fixed_width_int_decode_noinline(
    const int8_t* byte_stream,
    const int32_t byte_width,
    const int64_t pos) {
  return fixed_width_int_decode(byte_stream, byte_width, pos);
}

extern "C" __attribute__((always_inline))
int64_t diff_fixed_width_int_decode(
    const int8_t* byte_stream,
    const int32_t byte_width,
    const int64_t baseline,
    const int64_t pos) {
  return fixed_width_int_decode(byte_stream, byte_width, pos) + baseline;
}

extern "C" __attribute__((always_inline))
float fixed_width_float_decode(
    const int8_t* byte_stream,
    const int64_t pos) {
  return *(reinterpret_cast<const float*>(&byte_stream[pos * sizeof(float)]));
}

extern "C"
float fixed_width_float_decode_noinline(
    const int8_t* byte_stream,
    const int64_t pos) {
  return fixed_width_float_decode(byte_stream, pos);
}

extern "C" __attribute__((always_inline))
double fixed_width_double_decode(
    const int8_t* byte_stream,
    const int64_t pos) {
  return *(reinterpret_cast<const double*>(&byte_stream[pos * sizeof(double)]));
}

extern "C"
double fixed_width_double_decode_noinline(
    const int8_t* byte_stream,
    const int64_t pos) {
  return fixed_width_double_decode(byte_stream, pos);
}

// aggregator implementations

extern "C" __attribute__((always_inline))
void agg_count(int64_t* agg, const int64_t val) {
  ++*agg;
}

extern "C" __attribute__((always_inline))
void agg_count_distinct(int64_t* agg, const int64_t val) {
  reinterpret_cast<std::set<int64_t>*>(*agg)->insert(val);
}

extern "C" __attribute__((always_inline))
void agg_count_distinct_bitmap(int64_t* agg, const int64_t val, const int64_t min_val) {
  const uint64_t bitmap_idx = val - min_val;
  reinterpret_cast<int8_t*>(*agg)[bitmap_idx >> 3] |= (1 << (bitmap_idx & 7));
}

extern "C" __attribute__((always_inline))
void agg_sum(int64_t* agg, const int64_t val) {
  *agg += val;
}

extern "C" __attribute__((always_inline))
void agg_max(int64_t* agg, const int64_t val) {
  *agg = std::max(*agg, val);
}

extern "C" __attribute__((always_inline))
void agg_min(int64_t* agg, const int64_t val) {
  *agg = std::min(*agg, val);
}

extern "C" __attribute__((always_inline))
void agg_id(int64_t* agg, const int64_t val) {
  *agg = val;
}

extern "C" __attribute__((always_inline))
void agg_count_distinct_skip_val(int64_t* agg, const int64_t val, const int64_t skip_val) {
  if (val != skip_val) {
    agg_count_distinct(agg, val);
  }
}

extern "C" __attribute__((always_inline))
void agg_count_distinct_bitmap_skip_val(int64_t* agg, const int64_t val, const int64_t min_val, const int64_t skip_val) {
  if (val != skip_val) {
    agg_count_distinct_bitmap(agg, val, min_val);
  }
}

#define DEF_SKIP_AGG(base_agg_func)                                                       \
extern "C" __attribute__((always_inline))                                                 \
void base_agg_func##_skip_val(int64_t* agg, const int64_t val, const int64_t skip_val) {  \
  if (val != skip_val) {                                                                  \
    base_agg_func(agg, val);                                                              \
  }                                                                                       \
}

DEF_SKIP_AGG(agg_count)
DEF_SKIP_AGG(agg_sum)
DEF_SKIP_AGG(agg_max)
DEF_SKIP_AGG(agg_min)

#undef DEF_SKIP_AGG

// TODO(alex): fix signature, implement the rest

extern "C" __attribute__((always_inline))
void agg_count_double(int64_t* agg, const double val) {
  ++*agg;
}

extern "C" __attribute__((always_inline))
void agg_sum_double(int64_t* agg, const double val) {
  const auto r = *reinterpret_cast<const double*>(agg) + val;
  *agg = *reinterpret_cast<const int64_t*>(&r);
}

extern "C" __attribute__((always_inline))
void agg_max_double(int64_t* agg, const double val) {
  const auto r = std::max(*reinterpret_cast<const double*>(agg), val);
  *agg = *(reinterpret_cast<const int64_t*>(&r));
}

extern "C" __attribute__((always_inline))
void agg_min_double(int64_t* agg, const double val) {
  const auto r = std::min(*reinterpret_cast<const double*>(agg), val);
  *agg = *(reinterpret_cast<const int64_t*>(&r));
}

extern "C" __attribute__((always_inline))
void agg_id_double(int64_t* agg, const double val) {
  *agg = *(reinterpret_cast<const int64_t*>(&val));
}

// Shared memory aggregators. Should never be called,
// real implementations are in cuda_mapd_rt.cu.

#define DEF_SHARED_AGG_STUBS(base_agg_func)                                                      \
extern "C" __attribute__((noinline))                                                             \
void base_agg_func##_shared(int64_t* agg, const int64_t val) {                                   \
  abort();                                                                                       \
}                                                                                                \
                                                                                                 \
extern "C" __attribute__((noinline))                                                             \
void base_agg_func##_skip_val_shared(int64_t* agg, const int64_t val, const int64_t skip_val) {  \
  abort();                                                                                       \
}                                                                                                \
                                                                                                 \
extern "C" __attribute__((noinline))                                                             \
void base_agg_func##_double_shared(int64_t* agg, const double val) {                             \
  abort();                                                                                       \
}

DEF_SHARED_AGG_STUBS(agg_count)
DEF_SHARED_AGG_STUBS(agg_sum)
DEF_SHARED_AGG_STUBS(agg_max)
DEF_SHARED_AGG_STUBS(agg_min)
DEF_SHARED_AGG_STUBS(agg_id)

// x64 stride functions

extern "C" __attribute__((noinline))
int32_t pos_start_impl(const int32_t* row_index_resume) {
  return row_index_resume ? row_index_resume[0] : 0;
}

extern "C" __attribute__((noinline))
int32_t pos_step_impl() {
  return 1;
}

extern "C" __attribute__((noinline))
int8_t thread_warp_idx(const int8_t warp_sz) {
  abort();
}

extern "C" __attribute__((always_inline))
int32_t merge_error_code(const int32_t err_code, int32_t* merged_err_code) {
  if (err_code) {
    *merged_err_code = err_code;
  }
  return err_code;
}

// group by helpers

extern "C" __attribute__((noinline))
const int64_t* init_shared_mem_nop(const int64_t* groups_buffer,
                                   const int32_t groups_buffer_size) {
  return groups_buffer;
}

extern "C" __attribute__((noinline))
void write_back_nop(int64_t* dest, int64_t* src, const int32_t sz) {
  // the body is not really needed, just make sure the call is not optimized away
  assert(dest);
}

extern "C" __attribute__((noinline))
const int64_t* init_shared_mem(const int64_t* groups_buffer,
                               const int32_t groups_buffer_size) {
  return init_shared_mem_nop(groups_buffer, groups_buffer_size);
}

extern "C" __attribute__((noinline))
void write_back(int64_t* dest, int64_t* src, const int32_t sz) {
  write_back_nop(dest, src, sz);
}

extern "C" __attribute__((noinline))
void init_group_by_buffer_impl(int64_t* groups_buffer,
                               const int64_t* init_vals,
                               const int32_t groups_buffer_entry_count,
                               const int32_t key_qw_count,
                               const int32_t agg_col_count) {
  // the body is not really needed, just make sure the call is not optimized away
  assert(groups_buffer);
}

extern "C" __attribute__((always_inline))
int64_t* get_matching_group_value(int64_t* groups_buffer,
                                  const int32_t h,
                                  const int64_t* key,
                                  const int32_t key_qw_count,
                                  const int32_t agg_col_count,
                                  const int64_t* init_vals) {
  auto off = h * (key_qw_count + agg_col_count);
  if (groups_buffer[off] == EMPTY_KEY) {
    memcpy(groups_buffer + off, key, key_qw_count * sizeof(*key));
    return groups_buffer + off + key_qw_count;
  }
  if (memcmp(groups_buffer + off, key, key_qw_count * sizeof(*key)) == 0) {
    return groups_buffer + off + key_qw_count;
  }
  return nullptr;
}

extern "C" __attribute__((always_inline))
int32_t key_hash(const int64_t* key, const int32_t key_qw_count, const int32_t groups_buffer_entry_count) {
  int32_t hash = 0;
  for (int32_t i = 0; i < key_qw_count; ++i) {
    hash = ((hash << 5) - hash + key[i]) % groups_buffer_entry_count;
  }
  return static_cast<uint32_t>(hash) % groups_buffer_entry_count;
}

extern "C" __attribute__((noinline))
int64_t* get_group_value(int64_t* groups_buffer,
                         const int32_t groups_buffer_entry_count,
                         const int64_t* key,
                         const int32_t key_qw_count,
                         const int32_t agg_col_count,
                         const int64_t* init_vals) {
  auto h = key_hash(key, key_qw_count, groups_buffer_entry_count);
  auto matching_group = get_matching_group_value(groups_buffer, h, key, key_qw_count, agg_col_count, init_vals);
  if (matching_group) {
    return matching_group;
  }
  auto h_probe = h + 1;
  while (h_probe != h) {
    matching_group = get_matching_group_value(groups_buffer, h_probe, key, key_qw_count, agg_col_count, init_vals);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return nullptr;
}

extern "C" __attribute__((always_inline))
int64_t* get_group_value_fast(int64_t* groups_buffer,
                              const int64_t key,
                              const int64_t min_key,
                              const int32_t agg_col_count) {
  auto off = (key - min_key) * (1 + agg_col_count);
  if (groups_buffer[off] == EMPTY_KEY) {
    groups_buffer[off] = key;
  }
  return groups_buffer + off + 1;
}

extern "C" __attribute__((always_inline))
int64_t* get_group_value_one_key(int64_t* groups_buffer,
                                 const int32_t groups_buffer_entry_count,
                                 int64_t* small_groups_buffer,
                                 const int32_t small_groups_buffer_qw_count,
                                 const int64_t key,
                                 const int64_t min_key,
                                 const int32_t agg_col_count,
                                 const int64_t* init_vals) {
  auto off = (key - min_key) * (1 + agg_col_count);
  if (0 <= off && off < small_groups_buffer_qw_count) {
    return get_group_value_fast(small_groups_buffer, key, min_key, agg_col_count);
  }
  return get_group_value(groups_buffer, groups_buffer_entry_count, &key, 1, agg_col_count, init_vals);
}

extern "C" __attribute__((always_inline))
int64_t* get_group_value_fast_keyless(int64_t* groups_buffer,
                                      const int64_t key,
                                      const int64_t min_key,
                                      const int32_t agg_col_count) {
  return groups_buffer + agg_col_count * (key - min_key);
}

extern "C" __attribute__((always_inline))
int64_t* get_group_value_fast_keyless_semiprivate(int64_t* groups_buffer,
                                                  const int64_t key,
                                                  const int64_t min_key,
                                                  const int32_t agg_col_count,
                                                  const int8_t thread_warp_idx,
                                                  const int8_t warp_size) {
  return groups_buffer + agg_col_count * (warp_size * (key - min_key) + thread_warp_idx);
}

extern "C" __attribute__((always_inline))
int8_t* extract_str_ptr(const uint64_t str_and_len) {
  return reinterpret_cast<int8_t*>(str_and_len & 0xffffffffffff);
}

extern "C" __attribute__((always_inline))
int32_t extract_str_len(const uint64_t str_and_len) {
  return static_cast<int64_t>(str_and_len) >> 48;
}

extern "C" __attribute__((noinline))
int8_t* extract_str_ptr_noinline(const uint64_t str_and_len) {
  return extract_str_ptr(str_and_len);
}

extern "C" __attribute__((noinline))
int32_t extract_str_len_noinline(const uint64_t str_and_len) {
  return extract_str_len(str_and_len);
}

#ifdef __clang__
#include "ExtractFromTime.cpp"
#include "../Utils/StringLike.cpp"
#endif
