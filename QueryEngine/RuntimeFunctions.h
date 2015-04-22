#ifndef QUERYENGINE_RUNTIMEFUNCTIONS_H
#define QUERYENGINE_RUNTIMEFUNCTIONS_H

#include <cassert>
#include <cstdint>
#include <ctime>
#include <limits>


extern "C"
void agg_sum(int64_t* agg, const int64_t val);

extern "C"
void agg_max(int64_t* agg, const int64_t val);

extern "C"
void agg_min(int64_t* agg, const int64_t val);

extern "C"
void agg_sum_double(int64_t* agg, const double val);

extern "C"
void agg_max_double(int64_t* agg, const double val);

extern "C"
void agg_min_double(int64_t* agg, const double val);

extern "C"
void agg_max_skip_val(int64_t* agg, const int64_t val, const int64_t skip_val);

extern "C"
void agg_min_skip_val(int64_t* agg, const int64_t val, const int64_t skip_val);

extern "C"
void agg_max_double_skip_val(int64_t* agg, const double val, const double skip_val);

extern "C"
void agg_min_double_skip_val(int64_t* agg, const double val, const double skip_val);

#define EMPTY_KEY std::numeric_limits<int64_t>::max()

extern "C"
int64_t* get_group_value(int64_t* groups_buffer,
                         const int32_t groups_buffer_entry_count,
                         const int64_t* key,
                         const int32_t key_qw_count,
                         const int32_t agg_col_count,
                         const int64_t* init_val = nullptr);

// Regular fixed_width_*_decode are only available from the JIT,
// we need to call them for lazy fetch columns -- create wrappers.

extern "C"
int64_t fixed_width_int_decode_noinline(
    const int8_t* byte_stream,
    const int32_t byte_width,
    const int64_t pos);

extern "C"
float fixed_width_float_decode_noinline(
    const int8_t* byte_stream,
    const int64_t pos);

extern "C"
double fixed_width_double_decode_noinline(
    const int8_t* byte_stream,
    const int64_t pos);

extern "C"
int8_t* extract_str_ptr_noinline(const uint64_t str_and_len);

extern "C"
int32_t extract_str_len_noinline(const uint64_t str_and_len);

#endif  // QUERYENGINE_RUNTIMEFUNCTIONS_H
