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

#define EMPTY_KEY std::numeric_limits<int64_t>::min()

extern "C"
void init_groups(int64_t* groups_buffer,
                 const int32_t groups_buffer_entry_count,
                 const int32_t key_qw_count,
                 const int64_t* init_vals,
                 const int32_t agg_col_count);

extern "C"
int64_t* get_group_value(int64_t* groups_buffer,
                         const int32_t groups_buffer_entry_count,
                         const int64_t* key,
                         const int32_t key_qw_count,
                         const int32_t agg_col_count);

enum ExtractField : int32_t {
  kYEAR,
  kMONTH,
  kDAY,
  kHOUR,
  kMINUTE,
  kSECOND,
  kDOW,
  kDOY,
  kEPOCH
};

extern "C" __attribute__((noinline))
int64_t ExtractFromTime(ExtractField f, time_t t);

#endif  // QUERYENGINE_RUNTIMEFUNCTIONS_H
