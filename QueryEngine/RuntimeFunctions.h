#ifndef QUERYENGINE_RUNTIMEFUNCTIONS_H
#define QUERYENGINE_RUNTIMEFUNCTIONS_H

#include <cstdint>
#include <limits>


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

#endif  // QUERYENGINE_RUNTIMEFUNCTIONS_H
