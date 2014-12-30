#ifndef QUERYENGINE_EXECUTE_RUNTIMEFUNCTIONS_H
#define QUERYENGINE_EXECUTE_RUNTIMEFUNCTIONS_H

#include <cstdint>

extern "C"
void init_groups(int64_t* groups_buffer,
                 const int32_t groups_buffer_entry_count,
                 const int32_t key_qw_count,
                 const int64_t init_val);

extern "C" __attribute__((always_inline))
int64_t* get_group_value(int64_t* groups_buffer,
                         const int32_t groups_buffer_entry_count,
                         const int64_t* key,
                         const int32_t key_qw_count);

#endif  // QUERYENGINE_EXECUTE_RUNTIMEFUNCTIONS_H