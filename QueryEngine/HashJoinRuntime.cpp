#include "HashJoinRuntime.h"
#ifdef __CUDACC__
#include "GpuRtConstants.h"
#include "DecodersImpl.h"
#include "GroupByFastImpl.h"
#else
#include "RuntimeFunctions.h"
#endif
#include "../Shared/funcannotations.h"

DEVICE void SUFFIX(init_groups)(int64_t* groups_buffer,
                                const int32_t groups_buffer_entry_count,
                                const int32_t key_qw_count,
                                const int64_t* init_vals) {
  int32_t groups_buffer_entry_qw_count = groups_buffer_entry_count * (key_qw_count + 1);
  for (int32_t i = 0; i < groups_buffer_entry_qw_count; ++i) {
    groups_buffer[i] =
        (i % (key_qw_count + 1) < key_qw_count) ? EMPTY_KEY : init_vals[(i - key_qw_count) % (key_qw_count + 1)];
  }
}

DEVICE void SUFFIX(init_hash_join_buff)(int64_t* buff,
                                        const int32_t groups_buffer_entry_count,
                                        const int8_t* col_buff,
                                        const size_t num_elems,
                                        const size_t elem_sz,
                                        const int64_t min_val) {
  int64_t init_val = -1;
  SUFFIX(init_groups)(buff, groups_buffer_entry_count, 1, &init_val);
  for (size_t i = 0; i < num_elems; ++i) {
    int64_t* entry_ptr =
        SUFFIX(get_group_value_fast)(buff, SUFFIX(fixed_width_int_decode_noinline)(col_buff, elem_sz, i), min_val, 1);
    // TODO; must check if it's one to one
    *entry_ptr = i;
  }
}

#ifdef __CUDACC__
__global__ void init_hash_join_buff_wrapper(int64_t* buff,
                                            const int32_t groups_buffer_entry_count,
                                            const int8_t* col_buff,
                                            const size_t num_elems,
                                            const size_t elem_sz,
                                            const int64_t min_val) {
  SUFFIX(init_hash_join_buff)(buff, groups_buffer_entry_count, col_buff, num_elems, elem_sz, min_val);
}

void init_hash_join_buff_on_device(int64_t* buff,
                                   const int32_t groups_buffer_entry_count,
                                   const int8_t* col_buff,
                                   const size_t num_elems,
                                   const size_t elem_sz,
                                   const int64_t min_val,
                                   const size_t block_size_x,
                                   const size_t grid_size_x) {
  init_hash_join_buff_wrapper<<<grid_size_x, block_size_x>>>
      (buff, groups_buffer_entry_count, col_buff, num_elems, elem_sz, min_val);
}
#endif
