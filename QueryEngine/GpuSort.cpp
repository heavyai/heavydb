#include <cstdint>

#include "GpuSortImpl.h"

void sort_groups(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count) {
  sort_on_device(val_buff, idx_buff, entry_count);
}

void apply_permutation(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count) {
  apply_permutation_on_device(val_buff, idx_buff, entry_count);
}
