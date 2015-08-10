#include <cstdint>

#include "GpuSortImpl.h"

void sort_groups(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, const bool desc) {
  sort_on_device(val_buff, idx_buff, entry_count, desc);
}

void apply_permutation(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count) {
  apply_permutation_on_device(val_buff, idx_buff, entry_count);
}
