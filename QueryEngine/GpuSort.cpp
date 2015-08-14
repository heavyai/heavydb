#include <cstdint>

#ifdef HAVE_CUDA
#include "GpuSortImpl.h"
#endif

void sort_groups(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, const bool desc) {
#ifdef HAVE_CUDA
  sort_on_device(val_buff, idx_buff, entry_count, desc);
#endif
}

void apply_permutation(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, int64_t* tmp_buff) {
#ifdef HAVE_CUDA
  apply_permutation_on_device(val_buff, idx_buff, entry_count, tmp_buff);
#endif
}
