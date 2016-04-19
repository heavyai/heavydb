#include <cstdint>

#ifdef HAVE_CUDA
#include "InPlaceSortImpl.h"
#endif

void sort_groups_gpu(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, const bool desc) {
#ifdef HAVE_CUDA
  sort_on_gpu(val_buff, idx_buff, entry_count, desc);
#endif
}

void sort_groups_cpu(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, const bool desc) {
#ifdef HAVE_CUDA
  sort_on_cpu(val_buff, idx_buff, entry_count, desc);
#endif
}

void apply_permutation_gpu(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, int64_t* tmp_buff) {
#ifdef HAVE_CUDA
  apply_permutation_on_gpu(val_buff, idx_buff, entry_count, tmp_buff);
#endif
}

void apply_permutation_cpu(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, int64_t* tmp_buff) {
#ifdef HAVE_CUDA
  apply_permutation_on_cpu(val_buff, idx_buff, entry_count, tmp_buff);
#endif
}
