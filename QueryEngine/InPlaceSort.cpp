#include "ThrustAllocator.h"

#include <cstdint>
#include <glog/logging.h>

#ifdef HAVE_CUDA
#include "InPlaceSortImpl.h"
#endif

void sort_groups_gpu(int64_t* val_buff,
                     int32_t* idx_buff,
                     const uint64_t entry_count,
                     const bool desc,
                     const uint32_t chosen_bytes,
                     ThrustAllocator& alloc) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 4:
    case 8:
      sort_on_gpu(val_buff, idx_buff, entry_count, desc, chosen_bytes, alloc);
      break;
    default:
      CHECK(false);
  }
#endif
}

void sort_groups_cpu(int64_t* val_buff,
                     int32_t* idx_buff,
                     const uint64_t entry_count,
                     const bool desc,
                     const uint32_t chosen_bytes) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 4:
    case 8:
      sort_on_cpu(val_buff, idx_buff, entry_count, desc, chosen_bytes);
      break;
    default:
      CHECK(false);
  }
#endif
}

void apply_permutation_gpu(int64_t* val_buff,
                           int32_t* idx_buff,
                           const uint64_t entry_count,
                           const uint32_t chosen_bytes,
                           ThrustAllocator& alloc) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 4:
    case 8:
      apply_permutation_on_gpu(val_buff, idx_buff, entry_count, chosen_bytes, alloc);
      break;
    default:
      CHECK(false);
  }
#endif
}

void apply_permutation_cpu(int64_t* val_buff,
                           int32_t* idx_buff,
                           const uint64_t entry_count,
                           int64_t* tmp_buff,
                           const uint32_t chosen_bytes) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 4:
    case 8:
      apply_permutation_on_cpu(val_buff, idx_buff, entry_count, tmp_buff, chosen_bytes);
      break;
    default:
      CHECK(false);
  }
#endif
}
