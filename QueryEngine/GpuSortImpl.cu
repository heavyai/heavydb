#ifdef HAVE_CUDA
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#endif

#include "GpuSortImpl.h"

void sort_on_device(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, const bool desc) {
#ifdef HAVE_CUDA
  thrust::device_ptr<int64_t> key_ptr(val_buff);
  thrust::device_ptr<int64_t> idx_ptr(idx_buff);
  thrust::sequence(idx_ptr, idx_ptr + entry_count);
  if (desc) {
    thrust::sort_by_key(key_ptr, key_ptr + entry_count, idx_ptr, thrust::greater<int64_t>());
  } else {
    thrust::sort_by_key(key_ptr, key_ptr + entry_count, idx_ptr);
  }
#endif
}

void apply_permutation_on_device(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, int64_t* tmp_buff) {
#ifdef HAVE_CUDA
  thrust::device_ptr<int64_t> key_ptr(val_buff);
  thrust::device_ptr<int64_t> idx_ptr(idx_buff);
  thrust::device_ptr<int64_t> tmp_ptr(tmp_buff);
  thrust::copy(key_ptr, key_ptr + entry_count, tmp_ptr);
  thrust::gather(idx_ptr, idx_ptr + entry_count, tmp_ptr, key_ptr);
#endif
}
