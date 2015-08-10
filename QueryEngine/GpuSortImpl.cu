#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "GpuSortImpl.h"

void sort_on_device(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count, const bool desc) {
  thrust::device_ptr<int64_t> key_ptr(val_buff);
  thrust::device_ptr<int64_t> idx_ptr(idx_buff);
  thrust::sequence(idx_ptr, idx_ptr + entry_count);
  if (desc) {
    thrust::sort_by_key(key_ptr, key_ptr + entry_count, idx_ptr, thrust::greater<int64_t>());
  } else {
    thrust::sort_by_key(key_ptr, key_ptr + entry_count, idx_ptr);
  }
}

void apply_permutation_on_device(int64_t* val_buff, int64_t* idx_buff, const uint64_t entry_count) {
  thrust::device_ptr<int64_t> key_ptr(val_buff);
  thrust::device_ptr<int64_t> idx_ptr(idx_buff);
  thrust::device_vector<int64_t> temp(key_ptr, key_ptr + entry_count);
  thrust::gather(idx_ptr, idx_ptr + entry_count, temp.begin(), key_ptr);
}
