#ifdef HAVE_CUDA
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#endif

#include "DataMgr/Allocators/ThrustAllocator.h"
#include "InPlaceSortImpl.h"

#ifdef HAVE_CUDA
#include <cuda.h>
CUstream getQueryEngineCudaStreamForDevice(int device_num);

#include "Logger/Logger.h"
#define checkCudaErrors(err) CHECK_EQ(err, CUDA_SUCCESS)

template <typename T>
void sort_on_gpu(T* val_buff,
                 int32_t* idx_buff,
                 const uint64_t entry_count,
                 const bool desc,
                 ThrustAllocator& alloc,
                 const int device_id) {
  thrust::device_ptr<T> key_ptr(val_buff);
  thrust::device_ptr<int32_t> idx_ptr(idx_buff);
  thrust::sequence(idx_ptr, idx_ptr + entry_count);
  auto qe_cuda_stream = getQueryEngineCudaStreamForDevice(device_id);
  if (desc) {
    thrust::sort_by_key(thrust::cuda::par(alloc).on(qe_cuda_stream),
                        key_ptr,
                        key_ptr + entry_count,
                        idx_ptr,
                        thrust::greater<T>());
  } else {
    thrust::sort_by_key(thrust::cuda::par(alloc).on(qe_cuda_stream),
                        key_ptr,
                        key_ptr + entry_count,
                        idx_ptr);
  }
  checkCudaErrors(cuStreamSynchronize(qe_cuda_stream));
}

template <typename T>
void apply_permutation_on_gpu(T* val_buff,
                              int32_t* idx_buff,
                              const uint64_t entry_count,
                              ThrustAllocator& alloc,
                              const int device_id) {
  thrust::device_ptr<T> key_ptr(val_buff);
  thrust::device_ptr<int32_t> idx_ptr(idx_buff);
  const size_t buf_size = entry_count * sizeof(T);
  T* raw_ptr = reinterpret_cast<T*>(alloc.allocate(buf_size));
  thrust::device_ptr<T> tmp_ptr(raw_ptr);
  auto qe_cuda_stream = getQueryEngineCudaStreamForDevice(device_id);
  thrust::copy(thrust::cuda::par(alloc).on(qe_cuda_stream),
               key_ptr,
               key_ptr + entry_count,
               tmp_ptr);
  checkCudaErrors(cuStreamSynchronize(qe_cuda_stream));
  thrust::gather(thrust::cuda::par(alloc).on(qe_cuda_stream),
                 idx_ptr,
                 idx_ptr + entry_count,
                 tmp_ptr,
                 key_ptr);
  checkCudaErrors(cuStreamSynchronize(qe_cuda_stream));
  alloc.deallocate(reinterpret_cast<int8_t*>(raw_ptr), buf_size);
}

template <typename T>
void sort_on_cpu(T* val_buff,
                 int32_t* idx_buff,
                 const uint64_t entry_count,
                 const bool desc) {
  thrust::sequence(idx_buff, idx_buff + entry_count);
  if (desc) {
    thrust::sort_by_key(val_buff, val_buff + entry_count, idx_buff, thrust::greater<T>());
  } else {
    thrust::sort_by_key(val_buff, val_buff + entry_count, idx_buff);
  }
}

template <typename T>
void apply_permutation_on_cpu(T* val_buff,
                              int32_t* idx_buff,
                              const uint64_t entry_count,
                              T* tmp_buff) {
  thrust::copy(val_buff, val_buff + entry_count, tmp_buff);
  thrust::gather(idx_buff, idx_buff + entry_count, tmp_buff, val_buff);
}
#endif

void sort_on_gpu(int64_t* val_buff,
                 int32_t* idx_buff,
                 const uint64_t entry_count,
                 const bool desc,
                 const uint32_t chosen_bytes,
                 ThrustAllocator& alloc,
                 const int device_id) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 1:
      sort_on_gpu(reinterpret_cast<int8_t*>(val_buff),
                  idx_buff,
                  entry_count,
                  desc,
                  alloc,
                  device_id);
      break;
    case 2:
      sort_on_gpu(reinterpret_cast<int16_t*>(val_buff),
                  idx_buff,
                  entry_count,
                  desc,
                  alloc,
                  device_id);
      break;
    case 4:
      sort_on_gpu(reinterpret_cast<int32_t*>(val_buff),
                  idx_buff,
                  entry_count,
                  desc,
                  alloc,
                  device_id);
      break;
    case 8:
      sort_on_gpu(val_buff, idx_buff, entry_count, desc, alloc, device_id);
      break;
    default:
      // FIXME(miyu): CUDA linker doesn't accept assertion on GPU yet right now.
      break;
  }
#endif
}

void sort_on_cpu(int64_t* val_buff,
                 int32_t* idx_buff,
                 const uint64_t entry_count,
                 const bool desc,
                 const uint32_t chosen_bytes) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 1:
      sort_on_cpu(reinterpret_cast<int8_t*>(val_buff), idx_buff, entry_count, desc);
      break;
    case 2:
      sort_on_cpu(reinterpret_cast<int16_t*>(val_buff), idx_buff, entry_count, desc);
      break;
    case 4:
      sort_on_cpu(reinterpret_cast<int32_t*>(val_buff), idx_buff, entry_count, desc);
      break;
    case 8:
      sort_on_cpu(val_buff, idx_buff, entry_count, desc);
      break;
    default:
      // FIXME(miyu): CUDA linker doesn't accept assertion on GPU yet right now.
      break;
  }
#endif
}

void apply_permutation_on_gpu(int64_t* val_buff,
                              int32_t* idx_buff,
                              const uint64_t entry_count,
                              const uint32_t chosen_bytes,
                              ThrustAllocator& alloc,
                              const int device_id) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 1:
      apply_permutation_on_gpu(
          reinterpret_cast<int8_t*>(val_buff), idx_buff, entry_count, alloc, device_id);
      break;
    case 2:
      apply_permutation_on_gpu(
          reinterpret_cast<int16_t*>(val_buff), idx_buff, entry_count, alloc, device_id);
      break;
    case 4:
      apply_permutation_on_gpu(
          reinterpret_cast<int32_t*>(val_buff), idx_buff, entry_count, alloc, device_id);
      break;
    case 8:
      apply_permutation_on_gpu(val_buff, idx_buff, entry_count, alloc, device_id);
      break;
    default:
      // FIXME(miyu): CUDA linker doesn't accept assertion on GPU yet right now.
      break;
  }
#endif
}

void apply_permutation_on_cpu(int64_t* val_buff,
                              int32_t* idx_buff,
                              const uint64_t entry_count,
                              int64_t* tmp_buff,
                              const uint32_t chosen_bytes) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 1:
      apply_permutation_on_cpu(reinterpret_cast<int8_t*>(val_buff),
                               idx_buff,
                               entry_count,
                               reinterpret_cast<int8_t*>(tmp_buff));
      break;
    case 2:
      apply_permutation_on_cpu(reinterpret_cast<int16_t*>(val_buff),
                               idx_buff,
                               entry_count,
                               reinterpret_cast<int16_t*>(tmp_buff));
      break;
    case 4:
      apply_permutation_on_cpu(reinterpret_cast<int32_t*>(val_buff),
                               idx_buff,
                               entry_count,
                               reinterpret_cast<int32_t*>(tmp_buff));
      break;
    case 8:
      apply_permutation_on_cpu(val_buff, idx_buff, entry_count, tmp_buff);
      break;
    default:
      // FIXME(miyu): CUDA linker doesn't accept assertion on GPU yet right now.
      break;
  }
#endif
}
