/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef HAVE_CUDA
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#endif

#include "DataMgr/Allocators/ThrustAllocator.h"
#include "InPlaceSortImpl.h"

#include <cstring>

#ifdef HAVE_CUDA
#include <cuda.h>

#include "Logger/Logger.h"
#define checkCudaErrors(err) CHECK_EQ(err, CUDA_SUCCESS)

struct BytePermutationIndex {
  const int32_t* idx_buff;
  uint32_t chosen_bytes;

  __host__ __device__ uint64_t operator()(const uint64_t byte_idx) const {
    const auto row_idx = byte_idx / chosen_bytes;
    const auto byte_offset = byte_idx - row_idx * chosen_bytes;
    return static_cast<uint64_t>(idx_buff[row_idx]) * chosen_bytes + byte_offset;
  }
};

template <typename T>
void sort_on_gpu(T* val_buff,
                 int32_t* idx_buff,
                 const uint64_t entry_count,
                 const bool desc,
                 ThrustAllocator& alloc,
                 CUstream cuda_stream) {
  thrust::device_ptr<T> key_ptr(val_buff);
  thrust::device_ptr<int32_t> idx_ptr(idx_buff);
  thrust::sequence(idx_ptr, idx_ptr + entry_count);
  if (desc) {
    thrust::sort_by_key(thrust::cuda::par(alloc).on(cuda_stream),
                        key_ptr,
                        key_ptr + entry_count,
                        idx_ptr,
                        thrust::greater<T>());
  } else {
    thrust::sort_by_key(thrust::cuda::par(alloc).on(cuda_stream),
                        key_ptr,
                        key_ptr + entry_count,
                        idx_ptr);
  }
  checkCudaErrors(cuStreamSynchronize(cuda_stream));
}

template <typename T>
void apply_permutation_on_gpu(T* val_buff,
                              int32_t* idx_buff,
                              const uint64_t entry_count,
                              ThrustAllocator& alloc,
                              CUstream cuda_stream) {
  thrust::device_ptr<T> key_ptr(val_buff);
  thrust::device_ptr<int32_t> idx_ptr(idx_buff);
  const size_t buf_size = entry_count * sizeof(T);
  T* raw_ptr = reinterpret_cast<T*>(alloc.allocate(buf_size));
  thrust::device_ptr<T> tmp_ptr(raw_ptr);
  thrust::copy(
      thrust::cuda::par(alloc).on(cuda_stream), key_ptr, key_ptr + entry_count, tmp_ptr);
  checkCudaErrors(cuStreamSynchronize(cuda_stream));
  thrust::gather(thrust::cuda::par(alloc).on(cuda_stream),
                 idx_ptr,
                 idx_ptr + entry_count,
                 tmp_ptr,
                 key_ptr);
  checkCudaErrors(cuStreamSynchronize(cuda_stream));
  alloc.deallocate(reinterpret_cast<int8_t*>(raw_ptr), buf_size);
}

void apply_byte_permutation_on_gpu(int64_t* val_buff,
                                   int32_t* idx_buff,
                                   const uint64_t entry_count,
                                   const uint32_t chosen_bytes,
                                   ThrustAllocator& alloc,
                                   CUstream cuda_stream) {
  auto key_ptr = thrust::device_pointer_cast(reinterpret_cast<int8_t*>(val_buff));
  const size_t buf_size = entry_count * chosen_bytes;
  auto raw_ptr = alloc.allocate(buf_size);
  thrust::device_ptr<int8_t> tmp_ptr(raw_ptr);
  thrust::copy(
      thrust::cuda::par(alloc).on(cuda_stream), key_ptr, key_ptr + buf_size, tmp_ptr);
  checkCudaErrors(cuStreamSynchronize(cuda_stream));

  const auto offsets_begin =
      thrust::make_transform_iterator(thrust::make_counting_iterator<uint64_t>(0),
                                      BytePermutationIndex{idx_buff, chosen_bytes});
  thrust::gather(thrust::cuda::par(alloc).on(cuda_stream),
                 offsets_begin,
                 offsets_begin + buf_size,
                 tmp_ptr,
                 key_ptr);
  checkCudaErrors(cuStreamSynchronize(cuda_stream));
  alloc.deallocate(raw_ptr, buf_size);
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

void apply_byte_permutation_on_cpu(int64_t* val_buff,
                                   int32_t* idx_buff,
                                   const uint64_t entry_count,
                                   int64_t* tmp_buff,
                                   const uint32_t chosen_bytes) {
  auto val_bytes = reinterpret_cast<int8_t*>(val_buff);
  auto tmp_bytes = reinterpret_cast<int8_t*>(tmp_buff);
  const size_t buf_size = entry_count * chosen_bytes;
  std::memcpy(tmp_bytes, val_bytes, buf_size);
  for (uint64_t row_idx = 0; row_idx < entry_count; ++row_idx) {
    std::memcpy(val_bytes + row_idx * chosen_bytes,
                tmp_bytes + static_cast<uint64_t>(idx_buff[row_idx]) * chosen_bytes,
                chosen_bytes);
  }
}
#endif

void sort_on_gpu(int64_t* val_buff,
                 int32_t* idx_buff,
                 const uint64_t entry_count,
                 const bool desc,
                 const uint32_t chosen_bytes,
                 ThrustAllocator& alloc,
                 CUstream cuda_stream) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 1:
      sort_on_gpu(reinterpret_cast<int8_t*>(val_buff),
                  idx_buff,
                  entry_count,
                  desc,
                  alloc,
                  cuda_stream);
      break;
    case 2:
      sort_on_gpu(reinterpret_cast<int16_t*>(val_buff),
                  idx_buff,
                  entry_count,
                  desc,
                  alloc,
                  cuda_stream);
      break;
    case 4:
      sort_on_gpu(reinterpret_cast<int32_t*>(val_buff),
                  idx_buff,
                  entry_count,
                  desc,
                  alloc,
                  cuda_stream);
      break;
    case 8:
      sort_on_gpu(val_buff, idx_buff, entry_count, desc, alloc, cuda_stream);
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
                              CUstream cuda_stream) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 1:
      apply_permutation_on_gpu(
          reinterpret_cast<int8_t*>(val_buff), idx_buff, entry_count, alloc, cuda_stream);
      break;
    case 2:
      apply_permutation_on_gpu(reinterpret_cast<int16_t*>(val_buff),
                               idx_buff,
                               entry_count,
                               alloc,
                               cuda_stream);
      break;
    case 4:
      apply_permutation_on_gpu(reinterpret_cast<int32_t*>(val_buff),
                               idx_buff,
                               entry_count,
                               alloc,
                               cuda_stream);
      break;
    case 8:
      apply_permutation_on_gpu(val_buff, idx_buff, entry_count, alloc, cuda_stream);
      break;
    default:
      apply_byte_permutation_on_gpu(
          val_buff, idx_buff, entry_count, chosen_bytes, alloc, cuda_stream);
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
      apply_byte_permutation_on_cpu(
          val_buff, idx_buff, entry_count, tmp_buff, chosen_bytes);
  }
#endif
}
