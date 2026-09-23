/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif
#include <thrust/device_ptr.h>

#include "GfxDriver/Resources/BufferLayout.h"
#include "GfxDriver/Resources/Enums.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/QueryDataLayout.h"

namespace QueryRenderer {

template <typename, int>
class ThrustBufferColumnIterator;

template <typename T, int NUM_ELEMS = 1>
struct ThrustBufferColumnIteratorBase {
  using TypedIterator = typename ::thrust::device_ptr<T>;
  using Value = typename ::thrust::iterator_value<TypedIterator>::type;
  using System = typename ::thrust::iterator_system<TypedIterator>::type;
  using Traversal = typename ::thrust::iterator_traversal<TypedIterator>::type;
  using Reference = typename ::thrust::iterator_reference<TypedIterator>::type;
  using DifferenceType = std::ptrdiff_t;

  using type = ::thrust::iterator_facade<ThrustBufferColumnIterator<T, NUM_ELEMS>,
                                         Value,
                                         System,
                                         Traversal,
                                         Reference,
                                         std::ptrdiff_t>;
};

template <typename T, int NUM_ELEMS = 1>
class ThrustBufferColumnIterator
    : public ThrustBufferColumnIteratorBase<T, NUM_ELEMS>::type {
 public:
  __host__ __device__ ThrustBufferColumnIterator() {}

  __host__ __device__ explicit ThrustBufferColumnIterator(
      ::thrust::device_ptr<T> const& iter)
      : iterator_{iter} {}

  ThrustBufferColumnIterator(const ::thrust::device_ptr<int8_t>& first,
                             const std::ptrdiff_t offset,
                             const std::ptrdiff_t stride)
      : iterator_{}, stride_{stride} {
    int8_t* byte_ptr = ::thrust::raw_pointer_cast(first + offset);
    iterator_ = ::thrust::device_pointer_cast(reinterpret_cast<T*>(byte_ptr));
  }

 protected:
  using super_t = typename ThrustBufferColumnIteratorBase<T, NUM_ELEMS>::type;

 private:
  // __thrust_exec_check_disable__
  __host__ __device__ typename super_t::reference dereference() const {
    return *iterator_;
  }

  // __thrust_exec_check_disable__
  template <typename TT, int NE>
  __host__ __device__ bool equal(ThrustBufferColumnIterator<TT, NE> const& x) const {
    return iterator_ == x.iterator_ && stride_ == x.stride_;
  }

  // __thrust_exec_check_disable__
  __host__ __device__ void increment() {
    T* type_ptr = ::thrust::raw_pointer_cast(iterator_);
    auto byte_ptr =
        ::thrust::device_pointer_cast(reinterpret_cast<int8_t*>(type_ptr)) + stride_;
    int8_t* byte_type_ptr = ::thrust::raw_pointer_cast(byte_ptr);
    iterator_ = ::thrust::device_pointer_cast(reinterpret_cast<T*>(byte_type_ptr));
  }

  // __thrust_exec_check_disable__
  __host__ __device__ void decrement() {
    T* type_ptr = ::thrust::raw_pointer_cast(iterator_);
    auto byte_ptr =
        ::thrust::device_pointer_cast(reinterpret_cast<int8_t*>(type_ptr)) - stride_;
    int8_t* byte_type_ptr = ::thrust::raw_pointer_cast(byte_ptr);
    iterator_ = ::thrust::device_pointer_cast(reinterpret_cast<T*>(byte_type_ptr));
  }

  // __thrust_exec_check_disable__
  __host__ __device__ void advance(typename super_t::difference_type n) {
    T* type_ptr = ::thrust::raw_pointer_cast(iterator_);
    auto byte_ptr =
        ::thrust::device_pointer_cast(reinterpret_cast<int8_t*>(type_ptr)) + stride_ * n;
    int8_t* byte_type_ptr = ::thrust::raw_pointer_cast(byte_ptr);
    iterator_ = ::thrust::device_pointer_cast(reinterpret_cast<T*>(byte_type_ptr));
  }

  // __thrust_exec_check_disable__
  __host__ __device__ typename super_t::difference_type distance_to(
      ThrustBufferColumnIterator<T, NUM_ELEMS> const& y) const {
    T* their_type_ptr = ::thrust::raw_pointer_cast(y.iterator_);
    auto their_byte_ptr =
        ::thrust::device_pointer_cast(reinterpret_cast<int8_t*>(their_type_ptr));

    T* our_type_ptr = ::thrust::raw_pointer_cast(iterator_);
    auto our_byte_ptr =
        ::thrust::device_pointer_cast(reinterpret_cast<int8_t*>(our_type_ptr));

    return (their_byte_ptr - our_byte_ptr) / stride_;
  }

 private:
  friend class ::thrust::iterator_core_access;

  ::thrust::device_ptr<T> iterator_;
  typename super_t::difference_type stride_;
};

template <typename T, int NUM_ELEMS = 1>
class ThrustBufferColumn {
 public:
  using Iterator = ::thrust::device_ptr<int8_t>;
  using DifferenceType = std::ptrdiff_t;
  using ColumnIterator = ThrustBufferColumnIterator<T, NUM_ELEMS>;

  ThrustBufferColumn(const Iterator& first,
                     const DifferenceType offset,
                     const DifferenceType stride,
                     const size_t num_verts)
      : first_(first), offset_(offset), stride_(stride), num_verts_(num_verts) {}

  ColumnIterator begin(void) const { return ColumnIterator(first_, offset_, stride_); }

  ColumnIterator end(void) const {
    return ColumnIterator(first_, offset_, stride_) + num_verts_;
  }

 protected:
  Iterator first_;
  DifferenceType offset_;
  DifferenceType stride_;
  const size_t num_verts_;
};

template <typename T, int NUM_ELEMS = 1>
std::shared_ptr<ThrustBufferColumn<T, NUM_ELEMS>> createThrustBufferColumn(
    const gfx::BufferMemoryDescriptor& interop_descriptor,
    const std::string& attr_name,
    const gfx::BufferLayoutShPtr& data_layout) {
  const auto& layout_attr_info = data_layout->getAttributeInfo(attr_name);
  int offset{0}, stride{0};
  switch (data_layout->getLayoutType()) {
    case gfx::BufferLayoutType::kInterleaved:
      offset = layout_attr_info.offset;
      stride = data_layout->getNumBytesPerItem();
      break;
    default:
      throw std::runtime_error(
          "Cannot build a thrust vbo column iterator from layout of type " +
          gfx::to_string(data_layout->getLayoutType()));
  }
  const auto num_verts = interop_descriptor.num_bytes / data_layout->getNumBytesPerItem();
  return std::make_shared<ThrustBufferColumn<T, NUM_ELEMS>>(
      thrust::device_pointer_cast(interop_descriptor.handle), offset, stride, num_verts);
}

}  // namespace QueryRenderer
