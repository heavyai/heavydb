/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>

#include <DataMgr/Allocators/ThrustAllocator.h>

namespace Data_Namespace {

namespace detail {
/**
 * @brief A thrust memory resource wrapped around a Data_Namespace::ThrustAllocator that
 * allocates memory via DataMgr. This memory resource wrapper is required to properly use
 * ThrustAllocator as a custom allocator for thrust device containers like
 * thrust::device_vector.
 */
template <typename Pointer>
class DataMgrMemoryResource final : public thrust::mr::memory_resource<Pointer> {
  using base = thrust::mr::memory_resource<Pointer>;

 public:
  DataMgrMemoryResource(ThrustAllocator& thrust_allocator)
      : base(), thrust_allocator_(&thrust_allocator) {}
  DataMgrMemoryResource(const DataMgrMemoryResource& other)
      : base(other), thrust_allocator_(other.thrust_allocator_) {}
  ~DataMgrMemoryResource() final = default;

  /**
   * @brief Overrides a pure virtual function defined in thrust::mr::memory_resource to
   * allocate from a ThrustAllocator
   */
  Pointer do_allocate(std::size_t bytes,
                      std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) final {
    (void)alignment;  // dummy cast to avoid unused warnings
    return Pointer(
        reinterpret_cast<typename thrust::detail::pointer_traits<Pointer>::element_type*>(
            thrust_allocator_->allocate(bytes)));
  }

  /**
   * @brief Overrides a pure virtual function defined in thrust::mr::memory_resource to
   * deallocate memory from a ThrustAllocator
   */
  void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) final {
    (void)alignment;  // dummy cast to avoid unused warnings
    thrust_allocator_->deallocate(
        reinterpret_cast<int8_t*>(thrust::detail::pointer_traits<Pointer>::get(p)),
        bytes);
  }

  __host__ __device__ const ThrustAllocator* getThrustAllocator() const {
    return thrust_allocator_;
  }

 private:
  ThrustAllocator* thrust_allocator_;
};

/**
 * @brief Manages the underlying state of a TypedThrustAllocator. The state consists of:
 *   DataMgrMemoryResource: this instance holds onto a pointer of the ThrustAllocator
 *       which performs generic allocations
 *   thrust::device_ptr_memory_resource: this instance is an adapter that converts the
 *       pointer returned from the DataMgrMemoryResource to a thrust::device_ptr
 */
class TypedThrustAllocatorState {
 public:
  using Pointer = thrust::
      pointer<void, thrust::device_system_tag, thrust::use_default, thrust::use_default>;

  // Need to define a device_ptr_memory_resource here so any implied execution
  // policies can be defined as device execution policies
  using DeviceResource =
      thrust::device_ptr_memory_resource<DataMgrMemoryResource<Pointer>>;

  TypedThrustAllocatorState(ThrustAllocator& thrust_allocator)
      : data_mgr_mem_rsrc_(thrust_allocator), device_rsrc_(&data_mgr_mem_rsrc_) {}

  // Need to override the default copy constructor/operator and move constructor to ensure
  // that the device_rsrc_ is constructed with a pointer to the local data_mgr_mem_rsrc_
  TypedThrustAllocatorState(const TypedThrustAllocatorState& other)
      : data_mgr_mem_rsrc_(other.data_mgr_mem_rsrc_), device_rsrc_(&data_mgr_mem_rsrc_) {}

  TypedThrustAllocatorState(TypedThrustAllocatorState&& other)
      : data_mgr_mem_rsrc_(std::move(other.data_mgr_mem_rsrc_))
      , device_rsrc_(&data_mgr_mem_rsrc_) {}

  __host__ __device__ void operator=(const TypedThrustAllocatorState& other) {
    assert(data_mgr_mem_rsrc_.getThrustAllocator() ==
           other.data_mgr_mem_rsrc_.getThrustAllocator());
    // NOTE: only copying the data_mgr_mem_rsrc_
    // The device_rsrc_ should have already been constructed with a poitner to the local
    // data_mgr_mem_rsrc_ and is therefore up-to-date.
    data_mgr_mem_rsrc_ = other.data_mgr_mem_rsrc_;
  }

  // TODO(croot): handle rvalue operator=?

 protected:
  DataMgrMemoryResource<Pointer> data_mgr_mem_rsrc_;
  DeviceResource device_rsrc_;
};
}  // namespace detail

/**
 * @brief a Templated version of Data_Namespace::ThrustAllocator that can be used as a
 * custom allocator in thrust device containers such as thrust::device_vector.
 * Note that this class derives from thrust::mr::allocator in order to meet the
 * requirements of an Allocator
 * @code
 * // creates a thrust allocator on device 0
 * ThrustAllocator thrust_allocator(data_mgr, 0);
 * // creates a device vector that allocates storage via the above ThrustAllocator
 * thrust::device_vector<int, TypedThrustAllocator<int>>
 * vec(10, TypedThrustAllocator<int>(thrust_allocator));
 * @endcode
 */
template <typename T>
class TypedThrustAllocator
    : public detail::TypedThrustAllocatorState,
      public thrust::mr::allocator<T, detail::TypedThrustAllocatorState::DeviceResource> {
  using Base =
      thrust::mr::allocator<T, detail::TypedThrustAllocatorState::DeviceResource>;

 public:
  TypedThrustAllocator(ThrustAllocator& thrust_allocator)
      : detail::TypedThrustAllocatorState(thrust_allocator), Base(&device_rsrc_) {}

  // Need to override the default copy constructor/operator and move constructor to ensure
  // that our Base(thrust::mr::allocator) is constructed with a pointer to our
  // device_rsrc_ state
  TypedThrustAllocator(const TypedThrustAllocator& other)
      : detail::TypedThrustAllocatorState(other), Base(&device_rsrc_) {}

  TypedThrustAllocator(TypedThrustAllocator&& other)
      : detail::TypedThrustAllocatorState(std::move(other)), Base(&device_rsrc_) {}

  __host__ __device__ void operator=(const TypedThrustAllocator<T>& other) {
    // NOTE: only applying the copy operator to TypedThrustAllocatorState
    // The thrust::mr::allocator should have already been constructed with a poitner to
    // the local state and is therefore up-to-date
    detail::TypedThrustAllocatorState::operator=(other);
  }
};  // namespace Data_Namespace

template <typename T>
using ThrustAllocatorDeviceVector = thrust::device_vector<T, TypedThrustAllocator<T>>;

}  // namespace Data_Namespace
