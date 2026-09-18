/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <boost/noncopyable.hpp>

#include "DataMgr/Allocators/ThrustAllocator.h"
#include "DataMgr/Allocators/thrust/DataMgrAllocationPolicy.h"

class ThrustAllocator;

namespace QueryRenderer {

/**
 * @brief Execution context for thrust code. This context currently stores an allocator
 * and a device execution policy instance to be used when running thrust endpoints.
 * Defaults to using a ThrustAllocator and DataMgrAllocationPolicy.
 * @code
 * ThrustExecutionContext<Allocator, DeviceExecutionPolicy>
 * thrust_context(std::move(allocator));
 * ...
 * thrust::sort(thrust_context.getDevicePolicy(), ...);
 * @endcode
 */
class DataMgrThrustContext {
 public:
  DataMgrThrustContext(ThrustAllocator&& in_allocator);
  DataMgrThrustContext(DataMgrThrustContext&& other);
  ~DataMgrThrustContext();

  DataMgrThrustContext(const DataMgrThrustContext&) = delete;
  DataMgrThrustContext& operator=(const DataMgrThrustContext&) = delete;

  const Data_Namespace::DataMgrAllocationPolicy& getDevicePolicy() const;

  ThrustAllocator& getAllocator();

 private:
  ThrustAllocator allocator_;
  class details;
  std::unique_ptr<details> details_;
};

}  // namespace QueryRenderer
