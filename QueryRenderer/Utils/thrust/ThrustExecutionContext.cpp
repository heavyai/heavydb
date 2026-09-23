/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Utils/thrust/ThrustExecutionContext.h"

#include "DataMgr/Allocators/ThrustAllocator.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

DataMgrThrustContext::DataMgrThrustContext(ThrustAllocator&& in_allocator)
    : allocator_(std::move(in_allocator))
    , details_{std::make_unique<DataMgrThrustContext::details>(allocator_)} {}

DataMgrThrustContext::DataMgrThrustContext(DataMgrThrustContext&& other)
    : allocator_(std::move(other.allocator_))
    , details_{std::make_unique<DataMgrThrustContext::details>(allocator_)} {}

DataMgrThrustContext::~DataMgrThrustContext() {}

const Data_Namespace::DataMgrAllocationPolicy& DataMgrThrustContext::getDevicePolicy()
    const {
  return details_->device_policy;
}

ThrustAllocator& DataMgrThrustContext::getAllocator() {
  return allocator_;
}

}  // namespace QueryRenderer
