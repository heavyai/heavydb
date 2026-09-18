/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/Allocators/thrust/DataMgrAllocationPolicy.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContext.h"

namespace QueryRenderer {

class DataMgrThrustContext::details {
 public:
  explicit details(const ThrustAllocator& allocator) : device_policy(allocator) {}

  const Data_Namespace::DataMgrAllocationPolicy device_policy;
};

}  // namespace QueryRenderer
