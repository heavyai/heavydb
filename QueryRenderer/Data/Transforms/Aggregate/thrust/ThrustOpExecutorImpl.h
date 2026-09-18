/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorInterface.h"
#include "QueryRenderer/Utils/thrust/ThrustDeviceSystem.h"

namespace QueryRenderer {

template <ThrustDeviceSystem device_system>
class ThrustOpExecutorImpl : public ThrustOpExecutorInterface {
 public:
  ThrustOpExecutorImpl(DataMgrThrustContext& thrust_context)
      : ThrustOpExecutorInterface(thrust_context) {}
  ~ThrustOpExecutorImpl() override = default;
};

}  // namespace QueryRenderer
