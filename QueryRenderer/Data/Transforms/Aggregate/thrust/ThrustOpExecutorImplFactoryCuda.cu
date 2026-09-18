/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplFactory.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

namespace QueryRenderer {

std::unique_ptr<ThrustOpExecutorInterface>
ThrustExecutorImplFactory::createXformOpExecutorCuda(
    DataMgrThrustContext& thrust_context) {
  return std::make_unique<ThrustOpExecutorImpl<ThrustDeviceSystem::kCuda>>(
      thrust_context);
}

}  // namespace QueryRenderer
