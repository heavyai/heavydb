/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplFactory.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

namespace QueryRenderer {

std::unique_ptr<ThrustOpExecutorInterface>
ThrustExecutorImplFactory::createXformOpExecutorTbb(
    DataMgrThrustContext& thrust_context) {
  return std::make_unique<ThrustOpExecutorImpl<ThrustDeviceSystem::kTbb>>(thrust_context);
}

}  // namespace QueryRenderer
