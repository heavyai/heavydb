/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorInterface.h"
#include "QueryRenderer/Utils/thrust/ThrustDeviceSystem.h"

namespace QueryRenderer {

struct ThrustExecutorImplFactory {
 public:
  static std::unique_ptr<ThrustOpExecutorInterface> createXformOpExecutor(
      const ThrustDeviceSystem device_system,
      DataMgrThrustContext& thrust_context) {
    switch (device_system) {
      case ThrustDeviceSystem::kCuda:
#ifdef HAVE_CUDA
        return createXformOpExecutorCuda(thrust_context);
#else
        UNREACHABLE();
#endif
      case ThrustDeviceSystem::kTbb:
#ifndef HAVE_CUDA
        return createXformOpExecutorTbb(thrust_context);
#else
        UNREACHABLE();
#endif
    }

    UNREACHABLE();
    return nullptr;
  }

 private:
#ifdef HAVE_CUDA
  static std::unique_ptr<ThrustOpExecutorInterface> createXformOpExecutorCuda(
      DataMgrThrustContext& thrust_context);
#else
  static std::unique_ptr<ThrustOpExecutorInterface> createXformOpExecutorTbb(
      DataMgrThrustContext& thrust_context);
#endif  // HAVE_CUDA
};

}  // namespace QueryRenderer
