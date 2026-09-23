/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplFactory.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContext.h"

namespace QueryRenderer {

class ThrustOpExecutor {
 public:
  ThrustOpExecutor(DataMgrThrustContext& thrust_context) {
    switch (active_device_system) {
      case ThrustDeviceSystem::kCuda:
#ifdef HAVE_CUDA
        impl_ = ThrustExecutorImplFactory::createXformOpExecutor(
            ThrustDeviceSystem::kCuda, thrust_context);
        break;
#else
        UNREACHABLE();
#endif
      case ThrustDeviceSystem::kTbb:
#ifndef HAVE_CUDA
        impl_ = ThrustExecutorImplFactory::createXformOpExecutor(ThrustDeviceSystem::kTbb,
                                                                 thrust_context);
        break;
#else
        UNREACHABLE();
#endif
    }
  }

  AggDataList executeCountOp(const InteropBufferInfo& interop_buffer_info,
                             const LayoutAttrInfo& input_info) {
    return impl_->executeCountOp(interop_buffer_info, input_info);
  }

  AggDataList executeValidOp(const InteropBufferInfo& interop_buffer_info,
                             const LayoutAttrInfo& input_info) {
    return impl_->executeValidOp(interop_buffer_info, input_info);
  }

  AggDataList executeMissingOp(const InteropBufferInfo& interop_buffer_info,
                               const LayoutAttrInfo& input_info) {
    return impl_->executeMissingOp(interop_buffer_info, input_info);
  }

  AggDataList executeMinOp(const InteropBufferInfo& interop_buffer_info,
                           const LayoutAttrInfo& input_info) {
    return impl_->executeMinOp(interop_buffer_info, input_info);
  }

  AggDataList executeMaxOp(const InteropBufferInfo& interop_buffer_info,
                           const LayoutAttrInfo& input_info) {
    return impl_->executeMaxOp(interop_buffer_info, input_info);
  }

  AggDataList executeSumOp(const InteropBufferInfo& interop_buffer_info,
                           const LayoutAttrInfo& input_info) {
    return impl_->executeSumOp(interop_buffer_info, input_info);
  }

  AggDataList executeSqDiffSumOp(const InteropBufferInfo& interop_buffer_info,
                                 const LayoutAttrInfo& input_info,
                                 const AnyDataType& avg_result) {
    return impl_->executeSqDiffSumOp(interop_buffer_info, input_info, avg_result);
  }

  AggDataList executeDistinctOp(const InteropBufferInfo& interop_buffer_info,
                                const LayoutAttrInfo& input_info) {
    return impl_->executeDistinctOp(interop_buffer_info, input_info);
  }

  AggDataList executeDistinctHistogramOp(const InteropBufferInfo& interop_buffer_info,
                                         const LayoutAttrInfo& input_info,
                                         const DistinctHistogramProps& props) {
    return impl_->executeDistinctHistogramOp(interop_buffer_info, input_info, props);
  }

  AggDataList executeQuantileOp(const AggDataList& distinct_histogram_results,
                                const QuantileProps& props) {
    return impl_->executeQuantileOp(distinct_histogram_results, props);
  }

  AggDataList executeTopBottomKOp(const AggDataList& dependency_results,
                                  const bool is_ascending,
                                  const uint16_t k) {
    return impl_->executeTopBottomKOp(dependency_results, is_ascending, k);
  }

 private:
  std::unique_ptr<ThrustOpExecutorInterface> impl_;
};

}  // namespace QueryRenderer
