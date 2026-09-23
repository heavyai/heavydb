/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImpl.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorUtils.h"
#include "QueryRenderer/Utils/thrust/ThrustDeviceSystem.h"

namespace QueryRenderer {

constexpr static ThrustDeviceSystem active_device_system =
#ifdef HAVE_CUDA
    ThrustDeviceSystem::kCuda;
#else
    ThrustDeviceSystem::kTbb;
#endif  // HAVE_CUDA

// specialization based on the current active_device_system
template <>
class ThrustOpExecutorImpl<active_device_system> : public ThrustOpExecutorInterface {
 public:
  ThrustOpExecutorImpl(DataMgrThrustContext& thrust_context)
      : ThrustOpExecutorInterface(thrust_context) {}

  ~ThrustOpExecutorImpl() override = default;

  AggDataList executeCountOp(const InteropBufferInfo& interop_buffer_info,
                             const LayoutAttrInfo& input_info) final {
    return executeCountOp(ThrustOpExecutorUtils::createThrustOpExecutionState(
        interop_buffer_info, input_info));
  }

  AggDataList executeValidOp(const InteropBufferInfo& interop_buffer_info,
                             const LayoutAttrInfo& input_info) final {
    return executeValidOp(ThrustOpExecutorUtils::createThrustOpExecutionState(
        interop_buffer_info, input_info));
  }

  AggDataList executeMissingOp(const InteropBufferInfo& interop_buffer_info,
                               const LayoutAttrInfo& input_info) final {
    return executeMissingOp(ThrustOpExecutorUtils::createThrustOpExecutionState(
        interop_buffer_info, input_info));
  }

  AggDataList executeMinOp(const InteropBufferInfo& interop_buffer_info,
                           const LayoutAttrInfo& input_info) final {
    return executeMinOp(ThrustOpExecutorUtils::createThrustOpExecutionState(
        interop_buffer_info, input_info));
  }

  AggDataList executeMaxOp(const InteropBufferInfo& interop_buffer_info,
                           const LayoutAttrInfo& input_info) final {
    return executeMaxOp(ThrustOpExecutorUtils::createThrustOpExecutionState(
        interop_buffer_info, input_info));
  }

  AggDataList executeSumOp(const InteropBufferInfo& interop_buffer_info,
                           const LayoutAttrInfo& input_info) final {
    return executeSumOp(ThrustOpExecutorUtils::createThrustOpExecutionState(
        interop_buffer_info, input_info));
  }

  AggDataList executeSqDiffSumOp(const InteropBufferInfo& interop_buffer_info,
                                 const LayoutAttrInfo& input_info,
                                 const AnyDataType& avg_result) final {
    return executeSqDiffSumOp(ThrustOpExecutorUtils::createThrustOpExecutionState(
                                  interop_buffer_info, input_info),
                              avg_result);
  }

  AggDataList executeDistinctOp(const InteropBufferInfo& interop_buffer_info,
                                const LayoutAttrInfo& input_info) final {
    return executeDistinctOp(ThrustOpExecutorUtils::createThrustOpExecutionState(
        interop_buffer_info, input_info));
  }

  AggDataList executeDistinctHistogramOp(const InteropBufferInfo& interop_buffer_info,
                                         const LayoutAttrInfo& input_info,
                                         const DistinctHistogramProps& props) final {
    return executeDistinctHistogramOp(ThrustOpExecutorUtils::createThrustOpExecutionState(
                                          interop_buffer_info, input_info),
                                      props);
  }

  AggDataList executeQuantileOp(const AggDataList& distinct_histogram_results,
                                const QuantileProps& props) final;

  AggDataList executeTopBottomKOp(const AggDataList& dependency_results,
                                  const bool is_ascending,
                                  const uint16_t k) final;

 private:
  AggDataList executeCountOp(ThrustOpExecutionState execution_state);
  AggDataList executeValidOp(ThrustOpExecutionState execution_state);
  AggDataList executeMissingOp(ThrustOpExecutionState execution_state);
  AggDataList executeMinOp(ThrustOpExecutionState execution_state);
  AggDataList executeMaxOp(ThrustOpExecutionState execution_state);
  AggDataList executeSumOp(ThrustOpExecutionState execution_state);
  AggDataList executeSqDiffSumOp(ThrustOpExecutionState execution_state,
                                 const AnyDataType& avg_result);
  AggDataList executeDistinctOp(ThrustOpExecutionState execution_state);
  AggDataList executeDistinctHistogramOp(ThrustOpExecutionState execution_state,
                                         const DistinctHistogramProps& props);
};

}  // namespace QueryRenderer
