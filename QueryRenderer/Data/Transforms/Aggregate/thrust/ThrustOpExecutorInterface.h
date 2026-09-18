/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Types.h"
#include "QueryRenderer/Interface/AggDataTypes.h"
#include "QueryRenderer/Interop/InteropBufferInfo.h"
#include "QueryRenderer/Interop/LayoutAttrInfo.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContext.h"

namespace QueryRenderer {

struct DistinctHistogramProps;
struct QuantileProps;

class ThrustOpExecutorInterface {
 public:
  ThrustOpExecutorInterface(DataMgrThrustContext& thrust_context)
      : thrust_context_(thrust_context) {}

  virtual ~ThrustOpExecutorInterface() = default;

  virtual AggDataList executeCountOp(const InteropBufferInfo& interop_buffer_info,
                                     const LayoutAttrInfo& input_info) = 0;

  virtual AggDataList executeValidOp(const InteropBufferInfo& interop_buffer_info,
                                     const LayoutAttrInfo& input_info) = 0;

  virtual AggDataList executeMissingOp(const InteropBufferInfo& interop_buffer_info,
                                       const LayoutAttrInfo& input_info) = 0;

  virtual AggDataList executeMinOp(const InteropBufferInfo& interop_buffer_info,
                                   const LayoutAttrInfo& input_info) = 0;

  virtual AggDataList executeMaxOp(const InteropBufferInfo& interop_buffer_info,
                                   const LayoutAttrInfo& input_info) = 0;

  virtual AggDataList executeSumOp(const InteropBufferInfo& interop_buffer_info,
                                   const LayoutAttrInfo& input_info) = 0;

  virtual AggDataList executeSqDiffSumOp(const InteropBufferInfo& interop_buffer_info,
                                         const LayoutAttrInfo& input_info,
                                         const AnyDataType& avg_result) = 0;

  virtual AggDataList executeDistinctOp(const InteropBufferInfo& interop_buffer_info,
                                        const LayoutAttrInfo& input_info) = 0;

  virtual AggDataList executeDistinctHistogramOp(
      const InteropBufferInfo& interop_buffer_info,
      const LayoutAttrInfo& input_info,
      const DistinctHistogramProps& props) = 0;

  virtual AggDataList executeQuantileOp(const AggDataList& distinct_histogram_results,
                                        const QuantileProps& props) = 0;

  virtual AggDataList executeTopBottomKOp(const AggDataList& dependency_results,
                                          const bool is_ascending,
                                          const uint16_t k) = 0;

 protected:
  DataMgrThrustContext& thrust_context_;
};

}  // namespace QueryRenderer
