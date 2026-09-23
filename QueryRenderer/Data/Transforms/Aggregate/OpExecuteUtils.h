/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpInterface.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

struct OpExecuteUtils {
  static XformOp::OpResult executeThrustDependencyOp(
      const AggDepOp& base_op,
      ThrustDependencyOpInterface& thrust_op,
      const XformShPtr& parent_xform,
      const LayoutAttrInfoSet& inputs,
      Data_Namespace::DataMgr& data_mgr,
      QueryRendererContext& ctx,
      const std::string& evaluator_name,
      const XformOp::DependencyOpResultsMap& dependency_results);

  static XformOp::OpResult executeThrustOp(
      const AggOp& base_op,
      ThrustOpInterface& thrust_op,
      const XformShPtr& parent_xform,
      const LayoutAttrInfoSet& inputs,
      Data_Namespace::DataMgr& data_mgr,
      QueryRendererContext& ctx,
      const std::string& evaluator_name,
      InteropBufferMgr* mapped_buffers,
      const XformOp::DependencyOpResultsMap& dependency_results);

  static AggDataList mergeOpResults(const OpType op_type,
                                    std::vector<AggDataList>&& results_to_merge);
};

}  // namespace QueryRenderer
