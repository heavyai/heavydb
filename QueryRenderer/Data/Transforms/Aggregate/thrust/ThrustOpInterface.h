/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/XformOp.h"
#include "QueryRenderer/Interface/AggDataTypes.h"
#include "QueryRenderer/Interop/InteropBufferInfo.h"

namespace QueryRenderer {

class ThrustOpExecutor;

template <typename T, int NUM_ELEMS>
class ThrustBufferColumn;

struct ThrustOpResultUtils {
  static void checkSingleValueResults(const AggDataList& results);
  static void checkSingleValueResultsForMerge(const AggDataList& results1,
                                              const AggDataList& results2);
  static AggDataList createEmptyVectorType(const QueryDataType data_type);

  static AggDataList createSingularNullFromType(const QueryDataType data_type);
  static AggDataList createSingularValueFromType(const QueryDataType data_type);
};

class ThrustOpInterface {
 private:
  virtual AggDataList executeThrustOp(
      ThrustOpExecutor& executor,
      const InteropBufferInfo& interop_buffer_info,
      const LayoutAttrInfo& input_info,
      const XformOp::DependencyOpResultsMap& dependency_results) const = 0;

  friend struct OpExecuteUtils;
};

class ThrustDependencyOpInterface {
 private:
  virtual AggDataList executeThrustDependencyOp(
      ThrustOpExecutor& executor,
      const XformOp::DependencyOpResultsMap& dependency_results) const = 0;

  friend struct OpExecuteUtils;
};

}  // namespace QueryRenderer
