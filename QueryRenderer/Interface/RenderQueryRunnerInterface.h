/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/noncopyable.hpp>

#include "QueryRenderer/Interface/RenderQueryExecuteData.h"
#include "QueryRenderer/Interface/RenderQueryExecuteTimer.h"
#include "QueryRenderer/Interface/RenderQueryInterfaceDeclarations.h"
#include "Shared/Rendering/RenderQueryOptions.h"

class Executor;

namespace QueryRenderer {

class JSONLocation;
enum class RenderQuerySpecialtyType { kNone = 0, kPolys, kLines, kMesh2d };

class RenderQueryRunnerInterface : public boost::noncopyable {
 public:
  virtual ~RenderQueryRunnerInterface() {}

  virtual void notifyQueryExecutionComplete() const = 0;

  virtual RenderQueryParseData executeQueryParse(RenderQueryExecuteTimer&,
                                                 const std::string&,
                                                 const JSONLocation*,
                                                 const RenderQueryOptions&,
                                                 const RenderQuerySpecialtyType) = 0;

  virtual RenderQueryExecuteData executeQuery(RenderQueryExecuteTimer&,
                                              const std::string&,
                                              const JSONLocation*,
                                              const RenderQueryOptions&,
                                              const RenderQuerySpecialtyType,
                                              const heavyai::InSituFlags) = 0;

  virtual std::vector<int32_t> getStringIds(
      const QueryDataLayout& query_data_layout,
      const std::string& column_name,
      const std::vector<std::string>& column_values_to_convert,
      const ResultSet& results,
      const bool warn = false) const = 0;

  virtual std::vector<std::string> getStringsFromIds(
      const QueryDataLayout& query_data_layout,
      const std::string& column_name,
      const std::vector<int32_t>& column_value_ids,
      const ResultSet& results) const = 0;
};

using RenderQueryRunnerUqPtr = std::unique_ptr<RenderQueryRunnerInterface>;

}  // namespace QueryRenderer
