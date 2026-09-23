/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseQueryDataTable.h"
#include "QueryRenderer/Data/BaseRowDataTable.h"

#include <map>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

//
// class SqlQueryRowDataTableJSON
//
// Data table class for basic sql driven data tables
// Handles generic row based results (point type queries)
//
class SqlQueryRowDataTableJSON : public BaseRowDataTable,
                                 public BaseQueryDataTableSQLJSON {
 public:
  SqlQueryRowDataTableJSON(QueryRendererContext& ctx,
                           const std::string& name,
                           const JSONLocation& json_loc);
  ~SqlQueryRowDataTableJSON() override {}

  // from BaseQueryDataTable
  bool hasData() const final;
  bool hasAttribute(const std::string& attr_name) const final;
  std::set<std::string> getAllAttrNames() const final;
  QueryLayoutBufferWkPtr getAttributeDataBuffer(const GpuId gpu_id,
                                                const std::string& attr_name) final;
  std::map<GpuId, QueryLayoutBufferWkPtr> getAttributeDataBuffers(
      const std::string& attr_name) final;
  SQLTypeInfo getAttributeTypeInfo(const std::string& attr_name) const final;
  QueryDataType getAttributeType(const std::string& attr_name) const final;
  gfx::BufferAttrType getAttributeBufferType(const std::string& attr_name) const final;
  gfx::BufferLayoutShPtr getAttributeBufferLayout(const std::string& attr_name) final;

  std::vector<GpuId> getUsedGpuIds() const final;

  // from BaseQueryDataTableSQLJSON
  QueryDataLayoutShPtr getVboQueryDataLayout() const final;

 private:
  // from BaseQueryDataTableSQLJSON
  bool isInternalCacheUpToDate() final;
  bool updateSqlFromJSONObj(const JSONLocation& json_loc) final;
  bool updateFromJSONObjInternal(const JSONLocation& json_loc,
                                 bool do_execute_query) final;
  bool queueQuery(const JSONLocation* json_loc,
                  const heavyai::InSituFlags insitu_flags) final;
  void postRunQuery(bool did_query_execute) final;

  // from BaseDataTable
  bool update() final;
};

}  // namespace QueryRenderer
