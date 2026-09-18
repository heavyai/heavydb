/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseLineDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"

#include "GfxDriver/Resources/Types.h"

namespace QueryRenderer {

//
// class SqlQueryLineDataTableJSON
//
// Data table class for sql driven line data tables
//
class SqlQueryLineDataTableJSON : public BaseLineDataTable,
                                  public BaseQueryDataTableSQLJSON {
 public:
  SqlQueryLineDataTableJSON(QueryRendererContext& ctx,
                            const std::string& name,
                            const JSONLocation& json_loc);
  ~SqlQueryLineDataTableJSON() override = default;

  // from BaseDataTable
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

  std::vector<GpuId> getUsedGpuIds() const override;

  // from BaseQueryDataTableSQLJSON
  QueryDataLayoutShPtr getVboQueryDataLayout() const final;
  QueryDataLayoutShPtr getSsboQueryDataLayout() const final;

  bool jsonNeedsUpdating() const final { return true; }

 private:
  // from BaseQueryDataTableSQLJSON
  // always false for lines
  bool isInternalCacheUpToDate() final { return false; }

  // Update the Sql string from JSON
  // returns true if the query should run
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
