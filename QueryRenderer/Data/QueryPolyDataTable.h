/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BasePolyDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"

#include "GfxDriver/Resources/Types.h"

namespace QueryRenderer {

//
// class SqlQueryPolyDataTableJSON
//
// Data table class for sql driven poly data tables
//
class SqlQueryPolyDataTableJSON : public BasePolyDataTable,
                                  public BaseQueryDataTableSQLJSON {
 public:
  SqlQueryPolyDataTableJSON(QueryRendererContext& ctx,
                            const std::string& name,
                            const JSONLocation& json_loc);
  ~SqlQueryPolyDataTableJSON() override = default;

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

  bool jsonNeedsUpdating() const final { return true; }

  // local methods
  bool hasQueryTypeChanged() const { return query_type_changed_; }

 protected:
  // from BaseQueryDataTableSQLJSON
  void resetStateFlags() final;

 private:
  OptionalStr sql_query_str_override_;
  std::string polys_key_;
  std::string facts_key_;
  std::string agg_expr_;
  std::string filter_expr_;
  std::string facts_table_name_;
  bool query_type_changed_;

  // Need to preserve the table info from prior to updateSqlFromJSONObj
  // to check in updateFromJSONObjInternal
  SQLSelectedTableContainerShPtr curr_table_info_;

  // from BaseQueryDataTableSQLJSON
  bool isInternalCacheUpToDate() final { return false; }
  bool updateSqlFromJSONObj(const JSONLocation& json_loc) final;

  bool updateFromJSONObjInternal(const JSONLocation& json_loc,
                                 bool do_execute_query) final;

  bool queueQuery(const JSONLocation* json_loc,
                  const heavyai::InSituFlags insitu_flags) final;
  void postRunQuery(bool did_query_execute) final;

  // from BaseDataTable
  bool update() final;

  // local methods
  const std::string& getSqlStrToUseRef() const;
};

}  // namespace QueryRenderer
