/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"

#include "QueryRenderer/Data/Enums/DataFormatType.h"
#include "QueryRenderer/Data/MeshDataTableGpuResources.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

//
// class SqlQueryMeshDataTableJSON
//
// Implements mesh based data table
// Only support definition via sql, no inline vega data
//
class SqlQueryMeshDataTableJSON : public BaseDataTable, public BaseQueryDataTableSQLJSON {
 public:
  SqlQueryMeshDataTableJSON(QueryRendererContext& ctx,
                            const std::string& name,
                            const JSONLocation& json_loc);
  ~SqlQueryMeshDataTableJSON() override = default;

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

  std::vector<GpuId> getUsedGpuIds() const final;

  // local methods
  const gfx::IndexBuffer* getIndexBuffer(const GpuId gpu_id) const;

 private:
  DataFormatType mesh_data_format_type_;
  std::unique_ptr<MeshDataTableGpuResources> gpu_resources_;

  // from BaseQueryDataTableSQLJSON
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
