/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/QueryMeshDataTable.h"

#include <memory>
#include <optional>

#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/MeshDataTableGpuResources.h"
#include "QueryRenderer/Data/Parsers/CrossSectionFormatJson.h"
#include "QueryRenderer/Data/Parsers/RasterMeshFormatJson.h"
#include "QueryRenderer/Data/QueryDataTableQueues.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"
#include "Shared/StringTransform.h"

namespace QueryRenderer {

SqlQueryMeshDataTableJSON::SqlQueryMeshDataTableJSON(QueryRendererContext& ctx,
                                                     const std::string& name,
                                                     const JSONLocation& json_loc)
    : BaseDataTable(DataInputFormat::kSQL, DataOutputFormat::kMesh2d)
    , BaseQueryDataTableSQLJSON(ctx, name, json_loc, RenderQuerySpecialtyType::kMesh2d)
    , mesh_data_format_type_{DataFormatType::kUnknown}
    , gpu_resources_{std::make_unique<MeshDataTableGpuResources>(DataInputFormat::kSQL)} {
  RENDER_LOG_SCOPE();
}

std::vector<GpuId> SqlQueryMeshDataTableJSON::getUsedGpuIds() const {
  return gpu_resources_->getGpuDataMap().getGpuIds();
}

const gfx::IndexBuffer* SqlQueryMeshDataTableJSON::getIndexBuffer(
    const GpuId gpu_id) const {
  auto const& gpu_data = gpu_resources_->getGpuDataMap().getData(gpu_id);
  return (gpu_data.ibo ? gpu_data.ibo->unmapForDraw() : nullptr);
}

bool SqlQueryMeshDataTableJSON::hasData() const {
  auto vbo_layout = getVboQueryDataLayout();
  bool found_data{false};
  if (vbo_layout) {
    gpu_resources_->getGpuDataMap().visitData(
        [&](GpuId gpu_id, MeshDataTablePerGpuData& gpu_data) {
          if ((gpu_data.vbo && gpu_data.vbo->numVertices(vbo_layout) > 0) &&
              (gpu_data.ibo && gpu_data.ibo->numItems())) {
            found_data = true;
            return false;  // stop visitor
          }
          return true;
        });
  }
  return found_data;
}

bool SqlQueryMeshDataTableJSON::hasAttribute(const std::string& attr_name) const {
  const auto layout = getVboQueryDataLayout();
  return (layout ? layout->hasAttribute(attr_name) : false);
}

std::set<std::string> SqlQueryMeshDataTableJSON::getAllAttrNames() const {
  const auto layout = getVboQueryDataLayout();
  if (layout) {
    const auto attrs = layout->getAllAttrNames();
    return std::set<std::string>(attrs.begin(), attrs.end());
  }
  return {};
}

QueryLayoutBufferWkPtr SqlQueryMeshDataTableJSON::getAttributeDataBuffer(
    const GpuId gpu_id,
    const std::string& attr_name) {
  RENDER_LOG_SCOPE_P(gpu_id) << attr_name;
  auto& gpu_data = gpu_resources_->getGpuDataMap().getData(gpu_id);

  RUNTIME_EX_ASSERT(
      gpu_data.vbo,
      createJSONRefError("Cannot get the mesh2d data buffer for " + attr_name +
                         " in table " + query_sql_.getPrimaryTableName() +
                         ". The table's data has not been initialized yet."));

  auto layout = getVboQueryDataLayout();
  CHECK(layout);
  RUNTIME_EX_ASSERT(gpu_data.vbo->hasAttribute(attr_name, *layout),
                    createJSONRefError("Attribute \"" + attr_name +
                                       "\" does not exist in the mesh2d data buffer."));

  return gpu_data.vbo;
}

std::map<GpuId, QueryLayoutBufferWkPtr>
SqlQueryMeshDataTableJSON::getAttributeDataBuffers(const std::string& attr_name) {
  std::map<GpuId, QueryLayoutBufferWkPtr> rtn;

  if (hasData()) {
    const auto vbo_layout = getVboQueryDataLayout();
    gpu_resources_->getGpuDataMap().visitData([&](GpuId gpu_id,
                                                  MeshDataTablePerGpuData& gpu_data) {
      if (gpu_data.vbo && gpu_data.vbo->hasAttribute(attr_name, *vbo_layout)) {
        rtn.emplace(gpu_id, gpu_data.vbo);
      } else {
        RUNTIME_EX_ASSERT(
            !gpu_data.vbo,
            createJSONRefError("Cannot get data buffer for \"" + attr_name +
                               "\". The attribute does not exist in the mesh2d data."));
        // if we reach here, the data is empty, or in other words possible empty query
      }
      return true;
    });
  }

  return rtn;
}

SQLTypeInfo SqlQueryMeshDataTableJSON::getAttributeTypeInfo(
    const std::string& attr_name) const {
  auto layout = getVboQueryDataLayout();
  RUNTIME_EX_ASSERT(
      layout,
      createJSONRefError("Cannot get the layout for attribute \"" + attr_name +
                         "\". The mesh2d data table has no data."));
  return layout->getAttrSQLTypeInfoRef(attr_name);
}

QueryDataType SqlQueryMeshDataTableJSON::getAttributeType(
    const std::string& attr_name) const {
  return convertToQueryDataType(getAttributeBufferType(attr_name));
}

gfx::BufferAttrType SqlQueryMeshDataTableJSON::getAttributeBufferType(
    const std::string& attr_name) const {
  const auto layout = getVboQueryDataLayout();
  RUNTIME_EX_ASSERT(
      layout,
      createJSONRefError("Cannot get attribute type for attribute \"" + attr_name +
                         "\" in mesh2d data table \"" + getName() +
                         "\". The table has not been initialized with data."));

  return layout->getBufferLayout()->getAttributeType(attr_name);
}

gfx::BufferLayoutShPtr SqlQueryMeshDataTableJSON::getAttributeBufferLayout(
    const std::string& attr_name) {
  auto layout = getVboQueryDataLayout();
  RUNTIME_EX_ASSERT(
      layout,
      createJSONRefError("Cannot get the layout for attribute \"" + attr_name +
                         "\". The mesh2d data table '" + getName() + "' has no data."));

  RUNTIME_EX_ASSERT(
      layout->hasAttribute(attr_name),
      createJSONRefError("Cannot find a layout for the attribute \"" + attr_name +
                         "\" in the mesh2d data table '" + getName() + "'."));
  return layout->getBufferLayout();
}

bool SqlQueryMeshDataTableJSON::updateSqlFromJSONObj(const JSONLocation& json_loc) {
  RENDER_LOG_SCOPE();
  query_sql_.updateFromJSONObj(json_loc, ctx_.doHitTest(), name_);
  return true;  // always run the query
}

bool SqlQueryMeshDataTableJSON::updateFromJSONObjInternal(const JSONLocation& json_loc,
                                                          bool do_execute_query) {
  RENDER_LOG_SCOPE();

  const auto format_loc = json_loc.getMember(JSONSchema_v1::Data::kFormatProp);
  CHECK(format_loc.isValid());
  CHECK(format_loc.isObject());

  auto format_type_loc = format_loc.getMember(JSONSchema_v1::Data::kTypeProp);
  CHECK(format_type_loc.isValid());
  CHECK(format_type_loc.isString());

  auto const format_type = to_lower(format_type_loc.getString());

  mesh_data_format_type_ = get_data_format_from_string(format_type);
  switch (mesh_data_format_type_) {
    case DataFormatType::kRasterMesh2d:
      RasterMeshFormatJson::validate(format_loc);
      break;
    case DataFormatType::kCrossSection2d:
      CrossSectionFormatJson::validate(format_loc);
      break;
    case DataFormatType::kCrossSection1d:
    case DataFormatType::kLines:
    case DataFormatType::kUnknown:
      CHECK(false) << to_string(mesh_data_format_type_);
      break;
  }

  return true;  // always run the query
}

bool SqlQueryMeshDataTableJSON::queueQuery(const JSONLocation* json_loc,
                                           const heavyai::InSituFlags insitu_flags) {
  RENDER_LOG_SCOPE();

  bool will_query_run =
      ctx_.getDataTableQueues().addToQueryQueue(shared_from_this(),
                                                *json_loc,
                                                std::nullopt,
                                                heavyai::InSituFlags::kForcedNonInSitu);

  return will_query_run;
}

void SqlQueryMeshDataTableJSON::postRunQuery(bool did_query_execute) {
  RENDER_LOG_SCOPE() << "did_query_execute: " << did_query_execute;
  gpu_resources_->initGpuResourcesFromBuffers(ctx_.getGlobalContext(), name_);
}

bool SqlQueryMeshDataTableJSON::update() {
  // no-op, data should have been generated in updateFromJSONObjInternal
  return false;
}

}  // namespace QueryRenderer
