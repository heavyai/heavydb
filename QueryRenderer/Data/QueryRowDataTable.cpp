/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/QueryRowDataTable.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/QueryDataTableQueues.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/QueryRendererContext.h"

using ::gfx::BufferAttrType;
using ::gfx::BufferLayoutShPtr;

namespace QueryRenderer {

SqlQueryRowDataTableJSON::SqlQueryRowDataTableJSON(QueryRendererContext& ctx,
                                                   const std::string& name,
                                                   const JSONLocation& json_loc)
    : BaseRowDataTable(DataInputFormat::kSQL)
    , BaseQueryDataTableSQLJSON(ctx, name, json_loc, RenderQuerySpecialtyType::kNone) {
  RENDER_LOG_SCOPE();
}

bool SqlQueryRowDataTableJSON::hasData() const {
  return gpu_resources_->hasVerticesForLayout(getVboQueryDataLayout());
}

bool SqlQueryRowDataTableJSON::hasAttribute(const std::string& attr_name) const {
  const auto layout = getVboQueryDataLayout();
  return (layout ? layout->hasAttribute(attr_name) : false);
}

std::set<std::string> SqlQueryRowDataTableJSON::getAllAttrNames() const {
  const auto layout = getVboQueryDataLayout();
  if (layout) {
    const auto attrs = layout->getAllAttrNames();
    return std::set<std::string>(attrs.begin(), attrs.end());
  }
  return {};
}

QueryLayoutBufferWkPtr SqlQueryRowDataTableJSON::getAttributeDataBuffer(
    const GpuId gpu_id,
    const std::string& attr_name) {
  RENDER_LOG_SCOPE_P(gpu_id) << attr_name;
  auto& gpu_data = gpu_resources_->getGpuDataMap().getData(gpu_id);

  RUNTIME_EX_ASSERT(
      gpu_data.vbo,
      createJSONRefError("Cannot get the data buffer for " + attr_name + " in table " +
                         query_sql_.getPrimaryTableName() +
                         ". The table's vbo has not been initialized yet."));

  auto layout = getVboQueryDataLayout();
  CHECK(layout);
  RUNTIME_EX_ASSERT(
      gpu_data.vbo->hasAttribute(attr_name, *layout),
      createJSONRefError("Attribute \"" + attr_name + "\" does not exist in VBO."));

  return gpu_data.vbo;
}

std::map<GpuId, QueryLayoutBufferWkPtr> SqlQueryRowDataTableJSON::getAttributeDataBuffers(
    const std::string& attr_name) {
  return gpu_resources_->getDataBuffers();
}

SQLTypeInfo SqlQueryRowDataTableJSON::getAttributeTypeInfo(
    const std::string& attr_name) const {
  auto layout = getVboQueryDataLayout();
  RUNTIME_EX_ASSERT(
      layout,
      createJSONRefError("Cannot get the layout for attribute \"" + attr_name +
                         "\". The vega data table has no data."));
  return layout->getAttrSQLTypeInfoRef(attr_name);
}

QueryDataType SqlQueryRowDataTableJSON::getAttributeType(
    const std::string& attr_name) const {
  return convertToQueryDataType(getAttributeBufferType(attr_name));
}

BufferAttrType SqlQueryRowDataTableJSON::getAttributeBufferType(
    const std::string& attr_name) const {
  const auto layout = getVboQueryDataLayout();
  RUNTIME_EX_ASSERT(
      layout,
      createJSONRefError("Cannot get attribute type for attribute \"" + attr_name +
                         "\" in vega table \"" + getName() +
                         "\". The table has not been initialized with data yet."));

  return layout->getBufferLayout()->getAttributeType(attr_name);
}

BufferLayoutShPtr SqlQueryRowDataTableJSON::getAttributeBufferLayout(
    const std::string& attr_name) {
  auto layout = getVboQueryDataLayout();
  RUNTIME_EX_ASSERT(
      layout,
      createJSONRefError("Cannot get the layout for attribute \"" + attr_name +
                         "\". The vega data table has no data."));
  RUNTIME_EX_ASSERT(layout->hasAttribute(attr_name),
                    createJSONRefError("Cannot find a layout for the attribute \"" +
                                       attr_name + "\" in the vega data table."));
  return layout->getBufferLayout();
}

QueryDataLayoutShPtr SqlQueryRowDataTableJSON::getVboQueryDataLayout() const {
  auto rtn = BaseQueryDataTableSQLJSON::getVboQueryDataLayout();
  if (rtn) {
    return rtn;
  }

  return gpu_resources_->getDataLayout();
}

std::vector<GpuId> SqlQueryRowDataTableJSON::getUsedGpuIds() const {
  return gpu_resources_->getGpuDataMap().getGpuIds();
}

bool SqlQueryRowDataTableJSON::isInternalCacheUpToDate() {
  RENDER_LOG_SCOPE();
  if (!gpu_resources_->getGpuDataMap().isEmpty()) {
    if (!buffer_layouts_.vbo_layout) {
      RENDER_LOG() << "vbo_query_data_layout_ == null";
      return false;
    }

    return gpu_resources_->hasDataForLayout(buffer_layouts_.vbo_layout);
  }

  return true;
}

bool SqlQueryRowDataTableJSON::updateSqlFromJSONObj(const JSONLocation& json_loc) {
  RENDER_LOG_SCOPE();
  auto curr_sql_query_str = query_sql_.getSqlQueryStr();
  auto curr_query_opts = query_sql_.getRenderQueryOptions();

  query_sql_.updateFromJSONObj(json_loc, ctx_.doHitTest(), name_);

  if (curr_sql_query_str != query_sql_.getSqlQueryStr() ||
      curr_query_opts != query_sql_.getRenderQueryOptions()) {
    // sql changed, re-run the query
    return true;
  }
  return false;
}

bool SqlQueryRowDataTableJSON::updateFromJSONObjInternal(const JSONLocation& json_loc,
                                                         bool do_execute_query) {
  RENDER_LOG_SCOPE();
  if (do_execute_query || !isInternalCacheUpToDate()) {
    return true;
  }
  return do_execute_query;
}

bool SqlQueryRowDataTableJSON::queueQuery(const JSONLocation* json_loc,
                                          const heavyai::InSituFlags insitu_flags) {
  RENDER_LOG_SCOPE();
  bool will_query_run = false;
  if (is_update_pending_) {
    // first run the sql query, then initialize resources
    will_query_run =
        ctx_.getDataTableQueues().addToQueryQueue(shared_from_this(), *json_loc);

    is_update_pending_ = false;
  }
  return will_query_run;
}

void SqlQueryRowDataTableJSON::postRunQuery(bool did_query_execute) {
  RENDER_LOG_SCOPE() << "did_query_execute: " << did_query_execute;
  if (!did_query_execute) {
    // the query/results didn't change, so we can clear the byte offset state
    gpu_resources_->clearOffsetBytesMap();
  }

  auto curr_layout = buffer_layouts_.vbo_layout;

  // now initialize resources
  gpu_resources_->initGpuResourcesFromBuffers(ctx_.getGlobalContext(), curr_layout);

  if (!curr_layout && !gpu_resources_->getGpuDataMap().isEmpty()) {
    curr_layout = getVboQueryDataLayout();
  }

  if (gpu_resources_->updateOffsetBytesMap(curr_layout)) {
    layout_changed_flags_ |= QDTLayoutChangedFlags::kVboOffset;
  }
}

bool SqlQueryRowDataTableJSON::update() {
  RENDER_LOG_SCOPE();
  bool will_query_run = false;
  if (is_update_pending_) {
    if (!isInternalCacheUpToDate()) {
      auto json_obj = ctx_.getJSONObj(json_path_);
      will_query_run = queueQuery(&json_obj, render_query_result_.getInSituFlags());
    }
    is_update_pending_ = false;
  }
  RENDER_LOG() << "will query run: " << (will_query_run ? "true" : "false");
  return will_query_run;
}

}  // namespace QueryRenderer
