/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/BaseQueryDataTable.h"

#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

BaseQueryDataTableSQLJSON::BaseQueryDataTableSQLJSON(
    QueryRendererContext& ctx,
    const std::string& name,
    const JSONLocation& json_loc,
    const RenderQuerySpecialtyType render_query_type)
    : JSONRefObject(ctx, RefType::kData, name, json_loc.getPathRef())
    , is_update_pending_{true}
    , layout_changed_flags_{QDTLayoutChangedFlags::kNone}
    , render_query_type_{render_query_type} {}

BaseQueryDataTableSQLJSON::~BaseQueryDataTableSQLJSON() {
  clearQueryCache();
}

gfx::BufferLayoutShPtr BaseQueryDataTableSQLJSON::getVboBufferLayout() const {
  auto layout = getVboQueryDataLayout();
  return layout ? layout->getBufferLayout() : nullptr;
}

gfx::BufferLayoutShPtr BaseQueryDataTableSQLJSON::getSsboBufferLayout() const {
  auto layout = getSsboQueryDataLayout();
  return layout ? layout->getBufferLayout() : nullptr;
}

void BaseQueryDataTableSQLJSON::initFromJSONObjAndQueueQuery(
    const JSONLocation& json_loc) {
  RENDER_LOG_SCOPE();
  bool do_execute_query = updateSqlFromJSONObj(json_loc);
  do_execute_query = updateFromJSONObjInternal(json_loc, do_execute_query);
  if (do_execute_query) {
    queueQuery(&json_loc, render_query_result_.getInSituFlags());
  }
}

bool BaseQueryDataTableSQLJSON::updateFromJSONObjAndQueueQuery(
    const JSONLocation& json_loc,
    const bool force_update) {
  RENDER_LOG_SCOPE() << "force_update: " << force_update;
  bool cache_up_to_date = true;
  if (!force_update && ctx_.isJSONCacheUpToDate(json_path_, json_loc) &&
      (cache_up_to_date = isInternalCacheUpToDate())) {
    json_path_ = json_loc.getPathRef();
    return false;
  }
  bool did_sql_change = updateSqlFromJSONObj(json_loc);
  if (did_sql_change) {
    RENDER_LOG() << "** sql changed, will re-run query **";
  }
  bool are_internal_caches_dirty = updateFromJSONObjInternal(
      json_loc, did_sql_change || force_update || !cache_up_to_date);
  if (are_internal_caches_dirty) {
    RENDER_LOG() << "** internal caches dirty, will re-run query **";
  }
  if (did_sql_change || are_internal_caches_dirty) {
    queueQuery(&json_loc, render_query_result_.getInSituFlags());
  }

  json_path_ = json_loc.getPathRef();

  is_update_pending_ = false;
  return true;
}

void BaseQueryDataTableSQLJSON::resetStateFlags() {
  RENDER_LOG_SCOPE();
  is_update_pending_ = true;
  layout_changed_flags_ = QDTLayoutChangedFlags::kNone;
}

void BaseQueryDataTableSQLJSON::clearQueryCache() {
  query_sql_.clearCache(ctx_.getGlobalContext().getRenderQueryCacheMap());
}

void BaseQueryDataTableSQLJSON::clear() {
  query_sql_.getAllTableInfoPtr()->clear();
  clearQueryCache();
  render_query_result_.clear();
}

void BaseQueryDataTableSQLJSON::clearLayouts() {
  buffer_layouts_.reset();
}

void BaseQueryDataTableSQLJSON::addNonInSituResultsToHitTestCache(
    const std::string& sql_query,
    RenderQueryOutput render_query_output) {
  RENDER_LOG_SCOPE();
  query_sql_.addQueryToCache(
      sql_query, render_query_output, ctx_.getGlobalContext().getRenderQueryCacheMap());
}

void BaseQueryDataTableSQLJSON::compareLayoutsAndUpdateChangedFlags(
    QDTLayoutChangedFlags layout_change_type,
    const QueryDataLayout* orig_layout,
    const QueryDataLayout* curr_layout) {
  RENDER_LOG_SCOPE();
  if (orig_layout != curr_layout &&
      (!orig_layout || !curr_layout || *orig_layout != *curr_layout)) {
    layout_changed_flags_ |= layout_change_type;
  } else {
    layout_changed_flags_ &= ~layout_change_type;
  }
}

void BaseQueryDataTableSQLJSON::setQueryResult(const std::string& sql_query,
                                               RenderQueryExecuteData& query_result) {
  RENDER_LOG_SCOPE();
  auto orig_buffer_layouts = buffer_layouts_;
  buffer_layouts_ = query_result.render_buffer_layouts;

  compareLayoutsAndUpdateChangedFlags(QDTLayoutChangedFlags::kVboContents,
                                      orig_buffer_layouts.vbo_layout.get(),
                                      buffer_layouts_.vbo_layout.get());

  compareLayoutsAndUpdateChangedFlags(QDTLayoutChangedFlags::kSsboContents,
                                      orig_buffer_layouts.ssbo_layout.get(),
                                      buffer_layouts_.ssbo_layout.get());

  CHECK(query_result.getResultSetPtr());
  query_sql_.updateAllTableInfo(query_result.getSqlSelectedTables());
  render_query_result_ = query_result.render_query_output.render_query_result;

  if (query_sql_.getRenderQueryOptions().isHitTestingEnabled()) {
    // always adding query execution info to hit-test cache as the
    // output_target_entries may be used when resolving a positive hit-test
    addNonInSituResultsToHitTestCache(sql_query,
                                      std::move(query_result.render_query_output));
  } else {
    clearQueryCache();
  }
}

}  // namespace QueryRenderer
