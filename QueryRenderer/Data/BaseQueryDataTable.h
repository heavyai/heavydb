/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "GfxDriver/Resources/Types.h"
#include "QueryRenderer/Data/QueryDataTableSQL.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Interface/RenderQueryExecuteData.h"
#include "QueryRenderer/Interface/RenderQueryRunnerInterface.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

//
// class BaseQueryDataTableSQLJSON
//
// Mix-in base class for DataTables that are driven by Sql
// Also provides the Json interface for VegaParser
//
class BaseQueryDataTableSQLJSON
    : public JSONRefObject,
      public std::enable_shared_from_this<BaseQueryDataTableSQLJSON> {
 public:
  BaseQueryDataTableSQLJSON(QueryRendererContext& ctx,
                            const std::string& name,
                            const JSONLocation& json_loc,
                            const RenderQuerySpecialtyType render_query_type);

  ~BaseQueryDataTableSQLJSON() override;

  QueryDataTableSQL& getQuerySQL() { return query_sql_; }
  RenderQuerySpecialtyType getRenderQuerySpecialtyType() const {
    return render_query_type_;
  }

  // Layouts
  virtual QueryDataLayoutShPtr getVboQueryDataLayout() const {
    return buffer_layouts_.vbo_layout;
  }
  virtual QueryDataLayoutShPtr getSsboQueryDataLayout() const {
    return buffer_layouts_.ssbo_layout;
  }
  gfx::BufferLayoutShPtr getVboBufferLayout() const;
  gfx::BufferLayoutShPtr getSsboBufferLayout() const;

  inline bool hasLayoutChanged(const QDTLayoutChangedFlags layout_change_type) const {
    return (layout_changed_flags_ & layout_change_type) != QDTLayoutChangedFlags::kNone;
  }

  // Results
  void setQueryResult(const std::string& sql_query, RenderQueryExecuteData& query_result);
  const ResultSet* getResultSet() const { return render_query_result_.result_set.get(); }

  inline bool isNonInSitu() const { return render_query_result_.isNonInSitu(); }

  // Json parsing
  virtual bool jsonNeedsUpdating() const { return false; }

  // Post-create init Sql and internals
  void initFromJSONObjAndQueueQuery(const JSONLocation& json_loc);

  // returns pair<should run query, should notify>
  bool updateFromJSONObjAndQueueQuery(const JSONLocation& json_loc,
                                      const bool force_update = false);

 protected:
  void addNonInSituResultsToHitTestCache(const std::string& sql_query,
                                         RenderQueryOutput render_query_output);

  // Json serialization
  void toJSONInternal(rapidjson::Value& obj,
                      rapidjson::Document::AllocatorType& allocator) const final {
    THROW_RUNTIME_EX(
        createJSONRefError("Evaluating data table to JSON is not currently supported."));
  }

  //
  // Scene graph and Json parsing
  //
  virtual void postJSONUpdate() {}
  virtual void resetStateFlags();

  bool is_update_pending_;
  QDTLayoutChangedFlags layout_changed_flags_;

  QueryDataTableSQL query_sql_;
  RenderQuerySpecialtyType render_query_type_;
  RenderQueryBufferLayouts buffer_layouts_;
  RenderQueryResult render_query_result_;

 private:
  // Update Sql string, returning true if query should execute
  virtual bool updateSqlFromJSONObj(const JSONLocation& json_loc) = 0;

  // Update other internals, returning true if query should execute
  virtual bool updateFromJSONObjInternal(const JSONLocation& json_loc,
                                         bool do_execute_query) = 0;
  virtual bool isInternalCacheUpToDate() { return true; }

  virtual bool queueQuery(const JSONLocation* json_loc,
                          const heavyai::InSituFlags insitu_flags) = 0;

  virtual void postRunQuery(bool did_query_execute) = 0;

  void clear();
  void clearQueryCache();
  void clearLayouts();
  void compareLayoutsAndUpdateChangedFlags(const QDTLayoutChangedFlags layout_change_type,
                                           const QueryDataLayout* orig_layout,
                                           const QueryDataLayout* curr_layout);

  friend class QueryRendererContext;
  friend class QueryDataTableQueues;
};

}  // namespace QueryRenderer
