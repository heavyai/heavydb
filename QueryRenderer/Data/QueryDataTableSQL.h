/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>

#include <rapidjson/pointer.h>

#include "Logger/Logger.h"
#include "QueryRenderer/Cache/ResultCache.h"
#include "QueryRenderer/Interface/RenderQueryExecuteData.h"
#include "QueryRenderer/Interface/ResultCacheTypes.h"
#include "QueryRenderer/Interface/SqlSelectedTableInfo.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"
#include "Shared/Rendering/RenderQueryOptions.h"

namespace QueryRenderer {

using SQLSelectedTableContainerShPtr = std::shared_ptr<SQLSelectedTableContainer>;

// QueryDataTableSQL class
//
// Concrete class encapsulating the SQL query string, table info, and query opetions
// for a QueryDataTableSQLJSON class instance
// Also maintains the entry in the QueryResultCache
class QueryDataTableSQL {
 public:
  QueryDataTableSQL() : all_table_info_(std::make_shared<SQLSelectedTableContainer>()) {}

  rapidjson::Pointer updateFromJSONObj(const JSONLocation& json_loc,
                                       bool do_hit_test,
                                       const std::string& data_table_name);
  ResultCacheId getResultCacheId() const;

  std::string getPrimaryTableName() const {
    CHECK(all_table_info_);
    return (all_table_info_->phys_tables.empty()
                ? ""
                : (*all_table_info_).phys_tables[0].table_name.getSqlReference());
  }
  void setAllTableInfo(SQLSelectedTableContainerShPtr all_table_info) {
    all_table_info_ = all_table_info;
  }
  const SQLSelectedTableContainer& getAllTableInfoRef() const {
    CHECK(all_table_info_);
    return *all_table_info_;
  }
  SQLSelectedTableContainerShPtr getAllTableInfoPtr() const { return all_table_info_; }
  void updateAllTableInfo(const SQLSelectedTableContainer& sql_selected_tables);

  bool hasExecutableSql() const;
  const std::string& getSqlQueryStr() const { return sql_query_str_; }

  const RowIdHitTestOffsetData* getRowIdOffsetData() const;
  RenderQueryOptions& getRenderQueryOptions() { return query_opts_; }

  void addQueryToCache(const std::string& sql_query,
                       RenderQueryOutput& render_query_output,
                       QueryResultCache& cache_map);
  void clearTableInfo();
  void clearCache(QueryResultCache& cache_map);

 protected:
  QueryResultCacheItemShPtr query_cache_;
  SQLSelectedTableContainerShPtr all_table_info_;
  std::string sql_query_str_;
  RenderQueryOptions query_opts_;
};

}  // namespace QueryRenderer
