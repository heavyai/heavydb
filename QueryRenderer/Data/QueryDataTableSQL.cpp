/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/QueryDataTableSQL.h"

#include "GfxDriver/RenderError.h"
#include "QueryRenderer/Cache/RowIdHitTestOffsetData.h"
#include "QueryRenderer/Data/Utils.h"
#include "Shared/StringTransform.h"

namespace QueryRenderer {

rapidjson::Pointer QueryDataTableSQL::updateFromJSONObj(
    const JSONLocation& json_loc,
    bool do_hit_test,
    const std::string& data_table_name) {
  const auto sql_loc = json_loc.getMember(JSONSchema_v1::Data::kSqlProp);
  RUNTIME_EX_ASSERT(sql_loc.isValid() && sql_loc.isString(),
                    RapidJSONUtils::createJsonParseError(
                        (sql_loc.isValid() ? sql_loc : json_loc),
                        "SQL data object \"" + data_table_name + "\" must contain an \"" +
                            std::string(JSONSchema_v1::Data::kSqlProp) +
                            "\" property and it must be a string"));

  sql_query_str_ = sql_loc.getString();

  // TODO(croot) -- should we validate the sql?

  // remove newlines, linefeeds, and tabs not inside quotes (BE-3857)
  bool sql_string_has_consistent_quoting =
      remove_unquoted_newlines_linefeeds_and_tabs_from_sql_string(sql_query_str_);
  if (!sql_string_has_consistent_quoting) {
    LOG(WARNING) << "Render Query SQL string has inconsistent quoting!";
  }

  bool is_hit_testing_enabled = false;
  query_opts_.clearAllFlags();
  if (do_hit_test) {  // enabled in RenderSession
    const auto hittest_loc = json_loc.getMember(
        JSONSchema_v1::Data::kEnableHitTestingProp, JSONValueType::kBool, false);
    // Check if "enableHitTesting" field is in the vega data block
    if (hittest_loc.isValid()) {
      is_hit_testing_enabled = hittest_loc.getBool();
      if (is_hit_testing_enabled) {
        // Inject rowid into RA if necessary
        query_opts_.setFlags(RenderQueryOptions::FlagBits::kInjectRowIdForHitTesting);
      }
    } else {
      // No "enableHitTesting" set in data block. Use legacy hit test logic
      // This will automatically add rowid to non-insitu queries
      query_opts_.setFlags(RenderQueryOptions::FlagBits::kLegacyHitTestLogic);
      is_hit_testing_enabled = true;  // attempt it
    }
  }
  if (is_hit_testing_enabled) {
    // All hit-testing requires physical tables to be present
    query_opts_.setFlags(RenderQueryOptions::FlagBits::kRequiresPhysicalTables);
  }

  // TODO(croot) - for backwards compatibility, the dbTableName doesn't have to be present
  // but should it be required? Or can we somehow extract it from the sql?
  all_table_info_ = std::make_shared<SQLSelectedTableContainer>();
  const auto db_table_loc = json_loc.getMember(JSONSchema_v1::Data::kDbNameProp);
  if (db_table_loc.isValid()) {
    RUNTIME_EX_ASSERT(db_table_loc.isString(),
                      RapidJSONUtils::createJsonParseError(
                          db_table_loc,
                          "\"" + std::string(JSONSchema_v1::Data::kDbNameProp) +
                              "\" property must be a string"));

    // TODO(croot): should we throw a deprecation warning? Also, remove support for this
    // very old poly structure
    all_table_info_->phys_tables.emplace_back(
        shared::TableKey{},
        shared::FullyQualifiedTableName{"", db_table_loc.getString()},
        0);
  }
  // TODO(croot): call the query parse cb to get at all the tables used in the query here?

  return json_loc.getPathRef();
}

bool QueryDataTableSQL::hasExecutableSql() const {
  // NOTE: "select x, y from tweets;" was a placeholder sql in the vega
  // Need to check for that for backwards compatibility.
  return (sql_query_str_.length() && sql_query_str_ != "select x, y from tweets;");
}

ResultCacheId QueryDataTableSQL::getResultCacheId() const {
  CHECK(all_table_info_);
  return query_cache_ ? query_cache_->cache_id : QueryResultCache::kEmptyCacheId;
}

const RowIdHitTestOffsetData* QueryDataTableSQL::getRowIdOffsetData() const {
  return query_cache_ ? query_cache_->row_id_offset_data.get() : nullptr;
}

void QueryDataTableSQL::addQueryToCache(const std::string& sql_query,
                                        RenderQueryOutput& render_query_output,
                                        QueryResultCache& query_cache_map) {
  if (!query_cache_map.hasQueryCache(sql_query)) {
    if (query_cache_) {
      query_cache_map.removeQueryResultsFromCache(query_cache_);
    }
    query_cache_ =
        query_cache_map.addQueryResultToCache(sql_query, std::move(render_query_output));
  } else {
    query_cache_ = query_cache_map.updateQueryResultsInCache(
        sql_query, std::move(render_query_output));
  }
}

void QueryDataTableSQL::clearCache(QueryResultCache& cache_map) {
  if (query_cache_) {
    cache_map.removeQueryResultsFromCache(query_cache_);
  }
  query_cache_ = nullptr;
}

void QueryDataTableSQL::clearTableInfo() {
  CHECK(all_table_info_);
  all_table_info_->clear();
}

void QueryDataTableSQL::updateAllTableInfo(
    const SQLSelectedTableContainer& sql_selected_tables) {
  if (query_opts_.requiresPhysicalTables()) {
    RUNTIME_EX_ASSERT(!sql_selected_tables.phys_tables.empty(),
                      "Physical table references are required by the renderer. "
                      "Hit-testing must be disabled.");
  }
  CHECK(all_table_info_);
  LOG_IF(WARNING,
         !all_table_info_->phys_tables.empty() &&
             all_table_info_->phys_tables[0].table_name !=
                 sql_selected_tables.phys_tables[0].table_name)
      << "The primary table to render \"" << sql_selected_tables.phys_tables[0].table_name
      << "\" does not match the dbTableName json attribute: "
      << all_table_info_->phys_tables[0].table_name
      << ". Using the table referenced in the query.";
  all_table_info_ = std::make_shared<SQLSelectedTableContainer>(sql_selected_tables);
}

}  // namespace QueryRenderer
