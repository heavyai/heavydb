/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Cache/ResultCache.h"

#include "QueryEngine/ResultSet.h"
#include "QueryRenderer/Cache/RowIdHitTestOffsetData.h"
#include "QueryRenderer/Data/EmbeddedDataUtils.h"
#include "QueryRenderer/Utils/TimeUtils.h"

namespace QueryRenderer {

namespace {
inline uint64_t get_result_set_size_bytes(const ResultSet* results) {
  if (!results || results->definitelyHasNoRows()) {
    return 0;
  }
  if (!results->getStorage()) {
    // valid results but with no storage
    // this only happens for in-situ polys
    // must compute estimated size some other way
    // @TODO simon.eves
    // HOW TO ESTIMATE THE SIZE WITH NO ResultRows?
    return 0;
  }
  return results->getQueryMemDesc().getBufferSizeBytes(results->getDeviceType());
}
}  // namespace

QueryResultCacheItem::QueryResultCacheItem(const ResultCacheId in_cache_id,
                                           const std::string& in_sql_str,
                                           RenderQueryOutput in_render_query_output,
                                           const uint64_t in_cache_item_size_bytes)
    : cache_id{in_cache_id}
    , sql_str{in_sql_str}
    , render_query_output{std::move(in_render_query_output)}
    , cache_item_size_bytes{in_cache_item_size_bytes}
    , last_used_time{getCurrentTimeMS()} {
  if (!hasResults()) {
    // Only need to build the rowid offset data in the event of an insitu render when
    // rowid is added as output. hasResuilts() is essentially a check whether the result
    // set is empty or not. If it is empty means it is likely an insitu render.
    // TODO(croot): should we check against the insitu flag in the render_query_output
    // directly?
    row_id_offset_data =
        std::make_unique<RowIdHitTestOffsetData>(sql_str, render_query_output);
  }
}

bool QueryResultCacheItem::hasResults() const {
  auto const* result_set = render_query_output.getResultSetPtr();
  return result_set && !result_set->isExplain();
}

struct QueryResultCache::UpdateCacheGeoReturnType {
  UpdateCacheGeoReturnType(const ResultSet::GeoReturnType geo_return_type)
      : geo_return_type_(geo_return_type) {}
  void operator()(QueryResultCacheItemShPtr& render_cache);

 private:
  ResultSet::GeoReturnType geo_return_type_;
};

QueryResultCache::UpdateCacheResults::UpdateCacheResults(
    RenderQueryOutput render_query_output,
    const uint64_t cache_item_size_bytes,
    bool update_last_used_time)
    : new_last_used_time_{getCurrentTimeMS()}
    , new_render_query_output_{std::move(render_query_output)}
    , new_cache_item_size_bytes_{cache_item_size_bytes}
    , update_last_used_time_{update_last_used_time} {}

void QueryResultCache::UpdateCacheResults::operator()(
    QueryResultCacheItemShPtr& render_cache) {
  if (update_last_used_time_) {
    render_cache->last_used_time = new_last_used_time_;
  }
  render_cache->render_query_output = std::move(new_render_query_output_);
  render_cache->cache_item_size_bytes = new_cache_item_size_bytes_;

  // the new query results require an update of the rowid offsets to ensure pointers are
  // in-sync
  if (!render_cache->hasResults()) {
    // Only need to build the rowid offset data in the event of an insitu render when
    // rowid is added as output. hasResuilts() is essentially a check whether the result
    // set is empty or not. If it is empty means it is likely an insitu render.
    // TODO(croot): should we check against the insitu flag in the render_query_output
    // directly?
    render_cache->row_id_offset_data = std::make_unique<RowIdHitTestOffsetData>(
        render_cache->sql_str, render_cache->render_query_output);
  } else {
    render_cache->row_id_offset_data = nullptr;
  }
}

void QueryResultCache::UpdateCacheGeoReturnType::operator()(
    QueryResultCacheItemShPtr& render_cache) {
  auto* result_set = render_cache->render_query_output.getResultSetPtr();
  if (result_set && result_set->getGeoReturnType() != geo_return_type_) {
    result_set->setGeoReturnType(geo_return_type_);
  }
}

ResultCacheId QueryResultCache::getNextUnusedCacheId() {
  ResultCacheId id{0u};
  if (cache_id_free_list_.size()) {
    id = cache_id_free_list_.back();
    cache_id_free_list_.pop_back();
  } else {
    id = next_cache_id_++;
  }
  return id;
}

void QueryResultCache::purgeUnusedCaches() {
  uint64_t removed_size_bytes = 0;
  auto itr = query_result_map_.begin();
  for (; itr != query_result_map_.end();) {
    if (itr->use_count() == 1) {
      // no one else is referencing the cache, so it can be removed
      VLOG(1) << "QueryResultCache: purging results for query \"" << (*itr)->sql_str
              << "\". Cache id: " << (*itr)->cache_id << ". Freeing up "
              << (*itr)->cache_item_size_bytes << " bytes. Remaining use_count: "
              << (*itr)->render_query_output.getResultSetUseCount();
      removed_size_bytes += (*itr)->cache_item_size_bytes;
      cache_id_free_list_.push_back((*itr)->cache_id);
      itr = query_result_map_.erase(itr);
    } else {
      itr++;
    }
  }

  total_cache_size_bytes_ -= removed_size_bytes;
}

QueryResultCacheItemShPtr QueryResultCache::addQueryResultToCache(
    const std::string& sql_str,
    RenderQueryOutput render_query_output) {
  // TODO(croot): make thread safe?

  purgeUnusedCaches();

  auto result_set_size_bytes =
      get_result_set_size_bytes(render_query_output.getResultSetPtr());
  auto& cache_map_by_sql = query_result_map_.get<SqlStrTag>();
  auto itr = cache_map_by_sql.find(sql_str);
  RUNTIME_EX_ASSERT(itr == cache_map_by_sql.end(),
                    "The sql \"" + sql_str +
                        "\" already exists in the cache. If there are new results, call "
                        "updateQueryResultsInCache()");

  auto cache_id = getNextUnusedCacheId();
  CHECK(query_result_map_.find(cache_id) == query_result_map_.end())
      << " cache_id=" << cache_id;

  auto [item_itr, inserted] =
      query_result_map_.emplace(std::make_shared<QueryResultCacheItem>(
          cache_id, sql_str, std::move(render_query_output), result_set_size_bytes));

  CHECK(inserted);
  total_cache_size_bytes_ += result_set_size_bytes;

  VLOG(1) << "Caching query \"" << sql_str
          << "\" for hit testing with id: " << (*item_itr)->cache_id
          << ". Num bytes in cache: " << result_set_size_bytes
          << ". Total used bytes in cache: " << total_cache_size_bytes_;

  return *item_itr;
}

QueryResultCacheItemShPtr QueryResultCache::updateQueryResultsInCache(
    const std::string& sql_str,
    RenderQueryOutput render_query_output) {
  // TODO(croot): make thread safe?
  purgeUnusedCaches();
  auto& cache_map_by_sql = query_result_map_.get<SqlStrTag>();
  auto itr = cache_map_by_sql.find(sql_str);

  RUNTIME_EX_ASSERT(
      itr != cache_map_by_sql.end(),
      "A render query result cache for sql \"" + sql_str + "\" does not exist.");

  auto result_set_size_bytes =
      get_result_set_size_bytes(render_query_output.getResultSetPtr());
  auto curr_item_size_bytes = (*itr)->cache_item_size_bytes;

  cache_map_by_sql.modify(
      itr, UpdateCacheResults(std::move(render_query_output), result_set_size_bytes));
  if (result_set_size_bytes >= curr_item_size_bytes) {
    total_cache_size_bytes_ += (result_set_size_bytes - curr_item_size_bytes);
  } else {
    total_cache_size_bytes_ -= (curr_item_size_bytes - result_set_size_bytes);
  }
  return *itr;
}

void QueryResultCache::removeQueryResultsFromCache(
    const QueryResultCacheItemShPtr& item) {
  // TODO(croot): make thread safe?
  purgeUnusedCaches();
  auto& cache_map_by_ptr = query_result_map_.get<QueryResultItemPtrTag>();
  auto itr = cache_map_by_ptr.find(item);

  RUNTIME_EX_ASSERT(
      itr != cache_map_by_ptr.end(),
      "A render query result cache for sql \"" + item->sql_str + "\" does not exist.");

  const auto item_size_bytes = (*itr)->cache_item_size_bytes;
  if (itr->use_count() <= 2) {
    VLOG(1) << "QueryResultCache: removing results for query \"" << (*itr)->sql_str
            << "\". Cache id: " << (*itr)->cache_id << ". Freeing up "
            << (*itr)->cache_item_size_bytes << " bytes. Remaining use_count: "
            << (*itr)->render_query_output.getResultSetUseCount();
    cache_id_free_list_.push_back((*itr)->cache_id);
    cache_map_by_ptr.erase(itr);
    total_cache_size_bytes_ -= item_size_bytes;
  } else {
    VLOG(1) << "Did not remove results for query \"" << item->sql_str
            << "\" from hit-test cache as it's being referenced by others. Remaining use "
               "count: "
            << (itr->use_count() - 1) << " results use count: "
            << (*itr)->render_query_output.getResultSetUseCount()
            << ", cache id: " << (*itr)->cache_id << ", cached bytes: " << item_size_bytes
            << ".";
  }
}

void QueryResultCache::clear() {
  query_result_map_.clear();
  cache_id_free_list_.clear();
  next_cache_id_ = kEmptyCacheId + 1;
  total_cache_size_bytes_ = 0;
}

bool QueryResultCache::hasQueryCache(const std::string& sql_str) {
  purgeUnusedCaches();
  auto& cache_map_by_sql = query_result_map_.get<SqlStrTag>();
  return cache_map_by_sql.find(sql_str) != cache_map_by_sql.end();
}

HitTestCacheResults QueryResultCache::getQueryCacheResults(
    const ResultCacheId cache_id,
    const int64_t rowid_to_unpack/*,
    const ResultSet::GeoReturnType geo_return_type*/) const {
  auto itr = query_result_map_.find(cache_id);
  RUNTIME_EX_ASSERT(itr != query_result_map_.end(),
                    "A render query result cache for table id " +
                        std::to_string(cache_id) + " does not exist.");
  query_result_map_.modify(
      itr,
      UpdateCacheGeoReturnType(/*geo_return_type*/ ResultSet::GeoReturnType::WktString));

  const auto& cache_item = *(*itr);
  const auto& rowid_offset_data = cache_item.row_id_offset_data;

  auto const& sql_selected_tables = cache_item.render_query_output.getSqlSelectedTables();

  HitTestTableContainer hittest_table_info;
  if (rowid_offset_data) {
    hittest_table_info =
        rowid_offset_data->unpackRowId(rowid_to_unpack, sql_selected_tables.phys_tables);
  } else {
    if (cache_item.isProjectionQuery()) {
      // If a query is cached as a non-projection query, but it could run in-situ (i.e.
      // forced non-insitu for some reason, usually this means legacy poly rendering using
      // chunkitr interface), we can still utilize multiple rowids for backwards lookups,
      // so make sure to include all the table info for this rowid. NOTE: rowid here is
      // the index into a result set, and that result set could have multiple rowids
      std::transform(sql_selected_tables.phys_tables.begin(),
                     sql_selected_tables.phys_tables.end(),
                     std::back_inserter(hittest_table_info),
                     [rowid_to_unpack](const auto& used_table) {
                       return HitTestTableInfo{used_table,
                                               {{kDefaultIdColumnName, rowid_to_unpack}}};
                     });
    } else {
      hittest_table_info.push_back({*sql_selected_tables.phys_tables.begin(),
                                    {{kDefaultIdColumnName, rowid_to_unpack}}});
    }
  }

  // Add all view tables to the hit-test cache results if any tables the view references
  // currently exist in the hit-test results
  for (auto& view : sql_selected_tables.views) {
    HitTestTableInfo view_info{view, {}};
    auto& hittest_table_by_name = hittest_table_info.get<HitTestTableInfo::NameTag>();
    const auto& view_table_names = view.target_table_names;

    for (const auto& table : view_table_names) {
      auto itr = hittest_table_by_name.find(table);
      if (itr != hittest_table_by_name.end()) {
        auto& hittest_table_info = *itr;  // struct HitTestTableInfo
        CHECK_EQ(hittest_table_info.col_rowid_map.size(), 1u);
        auto rowid_item = hittest_table_info.col_rowid_map.begin();
        const auto& rowid_col_name = rowid_item->first;
        const auto& table_key = hittest_table_info.getTableKeyRef();
        auto catalog =
            Catalog_Namespace::SysCatalog::instance().getCatalog(table_key.db_id);
        CHECK(catalog);
        auto cd = catalog->getMetadataForColumn(table_key.table_id, rowid_col_name);
        CHECK(cd) << table_key << ":" << rowid_col_name;

        // gets the column name from the view if it directly aliases the table/column for
        // a rowid column
        auto column_name =
            view.getAliasedColumn(hittest_table_info.getTableKeyRef(), cd->columnId);
        if (column_name) {
          view_info.col_rowid_map[*column_name] = rowid_item->second;
        }
      }
    }

    if (view_info.col_rowid_map.size()) {
      hittest_table_info.push_back(view_info);
    }
  }

  return {cache_item.sql_str,
          cache_item.render_query_output.getResultSetPtr(),
          cache_item.render_query_output.getOutputTargetEntries(),
          hittest_table_info,
          cache_item.isProjectionQuery()};
}

}  // namespace QueryRenderer
