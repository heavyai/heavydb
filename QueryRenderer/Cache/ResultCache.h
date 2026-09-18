/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>

#include "QueryRenderer/Cache/HitTestCacheResults.h"
#include "QueryRenderer/Interface/RenderQueryExecuteData.h"
#include "QueryRenderer/Interface/ResultCacheTypes.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {

struct RowIdHitTestOffsetData;

struct QueryResultCacheItem {
  ResultCacheId cache_id;
  std::string sql_str;
  RenderQueryOutput render_query_output;
  uint64_t cache_item_size_bytes;
  std::unique_ptr<RowIdHitTestOffsetData> row_id_offset_data;
  std::chrono::milliseconds last_used_time;

  explicit QueryResultCacheItem(const ResultCacheId in_cache_id,
                                const std::string& in_sql_str,
                                RenderQueryOutput in_render_query_output,
                                const uint64_t in_cache_size_bytes);

  bool hasResults() const;
  bool isProjectionQuery() const {
    return !hasResults() || render_query_output.isInSitu() ||
           render_query_output.couldRunInSitu();
  }
};

class QueryResultCache {
 public:
  static constexpr ResultCacheId kEmptyCacheId = 0u;

  QueryResultCache() : total_cache_size_bytes_{0u}, next_cache_id_{kEmptyCacheId + 1} {}

  QueryResultCacheItemShPtr addQueryResultToCache(const std::string& sql_str,
                                                  RenderQueryOutput render_query_output);

  QueryResultCacheItemShPtr updateQueryResultsInCache(
      const std::string& sql_str,
      RenderQueryOutput render_query_output);

  void removeQueryResultsFromCache(const QueryResultCacheItemShPtr& item);
  void clear();

  bool hasQueryCache(const std::string& sql_str);

  HitTestCacheResults getQueryCacheResults(
      const ResultCacheId cache_id,
      const int64_t rowid_to_unpack/*,
      const ResultSet::GeoReturnType geoReturnType*/) const;

 private:
  static const std::chrono::milliseconds max_cache_idle_time_;

  uint64_t total_cache_size_bytes_;

  struct ResultCacheIdTag {};
  struct SqlStrTag {};
  struct QueryResultItemPtrTag {};

  struct UpdateCacheResults {
    UpdateCacheResults(RenderQueryOutput render_query_output,
                       const uint64_t cache_item_size_bytes,
                       bool update_last_used_time = true);

    void operator()(QueryResultCacheItemShPtr& render_cache);

   private:
    std::chrono::milliseconds new_last_used_time_;
    RenderQueryOutput new_render_query_output_;
    uint64_t new_cache_item_size_bytes_;
    bool update_last_used_time_;
  };

  struct UpdateCacheGeoReturnType;

  using QueryResultMap = ::boost::multi_index_container<
      QueryResultCacheItemShPtr,
      ::boost::multi_index::indexed_by<
          ::boost::multi_index::ordered_unique<
              ::boost::multi_index::tag<ResultCacheIdTag>,
              ::boost::multi_index::member<QueryResultCacheItem,
                                           ResultCacheId,
                                           &QueryResultCacheItem::cache_id>>,
          ::boost::multi_index::hashed_unique<
              ::boost::multi_index::tag<SqlStrTag>,
              ::boost::multi_index::member<QueryResultCacheItem,
                                           decltype(QueryResultCacheItem::sql_str),
                                           &QueryResultCacheItem::sql_str>>,
          ::boost::multi_index::hashed_unique<
              ::boost::multi_index::tag<QueryResultItemPtrTag>,
              ::boost::multi_index::identity<QueryResultCacheItemShPtr>>>>;

  using QueryResultMap_by_SqlStr = QueryResultMap::index<SqlStrTag>::type;
  using QueryResultMap_by_Pointer = QueryResultMap::index<QueryResultItemPtrTag>::type;

  mutable QueryResultMap query_result_map_;
  std::vector<ResultCacheId> cache_id_free_list_;
  ResultCacheId next_cache_id_;

  ResultCacheId getNextUnusedCacheId();
  void purgeUnusedCaches();
};

}  // namespace QueryRenderer
