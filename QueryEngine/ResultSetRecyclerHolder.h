/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "QueryEngine/DataRecycler/ChunkMetadataRecycler.h"
#include "QueryEngine/DataRecycler/ResultSetRecycler.h"

class ResultSetRecyclerHolder {
 public:
  static auto invalidateCache() {
    CHECK(query_resultset_cache_);
    query_resultset_cache_->clearCache();

    CHECK(chunk_metadata_cache_);
    chunk_metadata_cache_->clearCache();
  }

  static auto markCachedItemAsDirty(size_t table_key) {
    CHECK(query_resultset_cache_);
    CHECK(chunk_metadata_cache_);
    auto candidate_table_keys =
        query_resultset_cache_->getMappedQueryPlanDagsWithTableKey(table_key);
    if (candidate_table_keys.has_value()) {
      query_resultset_cache_->markCachedItemAsDirty(
          table_key,
          *candidate_table_keys,
          CacheItemType::QUERY_RESULTSET,
          DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);

      chunk_metadata_cache_->markCachedItemAsDirty(
          table_key,
          *candidate_table_keys,
          CacheItemType::CHUNK_METADATA,
          DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
    }
  }

  static ResultSetRecycler* getResultSetRecycler() {
    return query_resultset_cache_.get();
  }

  static ChunkMetadataRecycler* getChunkMetadataRecycler() {
    return chunk_metadata_cache_.get();
  }

  const ResultSetPtr getCachedQueryResultSet(const size_t key);

  std::optional<std::vector<TargetMetaInfo>> getOutputMetaInfo(QueryPlanHash key);

  bool hasCachedQueryResultSet(const size_t key);

  void putQueryResultSetToCache(
      const size_t key,
      const std::unordered_set<size_t>& input_table_keys,
      const ResultSetPtr query_result,
      size_t resultset_size,
      std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs);

  std::optional<ChunkMetadataMap> getCachedChunkMetadata(const size_t key);

  void putChunkMetadataToCache(const size_t key,
                               const std::unordered_set<size_t>& input_table_keys,
                               const ChunkMetadataMap& chunk_metadata);

  std::vector<std::shared_ptr<Analyzer::Expr>>& getTargetExprs(QueryPlanHash key) const;

 private:
  static std::unique_ptr<ResultSetRecycler> query_resultset_cache_;
  // let's manage chunk_metadata cache with resultset recycler together b/c
  // the only usage currently allowed accessing chunk_metadata_cache is
  // synthetizing chunk metadata for "resultset"
  static std::unique_ptr<ChunkMetadataRecycler> chunk_metadata_cache_;
};
