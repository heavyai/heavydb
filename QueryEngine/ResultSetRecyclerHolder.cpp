/*
 * Copyright 2022 OmniSci, Inc.
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

#include "ResultSetRecyclerHolder.h"

extern bool g_bigint_count;

std::unique_ptr<ResultSetRecycler> ResultSetRecyclerHolder::query_resultset_cache_ =
    std::make_unique<ResultSetRecycler>();
std::unique_ptr<ChunkMetadataRecycler> ResultSetRecyclerHolder::chunk_metadata_cache_ =
    std::make_unique<ChunkMetadataRecycler>();

bool ResultSetRecyclerHolder::hasCachedQueryResultSet(const size_t key) {
  return query_resultset_cache_->hasItemInCache(key);
}

const ResultSetPtr ResultSetRecyclerHolder::getCachedQueryResultSet(const size_t key) {
  return query_resultset_cache_->getItemFromCache(key,
                                                  CacheItemType::QUERY_RESULTSET,
                                                  DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                                                  std::nullopt);
}

std::optional<std::vector<TargetMetaInfo>> ResultSetRecyclerHolder::getOutputMetaInfo(
    QueryPlanHash key) {
  return query_resultset_cache_->getOutputMetaInfo(key);
}

void ResultSetRecyclerHolder::putQueryResultSetToCache(
    const size_t key,
    const std::unordered_set<size_t>& input_table_keys,
    const ResultSetPtr query_result,
    size_t resultset_size,
    std::vector<std::shared_ptr<Analyzer::Expr>>& target_exprs) {
  ResultSetMetaInfo resultset_meta_info{input_table_keys};
  resultset_meta_info.keepTargetExprs(target_exprs);
  query_resultset_cache_->putItemToCache(key,
                                         query_result,
                                         CacheItemType::QUERY_RESULTSET,
                                         DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                                         resultset_size,
                                         query_result->getExecTime(),
                                         resultset_meta_info);
}

std::optional<ChunkMetadataMap> ResultSetRecyclerHolder::getCachedChunkMetadata(
    const size_t key) {
  return chunk_metadata_cache_->getItemFromCache(
      key, CacheItemType::CHUNK_METADATA, CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER);
}

void ResultSetRecyclerHolder::putChunkMetadataToCache(
    const size_t key,
    const std::unordered_set<size_t>& input_table_keys,
    const ChunkMetadataMap& chunk_metadata_map) {
  if (!chunk_metadata_map.empty()) {
    ChunkMetadataMetaInfo meta_info{input_table_keys};
    chunk_metadata_cache_->putItemToCache(key,
                                          chunk_metadata_map,
                                          CacheItemType::CHUNK_METADATA,
                                          CHUNK_METADATA_CACHE_DEVICE_IDENTIFIER,
                                          0,
                                          0,
                                          meta_info);
  }
}

std::vector<std::shared_ptr<Analyzer::Expr>>& ResultSetRecyclerHolder::getTargetExprs(
    QueryPlanHash key) const {
  return query_resultset_cache_->getTargetExprs(key);
}
