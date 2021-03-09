/*
 * Copyright 2019 OmniSci, Inc.
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

#include "Catalog/Catalog.h"

class Executor;
struct TableUpdateMetadata;

struct DeletedColumnStats {
  size_t total_row_count{0};
  std::unordered_map<int, size_t> visible_row_count_per_fragment;
  std::unordered_map<int, ChunkStats> chunk_stats_per_fragment;
};

/**
 * @brief Driver for running cleanup processes on a table.
 * TableOptimizer provides functions for various cleanup processes that improve
 * performance on a table. Only tables that have been modified using updates or deletes
 * are candidates for cleanup.
 * If the table descriptor corresponds to a sharded table, table optimizer processes each
 * physical shard.
 */
class TableOptimizer {
 public:
  TableOptimizer(const TableDescriptor* td,
                 Executor* executor,
                 const Catalog_Namespace::Catalog& cat);

  /**
   * @brief Recomputes per-chunk metadata for each fragment in the table.
   * Updates and deletes can cause chunk metadata to become wider than the values in the
   * chunk. Recomputing the metadata narrows the range to fit the chunk, as well as
   * setting or unsetting the nulls flag as appropriate.
   */
  void recomputeMetadata() const;

  /**
   * @brief Recomputes column chunk metadata for the given set of fragments.
   * The caller of this method is expected to have already acquired the
   * executor lock.
   */
  void recomputeMetadataUnlocked(const TableUpdateMetadata& table_update_metadata) const;

  /**
   * @brief Compacts fragments to remove deleted rows.
   * When a row is deleted, a boolean deleted system column is set to true. Vacuuming
   * removes all deleted rows from a fragment. Note that vacuuming is a checkpointing
   * operation, so data on disk will increase even though the number of rows for the
   * current epoch has decreased.
   */
  void vacuumDeletedRows() const;

  /**
   * Vacuums fragments with a deleted rows percentage that exceeds the configured minimum
   * vacuum selectivity threshold.
   */
  void vacuumFragmentsAboveMinSelectivity(
      const TableUpdateMetadata& table_update_metadata) const;

 private:
  DeletedColumnStats recomputeDeletedColumnMetadata(
      const TableDescriptor* td,
      const std::set<size_t>& fragment_indexes = {}) const;

  void recomputeColumnMetadata(const TableDescriptor* td,
                               const ColumnDescriptor* cd,
                               const std::unordered_map<int, size_t>& tuple_count_map,
                               std::optional<Data_Namespace::MemoryLevel> memory_level,
                               const std::set<size_t>& fragment_indexes) const;

  std::set<size_t> getFragmentIndexes(const TableDescriptor* td,
                                      const std::set<int>& fragment_ids) const;

  void vacuumFragments(const TableDescriptor* td,
                       const std::set<int>& fragment_ids = {}) const;

  DeletedColumnStats getDeletedColumnStats(
      const TableDescriptor* td,
      const std::set<size_t>& fragment_indexes) const;

  const TableDescriptor* td_;
  Executor* executor_;
  const Catalog_Namespace::Catalog& cat_;

  // We can use a smaller block size here, since we won't be running projection queries
  static constexpr size_t ROW_SET_SIZE{1000000000};
};
