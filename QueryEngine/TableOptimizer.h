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

#include "Execute.h"

#include <Shared/measure.h>
#include <glog/logging.h>

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
                 const Catalog_Namespace::Catalog& cat)
      : td_(td), executor_(executor), cat_(cat) {
    CHECK(td);
  }

  /**
   * @brief Recomputes per-chunk metadata for each fragment in the table.
   * Updates and deletes can cause chunk metadata to become wider than the values in the
   * chunk. Recomputing the metadata narrows the range to fit the chunk, as well as
   * setting or unsetting the nulls flag as appropriate.
   */
  void recomputeMetadata() const;

  /**
   * @brief Compacts fragments to remove deleted rows.
   * When a row is deleted, a boolean deleted system column is set to true. Vacuuming
   * removes all deleted rows from a fragment. Note that vacuuming is a checkpointing
   * operation, so data on disk will increase even though the number of rows for the
   * current epoch has decreased.
   */
  void vacuumDeletedRows() const;

 private:
  const TableDescriptor* td_;
  Executor* executor_;
  const Catalog_Namespace::Catalog& cat_;
};
