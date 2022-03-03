/*
 * Copyright 2020 OmniSci, Inc.
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

/**
 * @file    AbstractFragmenter.h
 * @author  Todd Mostak <todd@omnisci.com>
 */

#pragma once

#include "Fragmenter/Fragmenter.h"

#include <boost/variant.hpp>
#include <string>
#include <vector>

#include "QueryEngine/TargetMetaInfo.h"
#include "QueryEngine/TargetValue.h"
#include "Shared/sqltypes.h"
#include "StringDictionary/StringDictionaryProxy.h"

// Should the ColumnInfo and FragmentInfo structs be in
// AbstractFragmenter?

class Executor;

namespace Chunk_NS {
class Chunk;
};

namespace Data_Namespace {
class AbstractBuffer;
class AbstractDataMgr;
}  // namespace Data_Namespace

namespace import_export {
class TypedImportBuffer;
}

namespace Catalog_Namespace {
class Catalog;
}
struct TableDescriptor;
struct ColumnDescriptor;

namespace Fragmenter_Namespace {

/**
 *  slim interface to wrap the result set into
 */
class RowDataProvider {
 public:
  virtual size_t const getRowCount() const = 0;
  virtual size_t const getEntryCount() const = 0;
  virtual StringDictionaryProxy* getLiteralDictionary() const = 0;
  virtual std::vector<TargetValue> getEntryAt(const size_t index) const = 0;
  virtual std::vector<TargetValue> getTranslatedEntryAt(const size_t index) const = 0;
};

struct UpdateValuesStats {
  bool has_null{false};
  double max_double{std::numeric_limits<double>::lowest()};
  double min_double{std::numeric_limits<double>::max()};
  int64_t max_int64t{std::numeric_limits<int64_t>::min()};
  int64_t min_int64t{std::numeric_limits<int64_t>::max()};
};

/*
 * @type ChunkUpdateStats
 * @brief struct containing stats from a column chunk update.
 * `new_values_stats` represents aggregate stats for the new
 * values that were put into the chunk. `old_values_stats`
 * represents aggregate stats for chunk values that were
 * replaced.
 */
struct ChunkUpdateStats {
  UpdateValuesStats new_values_stats;
  UpdateValuesStats old_values_stats;
  int64_t updated_rows_count{0};
  int64_t fragment_rows_count{0};
  std::shared_ptr<Chunk_NS::Chunk> chunk;
};

/*
 * @type AbstractFragmenter
 * @brief abstract base class for all table partitioners
 *
 * The virtual methods of this class provide an interface
 * for an interface for getting the id and type of a
 * partitioner, inserting data into a partitioner, and
 * getting the partitions (fragments) managed by a
 * partitioner that must be queried given a predicate
 */

class AbstractFragmenter {
 public:
  virtual ~AbstractFragmenter() {}

  /**
   * @brief Should get the partitions(fragments)
   * where at least one tuple could satisfy the
   * (optional) provided predicate, given any
   * statistics on data distribution the partitioner
   * keeps. May also prune the predicate.
   */
  // virtual void getFragmentsForQuery(QueryInfo &queryInfo, const void *predicate = 0) =
  // 0

  /**
   * @brief returns the number of fragments in a table
   */

  virtual size_t getNumFragments() = 0;

  /**
   * @brief Get all fragments for the current table.
   */
  virtual TableInfo getFragmentsForQuery() = 0;

  /**
   * @brief Given data wrapped in an InsertData struct,
   * inserts it into the correct partitions
   * with locks and checkpoints
   */
  virtual void insertData(InsertData& insert_data_struct) = 0;

  /**
   * @brief Given data wrapped in an InsertData struct,
   * inserts it into the correct partitions
   * No locks and checkpoints taken needs to be managed externally
   */
  virtual void insertDataNoCheckpoint(InsertData& insert_data_struct) = 0;

  /**
   * @brief Will truncate table to less than maxRows by dropping
   * fragments
   */
  virtual void dropFragmentsToSize(const size_t maxRows) = 0;

  /**
   * @brief Update chunk stats
   */
  virtual void updateChunkStats(
      const ColumnDescriptor* cd,
      std::unordered_map</*fragment_id*/ int, ChunkStats>& stats_map,
      std::optional<Data_Namespace::MemoryLevel> memory_level) = 0;

  /**
   * @brief Retrieve the fragment info object for an individual fragment for editing.
   */
  virtual FragmentInfo* getFragmentInfo(const int fragment_id) const = 0;

  /**
   * @brief Gets the id of the partitioner
   */
  virtual int getFragmenterId() = 0;

  /**
   * @brief Gets the string type of the partitioner
   * @todo have a method returning the enum type?
   */
  virtual std::string getFragmenterType() = 0;

  virtual size_t getNumRows() = 0;
  virtual void setNumRows(const size_t numTuples) = 0;

  virtual void dropColumns(const std::vector<int>& columnIds) = 0;

  /**
   * @brief Updates the metadata for a column chunk
   *
   * @param cd - ColumnDescriptor for the column
   * @param fragment_id - Fragment id of the chunk within the column
   * @param metadata -  shared_ptr  of the metadata to update column chunk with
   */
  virtual void updateColumnChunkMetadata(
      const ColumnDescriptor* cd,
      const int fragment_id,
      const std::shared_ptr<ChunkMetadata> metadata) = 0;

  /**
   * Resets the fragmenter's size related metadata using the internal fragment info
   * vector. This is typically done after operations, such as vacuuming, which can
   * change fragment sizes.
   */
  virtual void resetSizesFromFragments() = 0;
};

}  // namespace Fragmenter_Namespace
