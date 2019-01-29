/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * @author  Todd Mostak <todd@map-d.com
 */

#ifndef _ABSTRACT_FRAGMENTER_H
#define _ABSTRACT_FRAGMENTER_H

#include <boost/variant.hpp>
#include <string>
#include <vector>
#include "../QueryEngine/TargetValue.h"
#include "../Shared/UpdelRoll.h"
#include "../Shared/sqltypes.h"
#include "../StringDictionary/StringDictionary.h"
#include "Fragmenter.h"

// Should the ColumnInfo and FragmentInfo structs be in
// AbstractFragmenter?

namespace Chunk_NS {
class Chunk;
};

namespace Data_Namespace {
class AbstractBuffer;
class AbstractDataMgr;
};  // namespace Data_Namespace

namespace Importer_NS {
class TypedImportBuffer;
};

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
  virtual size_t count() const = 0;
  virtual std::vector<TargetValue> getEntryAt(const size_t index) const = 0;
  virtual std::vector<TargetValue> getTranslatedEntryAt(const size_t index) const = 0;
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
   * @brief Get all fragments for the current table.
   */
  virtual TableInfo getFragmentsForQuery() = 0;

  /**
   * @brief Given data wrapped in an InsertData struct,
   * inserts it into the correct partitions
   * with locks and checkpoints
   */
  virtual void insertData(InsertData& insertDataStruct) = 0;

  /**
   * @brief Given data wrapped in an InsertData struct,
   * inserts it into the correct partitions
   * No locks and checkpoints taken needs to be managed externally
   */
  virtual void insertDataNoCheckpoint(InsertData& insertDataStruct) = 0;

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
      std::unordered_map</*fragment_id*/ int, ChunkStats>& stats_map) = 0;

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

  virtual void updateColumn(const Catalog_Namespace::Catalog* catalog,
                            const TableDescriptor* td,
                            const ColumnDescriptor* cd,
                            const int fragment_id,
                            const std::vector<uint64_t>& frag_offsets,
                            const std::vector<ScalarTargetValue>& rhs_values,
                            const SQLTypeInfo& rhs_type,
                            const Data_Namespace::MemoryLevel memory_level,
                            UpdelRoll& updel_roll) = 0;

  virtual void updateColumns(const Catalog_Namespace::Catalog* catalog,
                             const TableDescriptor* td,
                             const int fragmentId,
                             const std::vector<const ColumnDescriptor*> columnDescriptors,
                             const RowDataProvider& sourceDataProvider,
                             const size_t indexOffFragmentOffsetColumn,
                             const Data_Namespace::MemoryLevel memoryLevel,
                             UpdelRoll& updelRoll) = 0;

  virtual void updateColumn(const Catalog_Namespace::Catalog* catalog,
                            const TableDescriptor* td,
                            const ColumnDescriptor* cd,
                            const int fragment_id,
                            const std::vector<uint64_t>& frag_offsets,
                            const ScalarTargetValue& rhs_value,
                            const SQLTypeInfo& rhs_type,
                            const Data_Namespace::MemoryLevel memory_level,
                            UpdelRoll& updel_roll) = 0;

  virtual void updateColumnMetadata(const ColumnDescriptor* cd,
                                    FragmentInfo& fragment,
                                    std::shared_ptr<Chunk_NS::Chunk> chunk,
                                    const bool null,
                                    const double dmax,
                                    const double dmin,
                                    const int64_t lmax,
                                    const int64_t lmin,
                                    const SQLTypeInfo& rhs_type,
                                    UpdelRoll& updel_roll) = 0;

  virtual void updateMetadata(const Catalog_Namespace::Catalog* catalog,
                              const MetaDataKey& key,
                              UpdelRoll& updel_roll) = 0;

  virtual void compactRows(const Catalog_Namespace::Catalog* catalog,
                           const TableDescriptor* td,
                           const int fragmentId,
                           const std::vector<uint64_t>& fragOffsets,
                           const Data_Namespace::MemoryLevel memoryLevel,
                           UpdelRoll& updelRoll) = 0;

  virtual const std::vector<uint64_t> getVacuumOffsets(
      const std::shared_ptr<Chunk_NS::Chunk>& chunk) = 0;
};

}  // namespace Fragmenter_Namespace

#endif  // _ABSTRACT_FRAGMENTER_H
