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
 * @file	InsertOrderFragmenter.h
 * @author	Todd Mostak <todd@mapd.com>
 */

#pragma once

#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/MemoryLevel.h"
#include "FragmentDefaultValues.h"
#include "Fragmenter/AbstractFragmenter.h"
#include "QueryEngine/TargetValue.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/types.h"

class Executor;

namespace Data_Namespace {
class DataMgr;
}

namespace Fragmenter_Namespace {

/**
 * @type InsertOrderFragmenter
 * @brief	The InsertOrderFragmenter is a child class of
 * AbstractFragmenter, and fragments data in insert
 * order. Likely the default fragmenter
 */

class InsertOrderFragmenter : public AbstractFragmenter {
 public:
  InsertOrderFragmenter(
      const std::vector<int> chunkKeyPrefix,
      std::vector<Chunk_NS::Chunk>& chunkVec,
      Data_Namespace::DataMgr* dataMgr,
      Catalog_Namespace::Catalog* catalog,
      const int physicalTableId,
      const size_t maxFragmentRows = DEFAULT_FRAGMENT_ROWS,
      const size_t maxChunkSize = DEFAULT_MAX_CHUNK_SIZE,
      const size_t pageSize = DEFAULT_PAGE_SIZE /*default 1MB*/,
      const size_t maxRows = DEFAULT_MAX_ROWS,
      const Data_Namespace::MemoryLevel defaultInsertLevel = Data_Namespace::DISK_LEVEL,
      const bool uses_foreign_storage = false);

  ~InsertOrderFragmenter() override;

  /**
   * @brief returns the number of fragments in a table
   */

  size_t getNumFragments() override;

  /**
   * @brief returns (inside QueryInfo) object all
   * ids and row sizes of fragments
   *
   */

  // virtual void getFragmentsForQuery(QueryInfo &queryInfo, const void *predicate = 0);
  TableInfo getFragmentsForQuery() override;

  /**
   * @brief appends data onto the most recently occuring
   * fragment, creating a new one if necessary
   *
   * @todo be able to fill up current fragment in
   * multi-row insert before creating new fragment
   */
  void insertData(InsertData& insert_data_struct) override;

  void insertDataNoCheckpoint(InsertData& insert_data_struct) override;

  void dropFragmentsToSize(const size_t maxRows) override;

  void updateColumnChunkMetadata(const ColumnDescriptor* cd,
                                 const int fragment_id,
                                 const std::shared_ptr<ChunkMetadata> metadata) override;

  void updateChunkStats(const ColumnDescriptor* cd,
                        std::unordered_map</*fragment_id*/ int, ChunkStats>& stats_map,
                        std::optional<Data_Namespace::MemoryLevel> memory_level) override;

  FragmentInfo* getFragmentInfo(const int fragment_id) const override;

  /**
   * @brief get fragmenter's id
   */
  inline int getFragmenterId() override { return chunkKeyPrefix_.back(); }
  inline std::vector<int> getChunkKeyPrefix() const { return chunkKeyPrefix_; }
  /**
   * @brief get fragmenter's type (as string
   */
  inline std::string getFragmenterType() override { return fragmenterType_; }
  size_t getNumRows() override { return numTuples_; }
  void setNumRows(const size_t numTuples) override { numTuples_ = numTuples; }

  void dropColumns(const std::vector<int>& columnIds) override;

  void resetSizesFromFragments() override;

 protected:
  std::vector<int> chunkKeyPrefix_;
  std::map<int, Chunk_NS::Chunk>
      columnMap_; /**< stores a map of column id to metadata about that column */
  std::deque<std::unique_ptr<FragmentInfo>>
      fragmentInfoVec_; /**< data about each fragment stored - id and number of rows */
  // int currentInsertBufferFragmentId_;
  Data_Namespace::DataMgr* dataMgr_;
  Catalog_Namespace::Catalog* catalog_;
  const int physicalTableId_;
  size_t maxFragmentRows_;
  size_t pageSize_; /* Page size in bytes of each page making up a given chunk - passed to
                       BufferMgr in createChunk() */
  size_t numTuples_;
  int maxFragmentId_;
  size_t maxChunkSize_;
  size_t maxRows_;
  std::string fragmenterType_;
  mapd_shared_mutex
      fragmentInfoMutex_;  // to prevent read-write conflicts for fragmentInfoVec_
  mapd_shared_mutex
      insertMutex_;  // to prevent race conditions on insert - only one insert statement
                     // should be going to a table at a time
  Data_Namespace::MemoryLevel defaultInsertLevel_;
  const bool uses_foreign_storage_;
  bool hasMaterializedRowId_;
  int rowIdColId_;
  std::unordered_map<int, size_t> varLenColInfo_;
  std::shared_ptr<std::mutex> mutex_access_inmem_states;

  /**
   * @brief creates new fragment, calling createChunk()
   * method of BufferMgr to make a new chunk for each column
   * of the table.
   *
   * Also unpins the chunks of the previous insert buffer
   */

  FragmentInfo* createNewFragment(
      const Data_Namespace::MemoryLevel memory_level = Data_Namespace::DISK_LEVEL);
  void deleteFragments(const std::vector<int>& dropFragIds);

  void conditionallyInstantiateFileMgrWithParams();
  void getChunkMetadata();

  void lockInsertCheckpointData(const InsertData& insertDataStruct);
  void insertDataImpl(InsertData& insert_data);
  void addColumns(const InsertData& insertDataStruct);

  InsertOrderFragmenter(const InsertOrderFragmenter&);
  InsertOrderFragmenter& operator=(const InsertOrderFragmenter&);
  // FIX-ME:  Temporary lock; needs removing.
  mutable std::mutex temp_mutex_;

  FragmentInfo& getFragmentInfoFromId(const int fragment_id);

 private:
  bool isAddingNewColumns(const InsertData& insert_data) const;
  void dropFragmentsToSizeNoInsertLock(const size_t max_rows);
  void setLastFragmentVarLenColumnSizes();
};

}  // namespace Fragmenter_Namespace
