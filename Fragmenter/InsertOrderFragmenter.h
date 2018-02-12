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
 * @file	InsertOrderFragmenter.h
 * @author	Todd Mostak <todd@mapd.com>
 */
#ifndef INSERT_ORDER_FRAGMENTER_H
#define INSERT_ORDER_FRAGMENTER_H

#include "../Shared/mapd_shared_mutex.h"
#include "../Shared/types.h"
#include "AbstractFragmenter.h"
#include "../DataMgr/MemoryLevel.h"
#include "../Chunk/Chunk.h"

#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>

namespace Data_Namespace {
class DataMgr;
}

#define DEFAULT_FRAGMENT_ROWS 32000000     // in tuples
#define DEFAULT_PAGE_SIZE 2097152          // in bytes
#define DEFAULT_MAX_ROWS (1L) << 62        // in rows
#define DEFAULT_MAX_CHUNK_SIZE 1073741824  // in bytes

namespace Fragmenter_Namespace {

/**
 * @type InsertOrderFragmenter
 * @brief	The InsertOrderFragmenter is a child class of
 * AbstractFragmenter, and fragments data in insert
 * order. Likely the default fragmenter
 */

class InsertOrderFragmenter : public AbstractFragmenter {
 public:
  InsertOrderFragmenter(const std::vector<int> chunkKeyPrefix,
                        std::vector<Chunk_NS::Chunk>& chunkVec,
                        Data_Namespace::DataMgr* dataMgr,
                        const int physicalTableId,
                        const int shard,
                        const size_t maxFragmentRows = DEFAULT_FRAGMENT_ROWS,
                        const size_t maxChunkSize = DEFAULT_MAX_CHUNK_SIZE,
                        const size_t pageSize = DEFAULT_PAGE_SIZE /*default 1MB*/,
                        const size_t maxRows = DEFAULT_MAX_ROWS,

                        const Data_Namespace::MemoryLevel defaultInsertLevel = Data_Namespace::DISK_LEVEL);

  virtual ~InsertOrderFragmenter();
  /**
   * @brief returns (inside QueryInfo) object all
   * ids and row sizes of fragments
   *
   */

  // virtual void getFragmentsForQuery(QueryInfo &queryInfo, const void *predicate = 0);
  virtual TableInfo getFragmentsForQuery();

  /**
   * @brief appends data onto the most recently occuring
   * fragment, creating a new one if necessary
   *
   * @todo be able to fill up current fragment in
   * multi-row insert before creating new fragment
   */
  virtual void insertData(const InsertData& insertDataStruct);

  virtual void insertDataNoCheckpoint(const InsertData& insertDataStruct);

  virtual void dropFragmentsToSize(const size_t maxRows);
  /**
   * @brief get fragmenter's id
   */

  inline int getFragmenterId() { return chunkKeyPrefix_.back(); }
  inline std::vector<int> getChunkKeyPrefix() const { return chunkKeyPrefix_; }
  /**
   * @brief get fragmenter's type (as string
   */
  inline std::string getFragmenterType() { return fragmenterType_; }

 private:
  std::vector<int> chunkKeyPrefix_;
  std::map<int, Chunk_NS::Chunk> columnMap_; /**< stores a map of column id to metadata about that column */
  std::deque<FragmentInfo> fragmentInfoVec_; /**< data about each fragment stored - id and number of rows */
  // int currentInsertBufferFragmentId_;
  Data_Namespace::DataMgr* dataMgr_;
  const int physicalTableId_;
  const int shard_;
  size_t maxFragmentRows_;
  size_t pageSize_; /* Page size in bytes of each page making up a given chunk - passed to BufferMgr in createChunk() */
  size_t numTuples_;
  int maxFragmentId_;
  size_t maxChunkSize_;
  size_t maxRows_;
  std::string fragmenterType_;
  mapd_shared_mutex fragmentInfoMutex_;  // to prevent read-write conflicts for fragmentInfoVec_
  mapd_shared_mutex insertMutex_;  // to prevent race conditions on insert - only one insert statement should be going
                                   // to a table at a time
  Data_Namespace::MemoryLevel defaultInsertLevel_;
  bool hasMaterializedRowId_;
  int rowIdColId_;
  std::unordered_map<int, size_t> varLenColInfo_;

  /**
   * @brief creates new fragment, calling createChunk()
   * method of BufferMgr to make a new chunk for each column
   * of the table.
   *
   * Also unpins the chunks of the previous insert buffer
   */

  FragmentInfo* createNewFragment(const Data_Namespace::MemoryLevel memoryLevel = Data_Namespace::DISK_LEVEL);
  void deleteFragments(const std::vector<int>& dropFragIds);

  void getChunkMetadata();

  void lockInsertCheckpointData(const InsertData& insertDataStruct);
  void insertDataImpl(const InsertData& insertDataStruct);

  InsertOrderFragmenter(const InsertOrderFragmenter&);
  InsertOrderFragmenter& operator=(const InsertOrderFragmenter&);
};

}  // Fragmenter_Namespace

#endif  // INSERT_ORDER_FRAGMENTER_H
