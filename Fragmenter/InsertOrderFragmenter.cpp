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

#include "InsertOrderFragmenter.h"
#include "../DataMgr/LockMgr.h"
#include "../DataMgr/DataMgr.h"
#include "../DataMgr/AbstractBuffer.h"
#include <glog/logging.h>
#include <math.h>
#include <iostream>
#include <thread>

#include <assert.h>
#include <boost/lexical_cast.hpp>

#define DROP_FRAGMENT_FACTOR 0.97  // drop to 97% of max so we don't keep adding and dropping fragments

using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;
using Chunk_NS::Chunk;

using namespace std;

namespace Fragmenter_Namespace {

InsertOrderFragmenter::InsertOrderFragmenter(const vector<int> chunkKeyPrefix,
                                             vector<Chunk>& chunkVec,
                                             Data_Namespace::DataMgr* dataMgr,
                                             const int physicalTableId,
                                             const int shard,
                                             const size_t maxFragmentRows,
                                             const size_t maxChunkSize,
                                             const size_t pageSize,
                                             const size_t maxRows,
                                             const Data_Namespace::MemoryLevel defaultInsertLevel)
    : chunkKeyPrefix_(chunkKeyPrefix),
      dataMgr_(dataMgr),
      physicalTableId_(physicalTableId),
      shard_(shard),
      maxFragmentRows_(maxFragmentRows),
      pageSize_(pageSize),
      numTuples_(0),
      maxFragmentId_(-1),
      maxChunkSize_(maxChunkSize),
      maxRows_(maxRows),
      fragmenterType_("insert_order"),
      defaultInsertLevel_(defaultInsertLevel),
      hasMaterializedRowId_(false) {
  // Note that Fragmenter is not passed virtual columns and so should only
  // find row id column if it is non virtual

  for (auto colIt = chunkVec.begin(); colIt != chunkVec.end(); ++colIt) {
    int columnId = colIt->get_column_desc()->columnId;
    columnMap_[columnId] = *colIt;
    if (colIt->get_column_desc()->columnName == "rowid") {
      hasMaterializedRowId_ = true;
      rowIdColId_ = columnId;
    }
  }
  getChunkMetadata();
}

InsertOrderFragmenter::~InsertOrderFragmenter() {}

void InsertOrderFragmenter::getChunkMetadata() {
  if (defaultInsertLevel_ ==
      Data_Namespace::MemoryLevel::DISK_LEVEL) {  // memory-resident tables won't have anything on disk
    std::vector<std::pair<ChunkKey, ChunkMetadata>> chunkMetadataVec;
    dataMgr_->getChunkMetadataVecForKeyPrefix(chunkMetadataVec, chunkKeyPrefix_);

    // data comes like this - database_id, table_id, column_id, fragment_id
    // but lets sort by database_id, table_id, fragment_id, column_id

    int fragmentSubKey = 3;
    std::sort(chunkMetadataVec.begin(),
              chunkMetadataVec.end(),
              [&](const std::pair<ChunkKey, ChunkMetadata>& pair1, const std::pair<ChunkKey, ChunkMetadata>& pair2) {
                return pair1.first[3] < pair2.first[3];
              });

    for (auto chunkIt = chunkMetadataVec.begin(); chunkIt != chunkMetadataVec.end(); ++chunkIt) {
      int curFragmentId = chunkIt->first[fragmentSubKey];

      if (fragmentInfoVec_.empty() || curFragmentId != fragmentInfoVec_.back().fragmentId) {
        maxFragmentId_ = curFragmentId;
        fragmentInfoVec_.push_back(FragmentInfo());
        fragmentInfoVec_.back().fragmentId = curFragmentId;
        fragmentInfoVec_.back().setPhysicalNumTuples(chunkIt->second.numElements);
        numTuples_ += fragmentInfoVec_.back().getPhysicalNumTuples();
        for (const auto levelSize : dataMgr_->levelSizes_) {
          fragmentInfoVec_.back().deviceIds.push_back(curFragmentId % levelSize);
        }
        fragmentInfoVec_.back().shadowNumTuples = fragmentInfoVec_.back().getPhysicalNumTuples();
        fragmentInfoVec_.back().physicalTableId = physicalTableId_;
        fragmentInfoVec_.back().shard = shard_;
      } else {
        if (chunkIt->second.numElements != fragmentInfoVec_.back().getPhysicalNumTuples()) {
          throw std::runtime_error("Inconsistency in num tuples within fragment");
        }
      }
      int columnId = chunkIt->first[2];
      fragmentInfoVec_.back().setChunkMetadata(columnId, chunkIt->second);
    }
  }

  ssize_t maxFixedColSize = 0;

  for (auto colIt = columnMap_.begin(); colIt != columnMap_.end(); ++colIt) {
    ssize_t size = colIt->second.get_column_desc()->columnType.get_size();
    if (size == -1) {  // variable length
      varLenColInfo_.insert(std::make_pair(colIt->first, 0));
      size = 8;  // b/c we use this for string and array indices - gross to have magic number here
    }
    maxFixedColSize = std::max(maxFixedColSize, size);
  }

  maxFragmentRows_ =
      std::min(maxFragmentRows_,
               maxChunkSize_ / maxFixedColSize);  // this is maximum number of rows assuming everything is fixed length

  if (fragmentInfoVec_.size() > 0) {
    // Now need to get the insert buffers for each column - should be last
    // fragment
    int lastFragmentId = fragmentInfoVec_.back().fragmentId;
    int deviceId = fragmentInfoVec_.back().deviceIds[static_cast<int>(defaultInsertLevel_)];
    for (auto colIt = columnMap_.begin(); colIt != columnMap_.end(); ++colIt) {
      ChunkKey insertKey = chunkKeyPrefix_;  // database_id and table_id
      insertKey.push_back(colIt->first);     // column id
      insertKey.push_back(lastFragmentId);   // fragment id
      colIt->second.getChunkBuffer(dataMgr_, insertKey, defaultInsertLevel_, deviceId);
      auto varLenColInfoIt = varLenColInfo_.find(colIt->first);
      if (varLenColInfoIt != varLenColInfo_.end()) {
        varLenColInfoIt->second = colIt->second.get_buffer()->size();
      }
    }
  }
}

void InsertOrderFragmenter::dropFragmentsToSize(const size_t maxRows) {
  // not safe to call from outside insertData
  // b/c depends on insertLock around numTuples_

  if (numTuples_ > maxRows) {
    size_t preNumTuples = numTuples_;
    vector<int> dropFragIds;
    size_t targetRows = maxRows * DROP_FRAGMENT_FACTOR;
    while (numTuples_ > targetRows) {
      assert(fragmentInfoVec_.size() > 0);
      size_t numFragTuples = fragmentInfoVec_[0].getPhysicalNumTuples();
      dropFragIds.push_back(fragmentInfoVec_[0].fragmentId);
      fragmentInfoVec_.pop_front();
      assert(numTuples_ >= numFragTuples);
      numTuples_ -= numFragTuples;
    }
    deleteFragments(dropFragIds);
    LOG(INFO) << "dropFragmentsToSize, numTuples pre: " << preNumTuples << " post: " << numTuples_
              << " maxRows: " << maxRows;
  }
}

void InsertOrderFragmenter::deleteFragments(const vector<int>& dropFragIds) {
  // need to keep lock seq as UpdateDeleteLock >> fragmentInfoMutex_ or
  // SELECT and COPY may enter a deadlock
  using namespace Lock_Namespace;
  mapd_unique_lock<mapd_shared_mutex> deleteLock(
      *LockMgr<mapd_shared_mutex, ChunkKey>::getMutex(LockType::UpdateDeleteLock, chunkKeyPrefix_));
  mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);

  for (const auto fragId : dropFragIds) {
    for (const auto& col : columnMap_) {
      int colId = col.first;
      vector<int> fragPrefix = chunkKeyPrefix_;
      fragPrefix.push_back(colId);
      fragPrefix.push_back(fragId);
      dataMgr_->deleteChunksWithPrefix(fragPrefix);
    }
  }
}

void InsertOrderFragmenter::insertData(const InsertData& insertDataStruct) {
  insertDataImpl(insertDataStruct);
  if (defaultInsertLevel_ == Data_Namespace::DISK_LEVEL) {  // only checkpoint if data is resident on disk
    dataMgr_->checkpoint(chunkKeyPrefix_[0],
                         chunkKeyPrefix_[1]);  // need to checkpoint here to remove window for corruption
  }
}

void InsertOrderFragmenter::insertDataNoCheckpoint(const InsertData& insertDataStruct) {
  insertDataImpl(insertDataStruct);
}

void InsertOrderFragmenter::insertDataImpl(const InsertData& insertDataStruct) {
  // mutex comes from datamgr so that lock can span more than a single component
  // TODO: this local lock will need to be centralized when ALTER COLUMN is added, bc
  // ALTER modifies or add chunks via the same fragmenter...
  mapd_unique_lock<mapd_shared_mutex> insertLock(
      insertMutex_);  // prevent two threads from trying to insert into the same table simultaneously
  std::unordered_map<int, int> inverseInsertDataColIdMap;

  for (size_t insertId = 0; insertId < insertDataStruct.columnIds.size(); ++insertId) {
    inverseInsertDataColIdMap.insert(std::make_pair(insertDataStruct.columnIds[insertId], insertId));
  }

  size_t numRowsLeft = insertDataStruct.numRows;
  size_t numRowsInserted = 0;
  vector<DataBlockPtr> dataCopy =
      insertDataStruct.data;  // bc append data will move ptr forward and this violates constness of InsertData
  if (numRowsLeft <= 0) {
    return;
  }

  FragmentInfo* currentFragment = 0;

  if (fragmentInfoVec_.empty()) {  // if no fragments exist for table
    currentFragment = createNewFragment(defaultInsertLevel_);
  } else {
    currentFragment = &(fragmentInfoVec_.back());
  }
  size_t startFragment = fragmentInfoVec_.size() - 1;

  while (numRowsLeft > 0) {  // may have to create multiple fragments for bulk insert
    // loop until done inserting all rows
    CHECK_LE(currentFragment->shadowNumTuples, maxFragmentRows_);
    size_t rowsLeftInCurrentFragment = maxFragmentRows_ - currentFragment->shadowNumTuples;
    size_t numRowsToInsert = min(rowsLeftInCurrentFragment, numRowsLeft);
    if (rowsLeftInCurrentFragment != 0) {
      for (auto& varLenColInfoIt : varLenColInfo_) {
        CHECK_LE(varLenColInfoIt.second, maxChunkSize_);
        size_t bytesLeft = maxChunkSize_ - varLenColInfoIt.second;
        auto insertIdIt = inverseInsertDataColIdMap.find(varLenColInfoIt.first);
        if (insertIdIt != inverseInsertDataColIdMap.end()) {
          auto colMapIt = columnMap_.find(varLenColInfoIt.first);
          numRowsToInsert = std::min(numRowsToInsert,
                                     colMapIt->second.getNumElemsForBytesInsertData(
                                         dataCopy[insertIdIt->second], numRowsToInsert, numRowsInserted, bytesLeft));
        }
      }
    }

    if (rowsLeftInCurrentFragment == 0 || numRowsToInsert == 0) {
      currentFragment = createNewFragment(defaultInsertLevel_);
      if (numRowsInserted == 0) {
        startFragment++;
      }
      rowsLeftInCurrentFragment = maxFragmentRows_;
      for (auto& varLenColInfoIt : varLenColInfo_) {
        varLenColInfoIt.second = 0;  // reset byte counter
      }
      numRowsToInsert = min(rowsLeftInCurrentFragment, numRowsLeft);
      for (auto& varLenColInfoIt : varLenColInfo_) {
        CHECK_LE(varLenColInfoIt.second, maxChunkSize_);
        size_t bytesLeft = maxChunkSize_ - varLenColInfoIt.second;
        auto insertIdIt = inverseInsertDataColIdMap.find(varLenColInfoIt.first);
        if (insertIdIt != inverseInsertDataColIdMap.end()) {
          auto colMapIt = columnMap_.find(varLenColInfoIt.first);
          numRowsToInsert = std::min(numRowsToInsert,
                                     colMapIt->second.getNumElemsForBytesInsertData(
                                         dataCopy[insertIdIt->second], numRowsToInsert, numRowsInserted, bytesLeft));
        }
      }
    }

    CHECK_GT(numRowsToInsert, size_t(0));  // would put us into an endless loop as we'd never be able to insert anything

    // for each column, append the data in the appropriate insert buffer
    for (size_t i = 0; i < insertDataStruct.columnIds.size(); ++i) {
      int columnId = insertDataStruct.columnIds[i];
      auto colMapIt = columnMap_.find(columnId);
      assert(colMapIt != columnMap_.end());
      currentFragment->shadowChunkMetadataMap[columnId] =
          colMapIt->second.appendData(dataCopy[i], numRowsToInsert, numRowsInserted);
      auto varLenColInfoIt = varLenColInfo_.find(columnId);
      if (varLenColInfoIt != varLenColInfo_.end()) {
        varLenColInfoIt->second = colMapIt->second.get_buffer()->size();
      }
    }
    if (hasMaterializedRowId_) {
      size_t startId = maxFragmentRows_ * currentFragment->fragmentId + currentFragment->shadowNumTuples;
      int64_t* rowIdData = new int64_t[numRowsToInsert];
      for (size_t i = 0; i < numRowsToInsert; ++i) {
        rowIdData[i] = i + startId;
      }
      DataBlockPtr rowIdBlock;
      rowIdBlock.numbersPtr = reinterpret_cast<int8_t*>(rowIdData);
      auto colMapIt = columnMap_.find(rowIdColId_);
      currentFragment->shadowChunkMetadataMap[rowIdColId_] =
          colMapIt->second.appendData(rowIdBlock, numRowsToInsert, numRowsInserted);
      delete[] rowIdData;
    }

    currentFragment->shadowNumTuples = fragmentInfoVec_.back().getPhysicalNumTuples() + numRowsToInsert;
    numRowsLeft -= numRowsToInsert;
    numRowsInserted += numRowsToInsert;
  }
  {  // Need to narrow scope of this lock, or SELECT and COPY_FROM enters a dead lock
    // after SELECT has locked UpdateDeleteLock and COPY_FROM has locked fragmentInfoMutex_
    // while SELECT waits for fragmentInfoMutex_ and COPY_FROM waits for UpdateDeleteLock

    mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
    for (auto partIt = fragmentInfoVec_.begin() + startFragment; partIt != fragmentInfoVec_.end(); ++partIt) {
      partIt->setPhysicalNumTuples(partIt->shadowNumTuples);
      partIt->setChunkMetadataMap(partIt->shadowChunkMetadataMap);
    }
  }
  numTuples_ += insertDataStruct.numRows;
  dropFragmentsToSize(maxRows_);
}

FragmentInfo* InsertOrderFragmenter::createNewFragment(const Data_Namespace::MemoryLevel memoryLevel) {
  // also sets the new fragment as the insertBuffer for each column

  maxFragmentId_++;
  FragmentInfo newFragmentInfo;
  newFragmentInfo.fragmentId = maxFragmentId_;
  newFragmentInfo.shadowNumTuples = 0;
  newFragmentInfo.setPhysicalNumTuples(0);
  for (const auto levelSize : dataMgr_->levelSizes_) {
    newFragmentInfo.deviceIds.push_back(newFragmentInfo.fragmentId % levelSize);
  }
  newFragmentInfo.physicalTableId = physicalTableId_;
  newFragmentInfo.shard = shard_;

  for (map<int, Chunk>::iterator colMapIt = columnMap_.begin(); colMapIt != columnMap_.end(); ++colMapIt) {
    // colMapIt->second.unpin_buffer();
    ChunkKey chunkKey = chunkKeyPrefix_;
    chunkKey.push_back(colMapIt->second.get_column_desc()->columnId);
    chunkKey.push_back(maxFragmentId_);
    colMapIt->second.createChunkBuffer(
        dataMgr_, chunkKey, memoryLevel, newFragmentInfo.deviceIds[static_cast<int>(memoryLevel)], pageSize_);
    colMapIt->second.init_encoder();
  }

  mapd_lock_guard<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
  fragmentInfoVec_.push_back(newFragmentInfo);
  return &(fragmentInfoVec_.back());
}

TableInfo InsertOrderFragmenter::getFragmentsForQuery() {
  mapd_shared_lock<mapd_shared_mutex> readLock(fragmentInfoMutex_);
  TableInfo queryInfo;
  queryInfo.chunkKeyPrefix = chunkKeyPrefix_;
  // right now we don't test predicate, so just return (copy of) all fragments
  bool fragmentsExist = false;
  if (fragmentInfoVec_.empty()) {
    // If we have no fragments add a dummy empty fragment to make the executor
    // not have separate logic for 0-row tables
    int maxFragmentId = 0;
    FragmentInfo emptyFragmentInfo;
    emptyFragmentInfo.fragmentId = maxFragmentId;
    emptyFragmentInfo.shadowNumTuples = 0;
    emptyFragmentInfo.setPhysicalNumTuples(0);
    emptyFragmentInfo.deviceIds.resize(dataMgr_->levelSizes_.size());
    emptyFragmentInfo.physicalTableId = physicalTableId_;
    emptyFragmentInfo.shard = shard_;
    queryInfo.fragments.push_back(emptyFragmentInfo);
  } else {
    fragmentsExist = true;
    queryInfo.fragments = fragmentInfoVec_;  // makes a copy
  }
  readLock.unlock();
  queryInfo.setPhysicalNumTuples(0);
  auto partIt = queryInfo.fragments.begin();
  if (fragmentsExist) {
    while (partIt != queryInfo.fragments.end()) {
      if (partIt->getPhysicalNumTuples() == 0) {
        // this means that a concurrent insert query inserted tuples into a new fragment but when the query came in we
        // didn't have this fragment.
        // To make sure we don't mess up the executor we delete this
        // fragment from the metadatamap (fixes earlier bug found
        // 2015-05-08)
        partIt = queryInfo.fragments.erase(partIt);
      } else {
        queryInfo.setPhysicalNumTuples(queryInfo.getPhysicalNumTuples() + partIt->getPhysicalNumTuples());
        ++partIt;
      }
    }
  } else {
    // We added a dummy fragment and know the table is empty
    queryInfo.setPhysicalNumTuples(0);
  }
  return queryInfo;
}

}  // Fragmenter_Namespace
