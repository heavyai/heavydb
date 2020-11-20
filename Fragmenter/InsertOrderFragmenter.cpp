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

#include "Fragmenter/InsertOrderFragmenter.h"

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/DataMgr.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "LockMgr/LockMgr.h"
#include "Logger/Logger.h"

#include "Shared/checked_alloc.h"
#include "Shared/thread_count.h"

#define DROP_FRAGMENT_FACTOR \
  0.97  // drop to 97% of max so we don't keep adding and dropping fragments

using Chunk_NS::Chunk;
using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;

bool g_use_table_device_offset{true};

using namespace std;

namespace Fragmenter_Namespace {

InsertOrderFragmenter::InsertOrderFragmenter(
    const vector<int> chunkKeyPrefix,
    vector<Chunk>& chunkVec,
    Data_Namespace::DataMgr* dataMgr,
    Catalog_Namespace::Catalog* catalog,
    const int physicalTableId,
    const int shard,
    const size_t maxFragmentRows,
    const size_t maxChunkSize,
    const size_t pageSize,
    const size_t maxRows,
    const Data_Namespace::MemoryLevel defaultInsertLevel,
    const bool uses_foreign_storage)
    : chunkKeyPrefix_(chunkKeyPrefix)
    , dataMgr_(dataMgr)
    , catalog_(catalog)
    , physicalTableId_(physicalTableId)
    , shard_(shard)
    , maxFragmentRows_(std::min<size_t>(maxFragmentRows, maxRows))
    , pageSize_(pageSize)
    , numTuples_(0)
    , maxFragmentId_(-1)
    , maxChunkSize_(maxChunkSize)
    , maxRows_(maxRows)
    , fragmenterType_("insert_order")
    , defaultInsertLevel_(defaultInsertLevel)
    , uses_foreign_storage_(uses_foreign_storage)
    , hasMaterializedRowId_(false)
    , mutex_access_inmem_states(new std::mutex) {
  // Note that Fragmenter is not passed virtual columns and so should only
  // find row id column if it is non virtual

  for (auto colIt = chunkVec.begin(); colIt != chunkVec.end(); ++colIt) {
    int columnId = colIt->getColumnDesc()->columnId;
    columnMap_[columnId] = *colIt;
    if (colIt->getColumnDesc()->columnName == "rowid") {
      hasMaterializedRowId_ = true;
      rowIdColId_ = columnId;
    }
  }
  conditionallyInstantiateFileMgrWithParams();
  getChunkMetadata();
}

InsertOrderFragmenter::~InsertOrderFragmenter() {}

namespace {

/**
 * Offset the fragment ID by the table ID, meaning single fragment tables end up balanced
 * across multiple GPUs instead of all falling to GPU 0.
 */
int compute_device_for_fragment(const int table_id,
                                const int fragment_id,
                                const int num_devices) {
  if (g_use_table_device_offset) {
    return (table_id + fragment_id) % num_devices;
  } else {
    return fragment_id % num_devices;
  }
}

}  // namespace

void InsertOrderFragmenter::conditionallyInstantiateFileMgrWithParams() {
  // Somewhat awkward to do this in Fragmenter, but FileMgrs are not instantiated until
  // first use by Fragmenter, and until maxRollbackEpochs param, no options were set in
  // storage per table
  if (defaultInsertLevel_ == Data_Namespace::MemoryLevel::DISK_LEVEL) {
    const TableDescriptor* td =
        catalog_->getMetadataForTable(physicalTableId_, false /*populateFragmenter*/);
    if (td->maxRollbackEpochs != DEFAULT_MAX_ROLLBACK_EPOCHS) {
      File_Namespace::FileMgrParams fileMgrParams;
      fileMgrParams.max_rollback_epochs = td->maxRollbackEpochs;
      dataMgr_->getGlobalFileMgr()->setFileMgrParams(
          chunkKeyPrefix_[0], chunkKeyPrefix_[1], fileMgrParams);
    }
  }
}

void InsertOrderFragmenter::getChunkMetadata() {
  if (uses_foreign_storage_ ||
      defaultInsertLevel_ == Data_Namespace::MemoryLevel::DISK_LEVEL) {
    // memory-resident tables won't have anything on disk
    ChunkMetadataVector chunk_metadata;
    dataMgr_->getChunkMetadataVecForKeyPrefix(chunk_metadata, chunkKeyPrefix_);

    // data comes like this - database_id, table_id, column_id, fragment_id
    // but lets sort by database_id, table_id, fragment_id, column_id

    int fragment_subkey_index = 3;
    std::sort(chunk_metadata.begin(),
              chunk_metadata.end(),
              [&](const auto& pair1, const auto& pair2) {
                return pair1.first[3] < pair2.first[3];
              });

    for (auto chunk_itr = chunk_metadata.begin(); chunk_itr != chunk_metadata.end();
         ++chunk_itr) {
      int cur_column_id = chunk_itr->first[2];
      int cur_fragment_id = chunk_itr->first[fragment_subkey_index];

      if (fragmentInfoVec_.empty() ||
          cur_fragment_id != fragmentInfoVec_.back()->fragmentId) {
        auto new_fragment_info = std::make_unique<Fragmenter_Namespace::FragmentInfo>();
        CHECK(new_fragment_info);
        maxFragmentId_ = cur_fragment_id;
        new_fragment_info->fragmentId = cur_fragment_id;
        new_fragment_info->setPhysicalNumTuples(chunk_itr->second->numElements);
        numTuples_ += new_fragment_info->getPhysicalNumTuples();
        for (const auto level_size : dataMgr_->levelSizes_) {
          new_fragment_info->deviceIds.push_back(
              compute_device_for_fragment(physicalTableId_, cur_fragment_id, level_size));
        }
        new_fragment_info->shadowNumTuples = new_fragment_info->getPhysicalNumTuples();
        new_fragment_info->physicalTableId = physicalTableId_;
        new_fragment_info->shard = shard_;
        fragmentInfoVec_.emplace_back(std::move(new_fragment_info));
      } else {
        if (chunk_itr->second->numElements !=
            fragmentInfoVec_.back()->getPhysicalNumTuples()) {
          LOG(FATAL) << "Inconsistency in num tuples within fragment for table " +
                            std::to_string(physicalTableId_) + ", Column " +
                            std::to_string(cur_column_id) + ". Fragment Tuples: " +
                            std::to_string(
                                fragmentInfoVec_.back()->getPhysicalNumTuples()) +
                            ", Chunk Tuples: " +
                            std::to_string(chunk_itr->second->numElements);
        }
      }
      CHECK(fragmentInfoVec_.back().get());
      fragmentInfoVec_.back().get()->setChunkMetadata(cur_column_id, chunk_itr->second);
    }
  }

  size_t maxFixedColSize = 0;

  for (auto colIt = columnMap_.begin(); colIt != columnMap_.end(); ++colIt) {
    auto size = colIt->second.getColumnDesc()->columnType.get_size();
    if (size == -1) {  // variable length
      varLenColInfo_.insert(std::make_pair(colIt->first, 0));
      size = 8;  // b/c we use this for string and array indices - gross to have magic
                 // number here
    }
    CHECK_GE(size, 0);
    maxFixedColSize = std::max(maxFixedColSize, static_cast<size_t>(size));
  }

  // this is maximum number of rows assuming everything is fixed length
  maxFragmentRows_ = std::min(maxFragmentRows_, maxChunkSize_ / maxFixedColSize);

  if (!uses_foreign_storage_ && fragmentInfoVec_.size() > 0) {
    // Now need to get the insert buffers for each column - should be last
    // fragment
    int lastFragmentId = fragmentInfoVec_.back()->fragmentId;
    // TODO: add accessor here for safe indexing
    int deviceId =
        fragmentInfoVec_.back()->deviceIds[static_cast<int>(defaultInsertLevel_)];
    for (auto colIt = columnMap_.begin(); colIt != columnMap_.end(); ++colIt) {
      ChunkKey insertKey = chunkKeyPrefix_;  // database_id and table_id
      insertKey.push_back(colIt->first);     // column id
      insertKey.push_back(lastFragmentId);   // fragment id
      colIt->second.getChunkBuffer(dataMgr_, insertKey, defaultInsertLevel_, deviceId);
      auto varLenColInfoIt = varLenColInfo_.find(colIt->first);
      if (varLenColInfoIt != varLenColInfo_.end()) {
        varLenColInfoIt->second = colIt->second.getBuffer()->size();
      }
    }
  }
}

void InsertOrderFragmenter::dropFragmentsToSize(const size_t maxRows) {
  // not safe to call from outside insertData
  // b/c depends on insertLock around numTuples_

  // don't ever drop the only fragment!
  if (numTuples_ == fragmentInfoVec_.back()->getPhysicalNumTuples()) {
    return;
  }

  if (numTuples_ > maxRows) {
    size_t preNumTuples = numTuples_;
    vector<int> dropFragIds;
    size_t targetRows = maxRows * DROP_FRAGMENT_FACTOR;
    while (numTuples_ > targetRows) {
      CHECK_GT(fragmentInfoVec_.size(), size_t(0));
      size_t numFragTuples = fragmentInfoVec_[0]->getPhysicalNumTuples();
      dropFragIds.push_back(fragmentInfoVec_[0]->fragmentId);
      fragmentInfoVec_.pop_front();
      CHECK_GE(numTuples_, numFragTuples);
      numTuples_ -= numFragTuples;
    }
    deleteFragments(dropFragIds);
    LOG(INFO) << "dropFragmentsToSize, numTuples pre: " << preNumTuples
              << " post: " << numTuples_ << " maxRows: " << maxRows;
  }
}

void InsertOrderFragmenter::deleteFragments(const vector<int>& dropFragIds) {
  // Fix a verified loophole on sharded logical table which is locked using logical
  // tableId while it's its physical tables that can come here when fragments overflow
  // during COPY. Locks on a logical table and its physical tables never intersect, which
  // means potential races. It'll be an overkill to resolve a logical table to physical
  // tables in DBHandler, ParseNode or other higher layers where the logical table is
  // locked with Table Read/Write locks; it's easier to lock the logical table of its
  // physical tables. A downside of this approach may be loss of parallel execution of
  // deleteFragments across physical tables. Because deleteFragments is a short in-memory
  // operation, the loss seems not a big deal.
  auto chunkKeyPrefix = chunkKeyPrefix_;
  if (shard_ >= 0) {
    chunkKeyPrefix[1] = catalog_->getLogicalTableId(chunkKeyPrefix[1]);
  }

  // need to keep lock seq as TableLock >> fragmentInfoMutex_ or
  // SELECT and COPY may enter a deadlock
  const auto delete_lock =
      lockmgr::TableDataLockMgr::getWriteLockForTable(chunkKeyPrefix);

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

void InsertOrderFragmenter::updateColumnChunkMetadata(
    const ColumnDescriptor* cd,
    const int fragment_id,
    const std::shared_ptr<ChunkMetadata> metadata) {
  // synchronize concurrent accesses to fragmentInfoVec_
  mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);

  CHECK(metadata.get());
  auto fragment_info = getFragmentInfo(fragment_id);
  CHECK(fragment_info);
  fragment_info->setChunkMetadata(cd->columnId, metadata);
}

void InsertOrderFragmenter::updateChunkStats(
    const ColumnDescriptor* cd,
    std::unordered_map</*fragment_id*/ int, ChunkStats>& stats_map,
    std::optional<Data_Namespace::MemoryLevel> memory_level) {
  // synchronize concurrent accesses to fragmentInfoVec_
  mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
  /**
   * WARNING: This method is entirely unlocked. Higher level locks are expected to prevent
   * any table read or write during a chunk metadata update, since we need to modify
   * various buffers and metadata maps.
   */
  if (shard_ >= 0) {
    LOG(WARNING) << "Skipping chunk stats update for logical table " << physicalTableId_;
  }

  CHECK(cd);
  const auto column_id = cd->columnId;
  const auto col_itr = columnMap_.find(column_id);
  CHECK(col_itr != columnMap_.end());

  for (auto const& fragment : fragmentInfoVec_) {
    auto stats_itr = stats_map.find(fragment->fragmentId);
    if (stats_itr != stats_map.end()) {
      auto chunk_meta_it = fragment->getChunkMetadataMapPhysical().find(column_id);
      CHECK(chunk_meta_it != fragment->getChunkMetadataMapPhysical().end());
      ChunkKey chunk_key{catalog_->getCurrentDB().dbId,
                         physicalTableId_,
                         column_id,
                         fragment->fragmentId};
      auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                             &catalog_->getDataMgr(),
                                             chunk_key,
                                             memory_level.value_or(defaultInsertLevel_),
                                             0,
                                             chunk_meta_it->second->numBytes,
                                             chunk_meta_it->second->numElements);
      auto buf = chunk->getBuffer();
      CHECK(buf);
      if (!buf->hasEncoder()) {
        throw std::runtime_error("No encoder for chunk " + show_chunk(chunk_key));
      }
      auto encoder = buf->getEncoder();

      auto chunk_stats = stats_itr->second;

      auto old_chunk_metadata = std::make_shared<ChunkMetadata>();
      encoder->getMetadata(old_chunk_metadata);
      auto& old_chunk_stats = old_chunk_metadata->chunkStats;

      const bool didResetStats = encoder->resetChunkStats(chunk_stats);
      // Use the logical type to display data, since the encoding should be ignored
      const auto logical_ti = cd->columnType.is_dict_encoded_string()
                                  ? SQLTypeInfo(kBIGINT)
                                  : get_logical_type_info(cd->columnType);
      if (!didResetStats) {
        VLOG(3) << "Skipping chunk stats reset for " << show_chunk(chunk_key);
        VLOG(3) << "Max: " << DatumToString(old_chunk_stats.max, logical_ti) << " -> "
                << DatumToString(chunk_stats.max, logical_ti);
        VLOG(3) << "Min: " << DatumToString(old_chunk_stats.min, logical_ti) << " -> "
                << DatumToString(chunk_stats.min, logical_ti);
        VLOG(3) << "Nulls: " << (chunk_stats.has_nulls ? "True" : "False");
        continue;  // move to next fragment
      }

      VLOG(2) << "Resetting chunk stats for " << show_chunk(chunk_key);
      VLOG(2) << "Max: " << DatumToString(old_chunk_stats.max, logical_ti) << " -> "
              << DatumToString(chunk_stats.max, logical_ti);
      VLOG(2) << "Min: " << DatumToString(old_chunk_stats.min, logical_ti) << " -> "
              << DatumToString(chunk_stats.min, logical_ti);
      VLOG(2) << "Nulls: " << (chunk_stats.has_nulls ? "True" : "False");

      // Reset fragment metadata map and set buffer to dirty
      auto new_metadata = std::make_shared<ChunkMetadata>();
      // Run through fillChunkStats to ensure any transformations to the raw metadata
      // values get applied (e.g. for date in days)
      encoder->getMetadata(new_metadata);

      fragment->setChunkMetadata(column_id, new_metadata);
      fragment->shadowChunkMetadataMap =
          fragment->getChunkMetadataMap();  // TODO(adb): needed?
      if (defaultInsertLevel_ == Data_Namespace::DISK_LEVEL) {
        buf->setDirty();
      }
    } else {
      LOG(WARNING) << "No chunk stats update found for fragment " << fragment->fragmentId
                   << ", table " << physicalTableId_ << ", "
                   << ", column " << column_id;
    }
  }
}

FragmentInfo* InsertOrderFragmenter::getFragmentInfo(const int fragment_id) const {
  auto fragment_it = std::find_if(fragmentInfoVec_.begin(),
                                  fragmentInfoVec_.end(),
                                  [fragment_id](const auto& fragment) -> bool {
                                    return fragment->fragmentId == fragment_id;
                                  });
  CHECK(fragment_it != fragmentInfoVec_.end());
  return fragment_it->get();
}

bool InsertOrderFragmenter::isAddingNewColumns(const InsertData& insert_data) const {
  bool all_columns_already_exist = true, all_columns_are_new = true;
  for (const auto column_id : insert_data.columnIds) {
    if (columnMap_.find(column_id) == columnMap_.end()) {
      all_columns_already_exist = false;
    } else {
      all_columns_are_new = false;
    }
  }
  // only one should be TRUE
  bool either_all_exist_or_all_new = all_columns_already_exist ^ all_columns_are_new;
  CHECK(either_all_exist_or_all_new);
  return all_columns_are_new;
}

void InsertOrderFragmenter::insertData(InsertData& insert_data_struct) {
  // TODO: this local lock will need to be centralized when ALTER COLUMN is added, bc
  try {
    // prevent two threads from trying to insert into the same table simultaneously
    mapd_unique_lock<mapd_shared_mutex> insertLock(insertMutex_);
    if (!isAddingNewColumns(insert_data_struct)) {
      insertDataImpl(insert_data_struct);
    } else {
      addColumns(insert_data_struct);
    }
    if (defaultInsertLevel_ ==
        Data_Namespace::DISK_LEVEL) {  // only checkpoint if data is resident on disk
      dataMgr_->checkpoint(
          chunkKeyPrefix_[0],
          chunkKeyPrefix_[1]);  // need to checkpoint here to remove window for corruption
    }
  } catch (...) {
    auto table_epochs = catalog_->getTableEpochs(insert_data_struct.databaseId,
                                                 insert_data_struct.tableId);
    // the statement below deletes *this* object!
    // relying on exception propagation at this stage
    // until we can sort this out in a cleaner fashion
    catalog_->setTableEpochs(insert_data_struct.databaseId, table_epochs);
    throw;
  }
}

void InsertOrderFragmenter::insertDataNoCheckpoint(InsertData& insert_data_struct) {
  // TODO: this local lock will need to be centralized when ALTER COLUMN is added, bc
  mapd_unique_lock<mapd_shared_mutex> insertLock(
      insertMutex_);  // prevent two threads from trying to insert into the same table
                      // simultaneously
  if (!isAddingNewColumns(insert_data_struct)) {
    insertDataImpl(insert_data_struct);
  } else {
    addColumns(insert_data_struct);
  }
}

void InsertOrderFragmenter::addColumns(const InsertData& insertDataStruct) {
  // synchronize concurrent accesses to fragmentInfoVec_
  mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
  size_t numRowsLeft = insertDataStruct.numRows;
  for (const auto columnId : insertDataStruct.columnIds) {
    CHECK(columnMap_.end() == columnMap_.find(columnId));
    const auto columnDesc = catalog_->getMetadataForColumn(physicalTableId_, columnId);
    CHECK(columnDesc);
    columnMap_.emplace(columnId, Chunk_NS::Chunk(columnDesc));
  }
  try {
    for (auto const& fragmentInfo : fragmentInfoVec_) {
      fragmentInfo->shadowChunkMetadataMap = fragmentInfo->getChunkMetadataMapPhysical();
      auto numRowsToInsert = fragmentInfo->getPhysicalNumTuples();  // not getNumTuples()
      size_t numRowsCanBeInserted;
      for (size_t i = 0; i < insertDataStruct.columnIds.size(); i++) {
        auto columnId = insertDataStruct.columnIds[i];
        auto colDesc = catalog_->getMetadataForColumn(physicalTableId_, columnId);
        CHECK(colDesc);
        CHECK(columnMap_.find(columnId) != columnMap_.end());

        ChunkKey chunkKey = chunkKeyPrefix_;
        chunkKey.push_back(columnId);
        chunkKey.push_back(fragmentInfo->fragmentId);

        auto colMapIt = columnMap_.find(columnId);
        auto& chunk = colMapIt->second;
        if (chunk.isChunkOnDevice(
                dataMgr_,
                chunkKey,
                defaultInsertLevel_,
                fragmentInfo->deviceIds[static_cast<int>(defaultInsertLevel_)])) {
          dataMgr_->deleteChunksWithPrefix(chunkKey);
        }
        chunk.createChunkBuffer(
            dataMgr_,
            chunkKey,
            defaultInsertLevel_,
            fragmentInfo->deviceIds[static_cast<int>(defaultInsertLevel_)]);
        chunk.initEncoder();

        try {
          DataBlockPtr dataCopy = insertDataStruct.data[i];
          auto size = colDesc->columnType.get_size();
          if (0 > size) {
            std::unique_lock<std::mutex> lck(*mutex_access_inmem_states);
            varLenColInfo_[columnId] = 0;
            numRowsCanBeInserted = chunk.getNumElemsForBytesInsertData(
                dataCopy, numRowsToInsert, 0, maxChunkSize_, true);
          } else {
            numRowsCanBeInserted = maxChunkSize_ / size;
          }

          // FIXME: abort a case in which new column is wider than existing columns
          if (numRowsCanBeInserted < numRowsToInsert) {
            throw std::runtime_error("new column '" + colDesc->columnName +
                                     "' wider than existing columns is not supported");
          }

          auto chunkMetadata = chunk.appendData(dataCopy, numRowsToInsert, 0, true);
          fragmentInfo->shadowChunkMetadataMap[columnId] = chunkMetadata;

          // update total size of var-len column in (actually the last) fragment
          if (0 > size) {
            std::unique_lock<std::mutex> lck(*mutex_access_inmem_states);
            varLenColInfo_[columnId] = chunk.getBuffer()->size();
          }
        } catch (...) {
          dataMgr_->deleteChunksWithPrefix(chunkKey);
          throw;
        }
      }
      numRowsLeft -= numRowsToInsert;
    }
    CHECK(0 == numRowsLeft);
  } catch (const std::exception& e) {
    for (const auto columnId : insertDataStruct.columnIds) {
      columnMap_.erase(columnId);
    }
    throw e;
  }

  for (auto const& fragmentInfo : fragmentInfoVec_) {
    fragmentInfo->setChunkMetadataMap(fragmentInfo->shadowChunkMetadataMap);
  }
}

void InsertOrderFragmenter::dropColumns(const std::vector<int>& columnIds) {
  // prevent concurrent insert rows and drop column
  mapd_unique_lock<mapd_shared_mutex> insertLock(insertMutex_);
  // synchronize concurrent accesses to fragmentInfoVec_
  mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
  for (auto const& fragmentInfo : fragmentInfoVec_) {
    fragmentInfo->shadowChunkMetadataMap = fragmentInfo->getChunkMetadataMapPhysical();
  }

  for (const auto columnId : columnIds) {
    auto cit = columnMap_.find(columnId);
    if (columnMap_.end() != cit) {
      columnMap_.erase(cit);
    }

    vector<int> fragPrefix = chunkKeyPrefix_;
    fragPrefix.push_back(columnId);
    dataMgr_->deleteChunksWithPrefix(fragPrefix);

    for (const auto& fragmentInfo : fragmentInfoVec_) {
      auto cmdit = fragmentInfo->shadowChunkMetadataMap.find(columnId);
      if (fragmentInfo->shadowChunkMetadataMap.end() != cmdit) {
        fragmentInfo->shadowChunkMetadataMap.erase(cmdit);
      }
    }
  }
  for (const auto& fragmentInfo : fragmentInfoVec_) {
    fragmentInfo->setChunkMetadataMap(fragmentInfo->shadowChunkMetadataMap);
  }
}

bool InsertOrderFragmenter::hasDeletedRows(const int delete_column_id) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(fragmentInfoMutex_);

  for (auto const& fragment : fragmentInfoVec_) {
    auto chunk_meta_it = fragment->getChunkMetadataMapPhysical().find(delete_column_id);
    CHECK(chunk_meta_it != fragment->getChunkMetadataMapPhysical().end());
    const auto& chunk_stats = chunk_meta_it->second->chunkStats;
    if (chunk_stats.max.tinyintval == 1) {
      return true;
    }
  }
  return false;
}

void InsertOrderFragmenter::insertDataImpl(InsertData& insert_data) {
  // populate deleted system column if it should exist, as it will not come from client
  std::unique_ptr<int8_t[]> data_for_deleted_column;
  for (const auto& cit : columnMap_) {
    if (cit.second.getColumnDesc()->isDeletedCol) {
      data_for_deleted_column.reset(new int8_t[insert_data.numRows]);
      memset(data_for_deleted_column.get(), 0, insert_data.numRows);
      insert_data.data.emplace_back(DataBlockPtr{data_for_deleted_column.get()});
      insert_data.columnIds.push_back(cit.second.getColumnDesc()->columnId);
      insert_data.is_default.push_back(false);
      break;
    }
  }
  CHECK(insert_data.is_default.size() == insert_data.columnIds.size());
  std::unordered_map<int, int> inverseInsertDataColIdMap;
  for (size_t insertId = 0; insertId < insert_data.columnIds.size(); ++insertId) {
    inverseInsertDataColIdMap.insert(
        std::make_pair(insert_data.columnIds[insertId], insertId));
  }

  size_t numRowsLeft = insert_data.numRows;
  size_t numRowsInserted = 0;
  vector<DataBlockPtr> dataCopy =
      insert_data.data;  // bc append data will move ptr forward and this violates
                         // constness of InsertData
  if (numRowsLeft <= 0) {
    return;
  }

  FragmentInfo* currentFragment{nullptr};

  if (fragmentInfoVec_.empty()) {  // if no fragments exist for table
    currentFragment = createNewFragment(defaultInsertLevel_);
  } else {
    currentFragment = fragmentInfoVec_.back().get();
  }
  CHECK(currentFragment);

  size_t startFragment = fragmentInfoVec_.size() - 1;

  while (numRowsLeft > 0) {  // may have to create multiple fragments for bulk insert
    // loop until done inserting all rows
    CHECK_LE(currentFragment->shadowNumTuples, maxFragmentRows_);
    size_t rowsLeftInCurrentFragment =
        maxFragmentRows_ - currentFragment->shadowNumTuples;
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
                                         dataCopy[insertIdIt->second],
                                         numRowsToInsert,
                                         numRowsInserted,
                                         bytesLeft,
                                         insert_data.is_default[insertIdIt->second]));
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
                                         dataCopy[insertIdIt->second],
                                         numRowsToInsert,
                                         numRowsInserted,
                                         bytesLeft,
                                         insert_data.is_default[insertIdIt->second]));
        }
      }
    }

    CHECK_GT(numRowsToInsert, size_t(0));  // would put us into an endless loop as we'd
                                           // never be able to insert anything

    {
      mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
      // for each column, append the data in the appropriate insert buffer
      for (size_t i = 0; i < insert_data.columnIds.size(); ++i) {
        int columnId = insert_data.columnIds[i];
        auto colMapIt = columnMap_.find(columnId);
        CHECK(colMapIt != columnMap_.end());
        currentFragment->shadowChunkMetadataMap[columnId] = colMapIt->second.appendData(
            dataCopy[i], numRowsToInsert, numRowsInserted, insert_data.is_default[i]);
        auto varLenColInfoIt = varLenColInfo_.find(columnId);
        if (varLenColInfoIt != varLenColInfo_.end()) {
          varLenColInfoIt->second = colMapIt->second.getBuffer()->size();
        }
      }
      if (hasMaterializedRowId_) {
        size_t startId = maxFragmentRows_ * currentFragment->fragmentId +
                         currentFragment->shadowNumTuples;
        auto row_id_data = std::make_unique<int64_t[]>(numRowsToInsert);
        for (size_t i = 0; i < numRowsToInsert; ++i) {
          row_id_data[i] = i + startId;
        }
        DataBlockPtr rowIdBlock;
        rowIdBlock.numbersPtr = reinterpret_cast<int8_t*>(row_id_data.get());
        auto colMapIt = columnMap_.find(rowIdColId_);
        currentFragment->shadowChunkMetadataMap[rowIdColId_] =
            colMapIt->second.appendData(rowIdBlock, numRowsToInsert, numRowsInserted);
      }

      currentFragment->shadowNumTuples =
          fragmentInfoVec_.back()->getPhysicalNumTuples() + numRowsToInsert;
      numRowsLeft -= numRowsToInsert;
      numRowsInserted += numRowsToInsert;
      for (auto partIt = fragmentInfoVec_.begin() + startFragment;
           partIt != fragmentInfoVec_.end();
           ++partIt) {
        auto fragment_ptr = partIt->get();
        fragment_ptr->setPhysicalNumTuples(fragment_ptr->shadowNumTuples);
        fragment_ptr->setChunkMetadataMap(fragment_ptr->shadowChunkMetadataMap);
      }
    }
  }
  numTuples_ += insert_data.numRows;
  dropFragmentsToSize(maxRows_);
}

FragmentInfo* InsertOrderFragmenter::createNewFragment(
    const Data_Namespace::MemoryLevel memoryLevel) {
  // also sets the new fragment as the insertBuffer for each column

  maxFragmentId_++;
  auto newFragmentInfo = std::make_unique<FragmentInfo>();
  newFragmentInfo->fragmentId = maxFragmentId_;
  newFragmentInfo->shadowNumTuples = 0;
  newFragmentInfo->setPhysicalNumTuples(0);
  for (const auto levelSize : dataMgr_->levelSizes_) {
    newFragmentInfo->deviceIds.push_back(compute_device_for_fragment(
        physicalTableId_, newFragmentInfo->fragmentId, levelSize));
  }
  newFragmentInfo->physicalTableId = physicalTableId_;
  newFragmentInfo->shard = shard_;

  for (map<int, Chunk>::iterator colMapIt = columnMap_.begin();
       colMapIt != columnMap_.end();
       ++colMapIt) {
    ChunkKey chunkKey = chunkKeyPrefix_;
    chunkKey.push_back(colMapIt->second.getColumnDesc()->columnId);
    chunkKey.push_back(maxFragmentId_);
    colMapIt->second.createChunkBuffer(
        dataMgr_,
        chunkKey,
        memoryLevel,
        newFragmentInfo->deviceIds[static_cast<int>(memoryLevel)],
        pageSize_);
    colMapIt->second.initEncoder();
  }

  mapd_lock_guard<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
  fragmentInfoVec_.push_back(std::move(newFragmentInfo));
  return fragmentInfoVec_.back().get();
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
    std::for_each(
        fragmentInfoVec_.begin(),
        fragmentInfoVec_.end(),
        [&queryInfo](const auto& fragment_owned_ptr) {
          queryInfo.fragments.emplace_back(*fragment_owned_ptr);  // makes a copy
        });
  }
  readLock.unlock();
  queryInfo.setPhysicalNumTuples(0);
  auto partIt = queryInfo.fragments.begin();
  if (fragmentsExist) {
    while (partIt != queryInfo.fragments.end()) {
      if (partIt->getPhysicalNumTuples() == 0) {
        // this means that a concurrent insert query inserted tuples into a new fragment
        // but when the query came in we didn't have this fragment. To make sure we
        // don't mess up the executor we delete this fragment from the metadatamap
        // (fixes earlier bug found 2015-05-08)
        partIt = queryInfo.fragments.erase(partIt);
      } else {
        queryInfo.setPhysicalNumTuples(queryInfo.getPhysicalNumTuples() +
                                       partIt->getPhysicalNumTuples());
        ++partIt;
      }
    }
  } else {
    // We added a dummy fragment and know the table is empty
    queryInfo.setPhysicalNumTuples(0);
  }
  return queryInfo;
}

}  // namespace Fragmenter_Namespace
