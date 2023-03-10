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
#include "DataMgr/DataConversion/ConversionFactory.h"
#include "DataMgr/DataMgr.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "LockMgr/LockMgr.h"
#include "Logger/Logger.h"
#include "Utils/DdlUtils.h"

#include "Shared/checked_alloc.h"
#include "Shared/scope.h"
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

ChunkKey get_chunk_key(const ChunkKey& prefix, int column_id, int fragment_id) {
  ChunkKey key = prefix;  // database_id and table_id
  key.push_back(column_id);
  key.push_back(fragment_id);  // fragment id
  return key;
}

struct ArrayElemTypeChunk {
  ColumnDescriptor temp_cd;
  Chunk_NS::Chunk chunk;
};

void create_array_elem_type_chunk(ArrayElemTypeChunk& array_chunk,
                                  const ColumnDescriptor* array_cd) {
  array_chunk.temp_cd = *array_cd;
  array_chunk.temp_cd.columnType = array_cd->columnType.get_elem_type();
  array_chunk.chunk = Chunk_NS::Chunk{&array_chunk.temp_cd, true};
}

class BaseAlterColumnContext {
 public:
  BaseAlterColumnContext(int device_id,
                         const ChunkKey& chunk_key_prefix,
                         Fragmenter_Namespace::FragmentInfo* fragment_info,
                         const ColumnDescriptor* src_cd,
                         const ColumnDescriptor* dst_cd,
                         const size_t num_elements,
                         Data_Namespace::DataMgr* data_mgr,
                         Catalog_Namespace::Catalog* catalog,
                         std::map<int, Chunk_NS::Chunk>& column_map)
      : device_id_(device_id)
      , chunk_key_prefix_(chunk_key_prefix)
      , fragment_info_(fragment_info)
      , src_cd_(src_cd)
      , dst_cd_(dst_cd)
      , num_elements_(num_elements)
      , data_mgr_(data_mgr)
      , catalog_(catalog)
      , column_map_(column_map)
      , buffer_(nullptr)
      , index_buffer_(nullptr)
      , disk_level_src_chunk_{src_cd}
      , mem_level_src_chunk_{src_cd} {
    key_ = get_chunk_key(chunk_key_prefix, src_cd->columnId, fragment_info->fragmentId);
  }

  static void unpinChunk(Chunk& chunk) {
    auto buffer = chunk.getBuffer();
    if (buffer) {
      buffer->unPin();
      chunk.setBuffer(nullptr);
    }

    auto index_buffer = chunk.getIndexBuf();
    if (index_buffer) {
      index_buffer->unPin();
      chunk.setIndexBuffer(nullptr);
    }
  }

  void readSourceData() {
    disk_level_src_chunk_.getChunkBuffer(
        data_mgr_, key_, Data_Namespace::MemoryLevel::DISK_LEVEL, device_id_);
    // FIXME: there appears to be a bug where if the `num_elements` is not specified
    // below, the wrong byte count is returned for index buffers
    mem_level_src_chunk_.getChunkBuffer(data_mgr_,
                                        key_,
                                        MemoryLevel::CPU_LEVEL,
                                        0,
                                        disk_level_src_chunk_.getBuffer()->size(),
                                        num_elements_);
    CHECK_EQ(num_elements_,
             mem_level_src_chunk_.getBuffer()->getEncoder()->getNumElems());

    auto db_id = catalog_->getDatabaseId();
    source = data_conversion::create_source(mem_level_src_chunk_, db_id);

    try {
      std::tie(src_data_, std::ignore) = source->getSourceData();
    } catch (std::exception& except) {
      src_data_ = nullptr;
      throw std::runtime_error("Column " + src_cd_->columnName + ": " + except.what());
    }
  }

 protected:
  void createChunkScratchBuffer(Chunk_NS::Chunk& chunk) {
    chunk.setBuffer(data_mgr_->alloc(MemoryLevel::CPU_LEVEL, 0, 0));
    if (chunk.getColumnDesc()->columnType.is_varlen_indeed()) {
      chunk.setIndexBuffer(data_mgr_->alloc(MemoryLevel::CPU_LEVEL, 0, 0));
    }
  }

  void freeChunkScratchBuffer(Chunk_NS::Chunk& chunk) {
    data_mgr_->free(chunk.getBuffer());
    chunk.setBuffer(nullptr);
    if (chunk.getColumnDesc()->columnType.is_varlen_indeed()) {
      data_mgr_->free(chunk.getIndexBuf());
      chunk.setIndexBuffer(nullptr);
    }
  }

  int device_id_;
  const ChunkKey& chunk_key_prefix_;
  Fragmenter_Namespace::FragmentInfo* fragment_info_;
  const ColumnDescriptor* src_cd_;
  const ColumnDescriptor* dst_cd_;
  const size_t num_elements_;
  Data_Namespace::DataMgr* data_mgr_;
  Catalog_Namespace::Catalog* catalog_;
  std::map<int, Chunk_NS::Chunk>& column_map_;

  data_conversion::ConversionFactoryParam param_;
  std::unique_ptr<data_conversion::BaseSource> source;
  AbstractBuffer* buffer_;
  AbstractBuffer* index_buffer_;
  Chunk_NS::Chunk disk_level_src_chunk_;
  Chunk_NS::Chunk mem_level_src_chunk_;
  ChunkKey key_;
  const int8_t* src_data_;
  ArrayElemTypeChunk scalar_temp_chunk_;
};

class GeoAlterColumnContext : public BaseAlterColumnContext {
 public:
  GeoAlterColumnContext(int device_id,
                        const ChunkKey& chunk_key_prefix,
                        Fragmenter_Namespace::FragmentInfo* fragment_info,
                        const ColumnDescriptor* src_cd,
                        const ColumnDescriptor* dst_cd,
                        const size_t num_elements,
                        Data_Namespace::DataMgr* data_mgr,
                        Catalog_Namespace::Catalog* catalog,
                        std::map<int, Chunk_NS::Chunk>& column_map,
                        const std::list<const ColumnDescriptor*>& columns)
      : BaseAlterColumnContext(device_id,
                               chunk_key_prefix,
                               fragment_info,
                               src_cd,
                               dst_cd,
                               num_elements,
                               data_mgr,
                               catalog,
                               column_map)
      , dst_columns_(columns) {}

  void createScratchBuffers() {
    std::list<Chunk_NS::Chunk>& geo_chunks = param_.geo_chunks;
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata = param_.geo_chunk_metadata;
    // create all geo chunk buffers
    for (auto dst_cd : dst_columns_) {
      geo_chunks.emplace_back(dst_cd, true);
      auto& dst_chunk = geo_chunks.back();

      createChunkScratchBuffer(dst_chunk);
      dst_chunk.initEncoder();

      chunk_metadata.push_back(std::make_unique<ChunkMetadata>());
    }
  }

  void deleteScratchBuffers() {
    for (auto& dst_chunk : param_.geo_chunks) {
      freeChunkScratchBuffer(dst_chunk);
    }
  }

  void encodeData(const bool geo_validate_geometry) {
    auto convert_encoder =
        data_conversion::create_string_view_encoder(param_, false, geo_validate_geometry);
    try {
      convert_encoder->encodeAndAppendData(src_data_, num_elements_);
      convert_encoder->finalize(num_elements_);
    } catch (std::exception& except) {
      throw std::runtime_error("Column " + (*dst_columns_.begin())->columnName + ": " +
                               except.what());
    }
  }

  void putBuffersToDisk() {
    auto metadata_it = param_.geo_chunk_metadata.begin();
    auto chunk_it = param_.geo_chunks.begin();
    for (auto dst_cd : dst_columns_) {
      auto& chunk = *chunk_it;
      auto& metadata = *metadata_it;

      auto encoder = chunk.getBuffer()->getEncoder();
      CHECK(encoder);
      encoder->resetChunkStats(metadata->chunkStats);
      encoder->setNumElems(num_elements_);

      ChunkKey dst_key =
          get_chunk_key(chunk_key_prefix_, dst_cd->columnId, fragment_info_->fragmentId);

      if (dst_cd->columnType.is_varlen_indeed()) {
        auto data_key = dst_key;
        data_key.push_back(1);
        auto index_key = dst_key;
        index_key.push_back(2);

        chunk.getBuffer()->setUpdated();
        chunk.getIndexBuf()->setUpdated();

        Chunk fragmenter_chunk{dst_cd, false};
        fragmenter_chunk.setBuffer(
            data_mgr_->getGlobalFileMgr()->putBuffer(data_key, chunk.getBuffer()));
        fragmenter_chunk.setIndexBuffer(
            data_mgr_->getGlobalFileMgr()->putBuffer(index_key, chunk.getIndexBuf()));
        column_map_[src_cd_->columnId] = fragmenter_chunk;

      } else {
        chunk.getBuffer()->setUpdated();

        Chunk fragmenter_chunk{dst_cd, false};
        fragmenter_chunk.setBuffer(
            data_mgr_->getGlobalFileMgr()->putBuffer(dst_key, chunk.getBuffer()));
        column_map_[src_cd_->columnId] = fragmenter_chunk;
      }

      chunk_it++;
      metadata_it++;
    }
  }

 private:
  const std::list<const ColumnDescriptor*>& dst_columns_;
};

class NonGeoAlterColumnContext : public BaseAlterColumnContext {
 public:
  NonGeoAlterColumnContext(int device_id,
                           const ChunkKey& chunk_key_prefix,
                           Fragmenter_Namespace::FragmentInfo* fragment_info,
                           const ColumnDescriptor* src_cd,
                           const ColumnDescriptor* dst_cd,
                           const size_t num_elements,
                           Data_Namespace::DataMgr* data_mgr,
                           Catalog_Namespace::Catalog* catalog_,
                           std::map<int, Chunk_NS::Chunk>& column_map)
      : BaseAlterColumnContext(device_id,
                               chunk_key_prefix,
                               fragment_info,
                               src_cd,
                               dst_cd,
                               num_elements,
                               data_mgr,
                               catalog_,
                               column_map) {}

  void createScratchBuffers() {
    auto db_id = catalog_->getDatabaseId();
    param_.db_id = db_id;
    param_.dst_chunk = Chunk_NS::Chunk{dst_cd_, true};
    if (dst_cd_->columnType.is_array()) {
      create_array_elem_type_chunk(scalar_temp_chunk_, dst_cd_);
      param_.scalar_temp_chunk = scalar_temp_chunk_.chunk;
    }

    auto& dst_chunk = param_.dst_chunk;

    createChunkScratchBuffer(dst_chunk);

    if (dst_cd_->columnType.is_array()) {
      createChunkScratchBuffer(param_.scalar_temp_chunk);
    }

    buffer_ = dst_chunk.getBuffer();
    index_buffer_ = dst_chunk.getIndexBuf();  // nullptr for non-varlen types
  }

  void deleteScratchBuffers() {
    freeChunkScratchBuffer(param_.dst_chunk);
    if (dst_cd_->columnType.is_array()) {
      freeChunkScratchBuffer(param_.scalar_temp_chunk);
    }
  }

  void reencodeData() {
    auto& dst_chunk = param_.dst_chunk;
    disk_level_src_chunk_.getBuffer()->syncEncoder(dst_chunk.getBuffer());
    if (disk_level_src_chunk_.getIndexBuf() && dst_chunk.getIndexBuf()) {
      disk_level_src_chunk_.getIndexBuf()->syncEncoder(dst_chunk.getIndexBuf());
    }

    dst_chunk.initEncoder();

    auto convert_encoder =
        data_conversion::create_string_view_encoder(param_, false, false);  // not geo

    try {
      convert_encoder->encodeAndAppendData(src_data_, num_elements_);
      convert_encoder->finalize(num_elements_);
    } catch (std::exception& except) {
      throw std::runtime_error("Column " + src_cd_->columnName + ": " + except.what());
    }

    auto metadata = convert_encoder->getMetadata(
        dst_cd_->columnType.is_array() ? param_.scalar_temp_chunk : dst_chunk);

    buffer_->getEncoder()->resetChunkStats(metadata->chunkStats);
    buffer_->getEncoder()->setNumElems(num_elements_);

    buffer_->setUpdated();
    if (index_buffer_) {
      index_buffer_->setUpdated();
    }
  }

  void putBuffersToDisk() {
    if (dst_cd_->columnType.is_varlen_indeed()) {
      auto data_key = key_;
      data_key.push_back(1);
      auto index_key = key_;
      index_key.push_back(2);

      Chunk fragmenter_chunk{dst_cd_, false};
      fragmenter_chunk.setBuffer(
          data_mgr_->getGlobalFileMgr()->putBuffer(data_key, buffer_));
      fragmenter_chunk.setIndexBuffer(
          data_mgr_->getGlobalFileMgr()->putBuffer(index_key, index_buffer_));
      column_map_[src_cd_->columnId] = fragmenter_chunk;

    } else {
      Chunk fragmenter_chunk{dst_cd_, false};
      fragmenter_chunk.setBuffer(data_mgr_->getGlobalFileMgr()->putBuffer(key_, buffer_));
      column_map_[src_cd_->columnId] = fragmenter_chunk;
    }
  }
};

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

size_t get_num_rows_to_insert(const size_t rows_left_in_current_fragment,
                              const size_t num_rows_left,
                              const size_t num_rows_inserted,
                              const std::unordered_map<int, size_t>& var_len_col_info,
                              const size_t max_chunk_size,
                              const InsertChunks& insert_chunks,
                              std::map<int, Chunk_NS::Chunk>& column_map,
                              const std::vector<size_t>& valid_row_indices) {
  size_t num_rows_to_insert = min(rows_left_in_current_fragment, num_rows_left);
  if (rows_left_in_current_fragment != 0) {
    for (const auto& var_len_col_info_it : var_len_col_info) {
      CHECK_LE(var_len_col_info_it.second, max_chunk_size);
      size_t bytes_left = max_chunk_size - var_len_col_info_it.second;
      auto find_it = insert_chunks.chunks.find(var_len_col_info_it.first);
      if (find_it == insert_chunks.chunks.end()) {
        continue;
      }
      const auto& chunk = find_it->second;
      auto column_type = chunk->getColumnDesc()->columnType;
      const int8_t* index_buffer_ptr =
          column_type.is_varlen_indeed() ? chunk->getIndexBuf()->getMemoryPtr() : nullptr;
      CHECK(column_type.is_varlen());

      auto col_map_it = column_map.find(var_len_col_info_it.first);
      num_rows_to_insert =
          std::min(num_rows_to_insert,
                   col_map_it->second.getNumElemsForBytesEncodedDataAtIndices(
                       index_buffer_ptr, valid_row_indices, bytes_left));
    }
  }
  return num_rows_to_insert;
}

}  // namespace

void InsertOrderFragmenter::conditionallyInstantiateFileMgrWithParams() {
  // Somewhat awkward to do this in Fragmenter, but FileMgrs are not instantiated until
  // first use by Fragmenter, and until maxRollbackEpochs param, no options were set in
  // storage per table
  if (!uses_foreign_storage_ &&
      defaultInsertLevel_ == Data_Namespace::MemoryLevel::DISK_LEVEL) {
    const TableDescriptor* td =
        catalog_->getMetadataForTable(physicalTableId_, false /*populateFragmenter*/);
    File_Namespace::FileMgrParams fileMgrParams;
    fileMgrParams.max_rollback_epochs = td->maxRollbackEpochs;
    dataMgr_->getGlobalFileMgr()->setFileMgrParams(
        chunkKeyPrefix_[0], chunkKeyPrefix_[1], fileMgrParams);
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
  setLastFragmentVarLenColumnSizes();
}

void InsertOrderFragmenter::dropFragmentsToSize(const size_t max_rows) {
  heavyai::unique_lock<heavyai::shared_mutex> insert_lock(insertMutex_);
  dropFragmentsToSizeNoInsertLock(max_rows);
}

void InsertOrderFragmenter::dropFragmentsToSizeNoInsertLock(const size_t max_rows) {
  // not safe to call from outside insertData
  // b/c depends on insertLock around numTuples_

  // don't ever drop the only fragment!
  if (fragmentInfoVec_.empty() ||
      numTuples_ == fragmentInfoVec_.back()->getPhysicalNumTuples()) {
    return;
  }

  if (numTuples_ > max_rows) {
    size_t preNumTuples = numTuples_;
    vector<int> dropFragIds;
    size_t targetRows = max_rows * DROP_FRAGMENT_FACTOR;
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
              << " post: " << numTuples_ << " maxRows: " << max_rows;
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

  heavyai::unique_lock<heavyai::shared_mutex> writeLock(fragmentInfoMutex_);

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
  heavyai::unique_lock<heavyai::shared_mutex> writeLock(fragmentInfoMutex_);

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
  heavyai::unique_lock<heavyai::shared_mutex> writeLock(fragmentInfoMutex_);
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
          fragment->getChunkMetadataMapPhysicalCopy();  // TODO(adb): needed?
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

void InsertOrderFragmenter::insertChunks(const InsertChunks& insert_chunk) {
  try {
    // prevent two threads from trying to insert into the same table simultaneously
    heavyai::unique_lock<heavyai::shared_mutex> insertLock(insertMutex_);
    insertChunksImpl(insert_chunk);
    if (defaultInsertLevel_ ==
        Data_Namespace::DISK_LEVEL) {  // only checkpoint if data is resident on disk
      dataMgr_->checkpoint(
          chunkKeyPrefix_[0],
          chunkKeyPrefix_[1]);  // need to checkpoint here to remove window for corruption
    }
  } catch (...) {
    auto db_id = insert_chunk.db_id;
    auto table_epochs = catalog_->getTableEpochs(db_id, insert_chunk.table_id);
    // the statement below deletes *this* object!
    // relying on exception propagation at this stage
    // until we can sort this out in a cleaner fashion
    catalog_->setTableEpochs(db_id, table_epochs);
    throw;
  }
}

void InsertOrderFragmenter::insertData(InsertData& insert_data_struct) {
  try {
    // prevent two threads from trying to insert into the same table simultaneously
    heavyai::unique_lock<heavyai::shared_mutex> insertLock(insertMutex_);
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

void InsertOrderFragmenter::insertChunksNoCheckpoint(const InsertChunks& insert_chunk) {
  // TODO: this local lock will need to be centralized when ALTER COLUMN is added, bc
  heavyai::unique_lock<heavyai::shared_mutex> insertLock(
      insertMutex_);  // prevent two threads from trying to insert into the same table
                      // simultaneously
  insertChunksImpl(insert_chunk);
}

void InsertOrderFragmenter::insertDataNoCheckpoint(InsertData& insert_data_struct) {
  // TODO: this local lock will need to be centralized when ALTER COLUMN is added, bc
  heavyai::unique_lock<heavyai::shared_mutex> insertLock(
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
  heavyai::unique_lock<heavyai::shared_mutex> writeLock(fragmentInfoMutex_);
  size_t numRowsLeft = insertDataStruct.numRows;
  for (const auto columnId : insertDataStruct.columnIds) {
    CHECK(columnMap_.end() == columnMap_.find(columnId));
    const auto columnDesc = catalog_->getMetadataForColumn(physicalTableId_, columnId);
    CHECK(columnDesc);
    columnMap_.emplace(columnId, Chunk_NS::Chunk(columnDesc));
  }
  try {
    for (auto const& fragmentInfo : fragmentInfoVec_) {
      fragmentInfo->shadowChunkMetadataMap =
          fragmentInfo->getChunkMetadataMapPhysicalCopy();
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
  heavyai::unique_lock<heavyai::shared_mutex> insertLock(insertMutex_);
  // synchronize concurrent accesses to fragmentInfoVec_
  heavyai::unique_lock<heavyai::shared_mutex> writeLock(fragmentInfoMutex_);
  for (auto const& fragmentInfo : fragmentInfoVec_) {
    fragmentInfo->shadowChunkMetadataMap =
        fragmentInfo->getChunkMetadataMapPhysicalCopy();
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
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(fragmentInfoMutex_);

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

void InsertOrderFragmenter::insertChunksIntoFragment(
    const InsertChunks& insert_chunks,
    const std::optional<int> delete_column_id,
    FragmentInfo* current_fragment,
    const size_t num_rows_to_insert,
    size_t& num_rows_inserted,
    size_t& num_rows_left,
    std::vector<size_t>& valid_row_indices,
    const size_t start_fragment) {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(fragmentInfoMutex_);
  // for each column, append the data in the appropriate insert buffer
  auto insert_row_indices = valid_row_indices;
  CHECK_GE(insert_row_indices.size(), num_rows_to_insert);
  insert_row_indices.erase(insert_row_indices.begin() + num_rows_to_insert,
                           insert_row_indices.end());
  CHECK_EQ(insert_row_indices.size(), num_rows_to_insert);
  for (auto& [column_id, chunk] : insert_chunks.chunks) {
    auto col_map_it = columnMap_.find(column_id);
    CHECK(col_map_it != columnMap_.end());
    current_fragment->shadowChunkMetadataMap[column_id] =
        col_map_it->second.appendEncodedDataAtIndices(*chunk, insert_row_indices);
    auto var_len_col_info_it = varLenColInfo_.find(column_id);
    if (var_len_col_info_it != varLenColInfo_.end()) {
      var_len_col_info_it->second = col_map_it->second.getBuffer()->size();
      CHECK_LE(var_len_col_info_it->second, maxChunkSize_);
    }
  }
  if (hasMaterializedRowId_) {
    size_t start_id = maxFragmentRows_ * current_fragment->fragmentId +
                      current_fragment->shadowNumTuples;
    std::vector<int64_t> row_id_data(num_rows_to_insert);
    for (size_t i = 0; i < num_rows_to_insert; ++i) {
      row_id_data[i] = i + start_id;
    }
    DataBlockPtr row_id_block;
    row_id_block.numbersPtr = reinterpret_cast<int8_t*>(row_id_data.data());
    auto col_map_it = columnMap_.find(rowIdColId_);
    CHECK(col_map_it != columnMap_.end());
    current_fragment->shadowChunkMetadataMap[rowIdColId_] = col_map_it->second.appendData(
        row_id_block, num_rows_to_insert, num_rows_inserted);
  }

  if (delete_column_id) {  // has delete column
    std::vector<int8_t> delete_data(num_rows_to_insert, false);
    DataBlockPtr delete_block;
    delete_block.numbersPtr = reinterpret_cast<int8_t*>(delete_data.data());
    auto col_map_it = columnMap_.find(*delete_column_id);
    CHECK(col_map_it != columnMap_.end());
    current_fragment->shadowChunkMetadataMap[*delete_column_id] =
        col_map_it->second.appendData(
            delete_block, num_rows_to_insert, num_rows_inserted);
  }

  current_fragment->shadowNumTuples =
      fragmentInfoVec_.back()->getPhysicalNumTuples() + num_rows_to_insert;
  num_rows_left -= num_rows_to_insert;
  num_rows_inserted += num_rows_to_insert;
  for (auto part_it = fragmentInfoVec_.begin() + start_fragment;
       part_it != fragmentInfoVec_.end();
       ++part_it) {
    auto fragment_ptr = part_it->get();
    fragment_ptr->setPhysicalNumTuples(fragment_ptr->shadowNumTuples);
    fragment_ptr->setChunkMetadataMap(fragment_ptr->shadowChunkMetadataMap);
  }

  // truncate the first `num_rows_to_insert` rows in `valid_row_indices`
  valid_row_indices.erase(valid_row_indices.begin(),
                          valid_row_indices.begin() + num_rows_to_insert);
}

void InsertOrderFragmenter::insertChunksImpl(const InsertChunks& insert_chunks) {
  std::optional<int> delete_column_id{std::nullopt};
  for (const auto& cit : columnMap_) {
    if (cit.second.getColumnDesc()->isDeletedCol) {
      delete_column_id = cit.second.getColumnDesc()->columnId;
    }
  }

  // verify that all chunks to be inserted have same number of rows, otherwise the input
  // data is malformed
  std::optional<size_t> num_rows{std::nullopt};
  for (const auto& [column_id, chunk] : insert_chunks.chunks) {
    auto buffer = chunk->getBuffer();
    CHECK(buffer);
    CHECK(buffer->hasEncoder());
    if (!num_rows.has_value()) {
      num_rows = buffer->getEncoder()->getNumElems();
    } else {
      CHECK_EQ(num_rows.value(), buffer->getEncoder()->getNumElems());
    }
  }

  auto valid_row_indices = insert_chunks.valid_row_indices;
  size_t num_rows_left = valid_row_indices.size();
  size_t num_rows_inserted = 0;

  if (num_rows_left == 0) {
    return;
  }

  FragmentInfo* current_fragment{nullptr};

  // Access to fragmentInfoVec_ is protected as we are under the insertMutex_ lock but it
  // feels fragile
  if (fragmentInfoVec_.empty()) {  // if no fragments exist for table
    current_fragment = createNewFragment(defaultInsertLevel_);
  } else {
    current_fragment = fragmentInfoVec_.back().get();
  }
  CHECK(current_fragment);

  size_t start_fragment = fragmentInfoVec_.size() - 1;

  while (num_rows_left > 0) {  // may have to create multiple fragments for bulk insert
    // loop until done inserting all rows
    CHECK_LE(current_fragment->shadowNumTuples, maxFragmentRows_);
    size_t rows_left_in_current_fragment =
        maxFragmentRows_ - current_fragment->shadowNumTuples;
    size_t num_rows_to_insert = get_num_rows_to_insert(rows_left_in_current_fragment,
                                                       num_rows_left,
                                                       num_rows_inserted,
                                                       varLenColInfo_,
                                                       maxChunkSize_,
                                                       insert_chunks,
                                                       columnMap_,
                                                       valid_row_indices);

    if (rows_left_in_current_fragment == 0 || num_rows_to_insert == 0) {
      current_fragment = createNewFragment(defaultInsertLevel_);
      if (num_rows_inserted == 0) {
        start_fragment++;
      }
      rows_left_in_current_fragment = maxFragmentRows_;
      for (auto& varLenColInfoIt : varLenColInfo_) {
        varLenColInfoIt.second = 0;  // reset byte counter
      }
      num_rows_to_insert = get_num_rows_to_insert(rows_left_in_current_fragment,
                                                  num_rows_left,
                                                  num_rows_inserted,
                                                  varLenColInfo_,
                                                  maxChunkSize_,
                                                  insert_chunks,
                                                  columnMap_,
                                                  valid_row_indices);
    }

    CHECK_GT(num_rows_to_insert, size_t(0));  // would put us into an endless loop as we'd
                                              // never be able to insert anything

    insertChunksIntoFragment(insert_chunks,
                             delete_column_id,
                             current_fragment,
                             num_rows_to_insert,
                             num_rows_inserted,
                             num_rows_left,
                             valid_row_indices,
                             start_fragment);
  }
  numTuples_ += *num_rows;
  dropFragmentsToSizeNoInsertLock(maxRows_);
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

  // Access to fragmentInfoVec_ is protected as we are under the insertMutex_ lock but it
  // feels fragile
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
      heavyai::unique_lock<heavyai::shared_mutex> writeLock(fragmentInfoMutex_);
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
  dropFragmentsToSizeNoInsertLock(maxRows_);
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
    auto& chunk = colMapIt->second;
    if (memoryLevel == Data_Namespace::MemoryLevel::CPU_LEVEL) {
      /* At the end of this function chunks from the previous fragment become 'rolled
       * off', temporaray tables will lose reference to any 'rolled off' chunks and are
       * not able to unpin these chunks. Keep reference to 'rolled off' chunks and unpin
       * at ~InsertOrderFragmenter, chunks wrapped by unique_ptr to avoid extraneous
       * ~Chunk calls with temporary chunks.*/
      tracked_in_memory_chunks_.emplace_back(std::make_unique<Chunk_NS::Chunk>(chunk));
    }
    ChunkKey chunkKey = chunkKeyPrefix_;
    chunkKey.push_back(chunk.getColumnDesc()->columnId);
    chunkKey.push_back(maxFragmentId_);
    chunk.createChunkBuffer(dataMgr_,
                            chunkKey,
                            memoryLevel,
                            newFragmentInfo->deviceIds[static_cast<int>(memoryLevel)],
                            pageSize_);
    chunk.initEncoder();
  }

  heavyai::lock_guard<heavyai::shared_mutex> writeLock(fragmentInfoMutex_);
  fragmentInfoVec_.push_back(std::move(newFragmentInfo));
  return fragmentInfoVec_.back().get();
}

size_t InsertOrderFragmenter::getNumFragments() {
  heavyai::shared_lock<heavyai::shared_mutex> readLock(fragmentInfoMutex_);
  return fragmentInfoVec_.size();
}

TableInfo InsertOrderFragmenter::getFragmentsForQuery() {
  heavyai::shared_lock<heavyai::shared_mutex> readLock(fragmentInfoMutex_);
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

void InsertOrderFragmenter::resetSizesFromFragments() {
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(fragmentInfoMutex_);
  numTuples_ = 0;
  for (const auto& fragment_info : fragmentInfoVec_) {
    numTuples_ += fragment_info->getPhysicalNumTuples();
  }
  setLastFragmentVarLenColumnSizes();
}

void InsertOrderFragmenter::alterColumnGeoType(
    const std::list<
        std::pair<const ColumnDescriptor*, std::list<const ColumnDescriptor*>>>&
        src_dst_column_pairs) {
  CHECK(defaultInsertLevel_ == Data_Namespace::MemoryLevel::DISK_LEVEL &&
        !uses_foreign_storage_)
      << "`alterColumnTypeTransactional` only supported for regular tables";
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(fragmentInfoMutex_);

  for (const auto& [src_cd, dst_columns] : src_dst_column_pairs) {
    auto logical_geo_column = *dst_columns.begin();
    CHECK(logical_geo_column->columnType.is_geometry());

    columnMap_.erase(
        src_cd->columnId);  // NOTE: Necessary to prevent unpinning issues with these
                            // chunks when fragmenter is destroyed later.

    for (const auto& fragment_info : fragmentInfoVec_) {
      int device_id = fragment_info->deviceIds[static_cast<int>(defaultInsertLevel_)];
      auto num_elements = fragment_info->chunkMetadataMap[src_cd->columnId]->numElements;

      CHECK_GE(dst_columns.size(), 1UL);

      std::list<const ColumnDescriptor*> columns = dst_columns;
      GeoAlterColumnContext alter_column_context{device_id,
                                                 chunkKeyPrefix_,
                                                 fragment_info.get(),
                                                 src_cd,
                                                 *dst_columns.begin(),
                                                 num_elements,
                                                 dataMgr_,
                                                 catalog_,
                                                 columnMap_,
                                                 columns};

      alter_column_context.readSourceData();

      alter_column_context.createScratchBuffers();

      ScopeGuard delete_temp_chunk = [&] { alter_column_context.deleteScratchBuffers(); };

      const bool geo_validate_geometry = false;
      alter_column_context.encodeData(geo_validate_geometry);

      alter_column_context.putBuffersToDisk();
    }
  }
}

void InsertOrderFragmenter::alterNonGeoColumnType(
    const std::list<const ColumnDescriptor*>& columns) {
  CHECK(defaultInsertLevel_ == Data_Namespace::MemoryLevel::DISK_LEVEL &&
        !uses_foreign_storage_)
      << "`alterColumnTypeTransactional` only supported for regular tables";

  heavyai::unique_lock<heavyai::shared_mutex> write_lock(fragmentInfoMutex_);

  for (const auto dst_cd : columns) {
    auto col_it = columnMap_.find(dst_cd->columnId);
    CHECK(col_it != columnMap_.end());

    auto src_cd = col_it->second.getColumnDesc();
    CHECK_EQ(col_it->first, src_cd->columnId);

    if (ddl_utils::alter_column_utils::compare_column_descriptors(src_cd, dst_cd)
            .sql_types_match) {
      continue;
    }

    for (const auto& fragment_info : fragmentInfoVec_) {
      int device_id = fragment_info->deviceIds[static_cast<int>(defaultInsertLevel_)];
      auto num_elements = fragment_info->chunkMetadataMap[src_cd->columnId]->numElements;

      NonGeoAlterColumnContext alter_column_context{device_id,
                                                    chunkKeyPrefix_,
                                                    fragment_info.get(),
                                                    src_cd,
                                                    dst_cd,
                                                    num_elements,
                                                    dataMgr_,
                                                    catalog_,
                                                    columnMap_};

      alter_column_context.readSourceData();

      alter_column_context.createScratchBuffers();

      ScopeGuard delete_temp_chunk = [&] { alter_column_context.deleteScratchBuffers(); };

      alter_column_context.reencodeData();

      alter_column_context.putBuffersToDisk();
    }
  }
}

void InsertOrderFragmenter::setLastFragmentVarLenColumnSizes() {
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
}  // namespace Fragmenter_Namespace
