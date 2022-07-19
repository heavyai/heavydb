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

#include <algorithm>
#include <numeric>
#include <vector>

#include "../Shared/shard_key.h"
#include "Geospatial/Types.h"
#include "InsertDataLoader.h"
#include "TargetValueConvertersFactories.h"

namespace Fragmenter_Namespace {

struct ShardDataOwner {
  std::vector<std::vector<uint8_t>> rawData;
  std::vector<std::vector<std::string>> stringData;
  std::vector<std::vector<ArrayDatum>> arrayData;
};

template <typename SRC>
std::vector<std::vector<size_t>> compute_row_indices_of_shards(
    size_t shard_count,
    size_t leaf_count,
    size_t row_count,
    SRC* src,
    bool duplicated_key_value) {
  const auto n_shard_tables = shard_count * leaf_count;
  std::vector<std::vector<size_t>> row_indices_of_shards(n_shard_tables);
  if (!duplicated_key_value) {
    for (size_t row = 0; row < row_count; row++) {
      // expecting unsigned data
      // thus, no need for double remainder
      auto shard_id = (std::is_unsigned<SRC>::value)
                          ? src[row] % n_shard_tables
                          : SHARD_FOR_KEY(src[row], n_shard_tables);
      row_indices_of_shards[shard_id].push_back(row);
    }
  } else {
    auto shard_id = (std::is_unsigned<SRC>::value)
                        ? src[0] % n_shard_tables
                        : SHARD_FOR_KEY(src[0], n_shard_tables);
    row_indices_of_shards[shard_id].reserve(row_count);
    for (size_t row = 0; row < row_count; row++) {
      row_indices_of_shards[shard_id].push_back(row);
    }
  }

  return row_indices_of_shards;
}

template <typename T>
size_t indexOf(std::vector<T>& vec, T val) {
  typename std::vector<T>::iterator it = std::find(vec.begin(), vec.end(), val);
  CHECK(it != vec.end());
  return std::distance(vec.begin(), it);
}

bool isStringVectorData(const ColumnDescriptor* cd) {
  return (cd->columnType.is_geometry()) ||
         (cd->columnType.is_string() &&
          cd->columnType.get_compression() == kENCODING_NONE);
}

bool isDatumVectorData(const ColumnDescriptor* cd) {
  return cd->columnType.is_array();
}

size_t size_of_raw_column(const Catalog_Namespace::Catalog& cat,
                          const ColumnDescriptor* cd,
                          const bool get_logical_size = true) {
  switch (cd->columnType.get_type()) {
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
    case kARRAY:
      throw std::runtime_error("geo and array columns have variable length elements");
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kFLOAT:
    case kDOUBLE:
    case kTIMESTAMP:
    case kTIME:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kDATE:
      return get_logical_size ? cd->columnType.get_logical_size()
                              : cd->columnType.get_size();
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (cd->columnType.get_compression() == kENCODING_NONE) {
        throw std::runtime_error(
            "non encoded string columns have variable length elements");
      }
      return cd->columnType.get_size();
    default:
      throw std::runtime_error("not supported column type: " + cd->columnName + " (" +
                               cd->columnType.get_type_name() + ")");
  }
}

std::vector<std::vector<size_t>> compute_row_indices_of_shards(
    const Catalog_Namespace::Catalog& cat,
    size_t leaf_count,
    const InsertChunks& insert_chunks) {
  const auto* td = cat.getMetadataForTable(insert_chunks.table_id);
  const auto* shard_cd = cat.getShardColumnMetadataForTable(td);
  auto find_it = insert_chunks.chunks.find(shard_cd->columnId);
  CHECK(find_it != insert_chunks.chunks.end());
  Chunk_NS::Chunk& shard_chunk = *find_it->second;
  auto row_count = shard_chunk.getBuffer()->getEncoder()->getNumElems();
  auto shard_count = td->nShards;

  CHECK(!isStringVectorData(shard_cd));
  CHECK(!isDatumVectorData(shard_cd));

  auto memory_ptr = shard_chunk.getBuffer()->getMemoryPtr();
  CHECK(memory_ptr);
  switch (size_of_raw_column(cat, shard_cd, false)) {
    case 1:
      return compute_row_indices_of_shards(shard_count,
                                           leaf_count,
                                           row_count,
                                           reinterpret_cast<uint8_t*>(memory_ptr),
                                           false);
    case 2:
      return compute_row_indices_of_shards(shard_count,
                                           leaf_count,
                                           row_count,
                                           reinterpret_cast<uint16_t*>(memory_ptr),
                                           false);
    case 4:
      return compute_row_indices_of_shards(shard_count,
                                           leaf_count,
                                           row_count,
                                           reinterpret_cast<uint32_t*>(memory_ptr),
                                           false);
    case 8:
      return compute_row_indices_of_shards(shard_count,
                                           leaf_count,
                                           row_count,
                                           reinterpret_cast<uint64_t*>(memory_ptr),
                                           false);
    default:
      UNREACHABLE() << "unexpected data element size of column";
  }
  return {};
}

std::vector<std::vector<size_t>> computeRowIndicesOfShards(
    const Catalog_Namespace::Catalog& cat,
    size_t leafCount,
    InsertData& insert_data) {
  const auto* td = cat.getMetadataForTable(insert_data.tableId);
  const auto* shard_cd = cat.getShardColumnMetadataForTable(td);
  auto shardDataBlockIndex = indexOf(insert_data.columnIds, shard_cd->columnId);
  DataBlockPtr& shardDataBlock = insert_data.data[shardDataBlockIndex];
  auto rowCount = insert_data.numRows;
  auto shardCount = td->nShards;

  CHECK(!isStringVectorData(shard_cd));
  CHECK(!isDatumVectorData(shard_cd));

  CHECK(insert_data.is_default.size() == insert_data.columnIds.size());
  bool is_default = insert_data.is_default[shardDataBlockIndex];
  switch (size_of_raw_column(cat, shard_cd)) {
    case 1:
      return compute_row_indices_of_shards(
          shardCount,
          leafCount,
          rowCount,
          reinterpret_cast<uint8_t*>(shardDataBlock.numbersPtr),
          is_default);
    case 2:
      return compute_row_indices_of_shards(
          shardCount,
          leafCount,
          rowCount,
          reinterpret_cast<uint16_t*>(shardDataBlock.numbersPtr),
          is_default);
    case 4:
      return compute_row_indices_of_shards(
          shardCount,
          leafCount,
          rowCount,
          reinterpret_cast<uint32_t*>(shardDataBlock.numbersPtr),
          is_default);
    case 8:
      return compute_row_indices_of_shards(
          shardCount,
          leafCount,
          rowCount,
          reinterpret_cast<uint64_t*>(shardDataBlock.numbersPtr),
          is_default);
  }
  throw std::runtime_error("Unexpected data block element size");
}

template <typename T>
void copyColumnDataOfShard(const std::vector<size_t>& rowIndices, T* src, T* dst) {
  for (size_t row = 0; row < rowIndices.size(); row++) {
    auto srcRowIndex = rowIndices[row];
    dst[row] = src[srcRowIndex];
  }
}

struct BlockWithColumnId {
  int columnId;
  DataBlockPtr block;
  bool is_default;
};

BlockWithColumnId copyColumnDataOfShard(const Catalog_Namespace::Catalog& cat,
                                        ShardDataOwner& dataOwner,
                                        const std::vector<size_t>& rowIndices,
                                        const ColumnDescriptor* pCol,
                                        size_t columnIndex,
                                        DataBlockPtr dataBlock,
                                        bool is_default) {
  DataBlockPtr ret;
  std::vector<size_t> single_row_idx({0ul});
  const std::vector<size_t>& rows = is_default ? single_row_idx : rowIndices;
  if (isStringVectorData(pCol)) {
    auto& data = dataOwner.stringData[columnIndex];
    data.resize(rows.size());
    copyColumnDataOfShard(rows, &(*(dataBlock.stringsPtr))[0], &data[0]);
    ret.stringsPtr = &data;

  } else if (isDatumVectorData(pCol)) {
    auto& data = dataOwner.arrayData[columnIndex];
    data.resize(rows.size());
    copyColumnDataOfShard(rows, &(*(dataBlock.arraysPtr))[0], &data[0]);
    ret.arraysPtr = &data;

  } else {
    auto rawArrayElementSize = size_of_raw_column(cat, pCol);
    auto& data = dataOwner.rawData[columnIndex];
    data.resize(rows.size() * rawArrayElementSize);

    switch (rawArrayElementSize) {
      case 1: {
        copyColumnDataOfShard(rows,
                              reinterpret_cast<uint8_t*>(dataBlock.numbersPtr),
                              reinterpret_cast<uint8_t*>(&data[0]));
        break;
      }
      case 2: {
        copyColumnDataOfShard(rows,
                              reinterpret_cast<uint16_t*>(dataBlock.numbersPtr),
                              reinterpret_cast<uint16_t*>(&data[0]));
        break;
      }
      case 4: {
        copyColumnDataOfShard(rows,
                              reinterpret_cast<uint32_t*>(dataBlock.numbersPtr),
                              reinterpret_cast<uint32_t*>(&data[0]));
        break;
      }
      case 8: {
        copyColumnDataOfShard(rows,
                              reinterpret_cast<uint64_t*>(dataBlock.numbersPtr),
                              reinterpret_cast<uint64_t*>(&data[0]));
        break;
      }
      default:
        throw std::runtime_error("Unexpected data block element size");
    }

    ret.numbersPtr = reinterpret_cast<int8_t*>(&data[0]);
  }

  return {pCol->columnId, ret, is_default};
}

std::pair<std::list<std::unique_ptr<foreign_storage::ForeignStorageBuffer>>, InsertChunks>
copy_data_of_shard(const Catalog_Namespace::Catalog& cat,
                   const InsertChunks& insert_chunks,
                   int shardTableIndex,
                   const std::vector<size_t>& rowIndices) {
  const auto* table = cat.getMetadataForTable(insert_chunks.table_id);
  const auto* physical_table = cat.getPhysicalTablesDescriptors(table)[shardTableIndex];

  InsertChunks insert_chunks_for_shard{
      physical_table->tableId, insert_chunks.db_id, {}, {}};

  std::list<std::unique_ptr<foreign_storage::ForeignStorageBuffer>> buffers;

  for (const auto& [column_id, chunk] : insert_chunks.chunks) {
    auto column = chunk->getColumnDesc();
    insert_chunks_for_shard.chunks[column_id] = std::make_shared<Chunk_NS::Chunk>(column);
    auto& chunk_for_shard = *insert_chunks_for_shard.chunks[column_id];
    chunk_for_shard.setBuffer(
        buffers.emplace_back(std::make_unique<foreign_storage::ForeignStorageBuffer>())
            .get());
    if (column->columnType.is_varlen_indeed()) {  // requires an index buffer
      chunk_for_shard.setIndexBuffer(
          buffers.emplace_back(std::make_unique<foreign_storage::ForeignStorageBuffer>())
              .get());
    }
    chunk_for_shard.initEncoder();
    chunk_for_shard.appendEncodedDataAtIndices(*chunk, rowIndices);
    CHECK_EQ(chunk_for_shard.getBuffer()->getEncoder()->getNumElems(), rowIndices.size());
  }

  // mark which row indices are valid for import
  auto row_count = rowIndices.size();
  insert_chunks_for_shard.valid_row_indices.reserve(row_count);
  for (size_t irow = 0; irow < row_count; ++irow) {
    auto row_index = rowIndices[irow];
    if (std::binary_search(insert_chunks.valid_row_indices.begin(),
                           insert_chunks.valid_row_indices.end(),
                           row_index)) {
      insert_chunks_for_shard.valid_row_indices.emplace_back(irow);
    }
  }

  return {std::move(buffers), insert_chunks_for_shard};
}

InsertData copyDataOfShard(const Catalog_Namespace::Catalog& cat,
                           ShardDataOwner& dataOwner,
                           InsertData& insert_data,
                           int shardTableIndex,
                           const std::vector<size_t>& rowIndices) {
  const auto* td = cat.getMetadataForTable(insert_data.tableId);
  const auto* ptd = cat.getPhysicalTablesDescriptors(td)[shardTableIndex];

  InsertData shardData;
  shardData.databaseId = insert_data.databaseId;
  shardData.tableId = ptd->tableId;
  shardData.numRows = rowIndices.size();

  std::vector<const ColumnDescriptor*> pCols;
  std::vector<int> lCols;

  {
    auto logicalColumns = cat.getAllColumnMetadataForTable(td->tableId, true, true, true);
    for (const auto& cd : logicalColumns) {
      lCols.push_back(cd->columnId);
    }

    auto physicalColumns =
        cat.getAllColumnMetadataForTable(ptd->tableId, true, true, true);
    for (const auto& cd : physicalColumns) {
      pCols.push_back(cd);
    }
  }

  for (size_t col = 0; col < insert_data.columnIds.size(); col++) {
    dataOwner.arrayData.emplace_back();
    dataOwner.rawData.emplace_back();
    dataOwner.stringData.emplace_back();
  }

  auto copycat = [&cat, &dataOwner, &rowIndices, &lCols, &pCols, &insert_data](int col) {
    const auto lColId = insert_data.columnIds[col];
    const auto pCol = pCols[indexOf(lCols, lColId)];
    return copyColumnDataOfShard(cat,
                                 dataOwner,
                                 rowIndices,
                                 pCol,
                                 col,
                                 insert_data.data[col],
                                 insert_data.is_default[col]);
  };

  std::vector<std::future<BlockWithColumnId>> worker_threads;
  for (size_t col = 0; col < insert_data.columnIds.size(); col++) {
    worker_threads.push_back(std::async(std::launch::async, copycat, col));
  }

  for (auto& child : worker_threads) {
    child.wait();
  }

  for (auto& child : worker_threads) {
    auto shardColumnData = child.get();
    shardData.columnIds.push_back(shardColumnData.columnId);
    shardData.data.push_back(shardColumnData.block);
    shardData.is_default.push_back(shardColumnData.is_default);
  }

  return shardData;
}

size_t InsertDataLoader::moveToNextLeaf() {
  std::unique_lock current_leaf_index_lock(current_leaf_index_mutex_);
  size_t starting_leaf_index = current_leaf_index_;
  current_leaf_index_++;
  if (current_leaf_index_ >= leaf_count_) {
    current_leaf_index_ = 0;
  }
  return starting_leaf_index;
}

void InsertDataLoader::insertChunks(const Catalog_Namespace::SessionInfo& session_info,
                                    const InsertChunks& insert_chunks) {
  const auto& cat = session_info.getCatalog();
  const auto* td = cat.getMetadataForTable(insert_chunks.table_id);

  CHECK(td);
  if (td->nShards == 0) {
    connector_.insertChunksToLeaf(session_info, moveToNextLeaf(), insert_chunks);
  } else {
    // we have a sharded target table, start spreading to physical tables
    auto row_indices_of_shards =
        compute_row_indices_of_shards(cat, connector_.leafCount(), insert_chunks);

    auto insert_shard_data =
        [this, &session_info, &insert_chunks, &cat, &td, &row_indices_of_shards](
            size_t shardId) {
          const auto shard_tables = cat.getPhysicalTablesDescriptors(td);
          auto stard_table_idx = shardId % td->nShards;
          auto shard_leaf_idx = shardId / td->nShards;

          const auto& row_indices_of_shard = row_indices_of_shards[shardId];

          auto [buffers, shard_insert_chunks] = copy_data_of_shard(
              cat, insert_chunks, stard_table_idx, row_indices_of_shard);
          connector_.insertChunksToLeaf(
              session_info, shard_leaf_idx, shard_insert_chunks);
        };

    std::vector<std::future<void>> worker_threads;
    for (size_t shard_id = 0; shard_id < row_indices_of_shards.size(); shard_id++) {
      if (row_indices_of_shards[shard_id].size() > 0) {
        worker_threads.push_back(
            std::async(std::launch::async, insert_shard_data, shard_id));
      }
    }
    for (auto& child : worker_threads) {
      child.wait();
    }
    for (auto& child : worker_threads) {
      child.get();
    }
  }
}

void InsertDataLoader::insertData(const Catalog_Namespace::SessionInfo& session_info,
                                  InsertData& insert_data) {
  const auto& cat = session_info.getCatalog();
  const auto* td = cat.getMetadataForTable(insert_data.tableId);

  CHECK(td);
  if (td->nShards == 0) {
    connector_.insertDataToLeaf(session_info, moveToNextLeaf(), insert_data);
  } else {
    // we have a sharded target table, start spreading to physical tables
    auto rowIndicesOfShards =
        computeRowIndicesOfShards(cat, connector_.leafCount(), insert_data);

    auto insertShardData =
        [this, &session_info, &insert_data, &cat, &td, &rowIndicesOfShards](
            size_t shardId) {
          const auto shard_tables = cat.getPhysicalTablesDescriptors(td);
          auto stardTableIdx = shardId % td->nShards;
          auto shardLeafIdx = shardId / td->nShards;

          const auto& rowIndicesOfShard = rowIndicesOfShards[shardId];
          ShardDataOwner shardDataOwner;

          InsertData shardData = copyDataOfShard(
              cat, shardDataOwner, insert_data, stardTableIdx, rowIndicesOfShard);
          CHECK(shardData.numRows > 0);
          connector_.insertDataToLeaf(session_info, shardLeafIdx, shardData);
        };

    std::vector<std::future<void>> worker_threads;
    for (size_t shardId = 0; shardId < rowIndicesOfShards.size(); shardId++) {
      if (rowIndicesOfShards[shardId].size() > 0) {
        worker_threads.push_back(
            std::async(std::launch::async, insertShardData, shardId));
      }
    }
    for (auto& child : worker_threads) {
      child.wait();
    }
    for (auto& child : worker_threads) {
      child.get();
    }
  }
}

void LocalInsertConnector::insertChunksToLeaf(
    const Catalog_Namespace::SessionInfo& session,
    const size_t leaf_idx,
    const Fragmenter_Namespace::InsertChunks& insert_chunks) {
  CHECK(leaf_idx == 0);
  auto& catalog = session.getCatalog();
  auto created_td = catalog.getMetadataForTable(insert_chunks.table_id);
  created_td->fragmenter->insertChunksNoCheckpoint(insert_chunks);
}

void LocalInsertConnector::insertDataToLeaf(const Catalog_Namespace::SessionInfo& session,
                                            const size_t leaf_idx,
                                            InsertData& insert_data) {
  CHECK(leaf_idx == 0);
  auto& catalog = session.getCatalog();
  auto created_td = catalog.getMetadataForTable(insert_data.tableId);
  created_td->fragmenter->insertDataNoCheckpoint(insert_data);
}

void LocalInsertConnector::checkpoint(const Catalog_Namespace::SessionInfo& session,
                                      int table_id) {
  auto& catalog = session.getCatalog();
  catalog.checkpointWithAutoRollback(table_id);
}

void LocalInsertConnector::rollback(const Catalog_Namespace::SessionInfo& session,
                                    int table_id) {
  auto& catalog = session.getCatalog();
  auto db_id = catalog.getDatabaseId();
  auto table_epochs = catalog.getTableEpochs(db_id, table_id);
  catalog.setTableEpochs(db_id, table_epochs);
}

}  // namespace Fragmenter_Namespace
