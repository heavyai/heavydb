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
#include <algorithm>
#include <boost/variant.hpp>
#include <boost/variant/get.hpp>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "Catalog/Catalog.h"
#include "DataMgr/DataMgr.h"
#include "DataMgr/FixedLengthArrayNoneEncoder.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "QueryEngine/TargetValue.h"
#include "Shared/ConfigResolve.h"
#include "Shared/DateConversions.h"
#include "Shared/TypedDataAccessors.h"
#include "Shared/thread_count.h"
#include "TargetValueConvertersFactories.h"

namespace Fragmenter_Namespace {

inline void wait_cleanup_threads(std::vector<std::future<void>>& threads) {
  for (auto& t : threads) {
    t.get();
  }
  threads.clear();
}

FragmentInfo& InsertOrderFragmenter::getFragmentInfoFromId(const int fragment_id) {
  auto fragment_it = std::find_if(
      fragmentInfoVec_.begin(), fragmentInfoVec_.end(), [=](const auto& f) -> bool {
        return f.fragmentId == fragment_id;
      });
  CHECK(fragment_it != fragmentInfoVec_.end());
  return *fragment_it;
}

inline bool is_integral(const SQLTypeInfo& t) {
  return t.is_integer() || t.is_boolean() || t.is_time() || t.is_timeinterval();
}

bool FragmentInfo::unconditionalVacuum_{false};

void InsertOrderFragmenter::updateColumn(const Catalog_Namespace::Catalog* catalog,
                                         const std::string& tab_name,
                                         const std::string& col_name,
                                         const int fragment_id,
                                         const std::vector<uint64_t>& frag_offsets,
                                         const std::vector<ScalarTargetValue>& rhs_values,
                                         const SQLTypeInfo& rhs_type,
                                         const Data_Namespace::MemoryLevel memory_level,
                                         UpdelRoll& updel_roll) {
  const auto td = catalog->getMetadataForTable(tab_name);
  CHECK(td);
  const auto cd = catalog->getMetadataForColumn(td->tableId, col_name);
  CHECK(cd);
  td->fragmenter->updateColumn(catalog,
                               td,
                               cd,
                               fragment_id,
                               frag_offsets,
                               rhs_values,
                               rhs_type,
                               memory_level,
                               updel_roll);
}

void InsertOrderFragmenter::updateColumn(const Catalog_Namespace::Catalog* catalog,
                                         const TableDescriptor* td,
                                         const ColumnDescriptor* cd,
                                         const int fragment_id,
                                         const std::vector<uint64_t>& frag_offsets,
                                         const ScalarTargetValue& rhs_value,
                                         const SQLTypeInfo& rhs_type,
                                         const Data_Namespace::MemoryLevel memory_level,
                                         UpdelRoll& updel_roll) {
  updateColumn(catalog,
               td,
               cd,
               fragment_id,
               frag_offsets,
               std::vector<ScalarTargetValue>(1, rhs_value),
               rhs_type,
               memory_level,
               updel_roll);
}

static int get_chunks(const Catalog_Namespace::Catalog* catalog,
                      const TableDescriptor* td,
                      const FragmentInfo& fragment,
                      const Data_Namespace::MemoryLevel memory_level,
                      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks) {
  for (int cid = 1, nc = 0; nc < td->nColumns; ++cid) {
    if (const auto cd = catalog->getMetadataForColumn(td->tableId, cid)) {
      ++nc;
      if (!cd->isVirtualCol) {
        auto chunk_meta_it = fragment.getChunkMetadataMapPhysical().find(cid);
        CHECK(chunk_meta_it != fragment.getChunkMetadataMapPhysical().end());
        ChunkKey chunk_key{
            catalog->getCurrentDB().dbId, td->tableId, cid, fragment.fragmentId};
        auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                               &catalog->getDataMgr(),
                                               chunk_key,
                                               memory_level,
                                               0,
                                               chunk_meta_it->second.numBytes,
                                               chunk_meta_it->second.numElements);
        chunks.push_back(chunk);
      }
    }
  }
  return chunks.size();
}

struct ChunkToInsertDataConverter {
 public:
  virtual ~ChunkToInsertDataConverter() {}

  virtual void convertToColumnarFormat(size_t row, size_t indexInFragment) = 0;

  virtual void addDataBlocksToInsertData(
      Fragmenter_Namespace::InsertData& insertData) = 0;
};

template <typename BUFFER_DATA_TYPE, typename INSERT_DATA_TYPE>
struct ScalarChunkConverter : public ChunkToInsertDataConverter {
  using ColumnDataPtr =
      std::unique_ptr<INSERT_DATA_TYPE, CheckedMallocDeleter<INSERT_DATA_TYPE>>;

  const Chunk_NS::Chunk* chunk_;
  ColumnDataPtr column_data_;
  const ColumnDescriptor* column_descriptor_;
  const BUFFER_DATA_TYPE* data_buffer_addr_;

  ScalarChunkConverter(const size_t num_rows, const Chunk_NS::Chunk* chunk)
      : chunk_(chunk), column_descriptor_(chunk->get_column_desc()) {
    column_data_ = ColumnDataPtr(reinterpret_cast<INSERT_DATA_TYPE*>(
        checked_malloc(num_rows * sizeof(INSERT_DATA_TYPE))));
    data_buffer_addr_ = (BUFFER_DATA_TYPE*)chunk->get_buffer()->getMemoryPtr();
  }

  virtual ~ScalarChunkConverter() {}

  virtual void convertToColumnarFormat(size_t row, size_t indexInFragment) {
    auto buffer_value = data_buffer_addr_[indexInFragment];
    auto insert_value = static_cast<INSERT_DATA_TYPE>(buffer_value);
    column_data_.get()[row] = insert_value;
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    DataBlockPtr dataBlock;
    dataBlock.numbersPtr = reinterpret_cast<int8_t*>(column_data_.get());
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

struct FixedLenArrayChunkConverter : public ChunkToInsertDataConverter {
  const Chunk_NS::Chunk* chunk_;
  const ColumnDescriptor* column_descriptor_;

  std::unique_ptr<std::vector<ArrayDatum>> column_data_;
  int8_t* data_buffer_addr_;
  size_t fixed_array_length_;

  FixedLenArrayChunkConverter(const size_t num_rows, const Chunk_NS::Chunk* chunk)
      : chunk_(chunk), column_descriptor_(chunk->get_column_desc()) {
    column_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    data_buffer_addr_ = chunk->get_buffer()->getMemoryPtr();
    fixed_array_length_ = chunk->get_column_desc()->columnType.get_size();
  }

  virtual ~FixedLenArrayChunkConverter() {}

  virtual void convertToColumnarFormat(size_t row, size_t indexInFragment) {
    auto src_value_ptr = data_buffer_addr_ + (indexInFragment * fixed_array_length_);
    (*column_data_)[row] =
        ArrayDatum(fixed_array_length_, (int8_t*)src_value_ptr, DoNothingDeleter());
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    DataBlockPtr dataBlock;
    dataBlock.arraysPtr = column_data_.get();
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

struct ArrayChunkConverter : public FixedLenArrayChunkConverter {
  StringOffsetT* index_buffer_addr_;

  ArrayChunkConverter(const size_t num_rows, const Chunk_NS::Chunk* chunk)
      : FixedLenArrayChunkConverter(num_rows, chunk) {
    index_buffer_addr_ =
        (StringOffsetT*)(chunk->get_index_buf() ? chunk->get_index_buf()->getMemoryPtr()
                                                : nullptr);
  }

  virtual ~ArrayChunkConverter() {}

  virtual void convertToColumnarFormat(size_t row, size_t indexInFragment) {
    size_t src_value_size =
        index_buffer_addr_[indexInFragment + 1] - index_buffer_addr_[indexInFragment];
    auto src_value_ptr = data_buffer_addr_ + index_buffer_addr_[indexInFragment];
    (*column_data_)[row] =
        ArrayDatum(src_value_size, (int8_t*)src_value_ptr, DoNothingDeleter());
  }
};

struct StringChunkConverter : public ChunkToInsertDataConverter {
  const Chunk_NS::Chunk* chunk_;
  const ColumnDescriptor* column_descriptor_;

  std::unique_ptr<std::vector<std::string>> column_data_;
  const int8_t* data_buffer_addr_;
  const StringOffsetT* index_buffer_addr_;

  StringChunkConverter(size_t num_rows, const Chunk_NS::Chunk* chunk)
      : chunk_(chunk), column_descriptor_(chunk->get_column_desc()) {
    column_data_ = std::make_unique<std::vector<std::string>>(num_rows);
    data_buffer_addr_ = chunk->get_buffer()->getMemoryPtr();
    index_buffer_addr_ =
        (StringOffsetT*)(chunk->get_index_buf() ? chunk->get_index_buf()->getMemoryPtr()
                                                : nullptr);
  }

  virtual ~StringChunkConverter() {}

  virtual void convertToColumnarFormat(size_t row, size_t indexInFragment) {
    size_t src_value_size =
        index_buffer_addr_[indexInFragment + 1] - index_buffer_addr_[indexInFragment];
    auto src_value_ptr = data_buffer_addr_ + index_buffer_addr_[indexInFragment];
    (*column_data_)[row] = std::string((const char*)src_value_ptr, src_value_size);
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    DataBlockPtr dataBlock;
    dataBlock.stringsPtr = column_data_.get();
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

void InsertOrderFragmenter::updateColumns(
    const Catalog_Namespace::Catalog* catalog,
    const TableDescriptor* td,
    const int fragmentId,
    const std::vector<const ColumnDescriptor*> columnDescriptors,
    const RowDataProvider& sourceDataProvider,
    const size_t indexOffFragmentOffsetColumn,
    const Data_Namespace::MemoryLevel memoryLevel,
    UpdelRoll& updelRoll) {
  if (!is_feature_enabled<VarlenUpdates>()) {
    throw std::runtime_error("varlen UPDATE path not enabled.");
  }

  updelRoll.is_varlen_update = true;
  updelRoll.catalog = catalog;
  updelRoll.logicalTableId = catalog->getLogicalTableId(td->tableId);
  updelRoll.memoryLevel = memoryLevel;

  size_t num_rows = sourceDataProvider.count();

  if (0 == num_rows) {
    // bail out early
    return;
  }

  TargetValueConverterFactory factory;

  auto& fragment = getFragmentInfoFromId(fragmentId);
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks;
  get_chunks(catalog, td, fragment, memoryLevel, chunks);
  std::vector<std::unique_ptr<TargetValueConverter>> sourceDataConverters(
      columnDescriptors.size());
  std::vector<std::unique_ptr<ChunkToInsertDataConverter>> chunkConverters;

  std::shared_ptr<Chunk_NS::Chunk> deletedChunk;
  for (size_t indexOfChunk = 0; indexOfChunk < chunks.size(); indexOfChunk++) {
    auto chunk = chunks[indexOfChunk];
    const auto chunk_cd = chunk->get_column_desc();

    if (chunk_cd->isDeletedCol) {
      deletedChunk = chunk;
      continue;
    }

    auto targetColumnIt = std::find_if(columnDescriptors.begin(),
                                       columnDescriptors.end(),
                                       [=](const ColumnDescriptor* cd) -> bool {
                                         return cd->columnId == chunk_cd->columnId;
                                       });

    if (targetColumnIt != columnDescriptors.end()) {
      auto indexOfTargetColumn = std::distance(columnDescriptors.begin(), targetColumnIt);

      auto sourceDescriptor = columnDescriptors[indexOfTargetColumn];
      auto targetDescriptor = columnDescriptors[indexOfTargetColumn];

      ConverterCreateParameter param{num_rows,
                                     *catalog,
                                     sourceDescriptor,
                                     targetDescriptor,
                                     targetDescriptor->columnType,
                                     !targetDescriptor->columnType.get_notnull(),
                                     true};
      auto converter = factory.create(param);
      sourceDataConverters[indexOfTargetColumn] = std::move(converter);

      if (targetDescriptor->columnType.is_geometry()) {
        // geometry columns are composites
        // need to skip chunks, depending on geo type
        switch (targetDescriptor->columnType.get_type()) {
          case kMULTIPOLYGON:
            indexOfChunk += 5;
            break;
          case kPOLYGON:
            indexOfChunk += 4;
            break;
          case kLINESTRING:
            indexOfChunk += 2;
            break;
          case kPOINT:
            indexOfChunk += 1;
            break;
          default:
            CHECK(false);  // not supported
        }
      }
    } else {
      if (chunk_cd->columnType.is_varlen() || chunk_cd->columnType.is_fixlen_array()) {
        std::unique_ptr<ChunkToInsertDataConverter> converter;

        if (chunk_cd->columnType.is_fixlen_array()) {
          converter =
              std::make_unique<FixedLenArrayChunkConverter>(num_rows, chunk.get());
        } else if (chunk_cd->columnType.is_string()) {
          converter = std::make_unique<StringChunkConverter>(num_rows, chunk.get());
        } else if (chunk_cd->columnType.is_geometry()) {
          // the logical geo column is a string column
          converter = std::make_unique<StringChunkConverter>(num_rows, chunk.get());
        } else {
          converter = std::make_unique<ArrayChunkConverter>(num_rows, chunk.get());
        }

        chunkConverters.push_back(std::move(converter));

      } else {
        std::unique_ptr<ChunkToInsertDataConverter> converter;
        SQLTypeInfo logical_type = get_logical_type_info(chunk_cd->columnType);
        int logical_size = logical_type.get_size();
        int physical_size = chunk_cd->columnType.get_size();

        if (logical_type.is_string()) {
          // for dicts -> logical = physical
          logical_size = physical_size;
        }

        if (8 == physical_size) {
          converter = std::make_unique<ScalarChunkConverter<int64_t, int64_t>>(
              num_rows, chunk.get());
        } else if (4 == physical_size) {
          if (8 == logical_size) {
            converter = std::make_unique<ScalarChunkConverter<int32_t, int64_t>>(
                num_rows, chunk.get());
          } else {
            converter = std::make_unique<ScalarChunkConverter<int32_t, int32_t>>(
                num_rows, chunk.get());
          }
        } else if (2 == chunk_cd->columnType.get_size()) {
          if (8 == logical_size) {
            converter = std::make_unique<ScalarChunkConverter<int16_t, int64_t>>(
                num_rows, chunk.get());
          } else if (4 == logical_size) {
            converter = std::make_unique<ScalarChunkConverter<int16_t, int32_t>>(
                num_rows, chunk.get());
          } else {
            converter = std::make_unique<ScalarChunkConverter<int16_t, int16_t>>(
                num_rows, chunk.get());
          }
        } else if (1 == chunk_cd->columnType.get_size()) {
          if (8 == logical_size) {
            converter = std::make_unique<ScalarChunkConverter<int8_t, int64_t>>(
                num_rows, chunk.get());
          } else if (4 == logical_size) {
            converter = std::make_unique<ScalarChunkConverter<int8_t, int32_t>>(
                num_rows, chunk.get());
          } else if (2 == logical_size) {
            converter = std::make_unique<ScalarChunkConverter<int8_t, int16_t>>(
                num_rows, chunk.get());
          } else {
            converter = std::make_unique<ScalarChunkConverter<int8_t, int8_t>>(
                num_rows, chunk.get());
          }
        } else {
          CHECK(false);  // unknown
        }

        chunkConverters.push_back(std::move(converter));
      }
    }
  }

  static boost_variant_accessor<ScalarTargetValue> SCALAR_TARGET_VALUE_ACCESSOR;
  static boost_variant_accessor<int64_t> OFFSET_VALUE__ACCESSOR;

  updelRoll.dirtyChunks[deletedChunk.get()] = deletedChunk;
  ChunkKey chunkey{updelRoll.catalog->getCurrentDB().dbId,
                   deletedChunk->get_column_desc()->tableId,
                   deletedChunk->get_column_desc()->columnId,
                   fragment.fragmentId};
  updelRoll.dirtyChunkeys.insert(chunkey);
  bool* deletedChunkBuffer =
      reinterpret_cast<bool*>(deletedChunk->get_buffer()->getMemoryPtr());

  auto row_converter = [&sourceDataProvider,
                        &sourceDataConverters,
                        &indexOffFragmentOffsetColumn,
                        &chunkConverters,
                        &deletedChunkBuffer](size_t indexOfRow) -> void {
    // convert the source data
    const auto row = sourceDataProvider.getTranslatedEntryAt(indexOfRow);

    for (size_t col = 0; col < sourceDataConverters.size(); col++) {
      if (sourceDataConverters[col]) {
        const auto& mapd_variant = row[col];
        sourceDataConverters[col]->convertToColumnarFormat(indexOfRow, &mapd_variant);
      }
    }

    auto scalar = checked_get(
        indexOfRow, &row[indexOffFragmentOffsetColumn], SCALAR_TARGET_VALUE_ACCESSOR);
    auto indexInChunkBuffer = *checked_get(indexOfRow, scalar, OFFSET_VALUE__ACCESSOR);

    // convert the remaining chunks
    for (size_t idx = 0; idx < chunkConverters.size(); idx++) {
      chunkConverters[idx]->convertToColumnarFormat(indexOfRow, indexInChunkBuffer);
    }

    // now mark the row as deleted
    deletedChunkBuffer[indexInChunkBuffer] = true;
  };

  bool can_go_parallel = num_rows > 20000;

  if (can_go_parallel) {
    const size_t num_worker_threads = cpu_threads();
    std::vector<std::future<void>> worker_threads;
    for (size_t i = 0,
                start_entry = 0,
                stride = (num_rows + num_worker_threads - 1) / num_worker_threads;
         i < num_worker_threads && start_entry < num_rows;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, num_rows);
      worker_threads.push_back(
          std::async(std::launch::async,
                     [&row_converter](const size_t start, const size_t end) {
                       for (size_t indexOfRow = start; indexOfRow < end; ++indexOfRow) {
                         row_converter(indexOfRow);
                       }
                     },
                     start_entry,
                     end_entry));
    }

    for (auto& child : worker_threads) {
      child.wait();
    }

  } else {
    for (size_t indexOfRow = 0; indexOfRow < num_rows; indexOfRow++) {
      row_converter(indexOfRow);
    }
  }

  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = catalog->getCurrentDB().dbId;
  insert_data.tableId = td->tableId;

  for (size_t i = 0; i < chunkConverters.size(); i++) {
    chunkConverters[i]->addDataBlocksToInsertData(insert_data);
    continue;
  }

  for (size_t i = 0; i < sourceDataConverters.size(); i++) {
    if (sourceDataConverters[i]) {
      sourceDataConverters[i]->addDataBlocksToInsertData(insert_data);
    }
    continue;
  }

  insert_data.numRows = num_rows;
  insertDataNoCheckpoint(insert_data);

  // update metdata
  if (!deletedChunk->get_buffer()->hasEncoder) {
    deletedChunk->init_encoder();
  }
  deletedChunk->get_buffer()->encoder->updateStats(static_cast<int64_t>(true), false);

  auto& shadowDeletedChunkMeta =
      fragment.shadowChunkMetadataMap[deletedChunk->get_column_desc()->columnId];
  if (shadowDeletedChunkMeta.numElements >
      deletedChunk->get_buffer()->encoder->getNumElems()) {
    // the append will have populated shadow meta data, otherwise use existing num
    // elements
    deletedChunk->get_buffer()->encoder->setNumElems(shadowDeletedChunkMeta.numElements);
  }
  deletedChunk->get_buffer()->setUpdated();
}

void InsertOrderFragmenter::updateColumn(const Catalog_Namespace::Catalog* catalog,
                                         const TableDescriptor* td,
                                         const ColumnDescriptor* cd,
                                         const int fragment_id,
                                         const std::vector<uint64_t>& frag_offsets,
                                         const std::vector<ScalarTargetValue>& rhs_values,
                                         const SQLTypeInfo& rhs_type,
                                         const Data_Namespace::MemoryLevel memory_level,
                                         UpdelRoll& updel_roll) {
  updel_roll.catalog = catalog;
  updel_roll.logicalTableId = catalog->getLogicalTableId(td->tableId);
  updel_roll.memoryLevel = memory_level;

  const size_t ncore = cpu_threads();
  const auto nrow = frag_offsets.size();
  const auto n_rhs_values = rhs_values.size();
  if (0 == nrow) {
    return;
  }
  CHECK(nrow == n_rhs_values || 1 == n_rhs_values);

  auto& fragment = getFragmentInfoFromId(fragment_id);
  auto chunk_meta_it = fragment.getChunkMetadataMapPhysical().find(cd->columnId);
  CHECK(chunk_meta_it != fragment.getChunkMetadataMapPhysical().end());
  ChunkKey chunk_key{
      catalog->getCurrentDB().dbId, td->tableId, cd->columnId, fragment.fragmentId};
  auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                         &catalog->getDataMgr(),
                                         chunk_key,
                                         Data_Namespace::CPU_LEVEL,
                                         0,
                                         chunk_meta_it->second.numBytes,
                                         chunk_meta_it->second.numElements);

  std::vector<int8_t> has_null_per_thread(ncore, 0);
  std::vector<double> max_double_per_thread(ncore, std::numeric_limits<double>::min());
  std::vector<double> min_double_per_thread(ncore, std::numeric_limits<double>::max());
  std::vector<int64_t> max_int64t_per_thread(ncore, std::numeric_limits<int64_t>::min());
  std::vector<int64_t> min_int64t_per_thread(ncore, std::numeric_limits<int64_t>::max());

  // parallel update elements
  std::vector<std::future<void>> threads;

  const auto segsz = (nrow + ncore - 1) / ncore;
  auto dbuf = chunk->get_buffer();
  auto dbuf_addr = dbuf->getMemoryPtr();
  dbuf->setUpdated();
  {
    std::lock_guard<std::mutex> lck(updel_roll.mutex);
    if (updel_roll.dirtyChunks.count(chunk.get()) == 0) {
      updel_roll.dirtyChunks.emplace(chunk.get(), chunk);
    }

    ChunkKey chunkey{updel_roll.catalog->getCurrentDB().dbId,
                     cd->tableId,
                     cd->columnId,
                     fragment.fragmentId};
    updel_roll.dirtyChunkeys.insert(chunkey);
  }
  for (size_t rbegin = 0, c = 0; rbegin < nrow; ++c, rbegin += segsz) {
    threads.emplace_back(std::async(
        std::launch::async,
        [=,
         &has_null_per_thread,
         &min_int64t_per_thread,
         &max_int64t_per_thread,
         &min_double_per_thread,
         &max_double_per_thread,
         &frag_offsets,
         &rhs_values] {
          SQLTypeInfo lhs_type = cd->columnType;

          // !! not sure if this is a undocumented convention or a bug, but for a sharded
          // table the dictionary id of a encoded string column is not specified by
          // comp_param in physical table but somehow in logical table :) comp_param in
          // physical table is always 0, so need to adapt accordingly...
          auto cdl = (shard_ < 0)
                         ? cd
                         : catalog->getMetadataForColumn(
                               catalog->getLogicalTableId(td->tableId), cd->columnId);
          CHECK(cdl);
          DecimalOverflowValidator decimalOverflowValidator(lhs_type);
          NullAwareValidator<DecimalOverflowValidator> nullAwareDecimalOverflowValidator(
              lhs_type, &decimalOverflowValidator);

          StringDictionary* stringDict{nullptr};
          if (lhs_type.is_string()) {
            CHECK(kENCODING_DICT == lhs_type.get_compression());
            auto dictDesc = const_cast<DictDescriptor*>(
                catalog->getMetadataForDict(cdl->columnType.get_comp_param()));
            CHECK(dictDesc);
            stringDict = dictDesc->stringDict.get();
            CHECK(stringDict);
          }

          for (size_t r = rbegin; r < std::min(rbegin + segsz, nrow); r++) {
            const auto roffs = frag_offsets[r];
            auto data_ptr = dbuf_addr + roffs * get_element_size(lhs_type);
            auto sv = &rhs_values[1 == n_rhs_values ? 0 : r];
            ScalarTargetValue sv2;

            // Subtle here is on the two cases of string-to-string assignments, when
            // upstream passes RHS string as a string index instead of a preferred "real
            // string".
            //   case #1. For "SET str_col = str_literal", it is hard to resolve temp str
            //   index
            //            in this layer, so if upstream passes a str idx here, an
            //            exception is thrown.
            //   case #2. For "SET str_col1 = str_col2", RHS str idx is converted to LHS
            //   str idx.
            if (rhs_type.is_string()) {
              if (const auto vp = boost::get<int64_t>(sv)) {
                auto dictDesc = const_cast<DictDescriptor*>(
                    catalog->getMetadataForDict(rhs_type.get_comp_param()));
                if (nullptr == dictDesc) {
                  throw std::runtime_error(
                      "UPDATE does not support cast from string literal to string "
                      "column.");
                }
                auto stringDict = dictDesc->stringDict.get();
                CHECK(stringDict);
                sv2 = NullableString(stringDict->getString(*vp));
                sv = &sv2;
              }
            }

            if (const auto vp = boost::get<int64_t>(sv)) {
              auto v = *vp;
              if (lhs_type.is_string()) {
#ifdef ENABLE_STRING_CONVERSION_AT_STORAGE_LAYER
                v = stringDict->getOrAdd(DatumToString(
                    rhs_type.is_time() ? Datum{.timeval = v} : Datum{.bigintval = v},
                    rhs_type));
#else
                throw std::runtime_error("UPDATE does not support cast to string.");
#endif
              }
              put_scalar<int64_t>(data_ptr, lhs_type, v, cd->columnName, &rhs_type);
              if (lhs_type.is_decimal()) {
                nullAwareDecimalOverflowValidator.validate<int64_t>(v);
                int64_t decimal;
                get_scalar<int64_t>(data_ptr, lhs_type, decimal);
                set_minmax<int64_t>(
                    min_int64t_per_thread[c], max_int64t_per_thread[c], decimal);
                if (!((v >= 0) ^ (decimal < 0))) {
                  throw std::runtime_error(
                      "Data conversion overflow on " + std::to_string(v) +
                      " from DECIMAL(" + std::to_string(rhs_type.get_dimension()) + ", " +
                      std::to_string(rhs_type.get_scale()) + ") to (" +
                      std::to_string(lhs_type.get_dimension()) + ", " +
                      std::to_string(lhs_type.get_scale()) + ")");
                }
              } else if (is_integral(lhs_type)) {
                if (lhs_type.is_date_in_days()) {
                  // Store meta values in seconds
                  int64_t days;
                  get_scalar<int64_t>(data_ptr, lhs_type, days);
                  const auto seconds = DateConverters::get_epoch_seconds_from_days(days);
                  set_minmax<int64_t>(
                      min_int64t_per_thread[c], max_int64t_per_thread[c], seconds);
                } else {
                  int64_t target_value;
                  if (rhs_type.is_decimal()) {
                    target_value = round(decimal_to_double(rhs_type, v));
                  } else {
                    target_value = v;
                  }

                  set_minmax<int64_t>(
                      min_int64t_per_thread[c], max_int64t_per_thread[c], target_value);
                }
              } else {
                set_minmax<double>(
                    min_double_per_thread[c],
                    max_double_per_thread[c],
                    rhs_type.is_decimal() ? decimal_to_double(rhs_type, v) : v);
              }
            } else if (const auto vp = boost::get<double>(sv)) {
              auto v = *vp;
              if (lhs_type.is_string()) {
#ifdef ENABLE_STRING_CONVERSION_AT_STORAGE_LAYER
                v = stringDict->getOrAdd(DatumToString(Datum{.doubleval = v}, rhs_type));
#else
                throw std::runtime_error("UPDATE does not support cast to string.");
#endif
              }
              put_scalar<double>(data_ptr, lhs_type, v, cd->columnName);
              if (lhs_type.is_integer()) {
                set_minmax<int64_t>(
                    min_int64t_per_thread[c], max_int64t_per_thread[c], v);
              } else {
                set_minmax<double>(min_double_per_thread[c], max_double_per_thread[c], v);
              }
            } else if (const auto vp = boost::get<float>(sv)) {
              auto v = *vp;
              if (lhs_type.is_string()) {
#ifdef ENABLE_STRING_CONVERSION_AT_STORAGE_LAYER
                v = stringDict->getOrAdd(DatumToString(Datum{.floatval = v}, rhs_type));
#else
                throw std::runtime_error("UPDATE does not support cast to string.");
#endif
              }
              put_scalar<float>(data_ptr, lhs_type, v, cd->columnName);
              if (lhs_type.is_integer()) {
                set_minmax<int64_t>(
                    min_int64t_per_thread[c], max_int64t_per_thread[c], v);
              } else {
                set_minmax<double>(min_double_per_thread[c], max_double_per_thread[c], v);
              }
            } else if (const auto vp = boost::get<NullableString>(sv)) {
              const auto s = boost::get<std::string>(vp);
              const auto sval = s ? *s : std::string("");
              if (lhs_type.is_string()) {
                decltype(stringDict->getOrAdd(sval)) sidx;
                {
                  std::unique_lock<std::mutex> lock(temp_mutex_);
                  sidx = stringDict->getOrAdd(sval);
                }
                put_scalar<int32_t>(data_ptr, lhs_type, sidx, cd->columnName);
                set_minmax<int64_t>(
                    min_int64t_per_thread[c], max_int64t_per_thread[c], sidx);
              } else if (sval.size() > 0) {
                auto dval = std::atof(sval.data());
                if (lhs_type.is_boolean()) {
                  dval = sval == "t" || sval == "true" || sval == "T" || sval == "True";
                } else if (lhs_type.is_time()) {
                  throw std::runtime_error(
                      "Date/Time/Timestamp update not supported through translated "
                      "string path.");
                }
                if (lhs_type.is_fp() || lhs_type.is_decimal()) {
                  put_scalar<double>(data_ptr, lhs_type, dval, cd->columnName);
                  set_minmax<double>(
                      min_double_per_thread[c], max_double_per_thread[c], dval);
                } else {
                  put_scalar<int64_t>(data_ptr, lhs_type, dval, cd->columnName);
                  set_minmax<int64_t>(
                      min_int64t_per_thread[c], max_int64t_per_thread[c], dval);
                }
              } else {
                put_null(data_ptr, lhs_type, cd->columnName);
                has_null_per_thread[c] = true;
              }
            } else {
              CHECK(false);
            }
          }
        }));
    if (threads.size() >= (size_t)cpu_threads()) {
      wait_cleanup_threads(threads);
    }
  }
  wait_cleanup_threads(threads);

  // for unit test
  if (Fragmenter_Namespace::FragmentInfo::unconditionalVacuum_) {
    if (cd->isDeletedCol) {
      const auto deleted_offsets = getVacuumOffsets(chunk);
      if (deleted_offsets.size() > 0) {
        compactRows(catalog, td, fragment_id, deleted_offsets, memory_level, updel_roll);
        return;
      }
    }
  }
  bool has_null_per_chunk{false};
  double max_double_per_chunk{std::numeric_limits<double>::min()};
  double min_double_per_chunk{std::numeric_limits<double>::max()};
  int64_t max_int64t_per_chunk{std::numeric_limits<int64_t>::min()};
  int64_t min_int64t_per_chunk{std::numeric_limits<int64_t>::max()};
  for (size_t c = 0; c < ncore; ++c) {
    has_null_per_chunk = has_null_per_chunk || has_null_per_thread[c];
    max_double_per_chunk =
        std::max<double>(max_double_per_chunk, max_double_per_thread[c]);
    min_double_per_chunk =
        std::min<double>(min_double_per_chunk, min_double_per_thread[c]);
    max_int64t_per_chunk =
        std::max<int64_t>(max_int64t_per_chunk, max_int64t_per_thread[c]);
    min_int64t_per_chunk =
        std::min<int64_t>(min_int64t_per_chunk, min_int64t_per_thread[c]);
  }
  updateColumnMetadata(cd,
                       fragment,
                       chunk,
                       has_null_per_chunk,
                       max_double_per_chunk,
                       min_double_per_chunk,
                       max_int64t_per_chunk,
                       min_int64t_per_chunk,
                       cd->columnType,
                       updel_roll);
}

void InsertOrderFragmenter::updateColumnMetadata(const ColumnDescriptor* cd,
                                                 FragmentInfo& fragment,
                                                 std::shared_ptr<Chunk_NS::Chunk> chunk,
                                                 const bool has_null_per_chunk,
                                                 const double max_double_per_chunk,
                                                 const double min_double_per_chunk,
                                                 const int64_t max_int64t_per_chunk,
                                                 const int64_t min_int64t_per_chunk,
                                                 const SQLTypeInfo& rhs_type,
                                                 UpdelRoll& updel_roll) {
  auto td = updel_roll.catalog->getMetadataForTable(cd->tableId);
  auto key = std::make_pair(td, &fragment);
  std::lock_guard<std::mutex> lck(updel_roll.mutex);
  if (0 == updel_roll.chunkMetadata.count(key)) {
    updel_roll.chunkMetadata[key] = fragment.getChunkMetadataMapPhysical();
  }
  if (0 == updel_roll.numTuples.count(key)) {
    updel_roll.numTuples[key] = fragment.shadowNumTuples;
  }
  auto& chunkMetadata = updel_roll.chunkMetadata[key];

  auto buffer = chunk->get_buffer();
  const auto& lhs_type = cd->columnType;
  if (is_integral(lhs_type) || (lhs_type.is_decimal() && rhs_type.is_decimal())) {
    buffer->encoder->updateStats(max_int64t_per_chunk, has_null_per_chunk);
    buffer->encoder->updateStats(min_int64t_per_chunk, has_null_per_chunk);
  } else if (lhs_type.is_fp()) {
    buffer->encoder->updateStats(max_double_per_chunk, has_null_per_chunk);
    buffer->encoder->updateStats(min_double_per_chunk, has_null_per_chunk);
  } else if (lhs_type.is_decimal()) {
    buffer->encoder->updateStats(
        (int64_t)(max_double_per_chunk * pow(10, lhs_type.get_scale())),
        has_null_per_chunk);
    buffer->encoder->updateStats(
        (int64_t)(min_double_per_chunk * pow(10, lhs_type.get_scale())),
        has_null_per_chunk);
  } else if (!lhs_type.is_array() && !lhs_type.is_geometry() &&
             !(lhs_type.is_string() && kENCODING_DICT != lhs_type.get_compression())) {
    buffer->encoder->updateStats(max_int64t_per_chunk, has_null_per_chunk);
    buffer->encoder->updateStats(min_int64t_per_chunk, has_null_per_chunk);
  }
  buffer->encoder->getMetadata(chunkMetadata[cd->columnId]);

  // removed as @alex suggests. keep it commented in case of any chance to revisit
  // it once after vacuum code is introduced. fragment.invalidateChunkMetadataMap();
}

void InsertOrderFragmenter::updateMetadata(const Catalog_Namespace::Catalog* catalog,
                                           const MetaDataKey& key,
                                           UpdelRoll& updel_roll) {
  mapd_unique_lock<mapd_shared_mutex> writeLock(fragmentInfoMutex_);
  if (updel_roll.chunkMetadata.count(key)) {
    auto& fragmentInfo = *key.second;
    const auto& chunkMetadata = updel_roll.chunkMetadata[key];
    fragmentInfo.shadowChunkMetadataMap = chunkMetadata;
    fragmentInfo.setChunkMetadataMap(chunkMetadata);
    fragmentInfo.shadowNumTuples = updel_roll.numTuples[key];
    fragmentInfo.setPhysicalNumTuples(fragmentInfo.shadowNumTuples);
    // TODO(ppan): When fragment-level compaction is enable, the following code
    // should suffice. When not (ie. existing code), we'll revert to update
    // InsertOrderFragmenter::varLenColInfo_
    /*
    for (const auto cit : chunkMetadata) {
      const auto& cd = *catalog->getMetadataForColumn(td->tableId, cit.first);
      if (cd.columnType.get_size() < 0)
        fragmentInfo.varLenColInfox[cd.columnId] = cit.second.numBytes;
    }
    */
  }
}

auto InsertOrderFragmenter::getChunksForAllColumns(
    const TableDescriptor* td,
    const FragmentInfo& fragment,
    const Data_Namespace::MemoryLevel memory_level) {
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks;
  // coming from updateColumn (on '$delete$' column) we dont have chunks for all columns
  for (int col_id = 1, ncol = 0; ncol < td->nColumns; ++col_id) {
    if (const auto cd = catalog_->getMetadataForColumn(td->tableId, col_id)) {
      ++ncol;
      if (!cd->isVirtualCol) {
        auto chunk_meta_it = fragment.getChunkMetadataMapPhysical().find(col_id);
        CHECK(chunk_meta_it != fragment.getChunkMetadataMapPhysical().end());
        ChunkKey chunk_key{
            catalog_->getCurrentDB().dbId, td->tableId, col_id, fragment.fragmentId};
        auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                               &catalog_->getDataMgr(),
                                               chunk_key,
                                               memory_level,
                                               0,
                                               chunk_meta_it->second.numBytes,
                                               chunk_meta_it->second.numElements);
        chunks.push_back(chunk);
      }
    }
  }
  return chunks;
}

// get a sorted vector of offsets of rows to vacuum
const std::vector<uint64_t> InsertOrderFragmenter::getVacuumOffsets(
    const std::shared_ptr<Chunk_NS::Chunk>& chunk) {
  const auto data_buffer = chunk->get_buffer();
  const auto data_addr = data_buffer->getMemoryPtr();
  const size_t nrows_in_chunk = data_buffer->size();
  const size_t ncore = cpu_threads();
  const size_t segsz = (nrows_in_chunk + ncore - 1) / ncore;
  std::vector<std::vector<uint64_t>> deleted_offsets;
  deleted_offsets.resize(ncore);
  std::vector<std::future<void>> threads;
  for (size_t rbegin = 0; rbegin < nrows_in_chunk; rbegin += segsz) {
    threads.emplace_back(std::async(std::launch::async, [=, &deleted_offsets] {
      const auto rend = std::min<size_t>(rbegin + segsz, nrows_in_chunk);
      const auto ithread = rbegin / segsz;
      CHECK(ithread < deleted_offsets.size());
      deleted_offsets[ithread].reserve(segsz);
      for (size_t r = rbegin; r < rend; ++r) {
        if (data_addr[r]) {
          deleted_offsets[ithread].push_back(r);
        }
      }
    }));
  }
  wait_cleanup_threads(threads);
  std::vector<uint64_t> all_deleted_offsets;
  for (size_t i = 0; i < ncore; ++i) {
    all_deleted_offsets.insert(
        all_deleted_offsets.end(), deleted_offsets[i].begin(), deleted_offsets[i].end());
  }
  return all_deleted_offsets;
}

template <typename T>
static void set_chunk_stats(const SQLTypeInfo& col_type,
                            int8_t* data_addr,
                            int8_t& has_null,
                            T& min,
                            T& max) {
  T v;
  const auto can_be_null = !col_type.get_notnull();
  const auto is_null = get_scalar<T>(data_addr, col_type, v);
  if (is_null) {
    has_null = has_null || (can_be_null && is_null);
  } else {
    set_minmax(min, max, v);
  }
}

static void set_chunk_metadata(const Catalog_Namespace::Catalog* catalog,
                               FragmentInfo& fragment,
                               const std::shared_ptr<Chunk_NS::Chunk>& chunk,
                               const size_t nrows_to_keep,
                               UpdelRoll& updel_roll) {
  auto cd = chunk->get_column_desc();
  auto td = catalog->getMetadataForTable(cd->tableId);
  auto data_buffer = chunk->get_buffer();
  std::lock_guard<std::mutex> lck(updel_roll.mutex);
  const auto key = std::make_pair(td, &fragment);
  if (0 == updel_roll.chunkMetadata.count(key)) {
    updel_roll.chunkMetadata[key] = fragment.getChunkMetadataMapPhysical();
  }
  auto& chunkMetadata = updel_roll.chunkMetadata[key];
  chunkMetadata[cd->columnId].numElements = nrows_to_keep;
  chunkMetadata[cd->columnId].numBytes = data_buffer->size();
  if (updel_roll.dirtyChunks.count(chunk.get()) == 0) {
    updel_roll.dirtyChunks.emplace(chunk.get(), chunk);
  }
}

auto InsertOrderFragmenter::vacuum_fixlen_rows(
    const FragmentInfo& fragment,
    const std::shared_ptr<Chunk_NS::Chunk>& chunk,
    const std::vector<uint64_t>& frag_offsets) {
  const auto cd = chunk->get_column_desc();
  const auto& col_type = cd->columnType;
  auto data_buffer = chunk->get_buffer();
  auto data_addr = data_buffer->getMemoryPtr();
  auto element_size =
      col_type.is_fixlen_array() ? col_type.get_size() : get_element_size(col_type);
  int64_t irow_of_blk_to_keep = 0;  // head of next row block to keep
  int64_t irow_of_blk_to_fill = 0;  // row offset to fit the kept block
  size_t nbytes_fix_data_to_keep = 0;
  auto nrows_to_vacuum = frag_offsets.size();
  auto nrows_in_fragment = fragment.getPhysicalNumTuples();
  for (size_t irow = 0; irow <= nrows_to_vacuum; irow++) {
    auto is_last_one = irow == nrows_to_vacuum;
    auto irow_to_vacuum = is_last_one ? nrows_in_fragment : frag_offsets[irow];
    auto maddr_to_vacuum = data_addr;
    int64_t nrows_to_keep = irow_to_vacuum - irow_of_blk_to_keep;
    if (nrows_to_keep > 0) {
      auto nbytes_to_keep = nrows_to_keep * element_size;
      if (irow_of_blk_to_fill != irow_of_blk_to_keep) {
        // move curr fixlen row block toward front
        memmove(maddr_to_vacuum + irow_of_blk_to_fill * element_size,
                maddr_to_vacuum + irow_of_blk_to_keep * element_size,
                nbytes_to_keep);
      }
      irow_of_blk_to_fill += nrows_to_keep;
      nbytes_fix_data_to_keep += nbytes_to_keep;
    }
    irow_of_blk_to_keep = irow_to_vacuum + 1;
  }
  return nbytes_fix_data_to_keep;
}

auto InsertOrderFragmenter::vacuum_varlen_rows(
    const FragmentInfo& fragment,
    const std::shared_ptr<Chunk_NS::Chunk>& chunk,
    const std::vector<uint64_t>& frag_offsets) {
  auto data_buffer = chunk->get_buffer();
  auto index_buffer = chunk->get_index_buf();
  auto data_addr = data_buffer->getMemoryPtr();
  auto indices_addr = index_buffer ? index_buffer->getMemoryPtr() : nullptr;
  auto index_array = (StringOffsetT*)indices_addr;
  int64_t irow_of_blk_to_keep = 0;  // head of next row block to keep
  int64_t irow_of_blk_to_fill = 0;  // row offset to fit the kept block
  size_t nbytes_fix_data_to_keep = 0;
  size_t nbytes_var_data_to_keep = 0;
  auto nrows_to_vacuum = frag_offsets.size();
  auto nrows_in_fragment = fragment.getPhysicalNumTuples();
  for (size_t irow = 0; irow <= nrows_to_vacuum; irow++) {
    auto is_last_one = irow == nrows_to_vacuum;
    auto irow_to_vacuum = is_last_one ? nrows_in_fragment : frag_offsets[irow];
    auto maddr_to_vacuum = data_addr;
    int64_t nrows_to_keep = irow_to_vacuum - irow_of_blk_to_keep;
    if (nrows_to_keep > 0) {
      auto ibyte_var_data_to_keep = nbytes_var_data_to_keep;
      auto nbytes_to_keep =
          (is_last_one ? data_buffer->size() : index_array[irow_to_vacuum]) -
          index_array[irow_of_blk_to_keep];
      if (irow_of_blk_to_fill != irow_of_blk_to_keep) {
        // move curr varlen row block toward front
        memmove(data_addr + ibyte_var_data_to_keep,
                data_addr + index_array[irow_of_blk_to_keep],
                nbytes_to_keep);

        const auto index_base = index_array[irow_of_blk_to_keep];
        for (int64_t i = 0; i < nrows_to_keep; ++i) {
          auto& index = index_array[irow_of_blk_to_keep + i];
          index = ibyte_var_data_to_keep + (index - index_base);
        }
      }
      nbytes_var_data_to_keep += nbytes_to_keep;
      maddr_to_vacuum = indices_addr;

      constexpr static auto index_element_size = sizeof(StringOffsetT);
      nbytes_to_keep = nrows_to_keep * index_element_size;
      if (irow_of_blk_to_fill != irow_of_blk_to_keep) {
        // move curr fixlen row block toward front
        memmove(maddr_to_vacuum + irow_of_blk_to_fill * index_element_size,
                maddr_to_vacuum + irow_of_blk_to_keep * index_element_size,
                nbytes_to_keep);
      }
      irow_of_blk_to_fill += nrows_to_keep;
      nbytes_fix_data_to_keep += nbytes_to_keep;
    }
    irow_of_blk_to_keep = irow_to_vacuum + 1;
  }
  return nbytes_var_data_to_keep;
}

void InsertOrderFragmenter::compactRows(const Catalog_Namespace::Catalog* catalog,
                                        const TableDescriptor* td,
                                        const int fragment_id,
                                        const std::vector<uint64_t>& frag_offsets,
                                        const Data_Namespace::MemoryLevel memory_level,
                                        UpdelRoll& updel_roll) {
  auto& fragment = getFragmentInfoFromId(fragment_id);
  auto chunks = getChunksForAllColumns(td, fragment, memory_level);
  const auto ncol = chunks.size();

  std::vector<int8_t> has_null_per_thread(ncol, 0);
  std::vector<double> max_double_per_thread(ncol, std::numeric_limits<double>::min());
  std::vector<double> min_double_per_thread(ncol, std::numeric_limits<double>::max());
  std::vector<int64_t> max_int64t_per_thread(ncol, std::numeric_limits<uint64_t>::min());
  std::vector<int64_t> min_int64t_per_thread(ncol, std::numeric_limits<uint64_t>::max());

  // parallel delete columns
  std::vector<std::future<void>> threads;
  auto nrows_to_vacuum = frag_offsets.size();
  auto nrows_in_fragment = fragment.getPhysicalNumTuples();
  auto nrows_to_keep = nrows_in_fragment - nrows_to_vacuum;

  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    auto chunk = chunks[ci];
    const auto cd = chunk->get_column_desc();
    const auto& col_type = cd->columnType;
    auto data_buffer = chunk->get_buffer();
    auto index_buffer = chunk->get_index_buf();
    auto data_addr = data_buffer->getMemoryPtr();
    auto indices_addr = index_buffer ? index_buffer->getMemoryPtr() : nullptr;
    auto index_array = (StringOffsetT*)indices_addr;
    bool is_varlen = col_type.is_varlen_indeed();

    auto fixlen_vacuum = [=,
                          &has_null_per_thread,
                          &max_double_per_thread,
                          &min_double_per_thread,
                          &min_int64t_per_thread,
                          &max_int64t_per_thread,
                          &updel_roll,
                          &frag_offsets,
                          &fragment] {
      size_t nbytes_fix_data_to_keep;
      nbytes_fix_data_to_keep = vacuum_fixlen_rows(fragment, chunk, frag_offsets);

      data_buffer->encoder->setNumElems(nrows_to_keep);
      data_buffer->setSize(nbytes_fix_data_to_keep);
      data_buffer->setUpdated();

      set_chunk_metadata(catalog, fragment, chunk, nrows_to_keep, updel_roll);

      auto daddr = data_addr;
      auto element_size =
          col_type.is_fixlen_array() ? col_type.get_size() : get_element_size(col_type);
      for (size_t irow = 0; irow < nrows_to_keep; ++irow, daddr += element_size) {
        if (col_type.is_fixlen_array()) {
          auto encoder =
              dynamic_cast<FixedLengthArrayNoneEncoder*>(data_buffer->encoder.get());
          CHECK(encoder);
          encoder->updateMetadata((int8_t*)daddr);
        } else if (col_type.is_fp()) {
          set_chunk_stats(col_type,
                          data_addr,
                          has_null_per_thread[ci],
                          min_double_per_thread[ci],
                          max_double_per_thread[ci]);
        } else {
          set_chunk_stats(col_type,
                          data_addr,
                          has_null_per_thread[ci],
                          min_int64t_per_thread[ci],
                          max_int64t_per_thread[ci]);
        }
      }
    };

    auto varlen_vacuum = [=, &updel_roll, &frag_offsets, &fragment] {
      size_t nbytes_var_data_to_keep;
      nbytes_var_data_to_keep = vacuum_varlen_rows(fragment, chunk, frag_offsets);

      data_buffer->encoder->setNumElems(nrows_to_keep);
      data_buffer->setSize(nbytes_var_data_to_keep);
      data_buffer->setUpdated();

      index_array[nrows_to_keep] = data_buffer->size();
      index_buffer->setSize(sizeof(*index_array) *
                            (nrows_to_keep ? 1 + nrows_to_keep : 0));
      index_buffer->setUpdated();

      set_chunk_metadata(catalog, fragment, chunk, nrows_to_keep, updel_roll);
    };

    if (is_varlen) {
      threads.emplace_back(std::async(std::launch::async, varlen_vacuum));
    } else {
      threads.emplace_back(std::async(std::launch::async, fixlen_vacuum));
    }
    if (threads.size() >= (size_t)cpu_threads()) {
      wait_cleanup_threads(threads);
    }
  }

  wait_cleanup_threads(threads);

  auto key = std::make_pair(td, &fragment);
  updel_roll.numTuples[key] = nrows_to_keep;
  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    auto chunk = chunks[ci];
    auto cd = chunk->get_column_desc();
    if (!cd->columnType.is_fixlen_array()) {
      updateColumnMetadata(cd,
                           fragment,
                           chunk,
                           has_null_per_thread[ci],
                           max_double_per_thread[ci],
                           min_double_per_thread[ci],
                           max_int64t_per_thread[ci],
                           min_int64t_per_thread[ci],
                           cd->columnType,
                           updel_roll);
    }
  }
}

}  // namespace Fragmenter_Namespace

void UpdelRoll::commitUpdate() {
  if (nullptr == catalog) {
    return;
  }
  const auto td = catalog->getMetadataForTable(logicalTableId);
  CHECK(td);
  // checkpoint all shards regardless, or epoch becomes out of sync
  if (td->persistenceLevel == Data_Namespace::MemoryLevel::DISK_LEVEL) {
    catalog->checkpoint(logicalTableId);
  }
  // for each dirty fragment
  for (auto& cm : chunkMetadata) {
    cm.first.first->fragmenter->updateMetadata(catalog, cm.first, *this);
  }
  dirtyChunks.clear();
  // flush gpu dirty chunks if update was not on gpu
  if (memoryLevel != Data_Namespace::MemoryLevel::GPU_LEVEL) {
    for (const auto& chunkey : dirtyChunkeys) {
      catalog->getDataMgr().deleteChunksWithPrefix(
          chunkey, Data_Namespace::MemoryLevel::GPU_LEVEL);
    }
  }
}

void UpdelRoll::cancelUpdate() {
  if (nullptr == catalog) {
    return;
  }

  if (is_varlen_update) {
    int databaseId = catalog->getCurrentDB().dbId;
    int32_t tableEpoch = catalog->getTableEpoch(databaseId, logicalTableId);

    dirtyChunks.clear();
    const_cast<Catalog_Namespace::Catalog*>(catalog)->setTableEpoch(
        databaseId, logicalTableId, tableEpoch);
  } else {
    const auto td = catalog->getMetadataForTable(logicalTableId);
    CHECK(td);
    if (td->persistenceLevel != memoryLevel) {
      for (auto dit : dirtyChunks) {
        catalog->getDataMgr().free(dit.first->get_buffer());
        dit.first->set_buffer(nullptr);
      }
    }
  }
}
