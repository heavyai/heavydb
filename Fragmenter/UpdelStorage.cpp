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
#include <mutex>
#include <string>
#include <vector>

#include <boost/variant.hpp>
#include <boost/variant/get.hpp>

#include "Catalog/Catalog.h"
#include "DataMgr/ArrayNoneEncoder.h"
#include "DataMgr/FixedLengthArrayNoneEncoder.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "LockMgr/LockMgr.h"
#include "QueryEngine/Execute.h"
#include "Shared/DateConverters.h"
#include "Shared/TypedDataAccessors.h"
#include "Shared/thread_count.h"
#include "TargetValueConvertersFactories.h"

extern bool g_enable_string_functions;

bool g_enable_auto_metadata_update{true};

namespace Fragmenter_Namespace {

inline void wait_cleanup_threads(std::vector<std::future<void>>& threads) {
  for (auto& t : threads) {
    t.get();
  }
  threads.clear();
}

inline bool is_integral(const SQLTypeInfo& t) {
  return t.is_integer() || t.is_boolean() || t.is_time() || t.is_timeinterval();
}

bool FragmentInfo::unconditionalVacuum_{false};

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
                                               chunk_meta_it->second->numBytes,
                                               chunk_meta_it->second->numElements);
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
      : chunk_(chunk), column_descriptor_(chunk->getColumnDesc()) {
    column_data_ = ColumnDataPtr(reinterpret_cast<INSERT_DATA_TYPE*>(
        checked_malloc(num_rows * sizeof(INSERT_DATA_TYPE))));
    data_buffer_addr_ = (BUFFER_DATA_TYPE*)chunk->getBuffer()->getMemoryPtr();
  }

  ~ScalarChunkConverter() override {}

  void convertToColumnarFormat(size_t row, size_t indexInFragment) override {
    auto buffer_value = data_buffer_addr_[indexInFragment];
    auto insert_value = static_cast<INSERT_DATA_TYPE>(buffer_value);
    column_data_.get()[row] = insert_value;
  }

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
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
      : chunk_(chunk), column_descriptor_(chunk->getColumnDesc()) {
    column_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    data_buffer_addr_ = chunk->getBuffer()->getMemoryPtr();
    fixed_array_length_ = chunk->getColumnDesc()->columnType.get_size();
  }

  ~FixedLenArrayChunkConverter() override {}

  void convertToColumnarFormat(size_t row, size_t indexInFragment) override {
    auto src_value_ptr = data_buffer_addr_ + (indexInFragment * fixed_array_length_);

    bool is_null = FixedLengthArrayNoneEncoder::is_null(column_descriptor_->columnType,
                                                        src_value_ptr);

    (*column_data_)[row] = ArrayDatum(
        fixed_array_length_, (int8_t*)src_value_ptr, is_null, DoNothingDeleter());
  }

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
    DataBlockPtr dataBlock;
    dataBlock.arraysPtr = column_data_.get();
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

struct ArrayChunkConverter : public FixedLenArrayChunkConverter {
  ArrayOffsetT* index_buffer_addr_;

  ArrayChunkConverter(const size_t num_rows, const Chunk_NS::Chunk* chunk)
      : FixedLenArrayChunkConverter(num_rows, chunk) {
    index_buffer_addr_ =
        (StringOffsetT*)(chunk->getIndexBuf() ? chunk->getIndexBuf()->getMemoryPtr()
                                              : nullptr);
  }

  ~ArrayChunkConverter() override {}

  void convertToColumnarFormat(size_t row, size_t indexInFragment) override {
    auto startIndex = index_buffer_addr_[indexInFragment];
    auto endIndex = index_buffer_addr_[indexInFragment + 1];
    size_t src_value_size = std::abs(endIndex) - std::abs(startIndex);
    auto src_value_ptr = data_buffer_addr_ + index_buffer_addr_[indexInFragment];
    (*column_data_)[row] = ArrayDatum(
        src_value_size, (int8_t*)src_value_ptr, endIndex < 0, DoNothingDeleter());
  }
};

struct StringChunkConverter : public ChunkToInsertDataConverter {
  const Chunk_NS::Chunk* chunk_;
  const ColumnDescriptor* column_descriptor_;

  std::unique_ptr<std::vector<std::string>> column_data_;
  const int8_t* data_buffer_addr_;
  const StringOffsetT* index_buffer_addr_;

  StringChunkConverter(size_t num_rows, const Chunk_NS::Chunk* chunk)
      : chunk_(chunk), column_descriptor_(chunk->getColumnDesc()) {
    column_data_ = std::make_unique<std::vector<std::string>>(num_rows);
    data_buffer_addr_ = chunk->getBuffer()->getMemoryPtr();
    index_buffer_addr_ =
        (StringOffsetT*)(chunk->getIndexBuf() ? chunk->getIndexBuf()->getMemoryPtr()
                                              : nullptr);
  }

  ~StringChunkConverter() override {}

  void convertToColumnarFormat(size_t row, size_t indexInFragment) override {
    size_t src_value_size =
        index_buffer_addr_[indexInFragment + 1] - index_buffer_addr_[indexInFragment];
    auto src_value_ptr = data_buffer_addr_ + index_buffer_addr_[indexInFragment];
    (*column_data_)[row] = std::string((const char*)src_value_ptr, src_value_size);
  }

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
    DataBlockPtr dataBlock;
    dataBlock.stringsPtr = column_data_.get();
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

template <typename BUFFER_DATA_TYPE>
struct DateChunkConverter : public ChunkToInsertDataConverter {
  using ColumnDataPtr = std::unique_ptr<int64_t, CheckedMallocDeleter<int64_t>>;

  const Chunk_NS::Chunk* chunk_;
  ColumnDataPtr column_data_;
  const ColumnDescriptor* column_descriptor_;
  const BUFFER_DATA_TYPE* data_buffer_addr_;

  DateChunkConverter(const size_t num_rows, const Chunk_NS::Chunk* chunk)
      : chunk_(chunk), column_descriptor_(chunk->getColumnDesc()) {
    column_data_ = ColumnDataPtr(
        reinterpret_cast<int64_t*>(checked_malloc(num_rows * sizeof(int64_t))));
    data_buffer_addr_ = (BUFFER_DATA_TYPE*)chunk->getBuffer()->getMemoryPtr();
  }

  ~DateChunkConverter() override {}

  void convertToColumnarFormat(size_t row, size_t indexInFragment) override {
    auto buffer_value = data_buffer_addr_[indexInFragment];
    auto insert_value = static_cast<int64_t>(buffer_value);
    column_data_.get()[row] = DateConverters::get_epoch_seconds_from_days(insert_value);
  }

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
    DataBlockPtr dataBlock;
    dataBlock.numbersPtr = reinterpret_cast<int8_t*>(column_data_.get());
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

void InsertOrderFragmenter::updateColumns(
    const Catalog_Namespace::Catalog* catalog,
    const TableDescriptor* td,
    const int fragmentId,
    const std::vector<TargetMetaInfo> sourceMetaInfo,
    const std::vector<const ColumnDescriptor*> columnDescriptors,
    const RowDataProvider& sourceDataProvider,
    const size_t indexOffFragmentOffsetColumn,
    const Data_Namespace::MemoryLevel memoryLevel,
    UpdelRoll& updelRoll,
    Executor* executor) {
  updelRoll.is_varlen_update = true;
  updelRoll.catalog = catalog;
  updelRoll.logicalTableId = catalog->getLogicalTableId(td->tableId);
  updelRoll.memoryLevel = memoryLevel;

  size_t num_entries = sourceDataProvider.getEntryCount();
  size_t num_rows = sourceDataProvider.getRowCount();

  if (0 == num_rows) {
    // bail out early
    return;
  }

  TargetValueConverterFactory factory;

  auto fragment_ptr = getFragmentInfo(fragmentId);
  auto& fragment = *fragment_ptr;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks;
  get_chunks(catalog, td, fragment, memoryLevel, chunks);
  std::vector<std::unique_ptr<TargetValueConverter>> sourceDataConverters(
      columnDescriptors.size());
  std::vector<std::unique_ptr<ChunkToInsertDataConverter>> chunkConverters;
  size_t indexOfDeletedColumn{0};
  std::shared_ptr<Chunk_NS::Chunk> deletedChunk;
  for (size_t indexOfChunk = 0; indexOfChunk < chunks.size(); indexOfChunk++) {
    auto chunk = chunks[indexOfChunk];
    const auto chunk_cd = chunk->getColumnDesc();

    if (chunk_cd->isDeletedCol) {
      indexOfDeletedColumn = chunk_cd->columnId;
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

      auto sourceDataMetaInfo = sourceMetaInfo[indexOfTargetColumn];
      auto targetDescriptor = columnDescriptors[indexOfTargetColumn];

      ConverterCreateParameter param{
          num_rows,
          sourceDataMetaInfo,
          targetDescriptor,
          *catalog,
          targetDescriptor->columnType,
          !targetDescriptor->columnType.get_notnull(),
          sourceDataProvider.getLiteralDictionary(),
          g_enable_string_functions &&
                  sourceDataMetaInfo.get_type_info().is_dict_encoded_string()
              ? executor->getStringDictionaryProxy(
                    sourceDataMetaInfo.get_type_info().getStringDictKey(),
                    executor->getRowSetMemoryOwner(),
                    true)
              : nullptr,
          nullptr};
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
          case kMULTILINESTRING:
            indexOfChunk += 3;
            break;
          case kLINESTRING:
          case kMULTIPOINT:
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

      } else if (chunk_cd->columnType.is_date_in_days()) {
        /* Q: Why do we need this?
           A: In variable length updates path we move the chunk content of column
           without decoding. Since it again passes through DateDaysEncoder
           the expected value should be in seconds, but here it will be in days.
           Therefore, using DateChunkConverter chunk values are being scaled to
           seconds which then ultimately encoded in days in DateDaysEncoder.
        */
        std::unique_ptr<ChunkToInsertDataConverter> converter;
        const size_t physical_size = chunk_cd->columnType.get_size();
        if (physical_size == 2) {
          converter =
              std::make_unique<DateChunkConverter<int16_t>>(num_rows, chunk.get());
        } else if (physical_size == 4) {
          converter =
              std::make_unique<DateChunkConverter<int32_t>>(num_rows, chunk.get());
        } else {
          CHECK(false);
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

  updelRoll.addDirtyChunk(deletedChunk, fragment.fragmentId);
  bool* deletedChunkBuffer =
      reinterpret_cast<bool*>(deletedChunk->getBuffer()->getMemoryPtr());

  std::atomic<size_t> row_idx{0};

  auto row_converter = [&sourceDataProvider,
                        &sourceDataConverters,
                        &indexOffFragmentOffsetColumn,
                        &chunkConverters,
                        &deletedChunkBuffer,
                        &row_idx](size_t indexOfEntry) -> void {
    // convert the source data
    const auto row = sourceDataProvider.getEntryAt(indexOfEntry);
    if (row.empty()) {
      return;
    }

    size_t indexOfRow = row_idx.fetch_add(1);

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
                stride = (num_entries + num_worker_threads - 1) / num_worker_threads;
         i < num_worker_threads && start_entry < num_entries;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, num_rows);
      worker_threads.push_back(std::async(
          std::launch::async,
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
    for (size_t entryIdx = 0; entryIdx < num_entries; entryIdx++) {
      row_converter(entryIdx);
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
  insert_data.is_default.resize(insert_data.columnIds.size(), false);
  insertDataNoCheckpoint(insert_data);

  // update metdata for deleted chunk as we are doing special handling
  auto chunkMetadata =
      updelRoll.getChunkMetadata({td, &fragment}, indexOfDeletedColumn, fragment);
  chunkMetadata->chunkStats.max.boolval = 1;

  // Im not completely sure that we need to do this in fragmented and on the buffer
  // but leaving this alone for now
  if (!deletedChunk->getBuffer()->hasEncoder()) {
    deletedChunk->initEncoder();
  }
  deletedChunk->getBuffer()->getEncoder()->updateStats(static_cast<int64_t>(true), false);

  if (fragment.shadowNumTuples > deletedChunk->getBuffer()->getEncoder()->getNumElems()) {
    // An append to the same fragment will increase shadowNumTuples.
    // Update NumElems in this case. Otherwise, use existing NumElems.
    deletedChunk->getBuffer()->getEncoder()->setNumElems(fragment.shadowNumTuples);
  }
  deletedChunk->getBuffer()->setUpdated();
}

namespace {
inline void update_metadata(SQLTypeInfo const& ti,
                            ChunkUpdateStats& update_stats,
                            int64_t const updated_val,
                            int64_t const old_val,
                            NullSentinelSupplier s = NullSentinelSupplier()) {
  if (ti.get_notnull()) {
    set_minmax(update_stats.new_values_stats.min_int64t,
               update_stats.new_values_stats.max_int64t,
               updated_val);
    set_minmax(update_stats.old_values_stats.min_int64t,
               update_stats.old_values_stats.max_int64t,
               old_val);
  } else {
    set_minmax(update_stats.new_values_stats.min_int64t,
               update_stats.new_values_stats.max_int64t,
               update_stats.new_values_stats.has_null,
               updated_val,
               s(ti, updated_val));
    set_minmax(update_stats.old_values_stats.min_int64t,
               update_stats.old_values_stats.max_int64t,
               update_stats.old_values_stats.has_null,
               old_val,
               s(ti, old_val));
  }
}

inline void update_metadata(SQLTypeInfo const& ti,
                            ChunkUpdateStats& update_stats,
                            double const updated_val,
                            double const old_val,
                            NullSentinelSupplier s = NullSentinelSupplier()) {
  if (ti.get_notnull()) {
    set_minmax(update_stats.new_values_stats.min_double,
               update_stats.new_values_stats.max_double,
               updated_val);
    set_minmax(update_stats.old_values_stats.min_double,
               update_stats.old_values_stats.max_double,
               old_val);
  } else {
    set_minmax(update_stats.new_values_stats.min_double,
               update_stats.new_values_stats.max_double,
               update_stats.new_values_stats.has_null,
               updated_val,
               s(ti, updated_val));
    set_minmax(update_stats.old_values_stats.min_double,
               update_stats.old_values_stats.max_double,
               update_stats.old_values_stats.has_null,
               old_val,
               s(ti, old_val));
  }
}

inline void update_metadata(UpdateValuesStats& agg_stats,
                            const UpdateValuesStats& new_stats) {
  agg_stats.has_null = agg_stats.has_null || new_stats.has_null;
  agg_stats.max_double = std::max<double>(agg_stats.max_double, new_stats.max_double);
  agg_stats.min_double = std::min<double>(agg_stats.min_double, new_stats.min_double);
  agg_stats.max_int64t = std::max<int64_t>(agg_stats.max_int64t, new_stats.max_int64t);
  agg_stats.min_int64t = std::min<int64_t>(agg_stats.min_int64t, new_stats.min_int64t);
}
}  // namespace

std::optional<ChunkUpdateStats> InsertOrderFragmenter::updateColumn(
    const Catalog_Namespace::Catalog* catalog,
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
    return {};
  }
  CHECK(nrow == n_rhs_values || 1 == n_rhs_values);

  auto fragment_ptr = getFragmentInfo(fragment_id);
  auto& fragment = *fragment_ptr;
  auto chunk_meta_it = fragment.getChunkMetadataMapPhysical().find(cd->columnId);
  CHECK(chunk_meta_it != fragment.getChunkMetadataMapPhysical().end());
  ChunkKey chunk_key{
      catalog->getCurrentDB().dbId, td->tableId, cd->columnId, fragment.fragmentId};
  auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                         &catalog->getDataMgr(),
                                         chunk_key,
                                         Data_Namespace::CPU_LEVEL,
                                         0,
                                         chunk_meta_it->second->numBytes,
                                         chunk_meta_it->second->numElements);

  std::vector<ChunkUpdateStats> update_stats_per_thread(ncore);

  // parallel update elements
  std::vector<std::future<void>> threads;

  const auto segsz = (nrow + ncore - 1) / ncore;
  auto dbuf = chunk->getBuffer();
  auto dbuf_addr = dbuf->getMemoryPtr();
  dbuf->setUpdated();
  updel_roll.addDirtyChunk(chunk, fragment.fragmentId);
  for (size_t rbegin = 0, c = 0; rbegin < nrow; ++c, rbegin += segsz) {
    threads.emplace_back(std::async(
        std::launch::async, [=, &update_stats_per_thread, &frag_offsets, &rhs_values] {
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
          DateDaysOverflowValidator dateDaysOverflowValidator(lhs_type);
          NullAwareValidator<DateDaysOverflowValidator> nullAwareDateOverflowValidator(
              lhs_type, &dateDaysOverflowValidator);

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
                throw std::runtime_error("UPDATE does not support cast to string.");
              }
              int64_t old_val;
              get_scalar<int64_t>(data_ptr, lhs_type, old_val);
              // Handle special case where date column with date in days encoding stores
              // metadata in epoch seconds.
              if (lhs_type.is_date_in_days()) {
                old_val = DateConverters::get_epoch_seconds_from_days(old_val);
              }
              put_scalar<int64_t>(data_ptr, lhs_type, v, cd->columnName, &rhs_type);
              if (lhs_type.is_decimal()) {
                nullAwareDecimalOverflowValidator.validate<int64_t>(v);
                int64_t decimal_val;
                get_scalar<int64_t>(data_ptr, lhs_type, decimal_val);
                int64_t target_value = (v == inline_int_null_value<int64_t>() &&
                                        lhs_type.get_notnull() == false)
                                           ? v
                                           : decimal_val;
                update_metadata(
                    lhs_type, update_stats_per_thread[c], target_value, old_val);
                auto const positive_v_and_negative_d = (v >= 0) && (decimal_val < 0);
                auto const negative_v_and_positive_d = (v < 0) && (decimal_val >= 0);
                if (positive_v_and_negative_d || negative_v_and_positive_d) {
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
                  if (lhs_type.get_size() == 2) {
                    nullAwareDateOverflowValidator.validate<int16_t>(v);
                  } else {
                    nullAwareDateOverflowValidator.validate<int32_t>(v);
                  }
                  int64_t days;
                  get_scalar<int64_t>(data_ptr, lhs_type, days);
                  const auto seconds = DateConverters::get_epoch_seconds_from_days(days);
                  int64_t target_value = (v == inline_int_null_value<int64_t>() &&
                                          lhs_type.get_notnull() == false)
                                             ? NullSentinelSupplier()(lhs_type, v)
                                             : seconds;
                  update_metadata(
                      lhs_type, update_stats_per_thread[c], target_value, old_val);
                } else {
                  int64_t target_value;
                  if (rhs_type.is_decimal()) {
                    target_value = round(decimal_to_double(rhs_type, v));
                  } else {
                    target_value = v;
                  }
                  update_metadata(
                      lhs_type, update_stats_per_thread[c], target_value, old_val);
                }
              } else {
                if (rhs_type.is_decimal()) {
                  update_metadata(lhs_type,
                                  update_stats_per_thread[c],
                                  decimal_to_double(rhs_type, v),
                                  double(old_val));
                } else {
                  update_metadata(lhs_type, update_stats_per_thread[c], v, old_val);
                }
              }
            } else if (const auto vp = boost::get<double>(sv)) {
              auto v = *vp;
              if (lhs_type.is_string()) {
                throw std::runtime_error("UPDATE does not support cast to string.");
              }
              double old_val;
              get_scalar<double>(data_ptr, lhs_type, old_val);
              put_scalar<double>(data_ptr, lhs_type, v, cd->columnName);
              if (lhs_type.is_integer()) {
                update_metadata(
                    lhs_type, update_stats_per_thread[c], int64_t(v), int64_t(old_val));
              } else if (lhs_type.is_fp()) {
                update_metadata(
                    lhs_type, update_stats_per_thread[c], double(v), double(old_val));
              } else {
                UNREACHABLE() << "Unexpected combination of a non-floating or integer "
                                 "LHS with a floating RHS.";
              }
            } else if (const auto vp = boost::get<float>(sv)) {
              auto v = *vp;
              if (lhs_type.is_string()) {
                throw std::runtime_error("UPDATE does not support cast to string.");
              }
              float old_val;
              get_scalar<float>(data_ptr, lhs_type, old_val);
              put_scalar<float>(data_ptr, lhs_type, v, cd->columnName);
              if (lhs_type.is_integer()) {
                update_metadata(
                    lhs_type, update_stats_per_thread[c], int64_t(v), int64_t(old_val));
              } else {
                update_metadata(lhs_type, update_stats_per_thread[c], double(v), old_val);
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
                int64_t old_val;
                get_scalar<int64_t>(data_ptr, lhs_type, old_val);
                put_scalar<int64_t>(data_ptr, lhs_type, sidx, cd->columnName);
                update_metadata(
                    lhs_type, update_stats_per_thread[c], int64_t(sidx), old_val);
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
                  double old_val;
                  get_scalar<double>(data_ptr, lhs_type, old_val);
                  put_scalar<double>(data_ptr, lhs_type, dval, cd->columnName);
                  update_metadata(
                      lhs_type, update_stats_per_thread[c], double(dval), old_val);
                } else {
                  int64_t old_val;
                  get_scalar<int64_t>(data_ptr, lhs_type, old_val);
                  put_scalar<int64_t>(data_ptr, lhs_type, dval, cd->columnName);
                  update_metadata(
                      lhs_type, update_stats_per_thread[c], int64_t(dval), old_val);
                }
              } else {
                put_null(data_ptr, lhs_type, cd->columnName);
                update_stats_per_thread[c].new_values_stats.has_null = true;
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
        return {};
      }
    }
  }
  ChunkUpdateStats update_stats;
  for (size_t c = 0; c < ncore; ++c) {
    update_metadata(update_stats.new_values_stats,
                    update_stats_per_thread[c].new_values_stats);
    update_metadata(update_stats.old_values_stats,
                    update_stats_per_thread[c].old_values_stats);
  }

  CHECK_GT(fragment.shadowNumTuples, size_t(0));
  updateColumnMetadata(
      cd, fragment, chunk, update_stats.new_values_stats, cd->columnType, updel_roll);
  update_stats.updated_rows_count = nrow;
  update_stats.fragment_rows_count = fragment.shadowNumTuples;
  update_stats.chunk = chunk;
  return update_stats;
}

void InsertOrderFragmenter::updateColumnMetadata(
    const ColumnDescriptor* cd,
    FragmentInfo& fragment,
    std::shared_ptr<Chunk_NS::Chunk> chunk,
    const UpdateValuesStats& new_values_stats,
    const SQLTypeInfo& rhs_type,
    UpdelRoll& updel_roll) {
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(fragmentInfoMutex_);
  auto buffer = chunk->getBuffer();
  const auto& lhs_type = cd->columnType;

  auto encoder = buffer->getEncoder();
  auto update_stats = [&encoder](auto min, auto max, auto has_null) {
    static_assert(std::is_same<decltype(min), decltype(max)>::value,
                  "Type mismatch on min/max");
    if (has_null) {
      encoder->updateStats(decltype(min)(), true);
    }
    if (max < min) {
      return;
    }
    encoder->updateStats(min, false);
    encoder->updateStats(max, false);
  };

  if (is_integral(lhs_type) || (lhs_type.is_decimal() && rhs_type.is_decimal())) {
    update_stats(new_values_stats.min_int64t,
                 new_values_stats.max_int64t,
                 new_values_stats.has_null);
  } else if (lhs_type.is_fp()) {
    update_stats(new_values_stats.min_double,
                 new_values_stats.max_double,
                 new_values_stats.has_null);
  } else if (lhs_type.is_decimal()) {
    update_stats((int64_t)(new_values_stats.min_double * pow(10, lhs_type.get_scale())),
                 (int64_t)(new_values_stats.max_double * pow(10, lhs_type.get_scale())),
                 new_values_stats.has_null);
  } else if (!lhs_type.is_array() && !lhs_type.is_geometry() &&
             !(lhs_type.is_string() && kENCODING_DICT != lhs_type.get_compression())) {
    update_stats(new_values_stats.min_int64t,
                 new_values_stats.max_int64t,
                 new_values_stats.has_null);
  }
  auto td = updel_roll.catalog->getMetadataForTable(cd->tableId);
  auto chunk_metadata =
      updel_roll.getChunkMetadata({td, &fragment}, cd->columnId, fragment);
  buffer->getEncoder()->getMetadata(chunk_metadata);
}

void InsertOrderFragmenter::updateMetadata(const Catalog_Namespace::Catalog* catalog,
                                           const MetaDataKey& key,
                                           UpdelRoll& updel_roll) {
  heavyai::unique_lock<heavyai::shared_mutex> writeLock(fragmentInfoMutex_);
  const auto chunk_metadata_map = updel_roll.getChunkMetadataMap(key);
  auto& fragmentInfo = *key.second;
  fragmentInfo.setChunkMetadataMap(chunk_metadata_map);
  fragmentInfo.shadowChunkMetadataMap = fragmentInfo.getChunkMetadataMapPhysicalCopy();
  fragmentInfo.shadowNumTuples = updel_roll.getNumTuple(key);
  fragmentInfo.setPhysicalNumTuples(fragmentInfo.shadowNumTuples);
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
                                               chunk_meta_it->second->numBytes,
                                               chunk_meta_it->second->numElements);
        chunks.push_back(chunk);
      }
    }
  }
  return chunks;
}

// get a sorted vector of offsets of rows to vacuum
const std::vector<uint64_t> InsertOrderFragmenter::getVacuumOffsets(
    const std::shared_ptr<Chunk_NS::Chunk>& chunk) {
  const auto data_buffer = chunk->getBuffer();
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
                            bool& has_null,
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
  auto cd = chunk->getColumnDesc();
  auto td = catalog->getMetadataForTable(cd->tableId);
  auto data_buffer = chunk->getBuffer();
  auto chunkMetadata =
      updel_roll.getChunkMetadata({td, &fragment}, cd->columnId, fragment);
  chunkMetadata->numElements = nrows_to_keep;
  chunkMetadata->numBytes = data_buffer->size();
  updel_roll.addDirtyChunk(chunk, fragment.fragmentId);
}

auto InsertOrderFragmenter::vacuum_fixlen_rows(
    const FragmentInfo& fragment,
    const std::shared_ptr<Chunk_NS::Chunk>& chunk,
    const std::vector<uint64_t>& frag_offsets) {
  const auto cd = chunk->getColumnDesc();
  const auto& col_type = cd->columnType;
  auto data_buffer = chunk->getBuffer();
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

// Gets the initial padding required for the chunk buffer. For variable length array
// columns, if the first element after vacuuming is going to be a null array, a padding
// with a value that is greater than 0 is expected.
size_t get_null_padding(bool is_varlen_array,
                        const std::vector<uint64_t>& frag_offsets,
                        const StringOffsetT* index_array,
                        size_t fragment_row_count) {
  if (is_varlen_array) {
    size_t first_non_deleted_row_index{0};
    for (auto deleted_offset : frag_offsets) {
      if (first_non_deleted_row_index < deleted_offset) {
        break;
      } else {
        first_non_deleted_row_index++;
      }
    }
    CHECK_LT(first_non_deleted_row_index, fragment_row_count);
    if (first_non_deleted_row_index == 0) {
      // If the first row in the fragment is not deleted, then the first offset in the
      // index buffer/array already contains expected padding.
      return index_array[0];
    } else {
      // If the first non-deleted element is a null array (indentified by a negative
      // offset), get a padding value for the chunk buffer.
      if (index_array[first_non_deleted_row_index + 1] < 0) {
        size_t first_non_zero_offset{0};
        for (size_t i = 0; i <= first_non_deleted_row_index; i++) {
          if (index_array[i] != 0) {
            first_non_zero_offset = index_array[i];
            break;
          }
        }
        CHECK_GT(first_non_zero_offset, static_cast<size_t>(0));
        return std::min(ArrayNoneEncoder::DEFAULT_NULL_PADDING_SIZE,
                        first_non_zero_offset);
      } else {
        return 0;
      }
    }
  } else {
    return 0;
  }
}

// Gets the indexes of variable length null arrays in the chunk after vacuuming.
std::set<size_t> get_var_len_null_array_indexes(const SQLTypeInfo sql_type_info,
                                                const std::vector<uint64_t>& frag_offsets,
                                                const StringOffsetT* index_array,
                                                size_t fragment_row_count) {
  std::set<size_t> null_array_indexes;
  if (sql_type_info.is_varlen_array() && !sql_type_info.get_notnull()) {
    size_t frag_offset_index{0};
    size_t vacuum_offset{0};
    for (size_t i = 0; i < fragment_row_count; i++) {
      if (frag_offset_index < frag_offsets.size() &&
          i == frag_offsets[frag_offset_index]) {
        frag_offset_index++;
        vacuum_offset++;
      } else if (index_array[i + 1] < 0) {
        null_array_indexes.emplace(i - vacuum_offset);
      }
    }
  }
  return null_array_indexes;
}

StringOffsetT get_buffer_offset(bool is_varlen_array,
                                const StringOffsetT* index_array,
                                size_t index) {
  auto offset = index_array[index];
  if (offset < 0) {
    // Variable length arrays encode null arrays as negative offsets
    CHECK(is_varlen_array);
    offset = -offset;
  }
  return offset;
}

auto InsertOrderFragmenter::vacuum_varlen_rows(
    const FragmentInfo& fragment,
    const std::shared_ptr<Chunk_NS::Chunk>& chunk,
    const std::vector<uint64_t>& frag_offsets) {
  auto is_varlen_array = chunk->getColumnDesc()->columnType.is_varlen_array();
  auto data_buffer = chunk->getBuffer();
  CHECK(data_buffer);
  auto index_buffer = chunk->getIndexBuf();
  CHECK(index_buffer);
  auto data_addr = data_buffer->getMemoryPtr();
  auto indices_addr = index_buffer->getMemoryPtr();
  CHECK(indices_addr);
  auto index_array = (StringOffsetT*)indices_addr;
  int64_t irow_of_blk_to_keep = 0;  // head of next row block to keep
  int64_t irow_of_blk_to_fill = 0;  // row offset to fit the kept block
  size_t nbytes_fix_data_to_keep = 0;
  auto nrows_in_fragment = fragment.getPhysicalNumTuples();
  size_t null_padding =
      get_null_padding(is_varlen_array, frag_offsets, index_array, nrows_in_fragment);
  size_t nbytes_var_data_to_keep = null_padding;
  auto null_array_indexes = get_var_len_null_array_indexes(
      chunk->getColumnDesc()->columnType, frag_offsets, index_array, nrows_in_fragment);
  auto nrows_to_vacuum = frag_offsets.size();
  for (size_t irow = 0; irow <= nrows_to_vacuum; irow++) {
    auto is_last_one = irow == nrows_to_vacuum;
    auto irow_to_vacuum = is_last_one ? nrows_in_fragment : frag_offsets[irow];
    auto maddr_to_vacuum = data_addr;
    int64_t nrows_to_keep = irow_to_vacuum - irow_of_blk_to_keep;
    if (nrows_to_keep > 0) {
      auto ibyte_var_data_to_keep = nbytes_var_data_to_keep;
      auto deleted_row_start_offset =
          get_buffer_offset(is_varlen_array, index_array, irow_to_vacuum);
      auto kept_row_start_offset =
          get_buffer_offset(is_varlen_array, index_array, irow_of_blk_to_keep);
      auto nbytes_to_keep =
          (is_last_one ? data_buffer->size() : deleted_row_start_offset) -
          kept_row_start_offset;
      if (irow_of_blk_to_fill != irow_of_blk_to_keep) {
        if (nbytes_to_keep > 0) {
          CHECK(data_addr);
          // move curr varlen row block toward front
          memmove(data_addr + ibyte_var_data_to_keep,
                  data_addr + kept_row_start_offset,
                  nbytes_to_keep);
        }

        const auto base_offset = kept_row_start_offset;
        for (int64_t i = 0; i < nrows_to_keep; ++i) {
          auto update_index = irow_of_blk_to_keep + i;
          auto offset = get_buffer_offset(is_varlen_array, index_array, update_index);
          index_array[update_index] = ibyte_var_data_to_keep + (offset - base_offset);
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

  // Set expected null padding, last offset, and negative values for null array offsets.
  index_array[0] = null_padding;
  auto post_vacuum_row_count = nrows_in_fragment - nrows_to_vacuum;
  index_array[post_vacuum_row_count] = nbytes_var_data_to_keep;
  if (!is_varlen_array) {
    CHECK(null_array_indexes.empty());
  }
  for (auto index : null_array_indexes) {
    index_array[index + 1] = -1 * std::abs(index_array[index + 1]);
  }
  return nbytes_var_data_to_keep;
}

void InsertOrderFragmenter::compactRows(const Catalog_Namespace::Catalog* catalog,
                                        const TableDescriptor* td,
                                        const int fragment_id,
                                        const std::vector<uint64_t>& frag_offsets,
                                        const Data_Namespace::MemoryLevel memory_level,
                                        UpdelRoll& updel_roll) {
  auto fragment_ptr = getFragmentInfo(fragment_id);
  auto& fragment = *fragment_ptr;
  auto chunks = getChunksForAllColumns(td, fragment, memory_level);
  const auto ncol = chunks.size();

  std::vector<ChunkUpdateStats> update_stats_per_thread(ncol);

  // parallel delete columns
  std::vector<std::future<void>> threads;
  auto nrows_to_vacuum = frag_offsets.size();
  auto nrows_in_fragment = fragment.getPhysicalNumTuples();
  auto nrows_to_keep = nrows_in_fragment - nrows_to_vacuum;

  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    auto chunk = chunks[ci];
    const auto cd = chunk->getColumnDesc();
    const auto& col_type = cd->columnType;
    auto data_buffer = chunk->getBuffer();
    auto index_buffer = chunk->getIndexBuf();
    auto data_addr = data_buffer->getMemoryPtr();
    auto indices_addr = index_buffer ? index_buffer->getMemoryPtr() : nullptr;
    auto index_array = (StringOffsetT*)indices_addr;
    bool is_varlen = col_type.is_varlen_indeed();

    auto fixlen_vacuum =
        [=, &update_stats_per_thread, &updel_roll, &frag_offsets, &fragment] {
          size_t nbytes_fix_data_to_keep;
          if (nrows_to_keep == 0) {
            nbytes_fix_data_to_keep = 0;
          } else {
            nbytes_fix_data_to_keep = vacuum_fixlen_rows(fragment, chunk, frag_offsets);
          }

          data_buffer->getEncoder()->setNumElems(nrows_to_keep);
          data_buffer->setSize(nbytes_fix_data_to_keep);
          data_buffer->setUpdated();

          set_chunk_metadata(catalog, fragment, chunk, nrows_to_keep, updel_roll);

          auto daddr = data_addr;
          auto element_size = col_type.is_fixlen_array() ? col_type.get_size()
                                                         : get_element_size(col_type);
          data_buffer->getEncoder()->resetChunkStats();
          for (size_t irow = 0; irow < nrows_to_keep; ++irow, daddr += element_size) {
            if (col_type.is_fixlen_array()) {
              auto encoder =
                  dynamic_cast<FixedLengthArrayNoneEncoder*>(data_buffer->getEncoder());
              CHECK(encoder);
              encoder->updateMetadata((int8_t*)daddr);
            } else if (col_type.is_fp()) {
              set_chunk_stats(col_type,
                              daddr,
                              update_stats_per_thread[ci].new_values_stats.has_null,
                              update_stats_per_thread[ci].new_values_stats.min_double,
                              update_stats_per_thread[ci].new_values_stats.max_double);
            } else {
              set_chunk_stats(col_type,
                              daddr,
                              update_stats_per_thread[ci].new_values_stats.has_null,
                              update_stats_per_thread[ci].new_values_stats.min_int64t,
                              update_stats_per_thread[ci].new_values_stats.max_int64t);
            }
          }
        };

    auto varlen_vacuum = [=, &updel_roll, &frag_offsets, &fragment] {
      size_t nbytes_var_data_to_keep;
      if (nrows_to_keep == 0) {
        nbytes_var_data_to_keep = 0;
      } else {
        nbytes_var_data_to_keep = vacuum_varlen_rows(fragment, chunk, frag_offsets);
      }

      data_buffer->getEncoder()->setNumElems(nrows_to_keep);
      data_buffer->setSize(nbytes_var_data_to_keep);
      data_buffer->setUpdated();

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

  updel_roll.setNumTuple({td, &fragment}, nrows_to_keep);
  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    auto chunk = chunks[ci];
    auto cd = chunk->getColumnDesc();
    if (!cd->columnType.is_fixlen_array()) {
      // For DATE_IN_DAYS encoded columns, data is stored in days but the metadata is
      // stored in seconds. Do the metadata conversion here before updating the chunk
      // stats.
      if (cd->columnType.is_date_in_days()) {
        auto& stats = update_stats_per_thread[ci].new_values_stats;
        stats.min_int64t = DateConverters::get_epoch_seconds_from_days(stats.min_int64t);
        stats.max_int64t = DateConverters::get_epoch_seconds_from_days(stats.max_int64t);
      }
      updateColumnMetadata(cd,
                           fragment,
                           chunk,
                           update_stats_per_thread[ci].new_values_stats,
                           cd->columnType,
                           updel_roll);
    }
  }
}

}  // namespace Fragmenter_Namespace

bool UpdelRoll::commitUpdate() {
  if (nullptr == catalog) {
    return false;
  }
  const auto td = catalog->getMetadataForTable(logicalTableId);
  CHECK(td);
  ChunkKey chunk_key{catalog->getDatabaseId(), td->tableId};
  const auto table_lock = lockmgr::TableDataLockMgr::getWriteLockForTable(chunk_key);

  // Checkpoint all shards. Otherwise, epochs can go out of sync.
  if (td->persistenceLevel == Data_Namespace::MemoryLevel::DISK_LEVEL) {
    auto table_epochs = catalog->getTableEpochs(catalog->getDatabaseId(), logicalTableId);
    try {
      // `checkpointWithAutoRollback` is not called here because, if a failure occurs,
      // `dirtyChunks` has to be cleared before resetting epochs
      catalog->checkpoint(logicalTableId);
    } catch (...) {
      dirty_chunks.clear();
      catalog->setTableEpochsLogExceptions(catalog->getDatabaseId(), table_epochs);
      throw;
    }
  }
  updateFragmenterAndCleanupChunks();
  return true;
}

void UpdelRoll::stageUpdate() {
  CHECK(catalog);
  auto db_id = catalog->getDatabaseId();
  CHECK(table_descriptor);
  auto table_id = table_descriptor->tableId;
  CHECK_EQ(memoryLevel, Data_Namespace::MemoryLevel::CPU_LEVEL);
  CHECK_EQ(table_descriptor->persistenceLevel, Data_Namespace::MemoryLevel::DISK_LEVEL);
  try {
    catalog->getDataMgr().checkpoint(db_id, table_id, memoryLevel);
  } catch (...) {
    dirty_chunks.clear();
    throw;
  }
  updateFragmenterAndCleanupChunks();
}

void UpdelRoll::updateFragmenterAndCleanupChunks() {
  // for each dirty fragment
  for (auto& cm : chunk_metadata_map_per_fragment) {
    cm.first.first->fragmenter->updateMetadata(catalog, cm.first, *this);
  }

  // flush gpu dirty chunks if update was not on gpu
  if (memoryLevel != Data_Namespace::MemoryLevel::GPU_LEVEL) {
    for (const auto& [chunk_key, chunk] : dirty_chunks) {
      catalog->getDataMgr().deleteChunksWithPrefix(
          chunk_key, Data_Namespace::MemoryLevel::GPU_LEVEL);
    }
  }
  dirty_chunks.clear();
}

void UpdelRoll::cancelUpdate() {
  if (nullptr == catalog) {
    return;
  }

  // TODO: needed?
  ChunkKey chunk_key{catalog->getDatabaseId(), logicalTableId};
  const auto table_lock = lockmgr::TableDataLockMgr::getWriteLockForTable(chunk_key);
  if (is_varlen_update) {
    int databaseId = catalog->getDatabaseId();
    auto table_epochs = catalog->getTableEpochs(databaseId, logicalTableId);

    dirty_chunks.clear();
    catalog->setTableEpochs(databaseId, table_epochs);
  } else {
    const auto td = catalog->getMetadataForTable(logicalTableId);
    CHECK(td);
    if (td->persistenceLevel != memoryLevel) {
      for (const auto& [chunk_key, chunk] : dirty_chunks) {
        catalog->getDataMgr().free(chunk->getBuffer());
        chunk->setBuffer(nullptr);
      }
    }
  }
}

void UpdelRoll::addDirtyChunk(std::shared_ptr<Chunk_NS::Chunk> chunk,
                              int32_t fragment_id) {
  heavyai::unique_lock<heavyai::shared_mutex> lock(chunk_update_tracker_mutex);
  CHECK(catalog);
  ChunkKey chunk_key{catalog->getDatabaseId(),
                     chunk->getColumnDesc()->tableId,
                     chunk->getColumnDesc()->columnId,
                     fragment_id};
  dirty_chunks[chunk_key] = chunk;
}

void UpdelRoll::initializeUnsetMetadata(
    const TableDescriptor* td,
    Fragmenter_Namespace::FragmentInfo& fragment_info) {
  heavyai::unique_lock<heavyai::shared_mutex> lock(chunk_update_tracker_mutex);
  MetaDataKey key{td, &fragment_info};
  if (chunk_metadata_map_per_fragment.count(key) == 0) {
    chunk_metadata_map_per_fragment[key] =
        fragment_info.getChunkMetadataMapPhysicalCopy();
  }
  if (num_tuples.count(key) == 0) {
    num_tuples[key] = fragment_info.shadowNumTuples;
  }
}

std::shared_ptr<ChunkMetadata> UpdelRoll::getChunkMetadata(
    const MetaDataKey& key,
    int32_t column_id,
    Fragmenter_Namespace::FragmentInfo& fragment_info) {
  initializeUnsetMetadata(key.first, fragment_info);
  heavyai::shared_lock<heavyai::shared_mutex> lock(chunk_update_tracker_mutex);
  auto metadata_map_it = chunk_metadata_map_per_fragment.find(key);
  CHECK(metadata_map_it != chunk_metadata_map_per_fragment.end());
  auto chunk_metadata_it = metadata_map_it->second.find(column_id);
  CHECK(chunk_metadata_it != metadata_map_it->second.end());
  return chunk_metadata_it->second;
}

ChunkMetadataMap UpdelRoll::getChunkMetadataMap(const MetaDataKey& key) const {
  heavyai::shared_lock<heavyai::shared_mutex> lock(chunk_update_tracker_mutex);
  auto metadata_map_it = chunk_metadata_map_per_fragment.find(key);
  CHECK(metadata_map_it != chunk_metadata_map_per_fragment.end());
  return metadata_map_it->second;
}

size_t UpdelRoll::getNumTuple(const MetaDataKey& key) const {
  heavyai::shared_lock<heavyai::shared_mutex> lock(chunk_update_tracker_mutex);
  auto it = num_tuples.find(key);
  CHECK(it != num_tuples.end());
  return it->second;
}

void UpdelRoll::setNumTuple(const MetaDataKey& key, size_t num_tuple) {
  heavyai::unique_lock<heavyai::shared_mutex> lock(chunk_update_tracker_mutex);
  num_tuples[key] = num_tuple;
}
