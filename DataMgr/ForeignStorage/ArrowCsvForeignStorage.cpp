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

#include "ArrowCsvForeignStorage.h"

#include <arrow/api.h>
#include <arrow/csv/reader.h>
#include <arrow/io/file.h>
#include <arrow/util/decimal.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <array>
#include <future>

#include "Catalog/DataframeTableDescriptor.h"
#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"
#include "DataMgr/StringNoneEncoder.h"
#include "QueryEngine/ArrowResultSet.h"
#include "Shared/ArrowUtil.h"
#include "Shared/Logger.h"
#include "Shared/measure.h"

struct Frag {
  int first_chunk;         // index of the first chunk assigned to the fragment
  int first_chunk_offset;  // offset from the begining of the first chunk
  int last_chunk;          // index of the last chunk
  int last_chunk_size;     // number of elements in the last chunk
};

class ArrowCsvForeignStorage : public PersistentForeignStorageInterface {
 public:
  ArrowCsvForeignStorage() {}

  void append(const std::vector<ForeignStorageColumnBuffer>& column_buffers) override;

  void read(const ChunkKey& chunk_key,
            const SQLTypeInfo& sql_type,
            int8_t* dest,
            const size_t numBytes) override;

  void prepareTable(const int db_id,
                    const std::string& type,
                    TableDescriptor& td,
                    std::list<ColumnDescriptor>& cols) override;
  void registerTable(Catalog_Namespace::Catalog* catalog,
                     std::pair<int, int> table_key,
                     const std::string& type,
                     const TableDescriptor& td,
                     const std::list<ColumnDescriptor>& cols,
                     Data_Namespace::AbstractBufferMgr* mgr) override;

  std::string getType() const override;

  struct ArrowFragment {
    int64_t offset;
    int64_t sz;
    std::vector<std::shared_ptr<arrow::ArrayData>> chunks;
  };

  void createDictionaryEncodedColumn(StringDictionary* dict,
                                     const ColumnDescriptor& c,
                                     std::vector<ArrowFragment>& col,
                                     arrow::ChunkedArray* arr_col_chunked_array,
                                     tbb::task_group& tg,
                                     const std::vector<Frag>& fragments,
                                     ChunkKey key,
                                     Data_Namespace::AbstractBufferMgr* mgr);

  template <typename T, typename ChunkType>
  void createDecimalColumn(const ColumnDescriptor& c,
                           std::vector<ArrowFragment>& col,
                           arrow::ChunkedArray* arr_col_chunked_array,
                           tbb::task_group& tg,
                           const std::vector<Frag>& fragments,
                           ChunkKey key,
                           Data_Namespace::AbstractBufferMgr* mgr);

  std::map<std::array<int, 3>, std::vector<ArrowFragment>> m_columns;
};

void registerArrowCsvForeignStorage(void) {
  ForeignStorageInterface::registerPersistentStorageInterface(
      std::make_unique<ArrowCsvForeignStorage>());
}

void ArrowCsvForeignStorage::append(
    const std::vector<ForeignStorageColumnBuffer>& column_buffers) {
  CHECK(false);
}

void ArrowCsvForeignStorage::read(const ChunkKey& chunk_key,
                                  const SQLTypeInfo& sql_type,
                                  int8_t* dest,
                                  const size_t numBytes) {
  std::array<int, 3> col_key{chunk_key[0], chunk_key[1], chunk_key[2]};
  auto& frag = m_columns.at(col_key).at(chunk_key[3]);

  CHECK(!frag.chunks.empty() || !chunk_key[3]);
  int64_t sz = 0, copied = 0;
  int varlen_offset = 0;
  size_t read_size = 0;
  for (size_t i = 0; i < frag.chunks.size(); i++) {
    auto& array_data = frag.chunks[i];
    int offset = (i == 0) ? frag.offset : 0;
    size_t size = (i == frag.chunks.size() - 1) ? (frag.sz - read_size)
                                                : (array_data->length - offset);
    read_size += size;
    arrow::Buffer* bp = nullptr;
    if (sql_type.is_dict_encoded_string()) {
      // array_data->buffers[1] stores dictionary indexes
      bp = array_data->buffers[1].get();
    } else if (sql_type.get_type() == kTEXT) {
      CHECK_GE(array_data->buffers.size(), 3UL);
      // array_data->buffers[2] stores string array
      bp = array_data->buffers[2].get();
    } else if (array_data->null_count != array_data->length) {
      // any type except strings (none encoded strings offsets go here as well)
      CHECK_GE(array_data->buffers.size(), 2UL);
      bp = array_data->buffers[1].get();
    }
    if (bp) {
      // offset buffer for none encoded strings need to be merged
      if (chunk_key.size() == 5 && chunk_key[4] == 2) {
        auto data = reinterpret_cast<const uint32_t*>(bp->data()) + offset;
        auto dest_ui32 = reinterpret_cast<uint32_t*>(dest);
        // as size contains count of string in chunk slice it would always be one less
        // then offsets array size
        sz = (size + 1) * sizeof(uint32_t);
        if (sz > 0) {
          if (i != 0) {
            // We merge arrow chunks with string offsets into a single contigous fragment.
            // Each string is represented by a pair of offsets, thus size of offset table
            // is num strings + 1. When merging two chunks, the last number in the first
            // chunk duplicates the first number in the second chunk, so we skip it.
            data++;
            sz -= sizeof(uint32_t);
          } else {
            // As we support cases when fragment starts with offset of arrow chunk we need
            // to substract the first element of the first chunk from all elements in that
            // fragment
            varlen_offset -= data[0];
          }
          // We also re-calculate offsets in the second chunk as it is a continuation of
          // the first one.
          std::transform(data,
                         data + (sz / sizeof(uint32_t)),
                         dest_ui32,
                         [varlen_offset](uint32_t val) { return val + varlen_offset; });
          varlen_offset += data[(sz / sizeof(uint32_t)) - 1];
        }
      } else {
        auto fixed_type = dynamic_cast<arrow::FixedWidthType*>(array_data->type.get());
        if (fixed_type) {
          std::memcpy(
              dest,
              bp->data() + (array_data->offset + offset) * (fixed_type->bit_width() / 8),
              sz = size * (fixed_type->bit_width() / 8));
        } else {
          auto offsets_buffer =
              reinterpret_cast<const uint32_t*>(array_data->buffers[1]->data());
          auto string_buffer_offset = offsets_buffer[offset + array_data->offset];
          auto string_buffer_size =
              offsets_buffer[offset + array_data->offset + size] - string_buffer_offset;
          std::memcpy(dest, bp->data() + string_buffer_offset, sz = string_buffer_size);
        }
      }
    } else {
      // TODO: nullify?
      auto fixed_type = dynamic_cast<arrow::FixedWidthType*>(array_data->type.get());
      if (fixed_type) {
        sz = size * (fixed_type->bit_width() / 8);
      } else {
        CHECK(false);  // TODO: what's else???
      }
    }
    dest += sz;
    copied += sz;
  }
  CHECK_EQ(numBytes, size_t(copied));
}

// TODO: this overlaps with getArrowType() from ArrowResultSetConverter.cpp but with few
// differences in kTEXT and kDATE
static std::shared_ptr<arrow::DataType> getArrowImportType(const SQLTypeInfo type) {
  using namespace arrow;
  auto ktype = type.get_type();
  if (IS_INTEGER(ktype)) {
    switch (type.get_size()) {
      case 1:
        return int8();
      case 2:
        return int16();
      case 4:
        return int32();
      case 8:
        return int64();
      default:
        CHECK(false);
    }
  }
  switch (ktype) {
    case kBOOLEAN:
      return boolean();
    case kFLOAT:
      return float32();
    case kDOUBLE:
      return float64();
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      return utf8();
    case kDECIMAL:
    case kNUMERIC:
      return decimal(type.get_precision(), type.get_scale());
    case kTIME:
      return time32(TimeUnit::SECOND);
    // case kDATE:
    // TODO(wamsi) : Remove date64() once date32() support is added in cuDF. date32()
    // Currently support for date32() is missing in cuDF.Hence, if client requests for
    // date on GPU, return date64() for the time being, till support is added.
    // return device_type_ == ExecutorDeviceType::GPU ? date64() : date32();
    case kTIMESTAMP:
      switch (type.get_precision()) {
        case 0:
          return timestamp(TimeUnit::SECOND);
        case 3:
          return timestamp(TimeUnit::MILLI);
        case 6:
          return timestamp(TimeUnit::MICRO);
        case 9:
          return timestamp(TimeUnit::NANO);
        default:
          throw std::runtime_error("Unsupported timestamp precision for Arrow: " +
                                   std::to_string(type.get_precision()));
      }
    case kARRAY:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    default:
      throw std::runtime_error(type.get_type_name() + " is not supported in Arrow.");
  }
  return nullptr;
}

void ArrowCsvForeignStorage::prepareTable(const int db_id,
                                          const std::string& type,
                                          TableDescriptor& td,
                                          std::list<ColumnDescriptor>& cols) {
  td.hasDeletedCol = false;
}

void getSizeAndOffset(const Frag& frag,
                      const std::shared_ptr<arrow::Array>& chunk,
                      size_t i,
                      int& size,
                      int& offset) {
  offset = (i == frag.first_chunk) ? frag.first_chunk_offset : 0;
  size = (i == frag.last_chunk) ? frag.last_chunk_size : (chunk->length() - offset);
}

void ArrowCsvForeignStorage::createDictionaryEncodedColumn(
    StringDictionary* dict,
    const ColumnDescriptor& c,
    std::vector<ArrowFragment>& col,
    arrow::ChunkedArray* arr_col_chunked_array,
    tbb::task_group& tg,
    const std::vector<Frag>& fragments,
    ChunkKey key,
    Data_Namespace::AbstractBufferMgr* mgr) {
  tg.run([dict, &c, &col, arr_col_chunked_array, &tg, &fragments, k = key, mgr]() {
    auto key = k;
    auto full_time = measure<>::execution([&]() {
      // calculate offsets for every fragment in bulk
      size_t bulk_size = 0;
      std::vector<int> offsets(fragments.size() + 1);
      for (size_t f = 0; f < fragments.size(); f++) {
        offsets[f] = bulk_size;
        for (int i = fragments[f].first_chunk, e = fragments[f].last_chunk; i <= e; i++) {
          int size, offset;
          getSizeAndOffset(
              fragments[f], arr_col_chunked_array->chunk(i), i, size, offset);
          bulk_size += size;
        }
      }
      offsets[fragments.size()] = bulk_size;
      std::vector<std::string_view> bulk(bulk_size);

      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, fragments.size()),
          [&bulk, &fragments, arr_col_chunked_array, &offsets](
              const tbb::blocked_range<size_t>& r) {
            for (auto f = r.begin(); f != r.end(); ++f) {
              auto bulk_offset = offsets[f];

              size_t current_ind = 0;
              for (int i = fragments[f].first_chunk, e = fragments[f].last_chunk; i <= e;
                   i++) {
                int size, offset;
                getSizeAndOffset(
                    fragments[f], arr_col_chunked_array->chunk(i), i, size, offset);

                auto stringArray = std::static_pointer_cast<arrow::StringArray>(
                    arr_col_chunked_array->chunk(i));
                for (int i = offset; i < offset + size; i++) {
                  auto view = stringArray->GetView(i);
                  bulk[bulk_offset + current_ind] =
                      std::string_view(view.data(), view.length());
                  current_ind++;
                }
              }
            }
          });

      std::shared_ptr<arrow::Buffer> indices_buf;
      ARROW_THROW_NOT_OK(
          arrow::AllocateBuffer(bulk_size * sizeof(int32_t), &indices_buf));
      auto raw_data = reinterpret_cast<int*>(indices_buf->mutable_data());
      auto time = measure<>::execution([&]() { dict->getOrAddBulk(bulk, raw_data); });

      VLOG(1) << "FSI dictionary for column created in: " << time
              << "ms, strings count: " << bulk_size
              << ", unique_count: " << dict->storageEntryCount();

      for (size_t f = 0; f < fragments.size(); f++) {
        auto bulk_offset = offsets[f];
        tg.run([k = key,
                f,
                &col,
                mgr,
                &c,
                arr_col_chunked_array,
                bulk_offset,
                indices_buf,
                &fragments]() {
          auto key = k;
          key[3] = f;
          auto& frag = col[f];
          frag.chunks.resize(fragments[f].last_chunk - fragments[f].first_chunk + 1);
          auto b = mgr->createBuffer(key);
          b->sql_type = c.columnType;
          b->encoder.reset(Encoder::Create(b, c.columnType));
          b->has_encoder = true;
          size_t current_ind = 0;
          for (int i = fragments[f].first_chunk, e = fragments[f].last_chunk; i <= e;
               i++) {
            int size, offset;
            getSizeAndOffset(
                fragments[f], arr_col_chunked_array->chunk(i), i, size, offset);
            auto indexArray = std::make_shared<arrow::Int32Array>(
                size, indices_buf, nullptr, -1, bulk_offset + current_ind);
            frag.chunks[i - fragments[f].first_chunk] = indexArray->data();
            frag.sz += size;
            current_ind += size;
            frag.offset = 0;
            auto len = frag.chunks[i - fragments[f].first_chunk]->length;
            auto data = frag.chunks[i - fragments[f].first_chunk]->GetValues<int32_t>(1);
            b->encoder->updateStats((const int8_t*)data, len);
          }

          b->setSize(frag.sz * b->sql_type.get_size());
          b->encoder->setNumElems(frag.sz);
        });
      }
    });
    VLOG(1) << "FSI: createDictionaryEncodedColumn time: " << full_time << "ms"
            << std::endl;
  });
}

template <typename T, typename ChunkType>
void ArrowCsvForeignStorage::createDecimalColumn(
    const ColumnDescriptor& c,
    std::vector<ArrowFragment>& col,
    arrow::ChunkedArray* arr_col_chunked_array,
    tbb::task_group& tg,
    const std::vector<Frag>& fragments,
    ChunkKey k,
    Data_Namespace::AbstractBufferMgr* mgr) {
  auto empty = arr_col_chunked_array->null_count() == arr_col_chunked_array->length();
  size_t column_size = 0;
  std::vector<int> offsets(fragments.size());
  for (size_t f = 0; f < fragments.size(); f++) {
    offsets[f] = column_size;
    auto& frag = col[f];
    for (int i = fragments[f].first_chunk, e = fragments[f].last_chunk; i <= e; i++) {
      int size, offset;
      getSizeAndOffset(fragments[f], arr_col_chunked_array->chunk(i), i, size, offset);
      // as we create new buffer, offsets are handled with arrow::ArrayData::offset
      frag.offset = 0;
      frag.sz += size;
    }
    column_size += frag.sz;
  }

  std::shared_ptr<arrow::Buffer> result_buffer;
  ARROW_THROW_NOT_OK(
      arrow::AllocateBuffer(column_size * c.columnType.get_size(), &result_buffer));

  T* buffer_data = reinterpret_cast<T*>(result_buffer->mutable_data());
  tbb::parallel_for(
      tbb::blocked_range(0UL, fragments.size()),
      [k,
       buffer_data,
       &offsets,
       &fragments,
       &col,
       arr_col_chunked_array,
       &result_buffer,
       mgr,
       &c,
       empty,
       &tg](auto& range) {
        auto key = k;
        for (size_t f = range.begin(); f < range.end(); f++) {
          T* fragment_data = buffer_data + offsets[f];
          size_t chunk_offset = 0;
          key[3] = f;
          auto& frag = col[f];
          frag.chunks.resize(fragments[f].last_chunk - fragments[f].first_chunk + 1);
          for (int i = fragments[f].first_chunk, e = fragments[f].last_chunk; i <= e;
               i++) {
            T* chunk_data = fragment_data + chunk_offset;
            int size, offset;
            getSizeAndOffset(
                fragments[f], arr_col_chunked_array->chunk(i), i, size, offset);

            auto decimalArray = std::static_pointer_cast<arrow::Decimal128Array>(
                arr_col_chunked_array->chunk(i));

            for (int j = 0; j < size; ++j) {
              if (empty || decimalArray->null_count() == decimalArray->length() ||
                  decimalArray->IsNull(j + offset)) {
                chunk_data[j] = inline_int_null_value<T>();
              } else {
                arrow::Decimal128 val(decimalArray->GetValue(j + offset));
                chunk_data[j] =
                    static_cast<int64_t>(val);  // arrow can cast only to int64_t
              }
            }

            auto converted_chunk = std::make_shared<ChunkType>(
                size, result_buffer, nullptr, -1, offsets[f] + chunk_offset);
            frag.chunks[i - fragments[f].first_chunk] = converted_chunk->data();

            chunk_offset += size;
          }

          auto b = mgr->createBuffer(key);
          b->sql_type = c.columnType;
          b->setSize(frag.sz * b->sql_type.get_size());
          b->encoder.reset(Encoder::Create(b, c.columnType));
          b->has_encoder = true;
          if (!empty) {
            tg.run([&frag, b]() {
              for (size_t i = 0; i < frag.chunks.size(); i++) {
                auto& chunk = frag.chunks[i];
                int offset = chunk->offset;
                size_t size = chunk->length;
                auto data = chunk->buffers[1]->data();
                b->encoder->updateStats(
                    (const int8_t*)data + offset * b->sql_type.get_size(), size);
              }
            });
          }
          b->encoder->setNumElems(frag.sz);
        }
      });
}

void ArrowCsvForeignStorage::registerTable(Catalog_Namespace::Catalog* catalog,
                                           std::pair<int, int> table_key,
                                           const std::string& info,
                                           const TableDescriptor& td,
                                           const std::list<ColumnDescriptor>& cols,
                                           Data_Namespace::AbstractBufferMgr* mgr) {
  const DataframeTableDescriptor* df_td =
      dynamic_cast<const DataframeTableDescriptor*>(&td);
  bool isDataframe = df_td ? true : false;
  if (!isDataframe) {
    df_td = new DataframeTableDescriptor(td);
  }
  auto memory_pool = arrow::default_memory_pool();
  auto arrow_parse_options = arrow::csv::ParseOptions::Defaults();
  arrow_parse_options.quoting = false;
  arrow_parse_options.escaping = false;
  arrow_parse_options.newlines_in_values = false;
  arrow_parse_options.delimiter = *df_td->delimiter.c_str();
  auto arrow_read_options = arrow::csv::ReadOptions::Defaults();
  arrow_read_options.use_threads = true;

  arrow_read_options.block_size = 20 * 1024 * 1024;
  arrow_read_options.autogenerate_column_names = false;
  arrow_read_options.skip_rows =
      df_td->hasHeader ? (df_td->skipRows + 1) : df_td->skipRows;

  auto arrow_convert_options = arrow::csv::ConvertOptions::Defaults();
  arrow_convert_options.check_utf8 = false;
  arrow_convert_options.include_columns = arrow_read_options.column_names;

  for (auto& c : cols) {
    if (c.isSystemCol) {
      continue;  // must be processed by base interface implementation
    }
    arrow_convert_options.column_types.emplace(c.columnName,
                                               getArrowImportType(c.columnType));
    arrow_read_options.column_names.push_back(c.columnName);
  }

  std::shared_ptr<arrow::io::ReadableFile> inp;
  auto file_result = arrow::io::ReadableFile::Open(info.c_str());
  ARROW_THROW_NOT_OK(file_result.status());
  inp = file_result.ValueOrDie();

  auto table_reader_result = arrow::csv::TableReader::Make(
      memory_pool, inp, arrow_read_options, arrow_parse_options, arrow_convert_options);
  ARROW_THROW_NOT_OK(table_reader_result.status());
  auto table_reader = table_reader_result.ValueOrDie();

  std::shared_ptr<arrow::Table> arrowTable;
  auto time = measure<>::execution([&]() {
    auto arrow_table_result = table_reader->Read();
    ARROW_THROW_NOT_OK(arrow_table_result.status());
    arrowTable = arrow_table_result.ValueOrDie();
  });

  VLOG(1) << "Read Arrow CSV file " << info << " in " << time << "ms";

  arrow::Table& table = *arrowTable.get();
  int cln = 0, num_cols = table.num_columns();
  int arr_frags = table.column(0)->num_chunks();
  arrow::ChunkedArray* c0p = table.column(0).get();

  // here we split arrow chunks between omnisci fragments

  std::vector<Frag> fragments;
  int64_t sz = 0;
  int64_t offset = 0;
  fragments.push_back({0, 0, 0, 0});

  for (int i = 0; i < arr_frags;) {
    auto& chunk = *c0p->chunk(i);
    auto& frag = *fragments.rbegin();
    if (df_td->maxFragRows - sz > chunk.length() - offset) {
      sz += chunk.length() - offset;
      if (i == arr_frags - 1) {
        fragments.rbegin()->last_chunk = arr_frags - 1;
        fragments.rbegin()->last_chunk_size =
            c0p->chunk(arr_frags - 1)->length() - offset;
      }
      offset = 0;
      i++;
    } else {
      frag.last_chunk = i;
      frag.last_chunk_size = df_td->maxFragRows - sz;
      offset += df_td->maxFragRows - sz;
      sz = 0;
      fragments.push_back({i, static_cast<int>(offset), 0, 0});
    }
  }
  if (fragments.rbegin()->first_chunk == fragments.rbegin()->first_chunk &&
      fragments.rbegin()->last_chunk_size == 0) {
    // remove empty fragment at the end if any
    fragments.pop_back();
  }
  // data comes like this - database_id, table_id, column_id, fragment_id
  ChunkKey key{table_key.first, table_key.second, 0, 0};
  std::array<int, 3> col_key{table_key.first, table_key.second, 0};

  tbb::task_group tg;

  for (auto& c : cols) {
    if (c.isSystemCol) {
      continue;  // must be processed by base interface implementation
    }

    if (cln >= num_cols) {
      LOG(ERROR) << "Number of columns read from Arrow (" << num_cols
                 << ") mismatch CREATE TABLE request: " << cols.size();
      break;
    }

    auto ctype = c.columnType.get_type();
    col_key[2] = key[2] = c.columnId;
    auto& col = m_columns[col_key];
    col.resize(fragments.size());
    auto arr_col_chunked_array = table.column(cln++).get();

    if (c.columnType.is_dict_encoded_string()) {
      auto dictDesc = const_cast<DictDescriptor*>(
          catalog->getMetadataForDict(c.columnType.get_comp_param()));
      StringDictionary* dict = dictDesc->stringDict.get();
      createDictionaryEncodedColumn(
          dict, c, col, arr_col_chunked_array, tg, fragments, key, mgr);
    } else if (ctype == kDECIMAL || ctype == kNUMERIC) {
      tg.run([this, &c, &col, arr_col_chunked_array, &tg, &fragments, key, mgr]() {
        switch (c.columnType.get_size()) {
          case 2:
            createDecimalColumn<int16_t, arrow::Int16Array>(
                c, col, arr_col_chunked_array, tg, fragments, key, mgr);
            break;
          case 4:
            createDecimalColumn<int32_t, arrow::Int32Array>(
                c, col, arr_col_chunked_array, tg, fragments, key, mgr);
            break;
          case 8:
            createDecimalColumn<int64_t, arrow::Int64Array>(
                c, col, arr_col_chunked_array, tg, fragments, key, mgr);
            break;
          default:
            // TODO: throw unsupported decimal type exception
            CHECK(false);
            break;
        }
      });
    } else {
      auto empty = arr_col_chunked_array->null_count() == arr_col_chunked_array->length();
      for (size_t f = 0; f < fragments.size(); f++) {
        key[3] = f;
        auto& frag = col[f];
        int64_t varlen = 0;
        frag.chunks.resize(fragments[f].last_chunk - fragments[f].first_chunk + 1);
        for (int i = fragments[f].first_chunk, e = fragments[f].last_chunk; i <= e; i++) {
          int size, offset;
          getSizeAndOffset(
              fragments[f], arr_col_chunked_array->chunk(i), i, size, offset);
          frag.offset += offset;
          frag.sz += size;
          frag.chunks[i - fragments[f].first_chunk] =
              arr_col_chunked_array->chunk(i)->data();
          auto& buffers = arr_col_chunked_array->chunk(i)->data()->buffers;
          if (!empty) {
            if (ctype == kTEXT) {
              if (buffers.size() <= 2) {
                LOG(FATAL) << "Type of column #" << cln
                           << " does not match between Arrow and description of "
                           << c.columnName;
              }
              auto offsets_buffer = reinterpret_cast<const uint32_t*>(buffers[1]->data());
              varlen += offsets_buffer[offset + size] - offsets_buffer[offset];
            } else if (buffers.size() != 2) {
              LOG(FATAL) << "Type of column #" << cln
                         << " does not match between Arrow and description of "
                         << c.columnName;
            }
          }
        }

        // create buffer descriptors
        if (ctype == kTEXT) {
          auto k = key;
          k.push_back(1);
          {
            auto b = mgr->createBuffer(k);
            b->setSize(varlen);
            b->encoder.reset(Encoder::Create(b, c.columnType));
            b->has_encoder = true;
            b->sql_type = c.columnType;
          }
          k[4] = 2;
          {
            auto b = mgr->createBuffer(k);
            b->sql_type = SQLTypeInfo(kINT, false);
            b->setSize(frag.sz * b->sql_type.get_size());
          }
        } else {
          auto b = mgr->createBuffer(key);
          b->sql_type = c.columnType;
          b->setSize(frag.sz * b->sql_type.get_size());
          b->encoder.reset(Encoder::Create(b, c.columnType));
          b->has_encoder = true;
          if (!empty) {
            size_t type_size = c.columnType.get_size();
            tg.run([b, fr = &frag, type_size]() {
              size_t sz = 0;
              for (size_t i = 0; i < fr->chunks.size(); i++) {
                auto& chunk = fr->chunks[i];
                int offset = (i == 0) ? fr->offset : 0;
                size_t size = (i == fr->chunks.size() - 1) ? (fr->sz - sz)
                                                           : (chunk->length - offset);
                sz += size;
                auto data = chunk->buffers[1]->data();
                b->encoder->updateStats((const int8_t*)data + offset * type_size, size);
              }
            });
          }
          b->encoder->setNumElems(frag.sz);
        }
      }
    }
  }  // each col and fragment

  // wait untill all stats have been updated
  tg.wait();

  VLOG(1) << "Created CSV backed temporary table with " << num_cols << " columns, "
          << arr_frags << " chunks, and " << fragments.size() << " fragments.";
  if (!isDataframe) {
    delete df_td;
  }
}

std::string ArrowCsvForeignStorage::getType() const {
  LOG(INFO) << "CSV backed temporary tables has been activated. Create table `with "
               "(storage_type='CSV:path/to/file.csv');`\n";
  return "CSV";
}
