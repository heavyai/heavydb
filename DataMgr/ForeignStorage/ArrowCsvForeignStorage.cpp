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
#include <arrow/util/task_group.h>
#include <arrow/util/thread_pool.h>
#include <array>
#include <future>

#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"
#include "DataMgr/StringNoneEncoder.h"
#include "QueryEngine/ArrowResultSet.h"
#include "Shared/ArrowUtil.h"
#include "Shared/Logger.h"
#include "Shared/measure.h"

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
    int offset = i == 0 ? frag.offset : 0;
    size_t size =
        i == frag.chunks.size() - 1 ? frag.sz - read_size : array_data->length - offset;
    read_size += array_data->length - offset;
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
          // We assume that none encoded string is only supported varlen type
          auto offsets_buffer =
              reinterpret_cast<const uint32_t*>(array_data->buffers[1]->data());
          auto string_buffer_offset = offsets_buffer[offset];
          auto string_buffer_size = offsets_buffer[offset + size] - string_buffer_offset;
          std::memcpy(dest, bp->data() + string_buffer_offset, sz = string_buffer_size);
        }
      }
    } else {
      // TODO: nullify?
      auto fixed_type = dynamic_cast<arrow::FixedWidthType*>(array_data->type.get());
      if (fixed_type) {
        sz = array_data->length * fixed_type->bit_width() / 8;
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

void ArrowCsvForeignStorage::registerTable(Catalog_Namespace::Catalog* catalog,
                                           std::pair<int, int> table_key,
                                           const std::string& info,
                                           const TableDescriptor& td,
                                           const std::list<ColumnDescriptor>& cols,
                                           Data_Namespace::AbstractBufferMgr* mgr) {
  auto memory_pool = arrow::default_memory_pool();
  auto arrow_parse_options = arrow::csv::ParseOptions::Defaults();
  arrow_parse_options.quoting = false;
  arrow_parse_options.escaping = false;
  arrow_parse_options.newlines_in_values = false;

  auto arrow_read_options = arrow::csv::ReadOptions::Defaults();
  arrow_read_options.use_threads = true;

  const size_t default_block_size = 2 * 1024 * 1024;
  arrow_read_options.block_size = default_block_size;
  auto arrow_convert_options = arrow::csv::ConvertOptions::Defaults();
  arrow_convert_options.check_utf8 = false;

  arrow_read_options.skip_rows = 0;  // TODO: add a way to switch csv header on
  arrow_read_options.autogenerate_column_names =
      false;  // read column names from first row after skip_rows
  arrow_convert_options.include_columns = arrow_read_options.column_names;

  for (auto c : cols) {
    if (c.isSystemCol) {
      continue;  // must be processed by base interface implementation
    }
    arrow_convert_options.column_types.emplace(c.columnName,
                                               getArrowImportType(c.columnType));
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
  struct fragment {
    int first_chunk;         // index of the first chunk assigned to the fragment
    int first_chunk_offset;  // offset from the begining of the first chunk
    int last_chunk;          // index of the last chunk
    int last_chunk_size;     // number of elements in the last chunk
  };

  std::vector<fragment> fragments;
  int64_t sz = 0;
  int64_t offset = 0;
  fragments.push_back({0, 0, 0, 0});

  for (int i = 0; i < arr_frags;) {
    auto& chunk = *c0p->chunk(i);
    auto& frag = *fragments.rbegin();
    if (td.maxFragRows - sz > chunk.length() - offset) {
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
      frag.last_chunk_size = td.maxFragRows - sz;
      offset += td.maxFragRows - sz;
      sz = 0;
      fragments.push_back({i, static_cast<int>(offset), 0, 0});
    }
  }

  // data comes like this - database_id, table_id, column_id, fragment_id
  ChunkKey key{table_key.first, table_key.second, 0, 0};
  std::array<int, 3> col_key{table_key.first, table_key.second, 0};

  auto tp = arrow::internal::GetCpuThreadPool();
  auto tg = arrow::internal::TaskGroup::MakeThreaded(tp);

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

    StringDictionary* dict{nullptr};
    if (c.columnType.is_dict_encoded_string()) {
      auto dictDesc = const_cast<DictDescriptor*>(
          catalog->getMetadataForDict(c.columnType.get_comp_param()));
      dict = dictDesc->stringDict.get();
    }
    auto empty = arr_col_chunked_array->null_count() == arr_col_chunked_array->length();

    // fill each fragment
    for (size_t f = 0; f < fragments.size(); f++) {
      key[3] = f;
      auto& frag = col[f];
      int64_t varlen = 0;
      // for each arrow chunk
      for (int i = fragments[f].first_chunk, e = fragments[f].last_chunk; i <= e; i++) {
        int offset = i == fragments[f].first_chunk ? fragments[f].first_chunk_offset : 0;
        int size = i == fragments[f].last_chunk
                       ? fragments[f].last_chunk_size
                       : arr_col_chunked_array->chunk(i)->length();
        CHECK_LT(i, arr_col_chunked_array->length());
        if (c.columnType.is_dict_encoded_string()) {
          arrow::Int32Builder indexBuilder;
          auto stringArray = std::static_pointer_cast<arrow::StringArray>(
              arr_col_chunked_array->chunk(i));
          indexBuilder.Reserve(size);
          for (int i = offset; i < offset + size; i++) {
            // TODO: use arrow dictionary encoding
            if (stringArray->IsNull(i) || empty ||
                stringArray->null_count() == stringArray->length()) {
              indexBuilder.Append(inline_int_null_value<int32_t>());
            } else {
              CHECK(dict);
              auto curStr = stringArray->GetString(i);
              indexBuilder.Append(dict->getOrAdd(curStr));
            }
          }
          std::shared_ptr<arrow::Array> indexArray;
          ARROW_THROW_NOT_OK(indexBuilder.Finish(&indexArray));
          frag.chunks.emplace_back(indexArray->data());
          frag.sz += size;
          frag.offset = 0;
        } else {
          // this will work as we have only one non zero offset for column
          frag.offset += offset;
          frag.sz += size;
          frag.chunks.emplace_back(arr_col_chunked_array->chunk(i)->data());
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
      }

      // create buffer descriptotrs
      if (ctype == kTEXT && !c.columnType.is_dict_encoded_string()) {
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
          size_t type_size = 0;
          auto fixed_type =
              dynamic_cast<arrow::FixedWidthType*>(frag.chunks[0]->type.get());
          if (fixed_type) {
            type_size = fixed_type->bit_width() / 8;
          } else {
            CHECK(false);
          }
          // asynchronously update stats for incoming data
          tg->Append([b, fr = &frag, type_size]() {
            size_t sz = 0;
            for (size_t i = 0; i < fr->chunks.size(); i++) {
              auto& chunk = fr->chunks[i];
              int offset = i == 0 ? fr->offset : 0;
              size_t size = i == fr->chunks.size() - 1 ? fr->sz - sz : chunk->length;
              sz += chunk->length - offset;
              auto data = chunk->buffers[1]->data();
              b->encoder->updateStats((const int8_t*)data + offset * type_size, size);
            }
            return arrow::Status::OK();
          });
        }
        b->encoder->setNumElems(frag.sz);
      }
    }
  }  // each col and fragment

  // wait untill all stats have been updated
  ARROW_THROW_NOT_OK(tg->Finish());
  VLOG(1) << "Created CSV backed temporary table with " << num_cols << " columns, "
          << arr_frags << " chunks, and " << fragments.size() << " fragments.";
}

std::string ArrowCsvForeignStorage::getType() const {
  LOG(INFO) << "CSV backed temporary tables has been activated. Create table `with "
               "(storage_type='CSV:path/to/file.csv');`\n";
  return "CSV";
}
