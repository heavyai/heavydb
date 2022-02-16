/*
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

#include "ArrowStorage.h"
#include "ArrowStorageUtils.h"

#include "Shared/ArrowUtil.h"
#include "Shared/measure.h"
#include "Shared/threading.h"

#include <arrow/csv/reader.h>
#include <arrow/io/file.h>
#include <arrow/util/decimal.h>

using namespace std::string_literals;

namespace {

size_t computeTotalStringsLength(std::shared_ptr<arrow::ChunkedArray> arr,
                                 size_t offset,
                                 size_t rows) {
  size_t start_offset = offset;
  size_t chunk_no = 0;
  while (static_cast<size_t>(arr->chunk(chunk_no)->length()) <= start_offset) {
    start_offset -= arr->chunk(chunk_no)->length();
    ++chunk_no;
  }

  size_t rows_remain = rows;
  size_t total_bytes = 0;
  while (rows_remain) {
    auto chunk = arr->chunk(chunk_no);
    size_t rows_in_chunk = std::min(rows_remain, chunk->length() - start_offset);
    const uint32_t* offsets = chunk->data()->GetValues<uint32_t>(1);
    total_bytes += offsets[start_offset + rows_in_chunk] - offsets[start_offset];
    rows_remain -= rows_in_chunk;
    start_offset = 0;
    ++chunk_no;
  }

  return total_bytes;
}

}  // anonymous namespace

void ArrowStorage::fetchBuffer(const ChunkKey& key,
                               Data_Namespace::AbstractBuffer* dest,
                               const size_t num_bytes) {
  CHECK_EQ(key[CHUNK_KEY_DB_IDX], db_id_);
  CHECK_EQ(tables_.count(key[CHUNK_KEY_TABLE_IDX]), (size_t)1);
  auto& table = *tables_.at(key[CHUNK_KEY_TABLE_IDX]);

  size_t col_idx = static_cast<size_t>(key[CHUNK_KEY_COLUMN_IDX] - 1);
  size_t frag_idx = static_cast<size_t>(key[CHUNK_KEY_FRAGMENT_IDX] - 1);
  CHECK_LT(frag_idx, table.fragments.size());
  CHECK_LT(col_idx, table.col_data.size());

  const auto* fixed_type =
      dynamic_cast<const arrow::FixedWidthType*>(table.col_data[col_idx]->type().get());
  if (fixed_type) {
    CHECK_EQ(key.size(), (size_t)4);
    size_t elem_size = fixed_type->bit_width() / CHAR_BIT;
    fetchFixedLenData(table, frag_idx, col_idx, dest, num_bytes, elem_size);
  } else {
    CHECK_EQ(key.size(), (size_t)5);
    if (key[CHUNK_KEY_VARLEN_IDX] == 1) {
      if (!dest->hasEncoder()) {
        dest->initEncoder(getColumnInfo(key[CHUNK_KEY_DB_IDX],
                                        key[CHUNK_KEY_TABLE_IDX],
                                        key[CHUNK_KEY_COLUMN_IDX])
                              ->type);
      }
      fetchVarLenData(table, frag_idx, col_idx, dest, num_bytes);
    } else {
      CHECK_EQ(key[CHUNK_KEY_VARLEN_IDX], 2);
      fetchVarLenOffsets(table, frag_idx, col_idx, dest, num_bytes);
    }
  }
}

void ArrowStorage::fetchFixedLenData(const TableData& table,
                                     size_t frag_idx,
                                     size_t col_idx,
                                     Data_Namespace::AbstractBuffer* dest,
                                     size_t num_bytes,
                                     size_t elem_size) const {
  auto& frag = table.fragments[frag_idx];
  size_t rows_to_fetch = num_bytes ? num_bytes / elem_size : frag.row_count;
  auto data_to_fetch = table.col_data[col_idx]->Slice(
      static_cast<int64_t>(frag.offset), static_cast<int64_t>(rows_to_fetch));
  int8_t* dst_ptr = dest->getMemoryPtr();
  for (auto& chunk : data_to_fetch->chunks()) {
    size_t chunk_size = chunk->length() * elem_size;
    const int8_t* src_ptr =
        chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * elem_size);
    memcpy(dst_ptr, src_ptr, chunk_size);
    dst_ptr += chunk_size;
  }
}

void ArrowStorage::fetchVarLenOffsets(const TableData& table,
                                      size_t frag_idx,
                                      size_t col_idx,
                                      Data_Namespace::AbstractBuffer* dest,
                                      size_t num_bytes) const {
  auto& frag = table.fragments[frag_idx];
  CHECK_EQ(num_bytes, (frag.row_count + 1) * sizeof(uint32_t));
  // Number of fetched offsets is 1 greater than number of fetched rows.
  size_t rows_to_fetch = num_bytes ? num_bytes / sizeof(uint32_t) - 1 : frag.row_count;
  auto data_to_fetch = table.col_data[col_idx]->Slice(
      static_cast<int64_t>(frag.offset), static_cast<int64_t>(rows_to_fetch));
  uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dest->getMemoryPtr());
  uint32_t delta = 0;
  for (auto& chunk : data_to_fetch->chunks()) {
    const uint32_t* src_ptr = chunk->data()->GetValues<uint32_t>(1);
    delta -= *src_ptr;
    dst_ptr = std::transform(
        src_ptr, src_ptr + chunk->length(), dst_ptr, [delta](uint32_t val) {
          return val + delta;
        });
    delta += src_ptr[chunk->length()];
  }
  *dst_ptr = delta;
}

void ArrowStorage::fetchVarLenData(const TableData& table,
                                   size_t frag_idx,
                                   size_t col_idx,
                                   Data_Namespace::AbstractBuffer* dest,
                                   size_t num_bytes) const {
  auto& frag = table.fragments[frag_idx];
  auto data_to_fetch =
      table.col_data[col_idx]->Slice(static_cast<int64_t>(frag.offset), frag.row_count);
  int8_t* dst_ptr = dest->getMemoryPtr();
  size_t remained = num_bytes;
  for (auto& chunk : data_to_fetch->chunks()) {
    if (remained == 0) {
      break;
    }

    const uint32_t* offsets = chunk->data()->GetValues<uint32_t>(1);
    size_t chunk_size = offsets[chunk->length()] - offsets[0];
    chunk_size = std::min(chunk_size, num_bytes);
    memcpy(dst_ptr, chunk->data()->GetValues<int8_t>(2, offsets[0]), chunk_size);
    remained -= chunk_size;
    dst_ptr += chunk_size;
  }
}

Fragmenter_Namespace::TableInfo ArrowStorage::getTableMetadata(int db_id,
                                                               int table_id) const {
  CHECK_EQ(db_id, db_id_);
  CHECK_EQ(tables_.count(table_id), (size_t)1);
  auto& table = *tables_.at(table_id);

  Fragmenter_Namespace::TableInfo res;
  res.setPhysicalNumTuples(table.row_count);
  for (size_t frag_idx = 0; frag_idx < table.fragments.size(); ++frag_idx) {
    auto& frag = table.fragments[frag_idx];
    auto& frag_info = res.fragments.emplace_back();
    frag_info.fragmentId = static_cast<int>(frag_idx + 1);
    frag_info.physicalTableId = table_id;
    frag_info.setPhysicalNumTuples(frag.row_count);
    frag_info.deviceIds.push_back(0);  // Data_Namespace::DISK_LEVEL
    frag_info.deviceIds.push_back(0);  // Data_Namespace::CPU_LEVEL
    frag_info.deviceIds.push_back(0);  // Data_Namespace::GPU_LEVEL
    for (size_t col_idx = 0; col_idx < frag.metadata.size(); ++col_idx) {
      frag_info.setChunkMetadata(static_cast<int>(col_idx + 1), frag.metadata[col_idx]);
    }
  }
  return res;
}

const DictDescriptor* ArrowStorage::getDictMetadata(int db_id,
                                                    int dict_id,
                                                    bool /*load_dict*/) {
  CHECK_EQ(db_id, db_id_);
  CHECK_EQ(dicts_.count(dict_id), (size_t)1);
  return dicts_.at(dict_id).get();
}

TableInfoPtr ArrowStorage::createTable(const std::string& table_name,
                                       const std::vector<ColumnDescription>& columns,
                                       const TableOptions& options) {
  TableInfoPtr res;
  int table_id;
  {
    mapd_unique_lock<mapd_shared_mutex> lock(schema_mutex_);
    table_id = next_table_id_++;
    checkNewTableParams(table_name, columns, options);
    res = addTableInfo(
        db_id_, table_id, table_name, false, Data_Namespace::MemoryLevel::CPU_LEVEL, 0);
    int next_col_id = 1;
    std::unordered_map<int, int> dict_ids;
    for (auto& col : columns) {
      SQLTypeInfo type = col.type;
      // Positive dictionary id means we use existing dictionary. Other values
      // mean we have to create new dictionaries. Columns with equal negative
      // dict ids will share dictionaries.
      if (type.is_dict_encoded_string() && type.get_comp_param() <= 0) {
        int sharing_id = type.get_comp_param();
        if (sharing_id < 0 && dict_ids.count(sharing_id)) {
          type.set_comp_param(dict_ids.at(sharing_id));
        } else {
          int dict_id = next_dict_id_++;
          auto dict_desc = std::make_unique<DictDescriptor>(
              db_id_, dict_id, col.name, 32, true, 1, table_name, true);
          dict_desc->stringDict = std::make_shared<StringDictionary>(
              table_name + "."s + col.name, true, false);
          if (sharing_id < 0) {
            dict_ids.emplace(sharing_id, dict_id);
          }
          dicts_.emplace(dict_id, std::move(dict_desc));
          type.set_comp_param(dict_id);
        }
      }
      addColumnInfo(db_id_, table_id, next_col_id++, col.name, type, false);
    }
    addRowidColumn(db_id_, table_id);
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    auto& name = columns[i].name;
    auto& type = columns[i].type;
    auto field = arrow::field(name, getArrowImportType(type), !type.get_notnull());
    fields.push_back(field);
  }
  auto schema = arrow::schema(fields);

  {
    mapd_unique_lock<mapd_shared_mutex> lock(data_mutex_);
    auto [iter, inserted] = tables_.emplace(table_id, std::make_unique<TableData>());
    CHECK(inserted);
    auto& table = *iter->second;
    table.fragment_size = options.fragment_size;
    table.schema = schema;
  }

  return res;
}

TableInfoPtr ArrowStorage::importArrowTable(std::shared_ptr<arrow::Table> at,
                                            const std::string& table_name,
                                            const std::vector<ColumnDescription>& columns,
                                            const TableOptions& options) {
  auto res = createTable(table_name, columns, options);
  appendArrowTable(at, table_name);
  return res;
}

TableInfoPtr ArrowStorage::importArrowTable(std::shared_ptr<arrow::Table> at,
                                            const std::string& table_name,
                                            const TableOptions& options) {
  std::vector<ColumnDescription> columns;
  for (auto& field : at->schema()->fields()) {
    ColumnDescription desc{field->name(), getOmnisciType(*field->type())};
    // getOmnisciType sets comp_param for dictionaries to 32 because Catalog
    // uses it to compute type size and then replaces it with dictionary id.
    // Reset comp_param here for dictionaries to avoid unknown dictionary id
    // error.
    if (desc.type.is_dict_encoded_string()) {
      desc.type.set_comp_param(0);
    }
    columns.emplace_back(std::move(desc));
  }
  return importArrowTable(at, table_name, columns, options);
}

void ArrowStorage::appendArrowTable(std::shared_ptr<arrow::Table> at,
                                    const std::string& table_name) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendArrowTable(at, tinfo->table_id);
}

void ArrowStorage::appendArrowTable(std::shared_ptr<arrow::Table> at, int table_id) {
  if (!tables_.count(table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto& table = *tables_.at(table_id);
  compareSchemas(table.schema, at->schema());

  mapd_unique_lock<mapd_shared_mutex> lock(table.mutex);
  std::vector<std::shared_ptr<arrow::ChunkedArray>> col_data;
  col_data.resize(at->columns().size());

  std::vector<DataFragment> fragments;
  // Compute size of the fragment. If the last existing fragment is not full, then it will
  // be merged with the first new fragment.
  size_t first_frag_size =
      std::min(table.fragment_size, static_cast<size_t>(at->num_rows()));
  if (!table.fragments.empty()) {
    auto& last_frag = table.fragments.back();
    if (last_frag.row_count < table.fragment_size) {
      first_frag_size =
          std::min(first_frag_size, table.fragment_size - last_frag.row_count);
    }
  }
  // Now we can compute number of fragments to create.
  size_t frag_count =
      (static_cast<size_t>(at->num_rows()) + table.fragment_size - 1 - first_frag_size) /
          table.fragment_size +
      1;
  fragments.resize(frag_count);
  for (auto& frag : fragments) {
    frag.metadata.resize(at->columns().size());
  }

  threading::parallel_for(
      threading::blocked_range(0, (int)at->columns().size()), [&](auto range) {
        for (auto col_idx = range.begin(); col_idx != range.end(); col_idx++) {
          auto& col_type = getColumnInfo(db_id_, table_id, col_idx + 1)->type;
          auto col_arr = at->column(col_idx);

          if (col_type.get_type() == kDECIMAL || col_type.get_type() == kNUMERIC) {
            col_arr = convertDecimalToInteger(col_arr, col_type);
          } else if (col_type.is_string()) {
            if (col_type.is_dict_encoded_string()) {
              StringDictionary* dict =
                  dicts_.at(col_type.get_comp_param())->stringDict.get();

              switch (col_arr->type()->id()) {
                case arrow::Type::STRING:
                  col_arr = createDictionaryEncodedColumn(dict, col_arr);
                  break;
                case arrow::Type::DICTIONARY:
                  col_arr = convertArrowDictionary(dict, col_arr);
                  break;
                default:
                  CHECK(false);
              }
            }
          } else {
            col_arr = replaceNullValues(col_arr, col_type.get_type());
          }

          col_data[col_idx] = col_arr;

          if (col_type.get_type() != kTEXT || col_type.is_dict_encoded_string()) {
            // Compute stats for each fragment.
            threading::parallel_for(
                threading::blocked_range(size_t(0), frag_count), [&](auto frag_range) {
                  for (size_t frag_idx = frag_range.begin(); frag_idx != frag_range.end();
                       ++frag_idx) {
                    auto& frag = fragments[frag_idx];

                    frag.offset =
                        frag_idx
                            ? ((frag_idx - 1) * table.fragment_size + first_frag_size)
                            : 0;
                    frag.row_count =
                        frag_idx
                            ? std::min(table.fragment_size,
                                       static_cast<size_t>(at->num_rows()) - frag.offset)
                            : first_frag_size;

                    auto meta = std::make_shared<ChunkMetadata>();
                    meta->sqlType = col_type;
                    meta->numElements = frag.row_count;
                    meta->numBytes = frag.row_count * col_type.get_size();
                    computeStats(col_arr->Slice(frag.offset, frag.row_count),
                                 col_type,
                                 meta->chunkStats);
                    frag.metadata[col_idx] = meta;
                  }
                });  // each fragment
          } else {
            for (size_t frag_idx = 0; frag_idx < frag_count; ++frag_idx) {
              auto& frag = fragments[frag_idx];
              frag.offset =
                  frag_idx ? ((frag_idx - 1) * table.fragment_size + first_frag_size) : 0;
              frag.row_count =
                  frag_idx ? std::min(table.fragment_size,
                                      static_cast<size_t>(at->num_rows()) - frag.offset)
                           : first_frag_size;
              auto meta = std::make_shared<ChunkMetadata>();
              meta->sqlType = col_type;
              meta->numElements = frag.row_count;
              meta->numBytes =
                  computeTotalStringsLength(col_arr, frag.offset, frag.row_count);
              meta->chunkStats.has_nulls =
                  col_arr->Slice(frag.offset, frag.row_count)->null_count();
              frag.metadata[col_idx] = meta;
            }
          }
        }
      });  // each column

  if (table.row_count) {
    // If table is not empty then we have to merge chunked arrays.
    CHECK_EQ(table.col_data.size(), col_data.size());
    for (size_t i = 0; i < table.col_data.size(); ++i) {
      arrow::ArrayVector lhs = table.col_data[i]->chunks();
      arrow::ArrayVector rhs = col_data[i]->chunks();
      lhs.insert(lhs.end(), rhs.begin(), rhs.end());
      table.col_data[i] = arrow::ChunkedArray::Make(std::move(lhs)).ValueOrDie();
    }

    // Probably need to merge the last existing fragment with the first new one.
    size_t start_frag = 0;
    auto& last_frag = table.fragments.back();
    if (last_frag.row_count < table.fragment_size) {
      auto& first_frag = fragments.front();
      last_frag.row_count += first_frag.row_count;
      for (size_t col_idx = 0; col_idx < last_frag.metadata.size(); ++col_idx) {
        auto col_type = getColumnInfo(db_id_, table_id, col_idx + 1)->type;
        last_frag.metadata[col_idx]->numElements +=
            first_frag.metadata[col_idx]->numElements;
        last_frag.metadata[col_idx]->numBytes += first_frag.metadata[col_idx]->numBytes;
        mergeStats(last_frag.metadata[col_idx]->chunkStats,
                   first_frag.metadata[col_idx]->chunkStats,
                   col_type);
      }
      start_frag = 1;
    }

    // Copy the rest of fragments adjusting offset.
    table.fragments.reserve(table.fragments.size() + fragments.size() - start_frag);
    for (size_t frag_idx = start_frag; frag_idx < fragments.size(); ++frag_idx) {
      table.fragments.emplace_back(std::move(fragments[frag_idx]));
      table.fragments.back().offset += table.row_count;
    }

    table.row_count += at->num_rows();
  } else {
    CHECK_EQ(table.row_count, (size_t)0);
    table.col_data = std::move(col_data);
    table.fragments = std::move(fragments);
    table.row_count = at->num_rows();
  }

  getTableInfo(db_id_, table_id)->fragments = table.fragments.size();
}

TableInfoPtr ArrowStorage::importCsvFile(const std::string& file_name,
                                         const std::string& table_name,
                                         const std::vector<ColumnDescription>& columns,
                                         const TableOptions& options,
                                         const CsvParseOptions parse_options) {
  auto res = createTable(table_name, columns, options);
  appendCsvFile(file_name, table_name, parse_options);
  return res;
}

TableInfoPtr ArrowStorage::importCsvFile(const std::string& file_name,
                                         const std::string& table_name,
                                         const TableOptions& options,
                                         const CsvParseOptions parse_options) {
  auto at = readCsvFile(file_name, parse_options);
  return importArrowTable(at, table_name, options);
}

void ArrowStorage::appendCsvFile(const std::string& file_name,
                                 const std::string& table_name,
                                 const CsvParseOptions parse_options) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendCsvFile(file_name, tinfo->table_id, parse_options);
}

void ArrowStorage::appendCsvFile(const std::string& file_name,
                                 int table_id,
                                 const CsvParseOptions parse_options) {
  if (!tables_.count(table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto col_infos = listColumns(db_id_, table_id);
  auto at = readCsvFile(file_name, parse_options, col_infos);
  appendArrowTable(at, table_id);
}

void ArrowStorage::checkNewTableParams(const std::string& table_name,
                                       const std::vector<ColumnDescription>& columns,
                                       const TableOptions& options) const {
  if (columns.empty()) {
    throw std::runtime_error("Cannot create table with no columns");
  }

  if (table_name.empty()) {
    throw std::runtime_error("Cannot create table with empty name");
  }

  auto tinfo = getTableInfo(db_id_, table_name);
  if (tinfo) {
    throw std::runtime_error("Table with name '"s + table_name + "' already exists"s);
  }

  std::unordered_set<std::string> col_names;
  for (auto& col : columns) {
    if (col.name.empty()) {
      throw std::runtime_error("Empty column name is not allowed");
    }

    if (col.name == "rowid") {
      throw std::runtime_error("Reserved column name is not allowed: "s + col.name);
    }

    if (col_names.count(col.name)) {
      throw std::runtime_error("Duplicated column name: "s + col.name);
    }

    switch (col.type.get_type()) {
      case kBOOLEAN:
      case kTINYINT:
      case kSMALLINT:
      case kINT:
      case kBIGINT:
      case kFLOAT:
      case kDOUBLE:
        break;
      case kCHAR:
      case kVARCHAR:
      case kTEXT:
        if (col.type.is_dict_encoded_string()) {
          if (col.type.get_comp_param() > 0 &&
              dicts_.count(col.type.get_comp_param()) == 0) {
            throw std::runtime_error("Unknown dictionary ID is referenced in column '"s +
                                     col.name + "': "s +
                                     std::to_string(col.type.get_comp_param()));
          }
        }
        break;
      case kDECIMAL:
      case kNUMERIC:
      case kTIME:
      case kDATE:
        break;
      case kTIMESTAMP:
        switch (col.type.get_precision()) {
          case 0:
          case 3:
          case 6:
          case 9:
            break;
          default:
            throw std::runtime_error(
                "Unsupported timestamp precision for Arrow import: " +
                std::to_string(col.type.get_precision()));
        }
        break;
      case kARRAY:
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
      default:
        throw std::runtime_error("Unsupported type for Arrow import: "s +
                                 col.type.get_type_name());
    }

    col_names.insert(col.name);
  }
}

void ArrowStorage::compareSchemas(std::shared_ptr<arrow::Schema> lhs,
                                  std::shared_ptr<arrow::Schema> rhs) {
  auto& lhs_fields = lhs->fields();
  auto& rhs_fields = rhs->fields();
  if (lhs_fields.size() != rhs_fields.size()) {
    throw std::runtime_error("Mismatched clumns count");
  }

  for (size_t i = 0; i < lhs_fields.size(); ++i) {
    auto lhs_type = lhs_fields[i]->type();
    auto rhs_type = rhs_fields[i]->type();

    if (!lhs_type->Equals(rhs_type)) {
      throw std::runtime_error("Mismatched type for column: "s + lhs_fields[i]->name());
    }
  }
}

void ArrowStorage::computeStats(std::shared_ptr<arrow::ChunkedArray> arr,
                                SQLTypeInfo type,
                                ChunkStats& stats) {
  std::unique_ptr<Encoder> encoder(Encoder::Create(nullptr, type));
  for (auto& chunk : arr->chunks()) {
    encoder->updateStatsEncoded(
        chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * type.get_size()),
        chunk->length());
  }

  encoder->fillChunkStats(stats, type);
}

std::shared_ptr<arrow::Table> ArrowStorage::readCsvFile(
    const std::string& file_name,
    const CsvParseOptions parse_options,
    const ColumnInfoList& col_infos) {
#ifdef ENABLE_ARROW_4
  auto io_context = arrow::io::default_io_context();
#else
  auto io_context = arrow::default_memory_pool();
#endif

  auto arrow_parse_options = arrow::csv::ParseOptions::Defaults();
  arrow_parse_options.quoting = false;
  arrow_parse_options.escaping = false;
  arrow_parse_options.newlines_in_values = false;
  arrow_parse_options.delimiter = parse_options.delimiter;

  auto arrow_read_options = arrow::csv::ReadOptions::Defaults();
  arrow_read_options.use_threads = true;
  arrow_read_options.block_size = parse_options.block_size;
  arrow_read_options.autogenerate_column_names =
      !parse_options.header && col_infos.empty();
  arrow_read_options.skip_rows = parse_options.header && !col_infos.empty()
                                     ? (parse_options.skip_rows + 1)
                                     : parse_options.skip_rows;

  auto arrow_convert_options = arrow::csv::ConvertOptions::Defaults();
  arrow_convert_options.check_utf8 = false;
  arrow_convert_options.include_columns = arrow_read_options.column_names;
  arrow_convert_options.strings_can_be_null = true;

  for (auto& col_info : col_infos) {
    if (!col_info->is_rowid) {
      arrow_read_options.column_names.push_back(col_info->name);
      arrow_convert_options.column_types.emplace(col_info->name,
                                                 getArrowImportType(col_info->type));
    }
  }

  std::shared_ptr<arrow::io::ReadableFile> inp;
  auto file_result = arrow::io::ReadableFile::Open(file_name.c_str());
  ARROW_THROW_NOT_OK(file_result.status());
  inp = file_result.ValueOrDie();

  auto table_reader_result = arrow::csv::TableReader::Make(
      io_context, inp, arrow_read_options, arrow_parse_options, arrow_convert_options);
  ARROW_THROW_NOT_OK(table_reader_result.status());
  auto table_reader = table_reader_result.ValueOrDie();

  std::shared_ptr<arrow::Table> at;
  auto time = measure<>::execution([&]() {
    auto arrow_table_result = table_reader->Read();
    ARROW_THROW_NOT_OK(arrow_table_result.status());
    at = arrow_table_result.ValueOrDie();
  });

  VLOG(1) << "Read Arrow CSV file " << file_name << " in " << time << "ms";

  return at;
}