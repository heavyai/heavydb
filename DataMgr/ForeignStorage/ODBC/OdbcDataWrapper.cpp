/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OdbcDataWrapper.h"

#include <regex>

#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "ForeignStorageException.h"
#include "ForeignTableSchema.h"
#include "FsiChunkUtils.h"
#include "ImportExport/CopyParams.h"
#include "Interval.h"
#include "OdbcGeospatialEncoder.h"
#include "Shared/JsonUtils.h"
#include "Shared/misc.h"
#include "Shared/thread_count.h"
#include "SharedMetadataValidator.h"
#include "odbc_utils.h"

bool g_enable_odbc_stats_scan{false};
extern size_t g_max_import_threads;
namespace {
struct RowSetResult {
  size_t num_rows;
  DataBlockPtr data_block_ptr;

  // Containers for underlying DataBlockPtr buffers
  std::vector<std::string> strings;
  std::vector<int32_t> string_ids;
  std::vector<int64_t> timestamps;
  std::vector<int8_t> numeric_types;
  std::unique_ptr<import_export::UnmanagedTypedImportBuffer> import_buffer;

  // Container used for error tracking
  std::set<size_t> rejected_rows;
  // Flag indicating error handling behaviour expected
  bool track_rejected_rows;
};

void update_delete_buffer(AbstractBuffer* delete_buffer,
                          const size_t row_start,
                          const size_t num_rows_to_append,
                          std::set<size_t>& rejected_rows,
                          std::mutex& delete_buffer_mutex) {
  std::unique_lock<std::mutex> delete_buffer_lock(delete_buffer_mutex);

  // ensure delete buffer is sized appropriately
  auto num_rows_in_chunk = row_start + num_rows_to_append;
  // ensure delete buffer is sized appropriately
  if (delete_buffer->size() < num_rows_in_chunk) {
    auto remaining_rows = num_rows_in_chunk - delete_buffer->size();
    std::vector<int8_t> data(remaining_rows, false);
    delete_buffer->append(data.data(), remaining_rows);
  }

  // mark deleted rows
  auto delete_data = delete_buffer->getMemoryPtr();
  CHECK(delete_data);
  for (auto rejected_index : rejected_rows) {
    CHECK_LT(rejected_index + row_start, delete_buffer->size());
    delete_data[rejected_index + row_start] = true;
  }
}

/*
 * Text data returned from the odbc layer is a block of fixed length strings, though
 * the strings themselves are null terminated.
 * For dictionary and non dictionary processing the data needs to be presented as a
 * vector of strings.
 * Note the 'accumulation' param strings does not need to be empty.
 * NULL values are returned as empty strings
 */
std::vector<std::string> convert_block_to_vector_string(
    size_t num_of_strings,
    size_t str_size,
    const foreign_storage::RemoteData& remote_data) {
  std::vector<std::string> strings;
  strings.reserve(strings.size() + num_of_strings);
  const char* text_block = reinterpret_cast<const char*>(remote_data.data_ptr);

  CHECK(remote_data.null_or_strlen.size() == num_of_strings);
  for (size_t i = 0; i < num_of_strings; i++) {
    auto null_or_strlen_value = remote_data.null_or_strlen[i];
    if (null_or_strlen_value) {
      CHECK(*null_or_strlen_value <= str_size)
          << "string exceeds ODBC buffer size resulting in truncation";
      strings.emplace_back(std::string{text_block + i * str_size, *null_or_strlen_value});
    } else {
      strings.emplace_back("");
    }
  }
  return strings;
}

void load_decimals_into_data_block(const foreign_storage::RemoteData& remote_data,
                                   size_t num_elements,
                                   size_t data_size,
                                   const ColumnDescriptor* column_descriptor,
                                   RowSetResult& result) {
  auto strings = convert_block_to_vector_string(num_elements, data_size, remote_data);
  //  Simply a place holder for the add_value method
  // not used in the kDECIMAL part of UnmanagedTypedImportBuffer code.
  import_export::CopyParams place_holder_copy_params;
  result.import_buffer = std::make_unique<import_export::UnmanagedTypedImportBuffer>(
      column_descriptor, nullptr);
  CHECK_EQ(strings.size(), num_elements);
  for (size_t i = 0; i < num_elements; i++) {
    try {
      result.import_buffer.get()->add_value(column_descriptor,
                                            strings[i],
                                            !remote_data.null_or_strlen[i].has_value(),
                                            place_holder_copy_params);
    } catch (const std::runtime_error& except) {
      if (result.track_rejected_rows) {
        result.import_buffer.get()->add_value(
            column_descriptor, "", true, place_holder_copy_params, false);
        result.rejected_rows.insert(i);
      } else {
        throw except;
      }
    }
  }
  result.data_block_ptr.numbersPtr = result.import_buffer->getAsBytes();
}

int64_t convert_positive_integer(const std::string& s) {
  auto l = std::stol(s);
  if (l <= 0) {
    throw foreign_storage::ForeignStorageException("Can not parse string '" + s +
                                                   "' into a positive integer");
  }
  return l;
}

size_t validate_and_get_odbc_buffer_byte_size(
    const foreign_storage::OptionsMap& options_map) {
  size_t buffer_byte_size;
  if (auto it = options_map.find("BUFFER_SIZE"); it != options_map.end()) {
    try {
      buffer_byte_size = convert_positive_integer(it->second);
    } catch (const std::exception& e) {
      throw foreign_storage::ForeignStorageException(
          std::string(e.what()) + " while validating option 'BUFFER_SIZE'");
    }
  } else {
    buffer_byte_size =
        foreign_storage::OdbcDataWrapper::DEFAULT_BUFFER_SIZE;  // default to 8MB
  }
  return buffer_byte_size;
}

void validate_odbc_buffer_byte_size(const foreign_storage::OptionsMap& options_map) {
  validate_and_get_odbc_buffer_byte_size(options_map);
}

size_t get_odbc_buffer_byte_size(const foreign_storage::OptionsMap& options_map) {
  size_t buffer_byte_size{0};
  try {
    buffer_byte_size = validate_and_get_odbc_buffer_byte_size(options_map);
  } catch (const std::exception& except) {
    UNREACHABLE() << "no exception is expected post validation";
  }
  return buffer_byte_size;
}

std::list<std::unique_ptr<ChunkMetadata>> create_geometry_chunk_metadata(
    std::list<Chunk_NS::Chunk>& chunks) {
  std::list<std::unique_ptr<ChunkMetadata>> chunk_metadata;
  for (auto chunks_iter = chunks.begin(); chunks_iter != chunks.end(); ++chunks_iter) {
    chunk_metadata.emplace_back(std::make_unique<ChunkMetadata>());
    auto& chunk_metadata_ptr = chunk_metadata.back();
    chunk_metadata_ptr->sqlType = chunks_iter->getColumnDesc()->columnType;
  }
  return chunk_metadata;
}

void handle_errors_with_strings_if_applicable(std::vector<std::string>& strings,
                                              RowSetResult& result,
                                              const SQLTypeInfo& sql_type_info) {
  if (!result.track_rejected_rows) {
    return;
  }

  for (size_t i = 0; i < strings.size(); ++i) {
    auto& str = strings[i];
    if (str.size() > StringDictionary::MAX_STRLEN) {
      str = "";  // set to null
      result.rejected_rows.insert(i);
    }
    if (str == "" && sql_type_info.get_notnull()) {
      result.rejected_rows.insert(i);
    }
  }
}

/*
 * Takes a vector of strings and if encoded runs them
 * through the dictionary process and add the results (numbers) to the chunk
 * or simply adds the strings themselves.
 */
void load_strings_into_data_block(const foreign_storage::RemoteData& remote_data,
                                  size_t num_elements,
                                  size_t data_size,
                                  bool is_dictionary_encoded,
                                  const SQLTypeInfo& sql_type_info,
                                  const DictDescriptor* dict_descriptor,
                                  RowSetResult& result) {
  // Note - DataBlockPtr will contain a raw ptr
  // to either the data in dest_ids or strings; when its
  // used in chunk append the relevant memory allocation
  // must still be in scope.
  if (is_dictionary_encoded) {
    // Encode strings and append encoded integers to chunk
    auto strings = convert_block_to_vector_string(num_elements, data_size, remote_data);
    handle_errors_with_strings_if_applicable(strings, result, sql_type_info);
    StringDictionary* string_dictionary = dict_descriptor->stringDict.get();
    result.string_ids.resize(strings.size());

    // getOrAddBulk() infers the encoding size based on the type of the result pointer we
    // pass in.  This matters because the value we use to represent nulls is different for
    // each type, so we need to make sure to cast to the correct size depending on the
    // column type.
    // TODO(Misiu): This is an easy to misuse inferface for getOrAddBulk().  We should fix
    // this to avoid further issues.
    if (auto encoding_size = sql_type_info.get_size(); encoding_size == 1) {
      string_dictionary->getOrAddBulk(strings,
                                      reinterpret_cast<uint8_t*>(&result.string_ids[0]));
    } else if (encoding_size == 2) {
      string_dictionary->getOrAddBulk(strings,
                                      reinterpret_cast<uint16_t*>(&result.string_ids[0]));
    } else if (encoding_size == 4) {
      string_dictionary->getOrAddBulk(strings, &result.string_ids[0]);
    } else {
      UNREACHABLE() << "Unsupported dictionary size: " << sql_type_info.get_size();
    }

    result.data_block_ptr.numbersPtr = reinterpret_cast<int8_t*>(&result.string_ids[0]);
  } else {
    // Append string vector to chunk
    result.strings = convert_block_to_vector_string(num_elements, data_size, remote_data);
    handle_errors_with_strings_if_applicable(result.strings, result, sql_type_info);
    result.data_block_ptr.setStringsPtr(result.strings);
  }
}

void load_temporal_into_data_block(const foreign_storage::RemoteData& remote_data,
                                   size_t num_elements,
                                   size_t data_size,
                                   const SQLTypeInfo& sql_type_info,
                                   RowSetResult& result) {
  auto dptr = remote_data.data_ptr;
  for (size_t i = 0; i < num_elements; i++) {
    if (!remote_data.null_or_strlen[i].has_value()) {
      // is null
      result.timestamps.push_back(inline_fixed_encoding_null_val(sql_type_info));
      if (sql_type_info.get_notnull() && result.track_rejected_rows) {
        result.rejected_rows.insert(i);
      }
    } else {
      try {
        size_t offset = i * data_size;
        int64_t epoch =
            foreign_storage::temporal_conversion_utility(dptr + offset, sql_type_info);
        result.timestamps.push_back(epoch);
      } catch (const std::runtime_error& except) {
        if (result.track_rejected_rows) {
          // track local rejected index and insert a null instead
          result.rejected_rows.insert(i);
          result.timestamps.push_back(inline_fixed_encoding_null_val(sql_type_info));
        } else {
          throw except;
        }
      }
    }
  }
  result.data_block_ptr.numbersPtr = reinterpret_cast<int8_t*>(&result.timestamps[0]);
}

void json_extract_column_descriptions(
    const rapidjson::Value& column_values,
    std::map<int, foreign_storage::RemoteColumnDescription>& remote_column_detail) {
  auto array = column_values.GetArray();
  for (auto& cd : array) {
    foreign_storage::RemoteColumnDescription rcd;
    rcd.column_name = cd.FindMember("column_name")->value.GetString();
    rcd.odbc_octet_transfer_size =
        cd.FindMember("odbc_octet_transfer_size")->value.GetInt64();
    rcd.decimal_digits = cd.FindMember("decimal_digits")->value.GetInt64();
    rcd.omnisci_type = (SQLTypes)cd.FindMember("omnisci_type")->value.GetInt();
    rcd.omnisci_type_name = cd.FindMember("omnisci_type_name")->value.GetString();
    rcd.omnisci_column_id = cd.FindMember("omnisci_column_id")->value.GetInt();
    rcd.odbc_base_type = cd.FindMember("odbc_base_type")->value.GetInt();
    rcd.is_unsigned = cd.HasMember("is_unsigned")
                          ? cd.FindMember("is_unsigned")->value.GetBool()
                          : false;
    remote_column_detail.insert({rcd.omnisci_column_id, rcd});
  }
}

void json_add_column_descriptions(
    rapidjson::Document& document,
    const std::map<int, foreign_storage::RemoteColumnDescription>& remote_column_detail) {
  // Add an array of column descriptions
  rapidjson::Value column_values(rapidjson::kArrayType);
  for (auto& [_, cd] : remote_column_detail) {
    rapidjson::Value column_value(rapidjson::kObjectType);
    column_value.AddMember("column_name", cd.column_name, document.GetAllocator());
    column_value.AddMember(
        "odbc_octet_transfer_size", cd.odbc_octet_transfer_size, document.GetAllocator());
    column_value.AddMember("decimal_digits", cd.decimal_digits, document.GetAllocator());
    column_value.AddMember("omnisci_type", cd.omnisci_type, document.GetAllocator());
    column_value.AddMember(
        "omnisci_column_id", cd.omnisci_column_id, document.GetAllocator());
    column_value.AddMember(
        "omnisci_type_name", cd.omnisci_type_name, document.GetAllocator());
    column_value.AddMember("odbc_base_type", cd.odbc_base_type, document.GetAllocator());
    column_value.AddMember("is_unsigned", cd.is_unsigned, document.GetAllocator());
    column_values.PushBack(column_value, document.GetAllocator());
  }
  document.AddMember("column_values", column_values, document.GetAllocator());
  return;
}

}  // namespace

namespace foreign_storage {

OdbcDataWrapper::OdbcDataWrapper()
    : db_id_(-1), foreign_table_(nullptr), user_mapping_(nullptr) {}

OdbcDataWrapper::OdbcDataWrapper(const int db_id,
                                 const ForeignTable* foreign_table,
                                 const UserMapping* user_mapping)
    : db_id_(db_id)
    , foreign_table_(foreign_table)
    , total_row_count_(0)
    , user_mapping_(user_mapping) {}

void OdbcDataWrapper::validateServerOptions(const ForeignServer* foreign_server) const {
  const auto& options = foreign_server->options;
  for (const auto& entry : options) {
    if (!shared::contains(supported_server_options_, entry.first)) {
      throw ForeignStorageException{"Invalid foreign server option \"" + entry.first +
                                    "\". Option must be one of the following: " +
                                    join(supported_server_options_, ", ") + "."};
    }
  }
  if ((options.find(ODBC_DSN_KEY) == options.end()) &&
      (options.find(ODBC_CONNECTION_KEY) == options.end())) {
    throw ForeignStorageException{
        "Foreign server options must contain a value for either \"" +
        std::string(ODBC_DSN_KEY) + "\" or \"" + std::string(ODBC_CONNECTION_KEY) +
        "\"."};
  }

  if ((options.find(ODBC_DSN_KEY) != options.end()) &&
      (options.find(ODBC_CONNECTION_KEY) != options.end())) {
    throw ForeignStorageException{"Foreign server options must contain only one of \"" +
                                  std::string(ODBC_DSN_KEY) + "\" or \"" +
                                  std::string(ODBC_CONNECTION_KEY) + "\"."};
  }
}

void OdbcDataWrapper::validateTableOptions(const ForeignTable* foreign_table) const {
  if (foreign_table->options.find(ODBC_SELECT_KEY) == foreign_table->options.end() ||
      foreign_table->options.find(ODBC_SELECT_KEY)->second.empty()) {
    throw ForeignStorageException{"Foreign table options must contain a value for \"" +
                                  std::string(ODBC_SELECT_KEY) + "\"."};
  }
  if (foreign_table->options.find(ODBC_ORDER_BY_KEY) == foreign_table->options.end() ||
      foreign_table->options.find(ODBC_ORDER_BY_KEY)->second.empty()) {
    throw ForeignStorageException{"Foreign table options must contain a value for \"" +
                                  std::string(ODBC_ORDER_BY_KEY) + "\"."};
  }
  validate_odbc_buffer_byte_size(foreign_table->options);
}

const std::set<std::string_view>& OdbcDataWrapper::getSupportedTableOptions() const {
  return supported_table_options_;
}

namespace {
std::string get_option_or_empty(const OptionsMap& options, const std::string& key) {
  if (auto option_it = options.find(key); option_it != options.end()) {
    return option_it->second;
  } else {
    return "";
  }
}
}  // namespace

void OdbcDataWrapper::validateUserMappingOptions(
    const UserMapping* user_mapping,
    const ForeignServer* foreign_server) const {
  const auto user_options = user_mapping->getUnencryptedOptions();
  const auto server_options = foreign_server->options;

  validate_odbc_credential_options(
      get_option_or_empty(server_options, ODBC_DSN_KEY),
      get_option_or_empty(server_options, ODBC_CONNECTION_KEY),
      get_option_or_empty(user_options, ODBC_CREDENTIAL),
      get_option_or_empty(user_options, ODBC_USERNAME),
      get_option_or_empty(user_options, ODBC_PASSWORD),
      false,
      foreign_server->name);
};

const std::set<std::string_view>& OdbcDataWrapper::getSupportedUserMappingOptions()
    const {
  return supported_user_mapping_options_;
}

namespace {
bool is_geometry_column(const ColumnDescriptor* column_descriptor) {
  return column_descriptor->columnType.is_geometry();
}

// Return a vector of n integers with a sum of m as evenly as possible
std::vector<size_t> divide_m_into_n(size_t m, size_t n) {
  std::vector<size_t> ret;
  size_t floor = m / n;
  for (size_t i = 0; i < n; i++) {
    ret.emplace_back(floor);
    m -= floor;
  }
  CHECK(m < n);
  for (size_t i = 0; i < m; i++) {
    ++ret[i];
  }
  return ret;
}
}  // namespace

void OdbcDataWrapper::populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) {
  // The ODBC data wrapper doesn't currently support refresh and therefore
  // populateChunkMetadata is only called prior to an initial load of any data.
  //
  // TODO examine error handling implications when implementing refresh

  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);

  std::list<ColumnDescriptor const*> logical_columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, false);

  std::list<ColumnDescriptor const*> logical_and_physical_columns =
      catalog->getAllColumnMetadataForTable(foreign_table_->tableId, false, false, true);

  std::unique_ptr<OdbcConnection> odbc_connection =
      OdbcConnection::create(getDbConnectionInfo(), user_mapping_);

  auto db_select = getDbSelect();
  auto db_order_by = foreign_table_->getOption(ODBC_ORDER_BY_KEY);
  if (!db_order_by.has_value()) {
    // Legacy foreign tables may not have this option set.  If this is the case then we
    // should prompt the user to update the table.
    throw std::runtime_error{
        "Odbc-backed foreign tables require an \"SQL_ORDER_BY\" option to specify the "
        "odbc result order.  Please add one to table \"" +
        foreign_table_->tableName +
        "\" using the \"ALTER FOREIGN TABLE SET (SQL_ORDER_BY='')\" command."};
  }

  std::vector<foreign_storage::RemoteColumnDescription> remote_column_details;
  odbc_connection->getRemoteColumnDetailsValidateRemoteMetaData(
      db_select, logical_columns, remote_column_details);
  for (const auto& remote_column_detail : remote_column_details) {
    remote_column_details_.insert(
        {remote_column_detail.omnisci_column_id, remote_column_detail});
  }

  auto rec_cnt = odbc_connection->getRecordCnt(db_select);
  int number_fragments =
      (rec_cnt + foreign_table_->maxFragRows - 1) / foreign_table_->maxFragRows;
  int rows_remaining = (rec_cnt % foreign_table_->maxFragRows);
  int last_fragment_row_count =
      (rows_remaining == 0 ? foreign_table_->maxFragRows : rows_remaining);

  int start_frag_num = 0;
  if (!foreign_table_->isAppendMode()) {
    // chunk_metadata_map keeps a copy of the metadata for use in populateChunkBuffers
    chunk_metadata_map_.clear();
  } else {
    if (rec_cnt < total_row_count_) {
      throw_removed_row_in_result_set_error(db_select);
    } else {
      int previous_number_fragments =
          (total_row_count_ + foreign_table_->maxFragRows - 1) /
          foreign_table_->maxFragRows;
      if (previous_number_fragments > 0) {
        start_frag_num = previous_number_fragments - 1;
      }
      for (const auto& [chunk_key, metadata] : chunk_metadata_map_) {
        if (chunk_key[CHUNK_KEY_FRAGMENT_IDX] < start_frag_num) {
          chunk_metadata_vector.emplace_back(chunk_key, metadata);
        }
      }
    }
  }

  foreign_storage::ForeignTableSchema schema(catalog->getDatabaseId(), foreign_table_);

  for (int frag_num = start_frag_num; frag_num < number_fragments; frag_num++) {
    int num_elements_this_fragment = (frag_num == number_fragments - 1)
                                         ? last_fragment_row_count
                                         : foreign_table_->maxFragRows;
    CHECK_GT(num_elements_this_fragment, 0);
    fragment_remote_db_rownumber_start_[frag_num] =
        frag_num * foreign_table_->maxFragRows;

    for (auto column : logical_and_physical_columns) {
      ChunkKey chunk_key{db_id_, foreign_table_->tableId, column->columnId, frag_num};
      if (column->columnType.is_varlen_indeed()) {
        chunk_key.emplace_back(1);
      }

      auto metadata = std::make_shared<ChunkMetadata>(
          get_placeholder_metadata(column->columnType, num_elements_this_fragment, {}));
      chunk_metadata_vector.emplace_back(chunk_key, metadata);
      chunk_metadata_map_[chunk_key] = metadata;

      if (g_enable_odbc_stats_scan && !column->columnType.is_dict_encoded_string() &&
          !is_geometry_column(schema.getLogicalColumn(column->columnId))) {
        // TODO(Misiu): Refactor this code to remove "int"s into size_t since they all use
        // absolute values.  This will remove the need for the size_t cast below.
        CHECK_GE(num_elements_this_fragment, 0);
        OdbcSelectDescriptor select_desc{fragment_remote_db_rownumber_start_[frag_num],
                                         static_cast<size_t>(num_elements_this_fragment),
                                         db_select,
                                         *db_order_by,
                                         *column,
                                         remote_column_details_[column->columnId]};

        size_t null_count = odbc_connection->getChunkNullCount(select_desc);
        metadata->chunkStats.has_nulls = null_count > 0;
        if (!column->columnType.is_varlen_indeed() &&
            null_count < static_cast<size_t>(num_elements_this_fragment)) {
          populateMinMaxChunkStats(metadata->chunkStats, column, chunk_key);
        }
      }
    }
  }
  total_row_count_ = rec_cnt;
}

OdbcSelectDescriptor OdbcDataWrapper::getDefaultSelectDescriptor(
    const ColumnDescriptor* column_descriptor,
    const ChunkKey& key) const {
  OdbcSelectDescriptor desc{
      getDbSelect(),
      foreign_table_->getOption(ODBC_ORDER_BY_KEY).value(),
      *column_descriptor,
      shared::get_from_map(remote_column_details_, key[CHUNK_KEY_COLUMN_IDX])};
  desc.select_type = OdbcSelectDescriptor::SelectQueryType::kSelect;
  return desc;
}

OdbcSelectDescriptor OdbcDataWrapper::getMinValueSelectDescriptor(
    const ColumnDescriptor* column_descriptor,
    const ChunkKey& key) const {
  OdbcSelectDescriptor desc{
      getDbSelect(),
      foreign_table_->getOption(ODBC_ORDER_BY_KEY).value(),
      *column_descriptor,
      shared::get_from_map(remote_column_details_, key[CHUNK_KEY_COLUMN_IDX])};
  desc.select_type = OdbcSelectDescriptor::SelectQueryType::kMinValue;
  return desc;
}

OdbcSelectDescriptor OdbcDataWrapper::getMaxValueSelectDescriptor(
    const ColumnDescriptor* column_descriptor,
    const ChunkKey& key) const {
  OdbcSelectDescriptor desc{
      getDbSelect(),
      foreign_table_->getOption(ODBC_ORDER_BY_KEY).value(),
      *column_descriptor,
      shared::get_from_map(remote_column_details_, key[CHUNK_KEY_COLUMN_IDX])};
  desc.select_type = OdbcSelectDescriptor::SelectQueryType::kMaxValue;
  return desc;
}

void OdbcDataWrapper::populateMinMaxChunkStats(ChunkStats& chunk_stats,
                                               const ColumnDescriptor* column_descriptor,
                                               const ChunkKey& key) {
  auto populate_stat = [&](const bool is_min) {
    ForeignStorageBuffer data_buffer;  // Memory use is expected to be negligble here
    auto chunk = Chunk_NS::Chunk{column_descriptor, false};
    // NOTE: the call below is only applicable to non varlen types, this condition is
    // checked at a higher level
    appendRemoteDataToChunk(column_descriptor,
                            &data_buffer,
                            nullptr,
                            nullptr,
                            key,
                            chunk,
                            1,
                            is_min ? getMinValueSelectDescriptor(column_descriptor, key)
                                   : getMaxValueSelectDescriptor(column_descriptor, key));
    auto chunk_metadata = data_buffer.getEncoder()->getMetadata();
    if (is_min) {
      chunk_stats.min = chunk_metadata.chunkStats.min;
    } else {
      chunk_stats.max = chunk_metadata.chunkStats.max;
    }
  };
  populate_stat(true);   // populate min
  populate_stat(false);  // populate max
}

void OdbcDataWrapper::processRemoteDataSource(const int64_t buffer_byte_size,
                                              ResultSetProcessor result_set_processor,
                                              const OdbcSelectDescriptor& select_desc) {
  /*
   * Having done all preparatory work, now connect to the remote db
   * and read a chunk of data from a column.
   *
   * TODO change to allow for a raw ptr to provided if the Abstractbuffer
   * used in the calling program supports direct writes.
   */

  std::unique_ptr<OdbcConnection> odbc_connection;
  {
    // Prevent ODBC connection failures when establishing multiple connections
    // concurrently
    std::unique_lock<std::mutex> connection_lock(odbc_connection_mutex_);
    odbc_connection = OdbcConnection::create(getDbConnectionInfo(), user_mapping_);
  }

  try {
    odbc_connection->runSelectCmd(select_desc, buffer_byte_size, result_set_processor);
  } catch (const ForeignStorageException& except) {
    throw ForeignStorageException(std::string(except.what()) +
                                  " Foreign table: " + foreign_table_->tableName);
  }
}

std::list<Chunk_NS::Chunk> OdbcDataWrapper::initializeGeoChunks(
    const std::map<ChunkKey, AbstractBuffer*>& required_buffers,
    const std::list<const ColumnDescriptor*>& column_descriptors,
    const int fragment_id) {
  std::list<Chunk_NS::Chunk> chunks;  // list of chunks to load for geo-type
  for (const auto& column_descriptor : column_descriptors) {
    const int column_id = column_descriptor->columnId;
    Chunk_NS::Chunk chunk{column_descriptor, false};
    if (column_descriptor->columnType.is_varlen_indeed()) {
      ChunkKey data_chunk_key = {
          db_id_, foreign_table_->tableId, column_id, fragment_id, 1};
      auto buffer = shared::get_from_map(required_buffers, data_chunk_key);
      chunk.setBuffer(buffer);
      CHECK_EQ(buffer->size(), 0U);
      ChunkKey index_chunk_key = {
          db_id_, foreign_table_->tableId, column_id, fragment_id, 2};
      auto index_buffer = shared::get_from_map(required_buffers, index_chunk_key);
      CHECK_EQ(index_buffer->size(), 0U);
      chunk.setIndexBuffer(index_buffer);
    } else {
      ChunkKey chunk_key = {db_id_, foreign_table_->tableId, column_id, fragment_id};
      auto buffer = shared::get_from_map(required_buffers, chunk_key);
      chunk.setBuffer(buffer);
      CHECK_EQ(buffer->size(), 0U);
    }
    chunks.emplace_back(chunk);
  }
  return chunks;
}

void OdbcDataWrapper::updateGeoChunkMetadata(
    const std::map<ChunkKey, AbstractBuffer*>& required_buffers,
    const std::list<Chunk_NS::Chunk>& chunks,
    const int fragment_id,
    std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata) {
  CHECK(chunks.size() == chunk_metadata.size());
  auto metada_iter = chunk_metadata.begin();
  auto chunk_iter = chunks.begin();
  for (; chunk_iter != chunks.end(); ++chunk_iter, ++metada_iter) {
    auto& chunk = *chunk_iter;
    auto& loaded_metadata = *metada_iter;
    auto column = chunk.getColumnDesc();
    auto column_id = column->columnId;
    ChunkKey data_chunk_key = {db_id_, foreign_table_->tableId, column_id, fragment_id};
    if (column->columnType.is_varlen_indeed()) {
      data_chunk_key.emplace_back(1);
    }

    // Allocate new shared_ptr for metadata so we dont modify old one which may be used by
    // executor
    auto cached_metadata_previous =
        shared::get_from_map(chunk_metadata_map_, data_chunk_key);
    shared::get_from_map(chunk_metadata_map_, data_chunk_key) =
        std::make_shared<ChunkMetadata>();
    auto cached_metadata = shared::get_from_map(chunk_metadata_map_, data_chunk_key);
    *cached_metadata = *cached_metadata_previous;

    CHECK(required_buffers.find(data_chunk_key) != required_buffers.end());
    auto required_buffer = shared::get_from_map(required_buffers, data_chunk_key);
    cached_metadata->numBytes = required_buffer->size();

    cached_metadata->chunkStats = loaded_metadata->chunkStats;

    // Update stats on buffer so it is saved in cache
    required_buffer->getEncoder()->setChunkStats(cached_metadata->chunkStats);
  }
}

void OdbcDataWrapper::processGeoBuffer(
    const std::map<ChunkKey, AbstractBuffer*>& required_buffers,
    const std::list<const ColumnDescriptor*>& column_descriptors,
    const ChunkKey& key,
    const size_t thread_count,
    AbstractBuffer* delete_buffer) {
  auto logical_column_descriptor = *column_descriptors.begin();
  CHECK(is_geometry_column(logical_column_descriptor));
  CHECK_GE(key.size(), size_t(4))
      << "Invalid chunk key supplied [" << show_chunk(key) << "]";
  auto metadata_itr = chunk_metadata_map_.find(key);
  CHECK(chunk_metadata_map_.end() != metadata_itr)
      << "Metadata key [" << show_chunk(key) << "] not found in meta data map";

  const int fragment_id = key[CHUNK_KEY_FRAGMENT_IDX];
  auto chunks = initializeGeoChunks(required_buffers, column_descriptors, fragment_id);

  auto chunk_metadata = create_geometry_chunk_metadata(chunks);
  const bool geo_validate_geometry =
      foreign_table_->getOptionAsBool(ForeignTable::GEO_VALIDATE_GEOMETRY_KEY);
  auto encoder = std::make_unique<foreign_storage::OdbcGeospatialEncoder>(
      chunks,
      chunk_metadata,
      delete_buffer,
      logical_column_descriptor->columnType,
      geo_validate_geometry);

  std::map<size_t, std::vector<std::string>> geo_string_container;
  std::mutex container_mutex;
  auto result_set_processor = [&](size_t num_rows,
                                  size_t data_size,
                                  size_t row_offset,
                                  RemoteData& remote_data) {
    std::lock_guard<std::mutex> callback_lock(container_mutex);
    geo_string_container.insert(
        {row_offset, convert_block_to_vector_string(num_rows, data_size, remote_data)});
  };

  processRemoteDataWithThreads(
      key,
      thread_count,
      result_set_processor,
      getDefaultSelectDescriptor(logical_column_descriptor, key));

  size_t chunk_size = 0;
  for (auto& [row_start, geo_strings] : geo_string_container) {
    std::optional<std::set<size_t>> rejected_row_local_indices;
    if (delete_buffer) {
      rejected_row_local_indices.emplace();
    }
    encoder->appendData(geo_strings, rejected_row_local_indices);

    if (delete_buffer) {
      update_delete_buffer(delete_buffer,
                           chunk_size,
                           geo_strings.size(),
                           rejected_row_local_indices.value(),
                           delete_buffer_mutex_);
      chunk_size += geo_strings.size();
    }
  }

  updateGeoChunkMetadata(required_buffers, chunks, fragment_id, chunk_metadata);
}

namespace {
// Encode NULL values into data array, also check if reserved NULL value is present in
// data array and throw exception
template <typename RemoteType, typename Type>
void encode_null_values(foreign_storage::RemoteData& remote_data,
                        int8_t* dest,
                        const SQLTypeInfo& type_info,
                        RowSetResult& result) {
  RemoteType* remote_data_ptr = reinterpret_cast<RemoteType*>(remote_data.data_ptr);
  Type* dest_ptr = reinterpret_cast<Type*>(dest);
  auto [min_value, max_value] = foreign_storage::get_min_max_bounds<Type>();

  for (size_t i = 0; i < remote_data.null_or_strlen.size(); i++) {
    if (!remote_data.null_or_strlen[i].has_value()) {
      // is null, encode null_value into data array
      dest_ptr[i] = foreign_storage::get_null_value<Type>();
      if (type_info.get_notnull() && result.track_rejected_rows) {
        result.rejected_rows.insert(i);
      }
      continue;
    }

    bool min_value_exceeded = false;
    if constexpr (std::is_signed<RemoteType>::value) {
      // check if min value exceeded only in signed case
      min_value_exceeded = remote_data_ptr[i] < min_value;
    }
    static_assert(static_cast<RemoteType>(std::numeric_limits<Type>::max()) <=
                  std::numeric_limits<RemoteType>::max());
    if (min_value_exceeded || remote_data_ptr[i] > static_cast<RemoteType>(max_value)) {
      if (!result.track_rejected_rows) {
        std::stringstream error_message;
        error_message << "ODBC column contains values that are outside the range of the "
                         "database column type "
                      << type_info.get_type_name()
                      << ". Min allowed value: " << +min_value
                      << ". Max allowed value: " << +max_value
                      << ". Encountered value: " << +remote_data_ptr[i] << ".";
        throw foreign_storage::ForeignStorageException(error_message.str());
      } else {
        result.rejected_rows.insert(i);
        dest_ptr[i] = foreign_storage::get_null_value<Type>();
      }
    } else {
      dest_ptr[i] = remote_data_ptr[i];
    }
  }
}

using NumericEncKey =
    std::pair<int32_t /*odbc_octet_transfer_size*/, int /*type_info_size*/>;
using NumericEnc =
    std::function<void(RemoteData&, int8_t*, const SQLTypeInfo&, RowSetResult& result)>;
const std::map<NumericEncKey, NumericEnc> signed_int_enc_map = {
    // one-to-one
    {{8, 8}, &encode_null_values<int64_t, int64_t>},
    {{4, 4}, &encode_null_values<int32_t, int32_t>},
    {{2, 2}, &encode_null_values<int16_t, int16_t>},
    {{1, 1}, &encode_null_values<int8_t, int8_t>},

    // narrowing
    {{8, 4}, &encode_null_values<int64_t, int32_t>},
    {{8, 2}, &encode_null_values<int64_t, int16_t>},
    {{8, 1}, &encode_null_values<int64_t, int8_t>},

    {{4, 2}, &encode_null_values<int32_t, int16_t>},
    {{4, 1}, &encode_null_values<int32_t, int8_t>},

    {{2, 1}, &encode_null_values<int16_t, int8_t>},

};

const std::map<NumericEncKey, NumericEnc> unsigned_int_enc_map = {
    // one-to-one
    {{4, 4}, &encode_null_values<uint32_t, int32_t>},
    {{2, 2}, &encode_null_values<uint16_t, int16_t>},
    {{1, 1}, &encode_null_values<uint8_t, int8_t>},

    // narrowing
    {{4, 2}, &encode_null_values<uint32_t, int16_t>},
    {{4, 1}, &encode_null_values<uint32_t, int8_t>},

    {{2, 1}, &encode_null_values<uint16_t, int8_t>},

    // widening
    {{4, 8}, &encode_null_values<uint32_t, int64_t>},
    {{2, 4}, &encode_null_values<uint16_t, int32_t>},
    {{1, 2}, &encode_null_values<uint8_t, int16_t>},
};
const std::map<NumericEncKey, NumericEnc> fp_enc_map = {
    // one-to-one
    {{8, 8}, &encode_null_values<double_t, double_t>},
    {{4, 4}, &encode_null_values<float_t, float_t>},

    // narrowing
    {{8, 4}, &encode_null_values<double_t, float_t>},
};
const std::map<NumericEncKey, NumericEnc> bool_enc_map = {
    // one-to-one
    {{1, 1}, &encode_null_values<int8_t, int8_t>},
};

void load_numerics_into_data_block(RemoteData& remote_data,
                                   const SQLTypeInfo& type_info,
                                   RowSetResult& result) {
  result.numeric_types.resize(type_info.get_size() * remote_data.null_or_strlen.size());
  int8_t* dest_ptr = &result.numeric_types[0];
  result.data_block_ptr.numbersPtr = dest_ptr;
  NumericEncKey key = {remote_data.odbc_octet_transfer_size, type_info.get_size()};

  if (type_info.is_integer()) {
    std::map<NumericEncKey, NumericEnc>::const_iterator itr;
    bool is_found;
    if (remote_data.is_unsigned) {
      itr = unsigned_int_enc_map.find(key);
      is_found = itr != unsigned_int_enc_map.end();
    } else {
      itr = signed_int_enc_map.find(key);
      is_found = itr != signed_int_enc_map.end();
    }
    CHECK(is_found) << "Unhandled type: key={" << key.first << ", " << key.second
                    << "}, type='" << type_info.get_type_name() << "'.";
    itr->second(remote_data, dest_ptr, type_info, result);

  } else if (type_info.is_fp()) {
    auto itr = fp_enc_map.find(key);
    CHECK(itr != fp_enc_map.end()) << "Unhandled type: " << type_info.get_type_name();
    itr->second(remote_data, dest_ptr, type_info, result);

  } else if (type_info.is_boolean()) {
    // ensure encode_null_values<int8_t> encodes NULLs as NULL_BOOLEAN
    static_assert(inline_int_null_value<int8_t>() == NULL_BOOLEAN);
    encode_null_values<int8_t, int8_t>(remote_data, dest_ptr, type_info, result);

  } else {
    UNREACHABLE() << "Unhandled type: " << type_info.get_type_name();
  }
}

}  // namespace

void OdbcDataWrapper::appendRemoteDataToChunk(
    const ColumnDescriptor* column_descriptor,
    Data_Namespace::AbstractBuffer* data_buffer,
    Data_Namespace::AbstractBuffer* index_buffer,
    Data_Namespace::AbstractBuffer* delete_buffer,
    const ChunkKey& key,
    Chunk_NS::Chunk& chunk,
    const int thread_count,
    const OdbcSelectDescriptor& select_desc) {
  chunk.setBuffer(data_buffer);

  chunk.setIndexBuffer(index_buffer);
  // Sets the correct encoder on data_buffer based on column_descriptor supplied
  // earlier.
  chunk.initEncoder();

  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);

  std::map<int64_t, RowSetResult> data_block_container;
  std::mutex container_mutex;
  auto result_set_processor = [&](size_t num_rows,
                                  size_t data_size,
                                  size_t row_offset,
                                  RemoteData& remote_data) {
    RowSetResult* data_block;
    {
      std::lock_guard<std::mutex> callback_lock(container_mutex);
      CHECK(data_block_container.find(row_offset) == data_block_container.end())
          << "Result set already exists";
      data_block = &data_block_container[row_offset];
      data_block->track_rejected_rows = delete_buffer;
    }
    data_block->num_rows = num_rows;
    if (column_descriptor->columnType.is_timestamp() ||
        column_descriptor->columnType.is_time() ||
        column_descriptor->columnType.is_date()) {
      load_temporal_into_data_block(
          remote_data, num_rows, data_size, column_descriptor->columnType, *data_block);
    } else if (column_descriptor->columnType.is_string()) {
      load_strings_into_data_block(
          remote_data,
          num_rows,
          data_size,
          column_descriptor->columnType.is_dict_encoded_string(),
          column_descriptor->columnType,
          catalog->getMetadataForDict(column_descriptor->columnType.get_comp_param(),
                                      true),
          *data_block);
    } else if (column_descriptor->columnType.is_decimal()) {
      load_decimals_into_data_block(
          remote_data, num_rows, data_size, column_descriptor, *data_block);
    } else {
      if (foreign_storage::is_decimal_odbc_type(remote_data.odbc_base_type)) {
        load_decimals_into_data_block(
            remote_data, num_rows, data_size, column_descriptor, *data_block);
      } else {
        load_numerics_into_data_block(
            remote_data, column_descriptor->columnType, *data_block);
      }
    }
  };
  processRemoteDataWithThreads(key, thread_count, result_set_processor, select_desc);

  size_t chunk_size = 0;
  for (auto& [row_start, data_block_container] : data_block_container) {
    if (delete_buffer) {
      update_delete_buffer(delete_buffer,
                           chunk_size,
                           data_block_container.num_rows,
                           data_block_container.rejected_rows,
                           delete_buffer_mutex_);
      chunk_size += data_block_container.num_rows;
    }

    chunk.appendData(
        data_block_container.data_block_ptr, data_block_container.num_rows, 0);
  }
}

void OdbcDataWrapper::processChunkbuffer(
    const std::map<ChunkKey, AbstractBuffer*>& required_buffers,
    const ColumnDescriptor* column_descriptor,
    const ChunkKey& key,
    const size_t thread_count,
    AbstractBuffer* delete_buffer) {
  CHECK(!is_varlen_index_key(key));
  auto data_buffer = shared::get_from_map(required_buffers, key);
  CHECK_EQ(data_buffer->size(), 0U);
  CHECK_GE(key.size(), size_t(4))
      << "Invalid chunk key supplied [" << show_chunk(key) << "]";

  CHECK(column_descriptor) << "Null column descriptor returned for chunk key ["
                           << show_chunk(key) << "]";

  AbstractBuffer* index_buffer = nullptr;

  if (is_varlen_data_key(key)) {
    // make sure there is an empty index buffer in the required buffer map
    ChunkKey index_key = key;
    index_key[CHUNK_KEY_VARLEN_IDX] = 2;
    auto itr = required_buffers.find(index_key);
    CHECK(itr != required_buffers.end())
        << "Variable data block (key " << show_chunk(key)
        << ") supplied without matching index block (key " << show_chunk(index_key)
        << ")";
    index_buffer = itr->second;
    CHECK_EQ(index_buffer->size(), 0U);
  }

  auto metadata_itr = chunk_metadata_map_.find(key);
  CHECK(chunk_metadata_map_.end() != metadata_itr)
      << "Metadata key [" << show_chunk(key) << "] not found in meta data map";

  /* Create a chunk, allocate its buffers and set encoder (partially done in downstream
   * call) */
  auto chunk = Chunk_NS::Chunk{column_descriptor, false};

  // if type is columnType.is_varlen_indeed() numBytes will be zero.
  data_buffer->reserve(metadata_itr->second->numBytes);

  appendRemoteDataToChunk(column_descriptor,
                          data_buffer,
                          index_buffer,
                          delete_buffer,
                          key,
                          chunk,
                          thread_count,
                          getDefaultSelectDescriptor(column_descriptor, key));

  // Update cached metadata and fragmenter
  auto& cached_metadata = chunk_metadata_map_[key];
  cached_metadata->numBytes = data_buffer->size();
  cached_metadata->chunkStats = chunk.getBuffer()->getEncoder()->getChunkStats();
}

namespace {

std::list<const ColumnDescriptor*> get_column_descriptors_for_geo_column(
    const foreign_storage::ForeignTableSchema& schema,
    const ChunkKey& key,
    const Catalog_Namespace::Catalog& catalog) {
  int column_id = key[CHUNK_KEY_COLUMN_IDX];
  auto logical_column = schema.getLogicalColumn(column_id);
  int logical_column_id = logical_column->columnId;
  CHECK(is_geometry_column(logical_column));

  const Interval<ColumnType> column_interval = {
      logical_column_id,
      logical_column_id + logical_column->columnType.get_physical_cols()};

  std::list<const ColumnDescriptor*> column_descriptors;
  std::list<ChunkKey> keys;
  for (int column_id = column_interval.start; column_id <= column_interval.end;
       ++column_id) {
    auto column_descriptor = schema.getColumnDescriptor(column_id);
    column_descriptors.emplace_back(column_descriptor);
  }

  return column_descriptors;
}
};  // namespace

void OdbcDataWrapper::populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                                           const ChunkToBufferMap& optional_buffers,
                                           AbstractBuffer* delete_buffer) {
  ChunkToBufferMap buffers_to_load;
  buffers_to_load.insert(required_buffers.begin(), required_buffers.end());
  buffers_to_load.insert(optional_buffers.begin(), optional_buffers.end());

  CHECK(!buffers_to_load.empty());

  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);
  CHECK(catalog);
  foreign_storage::ForeignTableSchema schema(catalog->getDatabaseId(), foreign_table_);
  std::set<ForeignStorageMgr::ParallelismHint> col_frag_hints;
  for (const auto& [chunk_key, buffer] : buffers_to_load) {
    CHECK_EQ(buffer->size(), static_cast<size_t>(0));
    col_frag_hints.emplace(
        schema.getLogicalColumn(chunk_key[CHUNK_KEY_COLUMN_IDX])->columnId,
        chunk_key[CHUNK_KEY_FRAGMENT_IDX]);
  }
  auto max_threads = std::min(static_cast<size_t>(cpu_threads()), g_max_import_threads);
  auto hints_per_thread = partition_for_threads(col_frag_hints, max_threads);
  std::vector<std::future<void>> futures;

  auto num_sub_threads = divide_m_into_n(max_threads, hints_per_thread.size());
  int thread = 0;
  for (const auto& hint_set : hints_per_thread) {
    auto sub_threads = num_sub_threads[thread++];
    futures.emplace_back(std::async(std::launch::async, [&, hint_set, sub_threads, this] {
      for (const auto& [column_id, fragment_id] : hint_set) {
        const ColumnDescriptor* column_descriptor = schema.getColumnDescriptor(column_id);
        ChunkKey key = {db_id_, foreign_table_->tableId, column_id, fragment_id};
        CHECK(column_descriptor) << "Null column descriptor returned for chunk key ["
                                 << show_chunk(key) << "]";
        if (column_descriptor->columnType.is_varlen_indeed()) {
          key.push_back(1);
        }
        if (is_geometry_column(column_descriptor)) {
          processGeoBuffer(buffers_to_load,
                           get_column_descriptors_for_geo_column(schema, key, *catalog),
                           key,
                           sub_threads,
                           delete_buffer);
        } else {
          // index chunks are processed at the same time as the data chunk.
          processChunkbuffer(
              buffers_to_load, column_descriptor, key, sub_threads, delete_buffer);
        }
      }
    }));
  }

  for (auto& future : futures) {
    // Wait for all threads to finish.
    future.wait();
  }
  for (auto& future : futures) {
    // Propagate any found exceptions.
    future.get();
  }
}

std::string OdbcDataWrapper::getSerializedDataWrapper() const {
  auto cached_metadata = chunk_metadata_map_;
  rapidjson::Document document;
  document.SetObject();
  json_utils::add_value_to_object(document,
                                  fragment_remote_db_rownumber_start_,
                                  "fragment_remote_db_rownumber_start_",
                                  document.GetAllocator());
  json_add_column_descriptions(document, remote_column_details_);
  json_utils::add_value_to_object(
      document, total_row_count_, "total_row_count_", document.GetAllocator());
  auto string_doc = json_utils::write_to_string(document);
  LOG(INFO) << string_doc;
  return string_doc;
}

void OdbcDataWrapper::restoreDataWrapperInternals(
    const std::string& file_path,
    const ChunkMetadataVector& chunk_metadata) {
  auto document = json_utils::read_from_file(file_path);
  CHECK(document.IsObject());
  json_utils::get_value_from_object(document,
                                    fragment_remote_db_rownumber_start_,
                                    "fragment_remote_db_rownumber_start_");
  auto cds = document.FindMember("column_values");
  CHECK(cds != document.MemberEnd()) << "restoreDataWrapperInternals error in json "
                                        "format. 'column_values' key not found";
  CHECK(cds->value.IsArray()) << "Error in json format. 'column_values' wrong type "
                              << json_utils::get_type_as_string(cds->value) << " ["
                              << cds->value.GetType() << "] found.";
  json_extract_column_descriptions(cds->value.GetArray(), remote_column_details_);
  json_utils::get_value_from_object(document, total_row_count_, "total_row_count_");

  std::copy(chunk_metadata.begin(),
            chunk_metadata.end(),
            std::inserter(chunk_metadata_map_, chunk_metadata_map_.begin()));

  is_restored_ = true;
}

bool OdbcDataWrapper::isRestored() const {
  return is_restored_;
}

OdbcConnectionInfo OdbcDataWrapper::getDbConnectionInfo() const {
  auto& server_options = foreign_table_->foreign_server->options;
  if (auto entry = server_options.find(ODBC_DSN_KEY); entry != server_options.end()) {
    return {entry->second, std::nullopt};
  } else {
    entry = server_options.find(ODBC_CONNECTION_KEY);
    CHECK(entry != server_options.end())
        << "Error db connection string not supplied.  Required for "
           "remote ODBC connection";
    return {std::nullopt, entry->second};
  }
}

void OdbcDataWrapper::processRemoteDataWithThreads(
    const ChunkKey& key,
    const size_t thread_count,
    ResultSetProcessor result_set_processor,
    const OdbcSelectDescriptor& select_desc) {
  auto buffer_byte_size = get_odbc_buffer_byte_size(foreign_table_->options);
  std::vector<std::future<void>> futures;
  auto start_row = fragment_remote_db_rownumber_start_[key[CHUNK_KEY_FRAGMENT_IDX]];
  auto metadata_itr = chunk_metadata_map_.find(key);
  CHECK(metadata_itr != chunk_metadata_map_.end());
  size_t num_rows_to_process = metadata_itr->second->numElements;
  auto num_rows_per_thread = divide_m_into_n(num_rows_to_process, thread_count);

  for (size_t i = 0; i < thread_count; i++) {
    if (num_rows_per_thread[i] == 0) {
      continue;
    }
    futures.emplace_back(std::async(std::launch::async, [&, start_row, i, this] {
      OdbcSelectDescriptor thread_local_select_desc = select_desc;
      thread_local_select_desc.offset = start_row;
      thread_local_select_desc.limit = num_rows_per_thread[i];
      processRemoteDataSource(
          buffer_byte_size, result_set_processor, thread_local_select_desc);
    }));
    start_row += num_rows_per_thread[i];
  }
  for (auto& future : futures) {
    // Wait for all threads to finish.
    future.wait();
  }
  for (auto& future : futures) {
    // Propagate any found exceptions.
    future.get();
  }
}

std::string OdbcDataWrapper::getDbSelect() const {
  const auto sql_select_entry = foreign_table_->getOption(ODBC_SELECT_KEY);
  CHECK(sql_select_entry.has_value() && !sql_select_entry.value().empty());

  // To wrap the statement in limits etc any ';'
  // in the sql will get in the way and need to be removed.
  // At the moment the UI takes a const pointer to the foreign_table_
  // so it seemed best to do it here.
  static std::regex trailing_semicolon("\\s*;\\s*$");
  return std::regex_replace(sql_select_entry.value(), trailing_semicolon, "");
}

namespace {
void set_column_data(SampleRows& sample_rows,
                     const ColumnRemoteData& column_remote_data,
                     size_t column_index) {
  for (size_t row_index = 0; row_index < column_remote_data.row_count; row_index++) {
    if (!column_remote_data.null_or_strlen[row_index].has_value()) {
      sample_rows[row_index][column_index] = "NULL";
    } else {
      const auto data_size =
          column_remote_data.remote_column_description.odbc_octet_transfer_size;
      const auto data_ptr = column_remote_data.data_ptr.get() + (row_index * data_size);
      const auto type = column_remote_data.remote_column_description.omnisci_type;
      if (type == kTIME || type == kTIMESTAMP || type == kDATE) {
        sample_rows[row_index][column_index] =
            temporal_to_string(data_ptr, SQLTypeInfo{type});
      } else if (type == kTEXT || type == kDECIMAL) {
        auto size = column_remote_data.null_or_strlen[row_index].value();
        sample_rows[row_index][column_index] =
            std::string{reinterpret_cast<char*>(data_ptr), size};
      } else if (type == kBOOLEAN) {
        auto value = reinterpret_cast<bool*>(data_ptr);
        sample_rows[row_index][column_index] = (*value) ? "true" : "false";
      } else if (type == kTINYINT) {
        auto value = reinterpret_cast<int8_t*>(data_ptr);
        sample_rows[row_index][column_index] = std::to_string(*value);
      } else if (type == kSMALLINT) {
        auto value = reinterpret_cast<int16_t*>(data_ptr);
        sample_rows[row_index][column_index] = std::to_string(*value);
      } else if (type == kINT) {
        auto value = reinterpret_cast<int32_t*>(data_ptr);
        sample_rows[row_index][column_index] = std::to_string(*value);
      } else if (type == kBIGINT) {
        auto value = reinterpret_cast<int64_t*>(data_ptr);
        sample_rows[row_index][column_index] = std::to_string(*value);
      } else if (type == kFLOAT) {
        auto value = reinterpret_cast<float*>(data_ptr);
        sample_rows[row_index][column_index] = std::to_string(*value);
      } else if (type == kDOUBLE) {
        auto value = reinterpret_cast<double*>(data_ptr);
        sample_rows[row_index][column_index] = std::to_string(*value);
      } else {
        UNREACHABLE() << "Unexpected type when setting column data. Type: "
                      << toString(type);
      }
    }
  }
}

SQLTypeInfo get_integer_type_for_precision(int32_t precision) {
  SQLTypes type;
  if (precision <= std::numeric_limits<int8_t>::digits10) {
    type = kTINYINT;
  } else if (precision <= std::numeric_limits<int16_t>::digits10) {
    type = kSMALLINT;
  } else if (precision <= std::numeric_limits<int32_t>::digits10) {
    type = kINT;
  } else {
    type = kBIGINT;
  }
  return {type};
}
}  // namespace

DataPreview OdbcDataWrapper::getDataPreview(size_t max_row_count) const {
  DataPreview preview;
  auto odbc_connection = OdbcConnection::create(getDbConnectionInfo(), user_mapping_);
  auto column_remote_data_vec =
      odbc_connection->runDataPreviewMultiColumnSelectCmd(getDbSelect(), max_row_count);
  auto column_count = column_remote_data_vec.size();
  for (size_t column_index = 0; column_index < column_count; column_index++) {
    const auto& remote_column_description =
        column_remote_data_vec[column_index].remote_column_description;
    preview.column_types.emplace_back(
        SQLTypeInfo{remote_column_description.omnisci_type});
    preview.column_names.emplace_back(remote_column_description.column_name);
    auto row_count = column_remote_data_vec[column_index].row_count;
    if (preview.sample_rows.empty()) {
      preview.sample_rows.resize(row_count);
    }
    for (size_t row_index = 0; row_index < row_count; row_index++) {
      if (preview.sample_rows[row_index].empty()) {
        preview.sample_rows[row_index].resize(column_count);
      }
    }
    set_column_data(
        preview.sample_rows, column_remote_data_vec[column_index], column_index);
    auto& type_info = preview.column_types.back();
    if (type_info.is_string()) {
      auto geo_type = foreign_storage::detect_geo_type(preview.sample_rows, column_index);
      if (geo_type.has_value()) {
        type_info.set_type(geo_type.value());
      } else {
        type_info.set_compression(kENCODING_DICT);
      }
    } else if (type_info.is_decimal()) {
      auto precision = remote_column_description.getDecimalPrecision();
      auto scale = remote_column_description.decimal_digits;
      if (scale == 0) {
        type_info = get_integer_type_for_precision(precision);
      } else if (precision > sql_constants::kMaxNumericPrecision) {
        std::string warning_message{
            "Remote column \"" + remote_column_description.column_name +
            "\" has decimal precision of " + std::to_string(precision) +
            " which exceeds the maximum supported decimal precision. Remote column "
            "data will be coerced to the supported decimal precision of " +
            std::to_string(sql_constants::kMaxNumericPrecision) +
            " and decimal scale of " + std::to_string(9) + "."};
        LOG(INFO) << warning_message;
        type_info.set_precision(sql_constants::kMaxNumericPrecision);
        type_info.set_scale(9);
      } else {
        type_info.set_precision(precision);
        type_info.set_scale(scale);
      }
    }
  }
  return preview;
}

std::set<std::string> OdbcDataWrapper::getSupportedSubDatasourceTypes() const {
  return OdbcConnection::getInstalledDrivers();
}

const std::set<std::string_view> OdbcDataWrapper::supported_table_options_{
    ODBC_SELECT_KEY,
    ODBC_BUFFER_SIZE_KEY,
    ODBC_ORDER_BY_KEY};
const std::set<std::string_view> OdbcDataWrapper::supported_server_options_{
    ODBC_DSN_KEY,
    ODBC_CONNECTION_KEY};
const std::set<std::string_view> OdbcDataWrapper::supported_user_mapping_options_{
    ODBC_USERNAME,
    ODBC_PASSWORD,
    ODBC_CREDENTIAL};

const std::set<std::string> OdbcDataWrapper::getAlterableTableOptions() const {
  return {ODBC_ORDER_BY_KEY, ODBC_SELECT_KEY};
}

namespace {
using IncompatibleOptions =
    std::pair<std::string /*credential_option*/, std::string /*connection_option*/>;

void odbc_option_incompatible(const bool is_import,
                              const IncompatibleOptions& import_options,
                              const IncompatibleOptions& fsi_options,
                              const std::optional<std::string>& server_name) {
  if (is_import) {
    throw ForeignStorageException{
        "The ODBC credential option \"" + import_options.first +
        "\" is incompatible with the ODBC connection option \"" + import_options.second +
        "\"."};
  } else {
    CHECK(server_name.has_value());
    throw ForeignStorageException{
        "The user mapping option \"" + fsi_options.first +
        "\" is incompatible with the foreign server \"" + server_name.value() +
        "\" which sets the server option \"" + fsi_options.second + "\"."};
  }
}
}  // namespace

void validate_odbc_credential_options(const std::string& dsn,
                                      const std::string& connection_string,
                                      const std::string& credential_string,
                                      const std::string& username,
                                      const std::string& password,
                                      const bool is_import,
                                      const std::optional<std::string>& server_name) {
  if (!password.empty() && username.empty()) {
    if (is_import) {
      throw ForeignStorageException{
          "ODBC option password requires a matching username option."};
    } else {
      throw ForeignStorageException{
          "User mapping option \"" + OdbcDataWrapper::ODBC_PASSWORD +
          "\" requires a matching \"" + OdbcDataWrapper::ODBC_USERNAME + "\" option."};
    }
  }

  if (!credential_string.empty() && !(username.empty() && password.empty())) {
    if (is_import) {
      throw ForeignStorageException{
          "ODBC options must contain only one of username/password or credential "
          "string."};
    } else {
      throw ForeignStorageException{"User mapping must contain only one of \"" +
                                    OdbcDataWrapper::ODBC_USERNAME + "/" +
                                    OdbcDataWrapper::ODBC_PASSWORD + "\" or \"" +
                                    OdbcDataWrapper::ODBC_CREDENTIAL + "\"."};
    }
  }

  if (!dsn.empty()) {
    if (!credential_string.empty()) {
      odbc_option_incompatible(
          is_import,
          {"credential_string", "data_source_name"},
          {OdbcDataWrapper::ODBC_CREDENTIAL, OdbcDataWrapper::ODBC_DSN_KEY},
          server_name);
    }
    return;
  } else if (!connection_string.empty()) {
    if (!username.empty()) {
      odbc_option_incompatible(
          is_import,
          {"username", "connection_string"},
          {OdbcDataWrapper::ODBC_USERNAME, OdbcDataWrapper::ODBC_CONNECTION_KEY},
          server_name);
    } else if (!password.empty()) {
      odbc_option_incompatible(
          is_import,
          {"password", "connection_string"},
          {OdbcDataWrapper::ODBC_PASSWORD, OdbcDataWrapper::ODBC_CONNECTION_KEY},
          server_name);
    }
    return;
  }
  // This should have already been checked at an earlier point
  UNREACHABLE() << "Either 'DATA_SOURCE_NAME' or 'CONNECTION_STRING' must be specified";
}
}  // namespace foreign_storage
