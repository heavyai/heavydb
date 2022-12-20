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

#include "ForeignDataImporter.h"

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <filesystem>

#include "Archive/S3Archive.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapperFactory.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "DataMgr/ForeignStorage/ParquetImporter.h"
#include "Importer.h"
#include "Parser/ParserNode.h"
#include "Shared/file_path_util.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/scope.h"
#include "UserMapping.h"

extern bool g_enable_legacy_delimited_import;
#ifdef ENABLE_IMPORT_PARQUET
extern bool g_enable_legacy_parquet_import;
#endif
extern bool g_enable_fsi_regex_import;

namespace {
ChunkMetadataVector metadata_scan(foreign_storage::ForeignDataWrapper* data_wrapper,
                                  foreign_storage::ForeignTable* foreign_table) {
  ChunkMetadataVector metadata_vector;
  try {
    data_wrapper->populateChunkMetadata(
        metadata_vector);  // explicitly invoke a metadata scan on data wrapper
  } catch (const foreign_storage::MetadataScanInfeasibleFragmentSizeException&
               metadata_scan_exception) {
    // if a metadata scan exception is thrown, check to see if we can adjust
    // the fragment size and retry

    auto min_feasible_fragment_size = metadata_scan_exception.min_feasible_fragment_size_;
    if (min_feasible_fragment_size < 0) {
      throw;  // no valid fragment size returned by exception
    }
    foreign_table->maxFragRows = min_feasible_fragment_size;
    data_wrapper->populateChunkMetadata(
        metadata_vector);  // attempt another metadata scan, note, we assume that the
                           // metadata scan can be reentered safely after throwing the
                           // exception
  }
  return metadata_vector;
}

std::string get_import_id(const import_export::CopyParams& copy_params,
                          const std::string& copy_from_source) {
  if (copy_params.source_type == import_export::SourceType::kDelimitedFile ||
      copy_params.source_type == import_export::SourceType::kRegexParsedFile
#ifdef ENABLE_IMPORT_PARQUET
      || copy_params.source_type == import_export::SourceType::kParquetFile
#endif
  ) {
    return boost::filesystem::path(copy_from_source).filename().string();
  }

  return copy_from_source;
}

void validate_copy_params(const import_export::CopyParams& copy_params) {
  if (copy_params.source_type == import_export::SourceType::kRegexParsedFile) {
    foreign_storage::validate_regex_parser_options(copy_params);
  }
}

struct FragmentBuffers {
  std::map<ChunkKey, std::unique_ptr<foreign_storage::ForeignStorageBuffer>>
      fragment_buffers_owner;
  foreign_storage::ChunkToBufferMap fragment_buffers;
  std::unique_ptr<foreign_storage::ForeignStorageBuffer> delete_buffer;
};

std::unique_ptr<FragmentBuffers> create_fragment_buffers(
    const int32_t fragment_id,
    Catalog_Namespace::Catalog& catalog,
    const TableDescriptor* table) {
  auto columns = catalog.getAllColumnMetadataForTable(table->tableId, false, false, true);

  std::set<ChunkKey> fragment_keys;
  for (const auto col_desc : columns) {
    ChunkKey key{
        catalog.getDatabaseId(), table->tableId, col_desc->columnId, fragment_id};
    if (col_desc->columnType.is_varlen_indeed()) {
      auto data_key = key;
      data_key.push_back(1);
      fragment_keys.insert(data_key);
      auto index_key = key;
      index_key.push_back(2);
      fragment_keys.insert(index_key);
    } else {
      fragment_keys.insert(key);
    }
  }

  // create buffers
  std::unique_ptr<FragmentBuffers> frag_buffers = std::make_unique<FragmentBuffers>();
  frag_buffers->delete_buffer = std::make_unique<foreign_storage::ForeignStorageBuffer>();
  for (const auto& key : fragment_keys) {
    frag_buffers->fragment_buffers_owner[key] =
        std::make_unique<foreign_storage::ForeignStorageBuffer>();
    frag_buffers->fragment_buffers[key] =
        shared::get_from_map(frag_buffers->fragment_buffers_owner, key).get();
  }

  return frag_buffers;
}

void load_foreign_data_buffers(
    Fragmenter_Namespace::InsertDataLoader::InsertConnector* connector,
    Catalog_Namespace::Catalog& catalog,
    const TableDescriptor* table,
    const Catalog_Namespace::SessionInfo* session_info,
    const import_export::CopyParams& copy_params,
    const std::string& copy_from_source,
    import_export::ImportStatus& import_status,
    std::mutex& communication_mutex,
    bool& continue_loading,
    bool& load_failed,
    bool& data_wrapper_error_occured,
    std::condition_variable& buffers_to_load_condition,
    std::list<std::unique_ptr<FragmentBuffers>>& buffers_to_load) {
  Fragmenter_Namespace::InsertDataLoader insert_data_loader(*connector);
  while (true) {
    {
      std::unique_lock communication_lock(communication_mutex);
      buffers_to_load_condition.wait(communication_lock, [&]() {
        return !buffers_to_load.empty() || !continue_loading ||
               data_wrapper_error_occured;
      });
      if ((buffers_to_load.empty() && !continue_loading) || data_wrapper_error_occured) {
        return;
      }
    }

    CHECK(!buffers_to_load.empty());

    try {
      std::unique_ptr<FragmentBuffers> grouped_fragment_buffers;
      {
        std::unique_lock communication_lock(communication_mutex);
        grouped_fragment_buffers.reset(buffers_to_load.front().release());
        buffers_to_load.pop_front();
        buffers_to_load_condition.notify_all();
      }
      auto& fragment_buffers = grouped_fragment_buffers->fragment_buffers;
      auto& delete_buffer = grouped_fragment_buffers->delete_buffer;

      // get chunks for import
      Fragmenter_Namespace::InsertChunks insert_chunks{
          table->tableId, catalog.getDatabaseId(), {}, {}};

      // create chunks from buffers
      for (const auto& [key, buffer] : fragment_buffers) {
        const auto col_id = key[CHUNK_KEY_COLUMN_IDX];
        const auto table_id = key[CHUNK_KEY_TABLE_IDX];
        const auto col_desc = catalog.getMetadataForColumn(table_id, col_id);

        if (col_desc->columnType.is_varlen_indeed()) {
          CHECK(key.size() > CHUNK_KEY_VARLEN_IDX);  // check for varlen key
          if (key[CHUNK_KEY_VARLEN_IDX] == 1) {      // data key
            auto index_key = key;
            index_key[CHUNK_KEY_VARLEN_IDX] = 2;
            insert_chunks.chunks[col_id] = Chunk_NS::Chunk::getChunk(
                col_desc,
                buffer,
                shared::get_from_map(fragment_buffers, index_key),
                false);
          }
        } else {  // regular non-varlen case with no index buffer
          insert_chunks.chunks[col_id] =
              Chunk_NS::Chunk::getChunk(col_desc, buffer, nullptr, false);
        }
      }

      // mark which row indices are valid for import
      auto row_count = fragment_buffers.begin()
                           ->second->getEncoder()
                           ->getNumElems();  // assume all chunks have same number of
                                             // rows, this is validated at a lower level
      insert_chunks.valid_row_indices.reserve(row_count);
      for (size_t irow = 0; irow < row_count; ++irow) {
        if (delete_buffer->size() > 0) {
          CHECK_LE(irow, delete_buffer->size());
          if (delete_buffer->getMemoryPtr()[irow]) {
            continue;
          }
        }
        insert_chunks.valid_row_indices.emplace_back(irow);
      }

      // import chunks
      insert_data_loader.insertChunks(*session_info, insert_chunks);

      CHECK_LE(insert_chunks.valid_row_indices.size(), row_count);
      import_status.rows_rejected += row_count - insert_chunks.valid_row_indices.size();
      import_status.rows_completed += insert_chunks.valid_row_indices.size();
      if (import_status.rows_rejected > copy_params.max_reject) {
        import_status.load_failed = true;
        import_status.load_msg =
            "Load was cancelled due to max reject rows being reached";
        import_export::Importer::set_import_status(
            get_import_id(copy_params, copy_from_source), import_status);
        std::unique_lock communication_lock(communication_mutex);
        load_failed = true;
        buffers_to_load_condition.notify_all();
        return;
      }
      import_export::Importer::set_import_status(
          get_import_id(copy_params, copy_from_source), import_status);
    } catch (...) {
      {
        std::unique_lock communication_lock(communication_mutex);
        load_failed = true;
        buffers_to_load_condition.notify_all();
      }
      throw;
    }
  }
}

import_export::ImportStatus import_foreign_data(
    const int32_t max_fragment_id,
    Fragmenter_Namespace::InsertDataLoader::InsertConnector* connector,
    Catalog_Namespace::Catalog& catalog,
    const TableDescriptor* table,
    foreign_storage::ForeignDataWrapper* data_wrapper,
    const Catalog_Namespace::SessionInfo* session_info,
    const import_export::CopyParams& copy_params,
    const std::string& copy_from_source,
    const size_t maximum_num_fragments_buffered) {
  import_export::ImportStatus import_status;

  std::mutex communication_mutex;
  bool continue_loading =
      true;  // when false, signals that the last buffer to load has been added, and
             // loading should cease after loading remaining buffers
  bool load_failed =
      false;  // signals loading has failed and buffer population should cease
  bool data_wrapper_error_occured = false;  // signals an error occured during buffer
                                            // population and loading should cease
  std::condition_variable buffers_to_load_condition;
  std::list<std::unique_ptr<FragmentBuffers>> buffers_to_load;

  // launch separate thread to load processed fragment buffers
  auto load_future = std::async(std::launch::async,
                                load_foreign_data_buffers,
                                connector,
                                std::ref(catalog),
                                table,
                                session_info,
                                std::cref(copy_params),
                                std::cref(copy_from_source),
                                std::ref(import_status),
                                std::ref(communication_mutex),
                                std::ref(continue_loading),
                                std::ref(load_failed),
                                std::ref(data_wrapper_error_occured),
                                std::ref(buffers_to_load_condition),
                                std::ref(buffers_to_load));

  for (int32_t fragment_id = 0; fragment_id <= max_fragment_id; ++fragment_id) {
    {
      std::unique_lock communication_lock(communication_mutex);
      buffers_to_load_condition.wait(communication_lock, [&]() {
        return buffers_to_load.size() < maximum_num_fragments_buffered || load_failed;
      });
      if (load_failed) {
        break;
      }
    }
    auto grouped_fragment_buffers = create_fragment_buffers(fragment_id, catalog, table);
    auto& fragment_buffers = grouped_fragment_buffers->fragment_buffers;
    auto& delete_buffer = grouped_fragment_buffers->delete_buffer;

    // get the buffers, accounting for the possibility of the requested fragment id being
    // out of bounds
    try {
      data_wrapper->populateChunkBuffers(fragment_buffers, {}, delete_buffer.get());
    } catch (const foreign_storage::RequestedFragmentIdOutOfBoundsException& except) {
      break;
    } catch (...) {
      std::unique_lock communication_lock(communication_mutex);
      data_wrapper_error_occured = true;
      buffers_to_load_condition.notify_all();
      throw;
    }

    std::unique_lock communication_lock(communication_mutex);
    buffers_to_load.emplace_back(std::move(grouped_fragment_buffers));
    buffers_to_load_condition.notify_all();
  }

  {  // data wrapper processing has finished, notify loading thread
    std::unique_lock communication_lock(communication_mutex);
    continue_loading = false;
    buffers_to_load_condition.notify_all();
  }

  // any exceptions in separate loading thread will occur here
  load_future.get();

  return import_status;
}

#ifdef HAVE_AWS_S3
struct DownloadedObjectToProcess {
  std::string object_key;
  std::atomic<bool> is_downloaded;
  std::string download_file_path;
  std::string import_file_path;
};

size_t get_number_of_digits(const size_t number) {
  return std::to_string(number).length();
}

std::tuple<std::string, import_export::CopyParams> get_local_copy_source_and_params(
    const import_export::CopyParams& s3_copy_params,
    std::vector<DownloadedObjectToProcess>& objects_to_process,
    const size_t begin_object_index,
    const size_t end_object_index) {
  import_export::CopyParams local_copy_params = s3_copy_params;
  // remove any members from `local_copy_params` that are only intended to be used at a
  // higher level
  local_copy_params.s3_access_key.clear();
  local_copy_params.s3_secret_key.clear();
  local_copy_params.s3_session_token.clear();
  local_copy_params.s3_region.clear();
  local_copy_params.s3_endpoint.clear();

  local_copy_params.regex_path_filter = std::nullopt;
  local_copy_params.file_sort_order_by = "PATHNAME";  // see comment below
  local_copy_params.file_sort_regex = std::nullopt;

  CHECK_GT(end_object_index, begin_object_index);
  CHECK_LT(begin_object_index, objects_to_process.size());

  size_t num_objects = end_object_index - begin_object_index;
  auto& first_object = objects_to_process[begin_object_index];
  std::string first_path = first_object.download_file_path;
  std::string temp_dir = first_path + "_import";

  if (!std::filesystem::create_directory(temp_dir)) {
    throw std::runtime_error("failed to create temporary directory for import: " +
                             temp_dir);
  }

  // construct a directory with files to import
  //
  // NOTE:
  // * files are moved into `temp_dir` in the exact order that they appear in
  // `objects_to_process`
  //
  // * the `PATHNAME` option is set for `file_sort_order_by` in order to
  // guarantee that import occurs in the order specified by user, provided the
  // data wrapper correctly supports the `PATHNAME` option
  //
  // * filenames are chosen such that they appear in lexicographical order by
  // pathname, thus require padding by appropriate number of zeros
  //
  // * filenames must maintain their suffixes for some features of data
  // wrappers to work correctly, especially in the case of archived data (such
  // as `.zip` or `.gz` or any variant.)
  std::filesystem::path temp_dir_path{temp_dir};
  size_t counter = 0;
  size_t num_zero = get_number_of_digits(num_objects);
  for (size_t i = begin_object_index; i < end_object_index; ++i) {
    auto& object = objects_to_process[i];
    std::filesystem::path old_path = object.download_file_path;
    auto counter_str = std::to_string(counter++);
    auto zero_padded_counter_str =
        std::string(num_zero - counter_str.length(), '0') + counter_str;
    auto new_path = (temp_dir_path / zero_padded_counter_str).string() +
                    std::filesystem::path{object.object_key}.extension().string();
    std::filesystem::rename(old_path, new_path);
    object.import_file_path = new_path;
  }
  return {temp_dir, local_copy_params};
}
#endif

}  // namespace

namespace import_export {

ForeignDataImporter::ForeignDataImporter(const std::string& copy_from_source,
                                         const CopyParams& copy_params,
                                         const TableDescriptor* table)
    : copy_from_source_(copy_from_source), copy_params_(copy_params), table_(table) {
  connector_ = std::make_unique<Fragmenter_Namespace::LocalInsertConnector>();
}

void ForeignDataImporter::finalize(
    const Catalog_Namespace::SessionInfo& parent_session_info,
    ImportStatus& import_status,
    const std::vector<std::pair<const ColumnDescriptor*, StringDictionary*>>&
        string_dictionaries) {
  if (table_->persistenceLevel ==
      Data_Namespace::MemoryLevel::DISK_LEVEL) {  // only checkpoint disk-resident
                                                  // tables
    if (!import_status.load_failed) {
      auto timer = DEBUG_TIMER("Dictionary Checkpointing");
      for (const auto& [column_desciptor, string_dictionary] : string_dictionaries) {
        if (!string_dictionary->checkpoint()) {
          LOG(ERROR) << "Checkpointing Dictionary for Column "
                     << column_desciptor->columnName << " failed.";
          import_status.load_failed = true;
          import_status.load_msg = "Dictionary checkpoint failed";
          break;
        }
      }
    }
  }
  if (import_status.load_failed) {
    connector_->rollback(parent_session_info, table_->tableId);
  } else {
    connector_->checkpoint(parent_session_info, table_->tableId);
  }
}

void ForeignDataImporter::finalize(
    const Catalog_Namespace::SessionInfo& parent_session_info,
    ImportStatus& import_status,
    const int32_t table_id) {
  auto& catalog = parent_session_info.getCatalog();

  auto logical_columns =
      catalog.getAllColumnMetadataForTable(table_id, false, false, false);

  std::vector<std::pair<const ColumnDescriptor*, StringDictionary*>> string_dictionaries;
  for (const auto& column_descriptor : logical_columns) {
    if (!column_descriptor->columnType.is_dict_encoded_string()) {
      continue;
    }
    auto dict_descriptor =
        catalog.getMetadataForDict(column_descriptor->columnType.get_comp_param(), true);
    string_dictionaries.emplace_back(column_descriptor,
                                     dict_descriptor->stringDict.get());
  }

  finalize(parent_session_info, import_status, string_dictionaries);
}

// This value is used only if it is non-zero; it is for testing purposes only
int32_t ForeignDataImporter::proxy_foreign_table_fragment_size_ = 0;

namespace {
int32_t get_proxy_foreign_table_fragment_size(
    const size_t maximum_num_fragments_buffered,
    const size_t max_import_batch_row_count,
    const Catalog_Namespace::SessionInfo& parent_session_info,
    const int32_t table_id) {
  if (ForeignDataImporter::proxy_foreign_table_fragment_size_ != 0) {
    return ForeignDataImporter::proxy_foreign_table_fragment_size_;
  }

  if (max_import_batch_row_count != 0) {
    return max_import_batch_row_count;
  }

  // This number is chosen as a reasonable default value to reserve for
  // intermediate buffering during import, it is about 2GB of memory. Note,
  // depending on the acutal size of var len values, this heuristic target may
  // be off. NOTE: `maximum_num_fragments_buffered` scales the allowed buffer
  // size with the assumption that in the worst case all buffers may be
  // buffered at once.
  const size_t max_buffer_byte_size =
      2 * 1024UL * 1024UL * 1024UL / maximum_num_fragments_buffered;

  auto& catalog = parent_session_info.getCatalog();

  auto logical_columns =
      catalog.getAllColumnMetadataForTable(table_id, false, false, false);

  size_t row_byte_size = 0;
  for (const auto& column_descriptor : logical_columns) {
    auto type = column_descriptor->columnType;
    size_t field_byte_length = 0;
    if (type.is_varlen_indeed()) {
      // use a heuristic where varlen types are assumed to be 256 bytes in length
      field_byte_length = 256;
    } else {
      field_byte_length = type.get_size();
    }
    row_byte_size += field_byte_length;
  }

  return std::min<size_t>((max_buffer_byte_size + row_byte_size - 1) / row_byte_size,
                          DEFAULT_FRAGMENT_ROWS);
}
}  // namespace

ImportStatus ForeignDataImporter::importGeneral(
    const Catalog_Namespace::SessionInfo* session_info,
    const std::string& copy_from_source,
    const CopyParams& copy_params) {
  auto& catalog = session_info->getCatalog();

  CHECK(foreign_storage::is_valid_source_type(copy_params));

  // validate copy params before import in order to print user friendly messages
  validate_copy_params(copy_params);

  ImportStatus import_status;
  {
    auto& current_user = session_info->get_currentUser();
    auto [server, user_mapping, foreign_table] =
        foreign_storage::create_proxy_fsi_objects(copy_from_source,
                                                  copy_params,
                                                  catalog.getDatabaseId(),
                                                  table_,
                                                  current_user.userId);

    // maximum number of fragments buffered in memory at any one time, affects
    // `maxFragRows` heuristic below
    const size_t maximum_num_fragments_buffered = 3;
    // set fragment size for proxy foreign table during import
    foreign_table->maxFragRows =
        get_proxy_foreign_table_fragment_size(maximum_num_fragments_buffered,
                                              copy_params.max_import_batch_row_count,
                                              *session_info,
                                              table_->tableId);

    // log for debugging purposes
    LOG(INFO) << "Import fragment row count is " << foreign_table->maxFragRows
              << " for table " << table_->tableName;

    auto data_wrapper =
        foreign_storage::ForeignDataWrapperFactory::createForGeneralImport(
            copy_params,
            catalog.getDatabaseId(),
            foreign_table.get(),
            user_mapping.get());

    int32_t max_fragment_id = std::numeric_limits<int32_t>::max();
    if (!data_wrapper->isLazyFragmentFetchingEnabled()) {
      ChunkMetadataVector metadata_vector =
          metadata_scan(data_wrapper.get(), foreign_table.get());
      if (metadata_vector.empty()) {  // an empty data source
        return {};
      }
      max_fragment_id = 0;
      for (const auto& [key, _] : metadata_vector) {
        max_fragment_id = std::max(max_fragment_id, key[CHUNK_KEY_FRAGMENT_IDX]);
      }
      CHECK_GE(max_fragment_id, 0);
    }

    import_status = import_foreign_data(max_fragment_id,
                                        connector_.get(),
                                        catalog,
                                        table_,
                                        data_wrapper.get(),
                                        session_info,
                                        copy_params,
                                        copy_from_source,
                                        maximum_num_fragments_buffered);

  }  // this scope ensures that fsi proxy objects are destroyed prior to checkpoint

  finalize(*session_info, import_status, table_->tableId);

  return import_status;
}

ImportStatus ForeignDataImporter::importGeneral(
    const Catalog_Namespace::SessionInfo* session_info) {
  return importGeneral(session_info, copy_from_source_, copy_params_);
}

void ForeignDataImporter::setDefaultImportPath(const std::string& base_path) {
  auto data_dir_path = boost::filesystem::canonical(base_path);
  default_import_path_ = (data_dir_path / shared::kDefaultImportDirName).string();
}

ImportStatus ForeignDataImporter::importGeneralS3(
    const Catalog_Namespace::SessionInfo* session_info) {
  CHECK(shared::is_s3_uri(copy_from_source_));

  if (!(copy_params_.source_type == SourceType::kDelimitedFile ||
#if ENABLE_IMPORT_PARQUET
        copy_params_.source_type == SourceType::kParquetFile ||
#endif
        copy_params_.source_type == SourceType::kRegexParsedFile)) {
    throw std::runtime_error("Attempting to load S3 resource '" + copy_from_source_ +
                             "' for unsupported 'source_type' (must be 'DELIMITED_FILE'"
#if ENABLE_IMPORT_PARQUET
                             ", 'PARQUET_FILE'"
#endif
                             " or 'REGEX_PARSED_FILE'");
  }

  const shared::FilePathOptions options{copy_params_.regex_path_filter,
                                        copy_params_.file_sort_order_by,
                                        copy_params_.file_sort_regex};
  shared::validate_sort_options(options);

#ifdef HAVE_AWS_S3

  auto uuid = boost::uuids::random_generator()();
  std::string base_path = "s3-import-" + boost::uuids::to_string(uuid);
  auto import_path = std::filesystem::path(default_import_path_) / base_path;

  // Ensure files & dirs are cleaned up, regardless of outcome
  ScopeGuard cleanup_guard = [&] {
    if (std::filesystem::exists(import_path)) {
      std::filesystem::remove_all(import_path);
    }
  };

  auto s3_archive = std::make_unique<S3Archive>(copy_from_source_,
                                                copy_params_.s3_access_key,
                                                copy_params_.s3_secret_key,
                                                copy_params_.s3_session_token,
                                                copy_params_.s3_region,
                                                copy_params_.s3_endpoint,
                                                copy_params_.plain_text,
                                                copy_params_.regex_path_filter,
                                                copy_params_.file_sort_order_by,
                                                copy_params_.file_sort_regex,
                                                import_path);
  s3_archive->init_for_read();

  const auto bucket_name = s3_archive->url_part(4);

  auto object_keys = s3_archive->get_objkeys();
  std::vector<DownloadedObjectToProcess> objects_to_process(object_keys.size());
  size_t object_count = 0;
  for (const auto& objkey : object_keys) {
    auto& object = objects_to_process[object_count++];
    object.object_key = objkey;
    object.is_downloaded = false;
  }

  ImportStatus aggregate_import_status;
  const int num_download_threads = copy_params_.s3_max_concurrent_downloads;

  std::mutex communication_mutex;
  bool continue_downloading = true;
  bool download_exception_occured = false;

  std::condition_variable files_download_condition;

  auto is_downloading_finished = [&] {
    std::unique_lock communication_lock(communication_mutex);
    return !continue_downloading || download_exception_occured;
  };

  std::function<void(const std::vector<size_t>&)> download_objects =
      [&](const std::vector<size_t>& partition) {
        for (const auto& index : partition) {
          DownloadedObjectToProcess& object = objects_to_process[index];
          const std::string& obj_key = object.object_key;
          if (is_downloading_finished()) {
            return;
          }
          std::exception_ptr eptr;  // unused
          std::string local_file_path;
          std::string exception_what;
          bool exception_occured = false;

          try {
            local_file_path = s3_archive->land(obj_key,
                                               eptr,
                                               false,
                                               /*allow_named_pipe_use=*/false,
                                               /*track_file_path=*/false);
          } catch (const std::exception& e) {
            exception_what = e.what();
            exception_occured = true;
          }

          if (is_downloading_finished()) {
            return;
          }
          if (exception_occured) {
            {
              std::unique_lock communication_lock(communication_mutex);
              download_exception_occured = true;
            }
            files_download_condition.notify_all();
            throw std::runtime_error("failed to fetch s3 object: '" + obj_key +
                                     "': " + exception_what);
          }

          object.download_file_path = local_file_path;
          object.is_downloaded =
              true;  // this variable is atomic and therefore acts as a lock, it must be
                     // set last to ensure no data race

          files_download_condition.notify_all();
        }
      };

  std::function<void()> import_local_files = [&]() {
    for (size_t object_index = 0; object_index < object_count;) {
      {
        std::unique_lock communication_lock(communication_mutex);
        files_download_condition.wait(
            communication_lock,
            [&download_exception_occured, object_index, &objects_to_process]() {
              return objects_to_process[object_index].is_downloaded ||
                     download_exception_occured;
            });
        if (download_exception_occured) {  // do not wait for object index if a download
                                           // error has occured
          return;
        }
      }

      // find largest range of files to import
      size_t end_object_index = object_count;
      for (size_t i = object_index + 1; i < object_count; ++i) {
        if (!objects_to_process[i].is_downloaded) {
          end_object_index = i;
          break;
        }
      }

      ImportStatus local_import_status;
      std::string local_import_dir;
      try {
        CopyParams local_copy_params;
        std::tie(local_import_dir, local_copy_params) = get_local_copy_source_and_params(
            copy_params_, objects_to_process, object_index, end_object_index);
        local_import_status =
            importGeneral(session_info, local_import_dir, local_copy_params);
        // clean up temporary files
        std::filesystem::remove_all(local_import_dir);
      } catch (const std::exception& except) {
        // replace all occurences of file names with the object keys for
        // users
        std::string what = except.what();

        for (size_t i = object_index; i < end_object_index; ++i) {
          auto& object = objects_to_process[i];
          what = boost::regex_replace(what,
                                      boost::regex{object.import_file_path},
                                      bucket_name + "/" + object.object_key);
        }
        {
          std::unique_lock communication_lock(communication_mutex);
          continue_downloading = false;
        }
        // clean up temporary files
        std::filesystem::remove_all(local_import_dir);
        throw std::runtime_error(what);
      }
      aggregate_import_status += local_import_status;
      import_export::Importer::set_import_status(copy_from_source_,
                                                 aggregate_import_status);
      if (aggregate_import_status.load_failed) {
        {
          std::unique_lock communication_lock(communication_mutex);
          continue_downloading = false;
        }
        return;
      }

      object_index =
          end_object_index;  // all objects in range [object_index,end_object_index)
                             // correctly imported at this point in excecution, move onto
                             // next range
    }
  };

  std::vector<size_t> partition_range(object_count);
  std::iota(partition_range.begin(), partition_range.end(), 0);
  auto download_futures = foreign_storage::create_futures_for_workers(
      partition_range, num_download_threads, download_objects);

  auto import_future = std::async(std::launch::async, import_local_files);

  for (auto& future : download_futures) {
    future.wait();
  }
  import_future.get();  // may throw an exception

  // get any remaining exceptions
  for (auto& future : download_futures) {
    future.get();
  }
  return aggregate_import_status;

#else
  throw std::runtime_error("AWS S3 support not available");

  return {};
#endif
}

#ifdef ENABLE_IMPORT_PARQUET
ImportStatus ForeignDataImporter::importParquet(
    const Catalog_Namespace::SessionInfo* session_info) {
  auto& catalog = session_info->getCatalog();

  CHECK(copy_params_.source_type == import_export::SourceType::kParquetFile);

  auto& current_user = session_info->get_currentUser();
  auto server = foreign_storage::ForeignDataWrapperFactory::createForeignServerProxy(
      catalog.getDatabaseId(), current_user.userId, copy_from_source_, copy_params_);

  auto user_mapping =
      foreign_storage::ForeignDataWrapperFactory::createUserMappingProxyIfApplicable(
          catalog.getDatabaseId(),
          current_user.userId,
          copy_from_source_,
          copy_params_,
          server.get());

  auto foreign_table =
      foreign_storage::ForeignDataWrapperFactory::createForeignTableProxy(
          catalog.getDatabaseId(), table_, copy_from_source_, copy_params_, server.get());

  foreign_table->validateOptionValues();

  auto data_wrapper = foreign_storage::ForeignDataWrapperFactory::createForImport(
      foreign_storage::DataWrapperType::PARQUET,
      catalog.getDatabaseId(),
      foreign_table.get(),
      user_mapping.get());

  if (auto parquet_import =
          dynamic_cast<foreign_storage::ParquetImporter*>(data_wrapper.get())) {
    Fragmenter_Namespace::InsertDataLoader insert_data_loader(*connector_);

    // determine the number of threads to use at each level

    int max_threads = 0;
    if (copy_params_.threads == 0) {
      max_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()),
                             g_max_import_threads);
    } else {
      max_threads = static_cast<size_t>(copy_params_.threads);
    }
    CHECK_GT(max_threads, 0);

    int num_importer_threads =
        std::min<int>(max_threads, parquet_import->getMaxNumUsefulThreads());
    parquet_import->setNumThreads(num_importer_threads);
    int num_outer_thread = 1;
    for (int thread_count = 1; thread_count <= max_threads; ++thread_count) {
      if (thread_count * num_importer_threads <= max_threads) {
        num_outer_thread = thread_count;
      }
    }

    std::shared_mutex import_status_mutex;
    ImportStatus import_status;  // manually update

    auto import_failed = [&import_status_mutex, &import_status] {
      std::shared_lock import_status_lock(import_status_mutex);
      return import_status.load_failed;
    };

    std::vector<std::future<void>> futures;

    for (int ithread = 0; ithread < num_outer_thread; ++ithread) {
      futures.emplace_back(std::async(std::launch::async, [&] {
        while (true) {
          auto batch_result = parquet_import->getNextImportBatch();
          if (import_failed()) {
            break;
          }
          auto batch = batch_result->getInsertData();
          if (!batch || import_failed()) {
            break;
          }
          insert_data_loader.insertData(*session_info, *batch);

          auto batch_import_status = batch_result->getImportStatus();
          {
            std::unique_lock import_status_lock(import_status_mutex);
            import_status.rows_completed += batch_import_status.rows_completed;
            import_status.rows_rejected += batch_import_status.rows_rejected;
            if (import_status.rows_rejected > copy_params_.max_reject) {
              import_status.load_failed = true;
              import_status.load_msg =
                  "Load was cancelled due to max reject rows being reached";
              break;
            }
          }
        }
      }));
    }

    for (auto& future : futures) {
      future.wait();
    }

    for (auto& future : futures) {
      future.get();
    }

    if (import_status.load_failed) {
      foreign_table.reset();  // this is to avoid calling the TableDescriptor dtor after
                              // the rollback in the checkpoint below
    }

    finalize(*session_info, import_status, parquet_import->getStringDictionaries());

    return import_status;
  }

  UNREACHABLE();
  return {};
}
#endif

ImportStatus ForeignDataImporter::import(
    const Catalog_Namespace::SessionInfo* session_info) {
  if (shared::is_s3_uri(copy_from_source_)) {
    return importGeneralS3(session_info);
  }
  return importGeneral(session_info);
}

}  // namespace import_export
