/*
 * Copyright 2021 OmniSci, Inc.
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
#include "DataMgr/ForeignStorage/ForeignDataWrapperFactory.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/ParquetImporter.h"
#include "Importer.h"
#include "Parser/ParserNode.h"
#include "Shared/enable_assign_render_groups.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "UserMapping.h"

#ifdef ENABLE_IMPORT_PARQUET
extern bool g_enable_parquet_import_fsi;
#endif
extern bool g_enable_general_import_fsi;

namespace {

std::string get_data_wrapper_type(const import_export::CopyParams& copy_params) {
  std::string data_wrapper_type;
  if (copy_params.source_type == import_export::SourceType::kDelimitedFile) {
    data_wrapper_type = foreign_storage::DataWrapperType::CSV;
  } else if (copy_params.source_type == import_export::SourceType::kRegexParsedFile) {
    data_wrapper_type = foreign_storage::DataWrapperType::REGEX_PARSER;
#ifdef ENABLE_IMPORT_PARQUET
  } else if (copy_params.source_type == import_export::SourceType::kParquetFile) {
    data_wrapper_type = foreign_storage::DataWrapperType::PARQUET;
#endif
  } else {
    UNREACHABLE();
  }
  return data_wrapper_type;
}

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

void validate_copy_params(const import_export::CopyParams& copy_params) {
  if (copy_params.source_type == import_export::SourceType::kRegexParsedFile) {
    foreign_storage::validate_regex_parser_options(copy_params);
  }
}

std::tuple<std::unique_ptr<foreign_storage::ForeignServer>,
           std::unique_ptr<foreign_storage::UserMapping>,
           std::unique_ptr<foreign_storage::ForeignTable>>
create_proxy_fsi_objects(const std::string& copy_from_source,
                         const import_export::CopyParams& copy_params,
                         Catalog_Namespace::Catalog& catalog,
                         const TableDescriptor* table,
                         const Catalog_Namespace::SessionInfo* session_info) {
  auto& current_user = session_info->get_currentUser();
  auto server = foreign_storage::ForeignDataWrapperFactory::createForeignServerProxy(
      catalog.getDatabaseId(), current_user.userId, copy_from_source, copy_params);

  CHECK(server);
  server->validate();

  auto user_mapping =
      foreign_storage::ForeignDataWrapperFactory::createUserMappingProxyIfApplicable(
          catalog.getDatabaseId(),
          current_user.userId,
          copy_from_source,
          copy_params,
          server.get());

  if (user_mapping) {
    user_mapping->validate(server.get());
  }

  auto foreign_table =
      foreign_storage::ForeignDataWrapperFactory::createForeignTableProxy(
          catalog.getDatabaseId(), table, copy_from_source, copy_params, server.get());

  CHECK(foreign_table);
  foreign_table->validateOptionValues();

  return {std::move(server), std::move(user_mapping), std::move(foreign_table)};
}

import_export::ImportStatus import_foreign_data(
    const ChunkMetadataVector& metadata_vector,
    Fragmenter_Namespace::InsertDataLoader::DistributedConnector* connector,
    Catalog_Namespace::Catalog& catalog,
    const TableDescriptor* table,
    foreign_storage::ForeignDataWrapper* data_wrapper,
    const Catalog_Namespace::SessionInfo* session_info) {
  int32_t max_fragment_id = -1;
  for (const auto& [key, _] : metadata_vector) {
    max_fragment_id = std::max(max_fragment_id, key[CHUNK_KEY_FRAGMENT_IDX]);
  }
  CHECK_GE(max_fragment_id, 0);

  if (g_enable_assign_render_groups) {
    // if render group assignment is globally enabled,
    // tell the wrapper to create any RenderGroupAnalyzers it
    // may need for any poly columns in the table, if that
    // wrapper type supports it
    data_wrapper->createRenderGroupAnalyzers();
  }

  Fragmenter_Namespace::InsertDataLoader insert_data_loader(*connector);
  import_export::ImportStatus import_status;  // manually update
  for (int32_t fragment_id = 0; fragment_id <= max_fragment_id; ++fragment_id) {
    // gather applicable keys to load for fragment
    std::set<ChunkKey> fragment_keys;
    for (const auto& [key, _] : metadata_vector) {
      if (key[CHUNK_KEY_FRAGMENT_IDX] == fragment_id) {
        fragment_keys.insert(key);

        const auto col_id = key[CHUNK_KEY_COLUMN_IDX];
        const auto table_id = key[CHUNK_KEY_TABLE_IDX];
        const auto col_desc = catalog.getMetadataForColumn(table_id, col_id);
        if (col_desc->columnType.is_varlen_indeed()) {
          CHECK(key.size() > CHUNK_KEY_VARLEN_IDX);
          if (key[CHUNK_KEY_VARLEN_IDX] == 1) {  // data chunk
            auto index_key = key;
            index_key[CHUNK_KEY_VARLEN_IDX] = 2;
            fragment_keys.insert(index_key);
          }
        }
      }
    }

    // create buffers
    std::map<ChunkKey, std::unique_ptr<foreign_storage::ForeignStorageBuffer>>
        fragment_buffers_owner;
    foreign_storage::ChunkToBufferMap fragment_buffers;
    for (const auto& key : fragment_keys) {
      fragment_buffers_owner[key] =
          std::make_unique<foreign_storage::ForeignStorageBuffer>();
      fragment_buffers_owner[key]->resetToEmpty();
      fragment_buffers[key] = shared::get_from_map(fragment_buffers_owner, key).get();
    }

    // get chunks for import
    Fragmenter_Namespace::InsertChunks insert_chunks{
        table->tableId, catalog.getDatabaseId(), {}};

    // get the buffers
    data_wrapper->populateChunkBuffers(fragment_buffers, {});

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
              col_desc, buffer, shared::get_from_map(fragment_buffers, index_key));
        }
      } else {  // regular non-varlen case with no index buffer
        insert_chunks.chunks[col_id] =
            Chunk_NS::Chunk::getChunk(col_desc, buffer, nullptr);
      }
    }

    // import chunks
    insert_data_loader.insertChunks(*session_info, insert_chunks);
  }
  return {};  // TODO: update import status appropriately once error handling is
              // implemented
}

}  // namespace

namespace import_export {

ForeignDataImporter::ForeignDataImporter(const std::string& copy_from_source,
                                         const CopyParams& copy_params,
                                         const TableDescriptor* table)
    : copy_from_source_(copy_from_source), copy_params_(copy_params), table_(table) {
  connector_ = std::make_unique<Parser::LocalConnector>();
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

// TODO: the `proxy_foreign_table_fragment_size_` parameter controls the amount
// of data buffered in memory while importing using the `ForeignDataImporter`
// may need to be tuned or exposed as configurable parameter
const int32_t ForeignDataImporter::proxy_foreign_table_fragment_size_ = 2000000;

ImportStatus ForeignDataImporter::importGeneral(
    const Catalog_Namespace::SessionInfo* session_info) {
  auto& catalog = session_info->getCatalog();

  CHECK(foreign_storage::is_valid_source_type(copy_params_));

  // validate copy params before import in order to print user friendly messages
  validate_copy_params(copy_params_);

  auto [server, user_mapping, foreign_table] = create_proxy_fsi_objects(
      copy_from_source_, copy_params_, catalog, table_, session_info);

  // set fragment size for prox foreign table during import
  foreign_table->maxFragRows = proxy_foreign_table_fragment_size_;

  auto data_wrapper = foreign_storage::ForeignDataWrapperFactory::createForGeneralImport(
      get_data_wrapper_type(copy_params_),
      catalog.getDatabaseId(),
      foreign_table.get(),
      user_mapping.get());

  ChunkMetadataVector metadata_vector =
      metadata_scan(data_wrapper.get(), foreign_table.get());
  if (metadata_vector.empty()) {  // an empty data source
    return {};
  }

  return import_foreign_data(metadata_vector,
                             connector_.get(),
                             catalog,
                             table_,
                             data_wrapper.get(),
                             session_info);
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
  if (g_enable_general_import_fsi) {
    return importGeneral(session_info);
#ifdef ENABLE_IMPORT_PARQUET
  } else if (g_enable_parquet_import_fsi) {
    return importParquet(session_info);
#endif
  } else {
    UNREACHABLE();
  }
  return {};
}

}  // namespace import_export
