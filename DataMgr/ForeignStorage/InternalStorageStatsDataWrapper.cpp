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

#include "InternalStorageStatsDataWrapper.h"

#include <filesystem>

#include "Catalog/SysCatalog.h"
#include "ImportExport/Importer.h"

namespace foreign_storage {
InternalStorageStatsDataWrapper::InternalStorageStatsDataWrapper()
    : InternalSystemDataWrapper() {}

InternalStorageStatsDataWrapper::InternalStorageStatsDataWrapper(
    const int db_id,
    const ForeignTable* foreign_table)
    : InternalSystemDataWrapper(db_id, foreign_table) {}

namespace {
void set_null(import_export::TypedImportBuffer* import_buffer) {
  import_buffer->add_value(import_buffer->getColumnDesc(), "", true, {});
}

void populate_import_buffers_for_storage_details(
    const std::vector<StorageDetails>& storage_details,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& storage_detail : storage_details) {
    if (import_buffers.find("node") != import_buffers.end()) {
      import_buffers["node"]->addString(storage_detail.node);
    }
    if (import_buffers.find("database_id") != import_buffers.end()) {
      import_buffers["database_id"]->addInt(storage_detail.database_id);
    }
    if (import_buffers.find("database_name") != import_buffers.end()) {
      import_buffers["database_name"]->addString(get_db_name(storage_detail.database_id));
    }
    if (import_buffers.find("table_id") != import_buffers.end()) {
      import_buffers["table_id"]->addInt(storage_detail.table_id);
    }
    if (import_buffers.find("table_name") != import_buffers.end()) {
      import_buffers["table_name"]->addString(
          get_table_name(storage_detail.database_id, storage_detail.table_id));
    }
    if (import_buffers.find("epoch") != import_buffers.end()) {
      import_buffers["epoch"]->addInt(storage_detail.storage_stats.epoch);
    }
    if (import_buffers.find("epoch_floor") != import_buffers.end()) {
      import_buffers["epoch_floor"]->addInt(storage_detail.storage_stats.epoch_floor);
    }
    if (import_buffers.find("fragment_count") != import_buffers.end()) {
      auto import_buffer = import_buffers["fragment_count"];
      if (storage_detail.storage_stats.fragment_count.has_value()) {
        import_buffer->addInt(storage_detail.storage_stats.fragment_count.value());
      } else {
        set_null(import_buffer);
      }
    }
    if (import_buffers.find("shard_id") != import_buffers.end()) {
      import_buffers["shard_id"]->addInt(storage_detail.shard_id);
    }
    if (import_buffers.find("data_file_count") != import_buffers.end()) {
      import_buffers["data_file_count"]->addInt(
          storage_detail.storage_stats.data_file_count);
    }
    if (import_buffers.find("metadata_file_count") != import_buffers.end()) {
      import_buffers["metadata_file_count"]->addInt(
          storage_detail.storage_stats.metadata_file_count);
    }
    if (import_buffers.find("total_data_file_size") != import_buffers.end()) {
      import_buffers["total_data_file_size"]->addBigint(
          storage_detail.storage_stats.total_data_file_size);
    }
    if (import_buffers.find("total_data_page_count") != import_buffers.end()) {
      import_buffers["total_data_page_count"]->addBigint(
          storage_detail.storage_stats.total_data_page_count);
    }
    if (import_buffers.find("total_free_data_page_count") != import_buffers.end()) {
      auto import_buffer = import_buffers["total_free_data_page_count"];
      if (storage_detail.storage_stats.total_free_data_page_count.has_value()) {
        import_buffer->addBigint(
            storage_detail.storage_stats.total_free_data_page_count.value());
      } else {
        set_null(import_buffer);
      }
    }
    if (import_buffers.find("total_metadata_file_size") != import_buffers.end()) {
      import_buffers["total_metadata_file_size"]->addBigint(
          storage_detail.storage_stats.total_metadata_file_size);
    }
    if (import_buffers.find("total_metadata_page_count") != import_buffers.end()) {
      import_buffers["total_metadata_page_count"]->addBigint(
          storage_detail.storage_stats.total_metadata_page_count);
    }
    if (import_buffers.find("total_free_metadata_page_count") != import_buffers.end()) {
      auto import_buffer = import_buffers["total_free_metadata_page_count"];
      if (storage_detail.storage_stats.total_free_metadata_page_count.has_value()) {
        import_buffer->addBigint(
            storage_detail.storage_stats.total_free_metadata_page_count.value());
      } else {
        set_null(import_buffer);
      }
    }
    if (import_buffers.find("total_dictionary_data_file_size") != import_buffers.end()) {
      import_buffers["total_dictionary_data_file_size"]->addBigint(
          storage_detail.total_dictionary_data_file_size);
    }
  }
}
}  // namespace

void InternalStorageStatsDataWrapper::initializeObjectsForTable(
    const std::string& table_name) {
  CHECK_EQ(table_name, Catalog_Namespace::STORAGE_DETAILS_SYS_TABLE_NAME);
  storage_details_.clear();
  const auto global_file_mgr =
      Catalog_Namespace::SysCatalog::instance().getDataMgr().getGlobalFileMgr();
  CHECK(global_file_mgr);
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  for (const auto& catalog : sys_catalog.getCatalogsForAllDbs()) {
    if (catalog->name() != INFORMATION_SCHEMA_DB) {
      std::set<std::string> found_dict_paths;
      for (const auto& [table_id, shard_id] :
           catalog->getAllPersistedTableAndShardIds()) {
        uint64_t total_dictionary_file_size{0};
        auto logical_table_id = catalog->getLogicalTableId(table_id);
        for (const auto& dict_path :
             catalog->getTableDictDirectoryPaths(logical_table_id)) {
          if (found_dict_paths.find(dict_path) == found_dict_paths.end()) {
            found_dict_paths.emplace(dict_path);
          } else {
            // Skip shared dictionaries.
            continue;
          }
          CHECK(std::filesystem::is_directory(dict_path));
          for (const auto& file_entry : std::filesystem::directory_iterator(dict_path)) {
            CHECK(file_entry.is_regular_file());
            total_dictionary_file_size += static_cast<uint64_t>(file_entry.file_size());
          }
        }
        auto db_id = catalog->getDatabaseId();
        storage_details_.emplace_back(db_id,
                                      logical_table_id,
                                      shard_id,
                                      total_dictionary_file_size,
                                      global_file_mgr->getStorageStats(db_id, table_id));
      }
    }
  }
  row_count_ = storage_details_.size();
}

void InternalStorageStatsDataWrapper::populateChunkBuffersForTable(
    const std::string& table_name,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  CHECK_EQ(table_name, Catalog_Namespace::STORAGE_DETAILS_SYS_TABLE_NAME);
  populate_import_buffers_for_storage_details(storage_details_, import_buffers);
}
}  // namespace foreign_storage
