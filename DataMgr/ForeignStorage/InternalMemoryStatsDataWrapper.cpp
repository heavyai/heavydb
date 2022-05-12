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

#include "InternalMemoryStatsDataWrapper.h"

#include "Catalog/SysCatalog.h"
#include "ImportExport/Importer.h"

namespace foreign_storage {
InternalMemoryStatsDataWrapper::InternalMemoryStatsDataWrapper()
    : InternalSystemDataWrapper() {}

InternalMemoryStatsDataWrapper::InternalMemoryStatsDataWrapper(
    const int db_id,
    const ForeignTable* foreign_table)
    : InternalSystemDataWrapper(db_id, foreign_table) {}

namespace {
void populate_import_buffers_for_memory_summary(
    const std::map<std::string, std::vector<MemoryInfo>>& memory_info_by_device_type,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& [device_type, memory_info_vector] : memory_info_by_device_type) {
    int32_t device_id{0};
    for (const auto& memory_info : memory_info_vector) {
      size_t used_page_count{0}, free_page_count{0};
      for (const auto& memory_data : memory_info.nodeMemoryData) {
        if (memory_data.memStatus == Buffer_Namespace::MemStatus::FREE) {
          free_page_count += memory_data.numPages;
        } else {
          used_page_count += memory_data.numPages;
        }
      }
      set_node_name(import_buffers);
      if (import_buffers.find("device_id") != import_buffers.end()) {
        import_buffers["device_id"]->addInt(device_id);
      }
      if (import_buffers.find("device_type") != import_buffers.end()) {
        import_buffers["device_type"]->addString(device_type);
      }
      if (import_buffers.find("max_page_count") != import_buffers.end()) {
        import_buffers["max_page_count"]->addBigint(memory_info.maxNumPages);
      }
      if (import_buffers.find("page_size") != import_buffers.end()) {
        import_buffers["page_size"]->addBigint(memory_info.pageSize);
      }
      if (import_buffers.find("allocated_page_count") != import_buffers.end()) {
        import_buffers["allocated_page_count"]->addBigint(memory_info.numPageAllocated);
      }
      if (import_buffers.find("used_page_count") != import_buffers.end()) {
        import_buffers["used_page_count"]->addBigint(used_page_count);
      }
      if (import_buffers.find("free_page_count") != import_buffers.end()) {
        import_buffers["free_page_count"]->addBigint(free_page_count);
      }
      device_id++;
    }
  }
}

void set_null(import_export::TypedImportBuffer* import_buffer) {
  import_buffer->add_value(import_buffer->getColumnDesc(), "", true, {});
}

bool is_table_chunk(const ChunkKey& chunk_key) {
  // Non-table chunks (temporary buffers) use a negative id for the database id.
  return (!chunk_key.empty() && chunk_key[CHUNK_KEY_DB_IDX] > 0);
}

std::string get_column_name(int32_t db_id, int32_t table_id, int32_t column_id) {
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  auto column_name = catalog->getColumnName(table_id, column_id);
  if (column_name.has_value()) {
    return column_name.value();
  } else {
    // It is possible for the column to be concurrently deleted while querying the system
    // table.
    return kDeletedValueIndicator;
  }
}

void populate_import_buffers_for_memory_details(
    const std::map<std::string, std::vector<MemoryInfo>>& memory_info_by_device_type,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  for (const auto& [device_type, memory_info_vector] : memory_info_by_device_type) {
    int32_t device_id{0};
    for (const auto& memory_info : memory_info_vector) {
      for (const auto& memory_data : memory_info.nodeMemoryData) {
        set_node_name(import_buffers);
        const auto& chunk_key = memory_data.chunk_key;
        if (import_buffers.find("database_id") != import_buffers.end()) {
          auto import_buffer = import_buffers["database_id"];
          if (is_table_chunk(chunk_key)) {
            import_buffer->addInt(chunk_key[CHUNK_KEY_DB_IDX]);
          } else {
            set_null(import_buffer);
          }
        }
        if (import_buffers.find("database_name") != import_buffers.end()) {
          auto import_buffer = import_buffers["database_name"];
          if (is_table_chunk(chunk_key)) {
            import_buffer->addString(get_db_name(chunk_key[CHUNK_KEY_DB_IDX]));
          } else {
            set_null(import_buffer);
          }
        }
        if (import_buffers.find("table_id") != import_buffers.end()) {
          auto import_buffer = import_buffers["table_id"];
          if (is_table_chunk(chunk_key)) {
            import_buffer->addInt(chunk_key[CHUNK_KEY_TABLE_IDX]);
          } else {
            set_null(import_buffer);
          }
        }
        if (import_buffers.find("table_name") != import_buffers.end()) {
          auto import_buffer = import_buffers["table_name"];
          if (is_table_chunk(chunk_key)) {
            import_buffer->addString(get_table_name(chunk_key[CHUNK_KEY_DB_IDX],
                                                    chunk_key[CHUNK_KEY_TABLE_IDX]));
          } else {
            set_null(import_buffer);
          }
        }
        if (import_buffers.find("column_id") != import_buffers.end()) {
          auto import_buffer = import_buffers["column_id"];
          if (is_table_chunk(chunk_key)) {
            import_buffer->addInt(chunk_key[CHUNK_KEY_COLUMN_IDX]);
          } else {
            set_null(import_buffer);
          }
        }
        if (import_buffers.find("column_name") != import_buffers.end()) {
          auto import_buffer = import_buffers["column_name"];
          if (is_table_chunk(chunk_key)) {
            import_buffer->addString(get_column_name(chunk_key[CHUNK_KEY_DB_IDX],
                                                     chunk_key[CHUNK_KEY_TABLE_IDX],
                                                     chunk_key[CHUNK_KEY_COLUMN_IDX]));
          } else {
            set_null(import_buffer);
          }
        }
        if (import_buffers.find("chunk_key") != import_buffers.end()) {
          import_buffers["chunk_key"]->addArray(ArrayDatum(
              chunk_key.size() * sizeof(int32_t),
              reinterpret_cast<int8_t*>(const_cast<int32_t*>(chunk_key.data())),
              DoNothingDeleter()));
        }
        if (import_buffers.find("device_id") != import_buffers.end()) {
          import_buffers["device_id"]->addInt(device_id);
        }
        if (import_buffers.find("device_type") != import_buffers.end()) {
          import_buffers["device_type"]->addString(device_type);
        }
        if (import_buffers.find("memory_status") != import_buffers.end()) {
          auto memory_status =
              (memory_data.memStatus == Buffer_Namespace::MemStatus::FREE ? "FREE"
                                                                          : "USED");
          import_buffers["memory_status"]->addString(memory_status);
        }
        if (import_buffers.find("page_count") != import_buffers.end()) {
          import_buffers["page_count"]->addBigint(memory_data.numPages);
        }
        if (import_buffers.find("page_size") != import_buffers.end()) {
          import_buffers["page_size"]->addBigint(memory_info.pageSize);
        }
        if (import_buffers.find("slab_id") != import_buffers.end()) {
          import_buffers["slab_id"]->addInt(memory_data.slabNum);
        }
        if (import_buffers.find("start_page") != import_buffers.end()) {
          import_buffers["start_page"]->addBigint(memory_data.startPage);
        }
        if (import_buffers.find("last_touch_epoch") != import_buffers.end()) {
          import_buffers["last_touch_epoch"]->addBigint(memory_data.touch);
        }
      }
      device_id++;
    }
  }
}
}  // namespace

void InternalMemoryStatsDataWrapper::initializeObjectsForTable(
    const std::string& table_name) {
  memory_info_by_device_type_.clear();
  row_count_ = 0;
  const auto& data_mgr = Catalog_Namespace::SysCatalog::instance().getDataMgr();
  // DataMgr::getMemoryInfoUnlocked() is used here because a lock on buffer_access_mutex_
  // is already acquired in DataMgr::getChunkMetadataVecForKeyPrefix()
  memory_info_by_device_type_["CPU"] =
      data_mgr.getMemoryInfoUnlocked(MemoryLevel::CPU_LEVEL);
  memory_info_by_device_type_["GPU"] =
      data_mgr.getMemoryInfoUnlocked(MemoryLevel::GPU_LEVEL);
  if (foreign_table_->tableName == Catalog_Namespace::MEMORY_SUMMARY_SYS_TABLE_NAME) {
    for (const auto& [device_type, memory_info_vector] : memory_info_by_device_type_) {
      row_count_ += memory_info_vector.size();
    }
  } else if (foreign_table_->tableName ==
             Catalog_Namespace::MEMORY_DETAILS_SYS_TABLE_NAME) {
    for (auto& [device_type, memory_info_vector] : memory_info_by_device_type_) {
      for (auto& memory_info : memory_info_vector) {
        for (auto& memory_data : memory_info.nodeMemoryData) {
          if (memory_data.memStatus == Buffer_Namespace::MemStatus::FREE) {
            memory_data.chunk_key.clear();
          } else {
            if (is_table_chunk(memory_data.chunk_key)) {
              CHECK_GE(memory_data.chunk_key.size(), static_cast<size_t>(4));
            }
          }
        }
        row_count_ += memory_info.nodeMemoryData.size();
      }
    }
  } else {
    UNREACHABLE() << "Unexpected table name: " << table_name;
  }
}

void InternalMemoryStatsDataWrapper::populateChunkBuffersForTable(
    const std::string& table_name,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  if (foreign_table_->tableName == Catalog_Namespace::MEMORY_SUMMARY_SYS_TABLE_NAME) {
    populate_import_buffers_for_memory_summary(memory_info_by_device_type_,
                                               import_buffers);
  } else if (foreign_table_->tableName ==
             Catalog_Namespace::MEMORY_DETAILS_SYS_TABLE_NAME) {
    populate_import_buffers_for_memory_details(memory_info_by_device_type_,
                                               import_buffers);
  } else {
    UNREACHABLE() << "Unexpected table name: " << foreign_table_->tableName;
  }
}
}  // namespace foreign_storage
