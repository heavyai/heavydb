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

#include "ForeignTableRefresh.h"
#include "Catalog/Catalog.h"
#include "DataMgr/ForeignStorage/CachingForeignStorageMgr.h"
#include "LockMgr/LockMgr.h"
#include "Shared/distributed.h"
#include "Shared/misc.h"

#include "QueryEngine/Execute.h"

namespace foreign_storage {
namespace {
void clear_cpu_and_gpu_cache(Data_Namespace::DataMgr& data_mgr,
                             const ChunkKey& key_prefix) {
  data_mgr.deleteChunksWithPrefix(key_prefix, MemoryLevel::CPU_LEVEL);
  data_mgr.deleteChunksWithPrefix(key_prefix, MemoryLevel::GPU_LEVEL);
}
}  // namespace

void refresh_foreign_table_unlocked(Catalog_Namespace::Catalog& catalog,
                                    const ForeignTable& td,
                                    const bool evict_cached_entries) {
  LOG(INFO) << "Starting refresh for table: " << td.tableName;
  auto& data_mgr = catalog.getDataMgr();
  ChunkKey table_key{catalog.getCurrentDB().dbId, td.tableId};
  Executor::clearExternalCaches(true, &td, catalog.getDatabaseId());
  catalog.removeFragmenterForTable(td.tableId);

  auto fsm = data_mgr.getPersistentStorageMgr()->getForeignStorageMgr();
  CHECK(fsm);
  if (auto cfm = dynamic_cast<CachingForeignStorageMgr*>(fsm)) {
    if (!cfm->hasStoredDataWrapper(table_key[CHUNK_KEY_DB_IDX],
                                   table_key[CHUNK_KEY_TABLE_IDX])) {
      // If there is no wrapper stored on disk, then we have not populated the metadata
      // for this table and we are free to skip the refresh.
      catalog.updateForeignTableRefreshTimes(td.tableId);
      return;
    }
  }

  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> old_chunk_metadata_by_chunk_key;
  if (td.isAppendMode() && !evict_cached_entries) {
    ChunkMetadataVector metadata_vec;
    data_mgr.getChunkMetadataVecForKeyPrefix(metadata_vec, table_key);
    int last_fragment_id = 0;
    for (const auto& [key, metadata] : metadata_vec) {
      if (key[CHUNK_KEY_FRAGMENT_IDX] > last_fragment_id) {
        last_fragment_id = key[CHUNK_KEY_FRAGMENT_IDX];
      }
      old_chunk_metadata_by_chunk_key[key] = metadata;
    }
    for (const auto& [key, metadata] : metadata_vec) {
      if (key[CHUNK_KEY_FRAGMENT_IDX] == last_fragment_id) {
        clear_cpu_and_gpu_cache(data_mgr, key);
      }
    }
  } else {
    clear_cpu_and_gpu_cache(data_mgr, table_key);
  }

  try {
    fsm->refreshTable(table_key, evict_cached_entries);
    catalog.updateForeignTableRefreshTimes(td.tableId);
  } catch (PostEvictionRefreshException& e) {
    catalog.updateForeignTableRefreshTimes(td.tableId);
    clear_cpu_and_gpu_cache(data_mgr, table_key);
    throw e.getOriginalException();
  } catch (...) {
    clear_cpu_and_gpu_cache(data_mgr, table_key);
    throw;
  }

  // Delete cached rolled off/updated chunks.
  if (!old_chunk_metadata_by_chunk_key.empty()) {
    ChunkMetadataVector new_metadata_vec;
    data_mgr.getChunkMetadataVecForKeyPrefix(new_metadata_vec, table_key);
    for (const auto& [key, metadata] : new_metadata_vec) {
      auto it = old_chunk_metadata_by_chunk_key.find(key);
      if (it != old_chunk_metadata_by_chunk_key.end() &&
          it->second->numElements != metadata->numElements) {
        clear_cpu_and_gpu_cache(data_mgr, key);
      }
    }
  }
  LOG(INFO) << "Completed refresh for table: " << td.tableName;
}

void refresh_foreign_table(Catalog_Namespace::Catalog& catalog,
                           const std::string& table_name,
                           const bool evict_cached_entries) {
  if (dist::is_leaf_node() &&
      shared::contains(Catalog_Namespace::kAggregatorOnlySystemTables, table_name)) {
    // Skip aggregator only system tables on leaf nodes.
    return;
  }
  auto table_lock =
      std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>>(
          lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
              catalog, table_name, false));

  const TableDescriptor* td = (*table_lock)();
  if (td->storageType != StorageType::FOREIGN_TABLE) {
    throw std::runtime_error{
        table_name +
        " is not a foreign table. Refreshes are applicable to only foreign tables."};
  }

  auto foreign_table = dynamic_cast<const ForeignTable*>(td);
  CHECK(foreign_table);
  refresh_foreign_table_unlocked(catalog, *foreign_table, evict_cached_entries);
}
}  // namespace foreign_storage
