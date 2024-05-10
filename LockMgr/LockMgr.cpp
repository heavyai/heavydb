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

#include "LockMgr/LockMgr.h"
#include "Catalog/Catalog.h"
#include "LockMgr/LockMgrImpl.h"

namespace lockmgr {

namespace helpers {

ChunkKey chunk_key_for_table(const Catalog_Namespace::Catalog& cat,
                             const std::string& tableName) {
  const auto table_id = cat.getTableId(tableName);
  if (table_id.has_value()) {
    ChunkKey chunk_key{cat.getCurrentDB().dbId, table_id.value()};
    return chunk_key;
  } else {
    throw Catalog_Namespace::TableNotFoundException(tableName, cat.getCurrentDB().dbName);
  }
}

}  // namespace helpers

void MutexTracker::lock() {
  ref_count_.fetch_add(1u);
  if (!g_multi_instance) {
    mutex_.lock();
  } else {
    dmutex_->lock();
  }
}

bool MutexTracker::try_lock() {
  bool gotlock{false};
  if (!g_multi_instance) {
    gotlock = mutex_.try_lock();
  } else {
    gotlock = dmutex_->try_lock();
  }
  if (gotlock) {
    ref_count_.fetch_add(1u);
  }
  return gotlock;
}

void MutexTracker::unlock() {
  if (!g_multi_instance) {
    mutex_.unlock();
  } else {
    dmutex_->unlock();
  }
  ref_count_.fetch_sub(1u);
}

void MutexTracker::lock_shared() {
  ref_count_.fetch_add(1u);
  if (!g_multi_instance) {
    mutex_.lock_shared();
  } else {
    dmutex_->lock_shared();
  }
}

bool MutexTracker::try_lock_shared() {
  bool gotlock{false};
  if (!g_multi_instance) {
    gotlock = mutex_.try_lock_shared();
  } else {
    gotlock = dmutex_->try_lock_shared();
  }
  if (gotlock) {
    ref_count_.fetch_add(1u);
  }
  return gotlock;
}

void MutexTracker::unlock_shared() {
  if (!g_multi_instance) {
    mutex_.unlock_shared();
  } else {
    dmutex_->unlock_shared();
  }
  ref_count_.fetch_sub(1u);
}

template <class T>
T& instance() {
  static T mgr;
  return mgr;
}

template <class T>
MutexTracker* TableLockMgrImpl<T>::getTableMutex(const ChunkKey& table_key) {
  std::lock_guard<std::mutex> access_map_lock(map_mutex_);
  auto mutex_it = table_mutex_map_.find(table_key);
  if (mutex_it != table_mutex_map_.end()) {
    return mutex_it->second.get();
  }

  // NOTE(sy): Only used by --multi-instance clusters.
  std::unique_ptr<heavyai::DistributedSharedMutex> dmutex =
      getClusterTableMutex(table_key);

  return table_mutex_map_
      .emplace(table_key, std::make_unique<MutexTracker>(std::move(dmutex)))
      .first->second.get();
}

template <class T>
std::set<ChunkKey> TableLockMgrImpl<T>::getLockedTables() const {
  std::set<ChunkKey> ret;
  std::lock_guard<std::mutex> access_map_lock(map_mutex_);
  for (const auto& kv : table_mutex_map_) {
    if (kv.second->isAcquired()) {
      ret.insert(kv.first);
    }
  }

  return ret;
}

template <class T>
WriteLock TableLockMgrImpl<T>::getWriteLockForTable(const Catalog_Namespace::Catalog& cat,
                                                    const std::string& table_name) {
  auto lock = WriteLock(getMutexTracker(cat, table_name));
  // Ensure table still exists after lock is acquired.
  validateExistingTable(cat, table_name);
  return lock;
}

template <class T>
WriteLock TableLockMgrImpl<T>::getWriteLockForTable(const ChunkKey& table_key) {
  auto& table_lock_mgr = T::instance();
  return WriteLock(table_lock_mgr.getTableMutex(table_key));
}

template <class T>
ReadLock TableLockMgrImpl<T>::getReadLockForTable(Catalog_Namespace::Catalog& cat,
                                                  const std::string& table_name) {
  auto lock = ReadLock(getMutexTracker(cat, table_name));
  // Ensure table still exists after lock is acquired.
  validateAndGetExistingTableId(cat, table_name);
  return lock;
}

template <class T>
ReadLock TableLockMgrImpl<T>::getReadLockForTable(const ChunkKey& table_key) {
  auto& table_lock_mgr = T::instance();
  return ReadLock(table_lock_mgr.getTableMutex(table_key));
}

template <class T>
std::unique_ptr<heavyai::DistributedSharedMutex>
TableLockMgrImpl<T>::getClusterTableMutex(const ChunkKey& table_key) const {
  std::unique_ptr<heavyai::DistributedSharedMutex> table_mutex;

  std::string table_key_as_text;
  for (auto n : table_key) {
    table_key_as_text += (!table_key_as_text.empty() ? "_" : "") + std::to_string(n);
  }

  // A callback used for syncing with most of the changed Catalog metadata, in-general,
  // such as the list of tables that exist, dashboards, etc. This is accomplished by
  // read locking, and immediately unlocking, dcatalogMutex_, so
  // cat->reloadCatalogMetadataUnlocked() will be called.
  auto cb_reload_catalog_metadata = [table_key](bool write) {
    if constexpr (T::kind == "insert") {
      CHECK(write);  // The insert lock is for writing, never for reading.
    }
    auto cat =
        Catalog_Namespace::SysCatalog::instance().getCatalog(table_key[CHUNK_KEY_DB_IDX]);
    CHECK(cat);
    heavyai::shared_lock<heavyai::DistributedSharedMutex> dread_lock(
        *cat->dcatalogMutex_);
  };

  if constexpr (T::kind == "schema") {
    // A callback used for reloading the Catalog schema for the one table being locked.
    auto cb_reload_table_metadata = [table_key, table_key_as_text](size_t version) {
      VLOG(2) << "reloading table metadata for: table_" << table_key_as_text;
      CHECK_EQ(table_key.size(), 2U);
      auto cat = Catalog_Namespace::SysCatalog::instance().getCatalog(
          table_key[CHUNK_KEY_DB_IDX]);
      CHECK(cat);
      heavyai::shared_lock<heavyai::DistributedSharedMutex> dread_lock(
          *cat->dcatalogMutex_);
      cat->reloadTableMetadataUnlocked(table_key[CHUNK_KEY_TABLE_IDX]);
    };

    // Create the table mutex.
    heavyai::DistributedSharedMutex::Callbacks cbs{
        cb_reload_catalog_metadata,  // pre_lock_callback
        cb_reload_table_metadata     // reload_cache_callback
    };
    auto schema_lockfile{
        std::filesystem::path(g_base_path) / shared::kLockfilesDirectoryName /
        shared::kCatalogDirectoryName /
        ("table_" + table_key_as_text + "." + T::kind.data() + ".lockfile")};
    table_mutex =
        std::make_unique<heavyai::DistributedSharedMutex>(schema_lockfile.string(), cbs);
  } else if constexpr (T::kind == "data" || T::kind == "insert") {
    // A callback used for reloading the DataMgr data for the one table being locked.
    auto cb_reload_table_data = [table_key, table_key_as_text](size_t version) {
      VLOG(2) << "invalidating table caches for new version " << version << " of: table_"
              << table_key_as_text;
      CHECK_EQ(table_key.size(), 2U);
      auto cat = Catalog_Namespace::SysCatalog::instance().getCatalog(
          table_key[CHUNK_KEY_DB_IDX]);
      CHECK(cat);
      cat->invalidateCachesForTable(table_key[CHUNK_KEY_TABLE_IDX]);
    };

    // Create the rows mutex.
    auto rows_lockfile{std::filesystem::path(g_base_path) /
                       shared::kLockfilesDirectoryName / shared::kDataDirectoryName /
                       ("table_" + table_key_as_text + ".rows.lockfile")};
    std::shared_ptr<heavyai::DistributedSharedMutex> rows_mutex =
        std::make_shared<heavyai::DistributedSharedMutex>(
            rows_lockfile.string(),
            cb_reload_table_data  // reload_cache_callback
        );

    // A callback used for syncing with outside changes to this table's row data.
    auto cb_reload_row_data = [table_key, rows_mutex](bool write) {
      heavyai::shared_lock<heavyai::DistributedSharedMutex> rows_read_lock(*rows_mutex);
    };

    // A callback to notify other nodes about our changes to this table's row data.
    auto cb_notify_about_row_data = [table_key, rows_mutex](bool write) {
      if (write) {
        heavyai::unique_lock<heavyai::DistributedSharedMutex> rows_write_lock(
            *rows_mutex);
      }
    };

    // Create the table mutex.
    heavyai::DistributedSharedMutex::Callbacks cbs{
        cb_reload_catalog_metadata,  // pre_lock_callback
        {},
        cb_reload_row_data,       // post_lock_callback
        cb_notify_about_row_data  // pre_unlock_callback
    };
    auto table_lockfile{
        std::filesystem::path(g_base_path) / shared::kLockfilesDirectoryName /
        shared::kDataDirectoryName /
        ("table_" + table_key_as_text + "." + T::kind.data() + ".lockfile")};
    table_mutex =
        std::make_unique<heavyai::DistributedSharedMutex>(table_lockfile.string(), cbs);
  } else {
    UNREACHABLE() << "unexpected lockmgr kind: " << T::kind;
  }

  return table_mutex;
}

template <class T>
MutexTracker* TableLockMgrImpl<T>::getMutexTracker(
    const Catalog_Namespace::Catalog& catalog,
    const std::string& table_name) {
  ChunkKey chunk_key{catalog.getDatabaseId(),
                     validateAndGetExistingTableId(catalog, table_name)};
  auto& table_lock_mgr = T::instance();
  MutexTracker* tracker = table_lock_mgr.getTableMutex(chunk_key);
  CHECK(tracker);
  return tracker;
}

template <class T>
void TableLockMgrImpl<T>::validateExistingTable(const Catalog_Namespace::Catalog& catalog,
                                                const std::string& table_name) {
  validateAndGetExistingTableId(catalog, table_name);
}

template <class T>
int32_t TableLockMgrImpl<T>::validateAndGetExistingTableId(
    const Catalog_Namespace::Catalog& catalog,
    const std::string& table_name) {
  auto table_id = catalog.getTableId(table_name);
  if (!table_id.has_value()) {
    throw Catalog_Namespace::TableNotFoundException(table_name, catalog.name());
  }
  return table_id.value();
}

template class TableLockMgrImpl<TableSchemaLockMgr>;
template class TableLockMgrImpl<TableDataLockMgr>;
template class TableLockMgrImpl<InsertDataLockMgr>;

}  // namespace lockmgr
