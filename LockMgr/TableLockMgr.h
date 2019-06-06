/*
 * Copyright 2019 OmniSci, Inc.
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

#pragma once

#include <Shared/mapd_shared_mutex.h>
#include <Shared/types.h>

#include <map>
#include <memory>
#include <string>

#include "LockMgr.h"

#include <Catalog/Catalog.h>

namespace Lock_Helpers {

template <typename LOCK_TYPE, typename LOCK_MGR_TYPE>
LOCK_TYPE getLockForTableImpl(const Catalog_Namespace::Catalog& cat,
                              const std::string& table_name) {
  const auto chunk_key = Lock_Namespace::getTableChunkKey(cat, table_name);

  auto& table_lock_mgr = LOCK_MGR_TYPE::instance();
  return LOCK_TYPE(table_lock_mgr.getTableMutex(chunk_key));
}

}  // namespace Lock_Helpers

namespace Lock_Namespace {

using MutexType = mapd_shared_mutex;

using WriteLock = mapd_unique_lock<MutexType>;
using ReadLock = mapd_shared_lock<MutexType>;

struct ReadWriteLockContainer {
  std::vector<ReadLock> read_locks;
  std::vector<WriteLock> write_locks;
};

template <class T>
class TableLockMgrImpl {
 public:
  MutexType& getTableMutex(const ChunkKey table_key) {
    std::lock_guard<std::mutex> access_map_lock(map_mutex_);
    return table_mutex_map_[table_key];
  }

  static void getTableLocks(const Catalog_Namespace::Catalog& cat,
                            const std::map<std::string, bool>& table_names,
                            std::vector<ReadLock>& read_locks,
                            std::vector<WriteLock>& write_locks) {
    for (const auto& table_name_itr : table_names) {
      if (table_name_itr.second) {
        write_locks.push_back(T::getWriteLockForTable(cat, table_name_itr.first));
      } else {
        read_locks.push_back(T::getReadLockForTable(cat, table_name_itr.first));
      }
    }
  }

  static void getTableLocks(const Catalog_Namespace::Catalog& cat,
                            const std::string& query_ra,
                            std::vector<ReadLock>& read_locks,
                            std::vector<WriteLock>& write_locks) {
    // parse ra to learn involved table names
    std::map<std::string, bool> table_names;
    getTableNames(table_names, query_ra);
    return T::getTableLocks(cat, table_names, read_locks, write_locks);
  }

  static WriteLock getWriteLockForTable(const Catalog_Namespace::Catalog& cat,
                                        const std::string& table_name) {
    return Lock_Helpers::getLockForTableImpl<WriteLock, T>(cat, table_name);
  }
  static WriteLock getWriteLockForTable(const ChunkKey table_key) {
    auto& table_lock_mgr = T::instance();
    return WriteLock(table_lock_mgr.getTableMutex(table_key));
  }

  static ReadLock getReadLockForTable(const Catalog_Namespace::Catalog& cat,
                                      const std::string& table_name) {
    return Lock_Helpers::getLockForTableImpl<ReadLock, T>(cat, table_name);
  }
  static ReadLock getReadLockForTable(const ChunkKey table_key) {
    auto& table_lock_mgr = T::instance();
    return ReadLock(table_lock_mgr.getTableMutex(table_key));
  }

 protected:
  TableLockMgrImpl() {}

  std::mutex map_mutex_;
  std::map<ChunkKey, MutexType> table_mutex_map_;
};

class TableLockMgr : public TableLockMgrImpl<TableLockMgr> {
 public:
  static TableLockMgr& instance() {
    static TableLockMgr table_lock_mgr;
    return table_lock_mgr;
  }

 private:
  TableLockMgr() {}
};

}  // namespace Lock_Namespace
