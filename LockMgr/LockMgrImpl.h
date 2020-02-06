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

#pragma once

#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include "Catalog/Catalog.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/types.h"
namespace Lock_Namespace {
void getTableNames(std::map<std::string, bool>& tableNames, const std::string query_ra);
}
namespace lockmgr {

using MutexType = mapd_shared_mutex;

using WriteLock = mapd_unique_lock<MutexType>;
using ReadLock = mapd_shared_lock<MutexType>;

struct TableLock {
  WriteLock write_lock;
  ReadLock read_lock;
};

template <typename T>
class AbstractLockContainer {
 public:
  virtual T operator()() const = 0;

  virtual ~AbstractLockContainer() {}
};

template <typename T, typename LOCK>
class LockContainerImpl : public AbstractLockContainer<T> {
 public:
  T operator()() const final { return obj_; }

 protected:
  LockContainerImpl(T obj, LOCK&& lock) : obj_(obj), lock_(std::move(lock)) {}

  T obj_;
  LOCK lock_;
};

namespace helpers {
ChunkKey chunk_key_for_table(const Catalog_Namespace::Catalog& cat,
                             const std::string& tableName);

template <typename LOCK_TYPE, typename LOCK_MGR_TYPE>
LOCK_TYPE getLockForKeyImpl(const ChunkKey& chunk_key) {
  auto& table_lock_mgr = LOCK_MGR_TYPE::instance();
  return LOCK_TYPE(table_lock_mgr.getTableMutex(chunk_key));
}

template <typename LOCK_TYPE, typename LOCK_MGR_TYPE>
LOCK_TYPE getLockForTableImpl(const Catalog_Namespace::Catalog& cat,
                              const std::string& table_name) {
  const auto chunk_key = chunk_key_for_table(cat, table_name);

  auto& table_lock_mgr = LOCK_MGR_TYPE::instance();
  return LOCK_TYPE(table_lock_mgr.getTableMutex(chunk_key));
}

}  // namespace helpers

template <class T>
class TableLockMgrImpl {
 public:
  MutexType& getTableMutex(const ChunkKey table_key) {
    std::lock_guard<std::mutex> access_map_lock(map_mutex_);
    return table_mutex_map_[table_key];
  }

  static void getTableLocks(const Catalog_Namespace::Catalog& cat,
                            const std::map<std::string, bool>& table_names,
                            std::vector<TableLock>& table_locks) {
    for (const auto& table_name_itr : table_names) {
      TableLock table_lock;
      if (table_name_itr.second) {
        table_lock.write_lock = T::getWriteLockForTable(cat, table_name_itr.first);
      } else {
        table_lock.read_lock = T::getReadLockForTable(cat, table_name_itr.first);
      }
      table_locks.emplace_back(std::move(table_lock));
    }
  }

  static void getTableLocks(const Catalog_Namespace::Catalog& cat,
                            const std::string& query_ra,
                            std::vector<TableLock>& table_locks) {
    // parse ra to learn involved table names
    std::map<std::string, bool> table_names;
    Lock_Namespace::getTableNames(table_names, query_ra);
    return T::getTableLocks(cat, table_names, table_locks);
  }

  static WriteLock getWriteLockForTable(const Catalog_Namespace::Catalog& cat,
                                        const std::string& table_name) {
    return helpers::getLockForTableImpl<WriteLock, T>(cat, table_name);
  }
  static WriteLock getWriteLockForTable(const ChunkKey table_key) {
    auto& table_lock_mgr = T::instance();
    return WriteLock(table_lock_mgr.getTableMutex(table_key));
  }

  static ReadLock getReadLockForTable(const Catalog_Namespace::Catalog& cat,
                                      const std::string& table_name) {
    return helpers::getLockForTableImpl<ReadLock, T>(cat, table_name);
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

}  // namespace lockmgr
