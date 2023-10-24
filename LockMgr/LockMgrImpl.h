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

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include "Catalog/Catalog.h"
#include "OSDependent/heavyai_locks.h"
#include "Shared/heavyai_shared_mutex.h"
#include "Shared/types.h"

namespace lockmgr {

class TableSchemaLockMgr;
class TableDataLockMgr;
class InsertDataLockMgr;

using MutexTypeBase = heavyai::shared_mutex;

class MutexTracker : public heavyai::SharedMutexInterface {
 public:
  MutexTracker(std::unique_ptr<heavyai::DistributedSharedMutex> dmutex)
      : ref_count_(0u), dmutex_(std::move(dmutex)) {}

  virtual void lock();
  virtual bool try_lock();
  virtual void unlock();

  virtual void lock_shared();
  virtual bool try_lock_shared();
  virtual void unlock_shared();

  virtual bool isAcquired() const { return ref_count_.load() > 0; }

 private:
  std::atomic<size_t> ref_count_;
  MutexTypeBase mutex_;
  std::unique_ptr<heavyai::DistributedSharedMutex> dmutex_;
};

using WriteLockBase = heavyai::unique_lock<MutexTracker>;
using ReadLockBase = heavyai::shared_lock<MutexTracker>;

template <typename LOCK>
class TrackedRefLock {
  static_assert(std::is_same_v<LOCK, ReadLockBase> ||
                std::is_same_v<LOCK, WriteLockBase>);

 public:
  TrackedRefLock(MutexTracker* m) : mutex_(checkPointer(m)), lock_(*mutex_) {}

  TrackedRefLock(TrackedRefLock&& other)
      : mutex_(other.mutex_), lock_(std::move(other.lock_)) {
    other.mutex_ = nullptr;
  }

  TrackedRefLock(const TrackedRefLock&) = delete;
  TrackedRefLock& operator=(const TrackedRefLock&) = delete;

 private:
  MutexTracker* mutex_;
  LOCK lock_;

  static MutexTracker* checkPointer(MutexTracker* m) {
    CHECK(m);
    return m;
  }
};

using WriteLock = TrackedRefLock<WriteLockBase>;
using ReadLock = TrackedRefLock<ReadLockBase>;

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
  static_assert(std::is_same_v<T, TableSchemaLockMgr> ||
                std::is_same_v<T, TableDataLockMgr> ||
                std::is_same_v<T, InsertDataLockMgr>);

 public:
  static T& instance();

  virtual ~TableLockMgrImpl() = default;

  virtual MutexTracker* getTableMutex(const ChunkKey& table_key);

  std::set<ChunkKey> getLockedTables() const;

  static WriteLock getWriteLockForTable(const Catalog_Namespace::Catalog& cat,
                                        const std::string& table_name);

  static WriteLock getWriteLockForTable(const ChunkKey table_key);

  static ReadLock getReadLockForTable(Catalog_Namespace::Catalog& cat,
                                      const std::string& table_name);

  static ReadLock getReadLockForTable(const ChunkKey table_key);

 protected:
  TableLockMgrImpl() {}

  virtual std::unique_ptr<heavyai::DistributedSharedMutex> getClusterTableMutex(
      const ChunkKey table_key) const;

  mutable std::mutex map_mutex_;
  std::map<ChunkKey, std::unique_ptr<MutexTracker>> table_mutex_map_;

 private:
  static MutexTracker* getMutexTracker(const Catalog_Namespace::Catalog& catalog,
                                       const std::string& table_name);

  static void validateExistingTable(const Catalog_Namespace::Catalog& catalog,
                                    const std::string& table_name);

  static int32_t validateAndGetExistingTableId(const Catalog_Namespace::Catalog& catalog,
                                               const std::string& table_name);
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const TableLockMgrImpl<T>& lock_mgr) {
  for (const auto& table_key : lock_mgr.getLockedTables()) {
    for (const auto& k : table_key) {
      os << k << " ";
    }
    os << "\n";
  }
  return os;
}

}  // namespace lockmgr
