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

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <type_traits>

#include "Catalog/Catalog.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/types.h"

namespace lockmgr {

using MutexTypeBase = mapd_shared_mutex;

using WriteLockBase = mapd_unique_lock<MutexTypeBase>;
using ReadLockBase = mapd_shared_lock<MutexTypeBase>;
class TrackedRefMutex {
 public:
  TrackedRefMutex() : ref_count_(0u) {}

  MutexTypeBase& lock() {
    ref_count_.fetch_add(1u);
    return mutex_;
  }

  void release() {
    const auto stored_ref_count = ref_count_.fetch_sub(1u);
    CHECK_GE(stored_ref_count, size_t(1));
  }

  bool isLocked() const { return ref_count_.load() > 0; }

 private:
  std::atomic<size_t> ref_count_;
  MutexTypeBase mutex_;
};

template <typename LOCK>
class TrackedRefLock {
 public:
  TrackedRefLock(TrackedRefMutex* m) : mutex_(m), lock_(mutex_->lock()) { CHECK(mutex_); }

  ~TrackedRefLock() {
    if (mutex_) {
      // This call only decrements the ref count. The actual release is done once the
      // mutex is destroyed.
      mutex_->release();
    }
  }

  TrackedRefLock(TrackedRefLock&& other)
      : mutex_(other.mutex_), lock_(std::move(other.lock_)) {
    other.mutex_ = nullptr;
  }

  TrackedRefLock(const TrackedRefLock&) = delete;
  TrackedRefLock& operator=(const TrackedRefLock&) = delete;

 private:
  TrackedRefMutex* mutex_;
  LOCK lock_;
};  // namespace lockmgr

using MutexType = TrackedRefMutex;

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
 public:
  MutexType* getTableMutex(const ChunkKey table_key) {
    std::lock_guard<std::mutex> access_map_lock(map_mutex_);
    auto mutex_it = table_mutex_map_.find(table_key);
    if (mutex_it == table_mutex_map_.end()) {
      table_mutex_map_.insert(std::make_pair(table_key, std::make_unique<MutexType>()));
    } else {
      return mutex_it->second.get();
    }
    return table_mutex_map_[table_key].get();
  }

  std::set<ChunkKey> getLockedTables() const {
    std::set<ChunkKey> ret;
    std::lock_guard<std::mutex> access_map_lock(map_mutex_);
    for (const auto& kv : table_mutex_map_) {
      if (kv.second->isLocked()) {
        ret.insert(kv.first);
      }
    }

    return ret;
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

  mutable std::mutex map_mutex_;
  std::map<ChunkKey, std::unique_ptr<MutexType>> table_mutex_map_;
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
