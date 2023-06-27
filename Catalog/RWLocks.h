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

#ifndef RW_LOCKS_H
#define RW_LOCKS_H

#include "OSDependent/heavyai_locks.h"
#include "Shared/heavyai_shared_mutex.h"

#include <atomic>
#include <thread>

namespace Catalog_Namespace {

/*
 *  The locking sequence for the locks below is as follows:
 *
 *  inter catalog / syscatalog it is always
 *
 *    read or write lock, then the sqlite_lock (if required)
 *
 *  intra catalog and syscatalog
 *
 *    always syscatalog locks (if required), then catalog locks
 */

template <typename T>
class read_lock {
  const T* catalog;
  heavyai::shared_lock<heavyai::shared_mutex> lock;
  heavyai::shared_lock<heavyai::DistributedSharedMutex> dlock;
  bool holds_lock;

  template <typename inner_type>
  void lock_catalog(const inner_type* cat) {
    std::thread::id tid = std::this_thread::get_id();

    if (cat->thread_holding_write_lock != tid && !inner_type::thread_holds_read_lock) {
      if (!g_multi_instance) {
        lock = heavyai::shared_lock<heavyai::shared_mutex>(cat->sharedMutex_);
      } else {
        dlock =
            heavyai::shared_lock<heavyai::DistributedSharedMutex>(*cat->dcatalogMutex_);
      }
      inner_type::thread_holds_read_lock = true;
      holds_lock = true;
    }
  }

 public:
  read_lock(const T* cat) : catalog(cat), holds_lock(false) { lock_catalog(cat); }

  ~read_lock() { unlock(); }

  void unlock() {
    if (holds_lock) {
      T::thread_holds_read_lock = false;
      if (!g_multi_instance) {
        lock.unlock();
      } else {
        dlock.unlock();
      }
      holds_lock = false;
    }
  }
};

template <typename T>
class sqlite_lock {
  // always obtain a read lock on catalog first
  // to ensure correct locking order
  read_lock<T> cat_read_lock;
  const T* catalog;
  heavyai::unique_lock<std::mutex> lock;
  heavyai::unique_lock<heavyai::DistributedSharedMutex> dlock;
  bool holds_lock;

  template <typename inner_type>
  void lock_catalog(const inner_type* cat) {
    std::thread::id tid = std::this_thread::get_id();

    if (cat->thread_holding_sqlite_lock != tid) {
      if (!g_multi_instance) {
        lock = heavyai::unique_lock<std::mutex>(cat->sqliteMutex_);
      } else {
        dlock =
            heavyai::unique_lock<heavyai::DistributedSharedMutex>(*cat->dsqliteMutex_);
      }
      cat->thread_holding_sqlite_lock = tid;
      holds_lock = true;
    }
  }

 public:
  sqlite_lock(const T* cat) : cat_read_lock(cat), catalog(cat), holds_lock(false) {
    lock_catalog(cat);
  }

  ~sqlite_lock() { unlock(); }

  void unlock() {
    if (holds_lock) {
      std::thread::id no_thread;
      catalog->thread_holding_sqlite_lock = no_thread;
      if (!g_multi_instance) {
        lock.unlock();
      } else {
        dlock.unlock();
      }
      cat_read_lock.unlock();
      holds_lock = false;
    }
  }
};

template <typename T>
class write_lock {
  const T* catalog;
  heavyai::unique_lock<heavyai::shared_mutex> lock;
  heavyai::unique_lock<heavyai::DistributedSharedMutex> dlock;
  bool holds_lock;

  template <typename inner_type>
  void lock_catalog(const inner_type* cat) {
    std::thread::id tid = std::this_thread::get_id();

    if (cat->thread_holding_write_lock != tid) {
      if (!g_multi_instance) {
        lock = heavyai::unique_lock<heavyai::shared_mutex>(cat->sharedMutex_);
      } else {
        dlock =
            heavyai::unique_lock<heavyai::DistributedSharedMutex>(*cat->dcatalogMutex_);
      }
      cat->thread_holding_write_lock = tid;
      holds_lock = true;
    }
  }

 public:
  write_lock(const T* cat) : catalog(cat), holds_lock(false) { lock_catalog(cat); }

  ~write_lock() { unlock(); }

  void unlock() {
    if (holds_lock) {
      std::thread::id no_thread;
      catalog->thread_holding_write_lock = no_thread;
      if (!g_multi_instance) {
        lock.unlock();
      } else {
        dlock.unlock();
      }
      holds_lock = false;
    }
  }
};

}  // namespace Catalog_Namespace

#endif  // RW_LOCKS_H
