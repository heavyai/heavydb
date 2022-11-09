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

#include "OSDependent/heavyai_locks.h"
#include "Shared/heavyai_shared_mutex.h"

namespace Catalog_Namespace {
class SysCatalog;
class Catalog;

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
class cat_lock {
 public:
  cat_lock(const T* cat);
  virtual ~cat_lock();
  virtual void lock() = 0;
  virtual void unlock() = 0;

 protected:
  const T* cat_;
  bool holds_lock_;
};

template <typename T>
class read_lock : public cat_lock<T> {
 public:
  read_lock(const T* cat);
  ~read_lock();
  void lock() override;
  void unlock() override;

 private:
  heavyai::shared_lock<heavyai::shared_mutex> lock_;
  heavyai::shared_lock<heavyai::DistributedSharedMutex> dlock_;
};

template <typename T>
class sqlite_lock : public cat_lock<T> {
 public:
  sqlite_lock(const T* cat);
  ~sqlite_lock();
  void lock() override;
  void unlock() override;

 private:
  // always obtain a read lock on catalog first
  // to ensure correct locking order
  read_lock<T> cat_read_lock_;
  heavyai::unique_lock<heavyai::shared_mutex> lock_;
  heavyai::unique_lock<heavyai::DistributedSharedMutex> dlock_;
};

template <typename T>
class write_lock : public cat_lock<T> {
 public:
  write_lock(const T* cat);
  ~write_lock();
  void lock() override;
  void unlock() override;

 private:
  heavyai::unique_lock<heavyai::shared_mutex> lock_;
  heavyai::unique_lock<heavyai::DistributedSharedMutex> dlock_;
};

class cat_init_lock : public cat_lock<SysCatalog> {
 public:
  cat_init_lock(const SysCatalog* cat);
  ~cat_init_lock();
  void lock() override;
  void unlock() override;

 private:
  // cat_init_lock does not need to support a distributed mutex because in multi-instance
  // mode the cat_lock/sqlite_locks will handle concurrent access as they lock based on
  // catalog db name, rather than catalog instance (they can lock between multiple catalog
  // instances for the same db).
  heavyai::unique_lock<heavyai::shared_mutex> lock_;
};

struct SharedMutexWrapper {
  std::atomic<std::thread::id> thread_holding_lock;
  heavyai::shared_mutex local_mutex;
  std::unique_ptr<heavyai::DistributedSharedMutex> dist_mutex;
};

}  // namespace Catalog_Namespace
