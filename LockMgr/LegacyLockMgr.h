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

#include <rapidjson/document.h>
#include <boost/variant.hpp>
#include <map>
#include <mutex>
#include <tuple>

#include "Catalog/Catalog.h"
#include "Shared/heavyai_shared_mutex.h"
#include "Shared/types.h"
#include "ThriftHandler/QueryState.h"

namespace legacylockmgr {

// NOTE(sy): Currently the legacylockmgr is only used (or "abused") as a global lock.

enum LockType { ExecutorOuterLock /*, LockMax*/ };

template <typename MutexType>
class WrapperType final {
 public:
  WrapperType()
      : dmutex_(std::filesystem::path(g_base_path) / shared::kLockfilesDirectoryName /
                "global.lockfile") {}

  void lock() {
    if (!g_multi_instance) {
      mutex_.lock();
    } else {
      dmutex_.lock();
    }
  }
  bool try_lock() {
    if (!g_multi_instance) {
      return mutex_.try_lock();
    } else {
      return dmutex_.try_lock();
    }
  }
  void unlock() {
    if (!g_multi_instance) {
      mutex_.unlock();
    } else {
      dmutex_.unlock();
    }
  }

  void lock_shared() {
    if (!g_multi_instance) {
      mutex_.lock_shared();
    } else {
      dmutex_.lock_shared();
    }
  }
  bool try_lock_shared() {
    if (!g_multi_instance) {
      return mutex_.try_lock_shared();
    } else {
      return dmutex_.try_lock_shared();
    }
  }
  void unlock_shared() {
    if (!g_multi_instance) {
      mutex_.unlock_shared();
    } else {
      dmutex_.unlock_shared();
    }
  }

 private:
  MutexType mutex_;
  heavyai::DistributedSharedMutex dmutex_;
};

template <typename MutexType, typename KeyType>
class LockMgr {
 public:
  static std::shared_ptr<WrapperType<MutexType>> getMutex(const LockType lockType,
                                                          const KeyType& key);

 private:
  static std::mutex aMutex_;
  static std::map<std::tuple<LockType, KeyType>, std::shared_ptr<WrapperType<MutexType>>>
      mutexMap_;
};

template <typename MutexType, typename KeyType>
std::mutex LockMgr<MutexType, KeyType>::aMutex_;
template <typename MutexType, typename KeyType>
std::map<std::tuple<LockType, KeyType>, std::shared_ptr<WrapperType<MutexType>>>
    LockMgr<MutexType, KeyType>::mutexMap_;

template <typename MutexType, typename KeyType>
std::shared_ptr<WrapperType<MutexType>> LockMgr<MutexType, KeyType>::getMutex(
    const LockType lock_type,
    const KeyType& key) {
  auto lock_key = std::make_tuple(lock_type, key);

  std::unique_lock<std::mutex> lck(aMutex_);
  auto mit = mutexMap_.find(lock_key);
  if (mit != mutexMap_.end()) {
    return mit->second;
  }

  auto tMutex = std::make_shared<WrapperType<MutexType>>();
  mutexMap_[lock_key] = tMutex;
  return tMutex;
}

using ExecutorWriteLock = std::unique_lock<WrapperType<std::shared_mutex>>;
using ExecutorReadLock = std::shared_lock<WrapperType<std::shared_mutex>>;

inline auto getExecuteWriteLock() {
  VLOG(1) << "Attempting to acquire the Executor Write Lock";
  auto ret = ExecutorWriteLock(
      *LockMgr<std::shared_mutex, bool>::getMutex(ExecutorOuterLock, true));
  VLOG(1) << "Acquired the ExecutorWriteLock";
  return ret;
}

inline auto getExecuteReadLock() {
  VLOG(1) << "Attempting to acquire the Executor Read Lock";
  auto ret = ExecutorReadLock(
      *LockMgr<std::shared_mutex, bool>::getMutex(ExecutorOuterLock, true));
  VLOG(1) << "Acquired the Executor Read Lock";
  return ret;
}

}  // namespace legacylockmgr
