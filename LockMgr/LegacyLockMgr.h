/*
 * Copyright 2017 MapD Technologies, Inc.
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
#include "Shared/mapd_shared_mutex.h"
#include "Shared/types.h"
#include "ThriftHandler/QueryState.h"

namespace legacylockmgr {

enum LockType { ExecutorOuterLock, LockMax };

template <typename MutexType, typename KeyType>
class LockMgr {
 public:
  static std::shared_ptr<MutexType> getMutex(const LockType lockType, const KeyType& key);

 private:
  static std::mutex aMutex_;
  static std::map<std::tuple<LockType, KeyType>, std::shared_ptr<MutexType>> mutexMap_;
};

template <typename MutexType, typename KeyType>
std::mutex LockMgr<MutexType, KeyType>::aMutex_;
template <typename MutexType, typename KeyType>
std::map<std::tuple<LockType, KeyType>, std::shared_ptr<MutexType>>
    LockMgr<MutexType, KeyType>::mutexMap_;

template <typename MutexType, typename KeyType>
std::shared_ptr<MutexType> LockMgr<MutexType, KeyType>::getMutex(const LockType lock_type,
                                                                 const KeyType& key) {
  auto lock_key = std::make_tuple(lock_type, key);

  std::unique_lock<std::mutex> lck(aMutex_);
  auto mit = mutexMap_.find(lock_key);
  if (mit != mutexMap_.end()) {
    return mit->second;
  }

  auto tMutex = std::make_shared<MutexType>();
  mutexMap_[lock_key] = tMutex;
  return tMutex;
}

}  // namespace legacylockmgr
