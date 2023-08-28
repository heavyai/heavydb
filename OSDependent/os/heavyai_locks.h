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

#include <filesystem>

namespace heavyai {

// MutexInterface:
// An exclusive mutex compatible with std::unique_lock.
class MutexInterface {
 public:
  virtual void lock() = 0;
  virtual bool try_lock() = 0;
  virtual void unlock() = 0;

  virtual ~MutexInterface() {}
};

// SharedMutexInterface:
// A sharable mutex compatible with std::unique_lock and std::shared_lock.
class SharedMutexInterface : public MutexInterface {
 public:
  virtual void lock_shared() = 0;
  virtual bool try_lock_shared() = 0;
  virtual void unlock_shared() = 0;

  virtual ~SharedMutexInterface() {}
};

class DistributedSharedMutex final : public SharedMutexInterface {
 public:
  struct Callbacks {
    std::function<void(bool /*write*/)> pre_lock_callback;
    std::function<void(size_t /*version*/)> reload_cache_callback;
    std::function<void(bool /*write*/)> post_lock_callback;
    std::function<void(bool /*write*/)> pre_unlock_callback;
    std::function<void(bool /*write*/)> post_unlock_callback;
  };

  DistributedSharedMutex(std::filesystem::path lockfilename,
                         std::function<void(size_t)> reload_cache_callback = {}) {}

  DistributedSharedMutex(std::filesystem::path lockfilename, Callbacks callbacks) {}

  ~DistributedSharedMutex() {}

  virtual void lock() {}
  virtual bool try_lock() { return true; }
  virtual void unlock() {}
  virtual void lock_shared() {}
  virtual bool try_lock_shared() { return true; }
  virtual void unlock_shared() {}
  virtual void convert_lock() {}
  virtual bool try_convert_lock() { return true; }
  virtual void convert_lock_shared() {}
  virtual bool try_convert_lock_shared() { return true; }

};  // class DistributedSharedMutex

}  // namespace heavyai
