/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
