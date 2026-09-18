/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if !defined(NDEBUG) || defined(HAVE_TSAN)
#define HEAVYDB_DEBUG_TEST_HOOKS_ENABLED 1
#else
#define HEAVYDB_DEBUG_TEST_HOOKS_ENABLED 0
#endif

#if HEAVYDB_DEBUG_TEST_HOOKS_ENABLED

#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace heavyai::test {

// Named test callbacks for observing or adjusting production state in Debug and TSAN
// builds. ScopedDebugTestHook owns registration in a test; HEAVYDB_DEBUG_TEST_HOOK
// invokes it from production code without holding the registry mutex.
class DebugTestHook {
 public:
  template <typename T, typename Callable>
  static void set(std::string point, Callable&& callable) {
    static_assert(std::is_invocable_v<std::decay_t<Callable>&, T*>,
                  "Debug test hook callback must accept T*");
    setCallback(std::move(point),
                [callback = std::forward<Callable>(callable)](void* data) mutable {
                  std::invoke(callback, static_cast<T*>(data));
                });
  }

  static void clear(const std::string& point) {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.erase(point);
  }

  static void run(const std::string& point, void* data) {
    Callback callback;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto callback_it = callbacks_.find(point);
      if (callback_it == callbacks_.end()) {
        return;
      }
      callback = callback_it->second;
    }
    callback(data);
  }

 private:
  using Callback = std::function<void(void*)>;

  static void setCallback(std::string point, Callback callback) {
    if (point.empty()) {
      throw std::invalid_argument("Debug test hook name must not be empty");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto [callback_it, inserted] =
        callbacks_.emplace(std::move(point), std::move(callback));
    if (!inserted) {
      throw std::logic_error("Debug test hook is already registered: " +
                             callback_it->first);
    }
  }

  inline static std::mutex mutex_;
  inline static std::unordered_map<std::string, Callback> callbacks_;
};

template <typename T>
class ScopedDebugTestHook {
 public:
  template <typename Callable>
  ScopedDebugTestHook(std::string point, Callable&& callable) : point_(std::move(point)) {
    DebugTestHook::set<T>(point_, std::forward<Callable>(callable));
  }

  ~ScopedDebugTestHook() { DebugTestHook::clear(point_); }

  ScopedDebugTestHook(const ScopedDebugTestHook&) = delete;
  ScopedDebugTestHook& operator=(const ScopedDebugTestHook&) = delete;

 private:
  const std::string point_;
};

}  // namespace heavyai::test

#define HEAVYDB_DEBUG_TEST_HOOK(point, data) \
  ::heavyai::test::DebugTestHook::run((point), (data))

#else

// Do not evaluate either argument or leave a hook call in ordinary release builds.
#define HEAVYDB_DEBUG_TEST_HOOK(point, data)

#endif
