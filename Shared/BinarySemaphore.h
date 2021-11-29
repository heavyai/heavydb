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

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace SemaphoreShim_Namespace {

/**
 * @type BinarySemaphore
 * @brief Utility type that implemnts behavior of a blocking binary
 * semaphore, with an optional timeout. May be removed in favor of
 * the C++20 std::binary_semaphore as soon as we can upgrade to gcc 11.
 */

class BinarySemaphore {
 public:
  /**
   * @brief Blocks calling thread until it can acquire the semaphore,
   * i.e. release is called by another thread.
   *
   * Note that this call will block indefinitely. If a timeout is required,
   * use try_acquire_for()
   */

  inline void try_acquire() {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    condition_.wait(state_lock, [& is_ready = is_ready_] { return is_ready; });
  }

  /**
   * @brief Blocks calling thread until it can acquire the semaphore,
   * i.e. release is called by another thread, or max_wait_in_ms
   * duration passes, whichever occurs first
   *
   * Note that this function does not throw to maximize its
   * performance in performance-critical scenarios. Instead, it returns
   * true if the call completed successfully and false if it timed out.
   *
   * @return A boolean value that is set to true if the semaphore lock
   * was able to be acquired successfully, or false if it timed out
   * (wait duration exceeded max_wait_in_ms)
   *
   */

  inline bool try_acquire_for(const size_t max_wait_in_ms) {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    return condition_.wait_for(state_lock,
                               std::chrono::milliseconds(max_wait_in_ms),
                               [&is_ready = is_ready_]() { return is_ready; });
  }

  /**
   * @brief Sets internal is_ready variable to true, allowing another
   * thread waiting for the semaphore to proceed
   */

  inline void release() {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    is_ready_ = true;
    // notify the waiting thread
    condition_.notify_one();
  }

  /**
   * @brief Resets the semaphore's ready condition to false
   *
   */

  inline void reset() {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    is_ready_ = false;
  }

  inline void set_ready() {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    is_ready_ = true;
  }

 private:
  bool is_ready_{false};
  std::mutex state_mutex_;
  std::condition_variable condition_;
};

}  // namespace SemaphoreShim_Namespace
