/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace foreign_storage {
class ForeignTableRefreshScheduler {
 public:
  static void start(std::atomic<bool>& is_program_running);
  static void stop();

  // The following methods are for testing purposes only
  static void setWaitDuration(int64_t duration_in_seconds);
  static bool isRunning();
  static bool hasRefreshedTable();
  static void resetHasRefreshedTable();

 private:
  static std::atomic<bool> is_scheduler_running_;
  static std::chrono::seconds thread_wait_duration_;
  static std::thread scheduler_thread_;
  static std::atomic<bool> has_refreshed_table_;
  static std::mutex wait_mutex_;
  static std::condition_variable wait_condition_;
};
}  // namespace foreign_storage
