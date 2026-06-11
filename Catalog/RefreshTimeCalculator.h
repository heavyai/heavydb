/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <string>

namespace foreign_storage {
class RefreshTimeCalculator {
 public:
  static int64_t getNextRefreshTime(
      const std::map<std::string, std::string, std::less<>>& foreign_table_options);
  static int64_t getCurrentTime();

  // For testing purposes only
  static void setMockCurrentTime(int64_t mock_current_time);
  static void resetMockCurrentTime();

 private:
  inline static std::atomic<bool> should_use_mock_current_time_{false};
  inline static std::atomic<int64_t> mock_current_time_{0};
};
}  // namespace foreign_storage
