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
