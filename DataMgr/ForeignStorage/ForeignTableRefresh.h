/*
 * Copyright 2020 OmniSci, Inc.
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

#include <string>

#include "Catalog/Catalog.h"

namespace foreign_storage {
void refresh_foreign_table(Catalog_Namespace::Catalog& catalog,
                           const std::string& table_name,
                           const bool evict_cached_entries);

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
