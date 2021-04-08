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

#include "ForeignTableRefreshScheduler.h"

#include "DataMgr/ForeignStorage/ForeignTableRefresh.h"
#include "LockMgr/LockMgr.h"
#include "QueryEngine/ExternalCacheInvalidators.h"

namespace foreign_storage {

void ForeignTableRefreshScheduler::invalidateQueryEngineCaches() {
  auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
      *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
          legacylockmgr::ExecutorOuterLock, true));
  UpdateTriggeredCacheInvalidator::invalidateCaches();
}

void ForeignTableRefreshScheduler::start(std::atomic<bool>& is_program_running) {
  if (is_program_running && !is_scheduler_running_) {
    is_scheduler_running_ = true;
    scheduler_thread_ = std::thread([&is_program_running]() {
      while (is_program_running && is_scheduler_running_) {
        auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
        // Exit if scheduler has been stopped asynchronously
        if (!is_program_running || !is_scheduler_running_) {
          return;
        }
        bool at_least_one_table_refreshed = false;
        for (const auto& catalog : sys_catalog.getCatalogsForAllDbs()) {
          // Exit if scheduler has been stopped asynchronously
          if (!is_program_running || !is_scheduler_running_) {
            return;
          }
          auto tables = catalog->getAllForeignTablesForRefresh();
          for (auto table : tables) {
            // Exit if scheduler has been stopped asynchronously
            if (!is_program_running || !is_scheduler_running_) {
              return;
            }
            try {
              refresh_foreign_table(*catalog, table->tableName, false);
            } catch (std::runtime_error& e) {
              LOG(ERROR) << "Scheduled refresh for table \"" << table->tableName
                         << "\" resulted in an error. " << e.what();
            }
            has_refreshed_table_ = true;
            at_least_one_table_refreshed = true;
          }
        }
        if (at_least_one_table_refreshed) {
          invalidateQueryEngineCaches();
        }
        // Exit if scheduler has been stopped asynchronously
        if (!is_program_running || !is_scheduler_running_) {
          return;
        }

        // A condition variable is used here (instead of a sleep call)
        // in order to allow for thread wake-up, even in the middle
        // of a wait interval.
        std::unique_lock<std::mutex> wait_lock(wait_mutex_);
        wait_condition_.wait_for(wait_lock, thread_wait_duration_);
      }
    });
  }
}

void ForeignTableRefreshScheduler::stop() {
  if (is_scheduler_running_) {
    is_scheduler_running_ = false;
    wait_condition_.notify_one();
    scheduler_thread_.join();
  }
}

void ForeignTableRefreshScheduler::setWaitDuration(int64_t duration_in_seconds) {
  thread_wait_duration_ = std::chrono::seconds{duration_in_seconds};
}

bool ForeignTableRefreshScheduler::isRunning() {
  return is_scheduler_running_;
}

bool ForeignTableRefreshScheduler::hasRefreshedTable() {
  return has_refreshed_table_;
}

void ForeignTableRefreshScheduler::resetHasRefreshedTable() {
  has_refreshed_table_ = false;
}

std::atomic<bool> ForeignTableRefreshScheduler::is_scheduler_running_{false};
std::chrono::seconds ForeignTableRefreshScheduler::thread_wait_duration_{60};
std::thread ForeignTableRefreshScheduler::scheduler_thread_;
std::atomic<bool> ForeignTableRefreshScheduler::has_refreshed_table_{false};
std::mutex ForeignTableRefreshScheduler::wait_mutex_;
std::condition_variable ForeignTableRefreshScheduler::wait_condition_;
}  // namespace foreign_storage
