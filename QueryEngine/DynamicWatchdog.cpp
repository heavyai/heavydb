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

#include <atomic>
#include <chrono>
#include <thread>

#include "DynamicWatchdog.h"
#include <glog/logging.h>

static __inline__ uint64_t read_cycle_counter(void) {
#if (defined(__x86_64__) || defined(__x86_64))
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
#else
  // Plug in other architectures' cycle counter reads, e.g. MRC on ARM
  return 0LL;
#endif
}

extern "C" uint64_t dynamic_watchdog_init(unsigned ms_budget) {
  static uint64_t dw_cycle_start = 0ULL;
  static uint64_t dw_cycle_budget = 0ULL;
  static std::atomic_bool dw_abort{false};

  if (ms_budget == static_cast<unsigned>(DW_DEADLINE)) {
    if (dw_abort.load())
      return 0LL;
    return dw_cycle_start + dw_cycle_budget;
  }
  if (ms_budget == static_cast<unsigned>(DW_ABORT)) {
    dw_abort = true;
    return 0LL;
  }
  if (ms_budget == static_cast<unsigned>(DW_RESET)) {
    dw_abort = false;
    return 0LL;
  }

  // Init cycle start, measure freq, set and return cycle budget
  dw_cycle_start = read_cycle_counter();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  auto freq_kHz = read_cycle_counter() - dw_cycle_start;
  dw_cycle_budget = freq_kHz * static_cast<uint64_t>(ms_budget);
  VLOG(1) << "INIT: thread " << std::this_thread::get_id() << ": ms_budget " << ms_budget << ", cycle_start "
          << dw_cycle_start << ", cycle_budget " << dw_cycle_budget << ", dw_deadline "
          << dw_cycle_start + dw_cycle_budget;
  return dw_cycle_budget;
}

// timeout detection
extern "C" bool dynamic_watchdog() {
  auto clock = read_cycle_counter();
  auto dw_deadline = dynamic_watchdog_init(static_cast<unsigned>(DW_DEADLINE));
  if (clock > dw_deadline) {
    LOG(INFO) << "TIMEOUT: thread " << std::this_thread::get_id() << ": clock " << clock << ", deadline "
              << dw_deadline;
    return true;
  }
  return false;
}
