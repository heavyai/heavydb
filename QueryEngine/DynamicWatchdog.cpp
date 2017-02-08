#include <chrono>
#include <thread>

#include "DynamicWatchdog.h"
#include <glog/logging.h>

#if (defined(__x86_64__) || defined(__x86_64))
static __inline__ uint64_t rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}
#endif

extern "C" uint64_t dynamic_watchdog_bark(unsigned ms_budget) {
#if (defined(__x86_64__) || defined(__x86_64))
  static uint64_t dw_cycle_start = 0ULL;
  static uint64_t dw_cycle_budget = 0ULL;
  if (ms_budget == 0) {
    // Return the deadline
    return dw_cycle_start + dw_cycle_budget;
  }
  // Init cycle start, measure freq, set and return cycle budget
  dw_cycle_start = rdtsc();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  auto freq_kHz = rdtsc() - dw_cycle_start;
  dw_cycle_budget = freq_kHz * static_cast<uint64_t>(ms_budget);
  VLOG(1) << "INIT: thread " << std::this_thread::get_id() << ": ms_budget " << ms_budget << ", cycle_start "
          << dw_cycle_start << ", cycle_budget " << dw_cycle_budget << ", dw_deadline "
          << dw_cycle_start + dw_cycle_budget;
  return dw_cycle_budget;
#else
  return 0LL;
#endif
}

// timeout detection
extern "C" bool dynamic_watchdog() {
#if (defined(__x86_64__) || defined(__x86_64))
  // Check if out of time
  auto clock = rdtsc();
  auto dw_deadline = dynamic_watchdog_bark(0);
  if (clock > dw_deadline) {
    LOG(INFO) << "TIMEOUT: thread " << std::this_thread::get_id() << ": clock " << clock << ", deadline "
              << dw_deadline;
    return true;
  }
#endif
  return false;
}
