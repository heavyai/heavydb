#ifndef QUERYENGINE_DYNAMICWATCHDOG_H
#define QUERYENGINE_DYNAMICWATCHDOG_H

enum DynamicWatchdogFlags { DW_DEADLINE = 0, DW_ABORT = -1, DW_RESET = -2 };

extern "C" uint64_t dynamic_watchdog_bark(unsigned ms_budget);

extern "C" bool dynamic_watchdog();

#endif  // QUERYENGINE_DYNAMICWATCHDOG_H
