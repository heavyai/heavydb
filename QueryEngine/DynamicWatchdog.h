#ifndef QUERYENGINE_DYNAMICWATCHDOG_H
#define QUERYENGINE_DYNAMICWATCHDOG_H

extern "C" uint64_t dynamic_watchdog_bark(unsigned ms_budget);

extern "C" bool dynamic_watchdog();

#endif  // QUERYENGINE_DYNAMICWATCHDOG_H
