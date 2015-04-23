#include "ExtractFromTime.h"

#ifndef __CUDACC__
#include <glog/logging.h>
#endif

/*
 * @brief support the SQL EXTRACT function
 */
extern "C" __attribute__((noinline))
#ifdef __CUDACC__
__device__
#endif
int64_t ExtractFromTime(ExtractField field, time_t timeval) {
  if (field == kEPOCH)
    return timeval;
  tm tm_struct;
#ifdef __CUDACC__
  gmtime_r_cuda(&timeval, &tm_struct);
#else
  gmtime_r(&timeval, &tm_struct);
#endif
  switch (field) {
    case kYEAR:
      return 1900 + tm_struct.tm_year;
    case kMONTH:
      return tm_struct.tm_mon + 1;
    case kDAY:
      return tm_struct.tm_mday;
    case kHOUR:
      return tm_struct.tm_hour;
    case kMINUTE:
      return tm_struct.tm_min;
    case kSECOND:
      return tm_struct.tm_sec;
    case kDOW:
      return tm_struct.tm_wday;
    case kDOY:
      return tm_struct.tm_yday + 1;
    default:
#ifdef __CUDACC__
      return -1;
#else
      CHECK(false);
#endif
  }
}
