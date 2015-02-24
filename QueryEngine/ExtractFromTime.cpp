#include "ExtractFromTime.h"

#ifndef __CUDACC__
#include <assert.h>
#endif

/*
 * @brief support the SQL EXTRACT function
 */
extern "C" __attribute__((noinline))
#ifdef __CUDACC__
__device__
#endif
int64_t ExtractFromTime(ExtractField field, time_t timeval) {
  int64_t result;
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
      result = 1900 + tm_struct.tm_year;
      break;
    case kMONTH:
      result = tm_struct.tm_mon + 1;
      break;
    case kDAY:
      result = tm_struct.tm_mday;
      break;
    case kHOUR:
      result = tm_struct.tm_hour;
      break;
    case kMINUTE:
      result = tm_struct.tm_min;
      break;
    case kSECOND:
      result = tm_struct.tm_sec;
      break;
    case kDOW:
      result = tm_struct.tm_wday;
      break;
    case kDOY:
      result = tm_struct.tm_yday + 1;
      break;
    default:
#ifdef __CUDACC__
      return -1;
#else
      assert(false);
#endif
  }
  return result;
}
