#ifndef QUERYENGINE_EXTRACTFROMTIME_H
#define QUERYENGINE_EXTRACTFROMTIME_H

#include <stdint.h>
#include <time.h>

enum ExtractField { kYEAR, kMONTH, kDAY, kHOUR, kMINUTE, kSECOND, kDOW, kISODOW, kDOY, kEPOCH };

extern "C" __attribute__((noinline))
#ifdef __CUDACC__
__device__
#endif
    int64_t ExtractFromTime(ExtractField field, time_t timeval);

#endif  // QUERYENGINE_EXTRACTFROMTIME_H
