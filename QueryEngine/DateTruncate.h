#ifndef QUERYENGINE_DATETRUNCATE_H
#define QUERYENGINE_DATETRUNCATE_H

#include <stdint.h>
#include <time.h>

#include "ExtractFromTime.h"

/*
 * year
 * month
 * day
 * hour
 * minute
 * second
 *
 * millennium
 * century
 * decade
 * milliseconds
 * microseconds
 * week
 * quarterday
 */
enum DatetruncField {
  dtYEAR,
  dtQUARTER,
  dtMONTH,
  dtDAY,
  dtHOUR,
  dtMINUTE,
  dtSECOND,
  dtMILLENNIUM,
  dtCENTURY,
  dtDECADE,
  dtMILLISECOND,
  dtMICROSECOND,
  dtWEEK,
  dtQUARTERDAY,
  dtINVALID
};

extern "C" __attribute__((noinline))
#ifdef __CUDACC__
__device__
#endif
    time_t
    DateTruncate(DatetruncField field, time_t timeval);

#endif  // QUERYENGINE_DATETRUNCATE_H
