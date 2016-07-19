#include "../Shared/funcannotations.h"
#ifndef __CUDACC__
#include <cstdint>
#endif
#include <math.h>

#define EXTENSION_NOINLINE extern "C" NEVER_INLINE DEVICE

/* Example extension function:
 *
 * EXTENSION_NOINLINE
 * int32_t diff(const int32_t x, const int32_t y) {
 *   return x - y;
 * }
 */

EXTENSION_NOINLINE
double Acos(const double x) {
  return acos(x);
}

EXTENSION_NOINLINE
double Asin(const double x) {
  return asin(x);
}

EXTENSION_NOINLINE
double Atan2(const double y, const double x) {
  return atan2(y, x);
}

EXTENSION_NOINLINE
double Cos(const double x) {
  return cos(x);
}

EXTENSION_NOINLINE
double Cot(const double x) {
  return 1 / tan(x);
}

EXTENSION_NOINLINE
double degrees(double x) {
  return x * (180.0 / M_PI);
}

EXTENSION_NOINLINE
double Exp(double x) {
  return exp(x);
}

EXTENSION_NOINLINE
double ln(const double x) {
  return log(x);
}

EXTENSION_NOINLINE
double Log(const double x) {
  return log(x);
}

EXTENSION_NOINLINE
double pi() {
  return M_PI;
}

EXTENSION_NOINLINE
double power(const double x, const double y) {
  return pow(x, y);
}

EXTENSION_NOINLINE
double radians(const double x) {
  return x * (M_PI / 180.0);
}

EXTENSION_NOINLINE
double Round(const double x) {
  return round(x);
}

EXTENSION_NOINLINE
double Sin(const double x) {
  return sin(x);
}

EXTENSION_NOINLINE
double Tan(const double x) {
  return tan(x);
}
