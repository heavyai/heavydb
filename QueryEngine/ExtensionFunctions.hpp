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
double Ceil(double x) {
  return ceil(x);
}

EXTENSION_NOINLINE
float Ceil__(float x) {
  return ceil(x);
}

EXTENSION_NOINLINE
int16_t Ceil__1(int16_t x) {
  return x;
}

EXTENSION_NOINLINE
int32_t Ceil__2(int32_t x) {
  return x;
}

EXTENSION_NOINLINE
int64_t Ceil__3(int64_t x) {
  return x;
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
double Floor(double x) {
  return floor(x);
}

EXTENSION_NOINLINE
float Floor__(float x) {
  return floor(x);
}

EXTENSION_NOINLINE
int16_t Floor__1(int16_t x) {
  return x;
}

EXTENSION_NOINLINE
int32_t Floor__2(int32_t x) {
  return x;
}

EXTENSION_NOINLINE
int64_t Floor__3(int64_t x) {
  return x;
}

EXTENSION_NOINLINE
double ln(const double x) {
  return log(x);
}

EXTENSION_NOINLINE
double ln__(const float x) {
  return logf(x);
}

EXTENSION_NOINLINE
double Log(const double x) {
  return log(x);
}

EXTENSION_NOINLINE
double Log__(const float x) {
  return logf(x);
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

EXTENSION_NOINLINE
double Tan__(const float x) {
  return tanf(x);
}

EXTENSION_NOINLINE
double Truncate(const double x, double y) {
  double p = pow((double)10L, y);
  int64_t temp = x * p;
  return temp / p;
}

EXTENSION_NOINLINE
double conv_4326_900913_x(const double x) {
  return x * 111319.490778;
}

EXTENSION_NOINLINE
double conv_4326_900913_x__(const float x) {
  return x * 111319.490778;
}

EXTENSION_NOINLINE
double conv_4326_900913_y(const double y) {
  return 6378136.99911 * log(tan(.00872664626 * y + .785398163397));
}

EXTENSION_NOINLINE
double conv_4326_900913_y__(const float y) {
  return 6378136.99911 * logf(tanf(.00872664626 * y + .785398163397));
}
