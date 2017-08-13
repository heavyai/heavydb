#include "../Shared/funcannotations.h"
#ifndef __CUDACC__
#include <cstdint>
#endif
#include <cmath>
#include <cstdlib>

#define EXTENSION_NOINLINE extern "C" NEVER_INLINE DEVICE

/* Example extension functions:
 *
 * EXTENSION_NOINLINE
 * int32_t diff(const int32_t x, const int32_t y) {
 *   return x - y;
 *
 * Arrays map to a pair of pointer and element count. The pointer type must be
 * consistent with the element type of the array column. Only array of numbers
 * are supported for the moment. For example, an ARRAY_AT function which works
 * for arrays of INTEGER and SMALLINT can be implemented as follows:
 *
 * EXTENSION_NOINLINE
 * int64_t array_at(const int32_t* arr, const size_t elem_count, const size_t idx) {
 *   return idx < elem_count ? arr[idx] : -1;
 * }
 *
 * EXTENSION_NOINLINE
 * int64_t array_at__(const int16_t* arr, const size_t elem_count, const size_t idx) {
 *   return idx < elem_count ? arr[idx] : -1;
 * }
 *
 * Note that the return type cannot be specialized and must be the same across
 * all specializations. We can remove this constraint in the future if necessary.
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
double Atan(const double x) {
  return atan(x);
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
double Log10(const double x) {
  return log10(x);
}

EXTENSION_NOINLINE
double Log10__(const float x) {
  return log10f(x);
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
double round_to_digit(const double x, const int32_t y) {
  double exp = pow(10, y);
  return round(x * exp) / exp;
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
double Truncate(const double x, const int32_t y) {
  double p = pow((double)10L, y);
  int64_t temp = x * p;
  return temp / p;
}

EXTENSION_NOINLINE
float Truncate__(const float x, const int32_t y) {
  float p = powf((float)10L, y);
  int64_t temp = x * p;
  return temp / p;
}

EXTENSION_NOINLINE
int16_t Truncate__1(const int16_t x, const int32_t y) {
  if (y >= 0) {
    return x;
  }
  int32_t p = pow((float)10L, std::abs(y));
  int64_t temp = x / p;
  return temp * p;
}

EXTENSION_NOINLINE
int32_t Truncate__2(const int32_t x, const int32_t y) {
  if (y >= 0) {
    return x;
  }
  int32_t p = pow((float)10L, std::abs(y));
  int64_t temp = x / p;
  return temp * p;
}

EXTENSION_NOINLINE
int64_t Truncate__3(const int64_t x, const int32_t y) {
  if (y >= 0) {
    return x;
  }
  int64_t p = pow((double)10L, std::abs(y));
  int64_t temp = x / p;
  return temp * p;
}

EXTENSION_NOINLINE
double conv_4326_900913_x(const double x) {
  return x * 111319.490778;
}

EXTENSION_NOINLINE
double conv_4326_900913_y(const double y) {
  return 6378136.99911 * log(tan(.00872664626 * y + .785398163397));
}

/** @brief  Computes the distance, in meters, between two WGS-84 positions.
  *
  * The result is equal to <code>EARTH_RADIUS_IN_METERS*ArcInRadians(from,to)</code>
  *
  * ArcInRadians is equal to <code>Distance(from,to)/EARTH_RADIUS_IN_METERS</code>
  *    <code>= 2*asin(sqrt(h(d/EARTH_RADIUS_IN_METERS )))</code>
  *
  * where:<ul>
  *    <li>d is the distance in meters between 'from' and 'to' positions.</li>
  *    <li>h is the haversine function: <code>h(x)=sin²(x/2)</code></li>
  * </ul>
  *
  * code attribution: http://blog.julien.cayzac.name/2008/10/arc-and-distance-between-two-points-on.html
  *
  * The haversine formula gives:
  *    <code>h(d/R) = h(from.lat-to.lat)+h(from.lon-to.lon)+cos(from.lat)*cos(to.lat)</code>
  *
  * @sa http://en.wikipedia.org/wiki/Law_of_haversines
  */
EXTENSION_NOINLINE
double distance_in_meters(const double fromlon, const double fromlat, const double tolon, const double tolat) {
  double latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886;
  double longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886;
  double latitudeH = sin(latitudeArc * 0.5);
  latitudeH *= latitudeH;
  double lontitudeH = sin(longitudeArc * 0.5);
  lontitudeH *= lontitudeH;
  double tmp = cos(fromlat * 0.017453292519943295769236907684886) * cos(tolat * 0.017453292519943295769236907684886);
  return 6372797.560856 * (2.0 * asin(sqrt(latitudeH + tmp * lontitudeH)));
}

EXTENSION_NOINLINE
double distance_in_meters__(const float fromlon, const float fromlat, const float tolon, const float tolat) {
  float latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886;
  float longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886;
  float latitudeH = sinf(latitudeArc * 0.5);
  latitudeH *= latitudeH;
  float lontitudeH = sinf(longitudeArc * 0.5);
  lontitudeH *= lontitudeH;
  float tmp = cosf(fromlat * 0.017453292519943295769236907684886) * cosf(tolat * 0.017453292519943295769236907684886);
  return 6372797.560856 * (2.0 * asinf(sqrtf(latitudeH + tmp * lontitudeH)));
}

EXTENSION_NOINLINE
double approx_distance_in_meters(const float fromlon, const float fromlat, const float tolon, const float tolat) {
#ifdef __CUDACC__
  float latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886;
  float longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886;
  float latitudeH = __sinf(latitudeArc * 0.5);
  latitudeH *= latitudeH;
  float lontitudeH = __sinf(longitudeArc * 0.5);
  lontitudeH *= lontitudeH;
  float tmp =
      __cosf(fromlat * 0.017453292519943295769236907684886) * __cosf(tolat * 0.017453292519943295769236907684886);
  return 6372797.560856 * (2.0 * asinf(__fsqrt_rd(latitudeH + tmp * lontitudeH)));
#else
  return distance_in_meters__(fromlon, fromlat, tolon, tolat);
#endif
}
