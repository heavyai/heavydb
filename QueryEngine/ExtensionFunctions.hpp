#include "../Shared/funcannotations.h"
#ifndef __CUDACC__
#include <cstdint>
#endif
#include <cmath>
#include <cstdlib>

#define EXTENSION_INLINE extern "C" ALWAYS_INLINE DEVICE
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
double Round(const double x, const int32_t y) {
  if (y == 0) {
    return round(x) + 0.0;
  }

  double exp = pow(10, y);
#if defined(__powerpc__) && !defined(__CUDACC__)
  int32_t yy = y - 1;
  exp = 10 * powf((float)10L, yy);
#endif
  return (round(x * exp) / exp) + 0.0;
}

EXTENSION_NOINLINE
float Round__(const float x, const int32_t y) {
  if (y == 0) {
    return roundf(x) + 0.0f;
  }

  float exp = powf((float)10L, y);
#if defined(__powerpc__) && !defined(__CUDACC__)
  int32_t yy = y - 1;
  exp = 10 * powf((float)10L, yy);
#endif
  return roundf(x * exp) / exp + 0.0f;
}

EXTENSION_NOINLINE
int16_t Round__1(const int16_t x, const int32_t y) {
  if (y >= 0) {
    return x;
  }

  int32_t p = pow((float)10L, std::abs(y));
  int32_t p_half = p >> 1;

  int64_t temp = x;
#if defined(__powerpc__) && !defined(__CUDACC__)
  int16_t xx = x;
  xx += 1;
  temp = xx;
  temp -= 1;
#endif
  temp = temp >= 0 ? temp + p_half : temp - p_half;
  temp = temp / p;
  return temp * p;
}

EXTENSION_NOINLINE
int32_t Round__2(const int32_t x, const int32_t y) {
  if (y >= 0) {
    return x;
  }

  int32_t p = pow((float)10L, std::abs(y));
  int32_t p_half = p >> 1;

  int64_t temp = x;
#if defined(__powerpc__) && !defined(__CUDACC__)
  int32_t xx = x;
  xx += 1;
  temp = xx;
  temp -= 1;
#endif
  temp = temp >= 0 ? temp + p_half : temp - p_half;
  temp = temp / p;
  return temp * p;
}

EXTENSION_NOINLINE
int64_t Round__3(const int64_t x, const int32_t y) {
  if (y >= 0) {
    return x;
  }

  int64_t p = pow((double)10L, std::abs(y));
  int64_t p_half = p >> 1;

  int64_t temp = x;
  temp = temp >= 0 ? temp + p_half : temp - p_half;
  temp = temp / p;
  return temp * p;
}

EXTENSION_NOINLINE
int64_t Round__4(const int64_t x, const int32_t y0, const int32_t scale) {
  int32_t y = y0 - scale;

  if (y >= 0) {
    return x;
  }

  int64_t p = pow((double)10L, std::abs(y));
  int64_t p_half = p >> 1;

  int64_t temp = x;
  temp = temp >= 0 ? temp + p_half : temp - p_half;
  temp = temp / p;
  return temp * p;
}

EXTENSION_NOINLINE
double Round2_to_digit(const double x, const int32_t y) {
  double exp = pow(10, y);
  return round(x * exp) / exp;
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
#if defined(__powerpc__) && !defined(__CUDACC__)
  int16_t xx = x;
  xx += 1;
  temp = xx;
  temp -= 1;
  temp /= p;
#endif
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
 * code attribution:
 * http://blog.julien.cayzac.name/2008/10/arc-and-distance-between-two-points-on.html
 *
 * The haversine formula gives:
 *    <code>h(d/R) =
 * h(from.lat-to.lat)+h(from.lon-to.lon)+cos(from.lat)*cos(to.lat)</code>
 *
 * @sa http://en.wikipedia.org/wiki/Law_of_haversines
 */
EXTENSION_NOINLINE
double distance_in_meters(const double fromlon,
                          const double fromlat,
                          const double tolon,
                          const double tolat) {
  double latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886;
  double longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886;
  double latitudeH = sin(latitudeArc * 0.5);
  latitudeH *= latitudeH;
  double lontitudeH = sin(longitudeArc * 0.5);
  lontitudeH *= lontitudeH;
  double tmp = cos(fromlat * 0.017453292519943295769236907684886) *
               cos(tolat * 0.017453292519943295769236907684886);
  return 6372797.560856 * (2.0 * asin(sqrt(latitudeH + tmp * lontitudeH)));
}

EXTENSION_NOINLINE
double distance_in_meters__(const float fromlon,
                            const float fromlat,
                            const float tolon,
                            const float tolat) {
  float latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886;
  float longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886;
  float latitudeH = sinf(latitudeArc * 0.5);
  latitudeH *= latitudeH;
  float lontitudeH = sinf(longitudeArc * 0.5);
  lontitudeH *= lontitudeH;
  float tmp = cosf(fromlat * 0.017453292519943295769236907684886) *
              cosf(tolat * 0.017453292519943295769236907684886);
  return 6372797.560856 * (2.0 * asinf(sqrtf(latitudeH + tmp * lontitudeH)));
}

EXTENSION_NOINLINE
double approx_distance_in_meters(const float fromlon,
                                 const float fromlat,
                                 const float tolon,
                                 const float tolat) {
#ifdef __CUDACC__
  float latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886;
  float longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886;
  float latitudeH = __sinf(latitudeArc * 0.5);
  latitudeH *= latitudeH;
  float lontitudeH = __sinf(longitudeArc * 0.5);
  lontitudeH *= lontitudeH;
  float tmp = __cosf(fromlat * 0.017453292519943295769236907684886) *
              __cosf(tolat * 0.017453292519943295769236907684886);
  return 6372797.560856 * (2.0 * asinf(__fsqrt_rd(latitudeH + tmp * lontitudeH)));
#else
  return distance_in_meters__(fromlon, fromlat, tolon, tolat);
#endif
}

EXTENSION_NOINLINE
float rect_pixel_bin(const double val,
                     const double min,
                     const double max,
                     const int32_t numbins,
                     const int32_t dimensionsize) {
  /** deprecated **/
  float numbinsf = float(numbins);
  return float(int32_t(float((val - min) / (max - min)) * numbinsf)) *
         float(dimensionsize) / numbinsf;
}

EXTENSION_NOINLINE
float rect_pixel_bin_x(const double valx,
                       const double minx,
                       const double maxx,
                       const double rectwidth,
                       const double offsetx,
                       const int32_t imgwidth) {
  const float imgwidthf = float(imgwidth);
  const float rectwidthf = float(rectwidth);
  double min = minx;
  float offset = offsetx;
  if (offset != 0) {
    offset = fmodf(offset, rectwidthf);
    if (offset > 0) {
      offset -= rectwidthf;
    }
    min += offset * (maxx - minx) / imgwidthf;
  }
  return float(int32_t(float((valx - min) / (maxx - min)) * (imgwidthf - offset) /
                       rectwidthf)) *
             rectwidthf +
         offset + rectwidthf / 2.0f;
}

EXTENSION_NOINLINE
float rect_pixel_bin_y(const double valy,
                       const double miny,
                       const double maxy,
                       const double rectheight,
                       const double offsety,
                       const int32_t imgheight) {
  const float imgheightf = float(imgheight);
  const float rectheightf = rectheight;
  double min = miny;
  float offset = offsety;
  if (offset != 0) {
    offset = fmodf(offset, rectheightf);
    if (offset > 0) {
      offset -= rectheightf;
    }
    min += offset * (maxy - miny) / imgheightf;
  }
  return float(int32_t(float((valy - min) / (maxy - min)) * (imgheightf - offset) /
                       rectheightf)) *
             rectheightf +
         offset + rectheightf / 2.0f;
}

EXTENSION_NOINLINE
float reg_hex_horiz_pixel_bin_x(const double valx,
                                const double minx,
                                const double maxx,
                                const double valy,
                                const double miny,
                                const double maxy,
                                const double hexwidth,
                                const double hexheight,
                                const double offsetx,
                                const double offsety,
                                const int32_t imgwidth,
                                const int32_t imgheight) {
  const float sqrt3 = 1.7320508075688772;
  const float imgwidthf = float(imgwidth);
  const float imgheightf = float(imgheight);
  const float hexwidthf = float(hexwidth);
  const float hexheightf = float(hexheight);

  // expand the bounds of the data according
  // to the input offsets. This is done because
  // we also expand the image size according to the
  // offsets because this algorithm layers the hexagon
  // bins starting at the bottom left corner
  double xmin = minx;
  float xoffset = offsetx;
  if (xoffset != 0) {
    xoffset = fmodf(xoffset, hexwidthf);
    if (xoffset > 0) {
      xoffset -= hexwidthf;
    }
    xmin += xoffset * (maxx - xmin) / imgwidthf;
  }

  double ymin = miny;
  float yoffset = offsety;
  if (yoffset != 0) {
    yoffset = fmodf(yoffset, 1.5f * hexheightf);
    if (yoffset > 0) {
      yoffset -= 1.5f * hexheightf;
    }
    ymin += yoffset * (maxy - ymin) / imgheightf;
  }

  // get the pixel position of the point
  // assumes a linear scale here
  // Rounds to the nearest pixel.
  const float pix_x =
      roundf((imgwidthf - xoffset) * float((valx - xmin) / (maxx - xmin)));
  const float pix_y =
      roundf((imgheightf - yoffset) * float((valy - ymin) / (maxy - ymin)));

  // Now convert the pixel position into a
  // cube-coordinate system representation
  const float hexsize = hexheightf / 2.0f;
  const float cube_x = ((pix_x / sqrt3) - (pix_y / 3.0)) / hexsize;
  const float cube_z = (pix_y * 2.0f / 3.0f) / hexsize;
  const float cube_y = -cube_x - cube_z;

  // need to round the cube coordinates above
  float rx = round(cube_x);
  float ry = round(cube_y);
  float rz = round(cube_z);
  const float x_diff = fabs(rx - cube_x);
  const float y_diff = fabs(ry - cube_y);
  const float z_diff = fabs(rz - cube_z);
  if (x_diff > y_diff && x_diff > z_diff) {
    rx = -ry - rz;
  } else if (y_diff > z_diff) {
    ry = -rx - rz;
  } else {
    rz = -rx - ry;
  }

  // now convert the cube/hex coord to a pixel location
  return hexsize * sqrt3 * (rx + rz / 2.0f) + xoffset;
}

EXTENSION_NOINLINE
float reg_hex_horiz_pixel_bin_y(const double valx,
                                const double minx,
                                const double maxx,
                                const double valy,
                                const double miny,
                                const double maxy,
                                const double hexwidth,
                                const double hexheight,
                                const double offsetx,
                                const double offsety,
                                const int32_t imgwidth,
                                const int32_t imgheight) {
  const float sqrt3 = 1.7320508075688772;
  const float imgwidthf = float(imgwidth);
  const float imgheightf = float(imgheight);
  const float hexwidthf = float(hexwidth);
  const float hexheightf = float(hexheight);

  // expand the bounds of the data according
  // to the input offsets. This is done because
  // we also expand the image size according to the
  // offsets because this algorithm layers the hexagon
  // bins starting at the bottom left corner
  double xmin = minx;
  float xoffset = offsetx;
  if (xoffset != 0) {
    xoffset = fmodf(xoffset, hexwidthf);
    if (xoffset > 0) {
      xoffset -= hexwidthf;
    }
    xmin += xoffset * (maxx - xmin) / imgwidthf;
  }

  double ymin = miny;
  float yoffset = offsety;
  if (yoffset != 0) {
    yoffset = fmodf(yoffset, 1.5f * hexheightf);
    if (yoffset > 0) {
      yoffset -= 1.5f * hexheightf;
    }
    ymin += yoffset * (maxy - ymin) / imgheightf;
  }

  // get the pixel position of the point
  // assumes a linear scale here
  // Rounds to the nearest pixel.
  const float pix_x =
      roundf((imgwidthf - xoffset) * float((valx - xmin) / (maxx - xmin)));
  const float pix_y =
      roundf((imgheightf - yoffset) * float((valy - ymin) / (maxy - ymin)));

  // Now convert the pixel position into a
  // cube-coordinate system representation
  const float hexsize = hexheightf / 2.0f;
  const float cube_x = ((pix_x / sqrt3) - (pix_y / 3.0f)) / hexsize;
  const float cube_z = (pix_y * 2.0f / 3.0f) / hexsize;
  const float cube_y = -cube_x - cube_z;

  // need to round the cube coordinates above
  float rx = round(cube_x);
  float ry = round(cube_y);
  float rz = round(cube_z);
  const float x_diff = fabs(rx - cube_x);
  const float y_diff = fabs(ry - cube_y);
  const float z_diff = fabs(rz - cube_z);
  if ((x_diff <= y_diff || x_diff <= z_diff) && y_diff <= z_diff) {
    rz = -rx - ry;
  }

  // now convert the cube/hex coord to a pixel location
  return hexsize * 3.0f / 2.0f * rz + yoffset;
}

EXTENSION_NOINLINE
float reg_hex_vert_pixel_bin_x(const double valx,
                               const double minx,
                               const double maxx,
                               const double valy,
                               const double miny,
                               const double maxy,
                               const double hexwidth,
                               const double hexheight,
                               const double offsetx,
                               const double offsety,
                               const int32_t imgwidth,
                               const int32_t imgheight) {
  const float sqrt3 = 1.7320508075688772;
  const float imgwidthf = float(imgwidth);
  const float imgheightf = float(imgheight);
  const float hexwidthf = float(hexwidth);
  const float hexheightf = float(hexheight);

  // expand the bounds of the data according
  // to the input offsets. This is done because
  // we also expand the image size according to the
  // offsets because this algorithm layers the hexagon
  // bins starting at the bottom left corner
  double xmin = minx;
  float xoffset = offsetx;
  if (xoffset != 0) {
    xoffset = fmodf(xoffset, 1.5f * hexwidthf);
    if (xoffset > 0) {
      xoffset -= 1.5f * hexwidthf;
    }
    xmin += xoffset * (maxx - xmin) / imgwidthf;
  }

  double ymin = miny;
  float yoffset = offsety;
  if (yoffset != 0) {
    yoffset = fmodf(yoffset, hexheightf);
    if (yoffset > 0) {
      yoffset -= hexheightf;
    }
    ymin += yoffset * (maxy - ymin) / imgheightf;
  }

  // get the pixel position of the point
  // assumes a linear scale here
  // Rounds to the nearest pixel.
  const float pix_x =
      roundf((imgwidthf - xoffset) * float((valx - xmin) / (maxx - xmin)));
  const float pix_y =
      roundf((imgheightf - yoffset) * float((valy - ymin) / (maxy - ymin)));

  // Now convert the pixel position into a
  // cube-coordinate system representation
  const float hexsize = hexwidthf / 2.0f;
  const float cube_x = (pix_x * 2.0f / 3.0f) / hexsize;
  const float cube_z = ((pix_y / sqrt3) - (pix_x / 3.0f)) / hexsize;
  const float cube_y = -cube_x - cube_z;

  // need to round the cube coordinates above
  float rx = round(cube_x);
  float ry = round(cube_y);
  float rz = round(cube_z);
  const float x_diff = fabs(rx - cube_x);
  const float y_diff = fabs(ry - cube_y);
  const float z_diff = fabs(rz - cube_z);
  if (x_diff > y_diff && x_diff > z_diff) {
    rx = -ry - rz;
  }

  // now convert the cube/hex coord to a pixel location
  return hexsize * 3.0f / 2.0f * rx + xoffset;
}

EXTENSION_NOINLINE
float reg_hex_vert_pixel_bin_y(const double valx,
                               const double minx,
                               const double maxx,
                               const double valy,
                               const double miny,
                               const double maxy,
                               const double hexwidth,
                               const double hexheight,
                               const double offsetx,
                               const double offsety,
                               const int32_t imgwidth,
                               const int32_t imgheight) {
  const float sqrt3 = 1.7320508075688772;
  const float imgwidthf = float(imgwidth);
  const float imgheightf = float(imgheight);
  const float hexwidthf = float(hexwidth);
  const float hexheightf = float(hexheight);

  // expand the bounds of the data according
  // to the input offsets. This is done because
  // we also expand the image size according to the
  // offsets because this algorithm layers the hexagon
  // bins starting at the bottom left corner
  float xmin = minx;
  float xoffset = offsetx;
  if (xoffset != 0) {
    xoffset = fmodf(xoffset, 1.5f * hexwidthf);
    if (xoffset > 0) {
      xoffset -= 1.5f * hexwidthf;
    }
    xmin += xoffset * (maxx - xmin) / imgwidthf;
  }

  float ymin = miny;
  float yoffset = offsety;
  if (yoffset != 0) {
    yoffset = fmodf(yoffset, hexheightf);
    if (yoffset > 0) {
      yoffset -= hexheightf;
    }
    ymin += yoffset * (maxy - ymin) / imgheightf;
  }

  // get the pixel position of the point
  // assumes a linear scale here
  // Rounds to the nearest pixel.
  const float pix_x = roundf((imgwidthf - xoffset) * (valx - xmin) / (maxx - xmin));
  const float pix_y = roundf((imgheightf - yoffset) * (valy - ymin) / (maxy - ymin));

  // Now convert the pixel position into a
  // cube-coordinate system representation
  const float hexsize = hexwidthf / 2.0f;
  const float cube_x = (pix_x * 2.0f / 3.0f) / hexsize;
  const float cube_z = ((pix_y / sqrt3) - (pix_x / 3.0f)) / hexsize;
  const float cube_y = -cube_x - cube_z;

  // need to round the cube coordinates above
  float rx = round(cube_x);
  float ry = round(cube_y);
  float rz = round(cube_z);
  const float x_diff = fabs(rx - cube_x);
  const float y_diff = fabs(ry - cube_y);
  const float z_diff = fabs(rz - cube_z);
  if (x_diff > y_diff && x_diff > z_diff) {
    rx = -ry - rz;
  } else if (y_diff > z_diff) {
    ry = -rx - rz;
  } else {
    rz = -rx - ry;
  }

  // now convert the cube/hex coord to a pixel location
  return hexsize * sqrt3 * (rz + rx / 2.0f) + yoffset;
}

EXTENSION_NOINLINE
double convert_meters_to_merc_pixel_width(const double meters,
                                          const double lon,
                                          const double lat,
                                          const double min_lon,
                                          const double max_lon,
                                          const int32_t img_width,
                                          const double min_width) {
  const double const1 = 0.017453292519943295769236907684886;
  const double const2 = 6372797.560856;
  double t1 = sinf(meters / (2.0 * const2));
  double t2 = cosf(const1 * lat);
  const double newlon = lon - (2.0 * asinf(t1 / t2)) / const1;
  t1 = conv_4326_900913_x(lon);
  t2 = conv_4326_900913_x(newlon);
  const double min_merc_x = conv_4326_900913_x(min_lon);
  const double max_merc_x = conv_4326_900913_x(max_lon);
  const double merc_diff = max_merc_x - min_merc_x;
  t1 = ((t1 - min_merc_x) / merc_diff) * static_cast<double>(img_width);
  t2 = ((t2 - min_merc_x) / merc_diff) * static_cast<double>(img_width);

  // TODO(croot): need to account for edge cases, such as getting close to the poles.
  const double sz = fabs(t1 - t2);
  return (sz < min_width ? min_width : sz);
}

EXTENSION_NOINLINE
double convert_meters_to_merc_pixel_height(const double meters,
                                           const double lon,
                                           const double lat,
                                           const double min_lat,
                                           const double max_lat,
                                           const int32_t img_height,
                                           const double min_height) {
  const double const1 = 0.017453292519943295769236907684886;
  const double const2 = 6372797.560856;
  const double latdiff = meters / (const1 * const2);
  const double newlat =
      (lat < 0) ? lat + latdiff : lat - latdiff;  // assumes a lat range of [-90, 90]
  double t1 = conv_4326_900913_y(lat);
  double t2 = conv_4326_900913_y(newlat);
  const double min_merc_y = conv_4326_900913_y(min_lat);
  const double max_merc_y = conv_4326_900913_y(max_lat);
  const double merc_diff = max_merc_y - min_merc_y;
  t1 = ((t1 - min_merc_y) / merc_diff) * static_cast<double>(img_height);
  t2 = ((t2 - min_merc_y) / merc_diff) * static_cast<double>(img_height);

  // TODO(croot): need to account for edge cases, such as getting close to the poles.
  const double sz = fabs(t1 - t2);
  return (sz < min_height ? min_height : sz);
}

EXTENSION_INLINE bool is_point_in_merc_view(const double lon,
                                            const double lat,
                                            const double min_lon,
                                            const double max_lon,
                                            const double min_lat,
                                            const double max_lat) {
  return !(lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat);
}

EXTENSION_NOINLINE bool is_point_size_in_merc_view(const double lon,
                                                   const double lat,
                                                   const double meters,
                                                   const double min_lon,
                                                   const double max_lon,
                                                   const double min_lat,
                                                   const double max_lat) {
  const double const1 = 0.017453292519943295769236907684886;
  const double const2 = 6372797.560856;
  const double latdiff = meters / (const1 * const2);
  const double t1 = sinf(meters / (2.0 * const2));
  const double t2 = cosf(const1 * lat);
  const double londiff = (2.0 * asinf(t1 / t2)) / const1;
  return !(lon + londiff < min_lon || lon - londiff > max_lon ||
           lat + latdiff < min_lat || lat - latdiff > max_lat);
}

#include "ExtensionFunctionsGeo.hpp"
