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

EXTENSION_NOINLINE
float rect_pixel_bin(const float val,
                     const float min,
                     const float max,
                     const int32_t numbins,
                     const int32_t dimensionsize) {
  /** deprecated **/
  float numbinsf = float(numbins);
  return float(int32_t((val - min) / (max - min) * numbinsf)) * float(dimensionsize) / numbinsf;
}

EXTENSION_NOINLINE
float rect_pixel_bin_x(const float valx,
                       const float minx,
                       const float maxx,
                       const float rectwidth,
                       const float offsetx,
                       const int32_t imgwidth) {
  const float imgwidthf = float(imgwidth);
  float min = minx;
  float offset = offsetx;
  if (offset != 0) {
    offset = fmodf(offset, rectwidth);
    if (offset > 0) {
      offset -= rectwidth;
    }
    min += offset * (maxx - minx) / imgwidthf;
  }
  return float(int32_t((valx - min) / (maxx - min) * (imgwidthf - offset) / rectwidth)) * rectwidth + offset +
         rectwidth / 2.0f;
}

EXTENSION_NOINLINE
float rect_pixel_bin_y(const float valy,
                       const float miny,
                       const float maxy,
                       const float rectheight,
                       const float offsety,
                       const int32_t imgheight) {
  const float imgheightf = float(imgheight);
  float min = miny;
  float offset = offsety;
  if (offset != 0) {
    offset = fmodf(offset, rectheight);
    if (offset > 0) {
      offset -= rectheight;
    }
    min += offset * (maxy - miny) / imgheightf;
  }
  return float(int32_t((valy - min) / (maxy - min) * (imgheightf - offset) / rectheight)) * rectheight + offset +
         rectheight / 2.0f;
}

EXTENSION_NOINLINE
float reg_hex_horiz_pixel_bin_x(const float valx,
                                const float minx,
                                const float maxx,
                                const float valy,
                                const float miny,
                                const float maxy,
                                const float hexwidth,
                                const float hexheight,
                                const float offsetx,
                                const float offsety,
                                const int32_t imgwidth,
                                const int32_t imgheight) {
  const float sqrt3 = 1.7320508075688772;
  const float imgwidthf = float(imgwidth);
  const float imgheightf = float(imgheight);

  // expand the bounds of the data according
  // to the input offsets. This is done because
  // we also expand the image size according to the
  // offsets because this algorithm layers the hexagon
  // bins starting at the bottom left corner
  float xmin = minx;
  float xoffset = offsetx;
  if (xoffset != 0) {
    xoffset = fmodf(xoffset, hexwidth);
    if (xoffset > 0) {
      xoffset -= hexwidth;
    }
    xmin += xoffset * (maxx - xmin) / imgwidthf;
  }

  float ymin = miny;
  float yoffset = offsety;
  if (yoffset != 0) {
    yoffset = fmodf(yoffset, 1.5f * hexheight);
    if (yoffset > 0) {
      yoffset -= 1.5f * hexheight;
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
  const float hexsize = hexheight / 2.0f;
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
float reg_hex_horiz_pixel_bin_y(const float valx,
                                const float minx,
                                const float maxx,
                                const float valy,
                                const float miny,
                                const float maxy,
                                const float hexwidth,
                                const float hexheight,
                                const float offsetx,
                                const float offsety,
                                const int32_t imgwidth,
                                const int32_t imgheight) {
  const float sqrt3 = 1.7320508075688772;
  const float imgwidthf = float(imgwidth);
  const float imgheightf = float(imgheight);

  // expand the bounds of the data according
  // to the input offsets. This is done because
  // we also expand the image size according to the
  // offsets because this algorithm layers the hexagon
  // bins starting at the bottom left corner
  float xmin = minx;
  float xoffset = offsetx;
  if (xoffset != 0) {
    xoffset = fmodf(xoffset, hexwidth);
    if (xoffset > 0) {
      xoffset -= hexwidth;
    }
    xmin += xoffset * (maxx - xmin) / imgwidthf;
  }

  float ymin = miny;
  float yoffset = offsety;
  if (yoffset != 0) {
    yoffset = fmodf(yoffset, 1.5f * hexheight);
    if (yoffset > 0) {
      yoffset -= 1.5f * hexheight;
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
  const float hexsize = hexheight / 2.0f;
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
float reg_hex_vert_pixel_bin_x(const float valx,
                               const float minx,
                               const float maxx,
                               const float valy,
                               const float miny,
                               const float maxy,
                               const float hexwidth,
                               const float hexheight,
                               const float offsetx,
                               const float offsety,
                               const int32_t imgwidth,
                               const int32_t imgheight) {
  const float sqrt3 = 1.7320508075688772;
  const float imgwidthf = float(imgwidth);
  const float imgheightf = float(imgheight);

  // expand the bounds of the data according
  // to the input offsets. This is done because
  // we also expand the image size according to the
  // offsets because this algorithm layers the hexagon
  // bins starting at the bottom left corner
  float xmin = minx;
  float xoffset = offsetx;
  if (xoffset != 0) {
    xoffset = fmodf(xoffset, 1.5f * hexwidth);
    if (xoffset > 0) {
      xoffset -= 1.5f * hexwidth;
    }
    xmin += xoffset * (maxx - xmin) / imgwidthf;
  }

  float ymin = miny;
  float yoffset = offsety;
  if (yoffset != 0) {
    yoffset = fmodf(yoffset, hexheight);
    if (yoffset > 0) {
      yoffset -= hexheight;
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
  const float hexsize = hexwidth / 2.0f;
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
float reg_hex_vert_pixel_bin_y(const float valx,
                               const float minx,
                               const float maxx,
                               const float valy,
                               const float miny,
                               const float maxy,
                               const float hexwidth,
                               const float hexheight,
                               const float offsetx,
                               const float offsety,
                               const int32_t imgwidth,
                               const int32_t imgheight) {
  const float sqrt3 = 1.7320508075688772;
  const float imgwidthf = float(imgwidth);
  const float imgheightf = float(imgheight);

  // expand the bounds of the data according
  // to the input offsets. This is done because
  // we also expand the image size according to the
  // offsets because this algorithm layers the hexagon
  // bins starting at the bottom left corner
  float xmin = minx;
  float xoffset = offsetx;
  if (xoffset != 0) {
    xoffset = fmodf(xoffset, 1.5f * hexwidth);
    if (xoffset > 0) {
      xoffset -= 1.5f * hexwidth;
    }
    xmin += xoffset * (maxx - xmin) / imgwidthf;
  }

  float ymin = miny;
  float yoffset = offsety;
  if (yoffset != 0) {
    yoffset = fmodf(yoffset, hexheight);
    if (yoffset > 0) {
      yoffset -= hexheight;
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
  const float hexsize = hexwidth / 2.0f;
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


DEVICE
double hypotenuse(double x, double y)
{
  x = fabs(x);
  y = fabs(y);
  if (x < y) {
    auto t = x; x = y; y = t;
  }
  if (y == 0.0)
    return x;
  return x * sqrt(1.0 + (y * y) / (x * x)); 
}

ALWAYS_INLINE DEVICE
double distance_point_point(double p1x, double p1y, double p2x, double p2y)
{
  return hypotenuse(p1x - p2x, p1y - p2y);
}

DEVICE
double distance_point_line(double px, double py, double *l)
{
  double length = distance_point_point(l[0], l[1], l[2], l[3]);
  if (length == 0.0)
    return distance_point_point(px, py, l[0], l[1]);

  // Find projection of point P onto the line segment AB:
  // Line containing that segment: A + k * (B - A)
  // Projection of point P onto the line touches it at
  //   k = dot(P-A,B-A) / length^2
  // AB segment is represented by k = [0,1]
  // Clamping k to [0,1] will give the shortest distance from P to AB segment
  double dotprod = (px - l[0]) * (l[2] - l[0]) + (py - l[1]) * (l[3] - l[1]);
  double k = dotprod / (length * length);
  k = fmax(0.0, fmin(1.0, k));
  double projx = l[0] + k * (l[2] - l[0]);
  double projy = l[1] + k * (l[3] - l[1]);
  return distance_point_point(px, py, projx, projy);
}

DEVICE
bool intersects_line_line(double *l1, double *l2)
{
  // Check if line segment AB intersects line segment CD
  double ex = l1[2] - l1[0]; // E = B - A
  double ey = l1[3] - l1[1];
  double fx = l2[2] - l2[0]; // F = D - C
  double fy = l2[3] - l2[1];
  double px = -ey; // P = ( -Ey, Ex )
  double py =  ex;
  double fp = fx * px + fy * py; // F * P
  if (fp == 0.0)
    return false; // lines are parallel
  double h = ((l1[0] - l2[0]) * px + (l1[1] - l2[1]) * py) / fp; // h = ( (A-C) * P ) / (F * P)
  return (h >= 0.0 && h <= 1.0);
}

DEVICE
double distance_line_line(double *l1, double *l2)
{
  if (intersects_line_line(l1, l2))
    return 0.0;
  double dist12 = fmin(distance_point_line(l1[0], l1[1], l2),
                       distance_point_line(l1[2], l1[3], l2));
  double dist21 = fmin(distance_point_line(l2[0], l2[1], l1),
                       distance_point_line(l2[2], l2[3], l1));
  return fmin(dist12, dist21);
}

EXTENSION_NOINLINE
double ST_Distance_Point_Point(double *p1, int64_t p1num, double *p2, int64_t p2num)
{
  return distance_point_point(p1[0], p1[1], p2[0], p2[1]);
}

EXTENSION_NOINLINE
double ST_Distance_Point_LineString(double *p, int64_t pnum, double *l, int64_t lnum)
{
  double *line = l;
  int64_t num_lines = lnum/2 - 1;
  double dist = distance_point_line(p[0], p[1], line);
  for (int i = 1; i < num_lines; i++) {
    line += 2; // adance one point
    double ldist = distance_point_line(p[0], p[1], line);
    if (dist > ldist)
      dist = ldist;
  }
  return dist;
}

EXTENSION_INLINE
double ST_Distance_LineString_Point(double *l, int64_t lnum, double *p, int64_t pnum)
{
  return ST_Distance_Point_LineString(p, pnum, l, lnum);
}

EXTENSION_NOINLINE
double ST_Distance_LineString_LineString(double *l1, int64_t l1num, double *l2, int64_t l2num)
{
  double dist = distance_point_point(l1[0], l1[1], l2[0], l2[1]);
  int64_t num_lines1 = l1num/2 - 1;
  int64_t num_lines2 = l2num/2 - 1;
  double *line1 = l1;
  for (int i = 0; i < num_lines1; i++) {
    double *line2 = l2;
    for (int j = 0; j < num_lines2; j++) {
      double ldist = distance_line_line(line1, line2);
      if (dist > ldist)
        dist = ldist;
      line2 += 2; // adance one point
    }
    line1 += 2; // adance one point
  }
  return dist;
}

EXTENSION_NOINLINE
bool ST_Contains_Point_Point(double *p1, int64_t p1num, double *p2, int64_t p2num)
{
  return (p1[0] == p2[0]) && (p1[1] == p2[1]); // TBD: sensitivity
}

EXTENSION_NOINLINE
bool ST_Contains_Point_LineString(double *p, int64_t pnum, double *l, int64_t lnum)
{
  for (int i = 0; i < lnum; i+=2) {
    if (p[0] == l[i] && p[1] == l[i+1])
      continue;
    return false;
  }
  return true;
}

EXTENSION_INLINE
bool ST_Contains_LineString_Point(double *l, int64_t lnum, double *p, int64_t pnum)
{
  return (ST_Distance_Point_LineString(p, pnum, l, lnum) == 0.0);  // TBD: sensitivity
}

EXTENSION_NOINLINE
bool ST_Contains_LineString_LineString(double *l1, int64_t l1num, double *l2, int64_t l2num)
{
  // TBD
  return false;
}
