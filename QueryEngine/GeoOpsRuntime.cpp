/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Geospatial/Utm.h"

extern "C" ALWAYS_INLINE double transform_4326_900913_x(const double x, double) {
  constexpr double a = 6378137;  // WGS84 Equatorial radius (m)
  constexpr double c = a * math_consts::rad_div_deg;
  return c * x;
}

// Required: -90 < y < 90 otherwise log() may be calculated for a negative number.
extern "C" ALWAYS_INLINE double transform_4326_900913_y(double, const double y) {
  constexpr double a = 6378137;  // WGS84 Equatorial radius (m)
  constexpr double rad_div_two_deg = math_consts::m_pi / (2 * 180);
  constexpr double pi_div_four = math_consts::m_pi / 4;
  return a * log(tan(rad_div_two_deg * y + pi_div_four));
}

extern "C" ALWAYS_INLINE double transform_900913_4326_x(const double x, double) {
  constexpr double a = 6378137;  // WGS84 Equatorial radius (m)
  constexpr double c = math_consts::deg_div_rad / a;
  return c * x;
}

extern "C" ALWAYS_INLINE double transform_900913_4326_y(double, const double y) {
  constexpr double a = 6378137;  // WGS84 Equatorial radius (m)
  constexpr double a_inv = 1 / a;
  constexpr double two_deg_div_rad = 2 * 180 / math_consts::m_pi;
  return two_deg_div_rad * atan(exp(a_inv * y)) - 90;
}

extern "C" ALWAYS_INLINE double transform_4326_utm_x(unsigned const utm_srid,
                                                     double const x,
                                                     double const y) {
  return Transform4326ToUTM(utm_srid, x, y).calculateX();
}

extern "C" ALWAYS_INLINE double transform_4326_utm_y(unsigned const utm_srid,
                                                     double const x,
                                                     double const y) {
  return Transform4326ToUTM(utm_srid, x, y).calculateY();
}

extern "C" ALWAYS_INLINE double transform_utm_4326_x(unsigned const utm_srid,
                                                     double const x,
                                                     double const y) {
  return TransformUTMTo4326(utm_srid, x, y).calculateX();
}

extern "C" ALWAYS_INLINE double transform_utm_4326_y(unsigned const utm_srid,
                                                     double const x,
                                                     double const y) {
  return TransformUTMTo4326(utm_srid, x, y).calculateY();
}

extern "C" ALWAYS_INLINE double transform_900913_utm_x(unsigned const utm_srid,
                                                       double const x,
                                                       double const y) {
  return transform_4326_utm_x(
      utm_srid, transform_900913_4326_x(x, {}), transform_900913_4326_y({}, y));
}

extern "C" ALWAYS_INLINE double transform_900913_utm_y(unsigned const utm_srid,
                                                       double const x,
                                                       double const y) {
  return transform_4326_utm_y(
      utm_srid, transform_900913_4326_x(x, {}), transform_900913_4326_y({}, y));
}

extern "C" ALWAYS_INLINE double transform_utm_900913_x(unsigned const utm_srid,
                                                       double const x,
                                                       double const y) {
  return transform_4326_900913_x(transform_utm_4326_x(utm_srid, x, y), {});
}

extern "C" ALWAYS_INLINE double transform_utm_900913_y(unsigned const utm_srid,
                                                       double const x,
                                                       double const y) {
  return transform_4326_900913_y({}, transform_utm_4326_y(utm_srid, x, y));
}
