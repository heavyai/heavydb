/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <string>

inline std::pair<double, double> geotransform_4326_to_900913(const double lon,
                                                             const double lat) {
  static const double e_circ = 40075016.68;  // Earth's circumference, meters
  static const double e_circ_360 = e_circ / 360;
  static const double pi = std::acos(-1);

  std::pair<double, double> ll;
  ll.first = lon * e_circ_360;
  ll.second = e_circ_360 * std::log(std::tan((90 + lat) * pi / 360)) / (pi / 180);
  return ll;
}

inline std::pair<double, double> geotransform(const std::string& src_proj,
                                              const std::string& dst_proj,
                                              const double x,
                                              const double y) {
  return geotransform_4326_to_900913(x, y);
}
