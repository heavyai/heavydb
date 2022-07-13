/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
