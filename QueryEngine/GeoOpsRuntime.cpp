/*
 * Copyright 2021 OmniSci, Inc.
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

#include <cmath>

extern "C" ALWAYS_INLINE double transform_4326_900913_x(const double x) {
  return x * 111319.490778;
}

extern "C" ALWAYS_INLINE double transform_4326_900913_y(const double y) {
  return 6378136.99911 * log(tan(.00872664626 * y + .785398163397));
}

extern "C" ALWAYS_INLINE double transform_900913_4326_x(const double x) {
  constexpr double e_circ = 40075016.68;  // Earth's circumference, meters
  constexpr double e_circ_360 = e_circ / 360.;
  constexpr double e_cir_360_inverse = 1. / e_circ_360;
  return x * e_cir_360_inverse;
}

extern "C" ALWAYS_INLINE double transform_900913_4326_y(const double y) {
  constexpr double e_circ = 40075016.68;        // Earth's circumference, meters
  constexpr double e_circ_360 = e_circ / 360.;  // 111319.491
  const double e = exp(1.0);
  const double pi = acos(-1);
  return atan(pow(e, (y / e_circ_360) * pi / 180.)) * 360. / pi - 90.;
}
