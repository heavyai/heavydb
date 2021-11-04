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
#pragma once

namespace math_consts {
constexpr double m_pi{3.14159265358979323846};
constexpr double rad_div_deg{m_pi / 180};
constexpr double deg_div_rad{180 / m_pi};
constexpr double radians_to_degrees{deg_div_rad};
}  // end namespace math_consts
