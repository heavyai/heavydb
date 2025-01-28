/*
 * Copyright 2025 HEAVY.AI, Inc.
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

#include <cstdint>
#include <string>

namespace Geospatial {

int64_t H3_LonLatToCell(const double lon, const double lat, const int32_t resolution);

std::pair<double, double> H3_CellToLonLat(const int64_t cell);

std::string H3_CellToString(const int64_t cell);

int64_t H3_StringToCell(const std::string& str);

int64_t H3_CellToParent(const int64_t cell, const int32_t resolution);

bool H3_IsValidCell(const int64_t cell);

std::string H3_CellToBoundary_WKT(const int64_t cell);

}  // namespace Geospatial
