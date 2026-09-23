/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Geospatial {

int64_t H3_LonLatToCell(const double lon, const double lat, const int32_t resolution);

std::pair<double, double> H3_CellToLonLat(const int64_t cell);

std::string H3_CellToString(const int64_t cell);

int64_t H3_StringToCell(const std::string& str);

int64_t H3_CellToParent(const int64_t cell, const int32_t resolution);

bool H3_IsValidCell(const int64_t cell);

std::vector<double> H3_CellToBoundary_POLYGON(const int64_t cell);

std::string H3_CellToBoundary_WKT(const int64_t cell);

}  // namespace Geospatial
