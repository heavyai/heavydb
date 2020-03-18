/*
 * Copyright 2020 OmniSci, Inc.
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
#include <vector>

#include "QueryEngine/TypePunning.h"
#include "Shared/geo_compression_runtime.h"
#include "Shared/sqltypes.h"
namespace geospatial {

uint64_t compress_coord(double coord, const SQLTypeInfo& ti, bool x);

uint64_t compress_null_point(const SQLTypeInfo& ti, bool x);

// Compress non-NULL geo coords; and also NULL POINT coords (special case)
std::vector<uint8_t> compress_coords(std::vector<double>& coords, const SQLTypeInfo& ti);

}  // namespace geospatial
