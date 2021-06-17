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

#include "Geospatial/CompressionRuntime.h"
#include "Shared/sqltypes.h"
namespace Geospatial {

int32_t get_compression_scheme(const SQLTypeInfo& ti);

uint64_t compress_coord(double coord, const SQLTypeInfo& ti, bool x);

uint64_t compress_null_point(const SQLTypeInfo& ti, bool x);

bool is_null_point(const SQLTypeInfo& geo_ti,
                   const int8_t* coords,
                   const size_t coords_sz);

// Compress non-NULL geo coords; and also NULL POINT coords (special case)
std::vector<uint8_t> compress_coords(const std::vector<double>& coords,
                                     const SQLTypeInfo& ti);

template <typename T>
void unpack_geo_vector(std::vector<T>& output, const int8_t* input_ptr, const size_t sz);

template <typename T, typename C>
std::shared_ptr<std::vector<T>> decompress_coords(const C& compression,
                                                  const int8_t* coords,
                                                  const size_t coords_sz);
}  // namespace Geospatial
