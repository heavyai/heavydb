/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
