/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    GeoOps.cpp
 * @brief   Functions to support geospatial operations used by the executor.
 *
 */

#include "../Geospatial/Compression.h"
#include "../Geospatial/Transforms.h"
#include "../Shared/funcannotations.h"
#include "TypePunning.h"

#ifdef EXECUTE_INCLUDE

extern "C" DEVICE RUNTIME_EXPORT double decompress_x_coord_geoint(const int32_t coord) {
  return Geospatial::decompress_longitude_coord_geoint32(coord);
}

extern "C" DEVICE RUNTIME_EXPORT double decompress_y_coord_geoint(const int32_t coord) {
  return Geospatial::decompress_latitude_coord_geoint32(coord);
}

extern "C" DEVICE RUNTIME_EXPORT int32_t compress_x_coord_geoint(const double coord) {
  return static_cast<int32_t>(Geospatial::compress_longitude_coord_geoint32(coord));
}

extern "C" DEVICE RUNTIME_EXPORT int32_t compress_y_coord_geoint(const double coord) {
  return static_cast<int32_t>(Geospatial::compress_latitude_coord_geoint32(coord));
}

#endif
