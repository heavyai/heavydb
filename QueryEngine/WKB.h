/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "Shared/funcannotations.h"

extern "C" RUNTIME_EXPORT NEVER_INLINE uint8_t* toWkb(
    size_t* wkb_size,
    int type,  // internal geometry type
    int8_t* coords,
    int64_t coords_size,
    int32_t* meta1,      // e.g. ring_sizes
    int64_t meta1_size,  // e.g. num_rings
    int32_t* meta2,      // e.g. rings (number of rings in each poly)
    int64_t meta2_size,  // e.g. num_polys
    int32_t ic,          // input compression
    int32_t srid_in,     // input srid
    int32_t srid_out,    // output srid
    int32_t* best_planar_srid_ptr);

extern "C" RUNTIME_EXPORT NEVER_INLINE bool fromWkb(const uint8_t* wkb_ptr,
                                                    const size_t wkb_size,
                                                    int* result_type,
                                                    int8_t** result_coords,
                                                    int64_t* result_coords_size,
                                                    int32_t** result_meta1,
                                                    int64_t* result_meta1_size,
                                                    int32_t** result_meta2,
                                                    int64_t* result_meta2_size,
                                                    int32_t result_srid_in,
                                                    int32_t result_srid_out,
                                                    int32_t* best_planar_srid_ptr);
