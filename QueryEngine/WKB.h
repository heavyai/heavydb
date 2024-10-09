/*
 * Copyright 2024 HEAVY.AI, Inc.
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
