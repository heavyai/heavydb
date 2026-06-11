/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "Shared/funcannotations.h"

extern "C" RUNTIME_EXPORT bool H3_CellToBoundary_POLYGON(const int64_t cell,
                                                         int* result_type,
                                                         int8_t** result_coords,
                                                         int64_t* result_coords_size,
                                                         int32_t** result_ring_sizes,
                                                         int64_t* result_ring_sizes_size);

extern "C" RUNTIME_EXPORT bool H3_CellToPoint(const int64_t cell,
                                              int* result_type,
                                              int8_t** result_coords,
                                              int64_t* result_coords_size);
