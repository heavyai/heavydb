/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace Geospatial {

bool geos_validation_available();
bool geos_validate_wkb(const unsigned char* wkb, const size_t wkb_size);

}  // namespace Geospatial
