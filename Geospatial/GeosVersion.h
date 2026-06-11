/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace Geospatial {

std::string geos_version_required();

bool geos_validate_version(const std::string& version_str);

}  // namespace Geospatial
