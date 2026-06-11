/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Geospatial/GeosVersion.h"

#define GEOS_USE_ONLY_R_API
#include "geos_c.h"

#include "Logger/Logger.h"
#include "Shared/StringTransform.h"

namespace Geospatial {

std::string geos_version_required() {
  return std::string(GEOS_CAPI_VERSION);
}

bool geos_validate_version(const std::string& version_str) {
  // version string will be of the form
  // 3.11.1-CAPI-1.17.1
  auto const outer_tokens = split(version_str, "-CAPI-");
  if (outer_tokens.size() != 2) {
    return false;
  }
  auto const& geos_version = outer_tokens[0];
  auto const& geos_c_version = outer_tokens[1];
  auto const geos_version_tokens = split(geos_version, ".");
  auto const geos_c_version_tokens = split(geos_c_version, ".");
  if (geos_version_tokens.size() != 3 || geos_c_version_tokens.size() != 3) {
    return false;
  }
  auto const geos_major = std::stoi(geos_version_tokens[0]);
  auto const geos_minor = std::stoi(geos_version_tokens[1]);
  auto const geos_patch = std::stoi(geos_version_tokens[2]);
  auto const geos_c_major = std::stoi(geos_c_version_tokens[0]);
  auto const geos_c_minor = std::stoi(geos_c_version_tokens[1]);
  auto const geos_c_patch = std::stoi(geos_c_version_tokens[2]);
  auto const geos_numeric_version =
      (geos_major * 10000) + (geos_minor * 100) + geos_patch;
  auto const geos_c_numeric_version =
      (geos_c_major * 10000) + (geos_c_minor * 100) + geos_c_patch;
  auto const geos_min_numeric_version =
      (GEOS_VERSION_MAJOR * 10000) + (GEOS_VERSION_MINOR * 100) + GEOS_VERSION_PATCH;
  auto const geos_c_min_numeric_version = (GEOS_CAPI_VERSION_MAJOR * 10000) +
                                          (GEOS_CAPI_VERSION_MINOR * 100) +
                                          GEOS_CAPI_VERSION_PATCH;
  return (geos_numeric_version >= geos_min_numeric_version &&
          geos_c_numeric_version >= geos_c_min_numeric_version);
}

}  // namespace Geospatial
