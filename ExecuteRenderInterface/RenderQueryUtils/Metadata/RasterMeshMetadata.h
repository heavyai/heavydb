/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Parsers/CrossSectionFormatJson.h"
#include "QueryRenderer/Data/Parsers/RasterMeshFormatJson.h"

namespace QueryRenderer {

struct RasterMeshMetadata : public RasterMeshFormatJson {
  uint32_t width = 0;
  uint32_t height = 0;
};

struct CrossSectionMetadata : public CrossSectionFormatJson {
  uint32_t width = 0;
  uint32_t height = 0;
};

}  // namespace QueryRenderer
