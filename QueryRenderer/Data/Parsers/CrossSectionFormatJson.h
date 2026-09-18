/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Parsers/RasterMeshFormatJson.h"

namespace QueryRenderer {

struct CrossSectionFormatJson : public RasterMeshFormatJson {
  static constexpr DataFormatType data_format_type = DataFormatType::kCrossSection2d;

  std::array<std::array<double, 2>, 2> linestring;

  static void validate(const JSONLocation& parent_loc);
  static void parse(CrossSectionFormatJson& cross_section_format_json,
                    const JSONLocation& parent_loc);
};

}  // namespace QueryRenderer
