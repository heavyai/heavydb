/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "QueryRenderer/Data/Enums/DataFormatType.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

struct RasterMeshFormatJson {
  static constexpr DataFormatType data_format_type = DataFormatType::kRasterMesh2d;

  std::string x_coord_name = "";
  std::string y_coord_name = "";

  static void validate(const JSONLocation& parent_loc);
  static void parse(RasterMeshFormatJson& mesh_format_json,
                    const JSONLocation& parent_loc);

 protected:
  static void validateCoordinateProp(const JSONLocation& parent_loc,
                                     const std::string_view prop_name);
};

}  // namespace QueryRenderer
