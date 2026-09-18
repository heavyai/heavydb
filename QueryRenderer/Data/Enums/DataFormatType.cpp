/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Enums/DataFormatType.h"

#include "Logger/Logger.h"

namespace QueryRenderer {

std::string to_string(const DataFormatType mesh_format_type) {
  switch (mesh_format_type) {
    case DataFormatType::kLines:
      return "lines";
    case DataFormatType::kRasterMesh2d:
      return "raster_mesh2d";
    case DataFormatType::kCrossSection1d:
      return "cross_section1d";
    case DataFormatType::kCrossSection2d:
      return "cross_section2d";
    case DataFormatType::kUnknown:
      return "unknown";
  }
  UNREACHABLE();
  return "";
}

DataFormatType get_data_format_from_string(const std::string& format_string) {
  for (auto i = 0u; i < static_cast<uint32_t>(DataFormatType::kUnknown); ++i) {
    auto const curr_mesh_format = static_cast<DataFormatType>(i);
    if (to_string(curr_mesh_format) == format_string) {
      return curr_mesh_format;
    }
  }
  return DataFormatType::kUnknown;
}

}  // namespace QueryRenderer
