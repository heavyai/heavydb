/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace QueryRenderer {

enum class DataFormatType {
  kLines,
  kRasterMesh2d,
  kCrossSection1d,
  kCrossSection2d,
  kUnknown
};

std::string to_string(const DataFormatType mesh_format_type);
DataFormatType get_data_format_from_string(const std::string& format_string);

}  // namespace QueryRenderer
