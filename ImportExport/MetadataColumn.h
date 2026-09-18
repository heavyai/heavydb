/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file MetadataColumn.h
 * @brief Metadata Column info struct and parser
 *
 */

#pragma once

#include <string>

#include "Catalog/ColumnDescriptor.h"

namespace import_export {

struct MetadataColumnInfo {
  ColumnDescriptor column_descriptor;
  std::string value;
};

using MetadataColumnInfos = std::vector<MetadataColumnInfo>;

MetadataColumnInfos parse_add_metadata_columns(const std::string& add_metadata_columns,
                                               const std::string& file_path);

}  // namespace import_export
