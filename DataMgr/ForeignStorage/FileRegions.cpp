/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "FileRegions.h"

#include "Shared/JsonUtils.h"

namespace foreign_storage {
// Serialization functions for FileRegion
void set_value(rapidjson::Value& json_val,
               const FileRegion& file_region,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetObject();
  json_utils::add_value_to_object(
      json_val, file_region.first_row_file_offset, "first_row_file_offset", allocator);
  json_utils::add_value_to_object(
      json_val, file_region.first_row_index, "first_row_index", allocator);
  json_utils::add_value_to_object(
      json_val, file_region.region_size, "region_size", allocator);
  json_utils::add_value_to_object(
      json_val, file_region.row_count, "row_count", allocator);
  if (file_region.file_path.size()) {
    json_utils::add_value_to_object(
        json_val, file_region.file_path, "file_path", allocator);
  }
}

void get_value(const rapidjson::Value& json_val, FileRegion& file_region) {
  CHECK(json_val.IsObject());
  json_utils::get_value_from_object(
      json_val, file_region.first_row_file_offset, "first_row_file_offset");
  json_utils::get_value_from_object(
      json_val, file_region.first_row_index, "first_row_index");
  json_utils::get_value_from_object(json_val, file_region.region_size, "region_size");
  json_utils::get_value_from_object(json_val, file_region.row_count, "row_count");
  if (json_val.HasMember("file_path")) {
    json_utils::get_value_from_object(json_val, file_region.file_path, "file_path");
  } else if (json_val.HasMember("filename")) {
    // Handle legacy "filename" field name
    json_utils::get_value_from_object(json_val, file_region.file_path, "filename");
  }
}
}  // namespace foreign_storage
