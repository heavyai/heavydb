/*
 * Copyright 2021 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "FileRegions.h"

#include "DataMgr/ForeignStorage/FsiJsonUtils.h"

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
  if (file_region.filename.size()) {
    json_utils::add_value_to_object(
        json_val, file_region.filename, "filename", allocator);
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
  if (json_val.HasMember("filename")) {
    json_utils::get_value_from_object(json_val, file_region.filename, "filename");
  }
}
}  // namespace foreign_storage
