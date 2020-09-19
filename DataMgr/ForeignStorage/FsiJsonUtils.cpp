/*
 * Copyright 2020 OmniSci, Inc.
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

#include "FsiJsonUtils.h"

namespace foreign_storage {
namespace json_utils {

// Basic types (more can be added as required)
void set_value(rapidjson::Value& json_val,
               const long unsigned int& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetUint64(value);
}
void get_value(const rapidjson::Value& json_val, long unsigned int& value) {
  CHECK(json_val.IsUint64());
  value = json_val.GetUint64();
}

void set_value(rapidjson::Value& json_val,
               const int& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetInt(value);
}

void get_value(const rapidjson::Value& json_val, int& value) {
  CHECK(json_val.IsInt());
  value = json_val.GetInt();
}

void set_value(rapidjson::Value& json_val,
               const std::string& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetString(value, allocator);
}

void get_value(const rapidjson::Value& json_val, std::string& value) {
  CHECK(json_val.IsString());
  value = json_val.GetString();
}

}  // namespace json_utils
}  // namespace foreign_storage
