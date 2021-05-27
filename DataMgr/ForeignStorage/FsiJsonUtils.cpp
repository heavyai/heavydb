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

#include <fstream>

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include <rapidjson/stringbuffer.h>

namespace foreign_storage {
namespace json_utils {

std::string get_type_as_string(const rapidjson::Value& object) {
  if (object.IsArray()) {
    return "array";
  } else if (object.IsBool()) {
    return "bool";
  } else if (object.IsDouble()) {
    return "double";
  } else if (object.IsFloat()) {
    return "float";
  } else if (object.IsInt64()) {
    return "int64";
  } else if (object.IsInt()) {
    return "int";
  } else if (object.IsNull()) {
    return "null";
  } else if (object.IsNumber()) {
    return "number";
  } else if (object.IsObject()) {
    return "onject";
  } else if (object.IsString()) {
    return "string";
  } else if (object.IsUint64()) {
    return "unit64";
  } else if (object.IsUint()) {
    return "unit";
  }
  return "unknown";
}

// Basic types (more can be added as required)
void set_value(rapidjson::Value& json_val,
               const size_t& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetUint64(value);
}
void get_value(const rapidjson::Value& json_val, size_t& value) {
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

// int64_t
void get_value(const rapidjson::Value& json_val, std::string& value) {
  CHECK(json_val.IsString());
  value = json_val.GetString();
}

void set_value(rapidjson::Value& json_val,
               const int64_t& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetInt64(value);
}

void get_value(const rapidjson::Value& json_val, int64_t& value) {
  CHECK(json_val.IsInt64());
  value = json_val.GetInt64();
}

rapidjson::Document read_from_file(const std::string& file_path) {
  std::ifstream ifs(file_path);
  if (!ifs) {
    throw std::runtime_error{"Error trying to open file \"" + file_path +
                             "\". The error was: " + std::strerror(errno)};
  }

  rapidjson::IStreamWrapper isw(ifs);
  rapidjson::Document d;
  d.ParseStream(isw);
  return d;
}

void write_to_file(const rapidjson::Document& document, const std::string& filepath) {
  std::ofstream ofs(filepath);
  if (!ofs) {
    throw std::runtime_error{"Error trying to create file \"" + filepath +
                             "\". The error was: " + std::strerror(errno)};
  }
  rapidjson::OStreamWrapper osw(ofs);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
  document.Accept(writer);
}

std::string write_to_string(const rapidjson::Document& document) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  document.Accept(writer);
  return buffer.GetString();
}
}  // namespace json_utils
}  // namespace foreign_storage
