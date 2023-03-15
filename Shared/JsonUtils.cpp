/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include "JsonUtils.h"

#include <fstream>

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include <rapidjson/stringbuffer.h>

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
    return "object";
  } else if (object.IsString()) {
    return "string";
  } else if (object.IsUint64()) {
    return "uint64";
  } else if (object.IsUint()) {
    return "uint";
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

// Boolean
void set_value(rapidjson::Value& json_val,
               const bool& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetBool(value);
}

void get_value(const rapidjson::Value& json_val, bool& value) {
  CHECK(json_val.IsBool());
  value = json_val.GetBool();
}

// SQLTypes
void set_value(rapidjson::Value& json_val,
               const SQLTypes& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetInt64(static_cast<int64_t>(value));
}

void get_value(const rapidjson::Value& json_val, SQLTypes& value) {
  CHECK(json_val.IsInt64());
  value = static_cast<SQLTypes>(json_val.GetInt64());
}

// EncodingType
void set_value(rapidjson::Value& json_val,
               const EncodingType& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetInt64(static_cast<int64_t>(value));
}

void get_value(const rapidjson::Value& json_val, EncodingType& value) {
  CHECK(json_val.IsInt64());
  value = static_cast<EncodingType>(json_val.GetInt64());
}

// StringDictKey
void set_value(rapidjson::Value& json_obj,
               const shared::StringDictKey& dict_key,
               rapidjson::Document::AllocatorType& allocator) {
  json_obj.SetObject();
  add_value_to_object(json_obj, dict_key.db_id, "db_id", allocator);
  add_value_to_object(json_obj, dict_key.dict_id, "dict_id", allocator);
}

void get_value(const rapidjson::Value& json_obj, shared::StringDictKey& dict_key) {
  CHECK(json_obj.IsObject());
  get_value_from_object(json_obj, dict_key.db_id, "db_id");
  get_value_from_object(json_obj, dict_key.dict_id, "dict_id");
}

// SQLTypeInfo
void set_value(rapidjson::Value& json_val,
               const SQLTypeInfo& type_info,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetObject();
  add_value_to_object(json_val, type_info.get_type(), "type", allocator);
  add_value_to_object(json_val, type_info.get_subtype(), "sub_type", allocator);
  add_value_to_object(json_val, type_info.get_dimension(), "dimension", allocator);
  add_value_to_object(json_val, type_info.get_scale(), "scale", allocator);
  add_value_to_object(json_val, type_info.get_notnull(), "notnull", allocator);
  add_value_to_object(json_val, type_info.get_compression(), "compression", allocator);
  add_value_to_object(json_val, type_info.get_comp_param(), "comp_param", allocator);
  add_value_to_object(json_val, type_info.get_size(), "size", allocator);
  add_value_to_object(json_val,
                      // Serialize StringDictKey without any checks, since the string
                      // dictionary ID may not be set at this point.
                      type_info.getStringDictKeySkipCompParamCheck(),
                      "string_dict_key",
                      allocator);
}

void get_value(const rapidjson::Value& json_val, SQLTypeInfo& type_info) {
  CHECK(json_val.IsObject());
  get_value_from_object<SQLTypes>(
      json_val, std::mem_fn(&SQLTypeInfo::set_type), type_info, "type");
  get_value_from_object<SQLTypes>(
      json_val, std::mem_fn(&SQLTypeInfo::set_subtype), type_info, "sub_type");
  get_value_from_object<int>(
      json_val, std::mem_fn(&SQLTypeInfo::set_dimension), type_info, "dimension");
  get_value_from_object<int>(
      json_val, std::mem_fn(&SQLTypeInfo::set_scale), type_info, "scale");
  get_value_from_object<bool>(
      json_val, std::mem_fn(&SQLTypeInfo::set_notnull), type_info, "notnull");
  get_value_from_object<EncodingType>(
      json_val, std::mem_fn(&SQLTypeInfo::set_compression), type_info, "compression");
  get_value_from_object<int>(
      json_val, std::mem_fn(&SQLTypeInfo::set_comp_param), type_info, "comp_param");
  get_value_from_object<int>(
      json_val, std::mem_fn(&SQLTypeInfo::set_size), type_info, "size");
  get_value_from_object<shared::StringDictKey>(
      json_val,
      // Deserialize StringDictKey without any checks.
      std::mem_fn(&SQLTypeInfo::setStringDictKeySkipCompParamCheck),
      type_info,
      "string_dict_key");
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

std::optional<std::string> get_optional_string_value_from_object(
    const rapidjson::Value& object,
    const std::string& key) {
  if (object.IsObject() && object.HasMember(key) && object[key].IsString()) {
    return object[key].GetString();
  }
  return {};
}
}  // namespace json_utils
