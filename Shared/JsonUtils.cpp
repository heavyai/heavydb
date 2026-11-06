/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  if (!json_val.IsUint64()) {
    throw json_schema_error("Expected uint64, got " + get_type_as_string(json_val));
  }
  value = json_val.GetUint64();
}

void set_value(rapidjson::Value& json_val,
               const int& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetInt(value);
}

void get_value(const rapidjson::Value& json_val, int& value) {
  if (!json_val.IsInt()) {
    throw json_schema_error("Expected int32, got " + get_type_as_string(json_val));
  }
  value = json_val.GetInt();
}

void set_value(rapidjson::Value& json_val,
               const std::string& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetString(value, allocator);
}

// int64_t
void get_value(const rapidjson::Value& json_val, std::string& value) {
  if (!json_val.IsString()) {
    throw json_schema_error("Expected string, got " + get_type_as_string(json_val));
  }
  value = json_val.GetString();
}

void set_value(rapidjson::Value& json_val,
               const int64_t& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetInt64(value);
}

void get_value(const rapidjson::Value& json_val, int64_t& value) {
  if (!json_val.IsInt64()) {
    throw json_schema_error("Expected int64, got " + get_type_as_string(json_val));
  }
  value = json_val.GetInt64();
}

// Boolean
void set_value(rapidjson::Value& json_val,
               const bool& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetBool(value);
}

void get_value(const rapidjson::Value& json_val, bool& value) {
  if (!json_val.IsBool()) {
    throw json_schema_error("Expected bool, got " + get_type_as_string(json_val));
  }
  value = json_val.GetBool();
}

// SQLTypes
void set_value(rapidjson::Value& json_val,
               const SQLTypes& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetInt64(static_cast<int64_t>(value));
}

void get_value(const rapidjson::Value& json_val, SQLTypes& value) {
  int64_t serialized_value;
  get_value(json_val, serialized_value);
  value = static_cast<SQLTypes>(serialized_value);
}

// EncodingType
void set_value(rapidjson::Value& json_val,
               const EncodingType& value,
               rapidjson::Document::AllocatorType& allocator) {
  json_val.SetInt64(static_cast<int64_t>(value));
}

void get_value(const rapidjson::Value& json_val, EncodingType& value) {
  int64_t serialized_value;
  get_value(json_val, serialized_value);
  value = static_cast<EncodingType>(serialized_value);
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

rapidjson::Document read_from_string(const std::string& input) {
  rapidjson::StringStream iss(input.c_str());
  rapidjson::Document d;
  d.ParseStream(iss);
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

const rapidjson::Value& get_member(const rapidjson::Value& value,
                                   const std::string& key) {
  if (!value.IsObject()) {
    throw json_schema_error("Expected object, got " + get_type_as_string(value));
  }
  const auto member = value.FindMember(key);
  if (member == value.MemberEnd()) {
    throw json_schema_error("Could not find member: " + key);
  }
  return member->value;
}

rapidjson::Value::ConstArray get_array(const rapidjson::Value& value) {
  if (!value.IsArray()) {
    throw json_schema_error("Expected array, got " + get_type_as_string(value));
  }
  return value.GetArray();
}

int32_t get_int(const rapidjson::Value& value) {
  int32_t result;
  get_value(value, result);
  return result;
}

int32_t get_int_member(const rapidjson::Value& value, const std::string& key) {
  const auto& member = get_member(value, key);
  return get_int(member);
}

std::vector<int32_t> get_int_vector(const rapidjson::Value& value) {
  std::vector<int32_t> ret;
  get_value(value, ret);
  return ret;
}
}  // namespace json_utils
