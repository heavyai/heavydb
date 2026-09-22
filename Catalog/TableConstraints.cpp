/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Copyright 2026 HEAVY.AI, Inc.
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

#include "Catalog/TableConstraints.h"

#include <cstdint>
#include <stdexcept>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <boost/algorithm/string.hpp>

namespace Catalog_Namespace {

namespace {

std::string json_string(const rapidjson::Value& value) {
  if (!value.IsString()) {
    throw std::runtime_error("Malformed table constraint metadata.");
  }
  return {value.GetString(), value.GetStringLength()};
}

std::vector<std::string> json_string_array(const rapidjson::Value& value) {
  if (!value.IsArray()) {
    throw std::runtime_error("Malformed table constraint metadata.");
  }
  std::vector<std::string> strings;
  strings.reserve(value.Size());
  for (const auto& element : value.GetArray()) {
    strings.emplace_back(json_string(element));
  }
  return strings;
}

std::vector<int32_t> json_int_array(const rapidjson::Value& value) {
  if (!value.IsArray()) {
    throw std::runtime_error("Malformed table constraint metadata.");
  }
  std::vector<int32_t> ints;
  ints.reserve(value.Size());
  for (const auto& element : value.GetArray()) {
    if (!element.IsInt()) {
      throw std::runtime_error("Malformed table constraint metadata.");
    }
    ints.emplace_back(element.GetInt());
  }
  return ints;
}

void add_string_member(rapidjson::Value& object,
                       const char* key,
                       const std::string& value,
                       rapidjson::Document::AllocatorType& allocator) {
  rapidjson::Value json_key;
  json_key.SetString(key, allocator);
  rapidjson::Value json_value;
  json_value.SetString(value.c_str(), value.size(), allocator);
  object.AddMember(json_key, json_value, allocator);
}

void add_string_array_member(rapidjson::Value& object,
                             const char* key,
                             const std::vector<std::string>& values,
                             rapidjson::Document::AllocatorType& allocator) {
  rapidjson::Value json_key;
  json_key.SetString(key, allocator);
  rapidjson::Value json_values(rapidjson::kArrayType);
  for (const auto& value : values) {
    rapidjson::Value json_value;
    json_value.SetString(value.c_str(), value.size(), allocator);
    json_values.PushBack(json_value, allocator);
  }
  object.AddMember(json_key, json_values, allocator);
}

void add_int_array_member(rapidjson::Value& object,
                          const char* key,
                          const std::vector<int32_t>& values,
                          rapidjson::Document::AllocatorType& allocator) {
  rapidjson::Value json_key;
  json_key.SetString(key, allocator);
  rapidjson::Value json_values(rapidjson::kArrayType);
  for (const auto value : values) {
    json_values.PushBack(value, allocator);
  }
  object.AddMember(json_key, json_values, allocator);
}

rapidjson::Document parse_key_metainfo_document(const std::string& key_metainfo) {
  rapidjson::Document document;
  if (key_metainfo.empty()) {
    document.SetArray();
    return document;
  }
  document.Parse(key_metainfo.c_str());
  if (document.HasParseError() || !document.IsArray()) {
    throw std::runtime_error("Malformed table key metadata.");
  }
  return document;
}

rapidjson::Value table_constraint_to_json(const TableConstraint& constraint,
                                          rapidjson::Document::AllocatorType& allocator) {
  rapidjson::Value object(rapidjson::kObjectType);
  add_string_member(
      object, "type", table_constraint_type_to_string(constraint.type), allocator);
  add_string_array_member(object, "columns", constraint.column_names, allocator);
  object.AddMember(rapidjson::StringRef("enforced"), constraint.enforced, allocator);

  if (constraint.name && !constraint.name->empty()) {
    add_string_member(object, "name", *constraint.name, allocator);
  }

  if (constraint.type == TableConstraintType::ForeignKey) {
    if (!constraint.foreign_key_reference) {
      throw std::runtime_error("Foreign key metadata is missing its reference.");
    }
    add_string_member(
        object, "foreign_table", constraint.foreign_key_reference->table_name, allocator);
    add_string_array_member(object,
                            "foreign_columns",
                            constraint.foreign_key_reference->column_names,
                            allocator);
    if (!constraint.foreign_key_reference->column_ordinals.empty()) {
      add_int_array_member(object,
                           "foreign_column_ordinals",
                           constraint.foreign_key_reference->column_ordinals,
                           allocator);
    }
  }

  return object;
}

std::optional<TableConstraint> table_constraint_from_json(
    const rapidjson::Value& object) {
  if (!object.IsObject() || !object.HasMember("type")) {
    return {};
  }

  const auto constraint_type =
      table_constraint_type_from_string(json_string(object["type"]));
  if (!constraint_type) {
    return {};
  }

  if (!object.HasMember("columns")) {
    throw std::runtime_error("Malformed table constraint metadata: missing columns.");
  }

  TableConstraint constraint;
  constraint.type = *constraint_type;
  constraint.column_names = json_string_array(object["columns"]);

  if (object.HasMember("name") && object["name"].IsString()) {
    constraint.name = json_string(object["name"]);
  }
  if (object.HasMember("enforced")) {
    if (!object["enforced"].IsBool()) {
      throw std::runtime_error(
          "Malformed table constraint metadata: invalid enforced flag.");
    }
    constraint.enforced = object["enforced"].GetBool();
  }

  if (constraint.type == TableConstraintType::ForeignKey) {
    if (!object.HasMember("foreign_table") || !object.HasMember("foreign_columns")) {
      throw std::runtime_error("Malformed foreign key metadata.");
    }
    ForeignKeyReference reference;
    reference.table_name = json_string(object["foreign_table"]);
    reference.column_names = json_string_array(object["foreign_columns"]);
    if (object.HasMember("foreign_column_ordinals")) {
      reference.column_ordinals = json_int_array(object["foreign_column_ordinals"]);
    }
    constraint.foreign_key_reference = reference;
  }

  return constraint;
}

}  // namespace

std::string table_constraint_type_to_string(const TableConstraintType type) {
  switch (type) {
    case TableConstraintType::PrimaryKey:
      return "PRIMARY KEY";
    case TableConstraintType::Unique:
      return "UNIQUE";
    case TableConstraintType::ForeignKey:
      return "FOREIGN KEY";
  }
  throw std::runtime_error("Unknown table constraint type.");
}

std::optional<TableConstraintType> table_constraint_type_from_string(
    const std::string& type) {
  const auto normalized_type = boost::to_upper_copy<std::string>(type);
  if (normalized_type == "PRIMARY KEY") {
    return TableConstraintType::PrimaryKey;
  }
  if (normalized_type == "UNIQUE") {
    return TableConstraintType::Unique;
  }
  if (normalized_type == "FOREIGN KEY") {
    return TableConstraintType::ForeignKey;
  }
  return {};
}

bool table_constraint_is_trusted(const TableConstraint& constraint,
                                 const bool trust_unenforced_constraints) {
  return constraint.enforced || trust_unenforced_constraints;
}

std::vector<TableConstraint> parse_table_constraints_from_key_metainfo(
    const std::string& key_metainfo) {
  auto document = parse_key_metainfo_document(key_metainfo);
  std::vector<TableConstraint> constraints;
  for (const auto& element : document.GetArray()) {
    auto constraint = table_constraint_from_json(element);
    if (constraint) {
      constraints.emplace_back(std::move(*constraint));
    }
  }
  return constraints;
}

std::string append_table_constraint_to_key_metainfo(const std::string& key_metainfo,
                                                    const TableConstraint& constraint) {
  auto document = parse_key_metainfo_document(key_metainfo);
  document.PushBack(table_constraint_to_json(constraint, document.GetAllocator()),
                    document.GetAllocator());

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  document.Accept(writer);
  return buffer.GetString();
}

std::string remove_table_constraint_from_key_metainfo(
    const std::string& key_metainfo,
    const std::string& constraint_name) {
  auto document = parse_key_metainfo_document(key_metainfo);
  std::optional<rapidjson::SizeType> matching_index;
  for (rapidjson::SizeType index = 0; index < document.Size(); ++index) {
    const auto constraint = table_constraint_from_json(document[index]);
    if (!constraint || !constraint->name ||
        !boost::iequals(*constraint->name, constraint_name)) {
      continue;
    }
    if (matching_index) {
      throw std::runtime_error("Duplicate table constraint name " + constraint_name +
                               " in catalog metadata.");
    }
    matching_index = index;
  }
  if (!matching_index) {
    throw std::runtime_error("Table constraint " + constraint_name + " does not exist.");
  }

  rapidjson::Value remaining_entries(rapidjson::kArrayType);
  auto& allocator = document.GetAllocator();
  for (rapidjson::SizeType index = 0; index < document.Size(); ++index) {
    if (index == *matching_index) {
      continue;
    }
    rapidjson::Value entry;
    entry.CopyFrom(document[index], allocator);
    remaining_entries.PushBack(entry, allocator);
  }
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  remaining_entries.Accept(writer);
  return buffer.GetString();
}

}  // namespace Catalog_Namespace
