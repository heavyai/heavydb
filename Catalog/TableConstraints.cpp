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
#include "Logger/Logger.h"
#include "Shared/JsonUtils.h"
#include "SqliteConnector/SqliteConnector.h"

#include <stdexcept>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <boost/algorithm/string.hpp>

namespace Catalog_Namespace {

namespace {

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
  json_utils::add_value_to_object(
      object, table_constraint_type_to_string(constraint.type), "type", allocator);
  json_utils::add_value_to_object(object, constraint.column_names, "columns", allocator);
  json_utils::add_value_to_object(object, constraint.enforced, "enforced", allocator);

  if (constraint.name && !constraint.name->empty()) {
    json_utils::add_value_to_object(object, *constraint.name, "name", allocator);
  }

  if (constraint.type == TableConstraintType::ForeignKey) {
    if (!constraint.foreign_key_reference) {
      throw std::runtime_error("Foreign key metadata is missing its reference.");
    }
    json_utils::add_value_to_object(
        object, constraint.foreign_key_reference->table_name, "foreign_table", allocator);
    json_utils::add_value_to_object(object,
                                    constraint.foreign_key_reference->column_names,
                                    "foreign_columns",
                                    allocator);
  }

  return object;
}

std::optional<TableConstraint> table_constraint_from_json(
    const rapidjson::Value& object) {
  if (!object.IsObject() || !object.HasMember("type")) {
    return {};
  }

  std::string type;
  json_utils::get_value_from_object(object, type, "type");
  const auto constraint_type = table_constraint_type_from_string(type);
  if (!constraint_type) {
    return {};
  }

  if (!object.HasMember("columns")) {
    throw std::runtime_error("Malformed table constraint metadata: missing columns.");
  }

  TableConstraint constraint;
  constraint.type = *constraint_type;
  json_utils::get_value_from_object(object, constraint.column_names, "columns");

  if (object.HasMember("name")) {
    std::string name;
    json_utils::get_value_from_object(object, name, "name");
    constraint.name = std::move(name);
  }
  if (object.HasMember("enforced")) {
    json_utils::get_value_from_object(object, constraint.enforced, "enforced");
  }

  if (constraint.type == TableConstraintType::ForeignKey) {
    if (!object.HasMember("foreign_table") || !object.HasMember("foreign_columns")) {
      throw std::runtime_error("Malformed foreign key metadata.");
    }
    ForeignKeyReference reference;
    json_utils::get_value_from_object(object, reference.table_name, "foreign_table");
    json_utils::get_value_from_object(object, reference.column_names, "foreign_columns");
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

SplitKeyMetadata split_key_metadata(const std::string& metadata) {
  auto document = parse_key_metainfo_document(metadata);
  SplitKeyMetadata result;
  rapidjson::Value directives(rapidjson::kArrayType);
  for (auto& entry : document.GetArray()) {
    const auto constraint = table_constraint_from_json(entry);
    if (!constraint) {
      directives.PushBack(entry, document.GetAllocator());
      continue;
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    entry.Accept(writer);
    result.constraints.push_back(
        {buffer.GetString(),
         constraint->foreign_key_reference
             ? std::make_optional(constraint->foreign_key_reference->table_name)
             : std::nullopt,
         directives.Size()});
  }
  if (result.constraints.empty()) {
    result.directives = metadata;
  } else {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    directives.Accept(writer);
    result.directives = buffer.GetString();
  }
  return result;
}

namespace {
std::string constraint_table(bool temporary) {
  return temporary ? "mapd_temp_constraints" : "mapd_constraints";
}

void create_constraint_table(SqliteConnector& db, bool temporary) {
  const auto table = constraint_table(temporary);
  db.query(
      std::string("CREATE ") + (temporary ? "TEMP " : "") + "TABLE IF NOT EXISTS " +
      table +
      "(owner INTEGER NOT NULL, position INTEGER NOT NULL, parent INTEGER, "
      "preceding_directives INTEGER NOT NULL, definition TEXT NOT NULL, "
      "CHECK ((parent IS NOT NULL) IS (upper(definition ->> 'type') = 'FOREIGN KEY')), "
      "PRIMARY KEY(owner,position)) WITHOUT ROWID");
  db.query("CREATE INDEX IF NOT EXISTS " + table + "_by_parent ON " + table +
           "(parent) WHERE parent IS NOT NULL");
}
}  // namespace

void delete_constraint_records(SqliteConnector& db, int owner, bool temporary) {
  db.query_with_text_param(
      "DELETE FROM " + constraint_table(temporary) + " WHERE owner = ?",
      std::to_string(owner));
}

void insert_constraint_records(
    SqliteConnector& db,
    int owner,
    bool temporary,
    const std::vector<StoredConstraint>& constraints,
    const std::function<int(const std::string&)>& resolve_parent) {
  size_t position = 0;
  for (const auto& entry : constraints) {
    db.query_with_text_params(
        "INSERT INTO " + constraint_table(temporary) +
            "(owner,position,parent,preceding_directives,definition) "
            "VALUES (?,?,NULLIF(?,''),?,?)",
        std::vector<std::string>{
            std::to_string(owner),
            std::to_string(position++),
            entry.parent ? std::to_string(resolve_parent(*entry.parent)) : "",
            std::to_string(entry.preceding_directives),
            entry.definition});
  }
}

void initialize_constraint_storage(SqliteConnector& db) {
  create_constraint_table(db, true);
  db.query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='mapd_constraints'");
  if (db.getNumRows()) {
    return;
  }
  db.query("BEGIN TRANSACTION");
  try {
    create_constraint_table(db, false);
    // Before constraint storage, master only wrote shard and dictionary directives.
    db.query(
        "SELECT name, key_metainfo FROM mapd_tables "
        "WHERE key_metainfo NOT IN ('','[]')");
    for (size_t i = 0; i < db.getNumRows(); ++i) {
      try {
        const auto metadata = db.getData<std::string>(i, 1);
        if (!parse_table_constraints_from_key_metainfo(metadata).empty()) {
          throw std::runtime_error(
              "Inline constraint declarations from a development build are not "
              "supported.");
        }
      } catch (const std::runtime_error& e) {
        const auto message = "Cannot migrate constraint metadata for table " +
                             db.getData<std::string>(i, 0) + ": " + e.what();
        LOG(ERROR) << message;
        throw std::runtime_error(message);
      }
    }
    db.query(
        "CREATE INDEX IF NOT EXISTS mapd_physical_table_lookup ON "
        "mapd_logical_to_physical(physical_table_id)");
    // The Thrift and embedded Calcite readers expose the original ordered JSON array.
    // Shards project the logical declaration, not another persisted copy.
    db.query(R"sql(CREATE VIEW mapd_table_key_metadata AS
      SELECT t.tableid, t.name,
        CASE WHEN coalesce(t.key_metainfo,'') IN ('','[]') AND NOT EXISTS
                    (SELECT 1 FROM mapd_constraints WHERE owner=coalesce(p.logical_table_id,t.tableid))
                  THEN t.key_metainfo
        WHEN NOT json_valid(coalesce(nullif(t.key_metainfo,''),'[]')) THEN t.key_metainfo
        WHEN json_type(coalesce(nullif(t.key_metainfo,''),'[]')) != 'array' THEN t.key_metainfo
        ELSE (SELECT json_group_array(json(value) ORDER BY slot, directive, pos) FROM (
          SELECT t.key_metainfo -> d.key AS value, d.key AS slot, 1 AS directive, 0 AS pos
            FROM json_each(coalesce(nullif(t.key_metainfo,''),'[]')) d
            WHERE CASE WHEN d.type='object' THEN
              coalesce(upper(d.value ->> 'type'),'') NOT IN ('FOREIGN KEY','PRIMARY KEY','UNIQUE')
              ELSE 1 END
          UNION ALL
          SELECT definition, preceding_directives, 0, position FROM mapd_constraints
            WHERE owner=coalesce(p.logical_table_id,t.tableid)))
        END AS key_metainfo
      FROM mapd_tables t LEFT JOIN mapd_logical_to_physical p
        ON p.physical_table_id=t.tableid)sql");
    db.query("END TRANSACTION");
  } catch (...) {
    end_transaction_or_abort(db, "ROLLBACK TRANSACTION", "migrating table constraints");
    throw;
  }
}

}  // namespace Catalog_Namespace
