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

#ifndef TABLE_CONSTRAINTS_H
#define TABLE_CONSTRAINTS_H

#include <optional>
#include <string>
#include <vector>

namespace Catalog_Namespace {

enum class TableConstraintType { PrimaryKey, Unique, ForeignKey };

struct ForeignKeyReference {
  std::string table_name;
  std::vector<std::string> column_names;
  std::vector<int32_t> column_ordinals;
};

struct TableConstraint {
  TableConstraintType type;
  std::optional<std::string> name;
  std::vector<std::string> column_names;
  std::optional<ForeignKeyReference> foreign_key_reference;
  bool enforced{false};
};

std::string table_constraint_type_to_string(const TableConstraintType type);

std::optional<TableConstraintType> table_constraint_type_from_string(
    const std::string& type);

bool table_constraint_is_trusted(const TableConstraint& constraint,
                                 bool trust_unenforced_constraints);

std::vector<TableConstraint> parse_table_constraints_from_key_metainfo(
    const std::string& key_metainfo);

std::string append_table_constraint_to_key_metainfo(const std::string& key_metainfo,
                                                    const TableConstraint& constraint);

std::string remove_table_constraint_from_key_metainfo(const std::string& key_metainfo,
                                                      const std::string& constraint_name);

}  // namespace Catalog_Namespace

#endif  // TABLE_CONSTRAINTS_H
