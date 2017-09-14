/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "SharedDictionaryValidator.h"

#include <memory>

#include "../Shared/unreachable.h"

const Parser::SharedDictionaryDef compress_reference_path(
    Parser::SharedDictionaryDef cur_node,
    const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs) {
  size_t idx = 0;
  for (; idx < shared_dict_defs.size(); idx++) {
    if (!shared_dict_defs[idx].get_column().compare(cur_node.get_column()) &&
        !shared_dict_defs[idx].get_foreign_table().compare(cur_node.get_foreign_table()) &&
        !shared_dict_defs[idx].get_foreign_column().compare(cur_node.get_foreign_column()))
      break;
  }
  // Make sure we have found the shared dictionary definition
  CHECK_LT(idx, shared_dict_defs.size());

  size_t ret_val_idx = idx;
  for (size_t j = 0; j < shared_dict_defs.size(); j++) {
    for (size_t i = 0; i < shared_dict_defs.size(); ++i) {
      if (!shared_dict_defs[i].get_column().compare(shared_dict_defs[ret_val_idx].get_foreign_column())) {
        ret_val_idx = i;
        break;
      }
    }
    if (shared_dict_defs[ret_val_idx].get_foreign_table().compare(cur_node.get_foreign_table())) {
      // found a dictionary share definition which shares the dict outside this table to be created
      break;
    }
  }

  return shared_dict_defs[ret_val_idx];
}

// Make sure the dependency of shared dictionaries does not form a cycle
void validate_shared_dictionary_order(const Parser::CreateTableStmt* stmt,
                                      const Parser::SharedDictionaryDef* shared_dict_def,
                                      const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs,
                                      const std::list<ColumnDescriptor>& columns) {
  std::string reference_col_qualified_name =
      shared_dict_def->get_foreign_column() + "." + shared_dict_def->get_foreign_table();
  if (!shared_dict_def->get_column().compare(shared_dict_def->get_foreign_column())) {
    throw std::runtime_error("Dictionary cannot be shared with itself. For dictionary : " +
                             reference_col_qualified_name);
  }
  auto table_name = stmt->get_table();
  CHECK(!shared_dict_def->get_foreign_table().compare(*table_name));
  auto col = std::find_if(columns.rbegin(), columns.rend(), [shared_dict_def](const ColumnDescriptor& elem) {
    return !elem.columnName.compare(shared_dict_def->get_column());
  });
  CHECK(col != columns.rend());
  auto ref_col = std::find_if(col, columns.rend(), [shared_dict_def](const ColumnDescriptor& elem) {
    return !elem.columnName.compare(shared_dict_def->get_foreign_column());
  });

  if (ref_col == columns.rend()) {
    throw std::runtime_error("Dictionary dependencies might create a cycle for " + shared_dict_def->get_column() +
                             "referencing " + reference_col_qualified_name);
  }
}

namespace {
const ColumnDescriptor* lookup_column(const std::string& name, const std::list<ColumnDescriptor>& columns) {
  for (const auto& cd : columns) {
    if (cd.columnName == name) {
      return &cd;
    }
  }
  return nullptr;
}

const ColumnDescriptor* lookup_column(const std::string& name, const std::list<const ColumnDescriptor*>& columns) {
  for (const auto& cd : columns) {
    if (cd->columnName == name) {
      return cd;
    }
  }
  return nullptr;
}

const Parser::CompressDef* get_compression_for_column(
    const std::string& name,
    const std::list<std::unique_ptr<Parser::TableElement>>& table_element_list) {
  for (const auto& e : table_element_list) {
    const auto col_def = dynamic_cast<Parser::ColumnDef*>(e.get());
    if (!col_def || *col_def->get_column_name() != name) {
      continue;
    }
    return col_def->get_compression();
  }
  UNREACHABLE();
  return nullptr;
}

}  // namespace

// Validate shared dictionary directive against the list of columns seen so far.
void validate_shared_dictionary(const Parser::CreateTableStmt* stmt,
                                const Parser::SharedDictionaryDef* shared_dict_def,
                                const std::list<ColumnDescriptor>& columns,
                                const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs_so_far,
                                const Catalog_Namespace::Catalog& catalog) {
  CHECK(shared_dict_def);
  auto table_name = stmt->get_table();
  const auto cd_ptr = lookup_column(shared_dict_def->get_column(), columns);
  const auto col_qualified_name = *table_name + "." + shared_dict_def->get_column();
  if (!cd_ptr) {
    throw std::runtime_error("Column " + col_qualified_name + " doesn't exist");
  }
  if (!cd_ptr->columnType.is_string() || cd_ptr->columnType.get_compression() != kENCODING_DICT) {
    throw std::runtime_error("Column " + col_qualified_name + " must be a dictionary encoded string");
  }
  const std::list<std::unique_ptr<Parser::TableElement>>& table_element_list = stmt->get_table_element_list();
  if (get_compression_for_column(shared_dict_def->get_column(), table_element_list)) {
    throw std::runtime_error("Column " + col_qualified_name +
                             " shouldn't specify an encoding, it borrows it from the referenced column");
  }
  const auto foreign_td = catalog.getMetadataForTable(shared_dict_def->get_foreign_table());
  if (!foreign_td && table_name->compare(shared_dict_def->get_foreign_table())) {
    throw std::runtime_error("Table " + shared_dict_def->get_foreign_table() + " doesn't exist");
  }

  if (foreign_td) {
    const auto reference_columns = catalog.getAllColumnMetadataForTable(foreign_td->tableId, false, false);
    const auto reference_cd_ptr = lookup_column(shared_dict_def->get_foreign_column(), reference_columns);
    const auto reference_col_qualified_name =
        reference_cd_ptr->columnName + "." + shared_dict_def->get_foreign_column();
    if (!reference_cd_ptr) {
      throw std::runtime_error("Column " + reference_col_qualified_name + " doesn't exist");
    }
    if (!reference_cd_ptr->columnType.is_string() || reference_cd_ptr->columnType.get_compression() != kENCODING_DICT) {
      throw std::runtime_error("Column " + reference_col_qualified_name + " must be a dictionary encoded string");
    }
  } else {
    // The dictionary is to be shared within table
    const auto reference_col_qualified_name = *table_name + "." + shared_dict_def->get_foreign_column();
    const auto reference_cd_ptr = lookup_column(shared_dict_def->get_foreign_column(), columns);
    if (!reference_cd_ptr) {
      throw std::runtime_error("Column " + reference_col_qualified_name + " doesn't exist");
    }
    if (!reference_cd_ptr->columnType.is_string() || reference_cd_ptr->columnType.get_compression() != kENCODING_DICT) {
      throw std::runtime_error("Column " + reference_col_qualified_name + " must be a dictionary encoded string");
    }
    validate_shared_dictionary_order(stmt, shared_dict_def, shared_dict_defs_so_far, columns);
  }
  const auto it = std::find_if(shared_dict_defs_so_far.begin(),
                               shared_dict_defs_so_far.end(),
                               [shared_dict_def](const Parser::SharedDictionaryDef& elem) {
                                 return elem.get_column() == shared_dict_def->get_column();
                               });
  if (it != shared_dict_defs_so_far.end()) {
    throw std::runtime_error("Duplicate shared dictionary hint for column " + *table_name + "." +
                             shared_dict_def->get_column());
  }
}
