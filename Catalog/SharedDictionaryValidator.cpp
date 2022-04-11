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

#include "Catalog/Catalog.h"
#include "Logger/Logger.h"

const Analyzer::SharedDictionaryDef compress_reference_path(
    Analyzer::SharedDictionaryDef cur_node,
    const std::vector<Analyzer::SharedDictionaryDef>& shared_dict_defs) {
  UNREACHABLE();
}

// Make sure the dependency of shared dictionaries does not form a cycle
void validate_shared_dictionary_order(
    const Parser::CreateTableBaseStmt* stmt,
    const Analyzer::SharedDictionaryDef* shared_dict_def,
    const std::vector<Analyzer::SharedDictionaryDef>& shared_dict_defs,
    const std::list<ColumnDescriptor>& columns) {
  UNREACHABLE();
}

// Validate shared dictionary directive against the list of columns seen so far.
void validate_shared_dictionary(
    const Parser::CreateTableBaseStmt* stmt,
    const Analyzer::SharedDictionaryDef* shared_dict_def,
    const std::list<ColumnDescriptor>& columns,
    const std::vector<Analyzer::SharedDictionaryDef>& shared_dict_defs_so_far,
    const Catalog_Namespace::Catalog& catalog) {
  UNREACHABLE();
}
