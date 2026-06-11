/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SHARED_DICTIONARY_VALIDATOR_H
#define SHARED_DICTIONARY_VALIDATOR_H

#include <vector>

#include "../Catalog/ColumnDescriptor.h"
#include "../Parser/ParserNode.h"

void validate_shared_dictionary(
    const Parser::CreateTableBaseStmt* stmt,
    const Parser::SharedDictionaryDef* shared_dict_def,
    const std::list<ColumnDescriptor>& columns,
    const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs_so_far,
    const Catalog_Namespace::Catalog& catalog);

const Parser::SharedDictionaryDef compress_reference_path(
    Parser::SharedDictionaryDef cur_node,
    const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs);

void validate_shared_dictionary_order(
    const Parser::CreateTableBaseStmt* stmt,
    const Parser::SharedDictionaryDef* shared_dict_def,
    const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs,
    const std::list<ColumnDescriptor>& columns);
#endif  // SHARED_DICTIONARY_VALIDATOR_H
