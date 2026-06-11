/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef THRIFTHANDLER_TOKENCOMPLETIONHINTS_H
#define THRIFTHANDLER_TOKENCOMPLETIONHINTS_H

#include "gen-cpp/completion_hints_types.h"
// completion_hints_types.h > Thrift.h > PlatformSocket.h > winsock2.h > windows.h
#include "Shared/cleanup_global_namespace.h"

#include <unordered_map>
#include <unordered_set>

// Find last "word" (can contain: alphanumeric, underscore, dot) from position
// `cursor` inside or at the end of `sql`.
std::string find_last_word_from_cursor(const std::string& sql, const int64_t cursor);

// Only allows a few whitelisted keywords, filters out everything else.
std::vector<TCompletionHint> just_whitelisted_keyword_hints(
    const std::vector<TCompletionHint>& hints);

// Given last_word = "table.prefix", returns column hints for all columns in "table" which
// start with "prefix" from `column_names_by_table["table"]`. Returns true iff `last_word`
// looks like a qualified name (contains a dot).
bool get_qualified_column_hints(
    std::vector<TCompletionHint>& hints,
    const std::string& last_word,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        column_names_by_table);

// Returns column hints for the flattened list of all values in `column_names_by_table`
// which start with `last_word`.
void get_column_hints(
    std::vector<TCompletionHint>& hints,
    const std::string& last_word,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        column_names_by_table);

// Returns true iff it should suggest columns or just the FROM keyword,
// should be called for partial queries after SELECT but before FROM.
bool should_suggest_column_hints(const std::string& partial_query);

#endif  // THRIFTHANDLER_TOKENCOMPLETIONHINTS_H
