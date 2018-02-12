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

#ifndef THRIFTHANDLER_TOKENCOMPLETIONHINTS_H
#define THRIFTHANDLER_TOKENCOMPLETIONHINTS_H

#include "gen-cpp/completion_hints_types.h"

#include <unordered_map>
#include <unordered_set>

// Find last "word" (can contain: alphanumeric, underscore, dot) from position
// `cursor` inside or at the end of `sql`.
std::string find_last_word_from_cursor(const std::string& sql, const ssize_t cursor);

// Only allows a few whitelisted keywords, filters out everything else.
std::vector<TCompletionHint> just_whitelisted_keyword_hints(const std::vector<TCompletionHint>& hints);

// Given last_word = "table.prefix", returns column hints for all columns in "table" which start with "prefix" from
// `column_names_by_table["table"]`. Returns true iff `last_word` looks like a qualified name (contains a dot).
bool get_qualified_column_hints(
    std::vector<TCompletionHint>& hints,
    const std::string& last_word,
    const std::unordered_map<std::string, std::unordered_set<std::string>>& column_names_by_table);

// Returns column hints for the flattened list of all values in `column_names_by_table` which start with `last_word`.
void get_column_hints(std::vector<TCompletionHint>& hints,
                      const std::string& last_word,
                      const std::unordered_map<std::string, std::unordered_set<std::string>>& column_names_by_table);

// Returns true iff it should suggest columns or just the FROM keyword,
// should be called for partial queries after SELECT but before FROM.
bool should_suggest_column_hints(const std::string& partial_query);

#endif  // THRIFTHANDLER_TOKENCOMPLETIONHINTS_H
