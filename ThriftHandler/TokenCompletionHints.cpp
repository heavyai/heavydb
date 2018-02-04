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

#include "TokenCompletionHints.h"
#include "Shared/StringTransform.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/tokenizer.hpp>

namespace {

bool is_qualified_identifier_part(const char ch) {
  return isalnum(ch) || ch == '_' || ch == '.';
}

}  // namespace

// Straightforward port from SqlAdvisor.getCompletionHints.
std::string find_last_word_from_cursor(const std::string& sql, const ssize_t cursor) {
  if (cursor > static_cast<ssize_t>(sql.size())) {
    return "";
  }
  auto word_start = cursor;
  bool quoted = false;
  while (word_start > 0 && is_qualified_identifier_part(sql[word_start - 1])) {
    --word_start;
  }
  if ((word_start > 0) && (sql[word_start - 1] == '"')) {
    quoted = true;
    --word_start;
  }

  if (word_start < 0) {
    return "";
  }

  // Search forwards to the end of the word we should remove. Eat up
  // trailing double-quote, if any
  auto word_end = cursor;
  while (word_end < static_cast<ssize_t>(sql.size()) && is_qualified_identifier_part(sql[word_end - 1])) {
    ++word_end;
  }
  if (quoted && (word_end < static_cast<ssize_t>(sql.size())) && (sql[word_end] == '"')) {
    ++word_end;
  }
  std::string last_word(sql.begin() + word_start + (quoted ? 1 : 0), sql.begin() + cursor);
  return last_word;
}

std::vector<TCompletionHint> just_whitelisted_keyword_hints(const std::vector<TCompletionHint>& hints) {
  static const std::unordered_set<std::string> whitelisted_keywords{
      "WHERE",    "GROUP", "BY",   "COUNT", "AVG",   "MAX",    "MIN", "SUM",   "STDDEV_POP", "STDDEV_SAMP", "AS",
      "HAVING",   "INNER", "JOIN", "LEFT",  "LIMIT", "OFFSET", "ON",  "ORDER", "OUTER",      "ASC",         "DESC",
      "DISTINCT", "IN",    "IS",   "NULL",  "NOT",   "AND",    "OR",  "LIKE",  "*",          "(",           ")"};
  std::vector<TCompletionHint> filtered;
  for (const auto& original_hint : hints) {
    if (original_hint.type != TCompletionHintType::KEYWORD) {
      filtered.push_back(original_hint);
      continue;
    }
    auto filtered_hint = original_hint;
    filtered_hint.hints.clear();
    for (const auto hint_token : original_hint.hints) {
      if (whitelisted_keywords.find(to_upper(hint_token)) != whitelisted_keywords.end()) {
        filtered_hint.hints.push_back(hint_token);
      }
    }
    if (!filtered_hint.hints.empty()) {
      filtered.push_back(filtered_hint);
    }
  }
  return filtered;
}

bool get_qualified_column_hints(
    std::vector<TCompletionHint>& hints,
    const std::string& last_word,
    const std::unordered_map<std::string, std::unordered_set<std::string>>& column_names_by_table) {
  std::vector<std::string> last_word_tokens;
  boost::split(last_word_tokens, last_word, boost::is_any_of("."));
  if (last_word_tokens.size() < 2) {
    return false;
  }
  const auto table_name = last_word_tokens[last_word_tokens.size() - 2];
  const auto col_names_it = column_names_by_table.find(table_name);
  if (col_names_it == column_names_by_table.end()) {
    return false;
  }
  TCompletionHint column_hint;
  column_hint.type = TCompletionHintType::COLUMN;
  column_hint.replaced = last_word;
  for (const auto& col_name : col_names_it->second) {
    if (boost::istarts_with(col_name, last_word_tokens.back())) {
      auto qualified_name = last_word_tokens;
      qualified_name.back() = col_name;
      column_hint.hints.push_back(boost::algorithm::join(qualified_name, "."));
    }
  }
  if (!column_hint.hints.empty()) {
    hints.push_back(column_hint);
  }
  return true;
}

void get_column_hints(std::vector<TCompletionHint>& hints,
                      const std::string& last_word,
                      const std::unordered_map<std::string, std::unordered_set<std::string>>& column_names_by_table) {
  TCompletionHint column_hint;
  column_hint.type = TCompletionHintType::COLUMN;
  column_hint.replaced = last_word;
  std::unordered_set<std::string> column_hints_deduped;
  for (const auto& kv : column_names_by_table) {
    for (const auto& col_name : kv.second) {
      if (boost::istarts_with(col_name, last_word)) {
        column_hints_deduped.insert(col_name);
      }
    }
  }
  column_hint.hints.insert(column_hint.hints.end(), column_hints_deduped.begin(), column_hints_deduped.end());
  if (!column_hint.hints.empty()) {
    hints.push_back(column_hint);
  }
}

bool should_suggest_column_hints(const std::string& partial_query) {
  boost::char_separator<char> sep(" \t\n", ",");
  boost::tokenizer<boost::char_separator<char>> tokens(partial_query, sep);
  const auto token_count = std::distance(tokens.begin(), tokens.end());
  if (token_count == 1) {
    return true;
  }
  std::string last_token;
  std::string prev_to_last_token;
  for (const auto& token : tokens) {
    prev_to_last_token = last_token;
    last_token = token;
  }
  return last_token == "," ||
         (!partial_query.empty() && !isspace(partial_query.back()) &&
          (prev_to_last_token.empty() || prev_to_last_token == "," || to_upper(prev_to_last_token) == "SELECT"));
}
