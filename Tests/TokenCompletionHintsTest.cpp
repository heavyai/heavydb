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

#include "../ThriftHandler/TokenCompletionHints.h"

#include <gtest/gtest.h>

TEST(FindLastWord, SimpleId) {
  std::string partial_query{"SELECT test"};
  ASSERT_EQ("test", find_last_word_from_cursor(partial_query, partial_query.size()));
}

TEST(FindLastWord, QualifiedId) {
  std::string partial_query{"SELECT test.x"};
  ASSERT_EQ("test.x", find_last_word_from_cursor(partial_query, partial_query.size()));
}

TEST(FindLastWord, CursorPastEnd) {
  std::string partial_query{"SELECT test"};
  ASSERT_EQ("", find_last_word_from_cursor(partial_query, partial_query.size() + 1));
}

TEST(FindLastWord, CursorInside) {
  std::string partial_query{"SELECT str FROM te LIMIT 10"};
  ASSERT_EQ("te", find_last_word_from_cursor(partial_query, 18));
}

TEST(FindLastWord, EmptyString) {
  ASSERT_EQ("", find_last_word_from_cursor("", 0));
  ASSERT_EQ("", find_last_word_from_cursor("", 1));
  ASSERT_EQ("", find_last_word_from_cursor("", -1));
}

namespace {

void assert_set_equals(const std::vector<std::string>& expected, const std::vector<std::string>& actual) {
  auto actual_sorted = actual;
  std::sort(actual_sorted.begin(), actual_sorted.end());
  auto expected_sorted = expected;
  std::sort(expected_sorted.begin(), expected_sorted.end());
  ASSERT_EQ(expected_sorted, actual_sorted);
}

}  // namespace

TEST(Completion, QualifiedColumnName) {
  std::unordered_map<std::string, std::unordered_set<std::string>> column_names_by_table;
  column_names_by_table["test"] = {"x", "ss", "str"};
  column_names_by_table["test_inner"] = {"x"};
  {
    std::vector<TCompletionHint> completion_hints;
    std::string last_word{"test.x"};
    ASSERT_TRUE(get_qualified_column_hints(completion_hints, last_word, column_names_by_table));
    ASSERT_EQ(size_t(1), completion_hints.size());
    ASSERT_TRUE(TCompletionHintType::COLUMN == completion_hints.front().type);
    ASSERT_EQ(last_word, completion_hints.front().replaced);
    assert_set_equals({"test.x"}, completion_hints.front().hints);
  }
  {
    std::vector<TCompletionHint> completion_hints;
    std::string last_word{"test.s"};
    ASSERT_TRUE(get_qualified_column_hints(completion_hints, last_word, column_names_by_table));
    ASSERT_EQ(size_t(1), completion_hints.size());
    ASSERT_TRUE(TCompletionHintType::COLUMN == completion_hints.front().type);
    ASSERT_EQ(last_word, completion_hints.front().replaced);
    assert_set_equals({"test.ss", "test.str"}, completion_hints.front().hints);
  }
  {
    std::vector<TCompletionHint> completion_hints;
    std::string last_word{"test."};
    ASSERT_TRUE(get_qualified_column_hints(completion_hints, last_word, column_names_by_table));
    ASSERT_EQ(size_t(1), completion_hints.size());
    ASSERT_TRUE(TCompletionHintType::COLUMN == completion_hints.front().type);
    ASSERT_EQ(last_word, completion_hints.front().replaced);
    assert_set_equals({"test.x", "test.ss", "test.str"}, completion_hints.front().hints);
  }
  {
    std::vector<TCompletionHint> completion_hints;
    ASSERT_TRUE(get_qualified_column_hints(completion_hints, "test.y", column_names_by_table));
    ASSERT_TRUE(completion_hints.empty());
  }
}

TEST(Completion, ColumnName) {
  std::unordered_map<std::string, std::unordered_set<std::string>> column_names_by_table;
  column_names_by_table["test"] = {"x", "ss", "str"};
  column_names_by_table["test_inner"] = {"x"};
  {
    std::vector<TCompletionHint> completion_hints;
    std::string last_word{"s"};
    get_column_hints(completion_hints, last_word, column_names_by_table);
    ASSERT_EQ(size_t(1), completion_hints.size());
    ASSERT_TRUE(TCompletionHintType::COLUMN == completion_hints.front().type);
    ASSERT_EQ(last_word, completion_hints.front().replaced);
    assert_set_equals({"ss", "str"}, completion_hints.front().hints);
  }
  {
    std::vector<TCompletionHint> completion_hints;
    std::string last_word{"x"};
    get_column_hints(completion_hints, last_word, column_names_by_table);
    ASSERT_EQ(size_t(1), completion_hints.size());
    ASSERT_TRUE(TCompletionHintType::COLUMN == completion_hints.front().type);
    ASSERT_EQ(last_word, completion_hints.front().replaced);
    assert_set_equals({"x"}, completion_hints.front().hints);
  }
  {
    std::vector<TCompletionHint> completion_hints;
    get_column_hints(completion_hints, "y", column_names_by_table);
    ASSERT_TRUE(completion_hints.empty());
  }
}

TEST(Completion, FilterKeywords) {
  std::vector<TCompletionHint> original_hints;
  std::vector<TCompletionHint> expected_filtered_hints;
  {
    TCompletionHint hint;
    hint.type = TCompletionHintType::COLUMN;
    hint.hints.emplace_back("foo");
    original_hints.push_back(hint);
    expected_filtered_hints.push_back(hint);
  }
  {
    TCompletionHint hint;
    hint.type = TCompletionHintType::KEYWORD;
    hint.hints.emplace_back("SUBSTR");
    original_hints.push_back(hint);
  }
  {
    TCompletionHint hint;
    hint.type = TCompletionHintType::KEYWORD;
    hint.hints.emplace_back("GROUP");
    original_hints.push_back(hint);
    expected_filtered_hints.push_back(hint);
  }
  {
    TCompletionHint hint;
    hint.type = TCompletionHintType::KEYWORD;
    hint.hints.emplace_back("ON");
    original_hints.push_back(hint);
    expected_filtered_hints.push_back(hint);
  }
  {
    TCompletionHint hint;
    hint.type = TCompletionHintType::KEYWORD;
    hint.hints.emplace_back("OUTER");
    original_hints.push_back(hint);
    expected_filtered_hints.push_back(hint);
  }
  const auto filtered_hints = just_whitelisted_keyword_hints(original_hints);
  ASSERT_EQ(expected_filtered_hints, filtered_hints);
}

TEST(Completion, ShouldSuggestColumnHints) {
  ASSERT_TRUE(should_suggest_column_hints("SELECT x"));
  ASSERT_FALSE(should_suggest_column_hints("SELECT x "));
  ASSERT_TRUE(should_suggest_column_hints("SELECT x,"));
  ASSERT_TRUE(should_suggest_column_hints("SELECT x , "));
  ASSERT_TRUE(should_suggest_column_hints("SELECT x, y"));
  ASSERT_FALSE(should_suggest_column_hints("SELECT x, y "));
  ASSERT_TRUE(should_suggest_column_hints("SELECT x, y,"));
  ASSERT_TRUE(should_suggest_column_hints("SELECT x, y , "));
  ASSERT_TRUE(should_suggest_column_hints("SELECT "));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
