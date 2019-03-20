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

#include "../Shared/StringTransform.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(StringTransform, CsvQuote) {
  std::vector<std::pair<std::string, std::string>> qa_pairs{
      {"", "\"\""},
      {"h", "\"h\""},
      {"hi", "\"hi\""},
      {"hi\\!", "\"hi\\!\""},
      {"\"", "\"\"\"\""},
      {"\"\"", "\"\"\"\"\"\""},
      {"\"h\"", "\"\"\"h\"\"\""},
      {"\"hi\"", "\"\"\"hi\"\"\""},
      {"\"hi\\!\"", "\"\"\"hi\\!\"\"\""},
      {"(5'11\")", "\"(5'11\"\")\""}};
  for (auto& test : qa_pairs) {
    std::ostringstream oss;
    oss << std::quoted(test.first, '"', '"');
    ASSERT_EQ(oss.str(), test.second);
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
