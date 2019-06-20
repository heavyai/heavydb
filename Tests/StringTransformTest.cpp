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
#include "TestHelpers.h"

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

TEST(StringTransform, HideSensitiveDataFromQuery) {
  std::vector<std::pair<std::string, std::string>> const tests{
      {"COPY testtable FROM 's3://example/*' WITH (header='true', geo='true', "
       "s3_region='us-west-1', "
       "s3_access_key='HelloWorldAccessKeys',s3_secret_key='abcxyz');",
       "COPY testtable FROM 's3://example/*' WITH (header='true', geo='true', "
       "s3_region='us-west-1', s3_access_key='XXXXXXXX',s3_secret_key='XXXXXXXX');"},
      {"CREATE USER jason (password = 'OmniSciRocks!', is_super = 'true')",
       "CREATE USER jason (password = 'XXXXXXXX', is_super = 'true')"},
      {"ALTER USER omnisci (password = 'OmniSciIsFast!')",
       "ALTER USER omnisci (password = 'XXXXXXXX')"},
      {"ALTER USER jason (is_super = 'false', password = 'SilkySmooth')",
       "ALTER USER jason (is_super = 'false', password = 'XXXXXXXX')"},
      {"ALTER USER omnisci (password = 'short')",
       "ALTER USER omnisci (password = 'XXXXXXXX')"},
      {"ALTER USER omnisci (password='short', future_parameter = 3)",
       "ALTER USER omnisci (password='XXXXXXXX', future_parameter = 3)"},
      {"CREATE USER jason (password = 'OmniSciRocks!', is_super = 'true'); CREATE "
       "USER omnisci (password = 'OmniSciIsFast!')",
       "CREATE USER jason (password = 'XXXXXXXX', is_super = 'true'); CREATE USER "
       "omnisci (password = 'XXXXXXXX')"},
      {"\\set_license DONTSHOWTHISSTRING", "\\set_license XXXXXXXX"},
      {"   \\set_license 'DONTSHOWTHISSTRING';", "   \\set_license XXXXXXXX"}};
  for (auto const& test : tests) {
    std::string const safe = hide_sensitive_data_from_query(test.first);
    ASSERT_EQ(safe, test.second);
  }
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
