// Copyright (c) 2021 OmniSci, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include "Logger/Logger.h"
#include "Shared/json.h"
#include "TestHelpers.h"

TEST(JSON, Types) {
  //////////

  std::string text1 = R"(
    false
  )";

  JSON json1(text1);
  EXPECT_EQ(json1.getType(), "False");
  EXPECT_EQ((bool)json1, false);

  //////////

  std::string text2 = R"(
    true
  )";

  JSON json2(text2);
  EXPECT_EQ(json2.getType(), "True");
  EXPECT_EQ((bool)json2, true);

  //////////

  std::string text3 = R"(
    12345
  )";

  JSON json3(text3);
  EXPECT_EQ(json3.getType(), "Number");
  EXPECT_EQ((uint64_t)json3, 12345U);

  //////////

  std::string text4 = R"(
    "somestring"
  )";

  JSON json4(text4);
  EXPECT_EQ(json4.getType(), "String");
  EXPECT_EQ((std::string)json4, "somestring");
}

TEST(JSON, Objects) {
  //////////
  std::string text1 = R"({
    "a": 1,
    "b": 2,
    "c": 3
  })";

  JSON json1(text1);
  EXPECT_EQ((size_t)json1["a"], 1U);
  EXPECT_EQ((size_t)json1["b"], 2U);
  EXPECT_EQ((size_t)json1["c"], 3U);

  //////////
  std::string text2 = R"text2(
    {"menu": {
      "id": "file",
      "value": "File",
      "popup": {
        "menuitem": [
          {"value": "New", "onclick": "CreateNewDoc()"},
          {"value": "Open", "onclick": "OpenDoc()"},
          {"value": "Close", "onclick": "CloseDoc()"}
        ]
      }
    }}
  )text2";

  JSON json2(text2);
  EXPECT_EQ((std::string)json2["menu"]["id"], "file");
  EXPECT_EQ(json2["menu"]["popup"]["menuitem"].getType(), "Array");
  EXPECT_EQ((std::string)json2["menu"]["popup"]["menuitem"][2]["value"], "Close");

  //////////
  JSON json3;
  json3["asdf"] = "1234";
  EXPECT_EQ(std::string(json3["asdf"]), "1234");
  EXPECT_EQ(json3.str(), "{\"asdf\":\"1234\"}");
}

TEST(JSON, Errors) {
  // empty input
  JSON json1;
  EXPECT_THROW(json1.parse(""), std::runtime_error);

  // missing commas
  std::string text2 = R"({
    "a": 1
    "b": 2
    "c": 3
  })";
  JSON json2;
  EXPECT_THROW(json2.parse(text2), std::runtime_error);
}

TEST(JSON, Copying) {
  JSON json1("{\"asdf\":\"1234\"}");

  JSON json2(json1);
  EXPECT_EQ((std::string)json2["asdf"], "1234");

  json2["asdf"] = JSON("[1, 2, 3]");
  EXPECT_EQ((int32_t)json2["asdf"][0], 1);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
