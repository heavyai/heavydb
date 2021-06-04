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
#include <string>
#include "Logger/Logger.h"
#include "Shared/json.h"
#include "TestHelpers.h"

using JSON = omnisci::JSON;

TEST(JSON, Types) {
  //////////

  std::string text1 = R"(
    false
  )";

  JSON json1(text1);
  EXPECT_TRUE(json1.isBoolean());
  EXPECT_EQ(static_cast<bool>(json1), false);

  //////////

  std::string text2 = R"(
    true
  )";

  JSON json2(text2);
  EXPECT_TRUE(json2.isBoolean());
  EXPECT_EQ(static_cast<bool>(json2), true);

  //////////

  std::string text3 = R"(
    12345
  )";

  JSON json3(text3);
  EXPECT_TRUE(json3.isNumber());
  EXPECT_EQ(static_cast<uint64_t>(json3), 12345U);

  //////////

  std::string text4 = R"(
    12345.67890
  )";

  JSON json4(text4);
  EXPECT_TRUE(json4.isNumber());
  EXPECT_FLOAT_EQ(static_cast<float>(json4), 12345.67890F);
  EXPECT_DOUBLE_EQ(static_cast<double>(json4), 12345.67890);

  //////////

  std::string text5 = R"(
    "somestring"
  )";

  JSON json5(text5);
  EXPECT_TRUE(json5.isString());
  EXPECT_EQ(static_cast<std::string>(json5), "somestring");
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
  EXPECT_EQ(json2["menu"]["popup"]["menuitem"].isArray(), true);
  EXPECT_EQ((std::string)json2["menu"]["popup"]["menuitem"][2]["value"], "Close");
  EXPECT_TRUE(json2.hasMember("menu"));
  EXPECT_FALSE(json2.hasMember("notamember"));
  EXPECT_TRUE(json2["menu"].hasMember("id"));
  EXPECT_TRUE(json2["menu"].hasMember("value"));
  EXPECT_TRUE(json2["menu"].hasMember("popup"));
  EXPECT_TRUE(json2["menu"]["popup"].hasMember("menuitem"));
  EXPECT_FALSE(json2["menu"]["popup"].hasMember("notamember"));

  //////////
  JSON json3;
  json3["asdf"] = "1234";
  EXPECT_EQ(std::string(json3["asdf"]), "1234");
  EXPECT_EQ(json3.stringify(), "{\"asdf\":\"1234\"}");
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

TEST(JSON, Comparisons) {
  using namespace std::string_literals;

  // strings: JSON vs C-style string literal
  JSON testString1("\"asdf\"");
  EXPECT_TRUE(testString1 == "asdf");
  EXPECT_TRUE("asdf" == testString1);
  EXPECT_TRUE(testString1 != "zxcv");
  EXPECT_TRUE("zxcv" != testString1);

  // strings: JSON vs std::string
  EXPECT_TRUE(testString1 == "asdf"s);
  EXPECT_TRUE("asdf"s == testString1);
  EXPECT_TRUE(testString1 != "zxcv"s);
  EXPECT_TRUE("zxcv"s != testString1);

  // strings: JSON vs JSON
  JSON testString2("\"asdf\"");
  JSON testString3("\"zxcv\"");
  EXPECT_TRUE(testString1 == testString2);
  EXPECT_TRUE(testString2 == testString1);
  EXPECT_TRUE(testString1 != testString3);
  EXPECT_TRUE(testString3 != testString1);

  // integers: JSON vs int literals
  JSON testInteger1("1234");
  EXPECT_TRUE(testInteger1 == 1234);
  EXPECT_TRUE(1234 == testInteger1);
  EXPECT_TRUE(testInteger1 != 5678);
  EXPECT_TRUE(5678 != testInteger1);

  // integers: JSON vs JSON
  JSON testInteger2("1234");
  JSON testInteger3("5678");
  EXPECT_TRUE(testInteger1 == testInteger2);
  EXPECT_TRUE(testInteger2 == testInteger1);
  EXPECT_TRUE(testInteger1 != testInteger3);
  EXPECT_TRUE(testInteger3 != testInteger1);

  // booleans: JSON vs bool literal
  JSON testBoolean1("true");
  EXPECT_TRUE(testBoolean1 == true);
  EXPECT_TRUE(true == testBoolean1);
  EXPECT_TRUE(testBoolean1 != false);
  EXPECT_TRUE(false != testBoolean1);

  // booleans: JSON vs JSON
  JSON testBoolean2("true");
  JSON testBoolean3("false");
  EXPECT_TRUE(testBoolean1 == testBoolean2);
  EXPECT_TRUE(testBoolean2 == testBoolean1);
  EXPECT_TRUE(testBoolean1 != testBoolean3);
  EXPECT_TRUE(testBoolean3 != testBoolean1);

  // type mismatch
  JSON testString4("\"1234\"");
  JSON testString5("\"true\"");
  EXPECT_FALSE(testString4 == testInteger1);
  EXPECT_FALSE(testString5 == testBoolean1);
}

TEST(JSON, IfStatements) {
  JSON json("{\"enabled\":true,\"driver_type\":\"Vulkan\"}");
  if (json["enabled"] != true) {
    EXPECT_TRUE(false);
  } else {
    EXPECT_TRUE(true);
  }
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
