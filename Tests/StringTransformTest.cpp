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

#include <type_traits>
#include "../Shared/StringTransform.h"
#include "../Shared/toString.h"
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

const std::vector<std::string> case_test_strings(
    {"",
     "ABCDEFG",
     "abcdefg",
     "aBcDeFg",
     "(aBcDeFg)",
     "aBcDeFg,aBcDeFg,aBcDeFg",
     "Four score and seven years ago our fathers brought forth on this continent, a new "
     "nation, conceived in Liberty, and dedicated to the proposition that all men are "
     "created equal. Now we are engaged in a great civil war, testing whether that "
     "nation, or any nation so conceived and so dedicated, can long endure. We are met "
     "on a great battle-field of that war. We have come to dedicate a portion of that "
     "field, as a final resting place for those who here gave their lives that that "
     "nation might live. It is altogether fitting and proper that we should do this. "
     "But, in a larger sense, we can not dedicate—we can not consecrate—we can not "
     "hallow—this ground. The brave men, living and dead, who struggled here, have "
     "consecrated it, far above our poor power to add or detract. The world will little "
     "note, nor long remember what we say here, but it can never forget what they did "
     "here. It is for us the living, rather, to be dedicated here to the unfinished work "
     "which they who fought here have thus far so nobly advanced. It is rather for us to "
     "be here dedicated to the great task remaining before us—that from these honored "
     "dead we take increased devotion to that cause for which they gave the last full "
     "measure of devotion—that we here highly resolve that these dead shall not have "
     "died in vain—that this nation, under God, shall have a new birth of freedom—and "
     "that government of the people, by the people, for the people, shall not perish "
     "from the earth. —Abraham Lincoln"});

TEST(StringTransform, ToUpper) {
  for (const auto& txt : case_test_strings) {
    std::string txt1 = to_upper(txt);
    std::string txt2 = boost::algorithm::to_upper_copy(txt);
    ASSERT_EQ(txt1, txt2);
  }
}

TEST(StringTransform, ToLower) {
  for (const std::string& txt : case_test_strings) {
    std::string txt1 = to_lower(txt);
    std::string txt2 = boost::algorithm::to_lower_copy(txt);
    ASSERT_EQ(txt1, txt2);
  }
}

const std::vector<std::string> split_test_strings({"",
                                                   "aBcDeFg",
                                                   "aBcDeFg,aBcDeFg,aBcDeFg",
                                                   "0 1 2 3 4 5 6 7 8 9",
                                                   "0  1  2  3  4  5  6  7  8  9",
                                                   " 0 1 2 3 4 5 6 7 8 9 ",
                                                   " 0  1  2  3  4  5  6  7  8  9 ",
                                                   "    0  1  2  3  4  5  6  7  8  9    ",
                                                   "        "});

TEST(StringTransform, Split) {
  {
    auto v = split(split_test_strings[0], ",");
    ASSERT_EQ(v.size(), 1U);
    ASSERT_EQ(v[0], "");
  }

  {
    auto v = split(split_test_strings[0]);
    ASSERT_EQ(v.size(), 0U);
  }

  {
    auto v = split(split_test_strings[1], ",");
    ASSERT_EQ(v.size(), 1U);
    ASSERT_EQ(v[0], "aBcDeFg");
  }

  {
    auto v = split(split_test_strings[2], ",");
    ASSERT_EQ(v.size(), 3U);
    ASSERT_EQ(v[0], "aBcDeFg");
    ASSERT_EQ(v[1], "aBcDeFg");
    ASSERT_EQ(v[2], "aBcDeFg");
  }

  for (size_t i = 3; i <= 7; ++i) {
    auto v = split(split_test_strings[i]);
    ASSERT_EQ(v.size(), 10U);
    ASSERT_EQ(v[0], "0");
    ASSERT_EQ(v[1], "1");
    ASSERT_EQ(v[2], "2");
    ASSERT_EQ(v[3], "3");
    ASSERT_EQ(v[4], "4");
    ASSERT_EQ(v[5], "5");
    ASSERT_EQ(v[6], "6");
    ASSERT_EQ(v[7], "7");
    ASSERT_EQ(v[8], "8");
    ASSERT_EQ(v[9], "9");
  }

  {
    auto v = split(split_test_strings[8]);
    ASSERT_EQ(v.size(), 0U);
  }
}

TEST(StringTransform, toString) {
  {
    ASSERT_EQ(::toString(123), "123");
    ASSERT_EQ(::toString(12.3f), "12.300000");
    ASSERT_EQ(::toString(std::string("123")), "\"123\"");
    {
      std::vector<int> v = {1, 2, 3};
      ASSERT_EQ(::toString(v), "[1, 2, 3]");
    }
    {
      std::vector<std::string> v = {"1", "2", "3"};
      ASSERT_EQ(::toString(v), "[\"1\", \"2\", \"3\"]");
    }
  }

  {
    class A {
     public:
    };
    A a;
    ASSERT_EQ(::toString(a), "StringTransform_toString_Test::TestBody()::A");
    ASSERT_EQ(::toString(&a), "&StringTransform_toString_Test::TestBody()::A");
  }

  {
    class A {
     public:
      std::string toString() const { return "A"; }
    };
    A a;
    ASSERT_EQ(::toString(a), "A");
    ASSERT_EQ(::toString(&a), "&A");
  }

  {
    class A {
     public:
      virtual std::string toString() const = 0;
    };

    class A1 : public A {
     public:
      std::string toString() const override { return "A1"; };

      virtual ~A1() {}
    };

    class A2 : public A1 {
     public:
      std::string toString() const override { return "A2"; };

      virtual ~A2() {}
    };

    A1 a1;
    ASSERT_EQ(::toString(a1), "A1");

    A1* p = &a1;
    ASSERT_EQ(::toString(p), "&A1");
    ASSERT_EQ(::toString(&p), "&&A1");

    A* p1 = &a1;
    ASSERT_EQ(::toString(p1), "&A1");

    A2 a2;
    ASSERT_EQ(::toString(a2), "A2");
    {
      std::vector<const A*> v = {&a1, &a2};
      ASSERT_EQ(::toString(v), "[&A1, &A2]");
      ASSERT_EQ(::toString(&v), "&[&A1, &A2]");
    }
    {
      auto a1 = std::make_shared<A1>();
      auto a2 = std::make_shared<A2>();
      std::vector<std::shared_ptr<const A>> v = {a1, a2};
      ASSERT_EQ(::toString(v), "[A1, A2]");
      ASSERT_EQ(::toString(&v), "&[A1, A2]");
    }
    {
      auto a1 = std::make_shared<A1>();
      std::vector<std::shared_ptr<const A1>> v = {a1};
      ASSERT_EQ(::toString(v), "[A1]");
      ASSERT_EQ(::toString(&v), "&[A1]");
    }
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
