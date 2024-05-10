/*
 * Copyright 2022 HEAVY.AI, Inc.
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

/**
 * @file StringFunctionsTest.cpp
 * @brief Test suite for string functions
 *
 */

#include <gtest/gtest.h>
#include <sstream>

#include "DBHandlerTestHelpers.h"
#include "Shared/scope.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace TestHelpers;

extern bool g_enable_string_functions;
extern bool g_enable_watchdog;
extern size_t g_watchdog_none_encoded_string_translation_limit;
extern std::string g_heavyiq_url;

// begin string function tests

/**
 * @brief Class used for setting up and tearing down tables and records that are required
 * by the string function test cases
 */
class StringFunctionTest : public DBHandlerTestFixture,
                           public ::testing::WithParamInterface<TExecuteMode::type> {
public:
  static void SetUpTestSuite() {
    const std::vector<std::string> setup_commands{
        "drop table if exists string_function_test_people;",

        "create table string_function_test_people(id int, first_name text, last_name "
        "text encoding none, full_name text, age integer, country_code text, "
        "us_phone_number text, zip_plus_4 text, personal_motto text, raw_email text);",

        "insert into string_function_test_people values(1, 'JOHN', 'SMITH', 'John "
        "SMITH', 25, 'us', '555-803-2144', '90210-7743', 'All for one and one for "
        "all.', 'Shoot me a note at therealjohnsmith@omnisci.com');",

        "insert into string_function_test_people values(2, 'John', 'Banks', 'John "
        "BANKS', 30, 'Us', '555-803-8244', '94104-8123', 'One plus one does not equal "
        "two.', 'Email: john_banks@mapd.com');",

        "insert into string_function_test_people values(3, 'JOHN', 'Wilson', 'John "
        "WILSON', 20, 'cA', '555-614-9814', null, 'What is the sound of one hand "
        "clapping?', 'JOHN.WILSON@geops.net');",

        "insert into string_function_test_people values(4, 'Sue', 'Smith', 'Sue "
        "SMITH', 25, 'CA', '555-614-2282', null, 'Nothing exists entirely alone. "
        "Everything is always in relation to everything else.', 'Find me at "
        "sue4tw@example.com, or reach me at sue.smith@example.com. I''d love to hear "
        "from you!');",

        "drop table if exists string_function_test_countries;",

        "create table string_function_test_countries(id int, code text, arrow_code "
        "text, name text, short_name text encoding none, capital text, capital_none "
        "text encoding none, largest_city text encoding none, lang text encoding none, "
        "json_data_none text encoding none);",

        "insert into string_function_test_countries values(1, 'US', '>>US<<', 'United "
        "States', null, 'Washington', 'Washington', 'New York City', 'en', "
        "'{\"capital\": \"Washington D.C.\", \"pop\": 329500000, \"independence_day\": "
        "\"1776-07-04\",  \"has_prime_minister\": false, \"prime_minister\": null, "
        "\"factoids\": {\"gdp_per_cap_2015_2020\": [56863, 58021, 60110, 63064, 65280, "
        "63544], \"Last 3 leaders\": [\"Barack Obama\", \"Donald Trump\", \"Joseph "
        "Biden\"], \"most_valuable_crop\": \"corn\"}}');",

        "insert into string_function_test_countries values(2, 'ca', '>>CA<<', "
        "'Canada', 'Canada', 'Ottawa', 'Ottawa', 'TORONTO', 'EN', '{\"capital\": "
        "\"Toronto\", \"pop\": 38010000, \"independence_day\": \"07/01/1867\", "
        "\"exchange_rate_usd\": \"0.78125\", \"has_prime_minister\": true, "
        "\"prime_minister\": \"Justin Trudeau\", \"factoids\": "
        "{\"gdp_per_cap_2015_2020\": [43596, 42316, 45129, 46454, 46327, 43242], "
        "\"Last 3 leaders\": [\"Paul Martin\", \"Stephen Harper\", \"Justin "
        "Trudeau\"], \"most valuable crop\": \"wheat\"}}');",

        "insert into string_function_test_countries values(3, 'Gb', '>>GB<<', 'United "
        "Kingdom', 'UK', 'London', 'London', 'LONDON', 'en', '{\"capital\": "
        "\"London\", \"pop\": 67220000, \"independence_day\": \"N/A\", "
        "\"exchange_rate_usd\": 1.21875, \"prime_minister\": \"Boris Johnson\", "
        "\"has_prime_minister\": true, \"factoids\": {\"gdp_per_cap_2015_2020\": "
        "[45039, 41048, 40306, 42996, 42354, 40285], \"most_valuable_crop\": "
        "\"wheat\"}}');",

        "insert into string_function_test_countries values(4, 'dE', '>>DE<<', "
        "'Germany', 'Germany', 'Berlin', 'Berlin', 'Berlin', 'de', "
        "'{\"capital\":\"Berlin\", \"independence_day\": \"1990-10-03\", "
        "\"exchange_rate_usd\": 1.015625, \"has_prime_minister\": false, "
        "\"prime_minister\": null, \"factoids\": {\"gdp_per_cap_2015_2020\": [41103, "
        "42136, 44453, 47811, 46468, 45724], \"most_valuable_crop\": \"wheat\"}}');",

        "drop table if exists numeric_to_string_test;",

        "create table numeric_to_string_test(b boolean, ti tinyint, si smallint, i "
        "int, bi bigint, flt float, dbl double, dec_5_2 decimal(5, 2), dec_18_10 "
        "decimal(18, 10), dt date, ts_0 timestamp(0), ts_3 timestamp(3), tm time, "
        "b_str text, ti_str text, si_str text, i_str text, bi_str text, flt_str text, "
        "dbl_str text, dec_5_2_str text, dec_18_10_str text, dt_str text, ts_0_str "
        "text, ts_3_str text, tm_str text) with (fragment_size=2);",

        "insert into numeric_to_string_test values (true, 21, 21, 21, 21, 1.25, 1.25, "
        "1.25, 1.25, '2013-09-10', '2013-09-10 12:43:23', '2013-09-10 12:43:23.123', "
        "'12:43:23', 'true', '21', '21', '21', '21', '1.250000', '1.250000', ' 1.25', "
        "'      1.2500000000', '2013-09-10', '2013-09-10 12:43:23', '2013-09-10 "
        "12:43:23.123', '12:43:23');",

        "insert into numeric_to_string_test values (false, 127, 32627, 2147483647, "
        "9223372036854775807,  0.78125, 0.78125, 123.45, 12345678.90123456789, "
        "'2013-09-11', '2013-09-11 12:43:23', '2013-09-11 12:43:23.123', '00:43:23', "
        "'false', '127', '32627', '2147483647', '9223372036854775807', '0.781250', "
        "'0.781250', '123.45', '12345678.9012345672', '2013-09-11', '2013-09-11 "
        "12:43:23', '2013-09-11 12:43:23.123', '00:43:23');",

        "insert into numeric_to_string_test values (null, null, null, null,  null, "
        "null, null, null, null, null, null, null, null, null, null, null, null, null, "
        "null, null, null, null, null, null, null, null);",

        "drop table if exists text_enc_test;",

        "create table text_enc_test (name text encoding none, short_name text encoding "
        "dict(32), code text encoding dict(32));",

        "insert into text_enc_test values ('United States', 'USA', '>>US USA<<');",

        "insert into text_enc_test values ('Canada', 'Canada', '>>CA CAN<<');",

        "insert into text_enc_test values ('United Kingdom', 'UK', '>>GB GBR<<');",

        "insert into text_enc_test values ('Germany', 'Germany', '>>DE DEN<<');"};
    executeCommands(setup_commands);
  }

  static void TearDownTestSuite() {
    const std::vector<std::string> teardown_commands{
        "drop table string_function_test_people;",
        "drop table string_function_test_countries;",
        "drop table numeric_to_string_test;",
        "drop table text_enc_test;"};
    executeCommands(teardown_commands);
  }

 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    auto device_type = GetParam();
    if (!setExecuteMode(device_type)) {
      GTEST_SKIP() << device_type << " is not enabled.";
    }
  }

 private:
  static void executeCommands(const std::vector<std::string>& commands) {
    for (const auto& command : commands) {
      sql(command);
    }
  }
};

TEST_P(StringFunctionTest, Lowercase) {
  sqlAndCompareResult(
      "select lower(first_name) from string_function_test_people order by id asc;",
      {{"john"}, {"john"}, {"john"}, {"sue"}});
}

TEST_P(StringFunctionTest, LowercaseLiteral) {
  sqlAndCompareResult("select lower('fUnNy CaSe');", {{"funny case"}});
}

TEST_P(StringFunctionTest, Uppercase) {
  sqlAndCompareResult(
      "select upper(first_name) from string_function_test_people order by id asc;",
      {{"JOHN"}, {"JOHN"}, {"JOHN"}, {"SUE"}});
}

TEST_P(StringFunctionTest, UppercaseLiteral) {
  sqlAndCompareResult("select upper('fUnNy CaSe');", {{"FUNNY CASE"}});
}

TEST_P(StringFunctionTest, InitCap) {
  sqlAndCompareResult(
      "select initcap(full_name) from string_function_test_people order by id asc",
      {{"John Smith"}, {"John Banks"}, {"John Wilson"}, {"Sue Smith"}});
}

TEST_P(StringFunctionTest, InitCapLiteral) {
  sqlAndCompareResult("select initcap('fUnNy CaSe');", {{"Funny Case"}});
}

TEST_P(StringFunctionTest, Reverse) {
  sqlAndCompareResult(
      "select reverse(full_name) from string_function_test_people order by id asc;",
      {{"HTIMS nhoJ"}, {"SKNAB nhoJ"}, {"NOSLIW nhoJ"}, {"HTIMS euS"}});
}

TEST_P(StringFunctionTest, ReverseLiteral) {
  sqlAndCompareResult("select reverse('fUnNy CaSe');", {{"eSaC yNnUf"}});
}

TEST_P(StringFunctionTest, Repeat) {
  sqlAndCompareResult(
      "select repeat(full_name, 2) from string_function_test_people order by id asc;",
      {{"John SMITHJohn SMITH"},
       {"John BANKSJohn BANKS"},
       {"John WILSONJohn WILSON"},
       {"Sue SMITHSue SMITH"}});
}

TEST_P(StringFunctionTest, RepeatLiteral) {
  sqlAndCompareResult("select repeat('fUnNy CaSe', 3);",
                      {{"fUnNy CaSefUnNy CaSefUnNy CaSe"}});
}

TEST_P(StringFunctionTest, Concat) {
  sqlAndCompareResult(
      "select name || ', Earth' from string_function_test_countries order by id asc;",
      {{"United States, Earth"},
       {"Canada, Earth"},
       {"United Kingdom, Earth"},
       {"Germany, Earth"}});
}

TEST_P(StringFunctionTest, ReverseConcat) {
  sqlAndCompareResult(
      "select 'Country: ' || code from string_function_test_countries order by id asc;",
      {{"Country: US"}, {"Country: ca"}, {"Country: Gb"}, {"Country: dE"}});
}

TEST_P(StringFunctionTest, ConcatLiteral) {
  sqlAndCompareResult("select 'fUnNy CaSe' || ' is the case.';",
                      {{"fUnNy CaSe is the case."}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_DictEncodedAndNoneEncoded) {
  sqlAndCompareResult(
      "select first_name || last_name from string_function_test_people order by "
      "id asc;",
      {{"JOHNSMITH"}, {"JohnBanks"}, {"JOHNWilson"}, {"SueSmith"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_NoneEncodedAndDictEncoded) {
  sqlAndCompareResult(
      "select last_name || first_name from string_function_test_people order by "
      "id asc;",
      {{"SMITHJOHN"}, {"BanksJohn"}, {"WilsonJOHN"}, {"SmithSue"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_DictEncodedAndLiteralAndNoneEncoded) {
  sqlAndCompareResult(
      "select first_name || ' ' || last_name from string_function_test_people "
      "order by id asc;",
      {{"JOHN SMITH"}, {"John Banks"}, {"JOHN Wilson"}, {"Sue Smith"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_NoneEncodedAndLiteralAndDictEncoded) {
  sqlAndCompareResult(
      "select last_name || ', ' || first_name from string_function_test_people "
      "order by id asc;",
      {{"SMITH, JOHN"}, {"Banks, John"}, {"Wilson, JOHN"}, {"Smith, Sue"}});
}

TEST_P(StringFunctionTest,
       ConcatTwoVarArg_StringOpDictEncodedAndLiteralAndStringOpNoneEncoded) {
  sqlAndCompareResult(
      "select UPPER(first_name) || ' ' || UPPER(last_name) from "
      "string_function_test_people order by id asc;",
      {{"JOHN SMITH"}, {"JOHN BANKS"}, {"JOHN WILSON"}, {"SUE SMITH"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_StringOpAndLiteral) {
  // StringOp(None-encoded) || literal || StringOp(dict-encoded)
  sqlAndCompareResult(
      "select UPPER(last_name) || ', ' || UPPER(first_name) from "
      "string_function_test_people order by id asc;",
      {{"SMITH, JOHN"}, {"BANKS, JOHN"}, {"WILSON, JOHN"}, {"SMITH, SUE"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_StringOpAndLiteralAndNoneEncodedColumn) {
  // StringOp(dict-encoded || literal || none-encoded)
  sqlAndCompareResult(
      "select INITCAP(first_name || ' ' || last_name) from "
      "string_function_test_people order by id asc;",
      {{"John Smith"}, {"John Banks"}, {"John Wilson"}, {"Sue Smith"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_StringOpAndLiteralAndDictEncodedColumn) {
  // StringOp(none-encoded || literal || dict-encoded)
  sqlAndCompareResult(
      "select INITCAP(last_name || ', ' || first_name) from "
      "string_function_test_people order by id asc;",
      {{"Smith, John"}, {"Banks, John"}, {"Wilson, John"}, {"Smith, Sue"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_DictEncodedColumnsAndLiterals) {
  // dict_encoded || literal || none-encoded || literal || cast(numeric to text) ||
  // literal
  sqlAndCompareResult(
      "select first_name || ' ' || last_name || ' (' || age || ')' from "
      "string_function_test_people order by id asc;",
      {{"JOHN SMITH (25)"},
       {"John Banks (30)"},
       {"JOHN Wilson (20)"},
       {"Sue Smith (25)"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_DictEncodedColumns) {
  // Concat two dictionary encoded text columns
  sqlAndCompareResult(
      "select full_name || ' (' || us_phone_number || ')' from "
      "string_function_test_people order by id asc;",
      {{"John SMITH (555-803-2144)"},
       {"John BANKS (555-803-8244)"},
       {"John WILSON (555-614-9814)"},
       {"Sue SMITH (555-614-2282)"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_MultipleStringOps) {
  // Concat two dictionary encoded text columns with string op
  sqlAndCompareResult(
      "select UPPER(full_name) || ' (' || REPLACE(us_phone_number, '-', '.') || "
      "')' from string_function_test_people order by id asc;",
      {{"JOHN SMITH (555.803.2144)"},
       {"JOHN BANKS (555.803.8244)"},
       {"JOHN WILSON (555.614.9814)"},
       {"SUE SMITH (555.614.2282)"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_NestedStringOp) {
  // Outer string op on concatenation of two dictionary encoded text columns
  sqlAndCompareResult(
      "select INITCAP(full_name || ' (' || REPLACE(us_phone_number, '-', '.') || "
      "')') from string_function_test_people order by id asc;",
      {{"John Smith (555.803.2144)"},
       {"John Banks (555.803.8244)"},
       {"John Wilson (555.614.9814)"},
       {"Sue Smith (555.614.2282)"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_EncodedTextWithNulls) {
  // Concat two dictionary encoded text columns with nulls
  sqlAndCompareResult(
      "select COALESCE(first_name || ' ' || zip_plus_4, 'null') from "
      "string_function_test_people order by id asc;",
      {{"JOHN 90210-7743"}, {"John 94104-8123"}, {"null"}, {"null"}});
}

TEST_P(StringFunctionTest, ConcatTwoVarArg_GroupBy) {
  sqlAndCompareResult(
      "select lower(first_name) || ' ' || lower(country_code) as t, count(*) as "
      "n from "
      "string_function_test_people group by t order by t asc;",
      {{"john ca", int64_t(1)}, {"john us", int64_t(2)}, {"sue ca", int64_t(1)}});
}

TEST_P(StringFunctionTest, LPad) {
  sqlAndCompareResult(
      "select lpad(name, 14) from string_function_test_countries order by id asc;",
      {{" United States"}, {"        Canada"}, {"United Kingdom"}, {"       Germany"}});
}

TEST_P(StringFunctionTest, LPadTruncate) {
  sqlAndCompareResult(
      "select lpad(name, 5) from string_function_test_countries order by id asc;",
      {{"Unite"}, {"Canad"}, {"Unite"}, {"Germa"}});
}

TEST_P(StringFunctionTest, LPadCustomChars) {
  sqlAndCompareResult(
      "select lpad(name, 14, '>|<') from string_function_test_countries order by id "
      "asc;",
      {{">United States"}, {">|<>|<>|Canada"}, {"United Kingdom"}, {">|<>|<>Germany"}});
}

TEST_P(StringFunctionTest, DISABLED_LPadLiteral) {
  sqlAndCompareResult("select lpad('123', 2);", {{"  123"}});
}

TEST_P(StringFunctionTest, RPad) {
  sqlAndCompareResult(
      "select rpad(name, 20) from string_function_test_countries order by id asc;",
      {{"United States       "},
       {"Canada              "},
       {"United Kingdom      "},
       {"Germany             "}});
}

TEST_P(StringFunctionTest, RPadLiteral) {
  sqlAndCompareResult(
      "select rpad('$323.', 8, '98') from string_function_test_countries order by id "
      "asc;",
      {{"$323.989"}, {"$323.989"}, {"$323.989"}, {"$323.989"}});
}

TEST_P(StringFunctionTest, TrimBothDefault) {
  // Will be a no-op as default trim character is space
  sqlAndCompareResult(
      "select trim(arrow_code) from string_function_test_countries order by id asc;",
      {{">>US<<"}, {">>CA<<"}, {">>GB<<"}, {">>DE<<"}});
}

TEST_P(StringFunctionTest, TrimBothCustomWithoutBothSyntax) {
  // Implicit 'BOTH
  sqlAndCompareResult(
      "select trim('<>' from arrow_code) from string_function_test_countries order "
      "by id asc;",
      {{"US"}, {"CA"}, {"GB"}, {"DE"}});
}

TEST_P(StringFunctionTest, TrimBothCustom_WithBothSyntax) {
  // explicit syntax
  sqlAndCompareResult(
      "select trim(both '<>' from arrow_code) from string_function_test_countries "
      "order by id asc;",
      {{"US"}, {"CA"}, {"GB"}, {"DE"}});
}

TEST_P(StringFunctionTest, TrimBothLiteralWithBothSyntax) {
  sqlAndCompareResult("select trim(both ' !' from ' Oops!');", {{"Oops"}});
}

TEST_P(StringFunctionTest, TrimBothLiteralWithoutBothSyntax) {
  sqlAndCompareResult("select trim(' !' from ' Oops!');", {{"Oops"}});
}

TEST_P(StringFunctionTest, LeftTrimLeadingSyntax) {
  // Trim with 'LEADING'
  sqlAndCompareResult(
      "select trim(leading '<>#' from arrow_code) from "
      "string_function_test_countries order by id asc;",
      {{"US<<"}, {"CA<<"}, {"GB<<"}, {"DE<<"}});
}

TEST_P(StringFunctionTest, LeftTrimTwoArgsSyntax) {
  // Explicit LTrim
  sqlAndCompareResult(
      "select ltrim(arrow_code, '<>#') from string_function_test_countries order by "
      "id asc;",
      {{"US<<"}, {"CA<<"}, {"GB<<"}, {"DE<<"}});
}

TEST_P(StringFunctionTest, LeftTrimLiteral) {
  // Trim with 'LEADING'
  sqlAndCompareResult("select trim(leading '$' from '$19.99$');", {{"19.99$"}});
  // LTrim
  sqlAndCompareResult("select ltrim('$19.99$', '$');", {{"19.99$"}});
}

TEST_P(StringFunctionTest, RightTrim) {
  // Trim with 'TRAILING'
  sqlAndCompareResult(
      "select trim(trailing '<> ' from arrow_code) from "
      "string_function_test_countries order by id asc;",
      {{">>US"}, {">>CA"}, {">>GB"}, {">>DE"}});
  // RTrim
  sqlAndCompareResult(
      "select rtrim(arrow_code, '<> ') from string_function_test_countries order by "
      "id asc;",
      {{">>US"}, {">>CA"}, {">>GB"}, {">>DE"}});
}

TEST_P(StringFunctionTest, RightTrimLiteral) {
  // Trim with 'TRAILING'
  sqlAndCompareResult("select trim(trailing '|' from '|half pipe||');", {{"|half pipe"}});

  // RTrim
  sqlAndCompareResult("select rtrim('|half pipe||', '|');", {{"|half pipe"}});
}

TEST_P(StringFunctionTest, Substring) {
  sqlAndCompareResult(
      "select substring(full_name, 1, 4) from string_function_test_people order by "
      "id asc;",
      {{"John"}, {"John"}, {"John"}, {"Sue "}});
  sqlAndCompareResult(
      "select substring(full_name from 1 for 4) from string_function_test_people "
      "order by "
      "id asc;",
      {{"John"}, {"John"}, {"John"}, {"Sue "}});
  // Test null inputs
  sqlAndCompareResult(
      "select substring(zip_plus_4, 1, 5) from string_function_test_people order by "
      "id asc;",
      {{"90210"}, {"94104"}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, SubstringNegativeWrap) {
  sqlAndCompareResult(
      "select substring(full_name, -3, 2) from string_function_test_people order by "
      "id asc;",
      {{"IT"}, {"NK"}, {"SO"}, {"IT"}});
}

TEST_P(StringFunctionTest, SubstringLengthOffEnd) {
  sqlAndCompareResult(
      "select substring(code, 2, 10) from string_function_test_countries order by id "
      "asc;",
      {{"S"}, {"a"}, {"b"}, {"E"}});
}

TEST_P(StringFunctionTest, SubstringLiteral) {
  sqlAndCompareResult("select substring('fUnNy CaSe', 4, 4);", {{"Ny C"}});
}

// Test that index of 0 is equivalent to index of 1 (first character)
TEST_P(StringFunctionTest, SubstringLengthZeroStartLiteral) {
  sqlAndCompareResult("select substring('12345', 1, 3);", {{"123"}});

  sqlAndCompareResult("select substring('12345', 0, 3);", {{"123"}});
}

TEST_P(StringFunctionTest, SubstrAlias) {
  sqlAndCompareResult(
      "select substr(us_phone_number, 5, 3) from string_function_test_people order "
      "by id asc;",
      {{"803"}, {"803"}, {"614"}, {"614"}});
}

TEST_P(StringFunctionTest, SubstrAliasLiteral) {
  sqlAndCompareResult("select substr('fUnNy CaSe', 4, 4);", {{"Ny C"}});
}

TEST_P(StringFunctionTest, Overlay) {
  sqlAndCompareResult(
      "select overlay(us_phone_number placing '6273' from 9) from "
      "string_function_test_people order by id asc;",
      {{"555-803-6273"}, {"555-803-6273"}, {"555-614-6273"}, {"555-614-6273"}});
}

TEST_P(StringFunctionTest, OverlayInsert) {
  sqlAndCompareResult(
      "select overlay(us_phone_number placing '+1-' from 1 for 0) from "
      "string_function_test_people order by id asc;",
      {{"+1-555-803-2144"},
       {"+1-555-803-8244"},
       {"+1-555-614-9814"},
       {"+1-555-614-2282"}});
}

TEST_P(StringFunctionTest, OverlayLiteralNoFor) {
  sqlAndCompareResult("select overlay('We all love big data.' PLACING 'fast' FROM 13);",
                      {{"We all love fastdata."}});
}

TEST_P(StringFunctionTest, OverlayLiteralWithFor) {
  sqlAndCompareResult(
      "select overlay('We all love big data.' PLACING 'fast' FROM 13 FOR 3);",
      {{"We all love fast data."}});
}

TEST_P(StringFunctionTest, Replace) {
  sqlAndCompareResult(
      "select replace(us_phone_number, '803', '#^!') from "
      "string_function_test_people order by id asc;",
      {{"555-#^!-2144"}, {"555-#^!-8244"}, {"555-614-9814"}, {"555-614-2282"}});
}

TEST_P(StringFunctionTest, DISABLED_ReplaceEmptyReplacement) {
  sqlAndCompareResult(
      "select replace(us_phone_number, '555-') from "
      "string_function_test_people order by id asc;",
      {{"803-2144"}, {"803-8244"}, {"614-9814"}, {"614-2282"}});
}

TEST_P(StringFunctionTest, ReplaceLiteral) {
  sqlAndCompareResult("select replace('We all love big data.', 'big', 'fast');",
                      {{"We all love fast data."}});
}

TEST_P(StringFunctionTest, DISABLED_ReplaceLiteralEmptyReplacement) {
  sqlAndCompareResult("select replace('We all love big data.', 'big');",
                      {{"We all love data."}});
}

TEST_P(StringFunctionTest, SplitPart) {
  sqlAndCompareResult(
      "select split_part(us_phone_number, '-', 2) from string_function_test_people "
      "order by id asc;",
      {{"803"}, {"803"}, {"614"}, {"614"}});
}

TEST_P(StringFunctionTest, SplitPartNegativeIndex) {
  sqlAndCompareResult(
      "select split_part(us_phone_number, '-', -1) from "
      "string_function_test_people order by id asc;",
      {{"2144"}, {"8244"}, {"9814"}, {"2282"}});
}

TEST_P(StringFunctionTest, SplitPartNonDelimiterMatch) {
  sqlAndCompareResult(
      "select split_part(name, ' ', -1) from "
      "string_function_test_countries order by id asc;",
      {{"States"}, {"Canada"}, {"Kingdom"}, {"Germany"}});

  for (int64_t split_idx = 0; split_idx <= 1; ++split_idx) {
    sqlAndCompareResult("select split_part(name, ' ', " + std::to_string(split_idx) +
                            ") from string_function_test_countries order by id asc;",
                        {{"United"}, {"Canada"}, {"United"}, {"Germany"}});
  }

  sqlAndCompareResult(
      ""
      "select split_part(name, ' ', 2) from "
      "string_function_test_countries order by id asc;",
      {{"States"}, {Null}, {"Kingdom"}, {Null}});
}

TEST_P(StringFunctionTest, SplitPartLiteral) {
  sqlAndCompareResult("select split_part('192.168.0.1', '.', 2);", {{"168"}});
}

TEST_P(StringFunctionTest, SplitPartLiteralNegativeIndex) {
  sqlAndCompareResult("select split_part('192.168.0.1', '.', -1);", {{"1"}});
}

TEST_P(StringFunctionTest, SplitPartLiteralNullIndex) {
  sqlAndCompareResult("select split_part('192.168.0.1', '.', 5);", {{Null}});
}

TEST_P(StringFunctionTest, RegexpReplace2Args) {
  sqlAndCompareResult(
      "select regexp_replace(name, 'United[[:space:]]') from "
      "string_function_test_countries order by id asc;",
      {{"States"}, {"Canada"}, {"Kingdom"}, {"Germany"}});
}

TEST_P(StringFunctionTest, RegexpReplace3Args) {
  sqlAndCompareResult(
      "select regexp_replace(name, 'United[[:space:]]([[:alnum:]])', 'The United "
      "$1') from string_function_test_countries order by id asc;",
      {{"The United States"}, {"Canada"}, {"The United Kingdom"}, {"Germany"}});
}

TEST_P(StringFunctionTest, RegexpReplace4Args) {
  sqlAndCompareResult(
      "select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
      "4) from string_function_test_people order by id asc",
      {{"All for one..two and one..two for all."},
       // Note we don't replace the first One due to start position argument of 4
       {"One plus one..two does not equal two."},
       {"What is the sound of one..two hand clapping?"},
       {"Nothing exists entirely alone. Everything is always in relation to "
        "everything else."}});

  // Test negative position, should wrap
  sqlAndCompareResult(
      "select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
      "-18) from string_function_test_people order by id asc",
      {{"All for one and one..two for all."},
       // Note we don't replace the first One due to start position argument of 4
       {"One plus one does not equal two."},
       {"What is the sound of one..two hand clapping?"},
       {"Nothing exists entirely alone. Everything is always in relation to "
        "everything else."}});
}

// 5th argument is occurrence

TEST_P(StringFunctionTest, RegexpReplace5Args) {
  // 0 for 5th (occurrence) arguments says to replace all matches
  sqlAndCompareResult(
      "select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
      "1, 0) from string_function_test_people order by id asc",
      {{"All for one..two and one..two for all."},
       {"One..two plus one..two does not equal two."},
       {"What is the sound of one..two hand clapping?"},
       {"Nothing exists entirely alone. Everything is always in relation to "
        "everything else."}});

  // Replace second match
  sqlAndCompareResult(
      "select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
      "1, 2) from string_function_test_people order by id asc",
      {{"All for one and one..two for all."},
       // Note we don't replace the first One due to start position argument of 4
       {"One plus one..two does not equal two."},
       {"What is the sound of one hand clapping?"},
       {"Nothing exists entirely alone. Everything is always in relation to "
        "everything else."}});

  // Replace second to last match via negative wrapping
  sqlAndCompareResult(
      "select regexp_replace(personal_motto, '([Oo]ne)[[:space:]]', '$1..two ', "
      "1, -2) from string_function_test_people order by id asc",
      {{"All for one..two and one for all."},
       // Note we don't replace the first One due to start position argument of 4
       {"One..two plus one does not equal two."},
       {"What is the sound of one hand clapping?"},
       {"Nothing exists entirely alone. Everything is always in relation to "
        "everything else."}});
}

// 6th argument is regex parameters
TEST_P(StringFunctionTest, RegexpReplace6Args) {
  // Currently only support 'c' (case sensitive-default) and 'i' (case insensitive) for
  // RegexpReplace

  // Test 'c' - case sensitive
  sqlAndCompareResult(
      "select regexp_replace(personal_motto, '(one)[[:space:]]', '$1..two ', 1, "
      "0, 'c') from string_function_test_people order by id asc",
      {{"All for one..two and one..two for all."},
       // Note "One" in next entry doesn't match due to case sensitive search
       {"One plus one..two does not equal two."},
       {"What is the sound of one..two hand clapping?"},
       {"Nothing exists entirely alone. Everything is always in relation to "
        "everything else."}});

  // Test 'i' - case insensitive

  sqlAndCompareResult(
      "select regexp_replace(personal_motto, '(one)[[:space:]]', '$1..two ', 1, "
      "0, 'i') from string_function_test_people order by id asc",
      {{"All for one..two and one..two for all."},
       // With case insensitive search, "One" will match
       {"One..two plus one..two does not equal two."},
       {"What is the sound of one..two hand clapping?"},
       {"Nothing exists entirely alone. Everything is always in relation to "
        "everything else."}});

  // Test that invalid regex param causes exception
  EXPECT_ANY_THROW(
      sql("select regexp_replace(personal_motto, '(one)[[:space:]]', '$1..two ', 1, "
          "0, 'iz') from string_function_test_people order by id asc;"));
}

TEST_P(StringFunctionTest, RegexpReplaceLiteral) {
  sqlAndCompareResult(
      "select regexp_replace('How much wood would a wood chuck chuck if a wood "
      "chuck could chuck wood?', 'wo[[:alnum:]]+d', 'metal', 1, 0, 'i');",
      {{"How much metal metal a metal chuck chuck if a metal chuck could chuck metal?"}});
}

TEST_P(StringFunctionTest, RegexpReplaceLiteralSpecificMatch) {
  sqlAndCompareResult(
      "select regexp_replace('How much wood would a wood chuck chuck if a wood "
      "chuck could chuck wood?', 'wo[[:alnum:]]+d', 'should', 1, 2, 'i');",
      {{"How much wood should a wood chuck chuck if a wood chuck could chuck wood?"}});
}

TEST_P(StringFunctionTest, RegexpSubstr2Args) {
  sqlAndCompareResult(
      "select regexp_substr(raw_email, '[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+') "
      "from string_function_test_people order by id asc;",
      {{"therealjohnsmith@omnisci.com"},
       {"john_banks@mapd.com"},
       {"JOHN.WILSON@geops.net"},
       {"sue4tw@example.com"}});
}

// 3rd arg is start position
TEST_P(StringFunctionTest, RegexpSubstr3Args) {
  sqlAndCompareResult(
      "select regexp_substr(raw_email, '[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
      "20) from string_function_test_people order by id asc;",
      {{"therealjohnsmith@omnisci.com"}, {Null}, {Null}, {"sue.smith@example.com"}});
}

// 4th arg is the occurence index
TEST_P(StringFunctionTest, RegexpSubstr4Args) {
  sqlAndCompareResult(
      "select regexp_substr(raw_email, '[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
      "1, 2) from string_function_test_people order by id asc;",
      {{Null}, {Null}, {Null}, {"sue.smith@example.com"}});

  // Test negative wrapping
  sqlAndCompareResult(
      "select regexp_substr(raw_email, '[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
      "1, -1) from string_function_test_people order by id asc;",
      {{"therealjohnsmith@omnisci.com"},
       {"john_banks@mapd.com"},
       {"JOHN.WILSON@geops.net"},
       {"sue.smith@example.com"}});
}

// 5th arg is regex params, 6th is sub-match index if 'e' is specified as regex param
TEST_P(StringFunctionTest, RegexpSubstr5Or6Args) {
  // case sensitive
  sqlAndCompareResult(
      "select regexp_substr(raw_email, "
      "'john[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', 1, 1, 'c') from "
      "string_function_test_people order by id asc;",
      {{"johnsmith@omnisci.com"}, {"john_banks@mapd.com"}, {Null}, {Null}});

  // case insensitive
  sqlAndCompareResult(
      "select regexp_substr(raw_email, "
      "'john[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', 1, 1, 'i') from "
      "string_function_test_people order by id asc;",
      {{"johnsmith@omnisci.com"},
       {"john_banks@mapd.com"},
       {"JOHN.WILSON@geops.net"},
       {Null}});

  // extract sub-matches

  // Get the email domain (second sub-match)
  sqlAndCompareResult(
      "select regexp_substr(raw_email, "
      "'([[:alnum:]._-]+)@([[:alnum:]]+.[[:alnum:]]+)', 1, 1, 'ce', 2) from "
      "string_function_test_people order by id asc;",
      {{"omnisci.com"}, {"mapd.com"}, {"geops.net"}, {"example.com"}});

  // Sub-match has no effect if extract ('e') is not specified

  sqlAndCompareResult(
      "select regexp_substr(raw_email, "
      "'([[:alnum:]._-]+)@([[:alnum:]]+.[[:alnum:]]+)', 1, 1, 'i', 2) from "
      "string_function_test_people order by id asc;",
      {{"therealjohnsmith@omnisci.com"},
       {"john_banks@mapd.com"},
       {"JOHN.WILSON@geops.net"},
       {"sue4tw@example.com"}});

  // Throw error if regex param is not valid
  EXPECT_ANY_THROW(
      sql("select regexp_substr(raw_email, "
          "'([[:alnum:]._-]+)@([[:alnum:]]+.[[:alnum:]]+)', 1, 1, 'z', 2) from "
          "string_function_test_people order by id asc;"));

  // Throw error if case regex param not specified
  EXPECT_ANY_THROW(
      sql("select regexp_substr(raw_email, "
          "'([[:alnum:]._-]+)@([[:alnum:]]+.[[:alnum:]]+)', 1, 1, 'e', 2) from "
          "string_function_test_people order by id asc;"));
}

TEST_P(StringFunctionTest, RegexpSubstrLiteral) {
  sqlAndCompareResult(
      "select regexp_substr('Feel free to send us an email at spam@devnull.com!', "
      "'[[:alnum:]]+@[[:alnum:]]+\\.[[:alnum:]]+',  1, -1, 'i', 0);",
      {{"spam@devnull.com"}});
}

TEST_P(StringFunctionTest, RegexpCount2Args) {
  sqlAndCompareResult(
      "select regexp_count(json_data_none, 'in') "
      "from string_function_test_countries order by id asc;",
      {{int64_t(4)}, {int64_t(6)}, {int64_t(3)}, {int64_t(4)}});
}

TEST_P(StringFunctionTest, RegexpCount3Args) {
   // 3rd argument to RegexpCount is starting position to search for matches
  sqlAndCompareResult(
      "select regexp_count(json_data_none, 'in', 50) "
      "from string_function_test_countries order by id asc;",
      {{int64_t(3)}, {int64_t(5)}, {int64_t(2)}, {int64_t(2)}});
}

TEST_P(StringFunctionTest, RegexpCount4Args) {
   // 4th argument to RegexpCount is for regex parameters.
   // Notably 'c' specifies case sensitive, and 'i' specifies case insensitive

  // Case-senstive default
  sqlAndCompareResult(
      "select regexp_count(personal_motto, 'one', 1) from "
      "string_function_test_people order by id asc;",
      {{int64_t(2)}, {int64_t(1)}, {int64_t(1)}, {int64_t(1)}});

  // Case-senstive default
  sqlAndCompareResult(
      "select regexp_count(personal_motto, 'one', 1, 'c') from "
      "string_function_test_people order by id asc;",
      {{int64_t(2)}, {int64_t(1)}, {int64_t(1)}, {int64_t(1)}});

  // Case-insenstive search
  sqlAndCompareResult(
      "select regexp_count(personal_motto, 'one', 1, 'i') from "
      "string_function_test_people order by id asc;",
      {{int64_t(2)}, {int64_t(2)}, {int64_t(1)}, {int64_t(1)}});
}

TEST_P(StringFunctionTest, RegexpCountLiteral) {
  sqlAndCompareResult(
      "select regexp_count('Feel free to send us an email at spam@devnull.com or "
      "to morespam@doa.com!', "
      "'[[:alnum:]]+@[[:alnum:]]+\\.[[:alnum:]]+',  1, 'i');",
      {{int64_t(2)}});
}

TEST_P(StringFunctionTest, JsonValue) {
  // String value with key at root level
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.capital') from "
      "string_function_test_countries;",
      {{"Washington D.C."}, {"Toronto"}, {"London"}, {"Berlin"}});

  // Numeric value with key at root level
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.pop') from "
      "string_function_test_countries;",
      {{"329500000"}, {"38010000"}, {"67220000"}, {Null}});

  // Boolean value with key at root level
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.has_prime_minister') from "
      "string_function_test_countries;",
      {{"false"}, {"true"}, {"true"}, {"false"}});

  // Null values with key at root level
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.prime_minister') from "
      "string_function_test_countries;",
      {{Null}, {"Justin Trudeau"}, {"Boris Johnson"}, {Null}});

  // Non-existent key at root level (actual key: "capital")
  // Should be all nulls
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.capitol') from "
      "string_function_test_countries;",
      {{Null}, {Null}, {Null}, {Null}});

  // Nested Accessor
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.factoids.most_valuable_crop') "
      "from string_function_test_countries;",
      {{"corn"}, {Null}, {"wheat"}, {"wheat"}});

  // Nested Accessor - non-existent key
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.factoids.nicest_view') "
      "from string_function_test_countries;",
      {{Null}, {Null}, {Null}, {Null}});

  // Nested Accessor - two non-existent nested keys
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.factoids.provinces.ottawa') "
      "from string_function_test_countries;",
      {{Null}, {Null}, {Null}, {Null}});

  // Nested Accessor - two non-existent nested keys - last is array
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.factoids.provinces.populations[3]') "
      "from string_function_test_countries;",
      {{Null}, {Null}, {Null}, {Null}});

  // Nested Accessor - Array (string)
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.factoids.\"Last 3 leaders\"[2]') "
      "from string_function_test_countries;",
      {{"Joseph Biden"}, {"Justin Trudeau"}, {Null}, {Null}});

  // Nested Accessor - Array (numeric)
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.factoids.gdp_per_cap_2015_2020[4]') "
      "from string_function_test_countries;",
      {{"65280"}, {"46327"}, {"42354"}, {"46468"}});

  // Nested Accessor - Array (numeric, off end)
  sqlAndCompareResult(
      "select json_value(json_data_none, '$.factoids.gdp_per_cap_2015_2020[7]') "
      "from string_function_test_countries;",
      {{Null}, {Null}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, JsonValueParseMode) {
  // Explicit Lax Mode (the default)
  sqlAndCompareResult(
      "select json_value(json_data_none, 'lax $.factoids.most_valuable_crop') "
      "from string_function_test_countries;",
      {{"corn"}, {Null}, {"wheat"}, {"wheat"}});

  // Explicit Lax Mode (test case-insensitivity)
  sqlAndCompareResult(
      "select json_value(json_data_none, 'LAX $.factoids.most_valuable_crop') "
      "from string_function_test_countries;",
      {{"corn"}, {Null}, {"wheat"}, {"wheat"}});

  // Test that Strict Mode is disabled
  queryAndAssertException(
      "select json_value(json_data_none, 'strict $.factoids.most_valuable_crop') "
      "from string_function_test_countries;",
      "Strict parsing not currently supported for JSON_VALUE.");
}

TEST_P(StringFunctionTest, Base64) {
  // Current behavior is that BASE64_ENCODE(NULL literal) and BASE64_DECODE(NULL
  // literal) returns NULL
  sqlAndCompareResult("select base64_encode(CAST(NULL AS TEXT)) IS NULL;", {{True}});

  sqlAndCompareResult("select base64_decode(CAST(NULL AS TEXT)) IS NULL;", {{True}});

  // Current behavior is that BASE64_ENCODE(NULL var) and BASE64_DECODE(NULL var)
  // returns NULL;
  sqlAndCompareResult(
      "SELECT base64_encode(json_value(json_data_none, '$.prime_minister'))"
      " IS NULL FROM string_function_test_countries ORDER BY rowid ASC;",
      {{True}, {False}, {False}, {True}});

  sqlAndCompareResult("select base64_encode('HEAVY.AI');", {{"SEVBVlkuQUk="}});

  sqlAndCompareResult("select base64_decode('SEVBVlkuQUk=');", {{"HEAVY.AI"}});

 sqlAndCompareResult("select base64_decode(base64_encode('HEAVY.AI'));", {{"HEAVY.AI"}});

  // Invalid base64 characters, should throw
  EXPECT_ANY_THROW(sql("select base64_decode('HEAVY.AI');"));

  // Below encodings validated independently
  sqlAndCompareResult(
      "select base64_encode(name) from string_function_test_countries ORDER by "
      "id ASC;",
      {{"VW5pdGVkIFN0YXRlcw=="},
       {"Q2FuYWRh"},
       {"VW5pdGVkIEtpbmdkb20="},
       {"R2VybWFueQ=="}});

  sqlAndCompareResult(
      "select base64_decode(base64_encode(name)) from "
      "string_function_test_countries ORDER by id ASC;",
      {{"United States"}, {"Canada"}, {"United Kingdom"}, {"Germany"}});
}

TEST_P(StringFunctionTest, UrlEncodeAndDecodeInversesAndNull) {
  // Verify URL_DECODE() is inverse of URL_ENCODE()
  sqlAndCompareResult(
      "SELECT COUNT(*) = COUNT_IF(personal_motto = "
      "URL_DECODE(URL_ENCODE(personal_motto))) FROM string_function_test_people;",
      {{True}});

  // Verify empty string and NULL behavior

  sqlAndCompareResult(
      "SELECT URL_ENCODE(b_str) IS NULL FROM numeric_to_string_test ORDER BY b_str "
      "NULLS FIRST LIMIT 1;",
      {{True}});

  sqlAndCompareResult(
      "SELECT URL_DECODE(b_str) IS NULL FROM numeric_to_string_test ORDER BY b_str "
      "NULLS FIRST LIMIT 1;",
      {{True}});

  sqlAndCompareResult("SELECT URL_ENCODE('') IS NULL;", {{True}});

  sqlAndCompareResult("SELECT URL_DECODE('') IS NULL;", {{True}});
}

TEST_P(StringFunctionTest, TryCastIntegerTypes) {
  // INT projected
  sqlAndCompareResult(
     "select try_cast(split_part(us_phone_number, '-', 2) as int) as digits "
      " from  string_function_test_people ORDER BY id ASC;",
      {{int64_t(803)}, {int64_t(803)}, {int64_t(614)}, {int64_t(614)}});

  // INT grouped
  sqlAndCompareResult(
      "select try_cast(split_part(us_phone_number, '-', 2) as int) as digits, "
      "count(*) as n from string_function_test_people group by digits "
      "ORDER BY digits ASC;",
      {{int64_t(614), int64_t(2)}, {int64_t(803), int64_t(2)}});

  // TINYINT Projecte
  // Todo: This test framework doesn't properly handle nulls, hence
  // the coalesce to a -1 sentinel value below. Fix this.
  sqlAndCompareResult(
      "select coalesce(try_cast(substring(zip_plus_4 from 3 for 3) "
      "as tinyint) , -1) as digits from string_function_test_people "
      "ORDER BY id ASC;",
      {{int64_t(-1)}, {int64_t(104)}, {int64_t(-1)}, {int64_t(-1)}});

  // SMALLINT Projected
  // Todo: This test framework doesn't properly handle nulls, hence
  // the coalesce to a -1 sentinel value below. Fix this.
  sqlAndCompareResult(
      "select coalesce(try_cast(substring(zip_plus_4 from 3 for 3) "
      "as smallint) , -1) as digits from string_function_test_people "
      "ORDER BY id ASC;",
      {{int64_t(210)}, {int64_t(104)}, {int64_t(-1)}, {int64_t(-1)}});

  // INT Projected
  // Todo: This test framework doesn't properly handle nulls, hence
  // the coalesce to a -1 sentinel value below. Fix this.
  sqlAndCompareResult(
      "select coalesce(try_cast(substring(zip_plus_4 from 3 for 3) "
      "as int) , -1) as digits from string_function_test_people "
      "ORDER BY id ASC;",
      {{int64_t(210)}, {int64_t(104)}, {int64_t(-1)}, {int64_t(-1)}});

  // BIGINT Projected
  // Todo: This test framework doesn't properly handle nulls, hence
  // the coalesce to a -1 sentinel value below. Fix this.
  sqlAndCompareResult(
      "select coalesce(try_cast(substring(zip_plus_4 from 3 for 3) "
      "as bigint) , -1) as digits from string_function_test_people "
      "ORDER BY id ASC;",
      {{int64_t(210)}, {int64_t(104)}, {int64_t(-1)}, {int64_t(-1)}});
}

TEST_P(StringFunctionTest, TryCastFPTypes) {
  // Todo: This test framework doesn't properly handle nulls, hence
  // the coalesce to a -1 sentinel value below. Fix this.

  // The actual conversion values, despite perhaps looking close to actuals (as of Aug
  // 2022), were choosen to be values that were exactly representable as floating
  // point values, so as to not need to introduce an epsilon range check
  sqlAndCompareResult(
      "select name, coalesce(try_cast(json_value(json_data_none, "
      "'$.exchange_rate_usd') "
      "as float), -1) as fp from string_function_test_countries ORDER BY name ASC;",
      {{"Canada", float(0.78125)},
       {"Germany", float(1.015625)},
       {"United Kingdom", float(1.21875)},
       {"United States", float(-1)}});

  // Todo: This test framework doesn't properly handle nulls, hence
  // the coalesce to a -1 sentinel value below. Fix this.

  // The actual exchange rates, despite being close to actuals (as of Aug
  // 2022), were choosen to be values that were exactly representable as floating
  // point values, so as to not need to introduce an epsilon range check
  sqlAndCompareResult(
      "select name, coalesce(try_cast(json_value(json_data_none, "
      "'$.exchange_rate_usd') "
      "as double), -1) as fp from string_function_test_countries ORDER BY name ASC;",
      {{"Canada", double(0.78125)},
       {"Germany", double(1.015625)},
       {"United Kingdom", double(1.21875)},
       {"United States", double(-1)}});
}

TEST_P(StringFunctionTest, TryCastDecimalTypes) {
  // Todo: This test framework doesn't properly handle nulls, hence
  // the coalesce to a -1 sentinel value below. Fix this.

  // The actual conversion values, despite perhaps looking close to actuals (as of Aug
  // 2022), were choosen to be values that were exactly representable as floating
  // point values, so as to not need to introduce an epsilon range check
  sqlAndCompareResult(
      "select name, coalesce(try_cast(json_value(json_data_none, "
      "'$.exchange_rate_usd') "
      "as decimal(7, 6)), -1) as dec_val from string_function_test_countries "
      "ORDER BY name ASC;",
      {{"Canada", double(0.78125)},
       {"Germany", double(1.015625)},
       {"United Kingdom", double(1.21875)},
       {"United States", double(-1)}});
}

TEST_P(StringFunctionTest, TryCastDateTypes) {
  // Projected
  sqlAndCompareResult(
      "select name, coalesce(extract(month from try_cast(json_value(json_data_none, "
      "'$.independence_day') as date)), -1) "
      "as independence_month from string_function_test_countries ORDER BY name ASC;",
      {{"Canada", int64_t(7)},
       {"Germany", int64_t(10)},
       {"United Kingdom", int64_t(-1)},
       {"United States", int64_t(7)}});

  // Group By
  sqlAndCompareResult(
      "select coalesce(extract(month from try_cast(json_value(json_data_none, "
      "'$.independence_day') as date)), -1) "
      "as independence_month, count(*) as n from string_function_test_countries group by "
      "independence_month "
      "ORDER BY independence_month ASC;",
      {{int64_t(-1), int64_t(1)}, {int64_t(7), int64_t(2)}, {int64_t(10), int64_t(1)}});
}

TEST_P(StringFunctionTest, TryCastTimestampTypes) {
  sqlAndCompareResult(
      "select extract(epoch from try_cast('2013-09-10 09:00:00' as timestamp));",
      {{int64_t(1378803600)}});

  sqlAndCompareResult(
      "select extract(millisecond from try_cast('2013-09-10 09:00:00.123' as "
      "timestamp(3)));",
      {{int64_t(123)}});

  // Null result
  sqlAndCompareResult(
      "select coalesce(try_cast('2020 -09/10 09:00:00' as timestamp), -1);", {{Null}});

  // Todo (todd): We're not coalescing the null value (MIN_BIGINT)
  // here so this test will currently fail. Investigate and fix.
  // compare_result_set(expected_result_set, result_set);
}

TEST_P(StringFunctionTest, TryCastTimeTypes) {
  sqlAndCompareResult("select extract(minute from try_cast('09:12:34' as time));",
                      {{int64_t(12)}});
}

TEST_P(StringFunctionTest, Position) {
  // Literal search
  sqlAndCompareResult("select position('ell' in 'hello');", {{int64_t(2)}});

  // Literal search with starting position
  sqlAndCompareResult("select position('ell' in 'hello' from 2);", {{int64_t(2)}});

  // Literal search with starting position past position index -
  // returns 0
  sqlAndCompareResult("select position('ell' in 'hello' from 3);", {{int64_t(0)}});

  // Literal search with negative "wraparound" starting position
  sqlAndCompareResult("select position('ell' in 'hello' from -4);", {{int64_t(2)}});

  // Literal search with negative "wraparound" starting position
  // past position index - returns 0
  sqlAndCompareResult("select position('ell' in 'hello' from -3);", {{int64_t(0)}});

  // All searches match
  sqlAndCompareResult(
      "select id, position('one' in personal_motto) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(9)},
       {int64_t(2), int64_t(10)},
       {int64_t(3), int64_t(22)},
       {int64_t(4), int64_t(27)}});

  // Some searches do not match, non-matches should return 0
  sqlAndCompareResult(
      "select id, position('for' in personal_motto) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(5)},
       {int64_t(2), int64_t(0)},
       {int64_t(3), int64_t(0)},
       {int64_t(4), int64_t(0)}});

  // Optional third start operand
  sqlAndCompareResult(
      "select id, position('one' in personal_motto from 12) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(17)},
       {int64_t(2), int64_t(0)},
       {int64_t(3), int64_t(22)},
       {int64_t(4), int64_t(27)}});

  // Negative optional third start operand
  sqlAndCompareResult(
      "select id, position('one' in personal_motto from -18) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(17)},
       {int64_t(2), int64_t(0)},
       {int64_t(3), int64_t(22)},
       {int64_t(4), int64_t(0)}});

  // Null inputs should output null
  sqlAndCompareResult(
      "select id, coalesce(position('94' in zip_plus_4), -1) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(0)},
       {int64_t(2), int64_t(1)},
       {int64_t(3), int64_t(-1)},
       {int64_t(4), int64_t(-1)}});

  // Empty search string should return start position (or 1 if it does
  // not exist) for all non-null inputs
  sqlAndCompareResult(
      "select id, coalesce(position('' in zip_plus_4), -1) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(1)},
       {int64_t(2), int64_t(1)},
       {int64_t(3), int64_t(-1)},
       {int64_t(4), int64_t(-1)}});

  // Empty search string should return start position (or 1 if it does
  // not exist) for all non-null inputs
  sqlAndCompareResult(
      "select id, coalesce(position('' in zip_plus_4 from 3), -1) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(3)},
       {int64_t(2), int64_t(3)},
       {int64_t(3), int64_t(-1)},
       {int64_t(4), int64_t(-1)}});

  // Chained string op
  sqlAndCompareResult(
      "select id, position('one' in lower(personal_motto)) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(9)},
       {int64_t(2), int64_t(1)},
       {int64_t(3), int64_t(22)},
       {int64_t(4), int64_t(27)}});

  // Text encoding none search
  sqlAndCompareResult(
      "select id, position('it' in last_name) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(0)},
       {int64_t(2), int64_t(0)},
       {int64_t(3), int64_t(0)},
       {int64_t(4), int64_t(3)}});

  // Text encoding none search chained
  sqlAndCompareResult(
      "select id, position('it' in lower(last_name)) from "
      "string_function_test_people order by id;",
      {{int64_t(1), int64_t(3)},
       {int64_t(2), int64_t(0)},
       {int64_t(3), int64_t(0)},
       {int64_t(4), int64_t(3)}});
}

TEST_P(StringFunctionTest, JarowinklerSimilarity) {
  // Literal similarity
  // Identical strings should score 100 similarity
  sqlAndCompareResult("select jarowinkler_similarity('hi', 'hi');", {{int64_t(100)}});

  // Literal similarity
  // Completely dissimilar strings should score 0 similarity
  // Note that Jaro-Winkler similarity is case sensitive
  sqlAndCompareResult("select jarowinkler_similarity('hi', 'HI');", {{int64_t(0)}});

  // Var-literal similarity, literal last
  sqlAndCompareResult(
      "select jarowinkler_similarity(personal_motto, 'one for all') from "
      "string_function_test_people order by id;",
      {{int64_t(63)}, {int64_t(45)}, {int64_t(48)}, {int64_t(59)}});

  // Var-literal similarity, literal first
  sqlAndCompareResult(
      "select jarowinkler_similarity('one for all', personal_motto) from "
      "string_function_test_people order by id;",
      {{int64_t(63)}, {int64_t(45)}, {int64_t(48)}, {int64_t(59)}});

  // Var-var similarity
  sqlAndCompareResult(
      "select jarowinkler_similarity(personal_motto, raw_email) from "
      "string_function_test_people order by id;",
      {{int64_t(56)}, {int64_t(43)}, {int64_t(42)}, {int64_t(60)}});

  // Var-var similarity, same string
  sqlAndCompareResult(
      "select jarowinkler_similarity(personal_motto, personal_motto) from "
      "string_function_test_people order by id;",
      {{int64_t(100)}, {int64_t(100)}, {int64_t(100)}, {int64_t(100)}});

  // Var-var similarity, one argument encoding none
  sqlAndCompareResult(
      "select jarowinkler_similarity(personal_motto, last_name) from "
      "string_function_test_people order by id;",
      {{int64_t(0)}, {int64_t(49)}, {int64_t(69)}, {int64_t(38)}});

  // Var-var similarity, both arguments encoding none
  sqlAndCompareResult(
      "select jarowinkler_similarity(last_name, last_name) from "
      "string_function_test_people order by id;",
      {{int64_t(100)}, {int64_t(100)}, {int64_t(100)}, {int64_t(100)}});

  // Var-var similarity, nested LOWER operator
  sqlAndCompareResult(
      "select jarowinkler_similarity(lower(first_name), lower(full_name)) from "
      "string_function_test_people order by id;",
      {{int64_t(88)}, {int64_t(88)}, {int64_t(87)}, {int64_t(84)}});
}

TEST_P(StringFunctionTest, LevenshteinDistance) {
  // Literal distance
  // Identical strings should have 0 distance
  sqlAndCompareResult("select levenshtein_distance('hi', 'hi');", {{int64_t(0)}});

  // Literal distance
  // Completely dissimilar strings should have distance equal to length of strings
  // Note that Levenshtein Distance is case sensitive
  sqlAndCompareResult("select levenshtein_distance('hi', 'HI');", {{int64_t(2)}});

  // Var-literal distance, literal last
  sqlAndCompareResult(
      "select levenshtein_distance(personal_motto, 'one for all') from "
      "string_function_test_people order by id;",
      {{int64_t(17)}, {int64_t(24)}, {int64_t(31)}, {int64_t(73)}});

  // Var-literal distance, literal first
  sqlAndCompareResult(
      "select levenshtein_distance('one for all', personal_motto) from "
      "string_function_test_people order by id;",
      {{int64_t(17)}, {int64_t(24)}, {int64_t(31)}, {int64_t(73)}});

  // Var-var distance
  sqlAndCompareResult(
      "select levenshtein_distance(personal_motto, raw_email) from "
      "string_function_test_people order by id;",
      {{int64_t(37)}, {int64_t(27)}, {int64_t(36)}, {int64_t(74)}});

  // Var-var distance, same string
  sqlAndCompareResult(
      "select levenshtein_distance(personal_motto, personal_motto) from "
      "string_function_test_people order by id;",
      {{int64_t(0)}, {int64_t(0)}, {int64_t(0)}, {int64_t(0)}});

  // Var-var distance, one argument encoding none
  sqlAndCompareResult(
      "select levenshtein_distance(personal_motto, last_name) from "
      "string_function_test_people order by id;",
      {{int64_t(28)}, {int64_t(30)}, {int64_t(34)}, {int64_t(80)}});

  // Var-var distance, both arguments encoding none
  sqlAndCompareResult(
      "select levenshtein_distance(last_name, last_name) from "
      "string_function_test_people order by id;",
      {{int64_t(0)}, {int64_t(0)}, {int64_t(0)}, {int64_t(0)}});

  // Var-var distance, nested LOWER operator
  sqlAndCompareResult(
      "select levenshtein_distance(lower(first_name), lower(full_name)) from "
      "string_function_test_people order by id;",
      {{int64_t(6)}, {int64_t(6)}, {int64_t(7)}, {int64_t(6)}});
}

TEST_P(StringFunctionTest, Hash) {
  // Literal hash
  sqlAndCompareResult("select hash('hi');", {{int64_t(1097802)}});

  // Literal null
  sqlAndCompareResult("select coalesce(hash(CAST(NULL AS TEXT)), 0);", {{int64_t(0)}});

  // Dictionary-encoded text column
  sqlAndCompareResult(
      "select hash(capital) from string_function_test_countries order by id;",
      {{int64_t(5703505280371710991)},
       {int64_t(1060071279222666409)},
       {int64_t(1057111063818803959)},
       {int64_t(1047250289947889561)}});

  // None-encoded text column
  sqlAndCompareResult(
      "select hash(capital_none) from string_function_test_countries order by id;",
      {{int64_t(5703505280371710991)},
       {int64_t(1060071279222666409)},
       {int64_t(1057111063818803959)},
       {int64_t(1047250289947889561)}});

  // Dictionary-encoded text column with nulls
  sqlAndCompareResult(
      "select coalesce(hash(zip_plus_4), 0) from string_function_test_people "
      "order by id;",
      {{int64_t(6345224789068548647)},
       {int64_t(-3868673234647279706)},
       {int64_t(0)},
       {int64_t(0)}});

  // None-encoded text column with nulls
  sqlAndCompareResult(
      "select coalesce(hash(short_name), 0) from string_function_test_countries "
      "order by id;",
      {{int64_t(0)},
       {int64_t(1048231423487679005)},
       {int64_t(1078829)},
       {int64_t(-2445200816347761128)}});

  // Hash comparison
  sqlAndCompareResult(
      "select count(*) from string_function_test_countries where "
      "hash(capital) = hash(capital_none);",
      {{int64_t(4)}});

  sqlAndCompareResult(
      "select hash(lower(first_name)), any_value(lower(first_name)), count(*) "
      "from string_function_test_people group by  hash(lower(first_name)) order "
      "by count(*) desc;",
      {{int64_t(1093213190016), "john", int64_t(3)},
       {int64_t(1105454758), "sue", int64_t(1)}});
}

TEST_P(StringFunctionTest, NullLiteralTest) {
  sqlAndCompareResult(
      "SELECT COUNT(str_fn) FROM (SELECT short_name, REGEXP_COUNT(CAST(NULL AS "
      "TEXT),'u',0,'i') AS str_fn FROM string_function_test_countries);",
      {{int64_t(0)}});

  sqlAndCompareResult(
      "SELECT COUNT(str_fn) FROM (SELECT short_name, REGEXP_SUBSTR(CAST(NULL AS "
      "TEXT),'u', 1, -1,'i', 0) AS str_fn FROM string_function_test_countries);",
      {{int64_t(0)}});

  sqlAndCompareResult(
      "SELECT COUNT(str_fn) FROM (SELECT short_name, POSITION('hi' in CAST(NULL "
      "AS TEXT)) AS str_fn FROM string_function_test_countries);",
      {{int64_t(0)}});

  sqlAndCompareResult(
      "SELECT COUNT(str_fn) FROM (SELECT short_name, "
      "JAROWINKLER_SIMILARITY(CAST(NULL AS TEXT), CAST(NULL AS TEXT)) AS str_fn FROM "
      "string_function_test_countries);",
      {{int64_t(0)}});
}

TEST_P(StringFunctionTest, ExplicitCastToNumeric) {
  sqlAndCompareResult(
      "select cast(age as text) from string_function_test_people order by id asc;",
      {{"25"}, {"30"}, {"20"}, {"25"}});

  sqlAndCompareResult(
      "select cast(age as text) || ' years'  from string_function_test_people order by "
      "id asc;",
      {{"25 years"}, {"30 years"}, {"20 years"}, {"25 years"}});

  sqlAndCompareResult(
      "select cast(age as text) || ' years' as age_years, count(*) as n "
      "from string_function_test_people group by age_years order by age_years asc;",
      {{"20 years", int64_t(1)}, {"25 years", int64_t(2)}, {"30 years", int64_t(1)}});
}

TEST_P(StringFunctionTest, ImplicitCastToNumeric) {
  sqlAndCompareResult(
      "select age || ' years'  from string_function_test_people order by id asc;",
      {{"25 years"}, {"30 years"}, {"20 years"}, {"25 years"}});

  sqlAndCompareResult(
      "select age || ' years' as age_years, count(*) as n "
      "from string_function_test_people group by age_years order by age_years asc;",
      {{"20 years", int64_t(1)}, {"25 years", int64_t(2)}, {"30 years", int64_t(1)}});
}

TEST_P(StringFunctionTest, CastTypesToString) {
  const std::vector col_type_strings = {"ti",
                                        "si",
                                        "i",
                                        "bi",
                                        "flt",
                                        "dbl",
                                        "dec_5_2",
                                        "dec_18_10",
                                        "dt",
                                        "ts_0",
                                        "ts_3",
                                        "tm"};
  // Explicit cast
  for (auto col_type : col_type_strings) {
    TQueryResult result_set;
    sql(result_set,
        "select cast(" + std::string(col_type) +
            " as text) || ' years' from numeric_to_string_test order by rowid asc;");
    TQueryResult expected_result_set;
    sql(expected_result_set,
        "select " + std::string(col_type) +
            "_str || ' years' from numeric_to_string_test order by rowid asc;");
    ASSERT_EQ(result_set.row_set.columns.size(), size_t(1));
    ASSERT_EQ(result_set.row_set.columns.size(),
              expected_result_set.row_set.columns.size());
    EXPECT_EQ(result_set.row_set.columns[0].data.str_col,
              expected_result_set.row_set.columns[0].data.str_col);
  }

  // Implicit cast
  for (auto col_type : col_type_strings) {
    TQueryResult result_set;
    sql(result_set,
        "select " + std::string(col_type) +
            " || ' years' from numeric_to_string_test order by rowid asc;");
    TQueryResult expected_result_set;
    sql(expected_result_set,
        "select " + std::string(col_type) +
            "_str || ' years' from numeric_to_string_test order by rowid asc;");
    ASSERT_EQ(result_set.row_set.columns.size(), size_t(1));
    ASSERT_EQ(result_set.row_set.columns.size(),
              expected_result_set.row_set.columns.size());
    EXPECT_EQ(result_set.row_set.columns[0].data.str_col,
              expected_result_set.row_set.columns[0].data.str_col);
  }

  // Direct equals
  for (auto col_type : col_type_strings) {
    // Last value is false/0 since in SQL null != null
    sqlAndCompareResult("select coalesce(cast(cast(" + std::string(col_type) +
                            " as text) || ' years' = " + std::string(col_type) +
                            "_str || ' years' as int), -1) from numeric_to_string_test "
                            "order by rowid asc;",
                        {{int64_t(1)}, {int64_t(1)}, {int64_t(-1)}});
  }
}

TEST_P(StringFunctionTest, StringFunctionEqualsFilterLHS) {
  sqlAndCompareResult(
      "select first_name, last_name from string_function_test_people "
      "where lower(country_code) = 'us';",
      {{"JOHN", "SMITH"}, {"John", "Banks"}});

  sqlAndCompareResult(
      "select COUNT(*) from string_function_test_people "
      "where initcap(first_name) = 'John';",
      {{int64_t(3)}});

  sqlAndCompareResult(
      "select lower(first_name), first_name from string_function_test_people "
      "where upper('johN') = first_name;",
      {{"john", "JOHN"}, {"john", "JOHN"}});
}

TEST_P(StringFunctionTest, StringFunctionEqualsFilterRHS) {
  sqlAndCompareResult(
      "select first_name, last_name from string_function_test_people "
      "where 'us' = lower(country_code);",
      {{"JOHN", "SMITH"}, {"John", "Banks"}});

  sqlAndCompareResult(
      "select COUNT(*) from string_function_test_people "
      "where 'John' = initcap(first_name);",
      {{int64_t(3)}});

  sqlAndCompareResult(
      "select lower(first_name), first_name from string_function_test_people "
      "where first_name = upper('johN');",
      {{"john", "JOHN"}, {"john", "JOHN"}});
}

TEST_P(StringFunctionTest, StringFunctionFilterBothSides) {
  sqlAndCompareResult(
      "select first_name, last_name from string_function_test_people "
      "where lower('US') = lower(country_code);",
      {{"JOHN", "SMITH"}, {"John", "Banks"}});

  sqlAndCompareResult(
      "select COUNT(*) from string_function_test_people "
      "where initcap('joHN') = initcap(first_name);",
      {{int64_t(3)}});

  sqlAndCompareResult(
      "select first_name, lower(first_name), first_name from "
      "string_function_test_people "
      "where upper(first_name) = upper('johN');",
      {{"JOHN", "john", "JOHN"}, {"John", "john", "John"}, {"JOHN", "john", "JOHN"}});

  sqlAndCompareResult(
      "select first_name, full_name from string_function_test_people "
      "where initcap(first_name) = split_part(full_name, ' ', 1);",
      {{"JOHN", "John SMITH"},
       {"John", "John BANKS"},
       {"JOHN", "John WILSON"},
       {"Sue", "Sue SMITH"}});
}

TEST_P(StringFunctionTest, MultipleFilters) {
  sqlAndCompareResult(
      "select first_name, last_name from string_function_test_people "
      "where lower(country_code) = 'us' or lower(first_name) = 'sue';",
      {{"JOHN", "SMITH"}, {"John", "Banks"}, {"Sue", "Smith"}});

  sqlAndCompareResult(
      "select first_name, last_name from string_function_test_people "
      "where lower(country_code) = 'us' or upper(country_code) = 'CA';",
      {{"JOHN", "SMITH"}, {"John", "Banks"}, {"JOHN", "Wilson"}, {"Sue", "Smith"}});
}

TEST_P(StringFunctionTest, MixedFilters) {
  sqlAndCompareResult(
      "select first_name, last_name from string_function_test_people "
      "where lower(country_code) = 'ca' and age > 20;",
      {{"Sue", "Smith"}});
}

TEST_P(StringFunctionTest, ChainedOperators) {
  sqlAndCompareResult(
      "select initcap(split_part(full_name, ' ', 2)) as surname from "
      "string_function_test_people order by id asc;",
      {{"Smith"}, {"Banks"}, {"Wilson"}, {"Smith"}});

  sqlAndCompareResult(
      "select upper(split_part(split_part(regexp_substr(raw_email, "
      "'[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
      "1, -1), '@', -1), '.', 1)) as upper_domain from "
      "string_function_test_people order by id asc;",
      {{"OMNISCI"}, {"MAPD"}, {"GEOPS"}, {"EXAMPLE"}});

  sqlAndCompareResult(
      "select lower(split_part(split_part(regexp_substr(raw_email, "
      "'[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
      "1, -1), '@', -1), '.', 2)), "
      "upper(split_part(split_part(regexp_substr(raw_email, "
      "'[[:alnum:]._-]+@[[:alnum:]]+.[[:alnum:]]+', "
      "1, -1), '@', -1), '.', 1)) as upper_domain from "
      "string_function_test_people where substring(replace(raw_email, 'com', "
      "'org') from -3 for 3) = 'org' order by id asc;",
      {{"com", "OMNISCI"}, {"com", "MAPD"}});
}

TEST_P(StringFunctionTest, CaseStatement) {
  // Single column, string op only on output
  sqlAndCompareResult(
      "select case when first_name = 'JOHN' then lower(first_name) else "
      "upper(first_name) end "
      "as case_stmt from string_function_test_people order by id asc;",
      {{"john"}, {"JOHN"}, {"john"}, {"SUE"}});

  // Single column, string ops on inputs and outputs, with additional literal
  sqlAndCompareResult(
      "select case when split_part(us_phone_number, '-', 2) = '614' then "
      "split_part(us_phone_number, '-', 3) "
      "when split_part(us_phone_number, '-', 3) = '2144' then "
      "substring(us_phone_number from 1 for 3) else "
      "'Surprise' end as case_stmt from string_function_test_people order by "
      "id asc;",
      {{"555"}, {"Surprise"}, {"9814"}, {"2282"}});

  // Multi-column, string ops on inputs and outputs, with null and additional literal
  sqlAndCompareResult(
      "select case when split_part(us_phone_number, '-', 2) = trim('614 ') "
      "then null "
      "when split_part(us_phone_number, '-', 3) = '214' || '4' then "
      "regexp_substr(zip_plus_4, "
      "'^[[:digit:]]+') else upper(country_code) end as case_stmt from "
      "string_function_test_people "
      "order by id asc;",
      {{"90210"}, {"US"}, {Null}, {Null}});

  // [QE-359] Currently unsupported string op on output of case statement
  EXPECT_ANY_THROW(
      sql("select upper(case when name like 'United%' then 'The ' || name "
          "else name end) from string_function_test_countries order by id asc;"));
}

TEST_P(StringFunctionTest, GroupBy) {
  sqlAndCompareResult(
      "select lower(first_name), count(*) from string_function_test_people "
      "group by lower(first_name) order by 2 desc;",
      {{"john", int64_t(3)}, {"sue", int64_t(1)}});

  sqlAndCompareResult(
      "select regexp_substr(raw_email, "
      "'([[:alnum:]._-]+)@([[:alnum:]]+).([[:alnum:]]+)', 1, 1, 'ie', 3) as tld, "
      "count(*) as n from string_function_test_people group by tld order by tld asc;",
      {{"com", int64_t(3)}, {"net", int64_t(1)}});
}

TEST_P(StringFunctionTest, Join) {
  // Turn off loop joins with the third parameter to ensure hash joins work

  // Ensure loop join throws
  EXPECT_ANY_THROW(
      sql("select /*+ disable_loop_join */ COUNT(*) from string_function_test_people, "
          "string_function_test_countries;"));

  // Both sides
  sqlAndCompareResult(
      "select first_name, name as country_name "
      "from string_function_test_people a "
      "join string_function_test_countries b on lower(country_code) = "
      "lower(code) order by a.id asc;",
      {{"JOHN", "United States"},
       {"John", "United States"},
       {"JOHN", "Canada"},
       {"Sue", "Canada"}});

  // String op lhs
  sqlAndCompareResult(
      "select first_name, name as country_name "
      "from string_function_test_people a "
      "join string_function_test_countries b on upper(country_code) = code order by a.id "
      "asc;",
      {{"JOHN", "United States"}, {"John", "United States"}});

  // String op rhs
  sqlAndCompareResult(
      "select first_name, name as country_name "
      "from string_function_test_people a "
      "join string_function_test_countries b on country_code = lower(code) order by a.id "
      "asc;",
      {{"JOHN", "United States"}});

  // Two arg baseline join
  sqlAndCompareResult(
      "select first_name, upper(name) as upper_country_name "
      "from string_function_test_people a "
      "join string_function_test_countries b on lower(country_code) = "
      "lower(code) and upper(country_code) = repeat(code, 1) order by a.id asc;",
      {{"JOHN", "UNITED STATES"}, {"John", "UNITED STATES"}});

  // [QE-359] Should throw when join predicate contains string op
  // on top of case statement
  EXPECT_ANY_THROW(
      sql("select count(*) from string_function_test_people a inner join "
          "string_function_test_countries b on lower(code) = lower(case when "
          "code = 'US' then repeat(code, 2) else code end);"));
}

TEST_P(StringFunctionTest, SelectLiteral) {
  sqlAndCompareResult(
      "select first_name, lower('SMiTH') from string_function_test_people;",
      {{"JOHN", "smith"}, {"John", "smith"}, {"JOHN", "smith"}, {"Sue", "smith"}});
}

TEST_P(StringFunctionTest, UpdateLowercase_EncodedColumnOnly) {
  sql("update string_function_test_people set country_code = lower(country_code);");
  sqlAndCompareResult("select country_code from string_function_test_people;",
                      {{"us"}, {"us"}, {"ca"}, {"ca"}});
}
/**
 * UPDATE statements with at least one non-encoded column follow a different code path
 * from those with only encoded columns (see StorageIOFacility::yieldUpdateCallback for
 * more details).InsertIntoSelectLowercase
 */
TEST_P(StringFunctionTest, UpdateLowercase_EncodedAndNonEncodedColumns) {
  sql("update string_function_test_people set last_name = last_name, country_code = "
      "lower(country_code);");
  sqlAndCompareResult(
      "select last_name, country_code from string_function_test_people;",
      {{"SMITH", "us"}, {"Banks", "us"}, {"Wilson", "ca"}, {"Smith", "ca"}});
}
// TODO-BE-4206: Re-enable after clear definition around handling non-ASCII characters
TEST_P(StringFunctionTest, DISABLED_LowercaseNonAscii) {
  sql("insert into string_function_test_people values('Ħ', 'Ħ', 25, 'GB')");
  sqlAndCompareResult(
      "select lower(first_name), last_name from string_function_test_people where "
      "country_code = 'GB';",
      {{"ħ", "Ħ"}});
}

TEST_P(StringFunctionTest, LowercaseNoneEncodedTextColumn) {
  sqlAndCompareResult(
      "select lower(last_name) from string_function_test_people order by id asc;",
      {{"smith"}, {"banks"}, {"wilson"}, {"smith"}});
}

TEST_P(StringFunctionTest, ChainNoneEncodedTextColumn) {
  sqlAndCompareResult(
      "select reverse(initcap(last_name)) from string_function_test_people order by id "
      "asc;",
      {{"htimS"}, {"sknaB"}, {"nosliW"}, {"htimS"}});
}

TEST_P(StringFunctionTest, NoneEncodedGroupByNoStringOps) {
  // No string ops
  sqlAndCompareResult(
      "select encode_text(lang) as g, count(*) as n from "
      "string_function_test_countries group by g order by g asc;",
      {{"EN", int64_t(1)}, {"de", int64_t(1)}, {"en", int64_t(2)}});
}

TEST_P(StringFunctionTest, NoneEncodedGroupByNullsNoStringOps) {
  // No string ops
  sqlAndCompareResult(
      "select encode_text(short_name) as g, count(*) as n from "
      "string_function_test_countries group by g order by g asc nulls last;",
      {{"Canada", int64_t(1)},
       {"Germany", int64_t(1)},
       {"UK", int64_t(1)},
       {Null, int64_t(1)}});
}

TEST_P(StringFunctionTest, NoneEncodedGroupByStringOps) {
  // String ops
  sqlAndCompareResult(
      "select lower(lang) as g, count(*) as n from "
      "string_function_test_countries group by g order by n desc;",
      {{"en", int64_t(3)}, {"de", int64_t(1)}});

  // With inert wrapping of ENCODE_TEXT
  sqlAndCompareResult(
      "select encode_text(lower(lang)) as g, count(*) as n from "
      "string_function_test_countries group by g order by n desc;",
      {{"en", int64_t(3)}, {"de", int64_t(1)}});

  // With inner ENCODE_TEXT cast
  sqlAndCompareResult(
      "select lower(encode_text(lang)) as g, count(*) as n from "
      "string_function_test_countries group by g order by n desc;",
      {{"en", int64_t(3)}, {"de", int64_t(1)}});

  sqlAndCompareResult(
      "select initcap(last_name) as g, count(*) as n from "
      "string_function_test_people group by g order by g asc;",
      {{"Banks", int64_t(1)}, {"Smith", int64_t(2)}, {"Wilson", int64_t(1)}});

  // String ops with filter
  sqlAndCompareResult(
      "select initcap(last_name) as g, count(*) as n from "
      "string_function_test_people where encode_text(last_name) <> "
      "upper(last_name) "
      "group by g order by g asc;",
      {{"Banks", int64_t(1)}, {"Smith", int64_t(1)}, {"Wilson", int64_t(1)}});
}

TEST_P(StringFunctionTest, NoneEncodedGroupByNullsStringOps) {
  // String ops
  sqlAndCompareResult(
      "select substring(short_name from 4 for 5) as g, count(*) as n from "
      "string_function_test_countries group by g order by g asc nulls last;",
      {{"ada", int64_t(1)}, {"many", int64_t(1)}, {Null, int64_t(2)}});
}

TEST_P(StringFunctionTest, NoneEncodedEncodedEquality) {
  // None encoded = encoded, no string ops
  sqlAndCompareResult(
      "select name from string_function_test_countries where "
      "name = short_name order by id asc;",
      {{"Canada"}, {"Germany"}});

  // Encoded = none-encoded, no string ops
  sqlAndCompareResult(
      "select UPPER(short_name) from string_function_test_countries where "
      "short_name = name order by id asc;",
      {{"CANADA"}, {"GERMANY"}});

  // None encoded = encoded, string ops both sides
  sqlAndCompareResult(
      "select upper(last_name) from string_function_test_people where "
      "initcap(last_name) = split_part(initcap(full_name), ' ', 2) order by id asc;",
      {{"SMITH"}, {"BANKS"}, {"WILSON"}, {"SMITH"}});

  // Encoded = none encoded, string ops both sides
  sqlAndCompareResult(
      "select upper(last_name) from string_function_test_people where "
      "split_part(initcap(full_name), ' ', 2) = initcap(last_name) order by id asc;",
      {{"SMITH"}, {"BANKS"}, {"WILSON"}, {"SMITH"}});

  // None encoded = encoded, string ops one side
  sqlAndCompareResult(
      "select repeat(largest_city, 2) from string_function_test_countries where "
      "initcap(largest_city) = capital order by id asc;",
      {{"LONDONLONDON"}, {"BerlinBerlin"}});

  // Encoded = none encoded, string ops one side
  sqlAndCompareResult(
      "select substring(capital from 0 for 3) from "
      "string_function_test_countries where "
      "capital = initcap(largest_city) order by id asc;",
      {{"Lon"}, {"Ber"}});
}

TEST_P(StringFunctionTest, NoneEncodedCaseStatementsNoStringOps) {
  // None-encoded + none-encoded, no string ops

  // Note if we don't project out the id column we get a
  // ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED: Columnar conversion not
  // supported for variable length types error
  // Need to address this separately (precedes the string function work in
  // master)
  sqlAndCompareResult(
      "select id, case when id <= 2 then short_name else lang end from "
      "string_function_test_countries order by id asc;",
      {{int64_t(1), Null},
       {int64_t(2), "Canada"},
       {int64_t(3), "en"},
       {int64_t(4), "de"}});

  // None-encoded + none-encoded + literal
  sqlAndCompareResult(
      "select id, case when id = 1 then 'USA' when id <= 3 then short_name "
      "else lang "
      "end from string_function_test_countries order by id asc;",
      {{int64_t(1), "USA"},
       {int64_t(2), "Canada"},
       {int64_t(3), "UK"},
       {int64_t(4), "de"}});

  // Dict-encoded + none-encoded + literal
  sqlAndCompareResult(
      "select case when id <= 2 then name when id <= 3 then short_name else 'DE' end "
      "from string_function_test_countries order by id asc;",
      {{"United States"}, {"Canada"}, {"UK"}, {"DE"}});

  // Group by
  // Dict-encoded + none-encoded + literal
  sqlAndCompareResult(
      "select case when lang = 'en' then lang when code = 'ca' then 'en' else "
      "code end "
      "as g, count(*) as n from string_function_test_countries group by g order by g "
      "asc;",
      {{"dE", int64_t(1)}, {"en", int64_t(3)}});
}

TEST_P(StringFunctionTest, NoneEncodedCaseStatementsStringOps) {
  // None-encoded + none-encoded, no string ops
  sqlAndCompareResult(
      "select case when id <= 2 then lower(short_name) else upper(lang) end from "
      "string_function_test_countries order by id asc;",
      {{Null}, {"canada"}, {"EN"}, {"DE"}});

  // None-encoded + none-encoded + literal
  sqlAndCompareResult(
      "select case when id = 1 then initcap('USA') when id <= 3 then "
      "upper(short_name) "
      "else reverse(lang) end from string_function_test_countries order by id asc;",
      {{"Usa"}, {"CANADA"}, {"UK"}, {"ed"}});

  // Dict-encoded + none-encoded + literal
  sqlAndCompareResult(
      "select case when id <= 2 then initcap(repeat(name, 2)) when id <= 3 then "
      "substring(short_name from 2 for 1) else 'DE' "
      "end from string_function_test_countries order by id asc;",
      {{"United Statesunited States"}, {"Canadacanada"}, {"K"}, {"DE"}});

  // Group by
  // Dict-encoded + none-encoded + literal
  sqlAndCompareResult(
      "select case when lang = 'en' then upper(lang) when code = 'ca' then 'en' else "
      "'Z' || trim(leading 'd' from repeat(code, 2)) end "
      "as g, count(*) as n from string_function_test_countries group by g order by g "
      "asc;",
      {{"EN", int64_t(2)}, {"ZEdE", int64_t(1)}, {"en", int64_t(1)}});
}

TEST_P(StringFunctionTest, LowercaseNullColumn) {
  sqlAndCompareResult(
      "select lower(zip_plus_4), last_name from string_function_test_people where "
      "id = 4;",
      {{Null, "Smith"}});
}

TEST_P(StringFunctionTest, SelectLowercase_StringFunctionsDisabled) {
  const auto previous_string_function_state = g_enable_string_functions;
  ScopeGuard reset_string_function_state = [&previous_string_function_state] {
    g_enable_string_functions = previous_string_function_state;
   };
  g_enable_string_functions = false;
  queryAndAssertException("select lower(first_name) from string_function_test_people;",
                          "Function LOWER not supported.");
}

TEST_P(StringFunctionTest, SelectLowercaseNoneEncoded_MoreRowsThanWatchdogLimit) {
  const auto previous_watchdog_state = g_enable_watchdog;
  const auto previous_none_encoded_translation_limit =
      g_watchdog_none_encoded_string_translation_limit;
  ScopeGuard reset_watchdog_state = [&previous_watchdog_state,
                                     &previous_none_encoded_translation_limit] {
    g_enable_watchdog = previous_watchdog_state;
    g_watchdog_none_encoded_string_translation_limit =
        previous_none_encoded_translation_limit;
  };
  g_enable_watchdog = true;
  g_watchdog_none_encoded_string_translation_limit = 3;
  std::ostringstream expected_error;
  expected_error
      << "Query requires one or more casts between none-encoded and "
         "dictionary-encoded "
      << "strings, and the estimated table size (4 rows) "
      << "exceeds the configured watchdog none-encoded string translation limit of "
      << g_watchdog_none_encoded_string_translation_limit << " rows.";
  queryAndAssertException("select lower(last_name) from string_function_test_people;",
                          expected_error.str());
}

TEST_P(StringFunctionTest, UDF_ExpandDefaults) {
  /*
  // ltrim has a defaulted second argument, test the parsing in group clause
  select ltrim(name) from heavyai_us_states;
  select ltrim(name) from heavyai_us_states group by 1;
  select ltrim(name) from heavyai_us_states group by name;
  select ltrim(name) as s from heavyai_us_states;
  select ltrim(name) as s from heavyai_us_states group by 1;
  select ltrim(name) as s from heavyai_us_states group by s;
  select ltrim(name) as s from heavyai_us_states group by name;

  // baseline verification
  select ltrim(name, 'New') from heavyai_us_states;
  select ltrim(name, 'New') from heavyai_us_states group by 1;
  select ltrim(name, 'New') from heavyai_us_states group by name;
  select ltrim(name, 'New') as s from heavyai_us_states;
  select ltrim(name, 'New') as s from heavyai_us_states group by 1;
  select ltrim(name, 'New') as s from heavyai_us_states group by s;
  select ltrim(name, 'New') as s from heavyai_us_states group by name;

  // note: "group by 1" causes problems with "order by"
  */

  // Second argument is defaulted to ""
  std::vector<std::vector<NullableTargetValue>> expected_result_set{
      {"JOHN"}, {"JOHN"}, {"John"}, {"Sue"}};
  std::vector<std::vector<NullableTargetValue>> expected_result_set_unique{
      {"JOHN"}, {"John"}, {"Sue"}};

  sqlAndCompareResult(
      "select ltrim(first_name) from string_function_test_people where id < 5 "
      "order by first_name asc;",
      expected_result_set);

  // just test the parsing ...
  // => an order by clause fails because of '1' substitution
  // => result set comparison could theorically fail due to ordering issues
  // compare_result_set(expected_result_set_unique, result_set2);
  sql("select ltrim(first_name) from string_function_test_people where id < 5 "
      "group by 1;");

  sqlAndCompareResult(
      "select ltrim(first_name) from string_function_test_people where id < 5 "
      "group by first_name order by first_name asc;",
      expected_result_set_unique);

  sqlAndCompareResult(
      "select ltrim(first_name) as s from string_function_test_people where id < 5 order "
      "by first_name asc;",
      expected_result_set);

  // just test the parsing ...
  // => an order by clause fails because of '1' substitution
  // => result set comparison could theorically fail due to ordering issues
  // compare_result_set(expected_result_set_unique, result_set5);
  sql("select ltrim(first_name) as s from string_function_test_people where id < 5 group "
      "by 1;");

  sqlAndCompareResult(
      "select ltrim(first_name) as s from string_function_test_people where id < 5 group "
      "by s order by ltrim(first_name) asc;",
      expected_result_set_unique);

  sqlAndCompareResult(
      "select ltrim(first_name) as s from string_function_test_people where id < 5 group "
      "by first_name order by first_name asc;",
      expected_result_set_unique);

  // fully specified call
  expected_result_set =
      std::vector<std::vector<NullableTargetValue>>{{"HN"}, {"ohn"}, {"HN"}, {"Sue"}};
  expected_result_set_unique =
      std::vector<std::vector<NullableTargetValue>>{{"HN"}, {"ohn"}, {"Sue"}};

  sqlAndCompareResult(
      "select ltrim(first_name, 'JO') from string_function_test_people where id "
      "< 5 order by id asc;",
      expected_result_set);

  // just test the parsing ...
  // => an order by clause fails because of '1' substitution
  // => result set comparison could theorically fail due to ordering issues
  // compare_result_set(expected_result_set, result_set2);
  sql("select ltrim(first_name, 'JO') from string_function_test_people where id "
      "< 5 group by 1;");

  sqlAndCompareResult(
      "select ltrim(first_name, 'JO') from string_function_test_people where id "
      "< 5 group by first_name order by first_name asc;",
      expected_result_set_unique);

  sqlAndCompareResult(
      "select ltrim(first_name, 'JO') as s from string_function_test_people "
      "where id < 5 order by id asc;",
      expected_result_set);

  // just test the parsing ...
  // => an order by clause fails because of '1' substitution
  // => result set comparison could theorically fail due to ordering issues
  // compare_result_set(expected_result_set, result_set5);
  sql("select ltrim(first_name, 'JO') as s "
      "from string_function_test_people "
      "where id < 5 "
      "group by 1;");

  // the grouping changes the intermediate results, so use a special result set
  sqlAndCompareResult(
      "select ltrim(first_name, 'JO') as s "
      "from string_function_test_people "
      "where id < 5 "
      "group by s "
      "order by ltrim(first_name, 'JO') asc;",
      {{"HN"}, {"Sue"}, {"ohn"}});

  sqlAndCompareResult(
      "select ltrim(first_name, 'JO') as s "
      "from string_function_test_people "
      "where id < 5 "
      "group by first_name "
      "order by first_name asc;",
      expected_result_set_unique);
}

// EXPANDED/REPLACED string operation tests
TEST_P(StringFunctionTest, contains) {
  sqlAndCompareResult("select contains('abcdefghijklmn', 'def');", {{True}});

  sqlAndCompareResult("select contains('abcdefghijklmn', 'xyz');", {{False}});

  sqlAndCompareResult("select contains('abcdefghijklmn', 'abc');", {{True}});

  sqlAndCompareResult("select contains('abcdefghijklmn', 'mnop');", {{False}});

  // Edge case: empty strings
  sqlAndCompareResult("select contains('', '');", {{True}});

  // contains is non-standard SQL, and returns -128 for NULL strings
  int64_t kNull = -128;

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult(
      "select contains(zip_plus_4, '94104') from string_function_test_people;",
      {{False}, {True}, {kNull}, {kNull}});

  // Note: pattern requires literal string so this is not currently valid
  //   "select startswith('94104-8123', zip_plus_4) from string_function_test_people;"
}

TEST_P(StringFunctionTest, endswith) {
  sqlAndCompareResult("select endswith('abcdefghijklmn', 'lmn');", {{True}});

  sqlAndCompareResult("select endswith('abcdef', 'aaabcdef');", {{False}});

  sqlAndCompareResult("select endswith('abcdefghijklmn', 'abcdefghijklmn');", {{True}});

  sqlAndCompareResult("select endswith('abcdefghijklmn', 'lmnop');", {{False}});

  // Edge case: empty strings
  sqlAndCompareResult("select endswith('', '');", {{True}});

  // endswith is non-standard SQL, and returns -128 for NULL strings
  int64_t kNull = -128;

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult(
      "select endswith(zip_plus_4, '94104') from string_function_test_people;",
      {{False}, {False}, {kNull}, {kNull}});

  // Note: pattern requires literal string so this is not currently valid
  //   "select endswith('94104-8123', zip_plus_4) from string_function_test_people;"

TEST_P(StringFunctionTest, lcase) {
  sqlAndCompareResult(
      "select lcase(largest_city) from string_function_test_countries where "
      "code = 'US';",
      {{"new york city"}});

  // Edge case: empty string
  sqlAndCompareResult("select lcase('');", {{Null}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult("select lcase(zip_plus_4) from string_function_test_people;",
                      {{"90210-7743"}, {"94104-8123"}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, left) {
  sqlAndCompareResult("select left('abcdef', -2);", {{Null}});

  sqlAndCompareResult("select left('abcdef', 0);", {{Null}});

  sqlAndCompareResult("select left('abcdef', 2);", {{"ab"}});

  sqlAndCompareResult("select left('abcdef', 10);", {{"abcdef"}});

  // Edge case: empty string
  sqlAndCompareResult("select left('', 2);", {{Null}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult("select left(zip_plus_4, 4) from string_function_test_people;",
                      {{"9021"}, {"9410"}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, len) {
  // LEN is an alias for LENGTH, just test the alias as
  //    LENGTH functionality should be covered by other tests
  sqlAndCompareResult("select len('abcdefghi');", {{int64_t(9)}});
  // Edge case: empty strings
  sqlAndCompareResult("select len('');", {{int64_t(0)}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult(
      "select len(zip_plus_4) from string_function_test_people;",
      {{int64_t(10)}, {int64_t(10)}, {int64_t(-2147483648)}, {int64_t(-2147483648)}});
}

TEST_P(StringFunctionTest, max) {
  sqlAndCompareResult("select MAX(7,4);", {{i(7)}});

  sqlAndCompareResult("select MAX(-7,220);", {{i(220)}});

  sqlAndCompareResult("select MAX('bf','sh');", {{"sh"}});

  sqlAndCompareResult("select MAX(123,456);", {{i(456)}});

  sqlAndCompareResult("select MIN(count(*),count(*)) from string_function_test_people;",
                      {{i(4)}});

  // this will assert as the types mismatch
  queryAndAssertException("select MAX(3,'f');", "Unable to parse f to INTEGER");

  // Edge case: empty strings
  sqlAndCompareResult("select max('', '');", {{""}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult(
      "select max(zip_plus_4, zip_plus_4) from string_function_test_people;",
      {{"90210-7743"}, {"94104-8123"}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, mid) {
  // MID is an alias for SUBSTRING, just test the alias as
  //    substring functionality should be covered by other tests
  sqlAndCompareResult("select mid('abcdef', 2,4);", {{"bcde"}});

  sqlAndCompareResult("select mid('abcdef', 4);", {{"def"}});

  // Edge case: empty strings
  sqlAndCompareResult("select mid('', 4);", {{Null}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult("select mid(zip_plus_4, 3,5) from string_function_test_people;",
                      {{"210-7"}, {"104-8"}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, min) {
  sqlAndCompareResult("select MIN(7,4);", {{i(4)}});

  sqlAndCompareResult("select MIN(-7,220);", {{i(-7)}});

  sqlAndCompareResult("select MIN('bf','sh');", {{"bf"}});

  sqlAndCompareResult("select MIN(123,456);", {{i(123)}});

  sqlAndCompareResult("select MIN(count(*),count(*)) from string_function_test_people;",
                      {{i(4)}});

  // this will assert as the types mismatch
  queryAndAssertException("select MIN(3,'f');", "Unable to parse f to INTEGER");

  // Edge case: empty strings
  sqlAndCompareResult("select min('', '');", {{""}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult(
      "select min(zip_plus_4, zip_plus_4) from string_function_test_people;",
      {{"90210-7743"}, {"94104-8123"}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, right) {
  sqlAndCompareResult("select right('abcdef', -2);", {{Null}});

  sqlAndCompareResult("select right('abcdef', 0);", {{Null}});

  sqlAndCompareResult("select right('abcdef', 2);", {{"ef"}});

  sqlAndCompareResult("select right('abcdef', 10);", {{"abcdef"}});

  // Edge case: empty string
  sqlAndCompareResult("select right('', 2);", {{Null}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult("select right(zip_plus_4, 4) from string_function_test_people;",
                      {{"7743"}, {"8123"}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, space) {
  sqlAndCompareResult("select space(0);", {{Null}});

  sqlAndCompareResult("select space(1);", {{" "}});

  sqlAndCompareResult("select space(8);", {{"        "}});
  // this will assert as the -1 is invalid
  queryAndAssertException("select space(-1);", "Number of repeats must be >= 0");

  // Edge case: non-fixed value will throw ...
  //   because SPACE is based upon REPEAT which does not accept non-fixed values
  queryAndAssertException(
      "select space(count(*)) from string_function_test_people;",
      "Error instantiating REPEAT operator. Currently only column inputs are allowed for "
      "argument 'operand', but a column input was received for argument 'num_repeats'.");
}

TEST_P(StringFunctionTest, split) {
  // SPLIT is an alias for SPLIT_PART, mainly test the alias as
  // SPLIT_PART functionality should be covered by other tests
  sqlAndCompareResult("select split('123-345-6789', '-', 2);", {{"345"}});

  // Edge case: empty strings
  sqlAndCompareResult("select split('', '', 3);", {{Null}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult("select split(zip_plus_4, '-',2) from string_function_test_people;",
                      {{"7743"}, {"8123"}, {Null}, {Null}});

  // Note: pattern requires literal string so this is not currently valid
  //   "select endswith('94104-8123', zip_plus_4) from string_function_test_people;"
}

TEST_P(StringFunctionTest, startswith) {
  sqlAndCompareResult("select startswith('abcdef', 'abc');", {{True}});

  sqlAndCompareResult("select startswith('abcdef', 'abcdef');", {{True}});

  sqlAndCompareResult("select startswith('abcdef', 'abcdefghi');", {{False}});

  sqlAndCompareResult("select startswith('abcdef', 'xyz');", {{False}});

  // Edge case: empty strings
  sqlAndCompareResult("select startswith('', '');", {{True}});

  // startswith is non-standard SQL, and returns -128 for NULL strings
  int64_t kNull = -128;

  // NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult(
      "select startswith(zip_plus_4, '94104') from string_function_test_people;",
      {{False}, {True}, {kNull}, {kNull}});

  // Note: pattern requires literal string so this is not currently valid
  //   "select startswith('94104-8123', zip_plus_4) from string_function_test_people;"
}

TEST_P(StringFunctionTest, substr) {
  // SUBSTR is an alias for SUBSTRING, mostly test the alias as
  //    substring functionality should be covered by other tests
  sqlAndCompareResult("select substr('abcdef', 2,4);", {{"bcde"}});

  // Edge case: empty strings
  sqlAndCompareResult("select substr('', 3, 5);", {{Null}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult("select substr(zip_plus_4, 3,5) from string_function_test_people;",
                      {{"210-7"}, {"104-8"}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, ucase) {
  sqlAndCompareResult(
      "select ucase(largest_city) from string_function_test_countries where "
      "code = 'US';",
      {{"NEW YORK CITY"}});

  // Edge case: empty string
  sqlAndCompareResult("select ucase('');", {{Null}});

  // Edge case: NULL string: zip_plus_4 has NULL values in some rows
  sqlAndCompareResult("select ucase(zip_plus_4) from string_function_test_people;",
                      {{"90210-7743"}, {"94104-8123"}, {Null}, {Null}});
}

TEST_P(StringFunctionTest, StrtokToArrayCommonCases) {
  // for (auto dt : {ExecutorDeviceType::CPU, /* ExecutorDeviceType::GPU */}) {
  sqlAndCompareResult("select strtok_to_array('a.b.c', '.');",
                      {{array({"a", "b", "c"})}});

  sqlAndCompareResult("select strtok_to_array('hello@world.com', '.@');",
                      {{array({"hello", "world", "com"})}});

  /* empty text */
  sqlAndCompareResult("select strtok_to_array('', '.@');", {{array({})}});

  /* empty delimiters */
  sqlAndCompareResult("select strtok_to_array('hello.world', '');", {{array({})}});
}

TEST_P(StringFunctionTest, StrtokToArrayTextEncodingNone) {
  sqlAndCompareResult("select strtok_to_array(name, ' ') from text_enc_test;",
                      {{array({"United", "States"})},
                       {array({"Canada"})},
                       {array({"United", "Kingdom"})},
                       {array({"Germany"})}});
}

TEST_P(StringFunctionTest, StrtokToArrayTextEncodingDict) {
  sqlAndCompareResult("select strtok_to_array(code, '> <') from text_enc_test;",
                      {{array({"US", "USA"})},
                       {array({"CA", "CAN"})},
                       {array({"GB", "GBR"})},
                       {array({"DE", "DEN"})}});
 }

TEST_P(StringFunctionTest, DISABLED_CardinalityStrtokToArrayTextEncodingDict) {
  sqlAndCompareResult(
      "select cardinality(strtok_to_array(code, '> <')) from text_enc_test;",
      {{int64_t(2)}, {int64_t(2)}, {int64_t(2)}, {int64_t(2)}});
}

TEST_P(StringFunctionTest, StrtokToArray_UDF) {
   // Apply STRTOK_TO_ARRAY on the output of an UDF
  sqlAndCompareResult(
      "select strtok_to_array(udf_identity(name), ' ') from text_enc_test;",
      {{array({"United", "States"})},
       {array({"Canada"})},
       {array({"United", "Kingdom"})},
       {array({"Germany"})}});
}

TEST_P(StringFunctionTest, UDFConcat) {
  sqlAndCompareResult("select 'hello ' || udf_identity(name) from text_enc_test;",
                      {{"hello United States"},
                       {"hello Canada"},
                       {"hello United Kingdom"},
                       {"hello Germany"}});
}

TEST_P(StringFunctionTest, AlterTable_RuntimeFunction) {
  sql("drop table if exists alter_column_test;");
  sql("create table alter_column_test (code TEXT ENCODING NONE);");
  sql("insert into alter_column_test values ('US USA');");
  sql("insert into alter_column_test values ('CA CAN');");
  sql("insert into alter_column_test values ('GB GBR');");
  sql("insert into alter_column_test values ('DE DEN');");
  sql("alter table alter_column_test add column tokenized_text TEXT[] ENCODING "
      "DICT(32);");
  sql("update alter_column_test set tokenized_text = strtok_to_array(code, ' ');");

  sqlAndCompareResult("select tokenized_text from alter_column_test;",
                      {{array({"US", "USA"})},
                       {array({"CA", "CAN"})},
                       {array({"GB", "GBR"})},
                       {array({"DE", "DEN"})}});
}

TEST_P(StringFunctionTest, TextEncodingNoneCopyUDF) {
  sqlAndCompareResult(
      "select text_encoding_none_copy(largest_city) from "
      "string_function_test_countries;",
      {{"New York City"}, {"TORONTO"}, {"LONDON"}, {"Berlin"}});
}

// TODO: Enable after fixing issue with text_encoding_none_concat returning wrong results
TEST_P(StringFunctionTest, DISABLED_TextEncodingNoneConcatUDF) {
  sqlAndCompareResult(
      "select text_encoding_none_concat('city:', largest_city) from "
      "string_function_test_countries",
      {{"city: New York City"}, {"city: TORONTO"}, {"city: LONDON"}, {"city: Berlin"}});
}

TEST_P(StringFunctionTest, TextEncodingNoneLengthUDF) {
  sqlAndCompareResult(
      "select text_encoding_none_length(largest_city) from "
      "string_function_test_countries;",
      {{int64_t(13)}, {int64_t(7)}, {int64_t(6)}, {int64_t(6)}});
}

TEST_P(StringFunctionTest, TextEncodingDictConcatUDF) {
  sqlAndCompareResult(
      "select text_encoding_dict_concat(short_name, name) from "
      "text_enc_test;",
      {{"USAUnited States"}, {"CanadaCanada"}, {"UKUnited Kingdom"}, {"GermanyGermany"}});

  sqlAndCompareResult(
      "select text_encoding_dict_concat2(name, short_name) from "
      "text_enc_test;",
      {{"United StatesUSA"}, {"CanadaCanada"}, {"United KingdomUK"}, {"GermanyGermany"}});

  sqlAndCompareResult(
      "select text_encoding_dict_concat2('short name: ', short_name) from "
      "text_enc_test;",
      {{"short name: USA"},
       {"short name: Canada"},
       {"short name: UK"},
       {"short name: Germany"}});

  sqlAndCompareResult(
      "select text_encoding_dict_concat3(short_name, "
      "text_encoding_dict_copy(code)) from "
      "text_enc_test;",
      {{"USA copy: >>US USA<<"},
       {"Canada copy: >>CA CAN<<"},
       {"UK copy: >>GB GBR<<"},
       {"Germany copy: >>DE DEN<<"}});
}

TEST_P(StringFunctionTest, TextEncodingDictCopyUDF) {
  sqlAndCompareResult(
      "select text_encoding_dict_copy(short_name) from "
      "text_enc_test;",
      {{"copy: USA"}, {"copy: Canada"}, {"copy: UK"}, {"copy: Germany"}});
}

TEST_P(StringFunctionTest, TextEncodingDictCopyFromUDF) {
  sqlAndCompareResult(
      "select text_encoding_dict_copy_from(short_name, code, 1) from "
      "text_enc_test;",
      {{"copy: USA"}, {"copy: Canada"}, {"copy: UK"}, {"copy: Germany"}});

  sqlAndCompareResult(
      "select text_encoding_dict_copy_from(short_name, code, 2) from "
      "text_enc_test;",
      {{"copy: >>US USA<<"},
       {"copy: >>CA CAN<<"},
       {"copy: >>GB GBR<<"},
       {"copy: >>DE DEN<<"}});
}

TEST_P(StringFunctionTest, LLMTransformCurlError) {
  ScopeGuard reset = [orig = g_heavyiq_url]() { g_heavyiq_url = orig; };
  g_heavyiq_url = "http://localhost:99999";
  queryAndAssertPartialException(
      "SELECT LLM_TRANSFORM(short_name, \'Return the capital of the following "
      "state\') FROM text_enc_test",
      "LLM_TRANSFORM failed");
}

INSTANTIATE_TEST_SUITE_P(CpuAndGpuExecutorDevices,
                         StringFunctionTest,
                         ::testing::Values(TExecuteMode::CPU, TExecuteMode::GPU),
                         ::testing::PrintToStringParamName());

class UrlEncodeTest : public StringFunctionTest {};

TEST_P(UrlEncodeTest, WhitespaceAndExclamationMark) {
  sqlAndCompareResult("SELECT URL_ENCODE('Hello World!');", {{"Hello+World%21"}});
}

INSTANTIATE_TEST_SUITE_P(CpuAndGpuExecutorDevices,
                         UrlEncodeTest,
                         ::testing::Values(TExecuteMode::CPU, TExecuteMode::GPU),
                         ::testing::PrintToStringParamName());

class UrlDecodeTest : public StringFunctionTest {};

TEST_P(UrlDecodeTest, DigitAndAlphabet) {
  sqlAndCompareResult("SELECT URL_DECODE('%3100%41');", {{"100A"}});
}

// If % is one of the last two characters, it should not be decoded by URL_DECODE().
TEST_P(UrlDecodeTest, DigitAndPercentLastCharacter) {
  sqlAndCompareResult("SELECT URL_DECODE('%3100%');", {{"100%"}});
}

TEST_P(UrlDecodeTest, DigitAndPercentSecondToLastCharacter) {
  sqlAndCompareResult("SELECT URL_DECODE('%3100%!');", {{"100%!"}});
}

INSTANTIATE_TEST_SUITE_P(CpuAndGpuExecutorDevices,
                         UrlDecodeTest,
                         ::testing::Values(TExecuteMode::CPU, TExecuteMode::GPU),
                         ::testing::PrintToStringParamName());

const char* kCreatePostgresOsmNamesDdl = R"(
    CREATE TABLE postgres_osm_names (
      name TEXT,
      name_none_encoded TEXT ENCODING NONE,
      name_lower TEXT,
      name_upper TEXT,
      name_initcap TEXT,
      name_reverse TEXT,
      name_repeat_2 TEXT,
      name_concat_comma TEXT,
      comma_concat_name TEXT,
      name_lpad_20 TEXT,
      name_rpad_20 TEXT,
      name_trib_abab TEXT,
      name_ltrim_abab TEXT,
      name_rtrim_abab TEXT,
      name_substring_from_3_for_8 TEXT,
      name_overlay_foobar_from_4_for_3 TEXT,
      name_replace_il_eel TEXT,
      name_split_part_space_2 TEXT);
    )";
const char* kEncodedTextColumnType{"EncodedTextColumn"};
const char* kNoneEncodedTextColumnType{"NoneEncodedTextColumn"};

// Todo(todd): Add more Postgres tests. Part of the issue is just
// getting in the data correctly, for example, when we import anything
// with leading or trailing spaces (i.e. the 'name_substring_from_3_for_8
// column), our importer automatically drops the spaces, and there seems
// no easy way in Postgres to force quoting everything.
class PostgresStringFunctionTest
    : public DBHandlerTestFixture,
      public ::testing::WithParamInterface<std::tuple<TExecuteMode::type, std::string>> {
 public:
  static void SetUpTestSuite() {
    ASSERT_NO_THROW(sql("DROP TABLE IF EXISTS postgres_osm_names;"));
    ASSERT_NO_THROW(sql(kCreatePostgresOsmNamesDdl));
    ASSERT_NO_THROW(
        sql("COPY postgres_osm_names FROM "
            "'../../Tests/Import/datafiles/string_funcs/"
            "postgres_osm_names_ascii_1k.csv.gz' WITH (header='true');"));

    TQueryResult result;
    sql(result, "SELECT COUNT(*) FROM postgres_osm_names;");
    CHECK_EQ(result.row_set.columns.size(), size_t(1));
    CHECK_EQ(result.row_set.columns[0].data.int_col.size(), size_t(1));
    CHECK_EQ(result.row_set.columns[0].data.int_col[0], expected_row_count_);
  }

  static void TearDownTestSuite() {
    ASSERT_NO_THROW(sql("DROP TABLE postgres_osm_names;"));
  }

 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    auto device_type = std::get<0>(GetParam());
    if (!setExecuteMode(device_type)) {
      GTEST_SKIP() << device_type << " is not enabled.";
    }

    auto column_type = std::get<1>(GetParam());
    if (column_type == kEncodedTextColumnType) {
      text_column_name_ = "name";
    } else {
      CHECK_EQ(column_type, kNoneEncodedTextColumnType);
      text_column_name_ = "name_none_encoded";
    }
  }

  std::string text_column_name_;
  const static inline int64_t expected_row_count_{1000};
};

TEST_P(PostgresStringFunctionTest, Lower) {
  sqlAndCompareResult("SELECT COUNT(*) FROM postgres_osm_names WHERE LOWER(" +
                          text_column_name_ + ") = name_lower;",
                      {{expected_row_count_}});

  sqlAndCompareResult("SELECT COUNT(*) FROM postgres_osm_names WHERE LOWER(" +
                          text_column_name_ + ") <> name_lower;",
                      {{i(0)}});
}

TEST_P(PostgresStringFunctionTest, Upper) {
  sqlAndCompareResult("SELECT COUNT(*) FROM postgres_osm_names WHERE UPPER(" +
                          text_column_name_ + ") = name_upper;",
                      {{expected_row_count_}});

  sqlAndCompareResult("SELECT COUNT(*) FROM postgres_osm_names WHERE UPPER(" +
                          text_column_name_ + ") <> name_upper;",
                      {{i(0)}});
}

TEST_P(PostgresStringFunctionTest, InitCap) {
  // Postgres seems to have different rules for INITCAP than the ones we use
  // following Snowflake, such as capitalizing letters after apostrophes (i.e.
  // Smith'S), so exclude these differences via additional SQL filters
  sqlAndCompareResult(
      "SELECT (SELECT COUNT(*) FROM postgres_osm_names "
      " where not name ilike '%''%' and not name ilike '%’%' and not name ilike '%–%') "
      " - (SELECT COUNT(*) FROM postgres_osm_names "
      " WHERE INITCAP(" +
          text_column_name_ +
          ") = name_initcap AND not name ilike '%''%' "
          " and not name ilike '%’%' and not name ilike '%–%');",
      {{i(0)}});
}

INSTANTIATE_TEST_SUITE_P(
    CpuAndGpuExecutorDevices,
    PostgresStringFunctionTest,
    ::testing::Combine(::testing::Values(TExecuteMode::CPU, TExecuteMode::GPU),
                       ::testing::Values(kEncodedTextColumnType,
                                         kNoneEncodedTextColumnType)),
    [](const auto& info) {
      std::stringstream ss;
      ss << std::get<0>(info.param) << "_" << std::get<1>(info.param);
      return ss.str();
    });

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  g_enable_string_functions = true;
  g_enable_watchdog = true;
  g_watchdog_none_encoded_string_translation_limit = 1000000UL;

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_string_functions = false;
  return err;
}
