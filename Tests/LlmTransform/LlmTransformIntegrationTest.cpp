/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file LlmTransformIntegrationTest.cpp
 * @brief Integration test suite for the LLM_TRANSFORM function
 *
 */

#include <gtest/gtest.h>
#include <thrift/protocol/TBinaryProtocol.h>

#include "Shared/SysDefinitions.h"
#include "Shared/ThriftClient.h"
#include "Shared/scope.h"
#include "Tests/TestHelpers.h"
#include "gen-cpp/Heavy.h"

std::string g_hostname{"localhost"};
int32_t g_tcp_port{6274};

std::string g_username{shared::kRootUsername};
std::string g_password{shared::kDefaultRootPasswd};
std::string g_database{shared::kDefaultDbName};

// Steps for running the integration tests:
// 1. Build the `heavydb` and `LlmTransformIntegrationTest` targets.
// 2. Navigate to the build directory.
// 3. Run the start_and_run_llm_transform_test.sh script.
class LlmTransformIntegrationTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    createNewSession(session_id_);
    for (const auto& table_name : {"test_table_1", "test_table_2"}) {
      executeQuery("CREATE TABLE " + std::string{table_name} +
                   " (id INTEGER, state_encoded TEXT ENCODING DICT(32), "
                   "state_none_encoded TEXT ENCODING NONE);");

      size_t i{0};
      for (const auto& state :
           {"Washington", "Oregon", "California", "Texas", "New York"}) {
        executeQuery("INSERT INTO " + std::string{table_name} + " VALUES (" +
                     std::to_string(i++) + ", '" + std::string{state} + "', '" +
                     std::string{state} + "');");
      }
    }
    // The test suite expects "llm-transform-max-num-unique-value" to be set to 5. Ensure
    // that test_table_2 exceeds that threshold.
    executeQuery(
        "INSERT INTO test_table_2 VALUES (5, 'Arizona', 'Arizona'), (6, 'Virginia', "
        "'Virginia');");
  }

  static void TearDownTestSuite() {
    for (const auto& table_name : {"test_table_1", "test_table_2"}) {
      executeQuery("DROP TABLE IF EXISTS " + std::string{table_name} + ";");
    }
    heavydb_client_.disconnect(session_id_);
  }

  static std::shared_ptr<TBinaryProtocol> getThriftProtocol() {
    auto client_connection = std::make_shared<ThriftClientConnection>();
    auto transport =
        client_connection->open_buffered_client_transport(g_hostname, g_tcp_port, {});
    transport->open();
    return std::make_shared<TBinaryProtocol>(transport);
  }

  static void createNewSession(TSessionId& session_id) {
    heavydb_client_.connect(session_id, g_username, g_password, g_database);
    ASSERT_FALSE(session_id.empty());
  }

  static void executeQuery(const std::string& query) {
    TQueryResult result;
    heavydb_client_.sql_execute(result, session_id_, query, true, "", -1, -1);
  }

  void executeQueryAndAssertResult(
      const std::string& query,
      const std::vector<std::vector<std::string>>& expected_result) {
    TQueryResult result;
    heavydb_client_.sql_execute(result, session_id_, query, true, "", -1, -1);
    ASSERT_TRUE(result.row_set.is_columnar);

    const auto& columns = result.row_set.columns;
    auto column_count = columns.size();
    ASSERT_GT(column_count, size_t(0));
    auto row_count = columns[0].data.str_col.size();

    std::vector<std::vector<std::string>> actual_result;
    for (size_t r = 0; r < row_count; r++) {
      actual_result.emplace_back();
      for (size_t c = 0; c < column_count; c++) {
        const auto& rows = columns[c].data.str_col;
        ASSERT_LT(r, rows.size());
        actual_result.back().emplace_back(rows[r]);
      }
    }
    EXPECT_EQ(expected_result, actual_result);
  }

  void executeQueryAndAssertException(const std::string& query,
                                      const std::string& expected_error) {
    try {
      executeQuery(query);
      FAIL() << "An exception should have been thrown for this test case.";
    } catch (const TDBException& e) {
      ASSERT_EQ(expected_error, e.error_msg);
    }
  }

  static inline HeavyClient heavydb_client_{getThriftProtocol()};
  static inline TSessionId session_id_;
};

TEST_F(LlmTransformIntegrationTest, EncodedTextColumn) {
  executeQueryAndAssertResult(
      "SELECT LLM_TRANSFORM(state_encoded, 'what is the capital of this US state') "
      "FROM test_table_1 ORDER BY id;",
      {{"Olympia"}, {"Salem"}, {"Sacramento"}, {"Austin"}, {"Albany"}});
}

TEST_F(LlmTransformIntegrationTest, EncodedTextColumnConstrainedChoices) {
  executeQueryAndAssertResult(
      "SELECT LLM_TRANSFORM(state_encoded, 'Return WEST if this state is "
      "west of the Mississippi river, otherwise EAST', 'WEST|EAST') "
      "FROM test_table_1 ORDER BY id;",
      {{"WEST"}, {"WEST"}, {"WEST"}, {"WEST"}, {"EAST"}});
}

TEST_F(LlmTransformIntegrationTest, EncodedTextColumnConstrainedRegex) {
  executeQueryAndAssertResult(
      "SELECT LLM_TRANSFORM(state_encoded, 'Is this state is west or "
      "east of the the Mississippi river?', '/[we].*/') "
      "FROM test_table_1 ORDER BY id;",
      {{"west"}, {"west"}, {"west"}, {"west"}, {"east"}});
}

TEST_F(LlmTransformIntegrationTest, ConstraintLiteralFormatError) {
  executeQueryAndAssertException(
      "SELECT LLM_TRANSFORM(state_encoded, 'When did this state enter the Union?', "
      "'1.*') FROM test_table_1;",
      "LLM_TRANSFORM constraint literal must either have at least two output choices, "
      "separated by a '|' character (i.e. 'west|east'), or must be bounded by '/' on "
      "each side (i.e. '/SELECT .*;/') to signify a regex.");
}

TEST_F(LlmTransformIntegrationTest, EncodedTextColumnStringDictionarySizeExceeded) {
  executeQueryAndAssertException(
      "SELECT LLM_TRANSFORM(state_encoded, 'what is the capital of this US state') "
      "FROM test_table_2;",
      "The number of entries of a string dictionary of the input argument of the "
      "LLM_TRANSFORM (=7) is larger than a threshold (=5)");
}

TEST_F(LlmTransformIntegrationTest, EncodedTextColumnFilteredStringDictionaryBelowLimit) {
  executeQueryAndAssertResult(
      "SELECT LLM_TRANSFORM(state_encoded, 'what is the capital of this US state') "
      "FROM test_table_2 WHERE id BETWEEN 3 AND 7 ORDER BY id;",
      {{"Austin"}, {"Albany"}, {"Phoenix"}, {"Richmond"}});
}

TEST_F(LlmTransformIntegrationTest,
       EncodedTextColumnMultipleLlmTransformCallsFilteredStringDictionaryBelowLimit) {
  executeQueryAndAssertResult(
      "SELECT LLM_TRANSFORM(state_encoded, 'what is the capital of this US state'), "
      "LLM_TRANSFORM(state_encoded, 'what is the most populated city in this US state') "
      "FROM test_table_2 WHERE id BETWEEN 1 AND 5 ORDER BY id;",
      {{"Salem", "Portland"},
       {"Sacramento", "Los Angeles"},
       {"Austin", "Houston"},
       {"Albany", "New York"},
       {"Phoenix", "Phoenix"}});
}

TEST_F(LlmTransformIntegrationTest, NoneEncodedTextColumn) {
  executeQueryAndAssertResult(
      "SELECT LLM_TRANSFORM(state_none_encoded, 'what is the capital of this US state') "
      "FROM test_table_1 ORDER BY id;",
      {{"Olympia"}, {"Salem"}, {"Sacramento"}, {"Austin"}, {"Albany"}});
}

TEST_F(LlmTransformIntegrationTest, NoneEncodedTextColumnNoLimit) {
  executeQueryAndAssertResult(
      "SELECT LLM_TRANSFORM(state_none_encoded, 'what is the capital of this US state') "
      "FROM test_table_2 ORDER BY id;",
      {{"Olympia"},
       {"Salem"},
       {"Sacramento"},
       {"Austin"},
       {"Albany"},
       {"Phoenix"},
       {"Richmond"}});
}

TEST_F(LlmTransformIntegrationTest, UpdateStringColumnByLLMTransformWithFilteredInput) {
  executeQuery("DROP TABLE IF EXISTS test_table_22;");
  ScopeGuard reset = []() { executeQuery("DROP TABLE IF EXISTS test_table_22;"); };
  executeQuery("CREATE TABLE test_table_22 AS (SELECT * FROM test_table_2);");
  executeQuery("ALTER TABLE test_table_22 ADD COLUMN capital TEXT;");
  executeQuery(
      "UPDATE test_table_22 SET capital = LLM_TRANSFORM(state_none_encoded, 'what is the "
      "capital of this US state') WHERE id > 3;");
}

TEST_F(LlmTransformIntegrationTest, PushdownFilterFromSubquery) {
  executeQueryAndAssertResult(
      "select state_encoded, LLM_TRANSFORM(state_encoded, 'Is in USA?') from "
      "test_table_1 where id in (select id from test_table_2 where state_none_encoded "
      "ilike '%C%');",
      {{"California", "Yes"}});
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()("hostname",
                     po::value<std::string>(&g_hostname)->default_value(g_hostname),
                     "HeavyDB server hostname");
  desc.add_options()("port",
                     po::value<int32_t>(&g_tcp_port)->default_value(g_tcp_port),
                     "HeavyDB server TCP port");
  desc.add_options()("username",
                     po::value<std::string>(&g_username)->default_value(g_username),
                     "Username to use for logging into the HeavyDB instance");
  desc.add_options()("password",
                     po::value<std::string>(&g_password)->default_value(g_password),
                     "Password to use for logging into the HeavyDB instance");
  desc.add_options()("database",
                     po::value<std::string>(&g_database)->default_value(g_database),
                     "Database to connect to when logging into the HeavyDB instance");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
