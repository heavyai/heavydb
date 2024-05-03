/*
 * Copyright 2024 HEAVY.AI, Inc.
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

#include "DBHandlerTestHelpers.h"
#include "TestHelpers.h"

#include "Logger/Logger.h"
#include "ThriftHandler/DBHandler.h"

using namespace TestHelpers;

class BaseTestFixture : public DBHandlerTestFixture {
 protected:
  void SetUp() override {}

  void TearDown() override {}

  const char* table_schema = "(x int, y int);";

  void buildTable(const std::string& table_name) {
    sql("DROP TABLE IF EXISTS " + table_name + ";");
    sql("CREATE TABLE " + table_name + " " + table_schema);
    ValuesGenerator gen(table_name);
    for (size_t i = 0; i < 10; i++) {
      sql(gen(i, i * 10));
    }
  }
};

class ValidateQueryTest : public BaseTestFixture {
 protected:
  void SetUp() override {
    BaseTestFixture::SetUp();
    buildTable("test");
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS test;");
    BaseTestFixture::TearDown();
  }
};

TEST_F(ValidateQueryTest, WidthBucketExprScalarSubquery) {
  setExecuteMode(TExecuteMode::CPU);
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TRowDescriptor validation_result;
  std::string q{
      "SELECT WIDTH_BUCKET(x,(SELECT MIN(x) FROM test), (SELECT MAX(x) FROM test), 10) "
      "AS x_bin, COUNT(*) AS n FROM test GROUP BY x_bin ORDER BY x_bin;"};
  EXPECT_NO_THROW(db_handler->sql_validate(validation_result, session_id, q));
  EXPECT_EQ(validation_result.size(), size_t(2));
  EXPECT_EQ(validation_result[0].col_type.type, TDatumType::INT);
  EXPECT_EQ(validation_result[1].col_type.type, TDatumType::INT);
}
TEST_F(ValidateQueryTest, WidthBucketExprHaveSameBound) {
  setExecuteMode(TExecuteMode::CPU);
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TRowDescriptor validation_result;
  std::string q{
      "SELECT WIDTH_BUCKET(x,0, 0, 10) AS x_bin, COUNT(*) AS n FROM test GROUP BY x_bin "
      "ORDER BY x_bin;"};
  EXPECT_ANY_THROW(db_handler->sql_validate(validation_result, session_id, q));
}

TEST_F(ValidateQueryTest, WidthBucketExprHaveNullValueAsBound) {
  setExecuteMode(TExecuteMode::CPU);
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TRowDescriptor validation_result;
  std::string q{
      "SELECT WIDTH_BUCKET(x,0, null, 10) AS x_bin, COUNT(*) AS n FROM test GROUP BY "
      "x_bin ORDER BY x_bin;"};
  EXPECT_ANY_THROW(db_handler->sql_validate(validation_result, session_id, q));
}

int main(int argc, char* argv[]) {
  g_is_test_env = true;
  ScopeGuard reset = [] { g_is_test_env = false; };
  TestHelpers::init_logger_stderr_only(argc, argv);
  namespace po = boost::program_options;
  po::options_description desc("Options");
  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(),
            vm);
  po::notify(vm);

  int err{0};
  try {
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
