/*
 * Copyright 2020 OmniSci, Inc.
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

#include <gtest/gtest.h>

#include "DBHandlerTestHelpers.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class EpochConsistencyTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    createDBHandler();
    dropUser();
    sql("create user non_super_user (password = 'HyperInteractive');");
    sql("grant all on database omnisci to non_super_user;");
  }

  static void TearDownTestSuite() { dropUser(); }

  static void dropUser() {
    switchToAdmin();
    try {
      sql("drop user non_super_user;");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
  }

  void TearDown() override {
    sql("drop table if exists test_table;");
    DBHandlerTestFixture::TearDown();
  }
};

class ImportLoaderFailureTest : public EpochConsistencyTest,
                                public testing::WithParamInterface<std::string> {
 protected:
  static void SetUpTestSuite() { EpochConsistencyTest::SetUpTestSuite(); }

  static void TearDownTestSuite() { EpochConsistencyTest::TearDownTestSuite(); }

  std::string getGoodFilePath() {
    return "../../Tests/Import/datafiles/sharded_example_1.csv";
  }

  std::string getBadFilePath() {
    return "../../Tests/Import/datafiles/sharded_example_2.csv";
  }

  void assertTableEpochs(const std::vector<int32_t>& expected_table_epochs) {
    auto [db_handler, session_id] = getDbHandlerAndSessionId();
    const auto& catalog = getCatalog();
    auto table_id = catalog.getMetadataForTable("test_table", false)->tableId;
    std::vector<TTableEpochInfo> table_epochs;
    db_handler->get_table_epochs(
        table_epochs, session_id, catalog.getDatabaseId(), table_id);

    ASSERT_EQ(expected_table_epochs.size(), table_epochs.size());
    for (size_t i = 0; i < expected_table_epochs.size(); i++) {
      EXPECT_EQ(expected_table_epochs[i], table_epochs[i].table_epoch);
    }
  }
};

TEST_P(ImportLoaderFailureTest, ShardedTableWithInconsistentEpochs) {
  login(GetParam(), "HyperInteractive");

  sql("create table test_table(a int, b int, shard key(a)) with (shard_count = 2);");
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({1, 1});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1)},
                       {i(1), i(10)},
                       {i(2), i(2)},
                       {i(2), i(20)}});
  // clang-format on

  // The following insert query should result in the epochs getting out of sync
  sql("insert into test_table values (1, 1000);");
  assertTableEpochs({1, 2});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1)},
                       {i(1), i(10)},
                       {i(1), i(1000)},
                       {i(2), i(2)},
                       {i(2), i(20)}});
  // clang-format on

  // The following copy from query should result in an error and rollback
  sql("copy test_table from '" + getBadFilePath() + "' with (max_reject = 0);");
  assertTableEpochs({1, 2});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1)},
                       {i(1), i(10)},
                       {i(1), i(1000)},
                       {i(2), i(2)},
                       {i(2), i(20)}});
  // clang-format on

  // Ensure that a subsequent import still works as expected
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({2, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1)},
                       {i(1), i(1)},
                       {i(1), i(10)},
                       {i(1), i(10)},
                       {i(1), i(1000)},
                       {i(2), i(2)},
                       {i(2), i(2)},
                       {i(2), i(20)},
                       {i(2), i(20)}});
  // clang-format on
}

INSTANTIATE_TEST_SUITE_P(ImportLoaderFailureTest,
                         ImportLoaderFailureTest,
                         testing::Values("admin", "non_super_user"),
                         [](const auto& param_info) { return param_info.param; });

class SetTableEpochsTest : public EpochConsistencyTest {
 protected:
  static void SetUpTestSuite() { EpochConsistencyTest::SetUpTestSuite(); }

  static void TearDownTestSuite() { EpochConsistencyTest::TearDownTestSuite(); }

  std::pair<int32_t, std::vector<TTableEpochInfo>> getDbIdAndTableEpochs() {
    const auto& catalog = getCatalog();
    auto table_id = catalog.getMetadataForTable("test_table", false)->tableId;
    std::vector<TTableEpochInfo> table_epoch_info_vector;
    for (size_t i = 0; i < 2; i++) {
      TTableEpochInfo table_epoch_info;
      table_epoch_info.table_epoch = 1;
      table_epoch_info.table_id = table_id;
      table_epoch_info.leaf_index = i;
      table_epoch_info_vector.emplace_back(table_epoch_info);
    }
    return {catalog.getDatabaseId(), table_epoch_info_vector};
  }
};

TEST_F(SetTableEpochsTest, Admin) {
  loginAdmin();
  sql("create table test_table(a int);");

  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  auto [db_id, epoch_vector] = getDbIdAndTableEpochs();
  db_handler->set_table_epochs(session_id, db_id, epoch_vector);
}

TEST_F(SetTableEpochsTest, NonSuperUser) {
  login("non_super_user", "HyperInteractive");
  sql("create table test_table(a int);");

  executeLambdaAndAssertException(
      [this] {
        auto [db_handler, session_id] = getDbHandlerAndSessionId();
        auto [db_id, epoch_vector] = getDbIdAndTableEpochs();
        db_handler->set_table_epochs(session_id, db_id, epoch_vector);
      },
      "Only super users can set table epochs");
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  DBHandlerTestFixture::initTestArgs(argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
