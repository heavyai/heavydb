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

/**
 * Class that provides an interface for checkpoint failure mocking
 */
class CheckpointFailureMock {
 public:
  CheckpointFailureMock() : throw_on_checkpoint_(false) {}
  virtual ~CheckpointFailureMock() = default;

  void throwOnCheckpoint(bool throw_on_checkpoint) {
    throw_on_checkpoint_ = throw_on_checkpoint;
  }

 protected:
  bool throw_on_checkpoint_;
};

/**
 * Mock file mgr class mainly used for testing checkpoint failure.
 * However, in addition to the `checkpoint` method, calls to
 * `putBuffer` and `fetchBuffer` are proxied to the original/parent
 * file mgr, since these methods are called as part of checkpoint
 * calls that originate from CPU buffers.
 */
class MockFileMgr : public CheckpointFailureMock, public File_Namespace::FileMgr {
 public:
  MockFileMgr(std::shared_ptr<File_Namespace::FileMgr> file_mgr)
      : File_Namespace::FileMgr(file_mgr->epoch()), parent_file_mgr_(file_mgr) {
    CHECK(parent_file_mgr_);
  }

  File_Namespace::FileBuffer* putBuffer(const ChunkKey& chunk_key,
                                        AbstractBuffer* buffer,
                                        const size_t num_bytes) override {
    return parent_file_mgr_->putBuffer(chunk_key, buffer, num_bytes);
  }

  void fetchBuffer(const ChunkKey& chunk_key,
                   AbstractBuffer* dest_buffer,
                   const size_t num_bytes) override {
    parent_file_mgr_->fetchBuffer(chunk_key, dest_buffer, num_bytes);
  }

  void checkpoint() override {
    if (throw_on_checkpoint_) {
      throw std::runtime_error{"Mock checkpoint exception"};
    } else {
      parent_file_mgr_->checkpoint();
    }
  }

 private:
  std::shared_ptr<File_Namespace::FileMgr> parent_file_mgr_;
};

class EpochRollbackTest
    : public EpochConsistencyTest,
      public testing::WithParamInterface<std::tuple<std::string, bool>> {
 public:
  static std::string testParamsToString(
      const testing::TestParamInfo<std::tuple<std::string, bool>>& param_info) {
    auto& [user, is_checkpoint_error] = param_info.param;
    return user + (is_checkpoint_error ? "_checkpoint_error" : "_query_error");
  }

 protected:
  static void SetUpTestSuite() { EpochConsistencyTest::SetUpTestSuite(); }

  static void TearDownTestSuite() { EpochConsistencyTest::TearDownTestSuite(); }

  std::string getGoodFilePath() {
    return "../../Tests/Import/datafiles/sharded_example_1.csv";
  }

  std::string getBadFilePath() {
    return "../../Tests/Import/datafiles/sharded_example_2.csv";
  }

  std::string getUser() { return std::get<0>(GetParam()); }

  bool isCheckpointError() { return std::get<1>(GetParam()); }

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

  void setUpTestTableWithInconsistentEpochs() {
    login(getUser(), "HyperInteractive");

    sql("create table test_table(a int, b tinyint, c text encoding none, shard key(a)) "
        "with (shard_count = 2);");
    sql("copy test_table from '" + getGoodFilePath() + "';");
    assertTableEpochs({1, 1});

    // clang-format off
    sqlAndCompareResult("select * from test_table order by a, b;",
                        {{i(1), i(1), "test_1"},
                         {i(1), i(10), "test_10"},
                         {i(2), i(2), "test_2"},
                         {i(2), i(20), "test_20"}});
    // clang-format on

    // The following insert query should result in the epochs getting out of sync
    sql("insert into test_table values (1, 100, 'test_100');");
    assertTableEpochs({1, 2});
    assertInitialInsertResultSet();
  }

  // Asserts that the result set is equal to the expected value after the initial insert
  void assertInitialInsertResultSet() {
    // clang-format off
    sqlAndCompareResult("select * from test_table order by a, b;",
                        {{i(1), i(1), "test_1"},
                         {i(1), i(10), "test_10"},
                         {i(1), i(100), "test_100"},
                         {i(2), i(2), "test_2"},
                         {i(2), i(20), "test_20"}});
    // clang-format on
  }

  void assertInitialTableState() {
    assertTableEpochs({1, 2});
    assertInitialInsertResultSet();
  }

  void sendFailedImportQuery() {
    if (isCheckpointError()) {
      queryAndAssertCheckpointError("copy test_table from '" + getGoodFilePath() + "';");
    } else {
      sql("copy test_table from '" + getBadFilePath() + "' with (max_reject = 0);");
    }
  }

  void sendFailedInsertQuery() {
    if (isCheckpointError()) {
      queryAndAssertCheckpointError(
          "insert into test_table values (1, 110, 'test_110');");
    } else {
      EXPECT_ANY_THROW(sql("insert into test_table values (1, 10000, 'test_10000');"));
    }
  }

  void sendFailedUpdateQuery() {
    if (isCheckpointError()) {
      queryAndAssertCheckpointError(
          "update test_table set b = b + 1 where b = 100 or b = 20;");
    } else {
      EXPECT_ANY_THROW(
          sql("update test_table set b = case when b = 100 then 10000 else b + 1 end;"));
    }
  }

  void sendFailedVarlenUpdateQuery() {
    if (isCheckpointError()) {
      queryAndAssertCheckpointError(
          "update test_table set b = 110, c = 'test_110' where b = 100;");
    } else {
      EXPECT_ANY_THROW(sql(
          "update test_table set b = case when b = 100 then 10000 else b + 1 end, c = "
          "'test';"));
    }
  }

  void sendFailedDeleteQuery() {
    if (isCheckpointError()) {
      queryAndAssertCheckpointError("delete from test_table where b = 100 or b = 20;");
    } else {
      EXPECT_ANY_THROW(sql("delete from test_table where b = 2 or b = 10/0;"));
    }
  }

  void sendFailedItasQuery() {
    if (isCheckpointError()) {
      queryAndAssertCheckpointError(
          "insert into test_table (select * from test_table where b = 1 or b = 20);");
    } else {
      EXPECT_ANY_THROW(
          sql("insert into test_table (select a, case when b = 100 then 10000 else b + 1 "
              "end, c from test_table);"));
    }
  }

  void queryAndAssertCheckpointError(const std::string& query) {
    initializeCheckpointFailureMock();
    queryAndAssertException(query, "Exception: Mock checkpoint exception");
    resetCheckpointFailureMock();
  }

  void initializeCheckpointFailureMock() {
    const auto& catalog = getCatalog();
    auto global_file_mgr = catalog.getDataMgr().getGlobalFileMgr();
    auto physical_tables = getPhysicalTestTables();
    auto file_mgr = getFileMgr();
    CHECK(file_mgr);

    auto mock_file_mgr = std::make_shared<MockFileMgr>(file_mgr);
    global_file_mgr->setFileMgr(
        catalog.getDatabaseId(), physical_tables.back()->tableId, mock_file_mgr);
    checkpoint_failure_mock_ = mock_file_mgr.get();
    ASSERT_EQ(checkpoint_failure_mock_, dynamic_cast<MockFileMgr*>(getFileMgr().get()));
    checkpoint_failure_mock_->throwOnCheckpoint(true);
  }

  void resetCheckpointFailureMock() {
    if (isDistributedMode()) {
      checkpoint_failure_mock_->throwOnCheckpoint(false);
    } else {
      // GlobalFileMgr resets the file mgr for the table on rollback
      // and so the assertion here is to ensure that the mock file
      // mgr is no longer used.
      ASSERT_EQ(dynamic_cast<MockFileMgr*>(getFileMgr().get()), nullptr);
      checkpoint_failure_mock_ = nullptr;
    }
  }

  std::vector<const TableDescriptor*> getPhysicalTestTables() {
    const auto& catalog = getCatalog();
    auto logical_table = catalog.getMetadataForTable("test_table", false);
    CHECK_GT(logical_table->nShards, 0);

    auto physical_tables = catalog.getPhysicalTablesDescriptors(logical_table);
    CHECK_EQ(physical_tables.size(), static_cast<size_t>(logical_table->nShards));
    return physical_tables;
  }

  std::shared_ptr<File_Namespace::FileMgr> getFileMgr() {
    const auto& catalog = getCatalog();
    auto global_file_mgr = catalog.getDataMgr().getGlobalFileMgr();
    auto physical_tables = getPhysicalTestTables();
    return global_file_mgr->getSharedFileMgr(catalog.getDatabaseId(),
                                             physical_tables.back()->tableId);
  }

  CheckpointFailureMock* checkpoint_failure_mock_;
};

TEST_P(EpochRollbackTest, Import) {
  setUpTestTableWithInconsistentEpochs();

  sendFailedImportQuery();
  assertInitialTableState();

  // Ensure that a subsequent import still works as expected
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({2, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(1), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(1), i(10), "test_10"},
                       {i(1), i(100), "test_100"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

TEST_P(EpochRollbackTest, Insert) {
  // In distributed mode, for insert queries, checkpoint
  // happens as part of the `sql_execute` Thrift request
  // to leaf nodes, and so, a checkpoint error is handled
  // in the same way as a query error (which also throws
  // a Thrift exception from that API). Skipping since
  // the same test path is covered by the query error
  // test case/param.
  if (isDistributedMode() && isCheckpointError()) {
    GTEST_SKIP();
  }
  setUpTestTableWithInconsistentEpochs();

  sendFailedInsertQuery();
  assertInitialTableState();

  // Ensure that a subsequent insert query still works as expected
  sql("insert into test_table values (1, 110, 'test_110');");
  assertTableEpochs({1, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(1), i(100), "test_100"},
                       {i(1), i(110), "test_110"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

TEST_P(EpochRollbackTest, Update) {
  // In distributed mode, for update queries, checkpoint
  // happens as part of the `execute_query_step` Thrift
  // request to leaf nodes, and so, a checkpoint error
  // is handled in the same way as a query error (which
  // also throws a Thrift exception from that API).
  // Skipping since the same test path is covered by
  // the query error test case/param.
  if (isDistributedMode() && isCheckpointError()) {
    GTEST_SKIP();
  }
  setUpTestTableWithInconsistentEpochs();

  sendFailedUpdateQuery();
  assertInitialTableState();

  // Ensure that a subsequent update query still works as expected
  sql("update test_table set b = b + 1 where b = 100 or b = 20;");
  assertTableEpochs({2, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(1), i(101), "test_100"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(21), "test_20"}});
  // clang-format on
}

// Updates execute different code paths when variable length columns are updated
TEST_P(EpochRollbackTest, VarlenUpdate) {
  // In distributed mode, for update queries involving
  // variable length column types, checkpoint happens
  // as part of the `execute_query_step` Thrift request
  // to leaf nodes, and so, a checkpoint error is
  // handled in the same way as a query error (which
  // also throws a Thrift exception from that API).
  // Skipping since the same test path is covered by
  // the query error test case/param.
  if (isDistributedMode() && isCheckpointError()) {
    GTEST_SKIP();
  }
  setUpTestTableWithInconsistentEpochs();

  sendFailedVarlenUpdateQuery();
  assertInitialTableState();

  // Ensure that a subsequent update query still works as expected
  sql("update test_table set b = 110, c = 'test_110' where b = 100;");
  assertTableEpochs({2, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(1), i(110), "test_110"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

TEST_P(EpochRollbackTest, Delete) {
  // In distributed mode, for delete queries, checkpoint
  // happens as part of the `execute_query_step` Thrift
  // request to leaf nodes, and so, a checkpoint error
  // is handled in the same way as a query error (which
  // also throws a Thrift exception from that API).
  // Skipping since the same test path is covered by
  // the query error test case/param.
  if (isDistributedMode() && isCheckpointError()) {
    GTEST_SKIP();
  }
  setUpTestTableWithInconsistentEpochs();

  sendFailedDeleteQuery();
  assertInitialTableState();

  // Ensure that a delete query still works as expected
  sql("delete from test_table where b = 100 or b = 20;");
  assertTableEpochs({2, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(2), i(2), "test_2"}});
  // clang-format on
}

TEST_P(EpochRollbackTest, InsertTableAsSelect) {
  setUpTestTableWithInconsistentEpochs();

  sendFailedItasQuery();
  assertInitialTableState();

  // Ensure that a subsequent ITAS query still works as expected
  sql("insert into test_table (select * from test_table where b = 1 or b = 20);");
  assertTableEpochs({2, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(1), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(1), i(100), "test_100"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

INSTANTIATE_TEST_SUITE_P(EpochRollbackTest,
                         EpochRollbackTest,
                         testing::Combine(testing::Values("admin", "non_super_user"),
                                          testing::Values(true, false)),
                         EpochRollbackTest::testParamsToString);

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
