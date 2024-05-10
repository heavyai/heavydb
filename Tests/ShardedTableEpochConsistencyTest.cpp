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

#include <gtest/gtest.h>

#include "DBHandlerTestHelpers.h"
#include "Shared/SysDefinitions.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

class EpochConsistencyTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    createDBHandler();
    dropUser();
    sql("create user non_super_user (password = 'HyperInteractive');");
    sql("grant all on database " + shared::kDefaultDbName + " to non_super_user;");
  }

  static void TearDownTestSuite() { dropUser(); }

  static void dropUser() {
    switchToAdmin();
    sql("drop user IF EXISTS non_super_user;");
  }

  static void loginTestUser() { login("non_super_user", "HyperInteractive"); }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
  }

  void TearDown() override {
    sql("drop table if exists test_table;");
    DBHandlerTestFixture::TearDown();
  }

  std::string getGoodFilePath() {
    return "../../Tests/Import/datafiles/sharded_example_1.csv";
  }

  std::string getBadFilePath() {
    return "../../Tests/Import/datafiles/sharded_example_2.csv";
  }

  std::string getShardedMultiFragmentFilePath() {
    return "../../Tests/Import/datafiles/sharded_2_col_1_20.csv";
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

  void setUpTestTableWithInconsistentEpochs(const std::string& db_name = {}) {
    sql("create table test_table(a int, b tinyint, c text encoding none, shard key(a)) "
        "with (shard_count = 2, max_rollback_epochs = 25);");
    sql("copy test_table from '" + getGoodFilePath() + "';");
    assertTableEpochs({1, 1});
    assertInitialImportResultSet();

    // Inconsistent epochs have to be set manually, since all write queries now level
    // epochs
    setTableEpochs({1, 2}, db_name);
    assertTableEpochs({1, 2});
    assertInitialImportResultSet();
  }

  // Asserts that the result set is equal to the expected value after the initial import
  void assertInitialImportResultSet() {
    // clang-format off
    sqlAndCompareResult("select * from test_table order by a, b;",
                        {{i(1), i(1), "test_1"},
                         {i(1), i(10), "test_10"},
                         {i(2), i(2), "test_2"},
                         {i(2), i(20), "test_20"}});
    // clang-format on
  }

  // Sets table epochs across shards and leaves as indicated by the flattened epoch
  // vector. For instance, in order to set the epochs for a table with 3 shards on
  // a distributed setup with 2 leaves, the epochs vector will contain epochs for
  // corresponding shards/leaves in the form: { shard_1_leaf_1, shard_2_leaf_1,
  // shard_3_leaf_1, shard_1_leaf_2, shard_2_leaf_2, shard_3_leaf_2 }
  void setTableEpochs(const std::vector<int32_t>& table_epochs,
                      const std::string& db_name = {}) {
    auto [db_handler, session_id] = getDbHandlerAndSessionId();
    const auto& catalog = getCatalog();
    auto logical_table = catalog.getMetadataForTable("test_table", false);
    auto physical_tables = catalog.getPhysicalTablesDescriptors(logical_table);
    std::vector<TTableEpochInfo> table_epoch_info_vector;
    for (size_t i = 0; i < table_epochs.size(); i++) {
      TTableEpochInfo table_epoch_info;
      auto table_index = i % physical_tables.size();
      table_epoch_info.table_id = physical_tables[table_index]->tableId;
      table_epoch_info.table_epoch = table_epochs[i];
      table_epoch_info.leaf_index = i / physical_tables.size();
      table_epoch_info_vector.emplace_back(table_epoch_info);
    }
    if (db_name.empty()) {
      switchToAdmin();
    } else {
      login("admin", "HyperInteractive", db_name);
    }
    db_handler->set_table_epochs(
        session_id, catalog.getDatabaseId(), table_epoch_info_vector);
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
      : File_Namespace::FileMgr(file_mgr->lastCheckpointedEpoch())
      , parent_file_mgr_(file_mgr) {
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

class EpochRollbackTest : public EpochConsistencyTest,
                          public testing::WithParamInterface<bool> {
 public:
  static std::string testParamsToString(const testing::TestParamInfo<bool>& param_info) {
    auto is_checkpoint_error = param_info.param;
    return is_checkpoint_error ? "checkpoint_error" : "query_error";
  }

 protected:
  static void SetUpTestSuite() { EpochConsistencyTest::SetUpTestSuite(); }

  static void TearDownTestSuite() { EpochConsistencyTest::TearDownTestSuite(); }

  void SetUp() override {
    EpochConsistencyTest::SetUp();
    loginTestUser();
  }

  bool isCheckpointError() { return GetParam(); }

  void setUpReplicatedTestTableWithInconsistentEpochs() {
    sql("create table test_table(a int, b tinyint, c text encoding none) "
        "with (partitions = 'REPLICATED');");
    sql("copy test_table from '" + getGoodFilePath() + "';");

    assertTableEpochs({1});
    assertInitialImportResultSet();

    // Inconsistent epochs have to be set manually, since all write queries now level
    // epochs
    setTableEpochs({2});
    assertTableEpochs({2});
    assertInitialImportResultSet();
  }

  void assertInitialTableState() {
    assertTableEpochs({1, 2});
    assertInitialImportResultSet();
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

  void sendReplicatedTableFailedInsertQuery() { sendFailedInsertQuery(); }

  void sendFailedUpdateQuery() {
    if (isCheckpointError()) {
      queryAndAssertCheckpointError(
          "update test_table set b = b + 1 where b = 100 or b = 20;");
    } else {
      EXPECT_ANY_THROW(
          sql("update test_table set b = case when b = 10 then 10000 else b + 1 end;"));
    }
  }

  void sendFailedVarlenUpdateQuery() {
    if (isCheckpointError()) {
      queryAndAssertCheckpointError(
          "update test_table set b = 110, c = 'test_110' where b = 10;");
    } else {
      EXPECT_ANY_THROW(
          sql("update test_table set b = case when b = 10 then 10000 else b + 1 end, c = "
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
          sql("insert into test_table (select a, case when b = 10 then 10000 else b + 1 "
              "end, c from test_table);"));
    }
  }

  void queryAndAssertCheckpointError(const std::string& query) {
    initializeCheckpointFailureMock();
    queryAndAssertException(query, "Mock checkpoint exception");
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
    // GlobalFileMgr resets the file mgr for the table on rollback
    // and so the assertion here is to ensure that the mock file
    // mgr is no longer used.
    ASSERT_EQ(dynamic_cast<MockFileMgr*>(getFileMgr().get()), nullptr);
    checkpoint_failure_mock_ = nullptr;
  }

  std::vector<const TableDescriptor*> getPhysicalTestTables() {
    const auto& catalog = getCatalog();
    auto logical_table = catalog.getMetadataForTable("test_table", false);
    if (logical_table->partitions != "REPLICATED") {
      CHECK_GT(logical_table->nShards, 0);
    }

    auto physical_tables = catalog.getPhysicalTablesDescriptors(logical_table);
    if (logical_table->partitions != "REPLICATED") {
      CHECK_EQ(physical_tables.size(), static_cast<size_t>(logical_table->nShards));
    }
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
  CheckpointFailureMock* checkpoint_failure_no_to_mock_;
};

TEST_P(EpochRollbackTest, Import) {
  setUpTestTableWithInconsistentEpochs();
  loginTestUser();

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
                       {i(2), i(2), "test_2"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

TEST_P(EpochRollbackTest, Insert) {
  setUpTestTableWithInconsistentEpochs();
  loginTestUser();

  sendFailedInsertQuery();
  assertInitialTableState();

  // Ensure that a subsequent insert query still works as expected
  sql("insert into test_table values (1, 110, 'test_110');");
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

TEST_P(EpochRollbackTest, InsertOnReplicatedTable) {
  setUpReplicatedTestTableWithInconsistentEpochs();
  loginTestUser();

  sendReplicatedTableFailedInsertQuery();
  assertTableEpochs({2});
  assertInitialImportResultSet();

  // Ensure that a subsequent insert query still works as expected
  sql("insert into test_table values (1, 110, 'test_110');");
  assertTableEpochs({3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(1), i(110), "test_110"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

TEST_P(EpochRollbackTest, Update) {
  // The checkpoint error case exercises the same path as the query error case in
  // distributed mode. Specifically, both come back as exceptions when
  // `execute_query_step` is called on leaf nodes.
  if (isDistributedMode() && isCheckpointError()) {
    GTEST_SKIP();
  }

  setUpTestTableWithInconsistentEpochs();
  loginTestUser();

  sendFailedUpdateQuery();
  assertInitialTableState();

  // Ensure that a subsequent update query still works as expected
  sql("update test_table set b = b + 1 where b = 10 or b = 20;");
  // 1 checkpoint for update and 1 checkpoint for automatic vacuum
  assertTableEpochs({3, 4});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(11), "test_10"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(21), "test_20"}});
  // clang-format on
}

// Updates execute different code paths when variable length columns are updated
TEST_P(EpochRollbackTest, VarlenUpdate) {
  // The checkpoint error case exercises the same path as the query error case in
  // distributed mode. Specifically, both come back as exceptions when
  // `execute_query_step` is called on leaf nodes.
  if (isDistributedMode() && isCheckpointError()) {
    GTEST_SKIP();
  }

  setUpTestTableWithInconsistentEpochs();
  loginTestUser();

  sendFailedVarlenUpdateQuery();
  assertInitialTableState();

  // Ensure that a subsequent update query still works as expected
  sql("update test_table set b = 110, c = 'test_110' where b = 10;");
  // 1 checkpoint for update and 1 checkpoint for automatic vacuum
  assertTableEpochs({3, 4});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(110), "test_110"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

TEST_P(EpochRollbackTest, Delete) {
  // The checkpoint error case exercises the same path as the query error case in
  // distributed mode. Specifically, both come back as exceptions when
  // `execute_query_step` is called on leaf nodes.
  if (isDistributedMode() && isCheckpointError()) {
    GTEST_SKIP();
  }

  setUpTestTableWithInconsistentEpochs();
  loginTestUser();

  sendFailedDeleteQuery();
  assertInitialTableState();

  // Ensure that a delete query still works as expected
  sql("delete from test_table where b = 10 or b = 20;");
  // 1 checkpoint for update and 1 checkpoint for automatic vacuum
  assertTableEpochs({3, 4});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(2), i(2), "test_2"}});
  // clang-format on
}

TEST_P(EpochRollbackTest, InsertTableAsSelect) {
  setUpTestTableWithInconsistentEpochs();
  loginTestUser();

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
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

INSTANTIATE_TEST_SUITE_P(EpochRollbackTest,
                         EpochRollbackTest,
                         testing::Values(true, false),
                         EpochRollbackTest::testParamsToString);

class EpochLevelingTest : public EpochConsistencyTest {
  void SetUp() override {
    EpochConsistencyTest::SetUp();
    sql("create table test_table(a int, b tinyint, c text encoding none, shard key(a)) "
        "with (shard_count = 4);");
    assertTableEpochs({0, 0, 0, 0});
  }
};

TEST_F(EpochLevelingTest, Import) {
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({1, 1, 1, 1});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

TEST_F(EpochLevelingTest, Insert) {
  sql("insert into test_table values (1, 100, 'test_100');");
  assertTableEpochs({1, 1, 1, 1});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(100), "test_100"}});
  // clang-format on
}

TEST_F(EpochLevelingTest, Update) {
  // Import data to be updated by the following query
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({1, 1, 1, 1});

  sql("update test_table set b = b + 1 where b = 1;");
  // 1 checkpoint for update and 1 checkpoint for automatic vacuum
  assertTableEpochs({3, 3, 3, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(2), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

// Updates execute different code paths when variable length columns are updated
TEST_F(EpochLevelingTest, VarlenUpdate) {
  // Import data to be updated by the following query
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({1, 1, 1, 1});

  sql("update test_table set b = 110, c = 'test_110' where b = 10;");
  // 1 checkpoint for update and 1 checkpoint for automatic vacuum
  assertTableEpochs({3, 3, 3, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(110), "test_110"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

TEST_F(EpochLevelingTest, UpdateQueryButDataNotChanged) {
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({1, 1, 1, 1});

  sql("update test_table set b = b + 1 where b = 1000;");
  // 1 checkpoint for update and 1 checkpoint for automatic vacuum
  assertTableEpochs({3, 3, 3, 3});
  assertInitialImportResultSet();
}

TEST_F(EpochLevelingTest, Delete) {
  // Import data to be deleted by the following query
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({1, 1, 1, 1});

  sql("delete from test_table where b = 10 or b = 20;");
  // 1 checkpoint for delete and 1 checkpoint for automatic vacuum
  assertTableEpochs({3, 3, 3, 3});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(2), i(2), "test_2"}});
  // clang-format on
}

TEST_F(EpochLevelingTest, DeleteQueryButNoDataDeleted) {
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({1, 1, 1, 1});

  sql("delete from test_table where b = 1000;");
  // 1 checkpoint for update and 1 checkpoint for automatic vacuum
  assertTableEpochs({3, 3, 3, 3});
  assertInitialImportResultSet();
}

TEST_F(EpochLevelingTest, InsertTableAsSelect) {
  // Import data to be selected by the following query
  sql("copy test_table from '" + getGoodFilePath() + "';");
  assertTableEpochs({1, 1, 1, 1});

  sql("insert into test_table (select * from test_table where b = 1);");
  assertTableEpochs({2, 2, 2, 2});

  // clang-format off
  sqlAndCompareResult("select * from test_table order by a, b;",
                      {{i(1), i(1), "test_1"},
                       {i(1), i(1), "test_1"},
                       {i(1), i(10), "test_10"},
                       {i(2), i(2), "test_2"},
                       {i(2), i(20), "test_20"}});
  // clang-format on
}

TEST_F(EpochLevelingTest, Optimize) {
  sql("drop table if exists test_table;");
  sql("create table test_table(a int, b int, shard key(a)) "
      "with (shard_count = 4, fragment_size = 2);");
  assertTableEpochs({0, 0, 0, 0});

  sql("copy test_table from '" + getShardedMultiFragmentFilePath() + "';");
  assertTableEpochs({1, 1, 1, 1});

  sql("delete from test_table where mod(b, 2) = 0;");
  // 1 checkpoint for update and 1 checkpoint for automatic vacuum
  assertTableEpochs({3, 3, 3, 3});

  sql("optimize table test_table with (vacuum = 'true');");
  // Vacuum does a checkpoint and metadata re-computation does another checkpoint
  assertTableEpochs({5, 5, 5, 5});

  // clang-format off
  // Assert subsequent query returns expected result
  sqlAndCompareResult("select * from test_table order by a;",
                      {{i(1), i(1)}, {i(3), i(3)}, {i(5), i(5)}, {i(7), i(7)},
                       {i(9), i(9)}, {i(11), i(11)}, {i(13), i(13)},
                       {i(15), i(15)}, {i(17), i(17)}, {i(19), i(19)}});
  // clang-format on
}

class SetTableEpochsTest : public EpochConsistencyTest {
 protected:
  static void SetUpTestSuite() { EpochConsistencyTest::SetUpTestSuite(); }

  static void TearDownTestSuite() { EpochConsistencyTest::TearDownTestSuite(); }

  void SetUp() override {
    EpochConsistencyTest::SetUp();
    sql("drop table if exists test_table;");
    sql("drop table if exists test_table_2;");
  }

  void TearDown() override {
    sql("drop table if exists test_table;");
    sql("drop table if exists test_table_2;");
    EpochConsistencyTest::TearDown();
  }

  std::pair<int32_t, std::vector<TTableEpochInfo>> getDbIdAndTableEpochs() {
    const auto& catalog = getCatalog();
    auto td = catalog.getMetadataForTable("test_table", false);
    auto tables = catalog.getPhysicalTablesDescriptors(td, false);
    std::vector<TTableEpochInfo> table_epoch_info_vector;
    for (auto table : tables) {
      for (size_t i = 0; i < 2; i++) {
        TTableEpochInfo table_epoch_info;
        table_epoch_info.table_epoch = 1;
        table_epoch_info.table_id = table->tableId;
        table_epoch_info.leaf_index = i;
        table_epoch_info_vector.emplace_back(table_epoch_info);
      }
    }
    return {catalog.getDatabaseId(), table_epoch_info_vector};
  }

  // assert all epochs are equal to current_epoch, then set to new_epoch
  void assertAndSetTableEpoch(int32_t current_epoch, int32_t new_epoch) {
    std::vector<TTableEpochInfo> epochs_vector;
    const auto& catalog = getCatalog();
    auto [db_handler, session_id] = getDbHandlerAndSessionId();
    auto [db_id, epoch_vector] = getDbIdAndTableEpochs();
    auto table_id = catalog.getMetadataForTable("test_table", false)->tableId;
    db_handler->get_table_epochs(epochs_vector, session_id, db_id, table_id);

    for (auto& tei : epochs_vector) {
      CHECK_EQ(tei.table_epoch, current_epoch);
      tei.table_epoch = new_epoch;
    }
    db_handler->set_table_epochs(session_id, db_id, epochs_vector);
  }

  std::string getTableIdString() {
    const auto& catalog = getCatalog();
    auto table_id = catalog.getMetadataForTable("test_table", false)->tableId;
    std::stringstream id_ss;
    // +1 as message will be from first shard
    id_ss << "(" << catalog.getDatabaseId() << ", " << table_id + 1 << ")";
    return id_ss.str();
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

TEST_F(SetTableEpochsTest, ShardedTable) {
  loginAdmin();
  sql("create table test_table (a int, shard key(a)) with (shard_count = 4);");

  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  auto [db_id, epoch_vector] = getDbIdAndTableEpochs();
  db_handler->set_table_epochs(session_id, db_id, epoch_vector);
}

TEST_F(SetTableEpochsTest, EpochWithDifferentLogicalTable) {
  loginAdmin();
  sql("create table test_table (a int, shard key(a)) with (shard_count = 4);");
  sql("create table test_table_2 (a int);");

  executeLambdaAndAssertException(
      [this] {
        auto [db_handler, session_id] = getDbHandlerAndSessionId();
        auto [db_id, epoch_vector] = getDbIdAndTableEpochs();
        ASSERT_GT(epoch_vector.size(), static_cast<size_t>(1));
        epoch_vector[0].table_id =
            getCatalog().getMetadataForTable("test_table_2")->tableId;
        db_handler->set_table_epochs(session_id, db_id, epoch_vector);
      },
      "Table epochs do not reference the same logical table");
}

TEST_F(SetTableEpochsTest, Capped) {
  loginAdmin();

  const int row_count = 20;
  sql("create table test_table(a int, shard key(a)) with (max_rollback_epochs = 5, "
      "shard_count =2);");

  for (int i = 0; i < row_count; i++) {
    sql("insert into test_table values (" + std::to_string(i) + ");");
  }
  assertTableEpochs({row_count, row_count});
  sqlAndCompareResult("select count(*) from test_table;", {{i(row_count)}});

  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  auto [db_id, not_good_to_use] = getDbIdAndTableEpochs();

  std::vector<TTableEpochInfo> epochs_vector;
  const auto& catalog = getCatalog();
  auto table_id = catalog.getMetadataForTable("test_table", false)->tableId;
  db_handler->get_table_epochs(epochs_vector, session_id, db_id, table_id);

  for (auto& tei : epochs_vector) {
    CHECK_EQ(tei.table_epoch, row_count);
    tei.table_epoch = 10;
  }

  // This should fail as we are attempting to rollback too far
  EXPECT_ANY_THROW(db_handler->set_table_epochs(session_id, db_id, epochs_vector));

  for (auto& tei : epochs_vector) {
    tei.table_epoch = 15;
  }
  db_handler->set_table_epochs(session_id, db_id, epochs_vector);

  sqlAndCompareResult("select count(*) from test_table;", {{i(15)}});
  assertTableEpochs({15, 15});
}

TEST_F(SetTableEpochsTest, CappedAlter) {
  loginAdmin();
  int row_count = 20;
  auto doInserts = [&row_count]() {
    for (int i = 0; i < row_count; i++) {
      sql("insert into test_table values (" + std::to_string(i) + ");");
    }
  };

  sql("create table test_table(a int, shard key(a)) with (shard_count = 2, "
      "max_rollback_epochs = 25);");

  doInserts();
  assertTableEpochs({row_count, row_count});
  sqlAndCompareResult("select count(*) from test_table;", {{i(row_count)}});

  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  auto [db_id, not_good_to_use] = getDbIdAndTableEpochs();

  std::vector<TTableEpochInfo> epochs_vector;
  const auto& catalog = getCatalog();
  auto table_id = catalog.getMetadataForTable("test_table", false)->tableId;
  db_handler->get_table_epochs(epochs_vector, session_id, db_id, table_id);

  // rollback to zero
  for (auto& tei : epochs_vector) {
    CHECK_EQ(tei.table_epoch, row_count);
    tei.table_epoch = 0;
  }

  db_handler->set_table_epochs(session_id, db_id, epochs_vector);
  assertTableEpochs({0, 0});

  // insert another 21 rows
  row_count = 21;
  doInserts();
  assertTableEpochs({row_count, row_count});
  sqlAndCompareResult("select count(*) from test_table;", {{i(row_count)}});

  sql("alter table test_table set max_rollback_epochs = 5;");

  for (auto& tei : epochs_vector) {
    tei.table_epoch = 12;
  }
  // This should fail as we are attempting to rollback too far
  EXPECT_ANY_THROW(db_handler->set_table_epochs(session_id, db_id, epochs_vector));

  sql("alter table test_table set epoch = 16;");

  sqlAndCompareResult("select count(*) from test_table;", {{i(16)}});
  assertTableEpochs({16, 16});
}

TEST_F(SetTableEpochsTest, AddColumn) {
  loginAdmin();
  std::string select = "select * from test_table order by a;";

  sql("create table test_table(a int, shard key(a)) with (shard_count = 2, "
      "max_rollback_epochs = 25);");
  sql("insert into test_table values (0);");
  assertTableEpochs({1, 1});
  sqlAndCompareResult(select, {{i(0)}});

  sql("alter table test_table add b int default 0;");
  assertTableEpochs({2, 2});
  sqlAndCompareResult(select, {{i(0), i(0)}});

  sql("insert into test_table values (1,1);");
  sqlAndCompareResult(select, {{i(0), i(0)}, {i(1), i(1)}});
  assertTableEpochs({3, 3});

  try {
    assertAndSetTableEpoch(3, 1);
  } catch (const std::exception& e) {
    ASSERT_EQ("Cannot set epoch for table " + getTableIdString() +
                  " lower than the minimum rollback epoch (2).",
              std::string(e.what()));
  }

  sqlAndCompareResult(select, {{i(0), i(0)}, {i(1), i(1)}});

  // Make sure rollback to just after schema change works
  assertAndSetTableEpoch(3, 2);
  sqlAndCompareResult(select, {{i(0), i(0)}});
}

TEST_F(SetTableEpochsTest, DropColumn) {
  loginAdmin();
  std::string select = "select * from test_table order by a;";

  sql("create table test_table(a int, shard key(a), b int ) with (shard_count = 2, "
      "max_rollback_epochs = 25);");

  sql("insert into test_table values (0,0);");
  assertTableEpochs({1, 1});
  sqlAndCompareResult(select, {{i(0), i(0)}});

  sql("alter table test_table drop b;");
  assertTableEpochs({2, 2});
  sqlAndCompareResult(select, {{i(0)}});

  sql("insert into test_table values (1);");
  sqlAndCompareResult(select, {{i(0)}, {i(1)}});
  assertTableEpochs({3, 3});

  try {
    assertAndSetTableEpoch(3, 1);
  } catch (const std::exception& e) {
    ASSERT_EQ("Cannot set epoch for table " + getTableIdString() +
                  " lower than the minimum rollback epoch (2).",
              std::string(e.what()));
  }

  sqlAndCompareResult(select, {{i(0)}, {i(1)}});

  // Make sure rollback to just after schema change works
  assertAndSetTableEpoch(3, 2);
  sqlAndCompareResult(select, {{i(0)}});
}

class EpochValidationTest : public EpochConsistencyTest {
 protected:
  static void SetUpTestSuite() {
    DBHandlerTestFixture::SetUpTestSuite();
    switchToAdmin();
    sql("DROP DATABASE IF EXISTS test_db;");
    sql("CREATE DATABASE test_db;");
  }

  static void TearDownTestSuite() {
    switchToAdmin();
    sql("DROP DATABASE IF EXISTS test_db;");
    DBHandlerTestFixture::TearDownTestSuite();
  }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    login("admin", "HyperInteractive", "test_db");
    dropTestTables();
  }

  void TearDown() override {
    dropTestTables();
    DBHandlerTestFixture::TearDown();
  }

  void dropTestTables() {
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP TABLE IF EXISTS test_temp_table;");
    sql("DROP TABLE IF EXISTS test_arrow_table;");
    sql("DROP VIEW IF EXISTS test_view;");
    if (!isDistributedMode()) {
      sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    }
  }

  std::string getValidateStatement() {
    if (isDistributedMode()) {
      return "VALIDATE CLUSTER;";
    } else {
      return "VALIDATE;";
    }
  }

  std::string getWrongValidateStatement() {
    if (isDistributedMode()) {
      return "VALIDATE;";
    } else {
      return "VALIDATE CLUSTER;";
    }
  }

  std::string getSuccessfulValidationResult() {
    if (isDistributedMode()) {
      return "Cluster OK";
    } else {
      return "Instance OK";
    }
  }

  std::string getInconsistentEpochsValidationResult() {
    if (isDistributedMode()) {
      return "\nEpoch values for table \"test_table\" are inconsistent:"
             "\nNode      Table Id  Epoch     "
             "\n========= ========= ========= "
             "\nLeaf 0    2         1         "
             "\nLeaf 1    2         2         "
             "\n";
    } else {
      return "\nEpoch values for table \"test_table\" are inconsistent:"
             "\nTable Id  Epoch     "
             "\n========= ========= "
             "\n2         1         "
             "\n3         2         "
             "\n";
    }
  }

  std::string getNegativeInconsistentEpochsValidationResult() {
    if (isDistributedMode()) {
      return "\nEpoch values for table \"test_table\" are inconsistent:"
             "\nNode      Table Id  Epoch     "
             "\n========= ========= ========= "
             "\nLeaf 0    2         -1        "
             "\nLeaf 1    2         -2        "
             "\n";
    } else {
      return "\nEpoch values for table \"test_table\" are inconsistent:"
             "\nTable Id  Epoch     "
             "\n========= ========= "
             "\n2         -1        "
             "\n3         -2        "
             "\n";
    }
  }
};

TEST_F(EpochValidationTest, ValidInstance) {
  sql("create table test_table(a int, b text, shard key(a)) with (shard_count = 2);");
  sql("insert into test_table values(10, 'abc');");
  sqlAndCompareResult(getValidateStatement(), {{getSuccessfulValidationResult()}});
}

TEST_F(EpochValidationTest, InconsistentEpochs) {
  setUpTestTableWithInconsistentEpochs("test_db");
  sqlAndCompareResult(getValidateStatement(),
                      {{getInconsistentEpochsValidationResult()}});
}

// TODO: Investigate simulating the negative epoch state via mocking or remove
// test case if this is not feasible
TEST_F(EpochValidationTest, DISABLED_NegativeEpochs) {
  sql("create table test_table(a int, b text, shard key(a)) with (shard_count = 2);");
  setTableEpochs({-10, -10}, "test_db");
  sqlAndCompareResult(
      getValidateStatement(),
      {{"\nNegative epoch value found for table \"test_table\". Epoch: -10."}});
}

// TODO: Investigate simulating the negative epoch state via mocking or remove
// test case if this is not feasible
TEST_F(EpochValidationTest, DISABLED_NegativeAndInconsistentEpochs) {
  sql("create table test_table(a int, b text, shard key(a)) with (shard_count = 2);");
  setTableEpochs({-1, -2}, "test_db");
  sqlAndCompareResult(getValidateStatement(),
                      {{getNegativeInconsistentEpochsValidationResult()}});
}

TEST_F(EpochValidationTest, WrongValidationType) {
  queryAndAssertException(getWrongValidateStatement(),
                          "Unexpected validation type specified. Only the \"" +
                              getValidateStatement() +
                              "\" command is currently supported.");
}

TEST_F(EpochValidationTest, DifferentTableTypes) {
  sql("create table test_table(a int, b text, shard key(a)) with (shard_count = 2);");
  sql("create temporary table test_temp_table (a int, b text);");
  sql("create dataframe test_arrow_table (a int) from 'CSV:" +
      boost::filesystem::canonical("../../Tests/FsiDataFiles/0.csv").string() + "';");
  sql("create view test_view as select * from test_table;");
  if (!isDistributedMode()) {
    sql("create foreign table test_foreign_table(a int) server default_local_delimited "
        "with (file_path = '" +
        boost::filesystem::canonical("../../Tests/FsiDataFiles/0.csv").string() + "');");
  }
  sqlAndCompareResult(getValidateStatement(), {{getSuccessfulValidationResult()}});
}

class EmptyChunkRolloffTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
  }

  void TearDown() override {
    sql("drop table if exists test_table;");
    DBHandlerTestFixture::SetUp();
  }

  void clearFragmenters() {
    auto& catalog = getCatalog();
    auto td = catalog.getMetadataForTable("test_table");
    auto physical_tables = catalog.getPhysicalTablesDescriptors(td);
    for (auto table : physical_tables) {
      catalog.removeFragmenterForTable(table->tableId);
    }
  }
};

TEST_F(EmptyChunkRolloffTest, NonShardedTable) {
  sql("create table test_table (i integer, t text encoding none) with "
      "(max_rollback_epochs = 0);");
  sql("insert into test_table values (1, null);");

  // Force roll-off
  sql("update test_table set i = i + 1;");

  sqlAndCompareResult("select * from test_table order by i;", {{i(2), Null}});
  clearFragmenters();
  sqlAndCompareResult("select * from test_table order by i;", {{i(2), Null}});
}

TEST_F(EmptyChunkRolloffTest, ShardedTable) {
  sql("create table test_table (i integer, i2 integer, t text encoding none, shard "
      "key(i)) with (max_rollback_epochs = 0, shard_count = 4);");
  for (int i = 1; i <= 4; i++) {
    sql("insert into test_table values (" + std::to_string(i) + ", 1, null);");
  }

  // Force roll-off
  sql("update test_table set i2 = i + 1;");

  sqlAndCompareResult(
      "select * from test_table order by i;",
      {{i(1), i(2), Null}, {i(2), i(3), Null}, {i(3), i(4), Null}, {i(4), i(5), Null}});
  clearFragmenters();
  sqlAndCompareResult(
      "select * from test_table order by i;",
      {{i(1), i(2), Null}, {i(2), i(3), Null}, {i(3), i(4), Null}, {i(4), i(5), Null}});
}

int main(int argc, char** argv) {
  g_enable_fsi = true;
  testing::InitGoogleTest(&argc, argv);

  po::options_description desc("Options");
  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all test");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()(
      "cluster",
      po::value<std::string>(&DBHandlerTestFixture::cluster_config_file_path_),
      "Path to data leaves list JSON file.");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_base_path(BASE_PATH);
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  logger::init(log_options);

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
