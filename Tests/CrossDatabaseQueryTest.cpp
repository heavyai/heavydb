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

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "DBHandlerTestHelpers.h"
#include "Shared/SysDefinitions.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class CrossDatabaseQueryTest : public DBHandlerTestFixture {
 public:
  static void SetUpTestSuite() {
    createDBHandler();
    sql("CREATE DATABASE db_1;");
    sql("CREATE DATABASE db_2;");

    createTestTable(shared::kDefaultDbName, "test_table");
    insertIntoTestTable(shared::kDefaultDbName, "test_table", {1, 2, 3}, {"a", "b", "c"});

    createTestTable("db_1", "db_1_table", true);
    insertIntoTestTable(
        "db_1", "db_1_table", {1, 2, 10, 20, 30}, {"a", "b", "aa", "bb", "cc"});

    createTestTable("db_2", "db_2_table", true);
    insertIntoTestTable("db_2", "db_2_table", {1, 10, 100}, {"a", "aa", "aaa"});

    sql("CREATE FOREIGN TABLE test_foreign_table(t TEXT, i INTEGER, d DOUBLE) "
        "SERVER default_local_delimited "
        "WITH (file_path = '" +
        std::filesystem::canonical("../../Tests/FsiDataFiles/example_2.csv").string() +
        "');");

    switchToAdmin();
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName + " TO test_user;");

    sql("CREATE VIEW test_view AS SELECT * FROM db_2.db_2_table;");
  }

  static void TearDownTestSuite() {
    switchToAdmin();
    sql("DROP USER IF EXISTS test_user;");
    sql("DROP DATABASE IF EXISTS db_1;");
    sql("DROP DATABASE IF EXISTS db_2;");
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP VIEW IF EXISTS test_view;");
    std::filesystem::remove_all(export_file_path_);
  }

  static void createTestTable(const std::string& db_name,
                              const std::string& table_name,
                              bool replicate = false) {
    login(shared::kRootUsername, shared::kDefaultRootPasswd, db_name);
    std::string options{"fragment_size = 2"};
    if (replicate && isDistributedMode()) {
      options += ", partitions = 'replicated'";
    }
    sql("CREATE TABLE " + table_name +
        " (i INTEGER, t TEXT ENCODING DICT(32), t2 TEXT ENCODING NONE) WITH (" + options +
        ");");
  }

  static void insertIntoTestTable(const std::string& db_name,
                                  const std::string& table_name,
                                  const std::vector<int>& int_values,
                                  const std::vector<std::string>& text_values) {
    ASSERT_EQ(int_values.size(), text_values.size());
    std::string insert_statement{"INSERT INTO " + table_name + " VALUES"};
    for (size_t i = 0, j = int_values.size() - 1; i < int_values.size(); i++, j--) {
      if (i > 0) {
        insert_statement += ",";
      }
      ASSERT_GE(j, size_t(0));
      insert_statement += " (" + std::to_string(int_values[i]) + ", '" + text_values[i] +
                          "', '" + text_values[j] + "')";
    }
    insert_statement += ";";
    sql(insert_statement);
  }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    switchToAdmin();
  }

  void TearDown() override {
    switchToAdmin();
    sql("REVOKE ALL ON DATABASE db_2 FROM test_user;");
    DBHandlerTestFixture::TearDown();
  }

  void assertExportedFileContent(const std::vector<std::string>& expected_rows) {
    std::ifstream file{export_file_path_};
    ASSERT_TRUE(file.good());
    std::string line;
    std::vector<std::string> file_rows;
    while (std::getline(file, line)) {
      file_rows.emplace_back(line);
    }
    EXPECT_EQ(expected_rows, file_rows);
  }

  inline static const std::string export_file_path_ =
      (std::filesystem::canonical(BASE_PATH) / shared::kDefaultExportDirName /
       "test_export_file.csv")
          .string();
};

TEST_F(CrossDatabaseQueryTest, ProjectTableFromAnotherDb) {
  // clang-format off
  sqlAndCompareResult("SELECT power(db_1.db_1_table.i, 2) AS c1, db_1.db_1_table.t, "
                        "db_1.db_1_table.t2 "
                      "FROM db_1.db_1_table ORDER BY c1;",
                      {{1.0, "a", "cc"},
                       {4.0, "b", "bb"},
                       {100.0, "aa", "aa"},
                       {400.0, "bb", "b"},
                       {900.0, "cc", "a"}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, ProjectForeignTableFromAnotherDb) {
  // clang-format off
  sqlAndCompareResult("SELECT UPPER(t) as t1, i, d "
                      "FROM db_2.test_foreign_table ORDER BY i, t1;",
                      {{"A", i(1), 1.1},
                       {"AA", i(1), 1.1},
                       {"AAA", i(1), 1.1},
                       {"AA", i(2), 2.2},
                       {"AAA", i(2), 2.2},
                       {"AAA", i(3), 3.3}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, JoinBetweenTableInCurrentDbAndAnotherDb) {
  // clang-format off
  sqlAndCompareResult("SELECT table_1.i, table_2.t, table_2.t2, table_2.i * 2, "
                          "ST_Point(table_1.i, table_2.i) "
                        "FROM test_table as table_1, db_1.db_1_table as table_2 "
                        "WHERE table_1.t = table_2.t "
                        "ORDER BY i;",
                      {{i(1), "a", "cc", i(2), "POINT (1 1)"},
                       {i(2), "b", "bb", i(4), "POINT (2 2)"}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, SubqueryReferencingTableInAnotherDb) {
  // clang-format off
  sqlAndCompareResult("SELECT i, t, t2 FROM test_table "
                        "WHERE i > (SELECT MIN(i) FROM db_1.db_1_table) "
                        "ORDER BY i;",
                      {{i(2), "b", "b"},
                       {i(3), "c", "a"}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, AggregateAndFilterPredicateUsingColumnsFromDifferentDbs) {
  // clang-format off
  sqlAndCompareResult("SELECT AVG(table_1.i), table_2.t "
                        "FROM db_1.db_1_table as table_1, db_2.db_2_table as table_2 "
                        "WHERE table_1.t = table_2.t AND table_1.t2 != table_2.t2 "
                        "GROUP BY table_2.t ORDER BY table_2.t;",
                      {{1.0, "a"}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, LiteralFilterPredicateUsingColumnsFromDifferentDbs) {
  // clang-format off
  sqlAndCompareResult("SELECT i, t, t2 FROM db_1.db_1_table WHERE t = 'aa';",
                      {{i(10), "aa", "aa"}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, WindowFunctionUsingTableFromAnotherDb) {
  // clang-format off
  sqlAndCompareResult("SELECT t, LEAD(i) OVER(ORDER BY i) - i AS c1 "
                        "FROM db_2.db_2_table ORDER BY c1;",
                      {{"a", i(9)},
                       {"aa", i(90)},
                       {"aaa", Null}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, TableFunctionQueryReferencingAnotherDb) {
  if (isDistributedMode()) {
    GTEST_SKIP() << "Table functions are not supported in distributed mode.";
  }
  // clang-format off
  sqlAndCompareResult("SELECT * FROM TABLE("
                        "tf_feature_self_similarity("
                          "primary_features => CURSOR("
                            "SELECT t, t, i FROM db_2.db_2_table"
                          "),"
                          "use_tf_idf => false"
                        ")"
                      ") ORDER BY class1, class2;",
                      {{"a", "a", 1.0f},
                       {"a", "aa", 0.0f},
                       {"a", "aaa", 0.0f},
                       {"aa", "aa", 1.0f},
                       {"aa", "aaa", 0.0f},
                       {"aaa", "aaa", 1.0f}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, UnionBetweenTablesInCurrentDbAndAnotherDb) {
  // clang-format off
  sqlAndCompareResult("SELECT i, t FROM test_table "
                      "UNION ALL "
                      "SELECT i, t FROM db_2.db_2_table "
                      "ORDER BY i;",
                      {{i(1), "a"},
                       {i(1), "a"},
                       {i(2), "b"},
                       {i(3), "c"},
                       {i(10), "aa"},
                       {i(100), "aaa"}});
  // clang-format on
}

// TODO: Re-enable when text case expression bug is fixed.
TEST_F(CrossDatabaseQueryTest, DISABLED_StringFunctionWithColumnsFromDifferentDbs) {
  // clang-format off
  sqlAndCompareResult("SELECT CASE "
                          "WHEN table_1.i = 1 THEN table_2.t "
                          "WHEN table_1.i = 2 THEN table_1.t2 "
                          "ELSE 'abc' "
                        "END, table_2.t, upper(table_2.t2) "
                        "FROM test_table as table_1, db_1.db_1_table as table_2 "
                        "WHERE lower(table_1.t) = lower(table_2.t) AND table_2.t2 <> 'bb'"
                        "ORDER BY table_1.i;",
                      {{"a", "a", "CC"}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, SelectWithTablePermissionNoDbAccess) {
  sql("GRANT SELECT ON DATABASE db_2 TO test_user;");

  login("test_user", "test_pass", shared::kDefaultDbName);
  queryAndAssertException(
      "SELECT * FROM db_2.db_2_table;",
      "Unauthorized Access: user test_user is not allowed to access database db_2.");
}

TEST_F(CrossDatabaseQueryTest, SelectWithDbAccessButNoTablePermission) {
  sql("GRANT ACCESS ON DATABASE db_2 TO test_user;");

  login("test_user", "test_pass", shared::kDefaultDbName);
  queryAndAssertException("SELECT * FROM db_2.db_2_table;",
                          "Violation of access privileges: user test_user has no proper "
                          "privileges for object db_2_table");
}

TEST_F(CrossDatabaseQueryTest, SelectNonAdminUserWithTableAndDbPermissions) {
  sql("GRANT ACCESS, SELECT ON DATABASE db_2 TO test_user;");

  login("test_user", "test_pass", shared::kDefaultDbName);
  sqlAndCompareResult("SELECT * FROM db_2.db_2_table WHERE i > 10 ORDER BY i;",
                      {{i(100), "aaa", "a"}});
}

TEST_F(CrossDatabaseQueryTest, ViewReferencingTableInAnotherDB) {
  // clang-format off
  sqlAndCompareResult("SELECT * FROM test_view ORDER BY i;",
                      {{i(1), "a", "aaa"},
                       {i(10), "aa", "aa"},
                       {i(100), "aaa", "a"}});
  // clang-format on
}

TEST_F(CrossDatabaseQueryTest, ExportQueryReferencingTableInAnotherDB) {
  sql("COPY (SELECT * FROM db_2.db_2_table WHERE t = 'a' OR t = 'aaa' ORDER BY i) TO '" +
      export_file_path_ + "' WITH (file_type = 'csv', quoted='false', header='true');");
  assertExportedFileContent({"i,t,t2", "1,a,aaa", "100,aaa,a"});
}

TEST_F(CrossDatabaseQueryTest, TableIdWithMoreThanTwoComponents) {
  queryAndAssertPartialException("SELECT * FROM db_2.db_2_table.i;",
                                 "Object 'i' not found within 'DB_2.db_2_table'");
}

TEST_F(CrossDatabaseQueryTest, ValidateTableInAnotherDb) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TRowDescriptor result;
  db_handler->sql_validate(result, session_id, "SELECT * FROM db_2.db_2_table;");
  ASSERT_EQ(result.size(), size_t(3));
}

class CrossDatabaseWriteQueryTest : public CrossDatabaseQueryTest {
 public:
  static void SetUpTestSuite() { CrossDatabaseQueryTest::SetUpTestSuite(); }

  static void TearDownTestSuite() {
    std::filesystem::remove_all(dump_path_);
    CrossDatabaseQueryTest::TearDownTestSuite();
  }

  void TearDown() override {
    login(shared::kRootUsername, shared::kDefaultRootPasswd, "db_1");
    sql("DROP TABLE IF EXISTS test_table;");
    switchToAdmin();
    sql("DROP TABLE IF EXISTS test_table_2;");
    sql("REVOKE ALL ON DATABASE db_1 FROM test_user;");
  }

  static inline std::string dump_path_{std::filesystem::absolute("./test_dump").string()};
};

TEST_F(CrossDatabaseWriteQueryTest, UpdateTableInAnotherDb) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 2, 3}, {"a", "b", "c"});

  switchToAdmin();
  sql("UPDATE db_1.test_table SET i = i * 2, t = upper(t);");
  // clang-format off
  sqlAndCompareResult("SELECT * FROM db_1.test_table ORDER BY i;",
                      {{i(2), "A", "c"},
                       {i(4), "B", "b"},
                       {i(6), "C", "a"}});
  // clang-format on
}

TEST_F(CrossDatabaseWriteQueryTest, UpdateTableNoAccessPermissions) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 2, 3}, {"a", "b", "c"});

  sql("GRANT UPDATE ON DATABASE db_1 TO test_user;");
  login("test_user", "test_pass", shared::kDefaultDbName);
  queryAndAssertException(
      "UPDATE db_1.test_table SET i = i * 2, t = upper(t);",
      "Unauthorized Access: user test_user is not allowed to access database db_1.");
}

TEST_F(CrossDatabaseWriteQueryTest, UpdateTableNoUpdatePermissions) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 2, 3}, {"a", "b", "c"});
  sql("GRANT ACCESS ON DATABASE db_1 TO test_user;");

  login("test_user", "test_pass", shared::kDefaultDbName);
  queryAndAssertException("UPDATE db_1.test_table SET i = i * 2, t = upper(t);",
                          "Violation of access privileges: user test_user has no proper "
                          "privileges for object test_table");
}

TEST_F(CrossDatabaseWriteQueryTest, UpdateTableBothUpdateAndAccessPermissions) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 2, 3}, {"a", "b", "c"});
  sql("GRANT ACCESS, UPDATE ON DATABASE db_1 TO test_user;");

  login("test_user", "test_pass", shared::kDefaultDbName);
  sql("UPDATE db_1.test_table SET i = i * 2, t = upper(t);");

  switchToAdmin();
  // clang-format off
  sqlAndCompareResult("SELECT * FROM db_1.test_table ORDER BY i;",
                      {{i(2), "A", "c"},
                       {i(4), "B", "b"},
                       {i(6), "C", "a"}});
  // clang-format on
}

TEST_F(CrossDatabaseWriteQueryTest, UpdateTableInAnotherDbWithCurrentDbSubquery) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 10, 100}, {"A", "B", "C"});

  switchToAdmin();
  sql("UPDATE db_1.test_table SET "
      "t = (SELECT t FROM db_2.db_2_table WHERE db_1.test_table.i = db_2.db_2_table.i);");
  // clang-format off
  sqlAndCompareResult("SELECT * FROM db_1.test_table ORDER BY i;",
                      {{i(1), "a", "C"},
                       {i(10), "aa", "B"},
                       {i(100), "aaa", "A"}});
  // clang-format on
}

TEST_F(CrossDatabaseWriteQueryTest, DeleteTableInAnotherDb) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 2, 3}, {"a", "b", "c"});

  switchToAdmin();
  sql("DELETE FROM db_1.test_table WHERE i > 2;");
  // clang-format off
  sqlAndCompareResult("SELECT * FROM db_1.test_table ORDER BY i;",
                      {{i(1), "a", "c"},
                       {i(2), "b", "b"}});
  // clang-format on
}

TEST_F(CrossDatabaseWriteQueryTest, DeleteNoAccessPermissions) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 2, 3}, {"a", "b", "c"});
  sql("GRANT DELETE ON DATABASE db_1 TO test_user;");

  login("test_user", "test_pass", shared::kDefaultDbName);
  queryAndAssertException(
      "DELETE FROM db_1.test_table WHERE i > 2;",
      "Unauthorized Access: user test_user is not allowed to access database db_1.");
}

TEST_F(CrossDatabaseWriteQueryTest, DeleteNoDeletePermissions) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 2, 3}, {"a", "b", "c"});
  sql("GRANT ACCESS ON DATABASE db_1 TO test_user;");

  login("test_user", "test_pass", shared::kDefaultDbName);
  queryAndAssertException("DELETE FROM db_1.test_table WHERE i > 2;",
                          "Violation of access privileges: user test_user has no proper "
                          "privileges for object test_table");
}

TEST_F(CrossDatabaseWriteQueryTest, DeleteBothAccessAndDeletePermissions) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 2, 3}, {"a", "b", "c"});
  sql("GRANT ACCESS, DELETE ON DATABASE db_1 TO test_user;");

  login("test_user", "test_pass", shared::kDefaultDbName);
  sql("DELETE FROM db_1.test_table WHERE i > 2;");

  switchToAdmin();
  // clang-format off
  sqlAndCompareResult("SELECT * FROM db_1.test_table ORDER BY i;",
                      {{i(1), "a", "c"},
                       {i(2), "b", "b"}});
  // clang-format on
}

TEST_F(CrossDatabaseWriteQueryTest,
       DISABLED_DeleteTableInAnotherDbWithCurrentDbSubquery) {
  createTestTable("db_1", "test_table");
  insertIntoTestTable("db_1", "test_table", {1, 2, 3}, {"a", "b", "c"});

  switchToAdmin();
  sql("DELETE FROM db_1.test_table WHERE i IN (SELECT i FROM db_2.db_2_table);");
  // clang-format off
  sqlAndCompareResult("SELECT * FROM db_1.test_table ORDER BY i;",
                      {{i(2), "b", "b"},
                       {i(3), "c", "aa"}});
  // clang-format on
}

TEST_F(CrossDatabaseWriteQueryTest, CtasWithQueryReferencingTableInAnotherDb) {
  sql("CREATE TABLE test_table_2 AS SELECT * FROM db_2.db_2_table;");
  // clang-format off
  sqlAndCompareResult("SELECT * FROM test_table_2 ORDER BY i;",
                      {{i(1), "a", "aaa"},
                       {i(10), "aa", "aa"},
                       {i(100), "aaa", "a"}});
  // clang-format on
}

TEST_F(CrossDatabaseWriteQueryTest, ItasWithQueryReferencingTableInAnotherDb) {
  createTestTable(shared::kDefaultDbName, "test_table_2");
  sql("INSERT INTO test_table_2 SELECT * FROM db_2.db_2_table;");
  // clang-format off
  sqlAndCompareResult("SELECT * FROM test_table_2 ORDER BY i;",
                      {{i(1), "a", "aaa"},
                       {i(10), "aa", "aa"},
                       {i(100), "aaa", "a"}});
  // clang-format on
}

TEST_F(CrossDatabaseWriteQueryTest, CtasCreateTableInAnotherDb) {
  queryAndAssertPartialException(
      "CREATE TABLE db_1.test_table AS SELECT * FROM db_2.db_2_table;",
      "SQL Error: Encountered \".\"");
}

TEST_F(CrossDatabaseWriteQueryTest, ItasInsertIntoTableFromAnotherDb) {
  createTestTable("db_1", "test_table");

  switchToAdmin();
  queryAndAssertException("INSERT INTO db_1.test_table SELECT * FROM db_2.db_2_table;",
                          "ITAS failed: table db_1.test_table does not exist.");
}

TEST_F(CrossDatabaseWriteQueryTest, InsertIntoTableFromAnotherDb) {
  createTestTable("db_1", "test_table");

  switchToAdmin();
  queryAndAssertException(
      "INSERT INTO db_1.test_table VALUES (1000, 'abc', 'abc');",
      "Table/View db_1.test_table for catalog heavyai does not exist");
}

TEST_F(CrossDatabaseWriteQueryTest, CreateTableInAnotherDb) {
  queryAndAssertPartialException("CREATE TABLE db_1.test_table (i INTEGER);",
                                 "SQL Error: Encountered \".\"");
}

TEST_F(CrossDatabaseWriteQueryTest, CreateForeignTableInAnotherDb) {
  queryAndAssertPartialException(
      "CREATE FOREIGN TABLE db_1.test_foreign_table(i INTEGER) "
      "SERVER default_local_delimited "
      "WITH (file_path = '../../Tests/FsiDataFiles/0.csv');",
      "SQL Error: Encountered \".\"");
}

TEST_F(CrossDatabaseWriteQueryTest, DropTableInAnotherDb) {
  queryAndAssertException(
      "DROP TABLE db_2.db_2_table;",
      "Table/View db_2.db_2_table for catalog heavyai does not exist");
}

TEST_F(CrossDatabaseWriteQueryTest, TruncateTableInAnotherDb) {
  queryAndAssertException(
      "TRUNCATE TABLE db_2.db_2_table;",
      "Table/View db_2.db_2_table for catalog heavyai does not exist");
}

TEST_F(CrossDatabaseWriteQueryTest, AlterTableInAnotherDb) {
  queryAndAssertException(
      "ALTER TABLE db_2.db_2_table RENAME COLUMN i TO i2;",
      "Table/View db_2.db_2_table for catalog heavyai does not exist");
}

TEST_F(CrossDatabaseWriteQueryTest, DumpTableInAnotherDb) {
  if (isDistributedMode()) {
    GTEST_SKIP() << "Dump/Restore is not supported in distributed mode.";
  }
  queryAndAssertException(
      "DUMP TABLE db_2.db_2_table TO '" + dump_path_ + "';",
      "Table/View db_2.db_2_table for catalog heavyai does not exist");
}

TEST_F(CrossDatabaseWriteQueryTest, RestoreTableInAnotherDb) {
  if (isDistributedMode()) {
    GTEST_SKIP() << "Dump/Restore is not supported in distributed mode.";
  }
  sql("DUMP TABLE test_table TO '" + dump_path_ + "';");
  ASSERT_TRUE(std::filesystem::exists(dump_path_));

  queryAndAssertPartialException(
      "RESTORE TABLE db_2.test_table FROM '" + dump_path_ + "';",
      "SQL Error: Encountered \".\"");
}

TEST_F(CrossDatabaseWriteQueryTest, RenameTableInAnotherDb) {
  queryAndAssertException("RENAME TABLE db_2.db_2_table TO db_2.db_2_table_2;",
                          "Source table 'db_2.db_2_table' does not exist.");
}

TEST_F(CrossDatabaseWriteQueryTest, OptimizeTableInAnotherDb) {
  queryAndAssertException(
      "OPTIMIZE TABLE db_2.db_2_table;",
      "Table/View db_2.db_2_table for catalog heavyai does not exist");
}

// TODO: Setup distributed tests

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  DBHandlerTestFixture::initTestArgs(argc, argv);

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  g_enable_fsi = false;
  return err;
}
