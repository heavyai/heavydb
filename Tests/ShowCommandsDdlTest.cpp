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

/**
 * @file ShowCommandsDdlTest.cpp
 * @brief Test suite for SHOW DDL commands
 */

#include <gtest/gtest.h>

#include "DBHandlerTestHelpers.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class ShowUserSessionsTest : public DBHandlerTestFixture {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    users = {"user1", "user2", "user3"};
    superusers = {"super1", "super2", "super3"};
    dbs = {"db1", "db2", "db3"};
    // Default connection string outside of thrift
    connection_string = "tcp:";
    createDBs();
    createUsers();
    createSuperUsers();
    // Check that default only user session exists
    TQueryResult result;
    sql(result, "SHOW USER SESSIONS;");
    assertExpectedFormat(result);
    assertNumSessions(result, 1);
    assertSessionResultFound(result, "admin", "omnisci", 1);
    getID(result, "admin", "omnisci", admin_id);
  }

  void TearDown() override {
    dropUsers();
    dropSuperUsers();
    dropDBs();

    // Check that default only user session still exists
    TQueryResult result;
    sql(result, "SHOW USER SESSIONS;");
    assertExpectedFormat(result);
    assertNumSessions(result, 1);
    assertSessionResultFound(result, "admin", "omnisci", admin_id);
    DBHandlerTestFixture::TearDown();
  }

  void createUsers() {
    for (const auto& user : users) {
      std::stringstream create;
      create << "CREATE USER " << user
             << " (password = 'HyperInteractive', is_super = 'false', "
                "default_db='omnisci');";
      sql(create.str());
      for (const auto& db : dbs) {
        std::stringstream grant;
        grant << "GRANT ALL ON DATABASE  " << db << " to " << user << ";";
        sql(grant.str());
      }
    }
  }

  void createSuperUsers() {
    for (const auto& user : superusers) {
      std::stringstream create;
      create
          << "CREATE USER " << user
          << " (password = 'HyperInteractive', is_super = 'true', default_db='omnisci');";
      sql(create.str());
      for (const auto& db : dbs) {
        std::stringstream grant;
        grant << "GRANT ALL ON DATABASE  " << db << " to " << user << ";";
        sql(grant.str());
      }
    }
  }

  void dropUsers() {
    for (const auto& user : users) {
      std::stringstream drop;
      drop << "DROP USER " << user << ";";
      sql(drop.str());
    }
  }

  void dropSuperUsers() {
    for (const auto& user : superusers) {
      std::stringstream drop;
      drop << "DROP USER " << user << ";";
      sql(drop.str());
    }
  }

  void createDBs() {
    for (const auto& db : dbs) {
      std::stringstream create;
      create << "CREATE DATABASE " << db << " (owner = 'admin');";
      sql(create.str());
    }
  }

  void dropDBs() {
    for (const auto& db : dbs) {
      std::stringstream drop;
      drop << "DROP DATABASE " << db << ";";
      sql(drop.str());
    }
  }

  enum ColumnIndex { ID, USERNAME, CONNECTION_INFO, DB_NAME };

  void assertExpectedFormat(const TQueryResult& result) {
    ASSERT_EQ(result.row_set.is_columnar, true);
    ASSERT_EQ(result.row_set.columns.size(), size_t(4));
    ASSERT_EQ(result.row_set.row_desc[ID].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[ID].col_name, "session_id");
    ASSERT_EQ(result.row_set.row_desc[USERNAME].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[USERNAME].col_name, "login_name");
    ASSERT_EQ(result.row_set.row_desc[CONNECTION_INFO].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[CONNECTION_INFO].col_name, "client_address");
    ASSERT_EQ(result.row_set.row_desc[DB_NAME].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[DB_NAME].col_name, "db_name");
  }

  // assert expected_matches results with username, database,
  void assertSessionResultFound(const TQueryResult& result,
                                const std::string& username,
                                const std::string& db,
                                int expected_matches) {
    int num_matches = 0;
    for (size_t i = 0; i < result.row_set.columns[ID].data.str_col.size(); ++i) {
      if (result.row_set.columns[USERNAME].data.str_col[i] == username &&
          result.row_set.columns[DB_NAME].data.str_col[i] == db) {
        num_matches++;
      }
    }
    ASSERT_EQ(num_matches, expected_matches);
  }

  // assert one result with username, database, and ID found
  void assertSessionResultFound(const TQueryResult& result,
                                const std::string& username,
                                const std::string& db,
                                const std::string& id) {
    int num_matches = 0;
    for (size_t i = 0; i < result.row_set.columns[ID].data.str_col.size(); ++i) {
      if (result.row_set.columns[USERNAME].data.str_col[i] == username &&
          result.row_set.columns[DB_NAME].data.str_col[i] == db &&
          result.row_set.columns[ID].data.str_col[i] == id &&
          result.row_set.columns[CONNECTION_INFO].data.str_col[i] == connection_string) {
        num_matches++;
      }
    }
    ASSERT_EQ(num_matches, 1);
  }

  // Get ID of unique session with username and database
  void getID(const TQueryResult& result,
             const std::string& username,
             const std::string& db,
             std::string& retval) {
    for (size_t i = 0; i < result.row_set.columns[ID].data.str_col.size(); ++i) {
      if (result.row_set.columns[USERNAME].data.str_col[i] == username &&
          result.row_set.columns[DB_NAME].data.str_col[i] == db &&
          result.row_set.columns[CONNECTION_INFO].data.str_col[i] == connection_string) {
        // Only one match should be found
        ASSERT_EQ(retval.length(), size_t(0));
        retval = result.row_set.columns[ID].data.str_col[i];
        ASSERT_GT(retval.length(), size_t(0));
      }
    }
    ASSERT_GT(retval.length(), size_t(0));
  }

  void assertNumSessions(const TQueryResult& result, size_t num_session) {
    ASSERT_EQ(num_session, result.row_set.columns[ID].data.str_col.size());
  }
  std::vector<std::string> get_users() { return users; }
  std::vector<std::string> get_superusers() { return superusers; }
  std::vector<std::string> get_dbs() { return dbs; }

 private:
  std::vector<std::string> users;
  std::vector<std::string> superusers;
  std::vector<std::string> dbs;

  std::string admin_id;
  std::string connection_string;
};

TEST_F(ShowUserSessionsTest, SHOW) {
  // check default admin session is created
  TQueryResult result;
  sql(result, "SHOW USER SESSIONS;");
  assertExpectedFormat(result);
  assertNumSessions(result, 1);
  assertSessionResultFound(result, "admin", "omnisci", 1);
}

TEST_F(ShowUserSessionsTest, SHOW_ADMIN_MULTIDB) {
  TSessionId new_session;
  login("admin", "HyperInteractive", "db1", new_session);
  TQueryResult result;
  sql(result, "SHOW USER SESSIONS;");
  assertExpectedFormat(result);
  assertNumSessions(result, 2);
  assertSessionResultFound(result, "admin", "db1", 1);
  assertSessionResultFound(result, "admin", "omnisci", 1);
  logout(new_session);
}

TEST_F(ShowUserSessionsTest, SHOW_ADMIN_MULTISESSION_SINGLEDB) {
  TSessionId new_session;
  login("admin", "HyperInteractive", "omnisci", new_session);
  TQueryResult result;
  std::string query{"SHOW USER SESSIONS;"};
  sql(result, query);
  assertExpectedFormat(result);
  assertNumSessions(result, 2);
  assertSessionResultFound(result, "admin", "omnisci", 2);
  logout(new_session);
}

TEST_F(ShowUserSessionsTest, SHOW_USERS_MULTISESSION) {
  TSessionId session1;
  login("user1", "HyperInteractive", "db1", session1);
  TSessionId session2;
  login("user2", "HyperInteractive", "db1", session2);
  TQueryResult result;
  std::string query{"SHOW USER SESSIONS;"};
  sql(result, query);
  assertExpectedFormat(result);
  assertNumSessions(result, 3);
  assertSessionResultFound(result, "admin", "omnisci", 1);
  assertSessionResultFound(result, "user1", "db1", 1);
  assertSessionResultFound(result, "user2", "db1", 1);
  logout(session1);
  logout(session2);
}

TEST_F(ShowUserSessionsTest, SHOW_USERS_MULTIDBS) {
  TSessionId session1;
  login("user1", "HyperInteractive", "db1", session1);
  TSessionId session2;
  login("user2", "HyperInteractive", "db2", session2);
  TQueryResult result;
  std::string query{"SHOW USER SESSIONS;"};
  sql(result, query);
  assertExpectedFormat(result);
  assertNumSessions(result, 3);
  assertSessionResultFound(result, "admin", "omnisci", 1);
  assertSessionResultFound(result, "user1", "db1", 1);
  assertSessionResultFound(result, "user2", "db2", 1);
  logout(session1);
  logout(session2);
}

TEST_F(ShowUserSessionsTest, SHOW_USERS_ALL) {
  std::vector<TSessionId> session_ids;

  for (int copies = 1; copies < 5; copies++) {
    for (auto const& user : get_users()) {
      for (auto const& db : get_dbs()) {
        TSessionId session;
        login(user, "HyperInteractive", db, session);
        session_ids.push_back(session);
      }
    }

    TQueryResult result;
    sql(result, "SHOW USER SESSIONS;");
    assertExpectedFormat(result);
    assertNumSessions(result, 1 + (copies * get_users().size() * get_dbs().size()));
    for (auto const& user : get_users()) {
      for (auto const& db : get_dbs()) {
        assertSessionResultFound(result, user, db, copies);
      }
    }
  }

  for (auto const& session_id : session_ids) {
    logout(session_id);
  }
}

TEST_F(ShowUserSessionsTest, SHOW_USERS_MULTIDB_LOGOUT) {
  TSessionId session1;
  login("user1", "HyperInteractive", "db1", session1);
  TSessionId session2;
  login("user2", "HyperInteractive", "db2", session2);
  std::string session2_id;
  {
    TQueryResult result;
    sql(result, "SHOW USER SESSIONS;");
    assertExpectedFormat(result);
    assertNumSessions(result, 3);
    assertSessionResultFound(result, "admin", "omnisci", 1);
    assertSessionResultFound(result, "user1", "db1", 1);
    assertSessionResultFound(result, "user2", "db2", 1);
    getID(result, "user2", "db2", session2_id);
  }

  logout(session1);
  {
    TQueryResult result;
    sql(result, "SHOW USER SESSIONS;");
    assertExpectedFormat(result);
    assertNumSessions(result, 2);
    assertSessionResultFound(result, "admin", "omnisci", 1);
    assertSessionResultFound(result, "user2", "db2", session2_id);
  }

  logout(session2);
  {
    TQueryResult result;
    sql(result, "SHOW USER SESSIONS;");
    assertExpectedFormat(result);
    assertNumSessions(result, 1);
    assertSessionResultFound(result, "admin", "omnisci", 1);
  }
}

TEST_F(ShowUserSessionsTest, PRIVILEGES_SUPERUSER) {
  TSessionId supersession;
  login("super1", "HyperInteractive", "db2", supersession);
  {
    TQueryResult result;
    std::string query{"SHOW USER SESSIONS;"};
    sql(result, query, supersession);
    assertExpectedFormat(result);
    assertNumSessions(result, 2);
    assertSessionResultFound(result, "admin", "omnisci", 1);
    assertSessionResultFound(result, "super1", "db2", 1);
  }
  logout(supersession);
}

TEST_F(ShowUserSessionsTest, PRIVILEGES_NONSUPERUSER) {
  TSessionId usersession;
  login("user1", "HyperInteractive", "db1", usersession);

  try {
    TQueryResult result;
    std::string query{"SHOW USER SESSIONS;"};
    sql(result, query, usersession);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TOmniSciException& e) {
    ASSERT_EQ(
        "Exception: SHOW USER SESSIONS failed, because it can only be executed by super "
        "user.",
        e.error_msg);
  }

  logout(usersession);
}

class ShowTableDdlTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    loginAdmin();
    createTestUser();
  }

  void TearDown() override {
    loginAdmin();
    dropTestTable();
    dropTestUser();
  }

  void createTestUser() {
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE omnisci TO test_user;");
  }

  void dropTestUser() {
    try {
      sql("DROP USER test_user;");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }

  void assertExpectedQueryFormat(const TQueryResult& result) const {
    ASSERT_EQ(result.row_set.is_columnar, true);
    ASSERT_EQ(result.row_set.columns.size(), 1UL);
    ASSERT_EQ(result.row_set.row_desc[0].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[0].col_name, "table_name");
  }

  void assertExpectedQuery(
      const TQueryResult& result,
      const std::vector<std::string>& expected_values,
      const std::vector<std::string>& expected_missing_values) const {
    assertExpectedQueryFormat(result);
    auto& result_values = result.row_set.columns[0].data.str_col;
    // TODO: at the moment, this checks that expected_values are a subset of
    // result_values; once other tests ensure they do not leave behind undropped tables,
    // this can be changed to be a check for equality of expected and result values
    std::unordered_set<std::string> result_values_set(result_values.begin(),
                                                      result_values.end());
    for (auto& value : expected_values) {
      ASSERT_FALSE(result_values_set.find(value) == result_values_set.end());
    }
    for (auto& value : expected_missing_values) {
      ASSERT_TRUE(result_values_set.find(value) == result_values_set.end());
    }
  }

  void assertExpectedQuery(const TQueryResult& result,
                           const std::vector<std::string>& expected_values) const {
    std::vector<std::string> expected_missing_values;
    assertExpectedQuery(result, expected_values, expected_missing_values);
  }

  void createTestTable() { sql("CREATE TABLE test_table ( test_val int );"); }

  void dropTestTable() { sql("DROP TABLE IF EXISTS test_table;"); }
};

TEST_F(ShowTableDdlTest, CreateTestTable) {
  createTestTable();
  TQueryResult result;
  std::vector<std::string> expected_result{"test_table"};
  sql(result, "SHOW TABLES;");
  assertExpectedQuery(result, expected_result);
}

TEST_F(ShowTableDdlTest, CreateTwoTestTablesDropOne) {
  createTestTable();
  sql("CREATE TABLE test_table2 ( test_val int );");
  {
    TQueryResult result;
    std::vector<std::string> expected_result{"test_table", "test_table2"};
    sql(result, "SHOW TABLES;");
    assertExpectedQuery(result, expected_result);
  }
  dropTestTable();
  {
    TQueryResult result;
    std::vector<std::string> expected_result{"test_table2"};
    std::vector<std::string> expected_missing_result{"test_table"};
    sql(result, "SHOW TABLES;");
    assertExpectedQuery(result, expected_result, expected_missing_result);
  }
  sql("DROP TABLE test_table2;");
}

TEST_F(ShowTableDdlTest, TestUserSeesNoTables) {
  login("test_user", "test_pass");
  TQueryResult result;
  std::vector<std::string> expected_result{};
  sql(result, "SHOW TABLES;");
  assertExpectedQuery(result, expected_result);
}

TEST_F(ShowTableDdlTest, CreateTestTableDropTestTable) {
  createTestTable();
  dropTestTable();
  TQueryResult result;
  std::vector<std::string> expected_missing_result{"test_table"};
  sql(result, "SHOW TABLES;");
  assertExpectedQuery(result, {}, expected_missing_result);
}

TEST_F(ShowTableDdlTest, TestUserSeesTestTableAfterGrantSelect) {
  createTestTable();
  sql("GRANT SELECT ON TABLE test_table TO test_user;");
  login("test_user", "test_pass");
  TQueryResult result;
  std::vector<std::string> expected_result{"test_table"};
  sql(result, "SHOW TABLES;");
  assertExpectedQuery(result, expected_result);
}

TEST_F(ShowTableDdlTest, TestUserSeesTestTableAfterGrantDrop) {
  createTestTable();
  sql("GRANT DROP ON TABLE test_table TO test_user;");
  login("test_user", "test_pass");
  TQueryResult result;
  std::vector<std::string> expected_result{"test_table"};
  sql(result, "SHOW TABLES;");
  assertExpectedQuery(result, expected_result);
}

TEST_F(ShowTableDdlTest, SuperUserSeesTestTableAfterTestUserCreates) {
  sql("GRANT CREATE TABLE ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");
  createTestTable();
  loginAdmin();
  TQueryResult result;
  std::vector<std::string> expected_result{"test_table"};
  sql(result, "SHOW TABLES;");
  assertExpectedQuery(result, expected_result);
}

TEST_F(ShowTableDdlTest, CreateTableCreateViewAndViewNotSeen) {
  createTestTable();
  sql("CREATE VIEW test_view AS SELECT * from test_table;");
  TQueryResult result;
  std::vector<std::string> expected_result{"test_table"};
  std::vector<std::string> expected_missing_result{"test_view"};
  sql(result, "SHOW TABLES;");
  assertExpectedQuery(result, expected_result, expected_missing_result);
  sql("DROP VIEW test_view;");
}

class ShowDatabasesTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    createTestUser("test_user_1", "test_pass_1");
    createTestUser("test_user_2", "test_pass_2");
    createTestUser("test_super_user", "test_pass", true);
  }

  void TearDown() override {
    dropTestUser("test_user_1");
    dropTestUser("test_user_2");
    dropTestUser("test_super_user");
    sql("DROP DATABASE IF EXISTS test_db_1;");
    sql("DROP DATABASE IF EXISTS test_db_2;");
    DBHandlerTestFixture::TearDown();
  }

  void assertExpectedResult(const std::vector<std::string> headers,
                            const std::vector<std::vector<std::string>> rows,
                            const TQueryResult& result) {
    const auto& row_set = result.row_set;
    const auto& row_descriptor = result.row_set.row_desc;

    ASSERT_TRUE(row_set.is_columnar);
    ASSERT_EQ(headers.size(), row_descriptor.size());
    ASSERT_FALSE(row_set.columns.empty());

    for (size_t i = 0; i < headers.size(); i++) {
      ASSERT_EQ(row_descriptor[i].col_name, headers[i]);
      ASSERT_EQ(TDatumType::type::STR, row_descriptor[i].col_type.type);
    }

    for (const auto& column : row_set.columns) {
      ASSERT_EQ(rows.size(), column.data.str_col.size());
    }

    for (size_t row = 0; row < rows.size(); row++) {
      for (size_t column = 0; column < rows[row].size(); column++) {
        ASSERT_EQ(rows[row][column], row_set.columns[column].data.str_col[row]);
        ASSERT_FALSE(row_set.columns[column].nulls[row]);
      }
    }
  }

  void createTestUser(const std::string& user_name,
                      const std::string& pass,
                      const bool is_super_user = false) {
    sql("CREATE USER " + user_name + " (password = '" + pass + "', is_super = '" +
        (is_super_user ? "true" : "false") + "');");
  }

  void dropTestUser(const std::string& user_name) {
    loginAdmin();
    try {
      sql("DROP USER " + user_name + ";");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }
};

TEST_F(ShowDatabasesTest, DefaultDatabase) {
  TQueryResult result;
  sql(result, "SHOW DATABASES;");
  // clang-format off
  assertExpectedResult({"Database", "Owner"},
                       {{"omnisci", "admin"}},
                       result);
  // clang-format on
}

TEST_F(ShowDatabasesTest, UserCreatedDatabase) {
  sql("CREATE DATABASE test_db_1 (owner = 'test_user_1');");
  login("test_user_1", "test_pass_1", "test_db_1");

  TQueryResult result;
  sql(result, "SHOW DATABASES;");
  // clang-format off
  assertExpectedResult({"Database", "Owner"},
                       {{"test_db_1", "test_user_1"}},
                       result);
  // clang-format on
}

TEST_F(ShowDatabasesTest, OtherUserDatabaseWithNoAccessPrivilege) {
  sql("CREATE DATABASE test_db_1 (owner = 'test_user_1');");
  sql("CREATE DATABASE test_db_2 (owner = 'test_user_2');");
  login("test_user_1", "test_pass_1", "test_db_1");

  TQueryResult result;
  sql(result, "SHOW DATABASES;");
  // clang-format off
  assertExpectedResult({"Database", "Owner"},
                       {{"test_db_1", "test_user_1"}},
                       result);
  // clang-format on
}

TEST_F(ShowDatabasesTest, OtherUserDatabaseWithAccessPrivilege) {
  sql("CREATE DATABASE test_db_1 (owner = 'test_user_1');");
  sql("CREATE DATABASE test_db_2 (owner = 'test_user_2');");
  sql("GRANT ACCESS ON DATABASE test_db_2 to test_user_1;");
  login("test_user_1", "test_pass_1", "test_db_1");

  TQueryResult result;
  sql(result, "SHOW DATABASES;");
  // clang-format off
  assertExpectedResult({"Database", "Owner"},
                       {{"test_db_1", "test_user_1"},
                        {"test_db_2", "test_user_2"}},
                       result);
  // clang-format on
}

TEST_F(ShowDatabasesTest, AdminLoginAndOtherUserDatabases) {
  sql("CREATE DATABASE test_db_1 (owner = 'test_user_1');");
  sql("CREATE DATABASE test_db_2 (owner = 'test_user_2');");

  TQueryResult result;
  sql(result, "SHOW DATABASES;");
  // clang-format off
  assertExpectedResult(
      {"Database", "Owner"},
      {{"omnisci", "admin"},
       {"test_db_1", "test_user_1"},
       {"test_db_2", "test_user_2"}},
      result);
  // clang-format on
}

TEST_F(ShowDatabasesTest, SuperUserLoginAndOtherUserDatabases) {
  sql("CREATE DATABASE test_db_1 (owner = 'test_user_1');");
  sql("CREATE DATABASE test_db_2 (owner = 'test_user_2');");
  login("test_super_user", "test_pass");

  TQueryResult result;
  sql(result, "SHOW DATABASES;");
  // clang-format off
  assertExpectedResult(
      {"Database", "Owner"},
      {{"omnisci", "admin"},
       {"test_db_1", "test_user_1"},
       {"test_db_2", "test_user_2"}},
      result);
  // clang-format on
}

class ShowCreateTableTest : public DBHandlerTestFixture {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("DROP TABLE IF EXISTS showcreatetabletest;");
    sql("DROP TABLE IF EXISTS showcreatetabletest1;");
    sql("DROP TABLE IF EXISTS showcreatetabletest2;");
    sql("DROP VIEW IF EXISTS showcreateviewtest;");
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS showcreatetabletest;");
    sql("DROP TABLE IF EXISTS showcreatetabletest1;");
    sql("DROP TABLE IF EXISTS showcreatetabletest2;");
    sql("DROP VIEW IF EXISTS showcreateviewtest;");
    DBHandlerTestFixture::TearDown();
  }
};

TEST_F(ShowCreateTableTest, Identity) {
  // clang-format off
  std::vector<std::string> creates = {
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (FRAGMENT_SIZE=123);",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (MAX_CHUNK_SIZE=123);",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (PAGE_SIZE=123);",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (MAX_ROWS=123);",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (VACUUM='IMMEDIATE');",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (PARTITIONS='SHARDED');",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (PARTITIONS='REPLICATED');",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER,\n  SHARD KEY (i))\nWITH (SHARD_COUNT=4);",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (SORT_COLUMN='i');",
    "CREATE TABLE showcreatetabletest (\n  i1 INTEGER,\n  i2 INTEGER)\nWITH (MAX_ROWS=123, VACUUM='IMMEDIATE');",
    "CREATE TABLE showcreatetabletest (\n  id TEXT ENCODING DICT(32),\n  abbr TEXT ENCODING DICT(32),\n  name TEXT ENCODING DICT(32),\n  omnisci_geo GEOMETRY(MULTIPOLYGON, 4326) NOT NULL);",
    "CREATE TABLE showcreatetabletest (\n  flight_year SMALLINT,\n  flight_month SMALLINT,\n  flight_dayofmonth SMALLINT,\n  flight_dayofweek SMALLINT,\n  deptime SMALLINT,\n  crsdeptime SMALLINT,\n  arrtime SMALLINT,\n  crsarrtime SMALLINT,\n  uniquecarrier TEXT ENCODING DICT(32),\n  flightnum SMALLINT,\n  tailnum TEXT ENCODING DICT(32),\n  actualelapsedtime SMALLINT,\n  crselapsedtime SMALLINT,\n  airtime SMALLINT,\n  arrdelay SMALLINT,\n  depdelay SMALLINT,\n  origin TEXT ENCODING DICT(32),\n  dest TEXT ENCODING DICT(32),\n  distance SMALLINT,\n  taxiin SMALLINT,\n  taxiout SMALLINT,\n  cancelled SMALLINT,\n  cancellationcode TEXT ENCODING DICT(32),\n  diverted SMALLINT,\n  carrierdelay SMALLINT,\n  weatherdelay SMALLINT,\n  nasdelay SMALLINT,\n  securitydelay SMALLINT,\n  lateaircraftdelay SMALLINT,\n  dep_timestamp TIMESTAMP(0),\n  arr_timestamp TIMESTAMP(0),\n  carrier_name TEXT ENCODING DICT(32),\n  plane_type TEXT ENCODING DICT(32),\n  plane_manufacturer TEXT ENCODING DICT(32),\n  plane_issue_date DATE ENCODING DAYS(32),\n  plane_model TEXT ENCODING DICT(32),\n  plane_status TEXT ENCODING DICT(32),\n  plane_aircraft_type TEXT ENCODING DICT(32),\n  plane_engine_type TEXT ENCODING DICT(32),\n  plane_year SMALLINT,\n  origin_name TEXT ENCODING DICT(32),\n  origin_city TEXT ENCODING DICT(32),\n  origin_state TEXT ENCODING DICT(32),\n  origin_country TEXT ENCODING DICT(32),\n  origin_lat FLOAT,\n  origin_lon FLOAT,\n  dest_name TEXT ENCODING DICT(32),\n  dest_city TEXT ENCODING DICT(32),\n  dest_state TEXT ENCODING DICT(32),\n  dest_country TEXT ENCODING DICT(32),\n  dest_lat FLOAT,\n  dest_lon FLOAT,\n  origin_merc_x FLOAT,\n  origin_merc_y FLOAT,\n  dest_merc_x FLOAT,\n  dest_merc_y FLOAT)\nWITH (FRAGMENT_SIZE=2000000);",
    "CREATE TEMPORARY TABLE showcreatetabletest (\n  i INTEGER);"
  };
  // clang-format on

  for (size_t i = 0; i < creates.size(); ++i) {
    TQueryResult result;
    sql(creates[i]);
    sql(result, "SHOW CREATE TABLE showcreatetabletest;");
    EXPECT_EQ(creates[i], result.row_set.columns[0].data.str_col[0]);
    sql("DROP TABLE IF EXISTS showcreatetabletest;");
  }
}

TEST_F(ShowCreateTableTest, Defaults) {
  std::vector<std::string> creates = {
      "CREATE TABLE showcreatetabletest (i INTEGER) WITH (FRAGMENT_SIZE=" +
          std::to_string(DEFAULT_FRAGMENT_ROWS) + ");",
      "CREATE TABLE showcreatetabletest (i INTEGER) WITH (MAX_CHUNK_SIZE=" +
          std::to_string(DEFAULT_MAX_CHUNK_SIZE) + ");",
      "CREATE TABLE showcreatetabletest (i INTEGER) WITH (PAGE_SIZE=" +
          std::to_string(DEFAULT_PAGE_SIZE) + ");",
      "CREATE TABLE showcreatetabletest (i INTEGER) WITH (MAX_ROWS=" +
          std::to_string(DEFAULT_MAX_ROWS) + ");",
      "CREATE TABLE showcreatetabletest (i INTEGER) WITH (VACUUM='DELAYED');"};

  for (size_t i = 0; i < creates.size(); ++i) {
    sql(creates[i]);
    TQueryResult result;
    sql(result, "SHOW CREATE TABLE showcreatetabletest;");
    EXPECT_EQ("CREATE TABLE showcreatetabletest (\n  i INTEGER);",
              result.row_set.columns[0].data.str_col[0]);
    sql("DROP TABLE IF EXISTS showcreatetabletest;");
  }
}

TEST_F(ShowCreateTableTest, Other) {
  {
    sql("CREATE TABLE showcreatetabletest (i INTEGER);");
    std::string sqltext =
        "CREATE VIEW showcreateviewtest AS SELECT * FROM showcreatetabletest;";
    sql(sqltext);
    TQueryResult result;
    sql(result, "SHOW CREATE TABLE showcreateviewtest;");
    EXPECT_EQ(sqltext, result.row_set.columns[0].data.str_col[0]);
    sql("DROP VIEW IF EXISTS showcreateviewtest;");
    sql("DROP TABLE IF EXISTS showcreatetabletest;");
  }

  {
    sql("CREATE TABLE showcreatetabletest1 (\n  t TEXT ENCODING DICT(32));");
    std::string sqltext =
        "CREATE TABLE showcreatetabletest2 (\n  t TEXT,\n  SHARED DICTIONARY (t) "
        "REFERENCES showcreatetabletest1(t))\nWITH (SORT_COLUMN='t');";
    sql(sqltext);
    TQueryResult result;
    sql(result, "SHOW CREATE TABLE showcreatetabletest2;");
    EXPECT_EQ(sqltext, result.row_set.columns[0].data.str_col[0]);
    sql("DROP TABLE IF EXISTS showcreatetabletest1;");
    sql("DROP TABLE IF EXISTS showcreatetabletest2;");
  }
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
