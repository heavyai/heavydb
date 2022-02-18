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
#include "Shared/File.h"
#include "TestHelpers.h"
#include "boost/filesystem.hpp"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

class ShowUserSessionsTest : public DBHandlerTestFixture {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    // Default connection string outside of thrift
    connection_string = "tcp:";
    // Check that default only user session exists
    TQueryResult result;
    sql(result, "SHOW USER SESSIONS;");
    assertExpectedFormat(result);
    assertNumSessions(result, 1);
    assertSessionResultFound(result, "admin", "omnisci", 1);
    getID(result, "admin", "omnisci", admin_id);
  }

  static void SetUpTestSuite() {
    createDBHandler();
    users_ = {"user1", "user2"};
    superusers_ = {"super1", "super2"};
    dbs_ = {"db1", "db2"};
    createDBs();
    createUsers();
    createSuperUsers();
  }

  static void TearDownTestSuite() {
    dropUsers();
    dropSuperUsers();
    dropDBs();
  }

  void TearDown() override {
    // Check that default only user session still exists
    TQueryResult result;
    sql(result, "SHOW USER SESSIONS;");
    assertExpectedFormat(result);
    assertNumSessions(result, 1);
    assertSessionResultFound(result, "admin", "omnisci", admin_id);
    DBHandlerTestFixture::TearDown();
  }

  static void createUsers() {
    for (const auto& user : users_) {
      std::stringstream create;
      create << "CREATE USER " << user
             << " (password = 'HyperInteractive', is_super = 'false', "
                "default_db='omnisci');";
      sql(create.str());
      for (const auto& db : dbs_) {
        std::stringstream grant;
        grant << "GRANT ALL ON DATABASE  " << db << " to " << user << ";";
        sql(grant.str());
      }
    }
  }

  static void createSuperUsers() {
    for (const auto& user : superusers_) {
      std::stringstream create;
      create
          << "CREATE USER " << user
          << " (password = 'HyperInteractive', is_super = 'true', default_db='omnisci');";
      sql(create.str());
      for (const auto& db : dbs_) {
        std::stringstream grant;
        grant << "GRANT ALL ON DATABASE  " << db << " to " << user << ";";
        sql(grant.str());
      }
    }
  }

  static void dropUsers() {
    for (const auto& user : users_) {
      std::stringstream drop;
      drop << "DROP USER " << user << ";";
      sql(drop.str());
    }
  }

  static void dropSuperUsers() {
    for (const auto& user : superusers_) {
      std::stringstream drop;
      drop << "DROP USER " << user << ";";
      sql(drop.str());
    }
  }

  static void createDBs() {
    for (const auto& db : dbs_) {
      std::stringstream create;
      create << "CREATE DATABASE " << db << " (owner = 'admin');";
      sql(create.str());
    }
  }

  static void dropDBs() {
    for (const auto& db : dbs_) {
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
  std::vector<std::string> get_users() { return users_; }
  std::vector<std::string> get_superusers() { return superusers_; }
  std::vector<std::string> get_dbs() { return dbs_; }

 private:
  static std::vector<std::string> users_;
  static std::vector<std::string> superusers_;
  static std::vector<std::string> dbs_;

  std::string admin_id;
  std::string connection_string;
};

std::vector<std::string> ShowUserSessionsTest::users_;
std::vector<std::string> ShowUserSessionsTest::superusers_;
std::vector<std::string> ShowUserSessionsTest::dbs_;

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
  for (int copies = 1; copies < 4; copies++) {
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
        "SHOW USER SESSIONS failed, because it can only be executed by super "
        "user.",
        e.error_msg);
  }

  logout(usersession);
}

class ShowUserDetailsTest : public DBHandlerTestFixture {
 public:
  void SetUp() override { DBHandlerTestFixture::SetUp(); }

  static void SetUpTestSuite() {
    createDBHandler();
    users_ = {"user1", "user2"};
    superusers_ = {"super1", "super2"};
    dbs_ = {"omnisci"};
    createUsers();
    createSuperUsers();
  }

  static void TearDownTestSuite() {
    dropUsers();
    dropSuperUsers();
  }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  static void createUsers() {
    for (const auto& user : users_) {
      std::stringstream create;
      create << "CREATE USER " << user
             << " (password = 'HyperInteractive', is_super = 'false', "
                "default_db='omnisci');";
      sql(create.str());
      for (const auto& db : dbs_) {
        std::stringstream grant;
        grant << "GRANT ALL ON DATABASE  " << db << " to " << user << ";";
        sql(grant.str());
      }
    }
  }

  static void createSuperUsers() {
    for (const auto& user : superusers_) {
      std::stringstream create;
      create
          << "CREATE USER " << user
          << " (password = 'HyperInteractive', is_super = 'true', default_db='omnisci');";
      sql(create.str());
      for (const auto& db : dbs_) {
        std::stringstream grant;
        grant << "GRANT ALL ON DATABASE  " << db << " to " << user << ";";
        sql(grant.str());
      }
    }
  }

  static void dropUsers() {
    for (const auto& user : users_) {
      std::stringstream drop;
      drop << "DROP USER " << user << ";";
      sql(drop.str());
    }
  }

  static void dropSuperUsers() {
    for (const auto& user : superusers_) {
      std::stringstream drop;
      drop << "DROP USER " << user << ";";
      sql(drop.str());
    }
  }

  enum ColumnIndex { NAME, ID, IS_SUPER, DEFAULT_DB, CAN_LOGIN };

  void assertExpectedFormat(const TQueryResult& result) {
    ASSERT_EQ(result.row_set.is_columnar, true);
    ASSERT_EQ(result.row_set.columns.size(), size_t(5));
    ASSERT_EQ(result.row_set.row_desc[NAME].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[NAME].col_name, "NAME");
    ASSERT_EQ(result.row_set.row_desc[ID].col_type.type, TDatumType::BIGINT);
    ASSERT_EQ(result.row_set.row_desc[ID].col_name, "ID");
    ASSERT_EQ(result.row_set.row_desc[IS_SUPER].col_type.type, TDatumType::BOOL);
    ASSERT_EQ(result.row_set.row_desc[IS_SUPER].col_name, "IS_SUPER");
    ASSERT_EQ(result.row_set.row_desc[DEFAULT_DB].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[DEFAULT_DB].col_name, "DEFAULT_DB");
    ASSERT_EQ(result.row_set.row_desc[CAN_LOGIN].col_type.type, TDatumType::BOOL);
    ASSERT_EQ(result.row_set.row_desc[CAN_LOGIN].col_name, "CAN_LOGIN");
  }

  void assertUserResultFound(const TQueryResult& result, const std::string& username) {
    int num_matches = 0;
    for (size_t i = 0; i < result.row_set.columns[NAME].data.str_col.size(); ++i) {
      if (result.row_set.columns[NAME].data.str_col[i] == username) {
        num_matches++;
      }
    }
    ASSERT_EQ(num_matches, 1);
  }

  template <typename T>
  void assertUserResultFound(const TQueryResult& result,
                             const std::string& username,
                             ColumnIndex col,
                             T val) {
    using CLEANT = std::remove_reference_t<std::remove_cv_t<T>>;
    static_assert(std::is_integral_v<CLEANT> || std::is_floating_point_v<CLEANT> ||
                  std::is_same_v<std::string, CLEANT>);
    int num_matches = 0;
    for (size_t i = 0; i < result.row_set.columns[NAME].data.str_col.size(); ++i) {
      if (result.row_set.columns[NAME].data.str_col[i] == username) {
        num_matches++;
        // integral
        if constexpr (std::is_integral_v<CLEANT>) {
          ASSERT_EQ(result.row_set.columns[col].data.int_col[i], val);
        }
        // floating point
        else if constexpr (std::is_floating_point_v<CLEANT>) {
          ASSERT_EQ(result.row_set.columns[col].data.real_col[i], val);
        }
        // string
        else if constexpr (std::is_same_v<std::string, CLEANT>) {
          ASSERT_EQ(result.row_set.columns[col].data.str_col[i], val);
        }
      }
    }
    ASSERT_EQ(num_matches, 1);
  }

  void assertNumUsers(const TQueryResult& result, size_t num_users) {
    ASSERT_EQ(num_users, result.row_set.columns[NAME].data.str_col.size());
  }
  std::vector<std::string> get_users() { return users_; }
  std::vector<std::string> get_superusers() { return superusers_; }

 private:
  static inline std::vector<std::string> users_;
  static inline std::vector<std::string> superusers_;
  static inline std::vector<std::string> dbs_;

  std::string admin_id;
};

TEST_F(ShowUserDetailsTest, AllUsers) {
  TQueryResult result;
  sql(result, "SHOW USER DETAILS;");
  assertExpectedFormat(result);
  assertNumUsers(result, 5);
  assertUserResultFound(result, "admin");
  assertUserResultFound(result, "user1");
  assertUserResultFound(result, "user2");
  assertUserResultFound(result, "super1");
  assertUserResultFound(result, "super2");
}

TEST_F(ShowUserDetailsTest, OneUser) {
  TQueryResult result;
  sql(result, "SHOW USER DETAILS user1;");
  assertNumUsers(result, 1);
  assertUserResultFound(result, "user1");
}

TEST_F(ShowUserDetailsTest, MultipleUsers) {
  TQueryResult result;
  sql(result, "SHOW USER DETAILS user1,super1;");
  assertNumUsers(result, 2);
  assertUserResultFound(result, "user1");
  assertUserResultFound(result, "super1");
}

TEST_F(ShowUserDetailsTest, Columns) {
  using namespace std::string_literals;
  TQueryResult result;
  sql(result, "SHOW USER DETAILS user1;");
  assertNumUsers(result, 1);
  assertUserResultFound(result, "user1", IS_SUPER, false);
  assertUserResultFound(result, "user1", DEFAULT_DB, "omnisci(1)"s);
  assertUserResultFound(result, "user1", CAN_LOGIN, true);
}

class ShowTest : public DBHandlerTestFixture {
 public:
  static void dropUserIfExists(const std::string& user_name) {
    switchToAdmin();
    try {
      sql("DROP USER " + user_name + ";");
    } catch (const std::exception& e) {
      ASSERT_NE(std::string(e.what()).find("Cannot drop user. User " + user_name +
                                           " does not exist."),
                std::string::npos);
    }
  }
};

class ShowTableDdlTest : public ShowTest {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    switchToAdmin();
    sql("DROP TABLE IF EXISTS test_table;");
  }

  void TearDown() override {
    switchToAdmin();
    sql("DROP TABLE IF EXISTS test_table;");
    DBHandlerTestFixture::TearDown();
  }

  static void SetUpTestSuite() {
    createDBHandler();
    createTestUser();
  }

  static void TearDownTestSuite() { dropTestUser(); }

  static void createTestUser() {
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE omnisci TO test_user;");
  }

  static void dropTestUser() {
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

  static void createTestTable() { sql("CREATE TABLE test_table ( test_val int );"); }
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
  sql("DROP TABLE IF EXISTS test_table;");
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
  sql("DROP TABLE IF EXISTS test_table;");
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
  switchToAdmin();
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
  void SetUp() override { DBHandlerTestFixture::SetUp(); }

  void TearDown() override {
    switchToAdmin();
    sql("DROP DATABASE IF EXISTS test_db_1;");
    sql("DROP DATABASE IF EXISTS test_db_2;");
    DBHandlerTestFixture::TearDown();
  }

  static void SetUpTestSuite() {
    createDBHandler();
    createTestUser("test_user_1", "test_pass_1");
    createTestUser("test_user_2", "test_pass_2");
    createTestUser("test_super_user", "test_pass", true);
  }

  static void TearDownTestSuite() {
    dropTestUser("test_user_1");
    dropTestUser("test_user_2");
    dropTestUser("test_super_user");
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

  static void createTestUser(const std::string& user_name,
                             const std::string& pass,
                             const bool is_super_user = false) {
    sql("CREATE USER " + user_name + " (password = '" + pass + "', is_super = '" +
        (is_super_user ? "true" : "false") + "');");
  }

  static void dropTestUser(const std::string& user_name) {
    switchToAdmin();
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
  assertExpectedResult({"Database", "Owner"}, {{"omnisci", "admin"}}, result);
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
  assertExpectedResult(
      {"Database", "Owner"},
      {{"omnisci", "admin"}, {"test_db_1", "test_user_1"}, {"test_db_2", "test_user_2"}},
      result);
}

TEST_F(ShowDatabasesTest, SuperUserLoginAndOtherUserDatabases) {
  sql("CREATE DATABASE test_db_1 (owner = 'test_user_1');");
  sql("CREATE DATABASE test_db_2 (owner = 'test_user_2');");
  login("test_super_user", "test_pass");

  TQueryResult result;
  sql(result, "SHOW DATABASES;");
  assertExpectedResult(
      {"Database", "Owner"},
      {{"omnisci", "admin"}, {"test_db_1", "test_user_1"}, {"test_db_2", "test_user_2"}},
      result);
}

class ShowCreateTableTest : public DBHandlerTestFixture {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    switchToAdmin();
    sql("DROP TABLE IF EXISTS showcreatetabletest;");
    sql("DROP TABLE IF EXISTS showcreatetabletest1;");
    sql("DROP TABLE IF EXISTS showcreatetabletest2;");
    sql("DROP VIEW IF EXISTS showcreateviewtest;");
  }

  void TearDown() override {
    switchToAdmin();
    sql("DROP TABLE IF EXISTS showcreatetabletest;");
    sql("DROP TABLE IF EXISTS showcreatetabletest1;");
    sql("DROP TABLE IF EXISTS showcreatetabletest2;");
    sql("DROP VIEW IF EXISTS showcreateviewtest;");
    DBHandlerTestFixture::TearDown();
  }

  std::string getTestFilePath() {
    return boost::filesystem::canonical("../../Tests/FsiDataFiles/example_1.csv")
        .string();
  }
};

TEST_F(ShowCreateTableTest, Identity) {
  // clang-format off
  std::vector<std::string> creates = {
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (FRAGMENT_SIZE=123);",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (MAX_CHUNK_SIZE=123);",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (PAGE_SIZE=123);",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (MAX_ROWS=123);",
    "CREATE TABLE showcreatetabletest (\n  i INTEGER)\nWITH (SORT_COLUMN='i');",
    "CREATE TABLE showcreatetabletest (\n  id TEXT ENCODING DICT(32),\n  abbr TEXT ENCODING DICT(32),\n  name TEXT ENCODING DICT(32));",
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
      "CREATE TABLE showcreatetabletest (i INTEGER);"};

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

TEST_F(ShowCreateTableTest, SharedComplex) {
  {
    sql("DROP TABLE IF EXISTS showcreatetabletest1;");
    sql("DROP TABLE IF EXISTS renamedcreatetabletest1;");
    sql("DROP TABLE IF EXISTS showcreatetabletest2;");
    sql("DROP TABLE IF EXISTS showcreatetabletest3;");

    sql("CREATE TABLE showcreatetabletest1 (\n  t1 TEXT ENCODING DICT(16));");
    std::string sqltext =
        "CREATE TABLE showcreatetabletest2 (\n  t2 TEXT,\n  SHARED DICTIONARY (t2) "
        "REFERENCES showcreatetabletest1(t1));";
    sql(sqltext);
    {
      TQueryResult result;
      sql(result, "SHOW CREATE TABLE showcreatetabletest2;");
      EXPECT_EQ(sqltext, result.row_set.columns[0].data.str_col[0]);
    }
    sql("CREATE TABLE showcreatetabletest3 (\n  t3 TEXT,\n SHARED DICTIONARY (t3) "
        "REFERENCES showcreatetabletest2(t2));");

    sql("ALTER TABLE showcreatetabletest1 RENAME TO renamedcreatetabletest1;");

    {
      TQueryResult result;
      sql(result, "SHOW CREATE TABLE showcreatetabletest3;");
      EXPECT_EQ(
          "CREATE TABLE showcreatetabletest3 (\n  t3 TEXT,\n  SHARED DICTIONARY (t3) "
          "REFERENCES renamedcreatetabletest1(t1));",
          result.row_set.columns[0].data.str_col[0]);
    }
    sql("DROP TABLE IF EXISTS renamedcreatetabletest1;");

    {
      TQueryResult result;
      sql(result, "SHOW CREATE TABLE showcreatetabletest2;");
      EXPECT_EQ("CREATE TABLE showcreatetabletest2 (\n  t2 TEXT ENCODING DICT(16));",
                result.row_set.columns[0].data.str_col[0]);
    }
    {
      TQueryResult result;
      sql(result, "SHOW CREATE TABLE showcreatetabletest3;");
      EXPECT_EQ(
          "CREATE TABLE showcreatetabletest3 (\n  t3 TEXT,\n  SHARED DICTIONARY (t3) "
          "REFERENCES showcreatetabletest2(t2));",
          result.row_set.columns[0].data.str_col[0]);
    }
    sql("DROP TABLE IF EXISTS showcreatetabletest2;");
    {
      TQueryResult result;
      sql(result, "SHOW CREATE TABLE showcreatetabletest3;");
      EXPECT_EQ("CREATE TABLE showcreatetabletest3 (\n  t3 TEXT ENCODING DICT(16));",
                result.row_set.columns[0].data.str_col[0]);
    }
    sql("DROP TABLE IF EXISTS showcreatetabletest3;");
  }
}

TEST_F(ShowCreateTableTest, TextArray) {
  sql("CREATE TABLE showcreatetabletest (t1 TEXT[], t2 TEXT[5]);");
  sqlAndCompareResult("SHOW CREATE TABLE showcreatetabletest;",
                      {{"CREATE TABLE showcreatetabletest (\n  t1 TEXT[] ENCODING "
                        "DICT(32),\n  t2 TEXT[5] ENCODING DICT(32));"}});
}

TEST_F(ShowCreateTableTest, TimestampArray) {
  sql("CREATE TABLE showcreatetabletest (tp TIMESTAMP, tpe TIMESTAMP ENCODING "
      "FIXED(32), "
      "tp1 TIMESTAMP(3), tp2 "
      "TIMESTAMP(6)[], tp3 TIMESTAMP(9)[2]);");
  sqlAndCompareResult("SHOW CREATE TABLE showcreatetabletest;",
                      {{"CREATE TABLE showcreatetabletest (\n  tp TIMESTAMP(0),\n  tpe "
                        "TIMESTAMP(0) ENCODING FIXED(32),\n  tp1 TIMESTAMP(3),\n  "
                        "tp2 TIMESTAMP(6)[],\n  tp3 TIMESTAMP(9)[2]);"}});
}

TEST_F(ShowCreateTableTest, TimestampEncoding) {
  // Timestamp encoding accepts a shorthand syntax (see above). Ensure the output of the
  // SHOW CREATE TABLE command using the short hand syntax can be passed back in as
  // input.
  sql("CREATE TABLE showcreatetabletest (tp TIMESTAMP(0), tpe TIMESTAMP(0) ENCODING "
      "FIXED(32));");
  sqlAndCompareResult("SHOW CREATE TABLE showcreatetabletest;",
                      {{"CREATE TABLE showcreatetabletest (\n  tp TIMESTAMP(0),\n  tpe "
                        "TIMESTAMP(0) ENCODING FIXED(32));"}});
}

TEST_F(ShowCreateTableTest, NotCaseSensitive) {
  sql("CREATE TABLE showcreatetabletest(c1 int);");

  sqlAndCompareResult("SHOW CREATE TABLE sHoWcReAtEtAbLeTeSt;",
                      {{"CREATE TABLE showcreatetabletest (\n  c1 INTEGER);"}});
}

TEST_F(ShowCreateTableTest, TableWithUncappedEpoch) {
  sql("CREATE TABLE showcreatetabletest (c1 INTEGER);");
  getCatalog().setUncappedTableEpoch("showcreatetabletest");
  sqlAndCompareResult("SHOW CREATE TABLE showcreatetabletest;",
                      {{"CREATE TABLE showcreatetabletest (\n  c1 INTEGER);"}});
}

TEST_F(ShowCreateTableTest, TableWithMaxRollbackEpochs) {
  sql("CREATE TABLE showcreatetabletest (c1 INTEGER) WITH (MAX_ROLLBACK_EPOCHS = 10);");
  sqlAndCompareResult("SHOW CREATE TABLE showcreatetabletest;",
                      {{"CREATE TABLE showcreatetabletest (\n  c1 INTEGER)\nWITH "
                        "(MAX_ROLLBACK_EPOCHS=10);"}});
}

TEST_F(ShowCreateTableTest, DefaultColumnValues) {
  sql("CREATE TABLE showcreatetabletest (idx INTEGER NOT NULL, i INTEGER DEFAULT 14,"
      "big_i BIGINT DEFAULT 314958734, null_i INTEGER, int_a INTEGER[] "
      "DEFAULT ARRAY[1, 2, 3], text_a TEXT[] DEFAULT ARRAY['a', 'b'] ENCODING DICT(32),"
      "dt TEXT DEFAULT 'World' ENCODING DICT(32), "
      "d DATE DEFAULT '2011-10-23' ENCODING DAYS(32), "
      "ta TIMESTAMP[] DEFAULT ARRAY['2011-10-23 07:15:01', '2012-09-17 11:59:11'], "
      "f FLOAT DEFAULT 1.15, n DECIMAL(3,2) DEFAULT 1.25 ENCODING FIXED(16));");
  sqlAndCompareResult(
      "SHOW CREATE TABLE showcreatetabletest;",
      {{"CREATE TABLE showcreatetabletest (\n  idx INTEGER NOT NULL,\n  i INTEGER "
        "DEFAULT 14,\n  big_i "
        "BIGINT DEFAULT 314958734,\n  null_i INTEGER,\n  int_a INTEGER[] DEFAULT "
        "ARRAY[1, 2, 3],\n  text_a TEXT[] DEFAULT ARRAY['a', 'b'] ENCODING DICT(32),\n  "
        "dt TEXT DEFAULT 'World' ENCODING DICT(32),\n  "
        "d DATE DEFAULT '2011-10-23' ENCODING DAYS(32),\n  ta "
        "TIMESTAMP(0)[] DEFAULT ARRAY['2011-10-23 07:15:01', '2012-09-17 11:59:11'],\n  "
        "f "
        "FLOAT DEFAULT 1.15,\n  n DECIMAL(3,2) DEFAULT 1.25 ENCODING FIXED(16));"}});
}

namespace {
const int64_t PAGES_PER_DATA_FILE =
    File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_DATA_FILE;
const int64_t PAGES_PER_METADATA_FILE =
    File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE;
const int64_t DEFAULT_DATA_FILE_SIZE{DEFAULT_PAGE_SIZE * PAGES_PER_DATA_FILE};
const int64_t DEFAULT_METADATA_FILE_SIZE{METADATA_PAGE_SIZE * PAGES_PER_METADATA_FILE};
}  // namespace

class ShowDiskCacheUsageTest : public ShowTest {
 public:
  static inline constexpr int64_t epoch_file_size{2 * sizeof(int64_t)};
  static inline constexpr int64_t empty_mgr_size{0};
  static inline constexpr int64_t chunk_size{DEFAULT_PAGE_SIZE + METADATA_PAGE_SIZE};
  // TODO(Misiu): These can be made constexpr once c++20 is supported.
  static inline std::string cache_path_ = to_string(BASE_PATH) + "/omnisci_disk_cache";
  static inline std::string table1{"table1"};

  static void SetUpTestSuite() {
    DBHandlerTestFixture::SetUpTestSuite();
    loginAdmin();
    sql("DROP DATABASE IF EXISTS test_db;");
    sql("CREATE DATABASE test_db;");
    login("admin", "HyperInteractive", "test_db");
    getCatalog().getDataMgr().getPersistentStorageMgr()->getDiskCache()->clear();
  }

  static void TearDownTestSuite() {
    sql("DROP DATABASE IF EXISTS test_db;");
    dropUserIfExists("test_user");
    DBHandlerTestFixture::TearDownTestSuite();
  }

  void SetUp() override {
    if (isDistributedMode()) {
      GTEST_SKIP() << "Test not supported in distributed mode.";
    }
    DBHandlerTestFixture::SetUp();
    login("admin", "HyperInteractive", "test_db");
    sql("DROP TABLE IF EXISTS " + table1 + ";");
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS " + table1 + ";");
    DBHandlerTestFixture::TearDown();
  }

  uint64_t getWrapperSizeForTable(const std::string& table_name) {
    uint64_t space_used = 0;
    auto& cat = getCatalog();
    auto td = cat.getMetadataForTable(table_name, false);
    std::string table_dir =
        cache_path_ + "/" +
        File_Namespace::get_dir_name_for_table(cat.getDatabaseId(), td->tableId);
    if (boost::filesystem::exists(table_dir)) {
      for (const auto& file :
           boost::filesystem::recursive_directory_iterator(table_dir)) {
        if (boost::filesystem::is_regular_file(file.path())) {
          space_used += boost::filesystem::file_size(file.path());
        }
      }
    }
    return space_used;
  }

  uint64_t getMinSizeForTable(std::string& table_name) {
    return chunk_size + getWrapperSizeForTable(table_name);
  }
};

TEST_F(ShowDiskCacheUsageTest, NoTables) {
  sqlAndCompareResult("SHOW DISK CACHE USAGE;", {});
}

class ShowDiskCacheUsageForNormalTableTest : public ShowDiskCacheUsageTest {
 public:
  static void SetUpTestSuite() {
    ShowDiskCacheUsageTest::SetUpTestSuite();
    resetPersistentStorageMgr(File_Namespace::DiskCacheLevel::all);
  }

  static void TearDownTestSuite() {
    resetPersistentStorageMgr(File_Namespace::DiskCacheLevel::fsi);
    ShowDiskCacheUsageTest::TearDownTestSuite();
  }

  static void resetPersistentStorageMgr(File_Namespace::DiskCacheLevel cache_level) {
    for (auto table_it : getCatalog().getAllTableMetadata()) {
      getCatalog().removeFragmenterForTable(table_it->tableId);
    }
    getCatalog().getDataMgr().resetPersistentStorage(
        {cache_path_, cache_level}, 0, getSystemParameters());
  }
};

TEST_F(ShowDiskCacheUsageForNormalTableTest, NormalTableEmptyUninitialized) {
  sql("CREATE TABLE " + table1 + " (s TEXT);");

  sqlAndCompareResult("SHOW DISK CACHE USAGE;", {{table1, empty_mgr_size}});
}

// If a table is initialized, but empty (it has a fileMgr, but no content), it will have
// created an epoch file, so it returns the size of that file only.  This is different
// from the case where no manager is found which returns 0.
TEST_F(ShowDiskCacheUsageForNormalTableTest, NormalTableEmptyInitialized) {
  sql("CREATE TABLE " + table1 + " (s TEXT);");

  sql("SELECT * FROM " + table1 + ";");

  sqlAndCompareResult("SHOW DISK CACHE USAGE;", {{table1, empty_mgr_size}});
}

TEST_F(ShowDiskCacheUsageForNormalTableTest, NormalTableMinimum) {
  sql("CREATE TABLE " + table1 + " (s TEXT);");

  sql("INSERT INTO " + table1 + " VALUES('1');");

  sqlAndCompareResult("SHOW DISK CACHE USAGE;",
                      {{table1, i(chunk_size + getWrapperSizeForTable(table1))}});
}

class ShowTableDetailsTest : public ShowTest,
                             public testing::WithParamInterface<int32_t> {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    login("admin", "HyperInteractive", "test_db");
    dropTestTables();
  }

  void TearDown() override {
    login("admin", "HyperInteractive", "test_db");
    dropTestTables();
    DBHandlerTestFixture::TearDown();
  }

  static void SetUpTestSuite() {
    DBHandlerTestFixture::SetUpTestSuite();
    switchToAdmin();
    sql("DROP DATABASE IF EXISTS test_db;");
    sql("CREATE DATABASE test_db;");
    createTestUser();
  }

  static void TearDownTestSuite() {
    switchToAdmin();
    dropTestUser();
    sql("DROP DATABASE IF EXISTS test_db;");
    DBHandlerTestFixture::TearDownTestSuite();
  }

  static void createTestUser() {
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE test_db TO test_user;");
  }

  static void dropTestUser() { dropUserIfExists("test_user"); }

  void loginTestUser() { login("test_user", "test_pass", "test_db"); }

  void dropTestTables() {
    sql("DROP TABLE IF EXISTS test_table_1;");
    sql("DROP TABLE IF EXISTS test_table_2;");
    sql("DROP TABLE IF EXISTS test_table_3;");
    sql("DROP TABLE IF EXISTS test_table_4;");
    sql("DROP TABLE IF EXISTS test_temp_table;");
    sql("DROP TABLE IF EXISTS test_arrow_table;");
    sql("DROP VIEW IF EXISTS test_view;");
  }

  void assertExpectedHeaders(const TQueryResult& result) {
    std::vector<std::string> headers{"table_id",
                                     "table_name",
                                     "column_count",
                                     "is_sharded_table",
                                     "shard_count",
                                     "max_rows",
                                     "fragment_size",
                                     "max_rollback_epochs",
                                     "min_epoch",
                                     "max_epoch",
                                     "min_epoch_floor",
                                     "max_epoch_floor",
                                     "metadata_file_count",
                                     "total_metadata_file_size",
                                     "total_metadata_page_count",
                                     "total_free_metadata_page_count",
                                     "data_file_count",
                                     "total_data_file_size",
                                     "total_data_page_count",
                                     "total_free_data_page_count"};
    if (isDistributedMode()) {
      headers.insert(headers.begin(), "leaf_index");
    }
    for (size_t i = 0; i < headers.size(); i++) {
      EXPECT_EQ(headers[i], result.row_set.row_desc[i].col_name);
    }
  }

  void assertMaxRollbackUpdateResult(int max_rollback_epochs,
                                     int used_metadata_pages,
                                     int used_data_pages,
                                     int epoch,
                                     int epoch_floor) {
    TQueryResult result;
    sql(result, "show table details;");
    assertExpectedHeaders(result);

    // clang-format off
    if (isDistributedMode()) {
      assertResultSetEqual({{i(0), i(1), "test_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                             i(DEFAULT_FRAGMENT_ROWS), i(max_rollback_epochs), i(epoch), i(epoch),
                             i(epoch_floor), i(epoch_floor), i(1), i(DEFAULT_METADATA_FILE_SIZE),
                             i(PAGES_PER_METADATA_FILE), i(PAGES_PER_METADATA_FILE - used_metadata_pages),
                             i(1), i(DEFAULT_DATA_FILE_SIZE), i(PAGES_PER_DATA_FILE),
                             i(PAGES_PER_DATA_FILE - used_data_pages)},
                            {i(1), i(1), "test_table_1", i(4), False, i(0), i(DEFAULT_MAX_ROWS),
                             i(DEFAULT_FRAGMENT_ROWS), i(max_rollback_epochs), i(epoch), i(epoch),
                             i(epoch_floor), i(epoch_floor), i(1), i(DEFAULT_METADATA_FILE_SIZE),
                             i(PAGES_PER_METADATA_FILE), i(PAGES_PER_METADATA_FILE - used_metadata_pages),
                             i(1), i(DEFAULT_DATA_FILE_SIZE), i(PAGES_PER_DATA_FILE),
                             i(PAGES_PER_DATA_FILE - used_data_pages)}},
                           result);
    } else {
      assertResultSetEqual({{i(1), "test_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                             i(DEFAULT_FRAGMENT_ROWS), i(max_rollback_epochs), i(epoch), i(epoch),
                             i(epoch_floor), i(epoch_floor), i(1), i(DEFAULT_METADATA_FILE_SIZE),
                             i(PAGES_PER_METADATA_FILE), i(PAGES_PER_METADATA_FILE - used_metadata_pages),
                             i(1), i(DEFAULT_DATA_FILE_SIZE), i(PAGES_PER_DATA_FILE),
                             i(PAGES_PER_DATA_FILE - used_data_pages)}},
                           result);
    }
    // clang-format on
  }

  void assertTablesWithContentResult(const TQueryResult result, int64_t data_page_size) {
    int64_t data_file_size;
    if (data_page_size == -1) {
      data_file_size = DEFAULT_PAGE_SIZE * PAGES_PER_DATA_FILE;
    } else {
      data_file_size = data_page_size * PAGES_PER_DATA_FILE;
    }

    // clang-format off
    assertResultSetEqual({{i(1), "test_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(1), i(1),
                           i(0), i(0), i(1), i(DEFAULT_METADATA_FILE_SIZE),
                           i(PAGES_PER_METADATA_FILE), i(PAGES_PER_METADATA_FILE - 2), i(1),
                           i(data_file_size), i(PAGES_PER_DATA_FILE), i(PAGES_PER_DATA_FILE - 2)},
                          {i(2), "test_table_3", i(2), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(1), i(1),
                           i(0), i(0), i(1), i(DEFAULT_METADATA_FILE_SIZE),
                           i(PAGES_PER_METADATA_FILE), i(PAGES_PER_METADATA_FILE - 1), i(1),
                           i(data_file_size), i(PAGES_PER_DATA_FILE), i(PAGES_PER_DATA_FILE - 1)}},
                         result);
    // clang-format on
  }

  // In the case where table page size is set to METADATA_PAGE_SIZE, both
  // the data and metadata content are stored in the data files
  void assertTablesWithContentAndSamePageSizeResult(const TQueryResult result) {
    int64_t data_file_size{METADATA_PAGE_SIZE * PAGES_PER_DATA_FILE};
    // clang-format off
    assertResultSetEqual({{i(1), "test_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(1), i(1),
                           i(0), i(0), i(0), i(0), i(0), i(0), i(1), i(data_file_size),
                           i(PAGES_PER_DATA_FILE), i(PAGES_PER_DATA_FILE - 4)},
                          {i(2), "test_table_3", i(2), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(1), i(1),
                           i(0), i(0), i(0), i(0), i(0), i(0), i(1), i(data_file_size),
                           i(PAGES_PER_DATA_FILE), i(PAGES_PER_DATA_FILE - 2)}},
                         result);
    // clang-format on
  }

  std::string getWithPageSize() {
    std::string with_page_size;
    auto page_size = GetParam();
    if (page_size != -1) {
      with_page_size = " with (page_size = " + std::to_string(page_size) + ")";
    }
    return with_page_size;
  }
};

TEST_F(ShowTableDetailsTest, EmptyTables) {
  sql("create table test_table_1 (c1 int, c2 text);");
  sql("create table test_table_3 (c1 int) with (fragment_size= 5);");

  TQueryResult result;
  sql(result, "show table details;");
  assertExpectedHeaders(result);

  // clang-format off
  assertResultSetEqual({{i(1), "test_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                         i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                         i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                         i(0)},
                        {i(2), "test_table_3", i(2), False, i(0), i(DEFAULT_MAX_ROWS),
                         i(5), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0), i(0), i(0), i(0),
                         i(0), i(0), i(0), i(0), i(0), i(0), i(0)}},
                       result);
  // clang-format on
}

TEST_F(ShowTableDetailsTest, NotCaseSensitive) {
  sql("create table TEST_table_1 (c1 int, c2 text);");

  TQueryResult result;
  sql(result, "show table details test_TABLE_1;");
  assertExpectedHeaders(result);

  // clang-format off
  if (isDistributedMode()) {
    assertResultSetEqual({{i(0), i(1), "TEST_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                           i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                           i(0)},
                          {i(1), i(1), "TEST_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                           i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                           i(0)}},
                         result);
  }
  else {
    assertResultSetEqual({{i(1), "TEST_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                           i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                           i(0)}},
                         result);
  }
  // clang-format on
}

TEST_P(ShowTableDetailsTest, TablesWithContent) {
  sql("create table test_table_1 (c1 int, c2 text) " + getWithPageSize() + ";");

  // Inserts for non-sharded tables are non-deterministic in distributed mode
  if (!isDistributedMode()) {
    sql("insert into test_table_1 values (10, 'abc');");
  }

  sql("create table test_table_3 (c1 int) " + getWithPageSize() + ";");
  sql("insert into test_table_3 values (50);");

  TQueryResult result;
  sql(result, "show table details;");
  assertExpectedHeaders(result);

  auto page_size = GetParam();
  if (page_size == METADATA_PAGE_SIZE) {
    assertTablesWithContentAndSamePageSizeResult(result);
  } else {
    assertTablesWithContentResult(result, page_size);
  }
}

INSTANTIATE_TEST_SUITE_P(
    DifferentPageSizes,
    ShowTableDetailsTest,
    testing::Values(-1 /* Use default */,
                    100 /* Arbitrary page size */,
                    METADATA_PAGE_SIZE,
                    65536 /* Results in the same file size as the metadata file */),
    [](const auto& param_info) {
      auto page_size = param_info.param;
      return "Page_Size_" + (page_size == -1 ? "Default" : std::to_string(page_size));
    });

TEST_F(ShowTableDetailsTest, CommandWithTableNames) {
  sql("create table test_table_1 (c1 int, c2 text);");
  sql("create table test_table_3 (c1 int);");

  TQueryResult result;
  sql(result, "show table details test_table_1, test_table_3;");
  assertExpectedHeaders(result);

  // clang-format off
  assertResultSetEqual({{i(1), "test_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                         i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                         i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                         i(0)},
                        {i(2), "test_table_3", i(2), False, i(0), i(DEFAULT_MAX_ROWS),
                         i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                         i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                         i(0)}},
                       result);
  // clang-format on
}

TEST_F(ShowTableDetailsTest, UserSpecificTables) {
  sql("create table test_table_1 (c1 int, c2 text);");
  sql("create table test_table_3 (c1 int);");
  sql("GRANT SELECT ON TABLE test_table_3 TO test_user;");

  loginTestUser();

  TQueryResult result;
  sql(result, "show table details;");
  assertExpectedHeaders(result);

  // clang-format off
  assertResultSetEqual({{i(2), "test_table_3", i(2), False, i(0), i(DEFAULT_MAX_ROWS),
                         i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                         i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                         i(0)}},
                       result);
  // clang-format on
}

TEST_F(ShowTableDetailsTest, InaccessibleTable) {
  sql("create table test_table_1 (c1 int, c2 text);");
  sql("create table test_table_3 (c1 int);");

  loginTestUser();
  queryAndAssertException("show table details test_table_1;",
                          "Unable to show table details for table: "
                          "test_table_1. Table does not exist.");
}

TEST_F(ShowTableDetailsTest, NonExistentTable) {
  sql("create table test_table_1 (c1 int, c2 text);");
  sql("create table test_table_3 (c1 int);");

  queryAndAssertException("show table details test_table_4;",
                          "Unable to show table details for table: "
                          "test_table_4. Table does not exist.");
}

TEST_F(ShowTableDetailsTest, UnsupportedTableTypes) {
  sql("create table test_table_1 (c1 int, c2 text);");
  sql("create temporary table test_temp_table (c1 int, c2 text);");
  sql("create dataframe test_arrow_table (c1 int) from 'CSV:" +
      boost::filesystem::canonical("../../Tests/FsiDataFiles/0.csv").string() + "';");
  sql("create view test_view as select * from test_table_1;");

  TQueryResult result;
  sql(result, "show table details;");
  assertExpectedHeaders(result);

  // clang-format off
  if (isDistributedMode()) {
    assertResultSetEqual({{i(0), i(1), "test_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                           i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                           i(0)},
                          {i(1), i(1), "test_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                           i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                           i(0)}},
                         result);
  } else {
    assertResultSetEqual({{i(1), "test_table_1", i(3), False, i(0), i(DEFAULT_MAX_ROWS),
                           i(DEFAULT_FRAGMENT_ROWS), i(DEFAULT_MAX_ROLLBACK_EPOCHS), i(0), i(0),
                           i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0), i(0),
                           i(0)}},
                         result);
  }
  // clang-format on
}

TEST_F(ShowTableDetailsTest, TemporaryTableSpecified) {
  sql("create temporary table test_temp_table (c1 int, c2 text);");

  queryAndAssertException("show table details test_temp_table;",
                          "SHOW TABLE DETAILS is not supported for temporary "
                          "tables. Table name: test_temp_table.");
}

TEST_F(ShowTableDetailsTest, ArrowFsiTableSpecified) {
  sql("create dataframe test_arrow_table (c1 int) from 'CSV:" +
      boost::filesystem::canonical("../../Tests/FsiDataFiles/0.csv").string() + "';");

  queryAndAssertException("show table details test_arrow_table;",
                          "SHOW TABLE DETAILS is not supported for temporary "
                          "tables. Table name: test_arrow_table.");
}

TEST_F(ShowTableDetailsTest, ViewSpecified) {
  sql("create table test_table_1 (c1 int, c2 text);");
  sql("create view test_view as select * from test_table_1;");

  queryAndAssertException("show table details test_view;",
                          "Unable to show table details for table: "
                          "test_view. Table does not exist.");
}

class ShowQueriesTest : public ShowTest {
 public:
  static void SetUpTestSuite() {
    createDBHandler();
    createTestUser();
    loginAdmin();
  }

  static void TearDownTestSuite() {
    switchToAdmin();
    sql("DROP TABLE IF EXISTS test_table;");
    dropTestUser();
  }

  void SetUp() override { DBHandlerTestFixture::SetUp(); }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  static void createTestUser() {
    dropUserIfExists("u1");
    dropUserIfExists("u2");
    sql("CREATE USER u1 (password = 'u1');");
    sql("GRANT ALL ON DATABASE omnisci TO u1;");
    sql("CREATE USER u2 (password = 'u2');");
    sql("GRANT ALL ON DATABASE omnisci TO u2;");
  }

  static void dropTestUser() {
    try {
      sql("DROP USER u1;");
      sql("DROP USER u2;");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }
};

TEST_F(ShowQueriesTest, NonAdminUser) {
  TQueryResult non_admin_res, admin_res, own_res, res1;
  TSessionId query_session;
  TSessionId show_queries_cmd_session;

  login("u1", "u1", "omnisci", query_session);
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, &getCatalog().getDataMgr());
  // mock the running query by just enrolling the meaningless query
  executor->enrollQuerySession(
      query_session, "MOCK_QUERY", "0", 0, QuerySessionStatus::RUNNING_QUERY_KERNEL);

  auto show_queries_thread1 = std::async(std::launch::deferred, [&] {
    login("u2", "u2", "omnisci", show_queries_cmd_session);
    sql(non_admin_res, "SHOW QUERIES;", show_queries_cmd_session);
  });

  auto show_queries_thread2 = std::async(std::launch::deferred, [&] {
    switchToAdmin();
    auto p = getDbHandlerAndSessionId();
    sql(admin_res, "SHOW QUERIES;", p.second);
  });

  auto show_queries_thread3 = std::async(
      std::launch::deferred, [&] { sql(own_res, "SHOW QUERIES;", query_session); });

  show_queries_thread1.get();
  show_queries_thread2.get();
  show_queries_thread3.get();

  EXPECT_TRUE(query_session.compare(show_queries_cmd_session) != 0);
  // non-admin && non-own session cannot see the query status
  EXPECT_TRUE(non_admin_res.row_set.columns[0].data.str_col.empty());
  // admin && own user can see the query status
  EXPECT_TRUE(!admin_res.row_set.columns[0].data.str_col.empty());
  EXPECT_TRUE(!own_res.row_set.columns[0].data.str_col.empty());
  {
    mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor->getSessionLock());
    executor->removeFromQuerySessionList(query_session, "0", session_write_lock);
  }
}

int main(int argc, char** argv) {
  g_enable_fsi = true;
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  DBHandlerTestFixture::initTestArgs(argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  g_enable_fsi = false;
  return err;
}
