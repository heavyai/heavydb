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
 * @file ForeignServerDdlTest.cpp
 * @brief Test suite for foreign server DDL commands
 */

#include <gtest/gtest.h>

#include "DBHandlerTestHelpers.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

class CreateForeignServerTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUp();
    getCatalog().dropForeignServer("test_server");
  }

  void TearDown() override {
    getCatalog().dropForeignServer("test_server");
    DBHandlerTestFixture::TearDown();
  }

  void assertExpectedForeignServer() {
    auto& catalog = getCatalog();
    auto foreign_server = catalog.getForeignServerFromStorage("test_server");

    ASSERT_GT(foreign_server->id, 0);
    ASSERT_EQ("test_server", foreign_server->name);
    ASSERT_EQ(foreign_storage::DataWrapperType::CSV, foreign_server->data_wrapper_type);
    ASSERT_EQ(OMNISCI_ROOT_USER_ID, foreign_server->user_id);

    ASSERT_TRUE(
        foreign_server->options.find(foreign_storage::ForeignServer::STORAGE_TYPE_KEY) !=
        foreign_server->options.end());
    ASSERT_EQ(
        foreign_storage::ForeignServer::LOCAL_FILE_STORAGE_TYPE,
        foreign_server->options.find(foreign_storage::ForeignServer::STORAGE_TYPE_KEY)
            ->second);

    ASSERT_TRUE(
        foreign_server->options.find(foreign_storage::ForeignServer::BASE_PATH_KEY) !=
        foreign_server->options.end());
    ASSERT_EQ("/test_path/",
              foreign_server->options.find(foreign_storage::ForeignServer::BASE_PATH_KEY)
                  ->second);
  }
};

TEST_F(CreateForeignServerTest, AllValidParameters) {
  sql("CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "
      "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  assertExpectedForeignServer();
}

TEST_F(CreateForeignServerTest, AllValidParametersAndReadFromCache) {
  sql("CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "
      "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  assertExpectedForeignServer();
}

TEST_F(CreateForeignServerTest, ExistingServerWithIfNotExists) {
  sql("CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "
      "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  sql("CREATE SERVER IF NOT EXISTS test_server FOREIGN DATA WRAPPER omnisci_csv "
      "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  assertExpectedForeignServer();
}

TEST_F(CreateForeignServerTest, ExistingServerWithoutIfNotExists) {
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  sql(query);
  queryAndAssertException(
      query, "Exception: A foreign server with name \"test_server\" already exists.");
}

TEST_F(CreateForeignServerTest, OmniSciPrefix) {
  std::string query{
      "CREATE SERVER omnisci_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  queryAndAssertException(query,
                          "Exception: Server names cannot start with \"omnisci\".");
}

TEST_F(CreateForeignServerTest, MissingStorageType) {
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(base_path = '/test_path/');"};
  queryAndAssertException(
      query, "Exception: Foreign server options must contain \"STORAGE_TYPE\".");
}

TEST_F(CreateForeignServerTest, MissingBasePath) {
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE');"};
  sql(query);
}

TEST_F(CreateForeignServerTest, InvalidOption) {
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(invalid_key = 'value', storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  std::string error_message{
      "Exception: Invalid option \"INVALID_KEY\". "
      "Option must be one of the following: STORAGE_TYPE, BASE_PATH."};
  queryAndAssertException(query, error_message);
}

TEST_F(CreateForeignServerTest, InvalidStorageType) {
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'INVALID_TYPE', base_path = '/test_path/');"};
  std::string error_message{
      "Exception: Invalid storage type value. Value must be one of the following: "
      "LOCAL_FILE."};
  queryAndAssertException(query, error_message);
}

TEST_F(CreateForeignServerTest, FsiDisabled) {
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  g_enable_fsi = false;
  queryAndAssertException(query, "Exception: Syntax error at: SERVER");
}

TEST_F(CreateForeignServerTest, InvalidDataWrapper) {
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER invalid_wrapper WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  std::string error_message{
      "Exception: Invalid data wrapper type \"INVALID_WRAPPER\". "
      "Data wrapper type must be one of the following: OMNISCI_CSV, OMNISCI_PARQUET."};
  queryAndAssertException(query, error_message);
}

class DropForeignServerTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUp();
    dropTestTable();
    sql("CREATE SERVER IF NOT EXISTS test_server FOREIGN DATA WRAPPER omnisci_csv "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  }

  void TearDown() override {
    g_enable_fsi = true;
    dropTestTable();
    getCatalog().dropForeignServer("test_server");
    getCatalog().dropForeignServer("test_server_1");
    DBHandlerTestFixture::TearDown();
  }

  void createTestServer1() {
    const auto base_path =
        boost::filesystem::canonical("../../Tests/FsiDataFiles/").string();
    sql("CREATE SERVER test_server_1 FOREIGN DATA WRAPPER omnisci_csv "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '" +
        base_path + "');");
  }

  void assertNullForeignServer(const std::string& server_name) {
    auto& catalog = getCatalog();
    auto foreign_server = catalog.getForeignServerFromStorage(server_name);
    ASSERT_EQ(nullptr, foreign_server);
  }

 private:
  void dropTestTable() { sql("DROP FOREIGN TABLE IF EXISTS test_table;"); }
};

TEST_F(DropForeignServerTest, ExistingServer) {
  sql("DROP SERVER test_server;");
  assertNullForeignServer("test_server");
}

TEST_F(DropForeignServerTest, NonExistingServerWithIfExists) {
  sql("DROP SERVER IF EXISTS test_server_2;");
  assertNullForeignServer("test_server_2");
}

TEST_F(DropForeignServerTest, NonExistingServerWithoutIfExists) {
  queryAndAssertException("DROP SERVER test_server_2;",
                          "Exception: Foreign server with name \"test_server_2\" can not "
                          "be dropped. Server does not exist.");
}

TEST_F(DropForeignServerTest, DefaultServers) {
  queryAndAssertException("DROP SERVER omnisci_local_csv;",
                          "Exception: OmniSci default servers cannot be dropped.");
  queryAndAssertException("DROP SERVER omnisci_local_parquet;",
                          "Exception: OmniSci default servers cannot be dropped.");
}

TEST_F(DropForeignServerTest, ForeignTableReferencingServer) {
  createTestServer1();
  sql("CREATE FOREIGN TABLE test_table (c1 int) SERVER test_server_1 "
      "WITH (file_path = 'example_1.csv');");
  std::string error_message{
      "Exception: Foreign server \"test_server_1\" is referenced by existing "
      "foreign tables and cannot be dropped."};
  queryAndAssertException("DROP SERVER test_server_1;", error_message);
}

TEST_F(DropForeignServerTest, FsiDisabled) {
  g_enable_fsi = false;
  queryAndAssertException("DROP SERVER test_server;",
                          "Exception: Syntax error at: SERVER");
}

class ForeignServerPrivilegesDdlTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    createDBHandler();
    switchToAdmin();
    createTestUser();
  }

  static void TearDownTestSuite() { dropTestUser(); }

  void SetUp() override {
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUp();
    loginAdmin();
    dropServer();
  }

  void TearDown() override {
    g_enable_fsi = true;
    loginAdmin();
    dropServer();
    revokeTestUserServerPrivileges();
  }

  void revokeTestUserServerPrivileges() {
    sql("REVOKE ALL ON DATABASE omnisci FROM test_user;");
    sql("GRANT ACCESS ON DATABASE omnisci TO test_user;");
  }

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

  void createTestServer() {
    sql("CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  }

  void dropServer() { sql("DROP SERVER IF EXISTS test_server;"); }

  void assertExpectedForeignServerCreatedByUser() {
    auto& catalog = getCatalog();
    auto foreign_server = catalog.getForeignServerFromStorage("test_server");
    ASSERT_GT(foreign_server->id, 0);
    ASSERT_EQ(foreign_server->user_id, getCurrentUser().userId);
    ASSERT_EQ(getCurrentUser().userName, "test_user");
    ASSERT_EQ(foreign_server->name, "test_server");
  }

  void assertNullForeignServer() {
    auto& catalog = getCatalog();
    auto foreign_server = catalog.getForeignServerFromStorage("test_server");
    ASSERT_EQ(nullptr, foreign_server);
  }
};

TEST_F(ForeignServerPrivilegesDdlTest, CreateServerWithoutPrivilege) {
  login("test_user", "test_pass");
  queryAndAssertException(
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "
      "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');",
      "Exception: Server test_server will not be created. "
      "User has no create privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, CreateServerWithPrivilege) {
  sql("GRANT CREATE SERVER ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");
  createTestServer();
  assertExpectedForeignServerCreatedByUser();
}

TEST_F(ForeignServerPrivilegesDdlTest,
       CreateServerWithPrivilegeThenDropWithoutDropPrivilege) {
  sql("GRANT CREATE SERVER ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");
  createTestServer();
  assertExpectedForeignServerCreatedByUser();
  sql("DROP SERVER test_server;");
  assertNullForeignServer();
}

TEST_F(ForeignServerPrivilegesDdlTest, DropServerWithoutPrivilege) {
  createTestServer();
  login("test_user", "test_pass");
  queryAndAssertException("DROP SERVER test_server;",
                          "Exception: Server test_server will not be dropped. "
                          "User has no DROP SERVER privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, DropServerWithPrivilege) {
  createTestServer();
  sql("GRANT DROP SERVER ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");
  sql("DROP SERVER test_server;");
  assertNullForeignServer();
}

TEST_F(ForeignServerPrivilegesDdlTest, DropServerWithSpecificPrivilege) {
  createTestServer();
  sql("GRANT DROP ON SERVER test_server TO test_user;");
  login("test_user", "test_pass");
  sql("DROP SERVER test_server;");
  assertNullForeignServer();
}

TEST_F(ForeignServerPrivilegesDdlTest, AlterServerWithoutPrivilege) {
  createTestServer();
  login("test_user", "test_pass");
  queryAndAssertException(
      "ALTER SERVER test_server SET FOREIGN DATA WRAPPER OMNISCI_PARQUET;",
      "Exception: Server test_server can not be altered. User has no ALTER SERVER "
      "privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, AlterServerWithPrivilege) {
  createTestServer();
  sql("GRANT ALTER SERVER ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");
  sql("ALTER SERVER test_server SET FOREIGN DATA WRAPPER OMNISCI_PARQUET;");
  auto foreign_server = getCatalog().getForeignServerFromStorage("test_server");
  ASSERT_EQ(foreign_server->data_wrapper_type, "OMNISCI_PARQUET");
}

TEST_F(ForeignServerPrivilegesDdlTest, AlterServerWithSpecificPrivilege) {
  createTestServer();
  sql("GRANT ALTER ON SERVER test_server TO test_user;");
  login("test_user", "test_pass");
  sql("ALTER SERVER test_server SET FOREIGN DATA WRAPPER OMNISCI_PARQUET;");
  auto foreign_server = getCatalog().getForeignServerFromStorage("test_server");
  ASSERT_EQ(foreign_server->data_wrapper_type, "OMNISCI_PARQUET");
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantRevokeAlterServerWithPrivilege) {
  createTestServer();
  sql("GRANT ALTER SERVER ON DATABASE omnisci TO test_user;");
  sql("REVOKE ALTER SERVER ON DATABASE omnisci FROM test_user;");
  login("test_user", "test_pass");
  queryAndAssertException(
      "ALTER SERVER test_server SET FOREIGN DATA WRAPPER OMNISCI_PARQUET;",
      "Exception: Server test_server can not be altered. User has no ALTER SERVER "
      "privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantRevokeAlterServerWithSpecificPrivilege) {
  createTestServer();
  sql("GRANT ALTER ON SERVER test_server TO test_user;");
  sql("REVOKE ALTER ON SERVER test_server FROM test_user;");
  login("test_user", "test_pass");
  queryAndAssertException(
      "ALTER SERVER test_server SET FOREIGN DATA WRAPPER OMNISCI_PARQUET;",
      "Exception: Server test_server can not be altered. User has no ALTER SERVER "
      "privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest,
       DropServerWithSpecificPrivilegeRepeatedWithoutPrivilege) {
  createTestServer();
  sql("GRANT DROP ON SERVER test_server TO test_user;");
  login("test_user", "test_pass");
  sql("DROP SERVER test_server;");
  assertNullForeignServer();
  loginAdmin();
  createTestServer();
  login("test_user", "test_pass");
  queryAndAssertException("DROP SERVER test_server;",
                          "Exception: Server test_server will not be dropped. "
                          "User has no DROP SERVER privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, CreateServerWithGrantThenRevokePrivilege) {
  sql("GRANT CREATE SERVER ON DATABASE omnisci TO test_user;");
  sql("REVOKE CREATE SERVER ON DATABASE omnisci FROM test_user;");
  login("test_user", "test_pass");
  queryAndAssertException(
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "
      "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');",
      "Exception: Server test_server will not be created. "
      "User has no create privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, DropServerWithGrantThenRevokePrivilege) {
  createTestServer();
  sql("GRANT DROP SERVER ON DATABASE omnisci TO test_user;");
  sql("REVOKE DROP SERVER ON DATABASE omnisci FROM test_user;");
  login("test_user", "test_pass");
  queryAndAssertException("DROP SERVER test_server;",
                          "Exception: Server test_server will not be dropped. "
                          "User has no DROP SERVER privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, DropServerWithGrantThenRevokeSpecificPrivilege) {
  createTestServer();
  sql("GRANT DROP ON SERVER test_server TO test_user;");
  sql("REVOKE DROP ON SERVER test_server FROM test_user;");
  login("test_user", "test_pass");
  queryAndAssertException("DROP SERVER test_server;",
                          "Exception: Server test_server will not be dropped. "
                          "User has no DROP SERVER privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, RevokeNonExistentPrivilege) {
  createTestServer();
  queryAndAssertException("REVOKE DROP SERVER ON DATABASE omnisci FROM test_user;",
                          "Exception: Can not revoke privileges because"
                          " test_user has no privileges to omnisci");
}

TEST_F(ForeignServerPrivilegesDdlTest, RevokeNonExistentSpecificPrivilege) {
  createTestServer();
  queryAndAssertException("REVOKE DROP ON SERVER test_server FROM test_user;",
                          "Exception: Can not revoke privileges because test_user"
                          " has no privileges to test_server");
}

TEST_F(ForeignServerPrivilegesDdlTest, RevokeSpecificPrivilegeOnNonExistentServer) {
  queryAndAssertException("REVOKE DROP ON SERVER test_server FROM test_user;",
                          "Exception: Failure generating DB object key. "
                          "Server test_server does not exist.");
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantPrivilegeOnNonExistentDatabase) {
  queryAndAssertException("GRANT CREATE SERVER ON DATABASE nonexistent_db TO test_user;",
                          "Exception: Failure generating DB object key."
                          " Database nonexistent_db does not exist.");
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantPrivilegeFsiDisabled) {
  g_enable_fsi = false;
  queryAndAssertException("GRANT CREATE SERVER ON DATABASE omnisci TO test_user;",
                          "Exception: GRANT failed. SERVER object unrecognized.");
}

TEST_F(ForeignServerPrivilegesDdlTest, RevokePrivilegeFsiDisabled) {
  g_enable_fsi = false;
  queryAndAssertException("REVOKE CREATE SERVER ON DATABASE omnisci FROM test_user;",
                          "Exception: REVOKE failed. SERVER object unrecognized.");
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantSpecificPrivilegeFsiDisabled) {
  createTestServer();
  g_enable_fsi = false;
  queryAndAssertException("GRANT DROP ON SERVER test_server TO test_user;",
                          "Exception: GRANT failed. SERVER object unrecognized.");
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantAllCreateServerDropServer) {
  sql("GRANT ALL ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");
  createTestServer();
  assertExpectedForeignServerCreatedByUser();
  sql("DROP SERVER test_server;");
  assertNullForeignServer();
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantAllDropServer) {
  createTestServer();
  sql("GRANT ALL ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");
  sql("DROP SERVER test_server;");
  assertNullForeignServer();
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantAllRevokeCreateServerCreateServer) {
  sql("GRANT ALL ON DATABASE omnisci TO test_user;");
  sql("REVOKE CREATE SERVER ON DATABASE omnisci FROM test_user;");
  login("test_user", "test_pass");
  queryAndAssertException(
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "
      "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');",
      "Exception: Server test_server will not be created. "
      "User has no create privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantCreateServerRevokeAllCreateServer) {
  sql("GRANT CREATE SERVER ON DATABASE omnisci TO test_user;");
  sql("REVOKE ALL ON DATABASE omnisci FROM test_user;");
  // must still allow access for login:
  sql("GRANT ACCESS ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");
  queryAndAssertException(
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "
      "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');",
      "Exception: Server test_server will not be created. "
      "User has no create privileges.");
}

TEST_F(ForeignServerPrivilegesDdlTest, GrantDropServerRevokeAllDropServer) {
  createTestServer();
  sql("GRANT DROP SERVER ON DATABASE omnisci TO test_user;");
  sql("REVOKE ALL ON DATABASE omnisci FROM test_user;");
  // must still allow access for login:
  sql("GRANT ACCESS ON DATABASE omnisci TO test_user;");
  login("test_user", "test_pass");
  queryAndAssertException("DROP SERVER test_server;",
                          "Exception: Server test_server will not be dropped. "
                          "User has no DROP SERVER privileges.");
}

class ShowForeignServerTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUp();
    // Check defeault server exist
    {
      auto foreign_server = getCatalog().getForeignServer("omnisci_local_csv");
      ASSERT_NE(foreign_server, nullptr);
      ASSERT_GT(foreign_server->id, 0);
    }
    {
      auto foreign_server = getCatalog().getForeignServer("omnisci_local_parquet");
      ASSERT_NE(foreign_server, nullptr);
      ASSERT_GT(foreign_server->id, 0);
    }
  }

  enum ColumnIndex { SERVER_NAME, DATA_WRAPPER, CREATED_AT, OPTIONS, NUM_FIELDS };

  void assertExpectedFormat(const TQueryResult& result) {
    ASSERT_EQ(result.row_set.is_columnar, true);
    ASSERT_EQ(result.row_set.columns.size(), size_t(NUM_FIELDS));
    ASSERT_EQ(result.row_set.row_desc[SERVER_NAME].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[SERVER_NAME].col_name, "server_name");
    ASSERT_EQ(result.row_set.row_desc[DATA_WRAPPER].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[DATA_WRAPPER].col_name, "data_wrapper");
    ASSERT_EQ(result.row_set.row_desc[CREATED_AT].col_type.type, TDatumType::TIMESTAMP);
    ASSERT_EQ(result.row_set.row_desc[CREATED_AT].col_name, "created_at");
    ASSERT_EQ(result.row_set.row_desc[OPTIONS].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[OPTIONS].col_name, "options");
  }

  // Allow times within a one second window for rounding
  static const int TIME_EPSILON = 1.0;

  bool isDateTimeAfterTimeStamp(std::time_t datetime, std::time_t reference) {
    return std::difftime(datetime, reference) >= -TIME_EPSILON;
  }

  bool isDateTimeInPast(std::time_t datetime) {
    std::time_t current_time = std::time(nullptr);
    return std::difftime(datetime, current_time) <= TIME_EPSILON;
  }

  void assertServerResultFound(const TQueryResult& result,
                               const std::string& name,
                               const std::string& wrapper,
                               const std::string& options) {
    int num_matches = 0;
    for (size_t index = 0;
         index < result.row_set.columns[SERVER_NAME].data.str_col.size();
         ++index) {
      if (result.row_set.columns[SERVER_NAME].data.str_col[index] == name &&
          result.row_set.columns[DATA_WRAPPER].data.str_col[index] == wrapper &&
          result.row_set.columns[OPTIONS].data.str_col[index] == options) {
        num_matches++;
        ASSERT_GT(result.row_set.columns[CREATED_AT].data.int_col[index], 0);
        ASSERT_TRUE(
            isDateTimeInPast(result.row_set.columns[CREATED_AT].data.int_col[index]));
      }
    }
    ASSERT_EQ(num_matches, 1);
  }
  void assertServerLocalCSVFound(const TQueryResult& result) {
    assertServerResultFound(
        result, "omnisci_local_csv", "OMNISCI_CSV", "{\"STORAGE_TYPE\":\"LOCAL_FILE\"}");
  }
  void assertServerLocalParquetFound(const TQueryResult& result) {
    assertServerResultFound(result,
                            "omnisci_local_parquet",
                            "OMNISCI_PARQUET",
                            "{\"STORAGE_TYPE\":\"LOCAL_FILE\"}");
  }

  void assertNumResults(const TQueryResult& result, size_t num_results) {
    ASSERT_EQ(result.row_set.columns[SERVER_NAME].data.str_col.size(), num_results);
    ASSERT_EQ(result.row_set.columns[DATA_WRAPPER].data.str_col.size(), num_results);
    ASSERT_EQ(result.row_set.columns[CREATED_AT].data.int_col.size(), num_results);
    ASSERT_EQ(result.row_set.columns[OPTIONS].data.str_col.size(), num_results);
  }
};

TEST_F(ShowForeignServerTest, SHOWALL_DEFAULT) {
  TQueryResult result;
  std::string query{"SHOW SERVERS;"};
  sql(result, query);
  assertExpectedFormat(result);
  // Two default servers
  assertNumResults(result, 2);
  assertServerLocalCSVFound(result);
  assertServerLocalParquetFound(result);
}

TEST_F(ShowForeignServerTest, SHOW_WHERE_EQ) {
  TQueryResult result;
  std::string query{"SHOW SERVERS WHERE server_name = 'omnisci_local_csv';"};
  sql(result, query);
  assertExpectedFormat(result);
  assertNumResults(result, 1);
  assertServerLocalCSVFound(result);
}

TEST_F(ShowForeignServerTest, SHOW_WHERE_LIKE) {
  TQueryResult result;
  std::string query{"SHOW SERVERS WHERE server_name LIKE '%_parquet';"};
  sql(result, query);
  assertExpectedFormat(result);
  assertNumResults(result, 1);
  assertServerLocalParquetFound(result);
}

TEST_F(ShowForeignServerTest, SHOW_WHERE_LIKE_OR_EQ) {
  TQueryResult result;
  std::string query{
      "SHOW SERVERS WHERE server_name LIKE '%_csv' OR server_name = "
      "'omnisci_local_parquet';"};
  sql(result, query);
  assertExpectedFormat(result);
  assertNumResults(result, 2);
  assertServerLocalCSVFound(result);
  assertServerLocalParquetFound(result);
}

TEST_F(ShowForeignServerTest, SHOW_WHERE_LIKE_AND_EQ) {
  TQueryResult result;
  std::string query{
      "SHOW SERVERS WHERE server_name LIKE '%_csv' AND data_wrapper = 'OMNISCI_CSV';"};
  sql(result, query);
  assertExpectedFormat(result);
  assertNumResults(result, 1);
  assertServerLocalCSVFound(result);
}

TEST_F(ShowForeignServerTest, SHOW_WHERE_LIKE_OR_LIKE_AND_LIKE) {
  TQueryResult result;
  std::string query{
      "SHOW SERVERS WHERE server_name LIKE '%_csv' OR data_wrapper = 'OMNISCI_PARQUET' "
      "AND options LIKE '%STORAGE_TYPE%';"};
  sql(result, query);
  assertExpectedFormat(result);
  assertNumResults(result, 2);
  assertServerLocalCSVFound(result);
  assertServerLocalParquetFound(result);
}

TEST_F(ShowForeignServerTest, SHOW_WHERE_LIKE_EMPTY) {
  TQueryResult result;
  std::string query{"SHOW SERVERS WHERE server_name LIKE 'unknown%';"};
  sql(result, query);
  assertNumResults(result, 0);
}

TEST_F(ShowForeignServerTest, SHOW_WHERE_EQ_EMPTY) {
  TQueryResult result;
  std::string query{"SHOW SERVERS WHERE data_wrapper = 'unknown';"};
  sql(result, query);
  assertNumResults(result, 0);
}

TEST_F(ShowForeignServerTest, SHOW_TIMESTAMP_LIKE) {
  // check that like operator with timestamp will error out
  queryAndAssertException("SHOW SERVERS WHERE created_at LIKE 'test';",
                          "Exception: LIKE operator is incompatible with TIMESTAMP data");
}

TEST_F(ShowForeignServerTest, SHOW_TIMESTAMP_EQ) {
  // check that = operator with timestamp works
  // create new server as default ones may have the same date

  std::time_t current_time = std::time(nullptr);
  {
    TQueryResult result;
    std::string query{
        "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
        "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
    sql(result, query);
  }
  std::string options_json =
      "{\"BASE_PATH\":\"/test_path/\",\"STORAGE_TYPE\":\"LOCAL_FILE\"}";

  time_t created_at;
  {
    TQueryResult result;
    std::string query{"SHOW SERVERS;"};
    sql(result, query);
    assertNumResults(result, 3);
    created_at = result.row_set.columns[CREATED_AT].data.int_col[2];
  }
  {
    TQueryResult result;
    std::tm* ptm = std::gmtime(&created_at);
    char buffer[256];
    std::strftime(
        buffer, 256, "SHOW SERVERS WHERE created_at = '%Y-%m-%d %H:%M:%S';", ptm);
    sql(result, buffer);
    // Result must return the test_server, but may include the default servers, if the
    // previous tests run in under 1s
    size_t num_results = result.row_set.columns[CREATED_AT].data.int_col.size();
    ASSERT_GT(num_results, size_t(0));
    ASSERT_LT(num_results, size_t(4));
    assertServerResultFound(result, "test_server", "OMNISCI_CSV", options_json);
    for (const auto& timestamp : result.row_set.columns[CREATED_AT].data.int_col) {
      // Check all returned values
      ASSERT_TRUE(isDateTimeAfterTimeStamp(timestamp, current_time));
    }
  }

  {
    TQueryResult result;
    std::string query{"DROP SERVER test_server;"};
    sql(result, query);
  }
}

TEST_F(ShowForeignServerTest, SHOW_ADD_DROP) {
  // check that server is added to resultset when created and removed when dropped
  std::time_t current_time = std::time(nullptr);
  {
    TQueryResult result;
    std::string query{
        "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
        "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
    sql(result, query);
  }
  std::string options_json =
      "{\"BASE_PATH\":\"/test_path/\",\"STORAGE_TYPE\":\"LOCAL_FILE\"}";
  {
    TQueryResult result;
    std::string query{"SHOW SERVERS;"};
    sql(result, query);
    assertNumResults(result, 3);
    assertServerLocalCSVFound(result);
    assertServerLocalParquetFound(result);
    assertServerResultFound(result, "test_server", "OMNISCI_CSV", options_json);
  }
  {
    TQueryResult result;
    std::string query{"SHOW SERVERS WHERE server_name = 'test_server';"};
    sql(result, query);
    assertNumResults(result, 1);
    assertServerResultFound(result, "test_server", "OMNISCI_CSV", options_json);
  }
  {
    TQueryResult result;
    std::string query{"SHOW SERVERS WHERE options LIKE '%test_path%';"};
    sql(result, query);
    assertNumResults(result, 1);
    assertServerResultFound(result, "test_server", "OMNISCI_CSV", options_json);
    ASSERT_TRUE(isDateTimeAfterTimeStamp(
        result.row_set.columns[CREATED_AT].data.int_col[0], current_time));
  }
  {
    TQueryResult result;
    std::string query{"DROP SERVER test_server;"};
    sql(result, query);
  }
  {
    TQueryResult result;
    std::string query{"SHOW SERVERS;"};
    sql(result, query);
    assertExpectedFormat(result);
    // Two default servers
    assertNumResults(result, 2);
    assertServerLocalCSVFound(result);
    assertServerLocalParquetFound(result);
  }
}

TEST_F(ShowForeignServerTest, SHOW_PRIVILEGE) {
  // check that servers are only shown to users with correct privilege

  sql("CREATE USER test (password = 'HyperInterative', is_super = 'false', "
      "default_db='omnisci');");
  sql("GRANT ACCESS ON DATABASE omnisci TO test;");
  TSessionId test_session_id;
  login("test", "HyperInteractive", "omnisci", test_session_id);
  {
    TQueryResult result;
    sql(result, "SHOW SERVERS;", test_session_id);
    assertNumResults(result, 0);
  }
  sql("GRANT ALL ON DATABASE omnisci TO test;");
  {
    TQueryResult result;
    sql(result, "SHOW SERVERS;", test_session_id);
    // Two default servers
    assertNumResults(result, 2);
    assertServerLocalCSVFound(result);
    assertServerLocalParquetFound(result);
  }
  logout(test_session_id);
  sql("DROP USER test;");
}

TEST_F(ShowForeignServerTest, BadTimestamp) {
  std::string query{"SHOW SERVERS WHERE created_at = 'test';"};
  queryAndAssertException(query, "Exception: Invalid DATE/TIMESTAMP string (test)");
}

TEST_F(ShowForeignServerTest, BadQuery) {
  std::string query{"SHOW SERVERS WHERE server_name 'x' ;"};
  queryAndAssertException(
      query,
      "Exception: Parse failed: Encountered \"\\'x\\'\" at line 1, column 32.\nWas "
      "expecting one of:\n"
      "    \"LIKE\" ...\n    \"=\" ...\n    \".\" ...\n    ");
}

TEST_F(ShowForeignServerTest, BadAttribute) {
  std::string query{"SHOW SERVERS WHERE invalid_param = 'x';"};
  queryAndAssertException(
      query, "Exception: Attribute with name \"invalid_param\" does not exist.");
}

TEST_F(ShowForeignServerTest, FsiDisabled) {
  g_enable_fsi = false;
  std::string query{"SHOW SERVERS;"};
  queryAndAssertException(query, "Exception: Unsupported command: SHOW FOREIGN SERVERS");
  g_enable_fsi = true;
}

class AlterForeignServerTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    createDBHandler();
    switchToAdmin();
    createTestUser();
    setTestUserId();
  }

  static void TearDownTestSuite() { dropTestUser(); }

  static int32_t test_user_id;

  void SetUp() override {
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUp();
    loginAdmin();
    dropServers();
  }

  void TearDown() override {
    g_enable_fsi = true;
    loginAdmin();
    dropServers();
  }

  static void createTestUser() {
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE omnisci TO test_user;");
    sql("GRANT CREATE SERVER ON DATABASE omnisci TO test_user;");
  }

  static void dropTestUser() {
    try {
      sql("DROP USER test_user;");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }

  void createTestServer() {
    sql("CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  }

  void dropServers() {
    sql("DROP SERVER IF EXISTS test_server;");
    sql("DROP SERVER IF EXISTS renamed_server;");
  }

  const std::map<std::string, std::string, std::less<>> DEFAULT_OPTIONS{
      {"base_path", "/test_path/"},
      {"storage_type", "LOCAL_FILE"},
  };

  foreign_storage::ForeignServer createExpectedForeignServer(
      const std::string& server_name,
      const std::string& data_wrapper,
      const std::map<std::string, std::string, std::less<>>& options,
      const int owner_user_id = OMNISCI_ROOT_USER_ID) {
    foreign_storage::OptionsContainer options_container(
        foreign_storage::OptionsContainer(options).getOptionsAsJsonString());
    foreign_storage::ForeignServer foreign_server{
        server_name, to_upper(data_wrapper), options_container.options, owner_user_id};
    return foreign_server;
  }

  void assertExpectedForeignServer(
      const foreign_storage::ForeignServer& expected_foreign_server) {
    auto& catalog = getCatalog();
    assertForeignServersEqual(catalog.getForeignServer(expected_foreign_server.name),
                              expected_foreign_server);
    assertForeignServersEqual(
        catalog.getForeignServerFromStorage(expected_foreign_server.name).get(),
        expected_foreign_server);
  }

  void assertNullForeignServer(const std::string& server_name) {
    auto& catalog = getCatalog();
    const auto inmemory_foreign_server = catalog.getForeignServer(server_name);
    ASSERT_EQ(nullptr, inmemory_foreign_server);
    const auto foreign_server = catalog.getForeignServerFromStorage(server_name);
    ASSERT_EQ(nullptr, foreign_server);
  }

  void assertNullForeignServer() { assertNullForeignServer("test_server"); }

 private:
  static void setTestUserId() {
    login("test_user", "test_pass");
    Catalog_Namespace::UserMetadata test_user;
    test_user_id = Catalog_Namespace::SysCatalog::instance().getMetadataForUser(
        "test_user", test_user);
    loginAdmin();
  }

  void assertForeignServersEqual(
      const foreign_storage::ForeignServer* foreign_server,
      const foreign_storage::ForeignServer& expected_foreign_server) {
    ASSERT_GT(foreign_server->id, 0);
    ASSERT_EQ(foreign_server->name, expected_foreign_server.name);
    ASSERT_EQ(foreign_server->data_wrapper_type,
              expected_foreign_server.data_wrapper_type);
    ASSERT_EQ(foreign_server->user_id, expected_foreign_server.user_id);
    ASSERT_EQ(foreign_server->getOptionsAsJsonString(),
              expected_foreign_server.getOptionsAsJsonString());
  }
};

int32_t AlterForeignServerTest::test_user_id;

TEST_F(AlterForeignServerTest, SetDataWrapper) {
  createTestServer();
  sql("ALTER SERVER test_server SET FOREIGN DATA WRAPPER omnisci_parquet;");
  assertExpectedForeignServer(
      createExpectedForeignServer("test_server", "omnisci_parquet", DEFAULT_OPTIONS));
}

TEST_F(AlterForeignServerTest, ModifyOption) {
  createTestServer();
  sql("ALTER SERVER test_server SET ( base_path = '/new_path/' );");
  assertExpectedForeignServer(
      createExpectedForeignServer("test_server",
                                  "omnisci_csv",
                                  std::map<std::string, std::string, std::less<>>{
                                      {"base_path", "/new_path/"},
                                      {"storage_type", "LOCAL_FILE"},
                                  }));
}

TEST_F(AlterForeignServerTest, ChangeOwner) {
  createTestServer();
  sql("ALTER SERVER test_server OWNER TO test_user;");
  assertExpectedForeignServer(createExpectedForeignServer(
      "test_server", "omnisci_csv", DEFAULT_OPTIONS, test_user_id));
}

TEST_F(AlterForeignServerTest, RenameServerDropServer) {
  createTestServer();
  sql("ALTER SERVER test_server RENAME TO renamed_server;");
  assertExpectedForeignServer(
      createExpectedForeignServer("renamed_server", "omnisci_csv", DEFAULT_OPTIONS));
  assertNullForeignServer();
  sql("DROP SERVER renamed_server;");
  assertNullForeignServer("renamed_server");
}

TEST_F(AlterForeignServerTest, UserCreateServerRenameServerDropServer) {
  login("test_user", "test_pass");
  createTestServer();
  sql("ALTER SERVER test_server RENAME TO renamed_server;");
  assertExpectedForeignServer(createExpectedForeignServer(
      "renamed_server", "omnisci_csv", DEFAULT_OPTIONS, test_user_id));
  assertNullForeignServer();
  sql("DROP SERVER renamed_server;");
  assertNullForeignServer("renamed_server");
}

TEST_F(AlterForeignServerTest, UserCreateServerAttemptChangeOwner) {
  login("test_user", "test_pass");
  createTestServer();
  queryAndAssertException(
      "ALTER SERVER test_server OWNER TO " + OMNISCI_ROOT_USER + ";",
      "Exception: Only a super user can change a foreign server's owner."
      " Current user is not a super-user. Foreign server with name"
      " \"test_server\" will not have owner changed.");
}

TEST_F(AlterForeignServerTest, ChangeOwnerSwicthUserDropServer) {
  createTestServer();
  sql("ALTER SERVER test_server OWNER TO test_user;");
  login("test_user", "test_pass");
  sql("DROP SERVER test_server;");
  assertNullForeignServer();
}

TEST_F(AlterForeignServerTest, UserCreateServerChangeOwnerDropServer) {
  login("test_user", "test_pass");
  createTestServer();
  loginAdmin();
  sql("ALTER SERVER test_server OWNER TO " + OMNISCI_ROOT_USER + ";");
  login("test_user", "test_pass");
  queryAndAssertException("DROP SERVER test_server;",
                          "Exception: Server test_server will not be dropped. User has"
                          " no DROP SERVER privileges.");
}

TEST_F(AlterForeignServerTest, UserCreateServerDropUserChangeOwner) {
  login("test_user", "test_pass");
  createTestServer();
  loginAdmin();
  dropTestUser();
  sql("ALTER SERVER test_server OWNER TO " + OMNISCI_ROOT_USER + ";");
  assertExpectedForeignServer(
      createExpectedForeignServer("test_server", "omnisci_csv", DEFAULT_OPTIONS));
  // test_user must be recreated for remaining tests
  createTestUser();
}

TEST_F(AlterForeignServerTest, AlterNonExistentForeignServer) {
  queryAndAssertException(
      "ALTER SERVER test_server SET FOREIGN DATA WRAPPER omnisci_parquet;",
      "Exception: Foreign server with name \"test_server\" does not exist "
      "and can not be altered.");
}

TEST_F(AlterForeignServerTest, ChangeOwnerToNonExistentUser) {
  createTestServer();
  queryAndAssertException("ALTER SERVER test_server OWNER TO non_existent_user;",
                          "Exception: User with username \"non_existent_user\" does not"
                          " exist. Foreign server with name \"test_server\" can not "
                          "have owner changed.");
  assertExpectedForeignServer(
      createExpectedForeignServer("test_server", "omnisci_csv", DEFAULT_OPTIONS));
}

TEST_F(AlterForeignServerTest, RenameToExistingServer) {
  createTestServer();
  sql("CREATE SERVER renamed_server FOREIGN DATA WRAPPER omnisci_csv "
      "WITH (storage_type = 'LOCAL_FILE', base_path = "
      "'/another_test_path/');");
  queryAndAssertException("ALTER SERVER test_server RENAME TO renamed_server;",
                          "Exception: Foreign server with name \"test_server"
                          "\" can not be renamed to \"renamed_server\"."
                          "Foreign server with name \"renamed_server\" exists.");
  assertExpectedForeignServer(
      createExpectedForeignServer("test_server", "omnisci_csv", DEFAULT_OPTIONS));
}

TEST_F(AlterForeignServerTest, FsiDisabled) {
  createTestServer();
  g_enable_fsi = false;
  queryAndAssertException("ALTER SERVER test_server OWNER TO test_user;",
                          "Exception: Syntax error at: SERVER");
}

TEST_F(AlterForeignServerTest, OmniSciPrefix) {
  queryAndAssertException("ALTER SERVER omnisci_local_csv RENAME TO renamed_server;",
                          "Exception: OmniSci default servers cannot be altered.");
}

TEST_F(AlterForeignServerTest, RenameServerToOmniSciPrefix) {
  createTestServer();
  queryAndAssertException("ALTER SERVER test_server RENAME TO omnisci_local_csv;",
                          "Exception: OmniSci prefix can not be used for new"
                          " name of server.");
  assertExpectedForeignServer(
      createExpectedForeignServer("test_server", "omnisci_csv", DEFAULT_OPTIONS));
}

TEST_F(AlterForeignServerTest, InvalidOption) {
  createTestServer();
  std::string query{"ALTER SERVER test_server SET (invalid_key = 'value');"};
  std::string error_message{
      "Exception: Invalid option \"INVALID_KEY\". "
      "Option must be one of the following: STORAGE_TYPE, BASE_PATH."};
  queryAndAssertException(query, error_message);
  assertExpectedForeignServer(
      createExpectedForeignServer("test_server", "omnisci_csv", DEFAULT_OPTIONS));
}

TEST_F(AlterForeignServerTest, InvalidStorageType) {
  createTestServer();
  std::string query{
      "ALTER SERVER test_server SET "
      "(storage_type = 'INVALID_TYPE', base_path = '/test_path/');"};
  std::string error_message{
      "Exception: Invalid storage type value. Value must be one of the following: "
      "LOCAL_FILE."};

  queryAndAssertException(query, error_message);
  assertExpectedForeignServer(
      createExpectedForeignServer("test_server", "omnisci_csv", DEFAULT_OPTIONS));
}

TEST_F(AlterForeignServerTest, InvalidDataWrapper) {
  createTestServer();
  std::string query{"ALTER SERVER test_server SET FOREIGN DATA WRAPPER invalid_wrapper;"};
  std::string error_message{
      "Exception: Invalid data wrapper type \"INVALID_WRAPPER\". "
      "Data wrapper type must be one of the following: OMNISCI_CSV, OMNISCI_PARQUET."};
  queryAndAssertException(query, error_message);
  assertExpectedForeignServer(
      createExpectedForeignServer("test_server", "omnisci_csv", DEFAULT_OPTIONS));
}

int main(int argc, char** argv) {
  g_enable_fsi = true;
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_fsi = false;
  return err;
}
