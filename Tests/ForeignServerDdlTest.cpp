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

#include "MapDHandlerTestHelpers.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

class CreateForeignServerTest : public MapDHandlerTestFixture {
 protected:
  void SetUp() override {
    g_enable_fsi = true;
    MapDHandlerTestFixture::SetUp();
    getCatalog().dropForeignServer("test_server", true);
  }

  void TearDown() override {
    getCatalog().dropForeignServer("test_server", true);
    MapDHandlerTestFixture::TearDown();
  }

  void assertExpectedForeignServer() {
    auto& catalog = getCatalog();
    auto foreign_server = catalog.getForeignServer("test_server", true);

    ASSERT_GT(foreign_server->id, 0);
    ASSERT_EQ("test_server", foreign_server->name);
    ASSERT_EQ(foreign_storage::DataWrapper::CSV_WRAPPER_NAME,
              foreign_server->data_wrapper.name);

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
  queryAndAssertException(
      query, "Exception: Foreign server options must contain \"BASE_PATH\".");
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
  queryAndAssertException(query, "Syntax error at: test_server");
}

TEST_F(CreateForeignServerTest, InvalidDataWrapper) {
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER invalid_wrapper WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  std::string error_message{
      "Exception: Invalid data wrapper type \"invalid_wrapper\". "
      "Data wrapper type must be one of the following: OMNISCI_CSV, OMNISCI_PARQUET."};
  queryAndAssertException(query, error_message);
}

class DropForeignServerTest : public MapDHandlerTestFixture {
 protected:
  void SetUp() override {
    g_enable_fsi = true;
    MapDHandlerTestFixture::SetUp();
    dropTestTable();
    sql("CREATE SERVER IF NOT EXISTS test_server FOREIGN DATA WRAPPER omnisci_csv "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  }

  void TearDown() override {
    g_enable_fsi = true;
    dropTestTable();
    getCatalog().dropForeignServer("test_server", true);
    MapDHandlerTestFixture::TearDown();
  }

  void assertNullForeignServer(const std::string& server_name) {
    auto& catalog = getCatalog();
    auto foreign_server = catalog.getForeignServer(server_name, true);
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
  queryAndAssertException(
      "DROP SERVER test_server_2;",
      "Exception: Foreign server with name \"test_server_2\" does not exist.");
}

TEST_F(DropForeignServerTest, DefaultServers) {
  queryAndAssertException("DROP SERVER omnisci_local_csv;",
                          "Exception: OmniSci default servers cannot be dropped.");
  queryAndAssertException("DROP SERVER omnisci_local_parquet;",
                          "Exception: OmniSci default servers cannot be dropped.");
}

TEST_F(DropForeignServerTest, ForeignTableReferencingServer) {
  sql("CREATE FOREIGN TABLE test_table (c1 int) SERVER test_server "
      "WITH (file_path = 'test_file.csv');");
  std::string error_message{
      "Exception: Foreign server \"test_server\" is referenced by existing "
      "foreign tables and cannot be dropped."};
  queryAndAssertException("DROP SERVER test_server;", error_message);
}

TEST_F(DropForeignServerTest, FsiDisabled) {
  g_enable_fsi = false;
  queryAndAssertException("DROP SERVER test_server;", "Syntax error at: SERVER");
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
