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
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  sql(result, query);
  assertExpectedForeignServer();
}

TEST_F(CreateForeignServerTest, AllValidParametersAndReadFromCache) {
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  sql(result, query);
  assertExpectedForeignServer();
}

TEST_F(CreateForeignServerTest, ExistingServerWithIfNotExists) {
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  sql(result, query);

  query =
      "CREATE SERVER IF NOT EXISTS test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');";
  sql(result, query);
  assertExpectedForeignServer();
}

TEST_F(CreateForeignServerTest, ExistingServerWithoutIfNotExists) {
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  sql(result, query);

  try {
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    ASSERT_EQ("Exception: A foreign server with name \"test_server\" already exists.",
              e.error_msg);
  }
}

TEST_F(CreateForeignServerTest, MissingStorageType) {
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(base_path = '/test_path/');"};
  try {
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    ASSERT_EQ("Exception: Foreign server options must contain \"STORAGE_TYPE\".",
              e.error_msg);
  }
}

TEST_F(CreateForeignServerTest, MissingBasePath) {
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE');"};
  try {
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    ASSERT_EQ("Exception: Foreign server options must contain \"BASE_PATH\".",
              e.error_msg);
  }
}

TEST_F(CreateForeignServerTest, InvalidOption) {
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(invalid_key = 'value', storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  try {
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    ASSERT_EQ(
        "Exception: Invalid option \"INVALID_KEY\". "
        "Option must be one of the following: STORAGE_TYPE, BASE_PATH.",
        e.error_msg);
  }
}

TEST_F(CreateForeignServerTest, InvalidStorageType) {
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'INVALID_TYPE', base_path = '/test_path/');"};
  try {
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    ASSERT_EQ(
        "Exception: Invalid storage type value. Value must be one of the following: "
        "LOCAL_FILE.",
        e.error_msg);
  }
}

TEST_F(CreateForeignServerTest, FsiDisabled) {
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  try {
    g_enable_fsi = false;
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    g_enable_fsi = true;
    ASSERT_EQ("Syntax error at: test_server", e.error_msg);
  } catch (...) {
    g_enable_fsi = true;
  }
}

TEST_F(CreateForeignServerTest, InvalidDataWrapper) {
  TQueryResult result;
  std::string query{
      "CREATE SERVER test_server FOREIGN DATA WRAPPER invalid_wrapper WITH "
      "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
  try {
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    ASSERT_EQ(
        "Exception: Invalid data wrapper type \"invalid_wrapper\". "
        "Data wrapper type must be one of the following: OMNISCI_CSV, OMNISCI_PARQUET.",
        e.error_msg);
  }
}

class DropForeignServerTest : public MapDHandlerTestFixture {
 protected:
  void SetUp() override {
    MapDHandlerTestFixture::SetUp();
    TQueryResult result;
    std::string query{
        "CREATE SERVER test_server FOREIGN DATA WRAPPER omnisci_csv WITH "
        "(storage_type = 'LOCAL_FILE', base_path = '/test_path/');"};
    sql(result, query);
  }

  void TearDown() override {
    getCatalog().dropForeignServer("test_server", true);
    MapDHandlerTestFixture::TearDown();
  }

  void assertNullForeignServer(const std::string& server_name) {
    auto& catalog = getCatalog();
    auto foreign_server = catalog.getForeignServer(server_name, true);

    ASSERT_EQ(nullptr, foreign_server);
  }
};

TEST_F(DropForeignServerTest, ExistingServer) {
  TQueryResult result;
  std::string query{"DROP SERVER test_server;"};
  sql(result, query);

  assertNullForeignServer("test_server");
}

TEST_F(DropForeignServerTest, NonExistingServerWithIfExists) {
  TQueryResult result;
  std::string query{"DROP SERVER IF EXISTS test_server_2;"};
  sql(result, query);

  assertNullForeignServer("test_server_2");
}

TEST_F(DropForeignServerTest, NonExistingServerWithoutIfExists) {
  TQueryResult result;
  std::string query{"DROP SERVER test_server_2;"};

  try {
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    ASSERT_EQ("Exception: Foreign server with name \"test_server_2\" does not exist.",
              e.error_msg);
  }
}

TEST_F(DropForeignServerTest, DefaultServers) {
  TQueryResult result;
  std::string query{"DROP SERVER omnisci_local_csv;"};

  try {
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    ASSERT_EQ("Exception: OmniSci default servers cannot be dropped.", e.error_msg);
  }

  query = "DROP SERVER omnisci_local_parquet;";
  try {
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    ASSERT_EQ("Exception: OmniSci default servers cannot be dropped.", e.error_msg);
  }
}

TEST_F(DropForeignServerTest, FsiDisabled) {
  TQueryResult result;
  std::string query{"DROP SERVER test_server;"};

  try {
    g_enable_fsi = false;
    sql(result, query);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TMapDException& e) {
    g_enable_fsi = true;
    ASSERT_EQ("Syntax error at: SERVER", e.error_msg);
  } catch (...) {
    g_enable_fsi = true;
  }
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
