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
 * @file CatalogMigrationTest.cpp
 * @brief Test suite for catalog migrations
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "Catalog/Catalog.h"
#include "DBHandlerTestHelpers.h"
#include "DataMgr/ForeignStorage/AbstractFileStorageDataWrapper.h"
#include "SqliteConnector/SqliteConnector.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;

class FsiSchemaTest : public testing::Test {
 protected:
  FsiSchemaTest()
      : sqlite_connector_(
            "omnisci",
            boost::filesystem::absolute("mapd_catalogs", BASE_PATH).string()) {}

  static void SetUpTestSuite() {
    g_enable_s3_fsi = true;
    Catalog_Namespace::SysCatalog::instance().init(
        BASE_PATH, nullptr, {}, nullptr, false, false, {});
  }

  void SetUp() override {
    g_enable_fsi = false;
    dropFsiTables();
  }

  void TearDown() override { dropFsiTables(); }

  std::vector<std::string> getTables() {
    sqlite_connector_.query("SELECT name FROM sqlite_master WHERE type='table';");
    std::vector<std::string> tables;
    for (size_t i = 0; i < sqlite_connector_.getNumRows(); i++) {
      tables.emplace_back(sqlite_connector_.getData<std::string>(i, 0));
    }
    return tables;
  }

  std::unique_ptr<Catalog_Namespace::Catalog> initCatalog() {
    Catalog_Namespace::DBMetadata db_metadata;
    db_metadata.dbName = "omnisci";
    std::vector<LeafHostInfo> leaves{};
    return std::make_unique<Catalog_Namespace::Catalog>(
        BASE_PATH, db_metadata, nullptr, leaves, nullptr, false);
  }

  void assertExpectedDefaultServer(Catalog_Namespace::Catalog* catalog,
                                   const std::string& server_name,
                                   const std::string& data_wrapper,
                                   const int32_t user_id) {
    auto foreign_server = catalog->getForeignServerFromStorage(server_name);

    ASSERT_GT(foreign_server->id, 0);
    ASSERT_EQ(server_name, foreign_server->name);
    ASSERT_EQ(data_wrapper, foreign_server->data_wrapper_type);
    ASSERT_EQ(user_id, foreign_server->user_id);

    ASSERT_TRUE(foreign_server->options.find(
                    foreign_storage::AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY) !=
                foreign_server->options.end());
    ASSERT_EQ(foreign_storage::AbstractFileStorageDataWrapper::LOCAL_FILE_STORAGE_TYPE,
              foreign_server->options
                  .find(foreign_storage::AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY)
                  ->second);

    ASSERT_TRUE(foreign_server->options.find(
                    foreign_storage::AbstractFileStorageDataWrapper::BASE_PATH_KEY) ==
                foreign_server->options.end());

    // check that server loaded from storage matches that in memory
    auto foreign_server_in_memory = catalog->getForeignServer(server_name);

    ASSERT_EQ(foreign_server->id, foreign_server_in_memory->id);
    ASSERT_EQ(foreign_server_in_memory->name, foreign_server->name);
    ASSERT_EQ(foreign_server_in_memory->data_wrapper_type,
              foreign_server->data_wrapper_type);
    ASSERT_EQ(foreign_server_in_memory->user_id, foreign_server->user_id);

    ASSERT_TRUE(foreign_server_in_memory->options.find(
                    foreign_storage::AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY) !=
                foreign_server_in_memory->options.end());
    ASSERT_EQ(foreign_storage::AbstractFileStorageDataWrapper::LOCAL_FILE_STORAGE_TYPE,
              foreign_server_in_memory->options
                  .find(foreign_storage::AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY)
                  ->second);

    ASSERT_TRUE(foreign_server_in_memory->options.find(
                    foreign_storage::AbstractFileStorageDataWrapper::BASE_PATH_KEY) ==
                foreign_server_in_memory->options.end());
  }

  void assertFsiTablesExist() {
    auto tables = getTables();
    ASSERT_FALSE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") ==
                 tables.end());
    ASSERT_FALSE(std::find(tables.begin(), tables.end(), "omnisci_foreign_tables") ==
                 tables.end());
  }

  void assertFsiTablesDoNotExist() {
    auto tables = getTables();
    ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") ==
                tables.end());
    ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_tables") ==
                tables.end());
  }

 private:
  SqliteConnector sqlite_connector_;

  void dropFsiTables() {
    sqlite_connector_.query("DROP TABLE IF EXISTS omnisci_foreign_servers;");
    sqlite_connector_.query("DROP TABLE IF EXISTS omnisci_foreign_tables;");
  }
};

TEST_F(FsiSchemaTest, FsiTablesNotCreatedWhenFsiIsDisabled) {
  assertFsiTablesDoNotExist();

  auto catalog = initCatalog();
  assertFsiTablesDoNotExist();
}

TEST_F(FsiSchemaTest, FsiTablesAreCreatedWhenFsiIsEnabled) {
  assertFsiTablesDoNotExist();

  g_enable_fsi = true;
  auto catalog = initCatalog();
  assertFsiTablesExist();
}

TEST_F(FsiSchemaTest, FsiTablesAreNotDroppedWhenFsiIsDisabled) {
  assertFsiTablesDoNotExist();

  g_enable_fsi = true;
  initCatalog();
  assertFsiTablesExist();

  g_enable_fsi = false;
  initCatalog();
  assertFsiTablesExist();
}

class ForeignTablesTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUpTestSuite();
  }

  static void TearDownTestSuite() {
    DBHandlerTestFixture::TearDownTestSuite();
    g_enable_fsi = false;
  }

  void SetUp() override {
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUp();
    dropTestTables();
  }

  void TearDown() override {
    dropTestTables();
    DBHandlerTestFixture::TearDown();
  }

 private:
  void dropTestTables() {
    g_enable_fsi = true;
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP VIEW IF EXISTS test_view;");
  }
};

TEST_F(ForeignTablesTest, ForeignTablesAreNotDroppedWhenFsiIsDisabled) {
  const auto file_path =
      boost::filesystem::canonical("../../Tests/FsiDataFiles/example_1.csv").string();
  sql("CREATE FOREIGN TABLE test_foreign_table (c1 int) SERVER omnisci_local_csv "
      "WITH (file_path = '" +
      file_path + "');");
  sql("CREATE TABLE test_table (c1 int);");
  sql("CREATE VIEW test_view AS SELECT * FROM test_table;");

  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_foreign_table", false));
  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_table", false));
  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_view", false));

  g_enable_fsi = false;
  resetCatalog();
  loginAdmin();

  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_foreign_table", false));
  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_table", false));
  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_view", false));
}

class DefaultForeignServersTest : public FsiSchemaTest {};

TEST_F(DefaultForeignServersTest, DefaultServersAreCreatedWhenFsiIsEnabled) {
  g_enable_fsi = true;
  auto catalog = initCatalog();
  g_enable_fsi = false;

  assertExpectedDefaultServer(catalog.get(),
                              "omnisci_local_csv",
                              foreign_storage::DataWrapperType::CSV,
                              OMNISCI_ROOT_USER_ID);

  assertExpectedDefaultServer(catalog.get(),
                              "omnisci_local_parquet",
                              foreign_storage::DataWrapperType::PARQUET,
                              OMNISCI_ROOT_USER_ID);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
