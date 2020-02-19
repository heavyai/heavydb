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
#include "SqliteConnector/SqliteConnector.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

class CatalogMigrationTest : public testing::Test {
 protected:
  CatalogMigrationTest()
      : sqlite_connector_(
            "omnisci",
            boost::filesystem::absolute("mapd_catalogs", BASE_PATH).string()) {}

  void SetUp() override {
    sqlite_connector_.query("DROP TABLE IF EXISTS omnisci_foreign_servers;");
  }

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
                                   const std::string& data_wrapper) {
    auto foreign_server = catalog->getForeignServer(server_name, true);

    ASSERT_GT(foreign_server->id, 0);
    ASSERT_EQ(server_name, foreign_server->name);
    ASSERT_EQ(data_wrapper, foreign_server->data_wrapper.name);

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
    ASSERT_EQ("/",
              foreign_server->options.find(foreign_storage::ForeignServer::BASE_PATH_KEY)
                  ->second);
  }

 private:
  SqliteConnector sqlite_connector_;
};

TEST_F(CatalogMigrationTest, FsiTablesNotCreatedWhenFsiIsDisabled) {
  auto tables = getTables();
  ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") ==
              tables.end());

  auto catalog = initCatalog();

  tables = getTables();
  ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") ==
              tables.end());
}

TEST_F(CatalogMigrationTest, FsiTablesAreCreatedWhenFsiIsEnabled) {
  auto tables = getTables();
  ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") ==
              tables.end());

  g_enable_fsi = true;
  auto catalog = initCatalog();
  g_enable_fsi = false;

  tables = getTables();
  ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") !=
              tables.end());
}

TEST_F(CatalogMigrationTest, FsiTablesAreDroppedWhenFsiIsDisabled) {
  auto tables = getTables();
  ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") ==
              tables.end());

  g_enable_fsi = true;
  initCatalog();
  g_enable_fsi = false;

  tables = getTables();
  ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") !=
              tables.end());

  initCatalog();
  tables = getTables();
  ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") ==
              tables.end());
}

TEST_F(CatalogMigrationTest, DefaultServersAreCreatedWhenFsiIsEnabled) {
  g_enable_fsi = true;
  auto catalog = initCatalog();
  g_enable_fsi = false;

  assertExpectedDefaultServer(
      catalog.get(), "omnisci_local_csv", foreign_storage::DataWrapper::CSV_WRAPPER_NAME);
  assertExpectedDefaultServer(catalog.get(),
                              "omnisci_local_parquet",
                              foreign_storage::DataWrapper::PARQUET_WRAPPER_NAME);
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
