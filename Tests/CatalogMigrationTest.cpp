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

/**
 * @file CatalogMigrationTest.cpp
 * @brief Test suite for catalog migrations
 *
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "Catalog/Catalog.h"
#include "DBHandlerTestHelpers.h"
#include "DataMgr/ForeignStorage/AbstractFileStorageDataWrapper.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapperFactory.h"
#include "Shared/SysDefinitions.h"
#include "SqliteConnector/SqliteConnector.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;
extern bool g_enable_system_tables;

namespace BF = boost::filesystem;
using SC = Catalog_Namespace::SysCatalog;

namespace {
bool table_exists(SqliteConnector& conn, const std::string& table_name) {
  conn.query("SELECT name FROM sqlite_master WHERE type='table' AND name='" + table_name +
             "'");
  return conn.getNumRows() > 0;
}

bool has_result(SqliteConnector& conn, const std::string& query) {
  conn.query(query);
  return conn.getNumRows() > 0;
}
}  // namespace

class CatalogTest : public DBHandlerTestFixture {
 protected:
  CatalogTest()
      : cat_conn_(shared::kDefaultDbName,
                  BF::absolute(shared::kCatalogDirectoryName, BASE_PATH).string()) {}

  static void SetUpTestSuite() {
    DBHandlerTestFixture::createDBHandler();
    initSysCatalog();
  }

  static void initSysCatalog() {
    auto db_handler = getDbHandlerAndSessionId().first;
    SC::instance().init(
        BASE_PATH, db_handler->data_mgr_, {}, db_handler->calcite_, false, false, {});
  }

  std::vector<std::string> getTables(SqliteConnector& conn) {
    conn.query("SELECT name FROM sqlite_master WHERE type='table';");
    std::vector<std::string> tables;
    for (size_t i = 0; i < conn.getNumRows(); i++) {
      tables.emplace_back(conn.getData<std::string>(i, 0));
    }
    return tables;
  }

  std::unique_ptr<Catalog_Namespace::Catalog> initCatalog() {
    Catalog_Namespace::DBMetadata db_metadata;
    db_metadata.dbName = shared::kDefaultDbName;
    db_metadata.dbId = 1;
    std::vector<LeafHostInfo> leaves{};
    return std::make_unique<Catalog_Namespace::Catalog>(
        BASE_PATH, db_metadata, nullptr, leaves, nullptr, false);
  }

  SqliteConnector cat_conn_;
};

class SysCatalogTest : public CatalogTest {
 protected:
  SysCatalogTest()
      : syscat_conn_(shared::kSystemCatalogName,
                     BF::absolute(shared::kCatalogDirectoryName, BASE_PATH).string()) {}

  void TearDown() override {
    if (tableExists("mapd_users")) {
      syscat_conn_.query("DELETE FROM mapd_users WHERE name='test_user'");
    }
    if (tableExists("mapd_object_permissions")) {
      syscat_conn_.query(
          "DELETE FROM mapd_object_permissions WHERE roleName='test_user'");
    }
  }

  bool hasResult(const std::string& query) { return has_result(syscat_conn_, query); }

  bool tableExists(const std::string& table_name) {
    return table_exists(syscat_conn_, table_name);
  }

  void createLegacyTestUser() {
    // This creates a test user in mapd_users syscat table, but does not properly add it
    // to mapd_object_permissions so it is incomplete by current standards.
    ASSERT_TRUE(table_exists(syscat_conn_, "mapd_users"));
    syscat_conn_.query("DELETE FROM mapd_users WHERE name='test_user'");
    syscat_conn_.query_with_text_params(
        "INSERT INTO mapd_users (name, passwd_hash, issuper, can_login) VALUES (?, ?, ?, "
        "?)",
        {"test_user", "passwd", "true", "true"});
  }

  static void reinitializeSystemCatalog() {
    SC::destroy();
    initSysCatalog();
  }

  SqliteConnector syscat_conn_;
};

// Check that we migrate correctly from pre 4.0 catalog.
TEST_F(SysCatalogTest, MigrateRoles) {
  // Make sure the post 4.0 tables do not exist to simulate migration.
  syscat_conn_.query("DROP TABLE IF EXISTS mapd_roles");
  syscat_conn_.query("DROP TABLE IF EXISTS mapd_object_permissions");
  createLegacyTestUser();

  // Create the pre 4.0 mapd_privileges table.
  syscat_conn_.query(
      "CREATE TABLE IF NOT EXISTS mapd_privileges (userid integer references mapd_users, "
      "dbid integer references mapd_databases, select_priv boolean, insert_priv boolean, "
      "UNIQUE(userid, dbid))");

  // Copy users who are not the admin (userid 0) into the pre 4.0 mapd_privileges table.
  syscat_conn_.query(
      "INSERT INTO mapd_privileges (userid, dbid) SELECT userid, default_db FROM "
      "mapd_users WHERE userid <> 0");

  // Re-initialization should perform migrations.
  reinitializeSystemCatalog();

  // Users should be inserted into mapd_object_permissions but not mapd_roles on
  // migration.
  ASSERT_TRUE(tableExists("mapd_roles"));
  ASSERT_FALSE(hasResult("SELECT roleName FROM mapd_roles WHERE roleName='test_user'"));

  ASSERT_TRUE(tableExists("mapd_object_permissions"));
  ASSERT_TRUE(hasResult(
      "SELECT roleName FROM mapd_object_permissions WHERE roleName='test_user'"));
}

TEST_F(SysCatalogTest, FixIncorrectRolesMigration) {
  ASSERT_TRUE(tableExists("mapd_roles"));
  createLegacyTestUser();

  // Setup an incorrect migration situation where we have usernames inserted into
  // mapd_roles.  This could occur between versions 4.0 and 5.7 and should now be fixed.
  ASSERT_TRUE(tableExists("mapd_users"));
  syscat_conn_.query("DELETE FROM mapd_roles WHERE roleName='test_user'");
  syscat_conn_.query_with_text_params("INSERT INTO mapd_roles VALUES (?, ?)",
                                      {"test_user", "test_user"});

  ASSERT_TRUE(hasResult("SELECT name FROM mapd_users WHERE name='test_user'"));
  ASSERT_TRUE(hasResult("SELECT roleName FROM mapd_roles WHERE roleName='test_user'"));

  // When we re-initialize the SysCatalog we should fix incorrect past migrations.
  reinitializeSystemCatalog();

  ASSERT_TRUE(hasResult("SELECT name FROM mapd_users WHERE name='test_user'"));
  ASSERT_FALSE(hasResult("SELECT roleName FROM mapd_roles WHERE roleName='test_user'"));
}

class FsiSchemaTest : public CatalogTest {
 protected:
  static void SetUpTestSuite() {
    g_enable_s3_fsi = true;
    g_enable_fsi = true;
    CatalogTest::SetUpTestSuite();
  }

  void SetUp() override {
    g_enable_fsi = false;
    g_enable_s3_fsi = false;
    g_enable_system_tables = false;
    dropFsiTables();
  }

  void TearDown() override { dropFsiTables(); }

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
    auto tables = getTables(cat_conn_);
    ASSERT_FALSE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") ==
                 tables.end());
    ASSERT_FALSE(std::find(tables.begin(), tables.end(), "omnisci_foreign_tables") ==
                 tables.end());
  }

  void assertFsiTablesDoNotExist() {
    auto tables = getTables(cat_conn_);
    ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_servers") ==
                tables.end());
    ASSERT_TRUE(std::find(tables.begin(), tables.end(), "omnisci_foreign_tables") ==
                tables.end());
  }

 private:
  void dropFsiTables() {
    cat_conn_.query("DROP TABLE IF EXISTS omnisci_foreign_servers;");
    cat_conn_.query("DROP TABLE IF EXISTS omnisci_foreign_tables;");
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
    g_enable_fsi = true;
    g_enable_system_tables = true;
    g_enable_s3_fsi = true;
    dropTestTables();
    DBHandlerTestFixture::TearDown();
  }

 private:
  void dropTestTables() {
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP VIEW IF EXISTS test_view;");
  }
};

TEST_F(ForeignTablesTest, ForeignTablesAreNotDroppedWhenFsiIsDisabled) {
  g_enable_fsi = true;
  resetCatalog();
  loginAdmin();

  const auto file_path = BF::canonical("../../Tests/FsiDataFiles/example_1.csv").string();
  sql("CREATE FOREIGN TABLE test_foreign_table (c1 int) SERVER default_local_delimited "
      "WITH (file_path = '" +
      file_path + "');");
  sql("CREATE TABLE test_table (c1 int);");
  sql("CREATE VIEW test_view AS SELECT * FROM test_table;");

  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_foreign_table", false));
  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_table", false));
  ASSERT_NE(nullptr, getCatalog().getMetadataForTable("test_view", false));

  g_enable_fsi = false;
  // The following flags should be false when FSI is disabled.
  g_enable_system_tables = false;
  g_enable_s3_fsi = false;

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
                              "default_local_delimited",
                              foreign_storage::DataWrapperType::CSV,
                              shared::kRootUserId);

  assertExpectedDefaultServer(catalog.get(),
                              "default_local_parquet",
                              foreign_storage::DataWrapperType::PARQUET,
                              shared::kRootUserId);
}

class SystemTableMigrationTest : public SysCatalogTest {
 protected:
  void SetUp() override {
    g_enable_system_tables = true;
    g_enable_fsi = true;
    dropInformationSchemaDb();
    deleteInformationSchemaMigration();
  }

  void TearDown() override {
    dropInformationSchemaDb();
    deleteInformationSchemaMigration();
    g_enable_system_tables = false;
    g_enable_fsi = false;
  }

  void dropInformationSchemaDb() {
    auto& system_catalog = SC::instance();
    Catalog_Namespace::DBMetadata db_metadata;
    if (system_catalog.getMetadataForDB(shared::kInfoSchemaDbName, db_metadata)) {
      system_catalog.dropDatabase(db_metadata);
    }
  }

  void deleteInformationSchemaMigration() {
    if (tableExists("mapd_version_history")) {
      syscat_conn_.query_with_text_param(
          "DELETE FROM mapd_version_history WHERE migration_history = ?",
          shared::kInfoSchemaMigrationName);
    }
  }

  bool isInformationSchemaMigrationRecorded() {
    return hasResult("SELECT * FROM mapd_version_history WHERE migration_history = '" +
                     shared::kInfoSchemaMigrationName + "';");
  }
};

TEST_F(SystemTableMigrationTest, SystemTablesEnabled) {
  g_enable_system_tables = true;
  g_enable_fsi = true;
  reinitializeSystemCatalog();
  ASSERT_TRUE(isInformationSchemaMigrationRecorded());
}

TEST_F(SystemTableMigrationTest, PreExistingInformationSchemaDatabase) {
  g_enable_system_tables = false;
  SC::instance().createDatabase("information_schema", shared::kRootUserId);

  g_enable_system_tables = true;
  g_enable_fsi = true;
  reinitializeSystemCatalog();
  ASSERT_FALSE(isInformationSchemaMigrationRecorded());
}

class LegacyDataWrapperMigrationTest : public FsiSchemaTest {
 protected:
  struct LegacyDataWrapperMapping {
    std::string test_server_name;
    std::string old_data_wrapper_name;
    std::string new_data_wrapper_name;
  };

  void insertForeignServer(const std::string& server_name,
                           const std::string& data_wrapper_type) {
    cat_conn_.query_with_text_params(
        "INSERT INTO omnisci_foreign_servers (name, data_wrapper_type, owner_user_id, "
        "creation_time, options) VALUES (?, ?, ?, ?, ?)",
        std::vector<std::string>{server_name,
                                 data_wrapper_type,
                                 shared::kRootUserIdStr,
                                 std::to_string(std::time(nullptr)),
                                 "{\"STORAGE_TYPE\":\"LOCAL_FILE\"}"});
  }

  void assertForeignServerCount(const std::string& server_name,
                                const std::string& data_wrapper_type,
                                size_t expected_count) {
    cat_conn_.query_with_text_params(
        "SELECT COUNT(*) FROM omnisci_foreign_servers WHERE name = ? AND "
        "data_wrapper_type = ?",
        std::vector<std::string>{server_name, data_wrapper_type});
    ASSERT_EQ(cat_conn_.getNumRows(), static_cast<size_t>(1));
    ASSERT_EQ(cat_conn_.getData<size_t>(0, 0), expected_count);
  }

  void clearMigration(const std::string& migration_name) {
    cat_conn_.query_with_text_params(
        "DELETE FROM mapd_version_history WHERE migration_history = ?",
        std::vector<std::string>{migration_name});
  }
};

TEST_F(LegacyDataWrapperMigrationTest, LegacyDataWrappersAreRenamed) {
  g_enable_fsi = true;
  initCatalog();
  assertFsiTablesExist();

  using foreign_storage::DataWrapperType;
  // clang-format off
  std::vector<LegacyDataWrapperMapping> legacy_data_wrapper_mappings{
    LegacyDataWrapperMapping{"test_csv_server",
                             "OMNISCI_CSV",
                             DataWrapperType::CSV},
    LegacyDataWrapperMapping{"test_parquet_server",
                             "OMNISCI_PARQUET",
                             DataWrapperType::PARQUET},
    LegacyDataWrapperMapping{"test_regex_server",
                             "OMNISCI_REGEX_PARSER",
                             DataWrapperType::REGEX_PARSER},
    LegacyDataWrapperMapping{"test_catalog_server",
                             "OMNISCI_INTERNAL_CATALOG",
                             DataWrapperType::INTERNAL_CATALOG},
    LegacyDataWrapperMapping{"test_memory_stats_server",
                             "INTERNAL_OMNISCI_MEMORY_STATS",
                             DataWrapperType::INTERNAL_MEMORY_STATS},
    LegacyDataWrapperMapping{"test_storage_stats_server",
                             "INTERNAL_OMNISCI_STORAGE_STATS",
                             DataWrapperType::INTERNAL_STORAGE_STATS}
  };
  // clang-format on

  for (const auto& mapping : legacy_data_wrapper_mappings) {
    // Insert foreign servers that use legacy data wrapper names
    insertForeignServer(mapping.test_server_name, mapping.old_data_wrapper_name);

    assertForeignServerCount(mapping.test_server_name, mapping.old_data_wrapper_name, 1);
    assertForeignServerCount(mapping.test_server_name, mapping.new_data_wrapper_name, 0);
  }

  clearMigration("rename_legacy_data_wrappers");
  initCatalog();
  for (const auto& mapping : legacy_data_wrapper_mappings) {
    // Assert that foreign servers now use the new data wrapper names
    assertForeignServerCount(mapping.test_server_name, mapping.old_data_wrapper_name, 0);
    assertForeignServerCount(mapping.test_server_name, mapping.new_data_wrapper_name, 1);
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
