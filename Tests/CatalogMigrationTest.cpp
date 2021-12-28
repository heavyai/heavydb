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
#include "SqliteConnector/SqliteConnector.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;

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
      : cat_conn_("omnisci", BF::absolute("mapd_catalogs", BASE_PATH).string()) {}

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
    db_metadata.dbName = "omnisci";
    std::vector<LeafHostInfo> leaves{};
    return std::make_unique<Catalog_Namespace::Catalog>(
        BASE_PATH, db_metadata, nullptr, leaves, nullptr, false);
  }

  SqliteConnector cat_conn_;
};

class SysCatalogTest : public CatalogTest {
 protected:
  SysCatalogTest()
      : syscat_conn_("omnisci_system_catalog",
                     BF::absolute("mapd_catalogs", BASE_PATH).string()) {}

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
