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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>

#include "Catalog/Catalog.h"
#include "DBHandlerTestHelpers.h"
#include "Logger/Logger.h"
#include "MigrationMgr/MigrationMgr.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/SysDefinitions.h"
#include "Shared/scope.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace TestHelpers;

using ::testing::_;
using ::testing::AtLeast;
using ::testing::NiceMock;
using ::testing::Return;

extern bool g_serialize_temp_tables;

class MockSqliteConnector : public SqliteConnector {
 public:
  MockSqliteConnector(sqlite3* db) : SqliteConnector(db) {}

  MOCK_METHOD(void, query, (const std::string& queryString));
  MOCK_METHOD(void, query_with_text_params, (std::string const& query_only));
  MOCK_METHOD(void,
              query_with_text_params,
              (const std::string& queryString,
               const std::vector<std::string>& text_param));
  MOCK_METHOD(void,
              query_with_text_param,
              (const std::string& queryString, const std::string& text_param));
  MOCK_METHOD(size_t, getNumRows, (), (const));

  void realQueryTextWithParams(const std::string& query_string,
                               const std::vector<std::string>& text_param) {
    return SqliteConnector::query_with_text_params(query_string, text_param);
  }

  void realQuery(const std::string& query_str) {
    return SqliteConnector::query(query_str);
  }
};

class DateInDaysMigrationTest : public DBHandlerTestFixture {
 public:
  static void SetUpTestSuite() {
    sql("DROP DATABASE IF EXISTS migration_mgr_db;");
    sql("CREATE DATABASE migration_mgr_db;");
    login(shared::kRootUsername, shared::kDefaultRootPasswd, "migration_mgr_db");
  }

  static void TearDownTestSuite() {
    loginAdmin();
    sql("DROP DATABASE IF EXISTS migration_mgr_db;");
  }

 protected:
  void SetUp() override {
    // create a real table
    sql("DROP TABLE IF EXISTS fake_date_in_days_metadata;");
    sql("CREATE TABLE fake_date_in_days_metadata(x INT, d DATE) WITH (FRAGMENT_SIZE=2);");

    // insert some data
    sql("INSERT INTO fake_date_in_days_metadata VALUES (1, '01/01/1991');");
    sql("INSERT INTO fake_date_in_days_metadata VALUES (2, '02/02/1992');");
    sql("INSERT INTO fake_date_in_days_metadata VALUES (3, '03/03/1993');");
    sql("INSERT INTO fake_date_in_days_metadata VALUES (4, '04/04/1994');");
  }

  void TearDown() override { sql("DROP TABLE IF EXISTS fake_date_in_days_metadata;"); }
};

TEST_F(DateInDaysMigrationTest, NoTables) {
  auto& cat = getCatalog();
  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  NiceMock<MockSqliteConnector> sqlite_mock(nullptr);
  EXPECT_CALL(sqlite_mock, query(_)).Times(8).WillRepeatedly(Return());
  EXPECT_CALL(sqlite_mock, getNumRows()).Times(2).WillRepeatedly(Return(0));
  EXPECT_NO_THROW(migrations::MigrationMgr::migrateDateInDaysMetadata(
      table_descriptors_map, cat.getCurrentDB().dbId, &cat, sqlite_mock));
}

auto get_chunk_metadata_vec =
    [](const std::string& table,
       const std::string& column,
       const Catalog_Namespace::Catalog& cat) -> std::shared_ptr<ChunkMetadata> {
  auto td = cat.getMetadataForTable(table);
  auto cd = cat.getMetadataForColumn(td->tableId, column);
  ChunkKey key{cat.getCurrentDB().dbId, td->tableId, cd->columnId, 1};  // second fragment
  auto& data_manager = cat.getDataMgr();
  ChunkMetadataVector chunk_metadata_vec;
  data_manager.getChunkMetadataVecForKeyPrefix(chunk_metadata_vec, key);
  CHECK_EQ(chunk_metadata_vec.size(), 1U);

  return chunk_metadata_vec[0].second;
};

TEST_F(DateInDaysMigrationTest, AlreadyMigrated) {
  auto& cat = getCatalog();

  // widen the metadata with an update query and add nulls
  sql("UPDATE fake_date_in_days_metadata SET d = NULL WHERE x = 3;");
  sql("UPDATE fake_date_in_days_metadata SET d = '04/04/2004' WHERE x = 4;");
  auto before_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d", cat);
  ASSERT_EQ(before_metadata->chunkStats.has_nulls, true);

  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  NiceMock<MockSqliteConnector> sqlite_mock(nullptr);
  EXPECT_CALL(sqlite_mock, query(_)).Times(4).WillRepeatedly(Return());
  EXPECT_CALL(sqlite_mock, getNumRows()).WillOnce(Return(1)).WillOnce(Return(1));
  EXPECT_NO_THROW(migrations::MigrationMgr::migrateDateInDaysMetadata(
      table_descriptors_map, cat.getCurrentDB().dbId, &cat, sqlite_mock));

  auto after_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d", cat);
  ASSERT_EQ(before_metadata->chunkStats.min.bigintval,
            after_metadata->chunkStats.min.bigintval);
  ASSERT_EQ(before_metadata->chunkStats.max.bigintval,
            after_metadata->chunkStats.max.bigintval);
  ASSERT_EQ(after_metadata->chunkStats.has_nulls, true);
}

TEST_F(DateInDaysMigrationTest, MigrateMetadata) {
  auto& cat = getCatalog();

  // get before update metadata
  auto before_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d", cat);
  ASSERT_EQ(before_metadata->chunkStats.has_nulls, false);

  // widen the metadata with an update query and add nulls
  sql("UPDATE fake_date_in_days_metadata SET d = NULL WHERE x = 3;");
  sql("UPDATE fake_date_in_days_metadata SET d = '04/04/2004' WHERE x = 4;");

  // check metadata after update
  auto after_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d", cat);
  ASSERT_EQ(before_metadata->chunkStats.min.bigintval,
            after_metadata->chunkStats.min.bigintval);
  ASSERT_NE(before_metadata->chunkStats.max.bigintval,
            after_metadata->chunkStats.max.bigintval);
  ASSERT_EQ(after_metadata->chunkStats.has_nulls, true);

  // return metadata to normal
  sql("UPDATE fake_date_in_days_metadata SET d = '03/03/1993' WHERE x = 3;");
  sql("UPDATE fake_date_in_days_metadata SET d = '04/04/1994' WHERE x = 4;");

  // run the migration
  auto table_descs = cat.getAllTableMetadata();
  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  for (auto descriptor : table_descs) {
    CHECK(table_descriptors_map
              .insert(std::make_pair(descriptor->tableId,
                                     const_cast<TableDescriptor*>(descriptor)))
              .second);
  }

  NiceMock<MockSqliteConnector> sqlite_mock(cat.getSqliteConnector().getSqlitePtr());

  ON_CALL(sqlite_mock, query_with_text_params(_, _))
      .WillByDefault([&sqlite_mock](const std::string& query_string,
                                    const std::vector<std::string>& text_param) {
        if (query_string ==
            "INSERT INTO mapd_version_history(version, migration_history) values(?,?)") {
          // ignore history updates
          return;
        }
        return sqlite_mock.realQueryTextWithParams(query_string, text_param);
      });

  ON_CALL(sqlite_mock, query(_))
      .WillByDefault([&sqlite_mock](const std::string& query_str) {
        // the mapd version history already exists and will already have registered
        // migrations, so ignore it
        if (!boost::algorithm::contains(query_str, "mapd_version_history")) {
          return sqlite_mock.realQuery(query_str);
        }
      });

  EXPECT_CALL(sqlite_mock, query(_)).Times(AtLeast(2));
  EXPECT_CALL(sqlite_mock, getNumRows())
      .WillOnce(Return(0))         // migration tables do not exist
      .WillRepeatedly(Return(1));  // 1 table to migrate
  EXPECT_NO_THROW(migrations::MigrationMgr::migrateDateInDaysMetadata(
      table_descriptors_map, cat.getCurrentDB().dbId, &cat, sqlite_mock));

  // check metadata after optimize
  auto optimized_metadata =
      get_chunk_metadata_vec("fake_date_in_days_metadata", "d", cat);
  ASSERT_EQ(optimized_metadata->chunkStats.min.bigintval,
            before_metadata->chunkStats.min.bigintval);
  ASSERT_EQ(optimized_metadata->chunkStats.max.bigintval,
            before_metadata->chunkStats.max.bigintval);
  ASSERT_EQ(optimized_metadata->chunkStats.has_nulls, false);
}

TEST_F(DateInDaysMigrationTest, RetryNotMigrated) {
  auto& cat = getCatalog();

  ScopeGuard reset = [&cat] {
    cat.getSqliteConnector().query(
        "DROP TABLE IF EXISTS mapd_date_in_days_column_migration_tmp");
  };

  // setup the retry table
  cat.getSqliteConnector().query(
      "CREATE TABLE mapd_date_in_days_column_migration_tmp(table_id integer primary "
      "key)");

  // get metadata before update
  auto before_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d", cat);

  // run update narrowing metadata
  sql("UPDATE fake_date_in_days_metadata SET d = '03/03/1993' WHERE x > 2;");

  // run the migration
  auto table_descs = cat.getAllTableMetadata();
  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  for (auto descriptor : table_descs) {
    CHECK(table_descriptors_map
              .insert(std::make_pair(descriptor->tableId,
                                     const_cast<TableDescriptor*>(descriptor)))
              .second);
  }

  NiceMock<MockSqliteConnector> sqlite_mock(cat.getSqliteConnector().getSqlitePtr());

  ON_CALL(sqlite_mock, query_with_text_params(_, _))
      .WillByDefault([&sqlite_mock](const std::string& query_string,
                                    const std::vector<std::string>& text_param) {
        if (query_string ==
            "INSERT INTO mapd_version_history(version, migration_history) values(?,?)") {
          // ignore history updates
          return;
        }
        return sqlite_mock.realQueryTextWithParams(query_string, text_param);
      });

  ON_CALL(sqlite_mock, query(_))
      .WillByDefault([&sqlite_mock](const std::string& query_str) {
        // the mapd version history already exists and will already have registered
        // migrations, so ignore it
        if (!boost::algorithm::contains(query_str, "mapd_version_history")) {
          return sqlite_mock.realQuery(query_str);
        }
      });

  EXPECT_CALL(sqlite_mock, query(_)).Times(AtLeast(2));
  EXPECT_CALL(sqlite_mock, getNumRows())
      .WillOnce(Return(1))         // migration tables exist
      .WillOnce(Return(0))         // date in days migration not done
      .WillOnce(Return(1))         // date in days migration table exists
      .WillOnce(Return(0))         // no tables migrated
      .WillRepeatedly(Return(1));  // one table to migrate
  EXPECT_NO_THROW(migrations::MigrationMgr::migrateDateInDaysMetadata(
      table_descriptors_map, cat.getCurrentDB().dbId, &cat, sqlite_mock));

  // check metadata after optimize
  auto optimized_metadata =
      get_chunk_metadata_vec("fake_date_in_days_metadata", "d", cat);
  ASSERT_EQ(optimized_metadata->chunkStats.min.bigintval,
            before_metadata->chunkStats.min.bigintval);
  ASSERT_EQ(
      optimized_metadata->chunkStats.max.bigintval,
      before_metadata->chunkStats.min.bigintval);  // all column values are identical
  ASSERT_EQ(optimized_metadata->chunkStats.has_nulls, false);
}

TEST_F(DateInDaysMigrationTest, RetryAlreadyMigrated) {
  auto& cat = getCatalog();

  ScopeGuard reset = [&cat] {
    cat.getSqliteConnector().query(
        "DROP TABLE IF EXISTS mapd_date_in_days_column_migration_tmp");
  };

  // setup the retry table
  cat.getSqliteConnector().query(
      "CREATE TABLE mapd_date_in_days_column_migration_tmp(table_id integer primary "
      "key)");

  // add the table ID for our test table
  auto td = cat.getMetadataForTable("fake_date_in_days_metadata");
  CHECK(td);
  cat.getSqliteConnector().query_with_text_params(
      "INSERT INTO mapd_date_in_days_column_migration_tmp VALUES(?)",
      std::vector<std::string>{std::to_string(td->tableId)});

  // get metadata before update
  auto before_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d", cat);

  // run update narrowing metadata
  sql("UPDATE fake_date_in_days_metadata SET d = '03/03/1993' WHERE x > 2;");

  // run the migration
  auto table_descs = cat.getAllTableMetadata();
  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  for (auto descriptor : table_descs) {
    CHECK(table_descriptors_map
              .insert(std::make_pair(descriptor->tableId,
                                     const_cast<TableDescriptor*>(descriptor)))
              .second);
  }

  NiceMock<MockSqliteConnector> sqlite_mock(cat.getSqliteConnector().getSqlitePtr());

  ON_CALL(sqlite_mock, query_with_text_params(_, _))
      .WillByDefault([&sqlite_mock](const std::string& query_string,
                                    const std::vector<std::string>& text_param) {
        if (query_string ==
            "INSERT INTO mapd_version_history(version, migration_history) values(?,?)") {
          // ignore history updates
          return;
        }
        return sqlite_mock.realQueryTextWithParams(query_string, text_param);
      });

  ON_CALL(sqlite_mock, query(_))
      .WillByDefault([&sqlite_mock](const std::string& query_str) {
        // the mapd version history already exists and will already have registered
        // migrations, so ignore it
        if (!boost::algorithm::contains(query_str, "mapd_version_history")) {
          return sqlite_mock.realQuery(query_str);
        }
      });

  EXPECT_CALL(sqlite_mock, query(_)).Times(AtLeast(2));
  EXPECT_CALL(sqlite_mock, getNumRows())
      .WillOnce(Return(1))         // migration tables exist
      .WillOnce(Return(0))         // date in days migration not done
      .WillOnce(Return(1))         // date in days migration table exists
      .WillOnce(Return(1))         // one table already migrated
      .WillRepeatedly(Return(1));  // one table to [potentially] migrate
  EXPECT_NO_THROW(migrations::MigrationMgr::migrateDateInDaysMetadata(
      table_descriptors_map, cat.getCurrentDB().dbId, &cat, sqlite_mock));

  // check metadata after optimize
  auto optimized_metadata =
      get_chunk_metadata_vec("fake_date_in_days_metadata", "d", cat);

  // No metadata narrowing
  ASSERT_EQ(optimized_metadata->chunkStats.min.bigintval,
            before_metadata->chunkStats.min.bigintval);
  ASSERT_EQ(optimized_metadata->chunkStats.max.bigintval,
            before_metadata->chunkStats.max.bigintval);
  ASSERT_EQ(optimized_metadata->chunkStats.has_nulls, false);
}

class RebrandMigrationTest : public ::testing::Test {
 public:
  static void setTestDir(const std::filesystem::path& test_dir) { test_dir_ = test_dir; }

 protected:
  void SetUp() override {
    std::filesystem::remove_all(test_dir_);
    std::filesystem::create_directory(test_dir_);
  }

  void TearDown() override {
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
  }

  void createLegacySystemFiles(bool create_optional_files) {
    for (const auto& dir_name : required_legacy_dirs_) {
      std::filesystem::create_directory(test_dir_ / dir_name);
    }
    createFile(test_dir_ / "mapd_catalogs" / "omnisci_system_catalog");
    if (create_optional_files) {
      for (const auto& dir_name : optional_legacy_dirs_) {
        std::filesystem::create_directory(test_dir_ / dir_name);
      }
      createFile(test_dir_ / "omnisci.license");
      createFile(test_dir_ / "omnisci_server_pid.lck");
      createFile(test_dir_ / "mapd_server_pid.lck");
      createFile(test_dir_ / "omnisci_key_store" / "omnisci.pem");
    }
  }

  void createNewSystemFiles(bool create_optional_files) {
    for (const auto& dir_name : required_new_dirs_) {
      std::filesystem::create_directory(test_dir_ / dir_name);
    }
    createFile(test_dir_ / shared::kCatalogDirectoryName / shared::kSystemCatalogName);
    if (create_optional_files) {
      for (const auto& dir_name : optional_new_dirs_) {
        std::filesystem::create_directory(test_dir_ / dir_name);
      }
      createFile(test_dir_ / shared::kDefaultLicenseFileName);
      createFile(test_dir_ / shared::kDefaultKeyStoreDirName /
                 shared::kDefaultKeyFileName);
    }
  }

  static void createFile(const std::filesystem::path& file_path) {
    std::ofstream fstream(file_path);
    fstream.close();
  }

  void assertExpectedRequiredFiles() {
    for (const auto& dir_name : required_new_dirs_) {
      assertDirectory(test_dir_ / dir_name);
    }

    for (const auto& dir_name : required_legacy_dirs_) {
      assertSymlink(test_dir_ / dir_name);
    }

    assertFile(test_dir_ / shared::kCatalogDirectoryName / shared::kSystemCatalogName);
    assertSymlink(test_dir_ / "mapd_catalogs" / "omnisci_system_catalog");
  }

  void assertExpectedOptionalFiles() {
    for (const auto& dir_name : optional_new_dirs_) {
      assertDirectory(test_dir_ / dir_name);
    }

    for (const auto& dir_name : optional_legacy_dirs_) {
      constexpr auto legacy_disk_cache_dir{"omnisci_disk_cache"};
      if (dir_name == legacy_disk_cache_dir) {
        ASSERT_FALSE(std::filesystem::exists(test_dir_ / legacy_disk_cache_dir));
      } else {
        assertSymlink(test_dir_ / dir_name);
      }
    }

    assertFile(test_dir_ / shared::kDefaultLicenseFileName);
    assertSymlink(test_dir_ / "omnisci.license");
    assertFile(test_dir_ / shared::kDefaultKeyStoreDirName / shared::kDefaultKeyFileName);
    assertSymlink(test_dir_ / "omnisci_key_store" / "omnisci.pem");

    ASSERT_FALSE(std::filesystem::exists(test_dir_ / "omnisci_server_pid.lck"));
    ASSERT_FALSE(std::filesystem::exists(test_dir_ / "mapd_server_pid.lck"));
    ASSERT_FALSE(std::filesystem::exists(test_dir_ / shared::kDefaultLogDirName /
                                         "omnisci_server.FATAL"));
    ASSERT_FALSE(std::filesystem::exists(test_dir_ / shared::kDefaultLogDirName /
                                         "omnisci_server.ERROR"));
    ASSERT_FALSE(std::filesystem::exists(test_dir_ / shared::kDefaultLogDirName /
                                         "omnisci_server.WARNING"));
    ASSERT_FALSE(std::filesystem::exists(test_dir_ / shared::kDefaultLogDirName /
                                         "omnisci_server.INFO"));
    ASSERT_FALSE(std::filesystem::exists(test_dir_ / shared::kDefaultLogDirName /
                                         "omnisci_web_server.ALL"));
    ASSERT_FALSE(std::filesystem::exists(test_dir_ / shared::kDefaultLogDirName /
                                         "omnisci_web_server.ACCESS"));
  }

  void assertDirectory(const std::filesystem::path& path) {
    ASSERT_TRUE(std::filesystem::exists(path)) << path;
    ASSERT_TRUE(std::filesystem::is_directory(path)) << path;
  }

  void assertFile(const std::filesystem::path& path) {
    ASSERT_TRUE(std::filesystem::exists(path)) << path;
    ASSERT_TRUE(std::filesystem::is_regular_file(path)) << path;
  }

  void assertSymlink(const std::filesystem::path& path) {
    ASSERT_TRUE(std::filesystem::exists(path)) << path;
    ASSERT_TRUE(std::filesystem::is_symlink(path)) << path;
  }

  static inline std::filesystem::path test_dir_;
  static inline const std::array<std::string, 3> required_legacy_dirs_{"mapd_data",
                                                                       "mapd_log",
                                                                       "mapd_catalogs"};
  static inline const std::array<std::string, 3> required_new_dirs_{
      shared::kDataDirectoryName,
      shared::kDefaultLogDirName,
      shared::kCatalogDirectoryName};
  static inline const std::array<std::string, 4> optional_legacy_dirs_{
      "mapd_export",
      "mapd_import",
      "omnisci_disk_cache",
      "omnisci_key_store"};
  static inline const std::array<std::string, 3> optional_new_dirs_{
      shared::kDefaultExportDirName,
      shared::kDefaultImportDirName,
      shared::kDefaultKeyStoreDirName};
};

TEST_F(RebrandMigrationTest, LegacyFiles) {
  createLegacySystemFiles(true);
  migrations::MigrationMgr::executeRebrandMigration(test_dir_.string());
  assertExpectedRequiredFiles();
  assertExpectedOptionalFiles();
}

TEST_F(RebrandMigrationTest, OptionalLegacyFilesMissing) {
  createLegacySystemFiles(false);
  migrations::MigrationMgr::executeRebrandMigration(test_dir_.string());
  assertExpectedRequiredFiles();
}

TEST_F(RebrandMigrationTest, NewFiles) {
  createNewSystemFiles(true);
  migrations::MigrationMgr::executeRebrandMigration(test_dir_.string());
  assertExpectedRequiredFiles();
  assertExpectedOptionalFiles();
}

TEST_F(RebrandMigrationTest, OptionalNewFilesMissing) {
  createNewSystemFiles(false);
  migrations::MigrationMgr::executeRebrandMigration(test_dir_.string());
  assertExpectedRequiredFiles();
}

class RenderGroupColumnMigrationTest : public DBHandlerTestFixture {
 public:
  void TearDown() override {
    switchToAdmin();
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    sql("DROP VIEW IF EXISTS test_view;");
    g_serialize_temp_tables = false;
  }

  static void TearDownTestSuite() { std::filesystem::remove_all(test_file_path_); }

  void assertPreMigrationTableColumnsAndRows(const std::string& table_name) {
    assertExpectedColumns(
        table_name, {"i", "poly", "poly_render_group", "mpoly", "mpoly_render_group"});
    assertPreMigrationTableRows(table_name);
  }

  void assertPreMigrationTableRows(const std::string& table_name) {
    // clang-format off
    sqlAndCompareResult("SELECT * FROM " + table_name + " ORDER BY i;",
                        {{i(1), "POLYGON ((0 0,1 0,1 1,0 1,0 0))", i(1),
                          "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((2 2,3 2,2 3,2 2)))", i(1)},
                         {i(2), "POLYGON ((1 1,3 1,2 3,1 1))", i(2),
                          "MULTIPOLYGON (((5 5,8 8,5 8,5 5)),((0 0,3 0,0 3,0 0)),"
                          "((11 11,10 12,10 10,11 11)))", i(2)}});
    // clang-format on
  }

  void assertPostMigrationTableColumnsAndRows(const std::string& table_name) {
    assertExpectedColumns(table_name, {"i", "poly", "mpoly"});
    assertPostMigrationTableRows(table_name);
  }

  void assertPostMigrationTableRows(const std::string& table_name) {
    // clang-format off
    sqlAndCompareResult("SELECT * FROM " + table_name + " ORDER BY i;",
                        {{i(1), "POLYGON ((0 0,1 0,1 1,0 1,0 0))",
                          "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((2 2,3 2,2 3,2 2)))"},
                         {i(2), "POLYGON ((1 1,3 1,2 3,1 1))",
                          "MULTIPOLYGON (((5 5,8 8,5 8,5 5)),((0 0,3 0,0 3,0 0)),"
                          "((11 11,10 12,10 10,11 11)))"}});
    // clang-format on
  }

  void assertExpectedColumns(const std::string& table_name,
                             const std::vector<std::string>& column_names) {
    auto td = getTableMetadata(table_name);
    auto logical_columns =
        getCatalog().getAllColumnMetadataForTable(td->tableId, false, false, false);
    ASSERT_EQ(column_names.size(), logical_columns.size());
    for (size_t i = 0; i < column_names.size(); i++) {
      auto it = std::next(logical_columns.begin(), i);
      EXPECT_EQ(column_names[i], (*it)->columnName);
    }
  }

  void createAndPopulateTestTable(bool is_temp_table) {
    sql("DROP TABLE IF EXISTS test_table;");
    // Create a table schema that would be identical to a table created before render
    // group column removal.
    sql(std::string{"CREATE "} + (is_temp_table ? "TEMPORARY " : "") +
        "TABLE test_table (i INTEGER, poly POLYGON, poly_render_group INTEGER, mpoly "
        "MULTIPOLYGON, mpoly_render_group INTEGER) WITH (fragment_size = 1);");
    sql("INSERT INTO test_table VALUES "
        "(1, 'POLYGON((0 0,1 0,1 1,0 1,0 0))', 1, "
        "'MULTIPOLYGON(((0 0,1 0,0 1,0 0)),((2 2,3 2,2 3,2 2)))', 1),"
        "(2, 'POLYGON((1 1,3 1,2 3,1 1))', 2, "
        "'MULTIPOLYGON(((5 5,8 8,5 8,5 5)),((0 0,3 0,0 3,0 0)),"
        "((11 11,10 12,10 10,11 11)))', 2);");
  }

  const TableDescriptor* getTableMetadata(const std::string& table_name) {
    auto& catalog = getCatalog();
    auto td = catalog.getMetadataForTable(table_name, false);
    CHECK(td);
    return td;
  }

  static void updateTestFile(bool add_render_group_columns) {
    // clang-format off
    static std::vector<std::vector<std::string>> file_lines{
        {"i", "poly", "poly_render_group", "mpoly", "mpoly_render_group"},
        {"1", "\"POLYGON((0 0,1 0,1 1,0 1,0 0))\"", "1",
         "\"MULTIPOLYGON(((0 0,1 0,0 1,0 0)),((2 2,3 2,2 3,2 2)))\"", "1"},
        {"2", "\"POLYGON((1 1,3 1,2 3,1 1))\"", "2",
         "\"MULTIPOLYGON(((5 5,8 8,5 8,5 5)),((0 0,3 0,0 3,0 0)),"
         "((11 11,10 12,10 10,11 11)))\"", "2"}};
    // clang-format on

    std::filesystem::remove_all(test_file_path_);
    std::ofstream file{test_file_path_};
    for (const auto& line : file_lines) {
      for (size_t i = 0; i < line.size(); i++) {
        if (!add_render_group_columns && (i == 2 || i == 4)) {
          continue;
        }
        file << line[i];
        if (i == line.size() - 1) {
          file << "\n";
        } else {
          file << ", ";
        }
      }
    }
  }

  static inline std::string test_file_path_{"./render_group_migration_test.csv"};
};

TEST_F(RenderGroupColumnMigrationTest, TableWithPolygonAndMultipolygonColumns) {
  createAndPopulateTestTable(false);

  static const std::string table_name{"test_table"};
  assertPreMigrationTableColumnsAndRows(table_name);

  auto td = getTableMetadata(table_name);
  migrations::MigrationMgr::dropRenderGroupColumns(
      {{td->tableId, const_cast<TableDescriptor*>(td)}}, &getCatalog());

  assertPostMigrationTableColumnsAndRows(table_name);
}

TEST_F(RenderGroupColumnMigrationTest, ForeignTableWithPolygonAndMultipolygonColumns) {
  updateTestFile(true);
  // Create a foreign table schema that would be identical to a table created before
  // render group column removal.
  sql("CREATE FOREIGN TABLE test_foreign_table (i INTEGER, poly POLYGON, "
      "poly_render_group INTEGER, mpoly MULTIPOLYGON, mpoly_render_group INTEGER) SERVER "
      "default_local_delimited WITH (file_path = '" +
      std::filesystem::canonical(test_file_path_).string() + "', fragment_size = 1);");

  static const std::string table_name{"test_foreign_table"};
  assertPreMigrationTableColumnsAndRows(table_name);

  auto td = getTableMetadata(table_name);
  migrations::MigrationMgr::dropRenderGroupColumns(
      {{td->tableId, const_cast<TableDescriptor*>(td)}}, &getCatalog());

  updateTestFile(false);
  assertPostMigrationTableColumnsAndRows(table_name);
}

TEST_F(RenderGroupColumnMigrationTest, ForeignTableInErrorState) {
  // Create a foreign table with mismatched number of columns.
  auto file_path = std::filesystem::canonical("../../Tests/FsiDataFiles/1.csv").string();
  sql("CREATE FOREIGN TABLE test_foreign_table (poly POLYGON, poly_render_group INTEGER, "
      "mpoly MULTIPOLYGON, mpoly_render_group INTEGER) SERVER default_local_delimited "
      "WITH (file_path = '" +
      file_path + "');");
  queryAndAssertException(
      "SELECT * FROM test_foreign_table;",
      "Mismatched number of logical columns: (expected 4 columns, has 1): in file '" +
          file_path + "'");

  auto td = getTableMetadata("test_foreign_table");
  migrations::MigrationMgr::dropRenderGroupColumns(
      {{td->tableId, const_cast<TableDescriptor*>(td)}}, &getCatalog());

  queryAndAssertException(
      "SELECT * FROM test_foreign_table;",
      "Mismatched number of logical columns: (expected 2 columns, has 1): in file '" +
          file_path + "'");
}

TEST_F(RenderGroupColumnMigrationTest, InformationSchemaDatabase) {
  login(shared::kRootUsername, shared::kDefaultRootPasswd, shared::kInfoSchemaDbName);
  auto& catalog = getCatalog();
  ASSERT_TRUE(catalog.isInfoSchemaDb());

  auto td = getTableMetadata("databases");
  sqlAndCompareResult("SELECT count(*) FROM databases WHERE database_name = '" +
                          shared::kInfoSchemaDbName + "';",
                      {{i(1)}});

  migrations::MigrationMgr::dropRenderGroupColumns(
      {{td->tableId, const_cast<TableDescriptor*>(td)}}, &catalog);

  sqlAndCompareResult("SELECT count(*) FROM databases WHERE database_name = '" +
                          shared::kInfoSchemaDbName + "';",
                      {{i(1)}});
}

TEST_F(RenderGroupColumnMigrationTest, View) {
  createAndPopulateTestTable(false);
  static const std::string table_name{"test_table"};
  sql("CREATE VIEW test_view AS SELECT * FROM " + table_name + ";");

  static const std::string view_name{"test_view"};
  assertPreMigrationTableRows(view_name);

  auto table = getTableMetadata(table_name);
  auto view = getTableMetadata(view_name);
  migrations::MigrationMgr::dropRenderGroupColumns(
      {{table->tableId, const_cast<TableDescriptor*>(table)},
       {view->tableId, const_cast<TableDescriptor*>(view)}},
      &getCatalog());

  assertPostMigrationTableRows(view_name);
}

// This test case is added for completeness, since temporary tables should not exist at
// server startup, when the migration is executed.
TEST_F(RenderGroupColumnMigrationTest, TemporaryTable) {
  g_serialize_temp_tables = true;
  createAndPopulateTestTable(true);

  static const std::string table_name{"test_table"};
  assertPreMigrationTableColumnsAndRows(table_name);

  auto td = getTableMetadata(table_name);
  migrations::MigrationMgr::dropRenderGroupColumns(
      {{td->tableId, const_cast<TableDescriptor*>(td)}}, &getCatalog());

  // Temporary tables are not touched.
  assertPreMigrationTableColumnsAndRows(table_name);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  RebrandMigrationTest::setTestDir(std::filesystem::canonical(argv[0]).parent_path() /
                                   "migration_test");

  // Disable automatic metadata update in order to ensure
  // that metadata is not automatically updated for other
  // tests that do and assert metadata updates.
  g_enable_auto_metadata_update = false;

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
