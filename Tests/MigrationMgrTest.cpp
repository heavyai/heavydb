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

#include "TestHelpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>

#include "Catalog/Catalog.h"
#include "MigrationMgr/MigrationMgr.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/Logger.h"
#include "Shared/scope.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

inline void run_ddl_statement(const std::string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(
      query_str, device_type, /*hoist_literals=*/true, /*allow_loop_joins=*/true);
}

using ::testing::_;
using ::testing::AtLeast;
using ::testing::NiceMock;
using ::testing::Return;

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

class DateInDaysMigrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // create a real table
    run_ddl_statement("DROP TABLE IF EXISTS fake_date_in_days_metadata;");
    run_ddl_statement(
        "CREATE TABLE fake_date_in_days_metadata(x INT, d DATE) WITH (FRAGMENT_SIZE=2);");

    // insert some data
    run_multiple_agg("INSERT INTO fake_date_in_days_metadata VALUES (1, '01/01/1991');",
                     ExecutorDeviceType::CPU);
    run_multiple_agg("INSERT INTO fake_date_in_days_metadata VALUES (2, '02/02/1992');",
                     ExecutorDeviceType::CPU);
    run_multiple_agg("INSERT INTO fake_date_in_days_metadata VALUES (3, '03/03/1993');",
                     ExecutorDeviceType::CPU);
    run_multiple_agg("INSERT INTO fake_date_in_days_metadata VALUES (4, '04/04/1994');",
                     ExecutorDeviceType::CPU);
  }

  void TearDown() override {
    run_ddl_statement("DROP TABLE IF EXISTS fake_date_in_days_metadata;");
  }
};

TEST_F(DateInDaysMigrationTest, NoTables) {
  auto cat = QR::get()->getCatalog();
  CHECK(cat);

  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  NiceMock<MockSqliteConnector> sqlite_mock(nullptr);
  EXPECT_CALL(sqlite_mock, query(_)).Times(8).WillRepeatedly(Return());
  EXPECT_CALL(sqlite_mock, getNumRows()).Times(2).WillRepeatedly(Return(0));
  EXPECT_NO_THROW(migrations::MigrationMgr::migrateDateInDaysMetadata(
      table_descriptors_map, cat->getCurrentDB().dbId, cat.get(), sqlite_mock));
}

auto get_chunk_metadata_vec = [](const std::string& table,
                                 const std::string& column) -> ChunkMetadata {
  auto cat = QR::get()->getCatalog();

  auto td = cat->getMetadataForTable(table);
  auto cd = cat->getMetadataForColumn(td->tableId, column);
  ChunkKey key{
      cat->getCurrentDB().dbId, td->tableId, cd->columnId, 1};  // second fragment
  auto& data_manager = cat->getDataMgr();
  std::vector<std::pair<ChunkKey, ChunkMetadata>> chunk_metadata_vec;
  data_manager.getChunkMetadataVecForKeyPrefix(chunk_metadata_vec, key);
  CHECK_EQ(chunk_metadata_vec.size(), 1U);

  return chunk_metadata_vec[0].second;
};

TEST_F(DateInDaysMigrationTest, AlreadyMigrated) {
  auto cat = QR::get()->getCatalog();
  CHECK(cat);

  // widen the metadata with an update query and add nulls
  run_multiple_agg("UPDATE fake_date_in_days_metadata SET d = NULL WHERE x = 3;",
                   ExecutorDeviceType::CPU);
  run_multiple_agg("UPDATE fake_date_in_days_metadata SET d = '04/04/2004' WHERE x = 4;",
                   ExecutorDeviceType::CPU);
  auto before_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d");
  ASSERT_EQ(before_metadata.chunkStats.has_nulls, true);

  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  NiceMock<MockSqliteConnector> sqlite_mock(nullptr);
  EXPECT_CALL(sqlite_mock, query(_)).Times(4).WillRepeatedly(Return());
  EXPECT_CALL(sqlite_mock, getNumRows()).WillOnce(Return(1)).WillOnce(Return(1));
  EXPECT_NO_THROW(migrations::MigrationMgr::migrateDateInDaysMetadata(
      table_descriptors_map, cat->getCurrentDB().dbId, cat.get(), sqlite_mock));

  auto after_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d");
  ASSERT_EQ(before_metadata.chunkStats.min.bigintval,
            after_metadata.chunkStats.min.bigintval);
  ASSERT_EQ(before_metadata.chunkStats.max.bigintval,
            after_metadata.chunkStats.max.bigintval);
  ASSERT_EQ(after_metadata.chunkStats.has_nulls, true);
}

TEST_F(DateInDaysMigrationTest, MigrateMetadata) {
  auto cat = QR::get()->getCatalog();
  CHECK(cat);

  // get before update metadata
  auto before_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d");
  ASSERT_EQ(before_metadata.chunkStats.has_nulls, false);

  // widen the metadata with an update query and add nulls
  run_multiple_agg("UPDATE fake_date_in_days_metadata SET d = NULL WHERE x = 3;",
                   ExecutorDeviceType::CPU);
  run_multiple_agg("UPDATE fake_date_in_days_metadata SET d = '04/04/2004' WHERE x = 4;",
                   ExecutorDeviceType::CPU);

  // check metadata after update
  auto after_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d");
  ASSERT_EQ(before_metadata.chunkStats.min.bigintval,
            after_metadata.chunkStats.min.bigintval);
  ASSERT_NE(before_metadata.chunkStats.max.bigintval,
            after_metadata.chunkStats.max.bigintval);
  ASSERT_EQ(after_metadata.chunkStats.has_nulls, true);

  // return metadata to normal
  run_multiple_agg("UPDATE fake_date_in_days_metadata SET d = '03/03/1993' WHERE x = 3;",
                   ExecutorDeviceType::CPU);
  run_multiple_agg("UPDATE fake_date_in_days_metadata SET d = '04/04/1994' WHERE x = 4;",
                   ExecutorDeviceType::CPU);

  // run the migration
  auto table_descs = cat->getAllTableMetadata();
  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  for (auto descriptor : table_descs) {
    CHECK(table_descriptors_map
              .insert(std::make_pair(descriptor->tableId,
                                     const_cast<TableDescriptor*>(descriptor)))
              .second);
  }

  NiceMock<MockSqliteConnector> sqlite_mock(cat->getSqliteConnector().getSqlitePtr());

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
      table_descriptors_map, cat->getCurrentDB().dbId, cat.get(), sqlite_mock));

  // check metadata after optimize
  auto optimized_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d");
  ASSERT_EQ(optimized_metadata.chunkStats.min.bigintval,
            before_metadata.chunkStats.min.bigintval);
  ASSERT_EQ(optimized_metadata.chunkStats.max.bigintval,
            before_metadata.chunkStats.max.bigintval);
  ASSERT_EQ(optimized_metadata.chunkStats.has_nulls, false);
}

TEST_F(DateInDaysMigrationTest, RetryNotMigrated) {
  auto cat = QR::get()->getCatalog();
  CHECK(cat);

  ScopeGuard reset = [&cat] {
    cat->getSqliteConnector().query(
        "DROP TABLE IF EXISTS mapd_date_in_days_column_migration_tmp");
  };

  // setup the retry table
  cat->getSqliteConnector().query(
      "CREATE TABLE mapd_date_in_days_column_migration_tmp(table_id integer primary "
      "key)");

  // get metadata before update
  auto before_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d");

  // run update narrowing metadata
  run_multiple_agg("UPDATE fake_date_in_days_metadata SET d = '03/03/1993' WHERE x > 2;",
                   ExecutorDeviceType::CPU);

  // run the migration
  auto table_descs = cat->getAllTableMetadata();
  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  for (auto descriptor : table_descs) {
    CHECK(table_descriptors_map
              .insert(std::make_pair(descriptor->tableId,
                                     const_cast<TableDescriptor*>(descriptor)))
              .second);
  }

  NiceMock<MockSqliteConnector> sqlite_mock(cat->getSqliteConnector().getSqlitePtr());

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
      table_descriptors_map, cat->getCurrentDB().dbId, cat.get(), sqlite_mock));

  // check metadata after optimize
  auto optimized_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d");
  ASSERT_EQ(optimized_metadata.chunkStats.min.bigintval,
            before_metadata.chunkStats.min.bigintval);
  ASSERT_EQ(optimized_metadata.chunkStats.max.bigintval,
            before_metadata.chunkStats.min.bigintval);  // all column values are identical
  ASSERT_EQ(optimized_metadata.chunkStats.has_nulls, false);
}

TEST_F(DateInDaysMigrationTest, RetryAlreadyMigrated) {
  auto cat = QR::get()->getCatalog();
  CHECK(cat);

  ScopeGuard reset = [&cat] {
    cat->getSqliteConnector().query(
        "DROP TABLE IF EXISTS mapd_date_in_days_column_migration_tmp");
  };

  // setup the retry table
  cat->getSqliteConnector().query(
      "CREATE TABLE mapd_date_in_days_column_migration_tmp(table_id integer primary "
      "key)");

  // add the table ID for our test table
  auto td = cat->getMetadataForTable("fake_date_in_days_metadata");
  CHECK(td);
  cat->getSqliteConnector().query_with_text_params(
      "INSERT INTO mapd_date_in_days_column_migration_tmp VALUES(?)",
      std::vector<std::string>{std::to_string(td->tableId)});

  // get metadata before update
  auto before_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d");

  // run update narrowing metadata
  run_multiple_agg("UPDATE fake_date_in_days_metadata SET d = '03/03/1993' WHERE x > 2;",
                   ExecutorDeviceType::CPU);

  // run the migration
  auto table_descs = cat->getAllTableMetadata();
  Catalog_Namespace::TableDescriptorMapById table_descriptors_map;
  for (auto descriptor : table_descs) {
    CHECK(table_descriptors_map
              .insert(std::make_pair(descriptor->tableId,
                                     const_cast<TableDescriptor*>(descriptor)))
              .second);
  }

  NiceMock<MockSqliteConnector> sqlite_mock(cat->getSqliteConnector().getSqlitePtr());

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
      table_descriptors_map, cat->getCurrentDB().dbId, cat.get(), sqlite_mock));

  // check metadata after optimize
  auto optimized_metadata = get_chunk_metadata_vec("fake_date_in_days_metadata", "d");

  // No metadata narrowing
  ASSERT_EQ(optimized_metadata.chunkStats.min.bigintval,
            before_metadata.chunkStats.min.bigintval);
  ASSERT_EQ(optimized_metadata.chunkStats.max.bigintval,
            before_metadata.chunkStats.max.bigintval);
  ASSERT_EQ(optimized_metadata.chunkStats.has_nulls, false);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  QR::init(BASE_PATH,
           std::string{OMNISCI_ROOT_USER},
           "HyperInteractive",
           "migration_mgr_db",
           {},
           {},
           "",
           true,
           0,
           256 << 20,
           false,
           true);  // create new db

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::get()->runDDLStatement("DROP DATABASE IF EXISTS migration_mgr_db;");
  QR::reset();
  return err;
}
