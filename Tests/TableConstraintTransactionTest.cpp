/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"
#include "Catalog/TableConstraints.h"
#include "CudaMgr/CudaMgr.h"
#include "Parser/ParserNode.h"
#include "Shared/SysDefinitions.h"
#include "TestHelpers.h"

namespace bf = boost::filesystem;
using Catalog_Namespace::SysCatalog;
using Catalog_Namespace::TableConstraint;
using Catalog_Namespace::TableConstraintType;

extern std::string g_base_path;
extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;
extern bool g_enable_system_tables;

namespace {

class MetadataOnlyCalcite : public Calcite {
 public:
  void updateMetadata(std::string, std::string) override { ++updates; }
  size_t updates{0};
};

int deny_rollback(void*,
                  int action,
                  const char* transaction,
                  const char*,
                  const char*,
                  const char*) {
  return action == SQLITE_TRANSACTION && std::string(transaction) == "ROLLBACK"
             ? SQLITE_DENY
             : SQLITE_OK;
}

}  // namespace

class TableConstraintTransactionTest : public testing::Test {
 protected:
  void SetUp() override {
    // The re-executed death-test child uses the same directory. Only it initializes
    // the catalog; the parent can inspect the stored metadata and clean up afterward.
    dir_ = bf::path(BASE_PATH) / "constraint_transactions" /
           testing::UnitTest::GetInstance()->current_test_info()->name();
    bf::create_directories(dir_);
    ASSERT_TRUE(bf::is_empty(dir_));
    original_base_path_ = g_base_path;
  }

  void TearDown() override {
    catalog_.reset();
    SysCatalog::destroy();
    calcite_.reset();
    g_base_path = original_base_path_;
    bf::remove_all(dir_);
  }

  void initializeCatalog(bool drop) {
    bf::create_directories(dir_ / shared::kCatalogDirectoryName);
    bf::create_directories(dir_ / shared::kDataDirectoryName);
    bf::create_directories(dir_ / shared::kLockfilesDirectoryName /
                           shared::kCatalogDirectoryName);
    bf::create_directories(dir_ / shared::kLockfilesDirectoryName /
                           shared::kDataDirectoryName);
    g_base_path = dir_.string();
    SystemParameters parameters;
    auto data_mgr = std::make_shared<Data_Namespace::DataMgr>(
        (dir_ / shared::kDataDirectoryName).string(), parameters, nullptr, false);
    calcite_ = std::make_shared<MetadataOnlyCalcite>();
    SysCatalog::instance().init(dir_.string(), data_mgr, auth_, calcite_, true, {});
    catalog_ = SysCatalog::instance().getCatalog(shared::kDefaultDbName);

    TableDescriptor table;
    table.tableName = "constraint_txn_test";
    table.userId = shared::kRootUserId;
    table.isView = false;
    table.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
    table.maxFragRows = DEFAULT_FRAGMENT_ROWS;
    table.maxChunkSize = DEFAULT_MAX_CHUNK_SIZE;
    table.fragPageSize = DEFAULT_PAGE_SIZE;
    table.maxRows = DEFAULT_MAX_ROWS;
    table.keyMetainfo = "[]";
    ColumnDescriptor column;
    column.columnName = "id";
    column.columnType = SQLTypeInfo(kINT, true);
    catalog_->createTable(table, {column}, {}, true);
    td_ = catalog_->getMetadataForTable(table.tableName, false);
    CHECK(td_);
    if (drop) {
      catalog_->addTableConstraint(td_, constraint());
    }
  }

  TableConstraint constraint() const {
    return {TableConstraintType::Unique, "id_unique", {"id"}, std::nullopt, false};
  }

  void changeConstraint(bool drop) {
    if (drop) {
      catalog_->dropTableConstraint(td_, "id_unique");
    } else {
      catalog_->addTableConstraint(td_, constraint());
    }
  }

  std::string catalogDirectory() const {
    return (dir_ / shared::kCatalogDirectoryName).string();
  }

  void failCommit(bool drop) {
    initializeCatalog(drop);
    SqliteConnector reader(shared::kDefaultDbName, catalogDirectory());
    reader.query("BEGIN TRANSACTION");
    reader.query("SELECT key_metainfo FROM mapd_tables");
    changeConstraint(drop);
  }

  void failRollback(bool drop) {
    initializeCatalog(drop);
    SqliteConnector blocker(shared::kDefaultDbName, catalogDirectory());
    blocker.query("BEGIN IMMEDIATE TRANSACTION");
    CHECK_EQ(sqlite3_set_authorizer(
                 catalog_->getSqliteConnector().getSqlitePtr(), deny_rollback, nullptr),
             SQLITE_OK);
    changeConstraint(drop);
  }

  void checkStoredConstraint(bool present) {
    SqliteConnector reader(shared::kDefaultDbName, catalogDirectory());
    reader.query(
        "SELECT key_metainfo FROM mapd_tables WHERE name = 'constraint_txn_test'");
    ASSERT_EQ(reader.getNumRows(), size_t(1));
    const auto constraints = Catalog_Namespace::parse_table_constraints_from_key_metainfo(
        reader.getData<std::string>(0, 0));
    ASSERT_EQ(constraints.size(), present ? size_t(1) : size_t(0));
    if (present) {
      EXPECT_EQ(constraints.front().name, "id_unique");
    }
  }

  void checkUpdateFailure(bool drop) {
    initializeCatalog(drop);
    const auto original_metadata = td_->keyMetainfo;
    const auto original_updates = calcite_->updates;
    SqliteConnector blocker(shared::kDefaultDbName, catalogDirectory());
    blocker.query("BEGIN IMMEDIATE TRANSACTION");
    try {
      changeConstraint(drop);
      FAIL() << "The competing writer must block the constraint update";
    } catch (const std::runtime_error& e) {
      EXPECT_EQ(std::string(e.what()), "Sqlite3 Error: database is locked");
    }
    EXPECT_NE(sqlite3_get_autocommit(catalog_->getSqliteConnector().getSqlitePtr()), 0);
    EXPECT_EQ(td_->keyMetainfo, original_metadata);
    EXPECT_EQ(calcite_->updates, original_updates);
    blocker.query("ROLLBACK TRANSACTION");
    checkStoredConstraint(drop);

    // A successful rollback leaves this same Catalog usable for another operation.
    changeConstraint(drop);
    EXPECT_EQ(catalog_->getTableConstraints(td_).size(), drop ? size_t(0) : size_t(1));
    EXPECT_EQ(calcite_->updates, original_updates + 1);
    checkStoredConstraint(!drop);
  }

  bf::path dir_;
  std::string original_base_path_;
  AuthMetadata auth_;
  std::shared_ptr<MetadataOnlyCalcite> calcite_;
  std::shared_ptr<Catalog_Namespace::Catalog> catalog_;
  const TableDescriptor* td_{nullptr};
};

TEST_F(TableConstraintTransactionTest, AddConstraintAbortsWhenCommitIsBusy) {
  ASSERT_DEATH(failCommit(false),
               "END TRANSACTION failed during ALTER TABLE constraint_txn_test ADD "
               "CONSTRAINT: Sqlite3 Error: database is locked");
  checkStoredConstraint(false);
}

TEST_F(TableConstraintTransactionTest, DropConstraintAbortsWhenCommitIsBusy) {
  ASSERT_DEATH(failCommit(true),
               "END TRANSACTION failed during ALTER TABLE constraint_txn_test DROP "
               "CONSTRAINT: Sqlite3 Error: database is locked");
  checkStoredConstraint(true);
}

TEST_F(TableConstraintTransactionTest, AddConstraintAbortsWhenRollbackFails) {
  ASSERT_DEATH(failRollback(false),
               "ROLLBACK TRANSACTION failed during ALTER TABLE constraint_txn_test ADD "
               "CONSTRAINT: Sqlite3 Error: not authorized");
  checkStoredConstraint(false);
}

TEST_F(TableConstraintTransactionTest, DropConstraintAbortsWhenRollbackFails) {
  ASSERT_DEATH(failRollback(true),
               "ROLLBACK TRANSACTION failed during ALTER TABLE constraint_txn_test DROP "
               "CONSTRAINT: Sqlite3 Error: not authorized");
  checkStoredConstraint(true);
}

TEST_F(TableConstraintTransactionTest, AddConstraintRollsBackFailedUpdate) {
  checkUpdateFailure(false);
}

TEST_F(TableConstraintTransactionTest, DropConstraintRollsBackFailedUpdate) {
  checkUpdateFailure(true);
}

int main(int argc, char** argv) {
  g_enable_fsi = false;
  g_enable_s3_fsi = false;
  g_enable_system_tables = false;
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
  return RUN_ALL_TESTS();
}
