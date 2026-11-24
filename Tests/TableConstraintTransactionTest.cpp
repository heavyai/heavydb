/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <functional>

#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"
#include "Catalog/TableConstraints.h"
#include "CudaMgr/CudaMgr.h"
#include "LogCaptureTestHelper.h"
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
  void updateMetadata(std::string, std::string) override {
    ++updates;
    if (on_update) {
      on_update();
    }
  }
  std::function<void()> on_update;
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

  void openCatalog(bool fresh) {
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
    SysCatalog::instance().init(dir_.string(), data_mgr, auth_, calcite_, fresh, {});
    catalog_ = SysCatalog::instance().getCatalog(shared::kDefaultDbName);
  }

  const TableDescriptor* createTable(const std::string& name) {
    TableDescriptor table;
    table.tableName = name;
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
    auto other = column;
    other.columnName = "other";
    catalog_->createTable(table, {column, other}, {}, true);
    return catalog_->getMetadataForTable(table.tableName, false);
  }

  void initializeCatalog(bool drop) {
    openCatalog(true);
    td_ = createTable("constraint_txn_test");
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
    reader.query("SELECT key_metainfo FROM mapd_table_key_metadata");
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
        "SELECT key_metainfo FROM mapd_table_key_metadata WHERE name = "
        "'constraint_txn_test'");
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

  void checkInvalidationFailure(bool drop) {
    initializeCatalog(drop);
    const auto updates = calcite_->updates;
    calcite_->on_update = [&] {
      // A direct catalog reader can reload as soon as Calcite evicts the table.
      checkStoredConstraint(!drop);
      throw std::runtime_error("no invalidation reply");
    };
    LogCapture logs;
    ASSERT_NO_THROW(changeConstraint(drop));
    EXPECT_TRUE(logs.contains("no invalidation reply"));
    EXPECT_EQ(updates + 1, calcite_->updates);
    EXPECT_NE(sqlite3_get_autocommit(catalog_->getSqliteConnector().getSqlitePtr()), 0);
    EXPECT_EQ(catalog_->getTableConstraints(td_).size(), drop ? 0u : 1u);
    catalog_.reset();
    SysCatalog::destroy();
    openCatalog(false);
    td_ = catalog_->getMetadataForTable("constraint_txn_test", false);
    ASSERT_NE(nullptr, td_);
    EXPECT_EQ(catalog_->getTableConstraints(td_).size(), drop ? 0u : 1u);
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

TEST_F(TableConstraintTransactionTest, MissingInvalidationReplyDoesNotUndoAdd) {
  checkInvalidationFailure(false);
}

TEST_F(TableConstraintTransactionTest, MissingInvalidationReplyDoesNotUndoDrop) {
  checkInvalidationFailure(true);
}

TEST_F(TableConstraintTransactionTest, DamagedChildDoesNotBlockUnrelatedDdlAfterRestart) {
  openCatalog(true);
  const auto parent = createTable("parent");
  const auto child = createTable("child");
  const auto unrelated = createTable("unrelated");
  catalog_->addTableConstraint(parent, constraint());
  catalog_->addTableConstraint(unrelated, constraint());
  catalog_->addTableConstraint(child,
                               {TableConstraintType::ForeignKey,
                                "child_fk",
                                {"id"},
                                Catalog_Namespace::ForeignKeyReference{"parent", {"id"}},
                                false});
  catalog_.reset();
  SysCatalog::destroy();
  {
    SqliteConnector db(shared::kDefaultDbName, catalogDirectory());
    db.query("UPDATE mapd_tables SET key_metainfo='not json' WHERE name='child'");
  }
  openCatalog(false);
  const auto reloaded_parent = catalog_->getMetadataForTable("parent", false);
  const auto reloaded_child = catalog_->getMetadataForTable("child", false);
  const auto reloaded_other = catalog_->getMetadataForTable("unrelated", false);
  ASSERT_NE(nullptr, reloaded_parent);
  ASSERT_NE(nullptr, reloaded_child);
  ASSERT_NE(nullptr, reloaded_other);
  ASSERT_EQ("not json", reloaded_child->keyMetainfo);
  try {
    catalog_->dropTable(reloaded_parent);
    FAIL() << "lost the incoming foreign key";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(
        "Cannot drop table parent because foreign key child_fk on table child references "
        "it.",
        std::string(e.what()));
  }
  EXPECT_THROW(catalog_->dropTableConstraint(reloaded_parent, "id_unique"),
               std::runtime_error);
  EXPECT_THROW(catalog_->renameTable(reloaded_parent, "renamed_parent"),
               std::runtime_error);
  auto col = catalog_->getMetadataForColumn(reloaded_other->tableId, "other");
  ASSERT_NE(nullptr, col);
  ASSERT_NO_THROW(catalog_->renameColumn(reloaded_other, col, "spare"));
  col = catalog_->getMetadataForColumn(reloaded_other->tableId, "spare");
  ASSERT_NE(nullptr, col);
  ASSERT_NO_THROW(catalog_->dropColumnTransactional(*reloaded_other, *col));
  ASSERT_NO_THROW(catalog_->renameTable(reloaded_other, "renamed_other"));
  ASSERT_NO_THROW(catalog_->dropTable(reloaded_other));
  ASSERT_NO_THROW(catalog_->dropTable(reloaded_child));
  EXPECT_NO_THROW(catalog_->dropTable(reloaded_parent));
}

TEST_F(TableConstraintTransactionTest, FailedAlterPreservesIncomingForeignKey) {
  openCatalog(true);
  const auto parent = createTable("parent");
  const auto child = createTable("child");
  catalog_->addTableConstraint(parent, constraint());
  catalog_->addTableConstraint(child,
                               {TableConstraintType::ForeignKey,
                                "fk",
                                {"id"},
                                Catalog_Namespace::ForeignKeyReference{"parent", {"id"}},
                                false});
  auto& db = catalog_->getSqliteConnector();
  db.query(
      "CREATE TRIGGER fail_second_record BEFORE INSERT ON mapd_constraints "
      "WHEN NEW.position=1 BEGIN SELECT RAISE(ABORT,'injected failure'); END");
  EXPECT_THROW(catalog_->addTableConstraint(child, constraint()), std::runtime_error);
  ASSERT_EQ(1u, catalog_->getTableConstraints(child).size());
  const auto keys = catalog_->getReferencingConstraints(parent->tableId);
  ASSERT_EQ(1u, keys.size());
  EXPECT_EQ("child", keys[0].first);
  EXPECT_EQ("fk", keys[0].second.name);
  EXPECT_THROW(catalog_->dropTable(parent), std::runtime_error);
  catalog_->dropTableConstraint(child, "fk");
  EXPECT_NO_THROW(catalog_->dropTable(parent));
}

TEST_F(TableConstraintTransactionTest, MigratesLegacyReferencesAtomically) {
  // Use a separate database with the pre-migration schema, not a mock of the index.
  const std::string fk =
      R"({"type":"FOREIGN KEY","name":"fk","columns":["id"],"foreign_table":"p","foreign_columns":["id"]})";
  const std::string directive = R"({"type":"SHARD KEY","name":"id"})";
  const std::string uq = R"({"type":"UNIQUE","name":"uq","columns":["other"]})";
  const std::string pk = R"({"type":"PRIMARY KEY","name":"pk","columns":["id"]})";
  // Two constraints share a slot before the directive; one follows it.
  const auto metadata = "[" + fk + "," + uq + "," + directive + "," + pk + "]";
  {
    SqliteConnector db("legacy", dir_.string());
    db.query(
        "CREATE TABLE mapd_tables(tableid INTEGER PRIMARY KEY,name TEXT,shard "
        "INTEGER,key_metainfo TEXT)");
    db.query(
        "CREATE TABLE mapd_logical_to_physical(physical_table_id "
        "INTEGER,logical_table_id INTEGER)");
    db.query_with_text_param(
        "INSERT INTO mapd_tables VALUES (1,'p',-1,?),(4,'broken',-1,'not json')",
        R"([{"type":"PRIMARY KEY","name":"pk","columns":["id"]}])");
    db.query_with_text_param(
        "INSERT INTO mapd_tables VALUES (2,'c',-1,?1),(3,'c_shard',0,?1)", metadata);
    db.query("INSERT INTO mapd_logical_to_physical VALUES (3,2)");
    {
      LogCapture logs;
      EXPECT_THROW(Catalog_Namespace::initialize_constraint_storage(db),
                   std::runtime_error);
      EXPECT_TRUE(
          logs.contains("Cannot migrate constraint metadata for table broken: Malformed "
                        "table key metadata."));
    }
    db.query("SELECT name FROM sqlite_master WHERE name='mapd_constraints'");
    EXPECT_EQ(0u, db.getNumRows());
    db.query("SELECT key_metainfo FROM mapd_tables WHERE tableid=2");
    EXPECT_EQ(metadata, db.getData<std::string>(0, 0));
    db.query("UPDATE mapd_tables SET key_metainfo='[]' WHERE tableid=4");
    // Fail after the parent's declaration has already moved. Neither half may commit.
    db.query("UPDATE mapd_tables SET name='missing' WHERE tableid=1");
    EXPECT_THROW(Catalog_Namespace::initialize_constraint_storage(db),
                 std::runtime_error);
    db.query("SELECT key_metainfo FROM mapd_tables WHERE tableid=1");
    EXPECT_NE(std::string::npos, db.getData<std::string>(0, 0).find("PRIMARY KEY"));
    db.query("UPDATE mapd_tables SET name='p' WHERE tableid=1");
  }
  {
    SqliteConnector db("legacy", dir_.string());
    Catalog_Namespace::initialize_constraint_storage(db);
    db.query("SELECT owner,parent FROM mapd_constraints WHERE parent IS NOT NULL");
    ASSERT_EQ(1u, db.getNumRows());
    EXPECT_EQ(2, db.getData<int>(0, 0));
    EXPECT_EQ(1, db.getData<int>(0, 1));
    db.query("SELECT key_metainfo FROM mapd_tables WHERE tableid IN (2,3)");
    ASSERT_EQ(2u, db.getNumRows());
    EXPECT_EQ("[" + directive + "]", db.getData<std::string>(0, 0));
    EXPECT_EQ("[" + directive + "]", db.getData<std::string>(1, 0));
    for (const auto reverse : {"OFF", "ON"}) {
      SCOPED_TRACE(reverse);
      db.query(std::string("PRAGMA reverse_unordered_selects=") + reverse);
      db.query("SELECT key_metainfo FROM mapd_table_key_metadata WHERE tableid IN (2,3)");
      ASSERT_EQ(2u, db.getNumRows());
      EXPECT_EQ(metadata, db.getData<std::string>(0, 0));
      EXPECT_EQ(metadata, db.getData<std::string>(1, 0));
    }
  }
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
