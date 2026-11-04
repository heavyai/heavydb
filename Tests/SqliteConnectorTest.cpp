/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file SqliteConnectorTest.cpp
 * @brief Unit tests for SqliteConnector
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "SqliteConnector/SqliteConnector.h"
#include "TestHelpers.h"

namespace bf = boost::filesystem;

class EndTransactionOrAbortTest : public testing::Test {
 protected:
  void SetUp() override {
    dir_ = bf::temp_directory_path() / bf::unique_path("sqlite_connector_%%%%-%%%%");
    bf::create_directories(dir_);
  }

  void TearDown() override { bf::remove_all(dir_); }

  bf::path dir_;
};

TEST_F(EndTransactionOrAbortTest, EndsAndRollsBackTransactions) {
  SqliteConnector connector("test_catalog", dir_.string());
  connector.query("CREATE TABLE t (a INTEGER)");

  connector.query("BEGIN TRANSACTION");
  connector.query("INSERT INTO t VALUES (1)");
  end_transaction_or_abort(connector, "END TRANSACTION", "test");

  connector.query("BEGIN TRANSACTION");
  connector.query("INSERT INTO t VALUES (2)");
  end_transaction_or_abort(connector, "ROLLBACK TRANSACTION", "test");

  connector.query("SELECT count(*) FROM t");
  ASSERT_EQ(connector.getData<int>(0, 0), 1);
}

// A second connection holding a read transaction makes the writer's commit fail with
// SQLITE_BUSY ("database is locked"): the writer cannot take its EXCLUSIVE lock while a
// SHARED lock is held. The catalog serializes its own SQLite access, so this is only
// reachable through a bug or an outside actor, and the process must abort instead of
// continuing with a transaction in an unknown state.
TEST_F(EndTransactionOrAbortTest, AbortsWhenCommitFails) {
  ASSERT_DEATH(
      {
        SqliteConnector writer("test_catalog", dir_.string());
        writer.query("CREATE TABLE t (a INTEGER)");
        SqliteConnector reader("test_catalog", dir_.string());
        reader.query("BEGIN TRANSACTION");
        reader.query("SELECT count(*) FROM t");

        writer.query("BEGIN TRANSACTION");
        writer.query("INSERT INTO t VALUES (1)");
        end_transaction_or_abort(writer, "END TRANSACTION", "test");
      },
      "END TRANSACTION failed during test: Sqlite3 Error: database is locked");
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  // This line can be replaced with 'GTEST_FLAG_SET(death_test_style, "threadsafe")' once
  // we update to a version of GTEST that supports it.
  (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
