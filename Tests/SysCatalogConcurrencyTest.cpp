/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file SysCatalogConcurrencyTest.cpp
 * @brief Concurrent access to SysCatalog::getCatalog
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "Catalog/Catalog.h"
#include "Catalog/SysCatalog.h"
#include "DBHandlerTestHelpers.h"
#include "Shared/scope.h"
#include "TestHelpers.h"

using Catalog_Namespace::Catalog;
using Catalog_Namespace::SysCatalog;

class SysCatalogConcurrencyTest : public DBHandlerTestFixture {
 protected:
  static constexpr const char* kDb = "concurrent_first_login_db";

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql(std::string("DROP DATABASE IF EXISTS ") + kDb + ";");
    sql(std::string("CREATE DATABASE ") + kDb + ";");
  }

  void TearDown() override {
    sql(std::string("DROP DATABASE IF EXISTS ") + kDb + ";");
    DBHandlerTestFixture::TearDown();
  }
};

// Concurrent first accesses to a database that has no cached Catalog must build exactly
// one Catalog. Each extra Catalog opens another SQLite connection on the same catalog
// file, a candidate source of "database is locked".
TEST_F(SysCatalogConcurrencyTest, ConcurrentFirstAccessBuildsOneCatalog) {
  constexpr size_t kThreads = 16;
  constexpr size_t kRounds = 20;
  auto& sys_cat = SysCatalog::instance();

  for (size_t round = 0; round < kRounds; ++round) {
    sys_cat.removeCatalog(kDb);  // forget the cached Catalog: next getCatalog rebuilds it

    const auto constructed_before = Catalog::numConstructed();
    std::atomic<size_t> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::shared_ptr<Catalog>> results(kThreads);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        ready++;
        while (!go.load()) {
          std::this_thread::yield();
        }
        results[t] = sys_cat.getCatalog(kDb);
      });
    }
    while (ready.load() < kThreads) {
      std::this_thread::yield();
    }
    go = true;
    for (auto& th : threads) {
      th.join();
    }

    for (size_t t = 0; t < kThreads; ++t) {
      ASSERT_NE(results[t], nullptr) << "round " << round;
      ASSERT_EQ(results[t].get(), results[0].get()) << "round " << round;
    }
    ASSERT_EQ(Catalog::numConstructed() - constructed_before, 1u)
        << "round " << round << ": concurrent first access built duplicate Catalogs";
  }
}

// Do duplicate Catalogs cause SQLITE_BUSY for DDL running on the
// Catalog that won the race? Concurrent first logins each run DDL right after login.
TEST_F(SysCatalogConcurrencyTest, ConcurrentFirstLoginDdlDoesNotFailWithSqliteBusy) {
  constexpr size_t kThreads = 8;
  constexpr size_t kRounds = 10;
  constexpr size_t kStatements = 6;
  auto& sys_cat = SysCatalog::instance();

  std::mutex errors_mutex;
  std::vector<std::string> errors;
  for (size_t round = 0; round < kRounds; ++round) {
    sys_cat.removeCatalog(kDb);
    std::atomic<size_t> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        ready++;
        while (!go.load()) {
          std::this_thread::yield();
        }
        try {
          TSessionId session;
          login("admin", "HyperInteractive", kDb, session);
          for (size_t i = 0; i < kStatements; ++i) {
            const auto table = "busy_t" + std::to_string(t) + "_" + std::to_string(i);
            TQueryResult result;
            sql(result, "CREATE TABLE " + table + " (a TEXT);", session);
            sql(result, "DROP TABLE " + table + ";", session);
          }
          logout(session);
        } catch (const TDBException& e) {
          std::lock_guard<std::mutex> lock(errors_mutex);
          errors.emplace_back(e.error_msg);
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lock(errors_mutex);
          errors.emplace_back(e.what());
        }
      });
    }
    while (ready.load() < kThreads) {
      std::this_thread::yield();
    }
    go = true;
    for (auto& th : threads) {
      th.join();
    }
  }
  for (const auto& e : errors) {
    ADD_FAILURE() << e;
  }
  ASSERT_TRUE(errors.empty()) << errors.size() << " failures";
}

// DROP USER holds the sys write lock and loads the Catalog of every database
// (getCatalogsForAllDbs), so it can reach getCatalog() for a database whose Catalog is
// being built by another thread. That must not deadlock: the builder needs the sys lock
// to finish. Building a small database's Catalog takes about a millisecond, so
// keep rebuilding it on background threads while DROP USERs run. A deadlock would hang
// the test, so a watchdog aborts it.
TEST_F(SysCatalogConcurrencyTest, ConcurrentCatalogBuildAndDropUserDoNotDeadlock) {
  constexpr size_t kRounds = 30;
  constexpr size_t kBuilders = 2;
  auto& sys_cat = SysCatalog::instance();

  std::atomic<bool> stop{false};
  std::vector<std::thread> builders;
  ScopeGuard stop_builders = [&] {
    stop = true;
    for (auto& th : builders) {
      th.join();
    }
  };
  for (size_t t = 0; t < kBuilders; ++t) {
    builders.emplace_back([&] {
      while (!stop.load()) {
        sys_cat.removeCatalog(kDb);  // the next getCatalog builds a new Catalog
        try {
          sys_cat.getCatalog(kDb);
        } catch (const std::exception&) {
          // Only the deadlock matters here; a failed build (e.g. SQLITE_BUSY from
          // duplicate Catalogs) is covered by the other tests.
        }
      }
    });
  }

  std::vector<std::string> users;
  ScopeGuard drop_users = [&] {
    for (const auto& user : users) {
      sql("DROP USER IF EXISTS " + user + ";");
    }
  };
  for (size_t round = 0; round < kRounds; ++round) {
    const auto user = "deadlock_user_" + std::to_string(round);
    users.push_back(user);
    sql("CREATE USER " + user + " (password = 'pw');");

    auto drop_done =
        std::async(std::launch::async, [&, user] { sql("DROP USER " + user + ";"); });
    if (drop_done.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
      // The stuck threads hold the sys lock, so neither joining nor tearing down can
      // work.
      std::cerr << "DEADLOCK: DROP USER did not finish within 60s while Catalogs were "
                   "being built (round "
                << round << ")" << std::endl;
      std::_Exit(1);
    }
    drop_done.get();
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
