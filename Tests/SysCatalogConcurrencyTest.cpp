/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file SysCatalogConcurrencyTest.cpp
 * @brief Concurrent access to SysCatalog::getCatalog (HERC-276)
 */

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

#include "Catalog/Catalog.h"
#include "Catalog/SysCatalog.h"
#include "DBHandlerTestHelpers.h"
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
// file, a candidate source of "database is locked" (HERC-276).
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
