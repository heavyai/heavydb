/*
 * Copyright 2021 OmniSci, Inc.
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
 * @file AlterSystemTest.cpp
 * @brief Test suite for ALTER SYSTEM DDL commands
 */

#include <gtest/gtest.h>

#include "DBHandlerTestHelpers.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

class AlterSystemTest : public DBHandlerTestFixture {
 public:
  void SetUp() override { DBHandlerTestFixture::SetUp(); }

  static void SetUpTestSuite() {
    createDBHandler();
    users_ = {"user1"};
    superusers_ = {"super1"};
    dbs_ = {"db1"};
    createDBs();
    createUsers();
    createSuperUsers();
  }

  static void TearDownTestSuite() {
    dropUsers();
    dropSuperUsers();
    dropDBs();
  }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  // Return the amount of memory caches used in the system
  int64_t getMemoryData(const TSessionId& session, const std::string& memory_level) {
    std::vector<TNodeMemoryInfo> memory_data;
    auto* handler = getDbHandlerAndSessionId().first;
    handler->get_memory(memory_data, session, memory_level);
    int64_t free_page_count = 0;
    int64_t used_page_count = 0;
    int64_t page_count = 0;
    int64_t memory_used = 0;
    for (auto nodeIt : memory_data) {
      for (auto segIt : nodeIt.node_memory_data) {
        page_count += segIt.num_pages;
        if (segIt.is_free) {
          free_page_count += segIt.num_pages;
        }
      }
      used_page_count += page_count - free_page_count;
      memory_used += used_page_count * nodeIt.page_size;
    }
    return memory_used;
  }

  void testClearMemory(TSessionId& session, TExecuteMode::type em) {
    TQueryResult result;
    sql(result, "DROP TABLE IF EXISTS clear_memory_table;", session);
    sql(result, "CREATE TABLE clear_memory_table (f1 bigint);", session);

    for (int i = 0; i < 100; i++) {
      sql(result,
          "INSERT INTO clear_memory_table values (" + std::to_string(i) + ");",
          session);
    }

    if (em == TExecuteMode::GPU && setExecuteMode(em) == false) {
      // we haven't a GPU or the database is set in
      // cpu_only mode
      return;
    }
    setExecuteMode(em);
    if (em == TExecuteMode::CPU) {
      sql(result, "ALTER SYSTEM CLEAR CPU MEMORY", session);
      sql(result, "select count(f1) from clear_memory_table;", session);
      ASSERT_LE(800, getMemoryData(session, "cpu"));
      sql(result, "ALTER SYSTEM CLEAR CPU MEMORY", session);
      ASSERT_EQ(0, getMemoryData(session, "cpu"));
    } else if (em == TExecuteMode::GPU) {
      sql(result, "ALTER SYSTEM CLEAR GPU MEMORY", session);
      sql(result, "select count(f1) from clear_memory_table;", session);
      ASSERT_LE(800, getMemoryData(session, "gpu"));
      sql(result, "ALTER SYSTEM CLEAR GPU MEMORY", session);
      ASSERT_EQ(0, getMemoryData(session, "gpu"));
    }
    sql(result, "DROP TABLE IF EXISTS clear_memory_table;", session);
  }

  static void createUsers() {
    for (const auto& user : users_) {
      std::stringstream create;
      create << "CREATE USER " << user
             << " (password = 'HyperInteractive', is_super = 'false', "
                "default_db='omnisci');";
      sql(create.str());
      for (const auto& db : dbs_) {
        std::stringstream grant;
        grant << "GRANT ALL ON DATABASE  " << db << " to " << user << ";";
        sql(grant.str());
      }
    }
  }

  static void createSuperUsers() {
    for (const auto& user : superusers_) {
      std::stringstream create;
      create
          << "CREATE USER " << user
          << " (password = 'HyperInteractive', is_super = 'true', default_db='omnisci');";
      sql(create.str());
    }
  }

  static void dropUsers() {
    for (const auto& user : users_) {
      std::stringstream drop;
      drop << "DROP USER " << user << ";";
      sql(drop.str());
    }
  }

  static void dropSuperUsers() {
    for (const auto& user : superusers_) {
      std::stringstream drop;
      drop << "DROP USER " << user << ";";
      sql(drop.str());
    }
  }

  static void createDBs() {
    for (const auto& db : dbs_) {
      std::stringstream create;
      create << "CREATE DATABASE " << db << " (owner = 'admin');";
      sql(create.str());
    }
  }

  static void dropDBs() {
    for (const auto& db : dbs_) {
      std::stringstream drop;
      drop << "DROP DATABASE " << db << ";";
      sql(drop.str());
    }
  }

 private:
  static std::vector<std::string> users_;
  static std::vector<std::string> superusers_;
  static std::vector<std::string> dbs_;
};

std::vector<std::string> AlterSystemTest::users_;
std::vector<std::string> AlterSystemTest::superusers_;
std::vector<std::string> AlterSystemTest::dbs_;

TEST_F(AlterSystemTest, CLEAR_MEMORY_CPU_SUPER) {
  TSessionId new_session;
  TQueryResult result;

  login("admin", "HyperInteractive", "db1", new_session);
  testClearMemory(new_session, TExecuteMode::CPU);

  logout(new_session);
}

TEST_F(AlterSystemTest, CLEAR_MEMORY_GPU_SUPER) {
  TSessionId super_session;
  TQueryResult result;

  login("super1", "HyperInteractive", "db1", super_session);
  testClearMemory(super_session, TExecuteMode::GPU);

  logout(super_session);
}

TEST_F(AlterSystemTest, CLEAR_MEMORY_NOSUPER) {
  TSessionId user_session;
  TQueryResult result;
  login("user1", "HyperInteractive", "db1", user_session);
  try {
    sql(result, "ALTER SYSTEM CLEAR CPU MEMORY", user_session);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TOmniSciException& e) {
    ASSERT_EQ(
        "TException - service has thrown: TOmniSciException(error_msg=Superuser "
        "privilege is required to run clear_cpu_memory)",
        e.error_msg);
  }

  try {
    sql(result, "ALTER SYSTEM CLEAR GPU MEMORY", user_session);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TOmniSciException& e) {
    ASSERT_EQ(
        "TException - service has thrown: TOmniSciException(error_msg=Superuser "
        "privilege is required to run clear_gpu_memory)",
        e.error_msg);
  }
  try {
    sql(result, "ALTER SYSTEM CLEAR RENDER MEMORY", user_session);
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TOmniSciException& e) {
    ASSERT_EQ(
        "TException - service has thrown: TOmniSciException(error_msg=Superuser "
        "privilege is required to run clear_render_memory)",
        e.error_msg);
  }

  logout(user_session);
}

int main(int argc, char** argv) {
  g_enable_fsi = true;
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  DBHandlerTestFixture::initTestArgs(argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  g_enable_fsi = false;
  return err;
}
