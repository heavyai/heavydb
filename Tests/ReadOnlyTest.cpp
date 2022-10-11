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
 * @file ReadOnlyTest.cpp
 * @brief Test suite for read-only commands
 */

#include <gtest/gtest.h>

#include "DBHandlerTestHelpers.h"
#include "Shared/SysDefinitions.h"
#include "TestHelpers.h"

class ReadOnlyTest : public DBHandlerTestFixture {
 public:
  void SetUp() override { DBHandlerTestFixture::SetUp(); }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }

 private:
};

TEST_F(ReadOnlyTest, SELECT) {
  EXPECT_NO_THROW(sql("SELECT COUNT(*) FROM heavyai_us_states;"));
};

TEST_F(ReadOnlyTest, UPDATE) {
  EXPECT_ANY_THROW(sql("UPDATE heavyai_us_states SET abbr='22' WHERE abbr='11';"));
};

TEST_F(ReadOnlyTest, DELETE) {
  EXPECT_ANY_THROW(sql("DELETE FROM heavyai_us_states WHERE abbr='11';"));
};

TEST_F(ReadOnlyTest, INSERT) {
  EXPECT_ANY_THROW(sql("INSERT INTO heavyai_us_states VALUES('11')"));
};

TEST_F(ReadOnlyTest, CREATE) {
  EXPECT_ANY_THROW(sql("CREATE TABLE ReadOnlyTest(i integer);"));
};

TEST_F(ReadOnlyTest, CTAS) {
  EXPECT_ANY_THROW(
      sql("CREATE TABLE heavyai_us_states2 AS SELECT * FROM heavyai_us_states WITH "
          "(USE_SHARED_DICTIONARIES='false');"));
};

TEST_F(ReadOnlyTest, ITAS) {
  EXPECT_ANY_THROW(
      sql("INSERT INTO heavyai_us_states SELECT * FROM heavyai_us_states WHERE abbr = "
          "'11';"));
};

TEST_F(ReadOnlyTest, DROP) {
  EXPECT_ANY_THROW(sql("DROP TABLE heavyai_us_states;"));
}

int main(int argc, char** argv) {
  g_enable_fsi = false;
  g_enable_system_tables = false;
  g_read_only = true;
  DBHandlerTestFixture::disk_cache_level_ = File_Namespace::DiskCacheLevel::none;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  DBHandlerTestFixture::initTestArgs(argc, argv);

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
