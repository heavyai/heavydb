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
 * @file ExplainTest.cpp
 * @brief Test suite for EXPLAIN commands
 */

#include <gtest/gtest.h>

#include "DBHandlerTestHelpers.h"
#include "Shared/SysDefinitions.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class ExplainTest : public DBHandlerTestFixture {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists table_trivial;");
    sql("create table table_trivial (c1 integer);");
  }

  void TearDown() override {
    sql("drop table if exists table_trivial;");
    DBHandlerTestFixture::TearDown();
  }

 private:
};

TEST_F(ExplainTest, ExplainCmds) {
  // Basic tests to simply validate that EXPLAIN can be run
  //   and successfully accomplish simple parsing

  // TODO: verify output as well ?

  // EXPLAIN
  ASSERT_NO_THROW(sql("EXPLAIN SELECT COUNT(*) FROM table_trivial;"));

  // EXPLAIN OPTIMIZED
  ASSERT_NO_THROW(sql("EXPLAIN OPTIMIZED SELECT COUNT(*) FROM table_trivial;"));

  // EXPLAIN CALCITE
  ASSERT_NO_THROW(sql("EXPLAIN CALCITE SELECT COUNT(*) FROM table_trivial;"));

  // EXPLAIN PLAN
  ASSERT_NO_THROW(sql("EXPLAIN PLAN SELECT COUNT(*) FROM table_trivial;"));

  // After the "EXPLAIN PLAN" is stripped off the input and
  //   the remainder sent to Calcite it should throw an exception
  //   because "nonexistant_table_name" does not exist
  EXPECT_THROW(sql("EXPLAIN PLAN SELECT COUNT(*) FROM nonexistant_table_name;"),
               TOmniSciException);
}

int main(int argc, char** argv) {
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
