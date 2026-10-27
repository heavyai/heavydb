/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file ReadOnlyTest.cpp
 * @brief Test suite for read-only commands
 *
 * Thin-client test: connects to an already running heavydb that was started with
 * --read-only (see scripts/ci/configs/tests/read-only.sh). It does not link the
 * query engine.
 */

#include <gtest/gtest.h>

#include "SqlTestFixture.h"
#include "TestHelpers.h"

class ReadOnlyTest : public SqlTestFixture {};

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
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    SqlTestFixture::initConnection(argc, argv);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = 1;
  }
  return err;
}
