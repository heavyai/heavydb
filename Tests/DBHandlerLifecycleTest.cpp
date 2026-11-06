/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>

#include "DBHandlerTestHelpers.h"
#include "QueryEngine/Execute.h"

using namespace TestHelpers;

class DBHandlerLifecycleTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("DROP TABLE IF EXISTS handler_lifecycle;");
    sql("CREATE TABLE handler_lifecycle (id INT);");
  }

  void TearDown() override {
    createDBHandler();
    sql("DROP TABLE IF EXISTS handler_lifecycle;");
    DBHandlerTestFixture::TearDown();
  }
};

TEST_F(DBHandlerLifecycleTest, ReleasesCachedExecutorsAcrossRestarts) {
  for (int64_t generation = 1; generation <= 3; ++generation) {
    SCOPED_TRACE(generation);
    sql("INSERT INTO handler_lifecycle VALUES (" + std::to_string(generation) + ");");
    sqlAndCompareResult("SELECT COUNT(*), SUM(id) FROM handler_lifecycle;",
                        {{i(generation), i(generation * (generation + 1) / 2)}});

    // Cover the direct executor, a query worker and the update/delete worker.
    // Weak references detect retained executors even if DataMgr's address is reused.
    std::vector<std::weak_ptr<Executor>> executors;
    for (const auto id :
         std::array<Executor::ExecutorId, 3>{Executor::UNITARY_EXECUTOR_ID, 1, 2}) {
      auto executor = Executor::getExecutor(id);
      ASSERT_EQ(&getCatalog().getDataMgr(), executor->getDataMgr());
      executors.emplace_back(executor);
    }
    destroyDBHandler();
    for (const auto& executor : executors) {
      EXPECT_TRUE(executor.expired());
    }
    // Avoid dereferencing a stale executor after reporting the lifetime failure.
    ASSERT_FALSE(HasFailure());
    createDBHandler();
    sqlAndCompareResult("SELECT COUNT(*), SUM(id) FROM handler_lifecycle;",
                        {{i(generation), i(generation * (generation + 1) / 2)}});
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
  return RUN_ALL_TESTS();
}
