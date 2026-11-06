/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <tuple>

#include "DBHandlerTestHelpers.h"
#include "LogCaptureTestHelper.h"
#include "TestHelpers.h"

using namespace TestHelpers;

extern float g_vacuum_min_selectivity;

class TableStatisticsTest : public DBHandlerTestFixture,
                            public testing::WithParamInterface<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    previous_rewrites_ = system_parameters_.enable_experimental_query_rewrites;
    previous_vacuum_selectivity_ = g_vacuum_min_selectivity;
    g_vacuum_min_selectivity = 1.1;
    system_parameters_.enable_experimental_query_rewrites = true;
    sql("DROP TABLE IF EXISTS statistics_maintenance;");
    const auto sharded = std::get<0>(GetParam());
    // Automatic vacuum is disabled so DELETE leaves physical slots until OPTIMIZE.
    // Small fragments exercise more than one fragment/shard.
    sql(std::string("CREATE TABLE statistics_maintenance (id INT") +
        (sharded ? ", SHARD KEY(id)" : "") + ") WITH (FRAGMENT_SIZE=2" +
        (sharded ? ", SHARD_COUNT=4" : "") + ");");
    sql("INSERT INTO statistics_maintenance VALUES (1),(2),(3),(4),(5),(6);");
    expectVisibleRows(6);
    expectPhysicalRows(6);
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS statistics_maintenance;");
    system_parameters_.enable_experimental_query_rewrites = previous_rewrites_;
    g_vacuum_min_selectivity = previous_vacuum_selectivity_;
    DBHandlerTestFixture::TearDown();
  }

  void expectVisibleRows(int64_t count) {
    sqlAndCompareResult("SELECT COUNT(*) FROM statistics_maintenance;", {{i(count)}});
  }

  void expectPhysicalRows(int64_t count) {
    auto [handler, session] = getDbHandlerAndSessionId();
    TTableDetails details;
    handler->get_table_details(details, session, "statistics_maintenance");
    ASSERT_TRUE(details.__isset.num_rows);
    EXPECT_EQ(count, details.num_rows);
  }

  void expectDeferredRefresh(LogCapture& log, int64_t count) {
    // The maintenance operation must not perform a metadata RPC under its locks.
    EXPECT_FALSE(log.contains(kMetadataRefresh));
    if (!system_parameters_.enable_experimental_query_rewrites) {
      expectVisibleRows(count);
      EXPECT_FALSE(log.contains(kMetadataRefresh));
      system_parameters_.enable_experimental_query_rewrites = true;
    }
    expectVisibleRows(count);
    EXPECT_TRUE(log.contains(kMetadataRefresh));
    expectPhysicalRows(count);
    expectVisibleRows(count);
    EXPECT_FALSE(log.contains(kMetadataRefresh));
  }

  // Observe the real Calcite invalidation rather than only a fresh native count.
  static constexpr const char* kMetadataRefresh = "Time to updateMetadata ";

 private:
  bool previous_rewrites_{};
  float previous_vacuum_selectivity_{};
};

TEST_P(TableStatisticsTest, TruncateInvalidatesBeforeNextRewrite) {
  system_parameters_.enable_experimental_query_rewrites = std::get<1>(GetParam());
  LogCapture log;
  sql("TRUNCATE TABLE statistics_maintenance;");
  expectDeferredRefresh(log, 0);

  sql("INSERT INTO statistics_maintenance VALUES (7),(8);");
  expectDeferredRefresh(log, 2);
}

TEST_P(TableStatisticsTest, VacuumInvalidatesOnlyWhenPhysicalCountChanges) {
  sql("DELETE FROM statistics_maintenance WHERE id <= 2;");
  expectVisibleRows(4);   // Drain DELETE's pending refresh before OPTIMIZE.
  expectPhysicalRows(6);  // Deleted slots still contribute to the estimate.

  system_parameters_.enable_experimental_query_rewrites = std::get<1>(GetParam());
  LogCapture log;
  sql("OPTIMIZE TABLE statistics_maintenance WITH (VACUUM='true');");
  expectDeferredRefresh(log, 4);

  sql("OPTIMIZE TABLE statistics_maintenance WITH (VACUUM='true');");
  expectVisibleRows(4);
  EXPECT_FALSE(log.contains(kMetadataRefresh));
}

INSTANTIATE_TEST_SUITE_P(NativeTables,
                         TableStatisticsTest,
                         testing::Combine(testing::Bool(), testing::Bool()));

int main(int argc, char** argv) {
  // INFO is needed for the metadata-RPC assertions, without changing logging
  // thresholds while the fixture's worker threads are running.
  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;
  log_options.parse_command_line(argc, argv);
  log_options.severity_clog_ = logger::Severity::INFO;
  logger::init(log_options);
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
  return RUN_ALL_TESTS();
}
