/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <boost/process.hpp>
#include <future>
#include <thread>
#include <tuple>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TServerSocket.h>

#include "DBHandlerTestHelpers.h"
#include "LogCaptureTestHelper.h"
#include "Shared/heavyai_path.h"
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
    handler->get_internal_table_details_for_database(
        details, session, "statistics_maintenance", shared::kDefaultDbName);
    ASSERT_TRUE(details.__isset.num_rows);
    EXPECT_EQ(count, details.num_rows);
  }

  void expectDeferredRefresh(LogCapture& log, int64_t count) {
    // The mutation must not perform a metadata RPC under its locks.
    EXPECT_FALSE(log.contains(kMetadataRefresh));
    if (!system_parameters_.enable_experimental_query_rewrites) {
      expectVisibleRows(count);
      EXPECT_FALSE(log.contains(kMetadataRefresh));
      system_parameters_.enable_experimental_query_rewrites = true;
    }
    expectVisibleRows(count);
    EXPECT_TRUE(log.contains("(ms), catalog=\"" + getCatalog().getCurrentDB().dbName +
                             "\", table=\"statistics_maintenance\""));
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

TEST_P(TableStatisticsTest, CopyInvalidatesBeforeNextRewrite) {
  TempFile file;
  file << "7\n8\n9\n10\n";
  file.close();

  system_parameters_.enable_experimental_query_rewrites = std::get<1>(GetParam());
  LogCapture log;
  sql("COPY statistics_maintenance FROM '" + file.path() + "' WITH (header='false');");
  expectDeferredRefresh(log, 10);
  sqlAndCompareResult(
      "SELECT id FROM statistics_maintenance ORDER BY id;",
      {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}, {i(6)}, {i(7)}, {i(8)}, {i(9)}, {i(10)}});
  EXPECT_FALSE(log.contains(kMetadataRefresh));
}

TEST_P(TableStatisticsTest, InsertSelectInvalidatesDestinationBeforeNextRewrite) {
  sql("DROP TABLE IF EXISTS statistics_source;");
  ScopeGuard drop_source = [] { sql("DROP TABLE IF EXISTS statistics_source;"); };
  sql("CREATE TABLE statistics_source (id INT);");
  sql("INSERT INTO statistics_source VALUES (7),(8),(9),(10);");
  sqlAndCompareResult("SELECT COUNT(*) FROM statistics_source;", {{i(4)}});

  system_parameters_.enable_experimental_query_rewrites = std::get<1>(GetParam());
  LogCapture log;
  LogCapture source_log;
  sql("INSERT INTO statistics_maintenance SELECT id FROM statistics_source;");
  expectDeferredRefresh(log, 10);
  EXPECT_FALSE(source_log.contains("(ms), catalog=\"" +
                                   getCatalog().getCurrentDB().dbName +
                                   "\", table=\"statistics_source\""));
  sqlAndCompareResult("SELECT id FROM statistics_source ORDER BY id;",
                      {{i(7)}, {i(8)}, {i(9)}, {i(10)}});
  EXPECT_FALSE(log.contains(kMetadataRefresh));
}

TEST_P(TableStatisticsTest, FailedCopyRollsBackBeforeDeferredRefresh) {
  TempFile file;
  // With one worker and 16-byte buffers, earlier batches reach the fragmenter
  // before the malformed final row cancels the load.
  for (int id = 7; id <= 46; ++id) {
    file << id << '\n';
  }
  file << "47,48\n";  // Two fields for a one-column table.
  file.close();

  auto [handler, session] = getDbHandlerAndSessionId();
  const auto& catalog = getCatalog();
  const auto db_id = catalog.getDatabaseId();
  const auto table_id =
      catalog.getMetadataForTable("statistics_maintenance", false)->tableId;
  std::vector<TTableEpochInfo> saved_epochs;
  handler->get_table_epochs(saved_epochs, session, db_id, table_id);

  system_parameters_.enable_experimental_query_rewrites = std::get<1>(GetParam());
  LogCapture log;
  TQueryResult result;
  sql(result,
      "COPY statistics_maintenance FROM '" + file.path() +
          "' WITH (header='false', threads=1, buffer_size=16, max_reject=0);");
  ASSERT_EQ(1u, result.row_set.columns.size());
  const auto& messages = result.row_set.columns.front().data.str_col;
  ASSERT_EQ(1u, messages.size());
  EXPECT_EQ(0u, messages.front().find("Loader Failed due to :"));
  const auto status =
      getImportStatus(boost::filesystem::path(file.path()).filename().string());
  EXPECT_GT(status.rows_completed, 16);  // More rows than fit in the failing batch.
  EXPECT_EQ(1, status.rows_rejected);

  std::vector<TTableEpochInfo> restored_epochs;
  handler->get_table_epochs(restored_epochs, session, db_id, table_id);
  EXPECT_EQ(saved_epochs, restored_epochs);
  expectDeferredRefresh(log, 6);
  sqlAndCompareResult("SELECT id FROM statistics_maintenance ORDER BY id;",
                      {{i(1)}, {i(2)}, {i(3)}, {i(4)}, {i(5)}, {i(6)}});
  EXPECT_FALSE(log.contains(kMetadataRefresh));

  sql("INSERT INTO statistics_maintenance VALUES (7),(8);");
  expectDeferredRefresh(log, 8);
}

TEST_P(TableStatisticsTest, EpochRollbackInvalidatesBeforeNextRewrite) {
  enum class EpochApi { Batch, ById, ByName, Sql };
  auto [handler, session] = getDbHandlerAndSessionId();
  const auto& catalog = getCatalog();
  const auto db_id = catalog.getDatabaseId();
  const auto table_id =
      catalog.getMetadataForTable("statistics_maintenance", false)->tableId;
  for (const auto api :
       {EpochApi::Batch, EpochApi::ById, EpochApi::ByName, EpochApi::Sql}) {
    SCOPED_TRACE(testing::Message() << "epoch API " << static_cast<int>(api));
    std::vector<TTableEpochInfo> epochs;
    handler->get_table_epochs(epochs, session, db_id, table_id);
    ASSERT_FALSE(epochs.empty());
    const auto saved_epoch = epochs.front().table_epoch;
    for (const auto& epoch : epochs) {
      ASSERT_EQ(saved_epoch, epoch.table_epoch);
    }
    system_parameters_.enable_experimental_query_rewrites = true;
    sql("INSERT INTO statistics_maintenance VALUES (7),(8),(9),(10);");
    expectVisibleRows(10);  // Drain the insert's pending refresh before rollback.
    expectPhysicalRows(10);

    system_parameters_.enable_experimental_query_rewrites = std::get<1>(GetParam());
    LogCapture log;
    switch (api) {
      case EpochApi::Batch:
        handler->set_table_epochs(session, db_id, epochs);
        break;
      case EpochApi::ById:
        handler->set_table_epoch(session, db_id, table_id, saved_epoch);
        break;
      case EpochApi::ByName:
        handler->set_table_epoch_by_name(session, "statistics_maintenance", saved_epoch);
        break;
      case EpochApi::Sql:
        sql("ALTER TABLE statistics_maintenance SET EPOCH = " +
            std::to_string(saved_epoch) + ";");
        break;
    }
    expectDeferredRefresh(log, 6);
  }
}

TEST_P(TableStatisticsTest, FailedEpochBatchRetainsSuccessfulShardInvalidation) {
  auto [handler, session] = getDbHandlerAndSessionId();
  const auto& catalog = getCatalog();
  const auto db_id = catalog.getDatabaseId();
  const auto table_id =
      catalog.getMetadataForTable("statistics_maintenance", false)->tableId;
  std::vector<TTableEpochInfo> epochs;
  handler->get_table_epochs(epochs, session, db_id, table_id);
  ASSERT_FALSE(epochs.empty());
  const bool sharded = std::get<0>(GetParam());
  if (sharded) {
    ASSERT_EQ(4u, epochs.size());
    // Restore one shard, then fail before touching either remaining shard.
    epochs.resize(2);
  }
  epochs.back().table_epoch = -2;
  sql("INSERT INTO statistics_maintenance VALUES (7),(8),(9),(10);");
  expectVisibleRows(10);
  expectPhysicalRows(10);

  system_parameters_.enable_experimental_query_rewrites = std::get<1>(GetParam());
  LogCapture log;
  EXPECT_THROW(handler->set_table_epochs(session, db_id, epochs), std::runtime_error);
  if (sharded) {
    // The insert added one row per shard; only the first shard was rolled back.
    expectDeferredRefresh(log, 9);
  } else {
    // Failure before any successful epoch reset must not enqueue a refresh.
    EXPECT_FALSE(log.contains(kMetadataRefresh));
    system_parameters_.enable_experimental_query_rewrites = true;
    expectVisibleRows(10);
    expectPhysicalRows(10);
    EXPECT_FALSE(log.contains(kMetadataRefresh));
  }
}

INSTANTIATE_TEST_SUITE_P(NativeTables,
                         TableStatisticsTest,
                         testing::Combine(testing::Bool(), testing::Bool()));

namespace {
namespace ts = apache::thrift::server;
namespace tt = apache::thrift::transport;
namespace tp = apache::thrift::protocol;

class ListeningEvent : public ts::TServerEventHandler {
 public:
  void preServe() override { ready.set_value(); }
  std::promise<void> ready;
};

class PlannerMetadataTest : public DBHandlerTestFixture,
                            public testing::WithParamInterface<std::tuple<bool, bool>> {
 protected:
  void runProbe(const std::string& scenario) {
    const auto [rewrites, sharded] = GetParam();
    const auto old_rewrites = system_parameters_.enable_experimental_query_rewrites;
    const auto old_vacuum_selectivity = g_vacuum_min_selectivity;
    system_parameters_.enable_experimental_query_rewrites = rewrites;
    g_vacuum_min_selectivity = 1.1;
    ScopeGuard restore = [&] {
      system_parameters_.enable_experimental_query_rewrites = old_rewrites;
      g_vacuum_min_selectivity = old_vacuum_selectivity;
    };

    auto [handler, session] = getDbHandlerAndSessionId();
    // The fixture owns the handler. Keep it alive until all Thrift workers stop.
    auto processor = std::make_shared<HeavyProcessor>(
        std::shared_ptr<DBHandler>(handler, [](DBHandler*) {}));
    auto socket = std::make_shared<tt::TServerSocket>("127.0.0.1", 0);
    ts::TThreadedServer server(processor,
                               socket,
                               std::make_shared<tt::TBufferedTransportFactory>(),
                               std::make_shared<tp::TBinaryProtocolFactory>());
    auto events = std::make_shared<ListeningEvent>();
    auto ready = events->ready.get_future();
    server.setServerEventHandler(events);
    std::exception_ptr server_error;
    std::thread worker([&] {
      try {
        server.serve();
      } catch (...) {
        server_error = std::current_exception();
        try {
          events->ready.set_exception(server_error);
        } catch (const std::future_error&) {
          // Startup completed; report the server exception after joining below.
        }
      }
    });
    ScopeGuard stop = [&] {
      if (worker.joinable()) {
        server.stop();
        worker.join();
      }
    };
    ASSERT_EQ(std::future_status::ready, ready.wait_for(std::chrono::seconds(30)));
    ready.get();

    const auto root = heavyai::get_root_abs_path();
    const auto classpath = root + "/Tests/java:" + root +
                           "/bin/calcite-1.41.0-SNAPSHOT-jar-with-dependencies.jar";
    boost::process::child probe(boost::process::search_path("java"),
                                "-cp",
                                classpath,
                                "com.mapd.calcite.parser.PlannerMetadataProbe",
                                std::to_string(socket->getPort()),
                                shared::kDefaultDbName,
                                scenario,
                                rewrites ? "true" : "false",
                                sharded ? "true" : "false");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
    while (probe.running() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (probe.running()) {
      probe.terminate();
      probe.wait();
      FAIL() << "Java metadata probe timed out: " << scenario;
    }
    probe.wait();
    EXPECT_EQ(0, probe.exit_code()) << "Java metadata probe failed: " << scenario;
    server.stop();
    worker.join();
    if (server_error) {
      std::rethrow_exception(server_error);
    }
  }
};

TEST_P(PlannerMetadataTest, RowCountsReachJavaStatisticsOverThrift) {
  runProbe("rows");
}

TEST_P(PlannerMetadataTest, ForeignKeysReachJavaStatisticsOverThrift) {
  runProbe("keys");
}

INSTANTIATE_TEST_SUITE_P(NativeTables,
                         PlannerMetadataTest,
                         testing::Combine(testing::Bool(), testing::Bool()));
}  // namespace

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
