/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <thrift/TApplicationException.h>
#include <thrift/transport/TTransportException.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Calcite/Calcite.h"
#include "Tests/LogCaptureTestHelper.h"

namespace {

using Table = std::pair<std::string, std::string>;
using Tables = std::vector<Table>;
using namespace std::chrono_literals;

class MetadataCalcite : public Calcite {
 public:
  MetadataCalcite() : Calcite(0, 0, "", 0, 0, false) {}

  std::function<void(const std::string&, const std::string&)> update;

 private:
  void updateMetadataImpl(const std::string& catalog, const std::string& table) override {
    update(catalog, table);
  }
};

class PausedUpdate {
 public:
  void pause() {
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    condition_.notify_all();
    condition_.wait(lock, [this] { return released_; });
  }

  bool waitForEntry() {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(lock, 5s, [this] { return entered_; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool entered_{false};
  bool released_{false};
};

}  // namespace

class CalciteMetadataTest : public testing::Test {
 protected:
  void refresh() { calcite_.refreshTableStatistics(); }

  MetadataCalcite calcite_;
};

TEST_F(CalciteMetadataTest, CoalescesWritesAndLeavesCleanReadsAlone) {
  Tables updates;
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
  };
  refresh();
  EXPECT_TRUE(updates.empty());
  calcite_.markTableStatisticsDirty("db", "a");
  calcite_.markTableStatisticsDirty("db", "a");
  calcite_.markTableStatisticsDirty("db", "b");
  refresh();
  refresh();
  EXPECT_EQ((Tables{{"db", "a"}, {"db", "b"}}), updates);
}

TEST_F(CalciteMetadataTest, RefreshDoesNotBlockWritersOrOtherPlanners) {
  PausedUpdate paused;
  std::atomic<size_t> updates{0};
  calcite_.update = [&](const auto&, const auto&) {
    if (++updates == 1) {
      paused.pause();
    }
  };
  calcite_.markTableStatisticsDirty("db", "a");
  auto owner = std::async(std::launch::async, [&] { refresh(); });
  const bool entered = paused.waitForEntry();
  auto writer = std::async(std::launch::async, [&] {
    calcite_.markTableStatisticsDirty("db", "a");
    calcite_.markTableStatisticsDirty("db", "b");
  });
  auto planner = std::async(std::launch::async, [&] { refresh(); });
  const auto writer_status = writer.wait_for(5s);
  const auto planner_status = planner.wait_for(5s);
  // Release before asserting or joining so a regression cannot strand the workers.
  paused.release();
  owner.get();
  writer.get();
  planner.get();
  EXPECT_TRUE(entered);
  EXPECT_EQ(std::future_status::ready, writer_status);
  EXPECT_EQ(std::future_status::ready, planner_status);
  EXPECT_EQ(1u, updates.load());
  refresh();
  EXPECT_EQ(3u, updates.load());
}

TEST_F(CalciteMetadataTest, FailedRefreshRetainsOnlyUnfinishedTables) {
  LogCapture log;
  Tables updates;
  bool fail = true;
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
    if (table == "b" && fail) {
      fail = false;
      throw apache::thrift::TApplicationException("metadata update failed");
    }
  };
  for (const auto* table : {"a", "b", "c"}) {
    calcite_.markTableStatisticsDirty("db", table);
  }
  EXPECT_NO_THROW(refresh());
  EXPECT_TRUE(log.contains("Statistics refresh failed for db.b: metadata update failed"));
  EXPECT_EQ((Tables{{"db", "a"}, {"db", "b"}}), updates);
  refresh();
  refresh();
  EXPECT_EQ((Tables{{"db", "a"}, {"db", "b"}, {"db", "b"}, {"db", "c"}}), updates);
}

TEST_F(CalciteMetadataTest, FailedRefreshMergesConcurrentWrites) {
  PausedUpdate paused;
  Tables updates;
  bool fail = true;
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
    if (table == "b" && fail) {
      fail = false;
      paused.pause();
      throw apache::thrift::transport::TTransportException("metadata connection failed");
    }
  };
  for (const auto* table : {"a", "b", "c"}) {
    calcite_.markTableStatisticsDirty("db", table);
  }
  auto owner = std::async(std::launch::async, [&] { refresh(); });
  const bool entered = paused.waitForEntry();
  auto writer = std::async(std::launch::async, [&] {
    for (const auto* table : {"a", "b", "d"}) {
      calcite_.markTableStatisticsDirty("db", table);
    }
  });
  const auto writer_status = writer.wait_for(5s);
  paused.release();
  EXPECT_NO_THROW(owner.get());
  writer.get();
  EXPECT_TRUE(entered);
  EXPECT_EQ(std::future_status::ready, writer_status);
  refresh();
  refresh();
  EXPECT_EQ(
      (Tables{
          {"db", "a"}, {"db", "b"}, {"db", "a"}, {"db", "b"}, {"db", "c"}, {"db", "d"}}),
      updates);
}

TEST_F(CalciteMetadataTest, RepeatedRpcFailureStopsSnapshotAndRecovers) {
  Tables updates;
  bool fail = true;
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
    if (table == "b" && fail) {
      throw apache::thrift::TException("metadata connection failed");
    }
  };
  for (const auto* table : {"a", "b", "c"}) {
    calcite_.markTableStatisticsDirty("db", table);
  }
  calcite_.markTableStatisticsDirty("other", "d");
  EXPECT_NO_THROW(refresh());
  EXPECT_EQ((Tables{{"db", "a"}, {"db", "b"}}), updates);
  EXPECT_NO_THROW(refresh());
  EXPECT_EQ((Tables{{"db", "a"}, {"db", "b"}, {"db", "b"}}), updates);
  fail = false;
  EXPECT_NO_THROW(refresh());
  refresh();
  EXPECT_EQ((Tables{{"db", "a"},
                    {"db", "b"},
                    {"db", "b"},
                    {"db", "b"},
                    {"db", "c"},
                    {"other", "d"}}),
            updates);
}

TEST_F(CalciteMetadataTest, UnexpectedRefreshFailureStillPropagates) {
  Tables updates;
  bool fail = true;
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
    if (table == "b" && fail) {
      fail = false;
      throw std::runtime_error("unexpected failure");
    }
  };
  for (const auto* table : {"a", "b", "c"}) {
    calcite_.markTableStatisticsDirty("db", table);
  }
  EXPECT_THROW(refresh(), std::runtime_error);
  refresh();
  refresh();
  EXPECT_EQ((Tables{{"db", "a"}, {"db", "b"}, {"db", "b"}, {"db", "c"}}), updates);
}

TEST_F(CalciteMetadataTest, AllocationFailureStillPropagates) {
  calcite_.markTableStatisticsDirty("db", "a");
  calcite_.update = [&](const auto&, const auto&) { throw std::bad_alloc(); };
  EXPECT_THROW(refresh(), std::bad_alloc);
  Tables updates;
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
  };
  refresh();
  refresh();
  EXPECT_EQ((Tables{{"db", "a"}}), updates);
}

TEST_F(CalciteMetadataTest, TableDdlClearsOnlyItsPendingRefresh) {
  Tables updates;
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
  };
  calcite_.markTableStatisticsDirty("db", "a");
  calcite_.markTableStatisticsDirty("db", "b");
  calcite_.markTableStatisticsDirty("other", "a");
  calcite_.updateMetadata("db", "a");
  refresh();
  EXPECT_EQ((Tables{{"db", "a"}, {"db", "b"}, {"other", "a"}}), updates);
}

TEST_F(CalciteMetadataTest, DatabaseDdlClearsOnlyItsPendingRefreshes) {
  Tables updates;
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
  };
  calcite_.markTableStatisticsDirty("db", "a");
  calcite_.markTableStatisticsDirty("db", "b");
  calcite_.markTableStatisticsDirty("other", "a");
  calcite_.updateMetadata("db", "");
  refresh();
  EXPECT_EQ((Tables{{"db", ""}, {"other", "a"}}), updates);
}

TEST_F(CalciteMetadataTest, DdlDoesNotLoseConcurrentWritesOrBlockStatisticsReaders) {
  PausedUpdate paused;
  std::atomic<size_t> updates{0};
  calcite_.update = [&](const auto&, const auto&) {
    if (++updates == 1) {
      paused.pause();
    }
  };
  calcite_.markTableStatisticsDirty("db", "a");
  auto ddl = std::async(std::launch::async, [&] { calcite_.updateMetadata("db", "a"); });
  const bool entered = paused.waitForEntry();
  auto writer = std::async(std::launch::async,
                           [&] { calcite_.markTableStatisticsDirty("db", "a"); });
  auto planner = std::async(std::launch::async, [&] { refresh(); });
  const auto writer_status = writer.wait_for(5s);
  const auto planner_status = planner.wait_for(5s);
  paused.release();
  ddl.get();
  writer.get();
  planner.get();
  EXPECT_TRUE(entered);
  EXPECT_EQ(std::future_status::ready, writer_status);
  EXPECT_EQ(std::future_status::ready, planner_status);
  EXPECT_EQ(1u, updates.load());
  refresh();
  EXPECT_EQ(2u, updates.load());
}

TEST_F(CalciteMetadataTest, DatabaseDdlRpcFailureStillPropagates) {
  Tables updates;
  calcite_.update = [&](const auto&, const auto&) {
    throw apache::thrift::TApplicationException("metadata update failed");
  };
  calcite_.markTableStatisticsDirty("db", "a");
  calcite_.markTableStatisticsDirty("db", "b");
  calcite_.markTableStatisticsDirty("other", "a");
  EXPECT_THROW(calcite_.updateMetadata("db", ""), apache::thrift::TApplicationException);
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
  };
  refresh();
  refresh();
  EXPECT_EQ((Tables{{"db", "a"}, {"db", "b"}, {"other", "a"}}), updates);
}

TEST_F(CalciteMetadataTest, TableDdlRpcFailureStillPropagates) {
  calcite_.markTableStatisticsDirty("db", "a");
  calcite_.markTableStatisticsDirty("db", "b");
  calcite_.update = [&](const auto&, const auto&) {
    throw apache::thrift::transport::TTransportException("metadata connection failed");
  };
  EXPECT_THROW(calcite_.updateMetadata("db", "a"),
               apache::thrift::transport::TTransportException);
  Tables updates;
  calcite_.update = [&](const auto& db, const auto& table) {
    updates.emplace_back(db, table);
  };
  refresh();
  refresh();
  EXPECT_EQ((Tables{{"db", "a"}, {"db", "b"}}), updates);
}

TEST_F(CalciteMetadataTest, DdlStillWaitsForItsInvalidation) {
  PausedUpdate paused;
  std::atomic<size_t> updates{0};
  calcite_.update = [&](const auto&, const auto&) {
    if (++updates == 1) {
      paused.pause();
    }
  };
  calcite_.markTableStatisticsDirty("db", "a");
  auto owner = std::async(std::launch::async, [&] { refresh(); });
  const bool entered = paused.waitForEntry();
  std::promise<void> ddl_started;
  auto started = ddl_started.get_future();
  auto ddl = std::async(std::launch::async, [&] {
    ddl_started.set_value();
    calcite_.updateMetadata("db", "a");
  });
  const auto started_status = started.wait_for(5s);
  const auto ddl_status = ddl.wait_for(100ms);
  paused.release();
  owner.get();
  ddl.get();
  EXPECT_TRUE(entered);
  EXPECT_EQ(std::future_status::ready, started_status);
  EXPECT_EQ(std::future_status::timeout, ddl_status);
  EXPECT_EQ(2u, updates.load());
}
