#include "QueryRunner/QueryRunner.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <functional>
#include <iostream>
#include <string>

#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/Execute.h"
#include "Shared/scope.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

bool skip_tests_on_gpu(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests_on_gpu(dt)) {                               \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

struct ExecutionContext {
  ExecutorDeviceType device_type;
  bool hash_join_enabled;

  std::string toString() const {
    const auto device_str = device_type == ExecutorDeviceType::CPU ? "CPU" : "GPU";

    return fmt::format(
        "Execution Context:\n"
        "  Device Type: {}\n"
        "  Hash Join Enabled: {}\n",
        device_str,
        hash_join_enabled);
  }
};

template <typename TEST_BODY>
void executeAllScenarios(TEST_BODY fn) {
  for (const auto overlaps_state : {true, false}) {
    const auto enable_overlaps_hashjoin_state = g_enable_overlaps_hashjoin;
    const auto enable_hashjoin_many_to_many_state = g_enable_hashjoin_many_to_many;

    g_enable_overlaps_hashjoin = overlaps_state;
    g_enable_hashjoin_many_to_many = overlaps_state;
    g_trivial_loop_join_threshold = overlaps_state ? 1 : 1000;

    ScopeGuard reset_overlaps_state = [&enable_overlaps_hashjoin_state,
                                       &enable_hashjoin_many_to_many_state] {
      g_enable_overlaps_hashjoin = enable_overlaps_hashjoin_state;
      g_enable_overlaps_hashjoin = enable_hashjoin_many_to_many_state;
      g_trivial_loop_join_threshold = 1000;
    };

    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      ExecutionContext execCtx{
          .device_type = dt,
          .hash_join_enabled = overlaps_state,
      };
      QR::get()->clearGpuMemory();
      QR::get()->clearCpuMemory();
      fn(execCtx);
    }
  }
}

const auto setup_stmts = {
    "CREATE TABLE t1_comp32 ( p1 GEOMETRY(POINT, 4326) ENCODING COMPRESSED(32) );",
    "CREATE TABLE t2_comp32 ( p1 GEOMETRY(POINT, 4326) ENCODING COMPRESSED(32) );",
    "CREATE TABLE t1 ( p1 GEOMETRY(POINT, 4326) ENCODING NONE );",
    "CREATE TABLE t2 ( p1 GEOMETRY(POINT, 4326) ENCODING NONE );",
};

const auto insert_data_stmts = {
    "INSERT INTO t1_comp32 VALUES ( 'point(0.123 0.123)' );",
    "INSERT INTO t1_comp32 VALUES ( 'point(1.123 0.123)' );",
    "INSERT INTO t1_comp32 VALUES ( 'point(2.123 2.123)' );",
    "INSERT INTO t1_comp32 VALUES ( 'point(3.123 30.123)' );",

    "INSERT INTO t1 VALUES ( 'point(0.123 0.123)' );",
    "INSERT INTO t1 VALUES ( 'point(1.123 0.123)' );",
    "INSERT INTO t1 VALUES ( 'point(2.123 2.123)' );",
    "INSERT INTO t1 VALUES ( 'point(3.123 30.123)' );",

    "INSERT INTO t2_comp32 VALUES ( 'point(0.1 0.1)' );",
    "INSERT INTO t2_comp32 VALUES ( 'point(10.123 40.123)' );",
    "INSERT INTO t2_comp32 VALUES ( 'point(102.123 2.123)' );",
    "INSERT INTO t2_comp32 VALUES ( 'point(103.123 30.123)' );",

    "INSERT INTO t2 VALUES ( 'point(0.1 0.1)' );",
    "INSERT INTO t2 VALUES ( 'point(10.123 40.123)' );",
    "INSERT INTO t2 VALUES ( 'point(102.123 2.123)' );",
    "INSERT INTO t2 VALUES ( 'point(103.123 30.123)' );",
};

const auto cleanup_stmts = {
    "DROP TABLE IF EXISTS t1_comp32;",
    "DROP TABLE IF EXISTS t2_comp32;",
    "DROP TABLE IF EXISTS t1;",
    "DROP TABLE IF EXISTS t2;",
};

struct BoundsWithValues {
  double upper_bound;
  int expected_value;

  std::string toString() const {
    return fmt::format(
        "BoundsWithValues:\n"
        "  upper_bound: {}\n"
        "  expected_value: {}\n",
        upper_bound,
        expected_value);
  }
};

class RangeJoinTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    for (const auto& stmt : cleanup_stmts) {
      QR::get()->runDDLStatement(stmt);
    }

    for (const auto& stmt : setup_stmts) {
      QR::get()->runDDLStatement(stmt);
    }

    for (const auto& stmt : insert_data_stmts) {
      QR::get()->runSQL(stmt, ExecutorDeviceType::CPU);
    }
  }

  static void TearDownTestSuite() {
    for (const auto& stmt : cleanup_stmts) {
      QR::get()->runDDLStatement(stmt);
    }
  }

  std::vector<BoundsWithValues> testBounds() {
    return {
        BoundsWithValues{.upper_bound = 1, .expected_value = 1},
        BoundsWithValues{.upper_bound = 6.33, .expected_value = 3},
        BoundsWithValues{.upper_bound = 60, .expected_value = 8},
        BoundsWithValues{.upper_bound = 1000, .expected_value = 16},
    };
  };
};

TargetValue execSQL(const std::string& stmt,
                    const ExecutionContext ctx,
                    const bool geo_return_geo_tv = true) {
  auto rows = QR::get()->runSQL(stmt, ctx.device_type, true, false);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << stmt;
  return crt_row[0];
}

TargetValue execSQLWithAllowLoopJoin(const std::string& stmt,
                                     const ExecutionContext ctx,
                                     const bool geo_return_geo_tv = true) {
  auto rows = QR::get()->runSQL(stmt, ctx.device_type, true, true);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << stmt;
  return crt_row[0];
}

TEST_F(RangeJoinTest, DistanceLessThanEqCompressedCols) {
  executeAllScenarios([this](const ExecutionContext ctx) -> void {
    size_t expected_hash_tables = 0;

    if (ctx.hash_join_enabled) {
      ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
          << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
      expected_hash_tables++;
    }
    const auto tableA = "t1_comp32";
    const auto tableB = "t2_comp32";

    for (const auto& b : testBounds()) {
      auto sql = fmt::format(
          "SELECT count(*) FROM {}, {} where ST_Distance({}.p1, {}.p1) <= {};",
          tableA,
          tableB,
          tableA,
          tableB,
          b.upper_bound);

      ASSERT_EQ(int64_t(b.expected_value), v<int64_t>(execSQL(sql, ctx)))
          << fmt::format("Failed <= 1 \n{}\n{}", ctx.toString(), b.toString());

      if (ctx.hash_join_enabled) {
        ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
            << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
        expected_hash_tables++;
      }
    }
  });
}

TEST_F(RangeJoinTest, DistanceLessThanEqUnCompressedCols) {
  const auto tableA = "t1";
  const auto tableB = "t2";

  executeAllScenarios([&](const ExecutionContext ctx) -> void {
    size_t expected_hash_tables = 0;

    if (ctx.hash_join_enabled) {
      ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
          << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
      expected_hash_tables++;
    }

    for (const auto& b : testBounds()) {
      auto sql = fmt::format(
          "SELECT count(*) FROM {}, {} where ST_Distance({}.p1, {}.p1) <= {};",
          tableA,
          tableB,
          tableA,
          tableB,
          b.upper_bound);

      ASSERT_EQ(int64_t(b.expected_value), v<int64_t>(execSQL(sql, ctx)))
          << fmt::format("Failed <= 1 \n{}\n{}", ctx.toString(), b.toString());

      if (ctx.hash_join_enabled) {
        ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
            << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
        expected_hash_tables++;
      }
    }
  });
}

TEST_F(RangeJoinTest, DistanceLessThanMixedEncoding) {
  {
    const auto tableA = "t1_comp32";
    const auto tableB = "t2";

    executeAllScenarios([&](const ExecutionContext ctx) -> void {
      size_t expected_hash_tables = 0;

      if (ctx.hash_join_enabled) {
        ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
            << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
        expected_hash_tables++;
      }

      for (const auto& b : testBounds()) {
        auto sql = fmt::format(
            "SELECT count(*) FROM {}, {} where ST_Distance({}.p1, {}.p1) <= {};",
            tableA,
            tableB,
            tableA,
            tableB,
            b.upper_bound);

        ASSERT_EQ(int64_t(b.expected_value), v<int64_t>(execSQL(sql, ctx)))
            << fmt::format("Failed <= 1 \n{}\n{}", ctx.toString(), b.toString());

        if (ctx.hash_join_enabled) {
          ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(),
                    expected_hash_tables)
              << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
          expected_hash_tables++;
        }
      }
    });
  }

  {  // run same tests again, transpose LHS & RHS
    const auto tableA = "t2";
    const auto tableB = "t1_comp32";

    executeAllScenarios([&](const ExecutionContext ctx) -> void {
      size_t expected_hash_tables = 0;

      if (ctx.hash_join_enabled) {
        ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
            << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
        expected_hash_tables++;
      }

      for (const auto& b : testBounds()) {
        auto sql = fmt::format(
            "SELECT count(*) FROM {}, {} where ST_Distance({}.p1, {}.p1) <= {};",
            tableA,
            tableB,
            tableA,
            tableB,
            b.upper_bound);

        ASSERT_EQ(int64_t(b.expected_value), v<int64_t>(execSQL(sql, ctx)))
            << fmt::format("Failed <= 1 \n{}\n{}", ctx.toString(), b.toString());

        if (ctx.hash_join_enabled) {
          ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(),
                    expected_hash_tables)
              << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
          expected_hash_tables++;
        }
      }
    });
  }
}

TEST_F(RangeJoinTest, IsEnabledByDefault) {
  QR::get()->clearGpuMemory();
  QR::get()->clearCpuMemory();
  ExecutionContext ctx{
      .device_type = ExecutorDeviceType::CPU,
      .hash_join_enabled = true,
  };
  int expected_hash_tables{0};

  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
      << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
  expected_hash_tables++;

  const auto tableA = "t1_comp32";
  const auto tableB = "t2_comp32";

  for (const auto& b : testBounds()) {
    auto sql =
        fmt::format("SELECT count(*) FROM {}, {} where ST_Distance({}.p1, {}.p1) <= {};",
                    tableA,
                    tableB,
                    tableA,
                    tableB,
                    b.upper_bound);

    ASSERT_EQ(int64_t(b.expected_value), v<int64_t>(execSQL(sql, ctx)))
        << fmt::format("Failed <= 1 \n{}\n{}", ctx.toString(), b.toString());

    ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
        << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
    expected_hash_tables++;
  }
}

TEST_F(RangeJoinTest, CanBeDisabled) {
  QR::get()->clearGpuMemory();
  QR::get()->clearCpuMemory();
  g_enable_distance_rangejoin = false;

  ExecutionContext ctx{
      .device_type = ExecutorDeviceType::CPU,
      .hash_join_enabled = false,
  };

  const int expected_hash_tables{0};

  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
      << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());

  const auto tableA = "t1_comp32";
  const auto tableB = "t2_comp32";

  for (const auto& b : testBounds()) {
    auto sql =
        fmt::format("SELECT count(*) FROM {}, {} where ST_Distance({}.p1, {}.p1) <= {};",
                    tableA,
                    tableB,
                    tableA,
                    tableB,
                    b.upper_bound);

    ASSERT_EQ(int64_t(b.expected_value), v<int64_t>(execSQL(sql, ctx)))
        << fmt::format("Failed <= 1 \n{}\n{}", ctx.toString(), b.toString());

    ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTables(), expected_hash_tables)
        << fmt::format("Returned incorrect # of cached tables. {}", ctx.toString());
  }
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  QR::init(BASE_PATH);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  QR::reset();
  return err;
}
