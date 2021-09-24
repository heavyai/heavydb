/*
 * Copyright 2020, OmniSci, Inc.
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
      ExecutionContext execution_context{
          .device_type = dt,
          .hash_join_enabled = overlaps_state,
      };
      QR::get()->clearGpuMemory();
      QR::get()->clearCpuMemory();
      fn(execution_context);
    }
  }
}

// clang-format off
const auto cleanup_stmts = {
    R"(drop table if exists does_intersect_a;)",
    R"(drop table if exists does_intersect_b;)",
    R"(drop table if exists does_not_intersect_a;)",
    R"(drop table if exists does_not_intersect_b;)",
    R"(drop table if exists empty_table;)"
};

// clang-format off
const auto init_stmts_ddl = {
    R"(create table does_intersect_a (id int,
                                      poly geometry(polygon, 4326),
                                      mpoly geometry(multipolygon, 4326),
                                      pt geometry(point, 4326));
    )",
    R"(create table does_intersect_b (id int,
                                      poly geometry(polygon, 4326),
                                      mpoly geometry(multipolygon, 4326),
                                      pt geometry(point, 4326),
                                      x DOUBLE,
                                      y DOUBLE);
    )",
    R"(create table does_not_intersect_a (id int,
                                        poly geometry(polygon, 4326),
                                        mpoly geometry(multipolygon, 4326),
                                        pt geometry(point, 4326));
    )",
    R"(create table does_not_intersect_b (id int,
                                        poly geometry(polygon, 4326),
                                        mpoly geometry(multipolygon, 4326),
                                        pt geometry(point, 4326));
    )",
    R"(create table empty_table (id int,
                           poly geometry(polygon, 4326),
                           mpoly geometry(multipolygon, 4326),
                           pt geometry(point, 4326));
    )"
};

const auto init_stmts_dml = {
    R"(insert into does_intersect_a
       values (0,
              'polygon((25 25,30 25,30 30,25 30,25 25))',
              'multipolygon(((25 25,30 25,30 30,25 30,25 25)))',
              'point(22 22)');
    )",
    R"(insert into does_intersect_a 
       values (1,
              'polygon((2 2,10 2,10 10,2 10,2 2))',
              'multipolygon(((2 2,10 2,10 10,2 10,2 2)))',
              'point(8 8)');
    )",
    R"(insert into does_intersect_a
       values (2,
              'polygon((2 2,10 2,10 10,2 10,2 2))',
              'multipolygon(((2 2,10 2,10 10,2 10,2 2)))',
              'point(8 8)');
    )",
    R"(insert into does_intersect_b
       values (0,
              'polygon((0 0,30 0,30 0,30 30,0 0))',
              'multipolygon(((0 0,30 0,30 0,30 30,0 0)))',
              'point(8 8)',
              8, 8);
    )",
    R"(insert into does_intersect_b
       values (1,
              'polygon((25 25,30 25,30 30,25 30,25 25))',
              'multipolygon(((25 25,30 25,30 30,25 30,25 25)))',
              'point(28 28)',
              28, 28);
    )",
    R"(insert into does_not_intersect_a
       values (1,
              'polygon((0 0,0 1,1 0,1 1,0 0))',
              'multipolygon(((0 0,0 1,1 0,1 1,0 0)))',
              'point(0 0)');
    )",
    R"(insert into does_not_intersect_a
       values (1,
              'polygon((0 0,0 1,1 0,1 1,0 0))',
              'multipolygon(((0 0,0 1,1 0,1 1,0 0)))',
              'point(0 0)');
    )",
    R"(insert into does_not_intersect_a
       values (1,
              'polygon((0 0,0 1,1 0,1 1,0 0))',
              'multipolygon(((0 0,0 1,1 0,1 1,0 0)))',
              'point(0 0)');
    )",
    R"(insert into does_not_intersect_b
       values (1,
              'polygon((2 2,2 4,4 2,4 4,2 2))',
              'multipolygon(((2 2,2 4,4 2,4 4,2 2)))',
              'point(2 2)');
    )"
};
// clang-format on

class OverlapsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    for (const auto& stmt : cleanup_stmts) {
      QR::get()->runDDLStatement(stmt);
    }

    for (const auto& stmt : init_stmts_ddl) {
      QR::get()->runDDLStatement(stmt);
    }

    for (const auto& stmt : init_stmts_dml) {
      QR::get()->runSQL(stmt, ExecutorDeviceType::CPU);
    }
  }

  static void TearDownTestSuite() {
    for (const auto& stmt : cleanup_stmts) {
      QR::get()->runDDLStatement(stmt);
    }
  }
};

TargetValue execSQL(const std::string& stmt,
                    const ExecutorDeviceType dt,
                    const bool geo_return_geo_tv = true) {
  auto rows = QR::get()->runSQL(stmt, dt, true, false);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << stmt;
  return crt_row[0];
}

TargetValue execSQLWithAllowLoopJoin(const std::string& stmt,
                                     const ExecutorDeviceType dt,
                                     const bool geo_return_geo_tv = true) {
  auto rows = QR::get()->runSQL(stmt, dt, true, true);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << stmt;
  return crt_row[0];
}

TEST_F(OverlapsTest, SimplePointInPolyIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    const auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_a WHERE ST_Intersects(poly, pt);";
    ASSERT_EQ(static_cast<int64_t>(2), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, InnerJoinPointInPolyIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON ST_Intersects(a.poly, "
        "b.pt);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, ctx.device_type)));

    sql =
        "SELECT count(*) from does_intersect_b as b JOIN "
        "does_intersect_a as a ON ST_Intersects(a.poly, b.pt);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, ctx.device_type)));

    sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON ST_Intersects(a.poly, "
        "ST_SetSRID(ST_Point(b.x, b.y), 4326));";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, ctx.device_type)));

    sql =
        "SELECT count(*) from does_intersect_b as b JOIN "
        "does_intersect_a as a ON ST_Intersects(a.poly, "
        "ST_SetSRID(ST_Point(b.x, b.y), 4326));";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

// TODO(jclay): This should succeed without failure.
// For now, we test against the (incorrect) failure.
TEST_F(OverlapsTest, InnerJoinPolyInPointIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    const auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON ST_Intersects(a.pt, "
        "b.poly);";
    if (g_enable_hashjoin_many_to_many) {
      EXPECT_ANY_THROW(execSQL(sql, ctx.device_type));
    } else {
      // Note(jclay): We return 0, postgis returns 4
      // Note(adb): Now we return 3. Progress?
      ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, ctx.device_type)));
    }
  });
}

TEST_F(OverlapsTest, InnerJoinPolyPolyIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Intersects(a.poly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, InnerJoinMPolyPolyIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Intersects(a.mpoly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, InnerJoinMPolyMPolyIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Intersects(a.mpoly, b.mpoly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, LeftJoinMPolyPolyIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  LEFT JOIN does_intersect_b as b
                  ON ST_Intersects(a.mpoly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, LeftJoinMPolyMPolyIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  LEFT JOIN does_intersect_b as b
                  ON ST_Intersects(a.mpoly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, InnerJoinPolyPolyIntersectsTranspose) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    const auto sql = R"(SELECT count(*) from does_intersect_a as a
                        JOIN does_intersect_b as b
                        ON ST_Intersects(b.poly, a.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, LeftJoinPolyPolyIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_b as b
                      LEFT JOIN does_intersect_a as a
                      ON ST_Intersects(a.poly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, LeftJoinPointInPolyIntersects) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                      LEFT JOIN does_intersect_b as b
                      ON ST_Intersects(b.poly, a.pt);)";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

// TODO(jclay): This should succeed without failure.
// Look into rewriting this in overlaps rewrite.
// For now, we test against the (incorrect) failure.
// It should return 3.
TEST_F(OverlapsTest, LeftJoinPointInPolyIntersectsWrongLHS) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                      LEFT JOIN does_intersect_b as b
                      ON ST_Intersects(a.poly, b.pt);)";
    if (g_enable_hashjoin_many_to_many) {
      EXPECT_ANY_THROW(execSQL(sql, ctx.device_type));
    } else {
      ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, ctx.device_type)));
    }
  });
}

TEST_F(OverlapsTest, InnerJoinPolyPolyContains) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_b as b
                  JOIN does_intersect_a as a
                  ON ST_Contains(a.poly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(0), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

// TODO(jclay): The following runtime functions are not implemented:
// - ST_Contains_MultiPolygon_MultiPolygon
// - ST_Contains_MultiPolygon_Polygon
// As a result, the following should succeed rather than throw error.
TEST_F(OverlapsTest, InnerJoinMPolyPolyContains) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Contains(a.mpoly, b.poly);)";
    EXPECT_ANY_THROW(execSQL(sql, ctx.device_type));
  });
}

TEST_F(OverlapsTest, InnerJoinMPolyMPolyContains) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Contains(a.mpoly, b.poly);)";
    // should return 4
    EXPECT_ANY_THROW(execSQL(sql, ctx.device_type));
  });
}

// NOTE(jclay): We don't support multipoly / poly ST_Contains
TEST_F(OverlapsTest, LeftJoinMPolyPolyContains) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_b as b
                  LEFT JOIN does_intersect_a as a
                  ON ST_Contains(a.mpoly, b.poly);)";
    // should return 4
    EXPECT_ANY_THROW(execSQL(sql, ctx.device_type));
  });
}

TEST_F(OverlapsTest, LeftJoinMPolyMPolyContains) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_b as b
                  LEFT JOIN does_intersect_a as a
                  ON ST_Contains(a.mpoly, b.mpoly);)";
    // should return 4
    EXPECT_ANY_THROW(execSQL(sql, ctx.device_type));
  });
}

TEST_F(OverlapsTest, JoinPolyPointContains) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON ST_Contains(a.poly, b.pt);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, ctx.device_type)));

    sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON "
        "ST_Contains(a.poly, ST_SetSRID(ST_Point(b.x, b.y), 4326));";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, ctx.device_type)));

    // sql =
    //     "SELECT "
    //     "count(*) from "
    //     "does_intersect_b as b JOIN does_intersect_a as a ON ST_Contains(a.pt,
    //     b.poly);";
    // ASSERT_EQ(static_cast<int64_t>(0), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, JoinPolyCentroidContains) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON "
        "ST_Contains(a.poly, ST_SetSRID(ST_Centroid(b.poly), 4326));";
    ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(execSQL(sql, ctx.device_type)));

    sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON "
        "ST_Contains(a.poly, ST_SetSRID(ST_Centroid(b.mpoly), 4326));";
    ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, PolyPolyDoesNotIntersect) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(execSQL("SELECT count(*) FROM does_not_intersect_b as b "
                                 "JOIN does_not_intersect_a as a "
                                 "ON ST_Intersects(a.poly, b.poly);",
                                 ctx.device_type)));
  });
}

TEST_F(OverlapsTest, EmptyPolyPolyJoin) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    const auto sql =
        "SELECT count(*) FROM does_not_intersect_a as a "
        "JOIN empty_table as b "
        "ON ST_Intersects(a.poly, b.poly);";
    ASSERT_EQ(static_cast<int64_t>(0), v<int64_t>(execSQL(sql, ctx.device_type)));
  });
}

TEST_F(OverlapsTest, SkipHashtableCaching) {
  const auto enable_overlaps_hashjoin_state = g_enable_overlaps_hashjoin;
  const auto enable_hashjoin_many_to_many_state = g_enable_hashjoin_many_to_many;

  g_enable_overlaps_hashjoin = true;
  g_enable_hashjoin_many_to_many = true;
  g_trivial_loop_join_threshold = 1;

  ScopeGuard reset_overlaps_state = [&enable_overlaps_hashjoin_state,
                                     &enable_hashjoin_many_to_many_state] {
    g_enable_overlaps_hashjoin = enable_overlaps_hashjoin_state;
    g_enable_overlaps_hashjoin = enable_hashjoin_many_to_many_state;
    g_trivial_loop_join_threshold = 1000;
  };

  QR::get()->clearCpuMemory();
  // check whether overlaps hashtable caching works properly
  const auto q1 =
      "SELECT count(*) FROM does_not_intersect_b as b JOIN does_not_intersect_a as a ON "
      "ST_Intersects(a.poly, b.poly);";
  execSQL(q1, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)2);

  const auto q2 =
      "SELECT /*+ overlaps_bucket_threshold(0.2), overlaps_no_cache */ count(*) FROM "
      "does_not_intersect_b as b JOIN does_not_intersect_a as a ON "
      "ST_Intersects(a.poly, b.poly);";
  execSQL(q2, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)2);

  QR::get()->clearCpuMemory();
  execSQL(q2, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)0);

  const auto q3 =
      "SELECT /*+ overlaps_no_cache */ count(*) FROM does_not_intersect_b as b JOIN "
      "does_not_intersect_a as a ON "
      "ST_Intersects(a.poly, b.poly);";
  execSQL(q3, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)0);

  const auto q4 =
      "SELECT /*+ overlaps_max_size(1000), overlaps_no_cache */ count(*) FROM "
      "does_not_intersect_b as b JOIN does_not_intersect_a as a ON "
      "ST_Intersects(a.poly, b.poly);";
  execSQL(q4, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)0);

  const auto q5 =
      "SELECT /*+ overlaps_bucket_threshold(0.2), overlaps_max_size(1000), "
      "overlaps_no_cache */ count(*) FROM does_not_intersect_b as b JOIN "
      "does_not_intersect_a as a ON "
      "ST_Intersects(a.poly, b.poly);";
  execSQL(q5, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)0);
}

TEST_F(OverlapsTest, CacheBehaviorUnderQueryHint) {
  // consider the following symbols:
  // T_E: bucket_threshold_hint_enabled
  // T_D: bucket_threshold_hint_disabled (use default value)
  // T_C: use calculated bucket_threshold value
  //      by performing auto tuner with an initial value of T_D
  // M_E: hashtable_max_size_hint_enabled
  // M_D: hashtable_max_size_hint_disabled (use default value)

  // here, we only add param setting to auto_tuner iff the initial setting is <T_D, *>
  // and replace the cached hashtable when we have already cached one
  // (and have different params)

  // let say a hashtable T is built from the setting C as C ----> T
  // then we reuse the hashtable T iff we have a cached hashtable which is mapped to C
  // all combinations of <chosen bucket_threshold, max_hashtable_size> are:
  // <T_E, M_E> --> impossible, we use <T_E, M_D> instead since we skip M_E and set M_D
  // <T_E, M_D> --> possible, but do not add the pair to auto_tuner_cache
  //                and map <T_E, M_D> ----> T to hashtable cache
  // <T_D, M_E> --> possible, and auto tuner calculates <T_C, M_D>
  //                add map <T_D, M_D> ----> <T_C, M_E> to auto_tuner_cache
  //                add map <T_C, M_E> ----> T to hashtable cache
  // <T_D, M_D> --> possible, and auto tuner calculates <T_C, M_D>
  //                add map <T_D, M_D> ----> <T_C, M_D> to auto_tuner_cache
  //                add map <T_C, M_D> ----> T to hashtable cache
  // <T_C, M_E> --> possible, and comes from the initial setting of <T_D, M_E>
  // <T_C, M_D> --> possible, and comes from the initial setting of <T_D, M_D>

  QR::get()->clearCpuMemory();
  const auto enable_overlaps_hashjoin_state = g_enable_overlaps_hashjoin;
  const auto enable_hashjoin_many_to_many_state = g_enable_hashjoin_many_to_many;

  g_enable_overlaps_hashjoin = true;
  g_enable_hashjoin_many_to_many = true;
  g_trivial_loop_join_threshold = 1;

  ScopeGuard reset_overlaps_state = [&enable_overlaps_hashjoin_state,
                                     &enable_hashjoin_many_to_many_state] {
    g_enable_overlaps_hashjoin = enable_overlaps_hashjoin_state;
    g_enable_overlaps_hashjoin = enable_hashjoin_many_to_many_state;
    g_trivial_loop_join_threshold = 1000;
  };

  // <T_D, M_D> case, add both <T_C, M_D> to auto tuner and its hashtable to cache
  const auto q1 =
      "SELECT count(*) FROM does_not_intersect_b as b JOIN does_not_intersect_a as a ON "
      "ST_Intersects(a.poly, b.poly);";
  execSQL(q1, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)2);

  // <T_E, M_D> case, only add hashtable to cache with <T_E: 0.1, M_D>
  const auto q2 =
      "SELECT /*+ overlaps_bucket_threshold(0.1) */ count(*) FROM does_not_intersect_b "
      "as b JOIN does_not_intersect_a as a ON ST_Intersects(a.poly, b.poly);";
  execSQL(q2, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)3);

  // <T_E, M_D> case... only add hashtable to cache with <T_E: 0.2, M_D>
  const auto q3 =
      "SELECT /*+ overlaps_bucket_threshold(0.2) */ count(*) FROM does_not_intersect_b "
      "as b JOIN does_not_intersect_a as a ON ST_Intersects(a.poly, b.poly);";
  execSQL(q3, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)4);

  // only reuse cached hashtable for <T_E: 0.1, M_D>
  const auto q4 =
      "SELECT /*+ overlaps_bucket_threshold(0.1) */ count(*) FROM does_not_intersect_b "
      "as b JOIN does_not_intersect_a as a ON ST_Intersects(a.poly, b.poly);";
  execSQL(q4, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)4);

  // skip max_size hint, so <T_E, M_D> case and only reuse <T_E: 0.1, M_D> hashtable
  const auto q5 =
      "SELECT /*+ overlaps_bucket_threshold(0.1), overlaps_max_size(1000) */ count(*) "
      "FROM does_not_intersect_b as b JOIN does_not_intersect_a as a ON "
      "ST_Intersects(a.poly, b.poly);";
  execSQL(q5, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)4);

  // <T_D, M_E> case, so it now becomes <T_C, M_E>
  // add <T_D, M_E> --> <T_C, M_E: 1000> mapping to auto_tuner
  // add <T_C, M_E: 1000> hashtable to cache
  const auto q6 =
      "SELECT /*+ overlaps_max_size(1000) */ count(*) FROM does_not_intersect_b as b "
      "JOIN does_not_intersect_a as a ON ST_Intersects(a.poly, b.poly);";
  execSQL(q6, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)6);

  // <T_E, M_D> case, only reuse cached hashtable of <T_E: 0.2, M_D>
  const auto q7 =
      "SELECT /*+ overlaps_max_size(1000), overlaps_bucket_threshold(0.2) */ count(*) "
      "FROM does_not_intersect_b as b JOIN does_not_intersect_a as a ON "
      "ST_Intersects(a.poly, b.poly);";
  execSQL(q7, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)6);

  // <T_E, M_D> case... only add hashtable to cache with <T_E: 0.3, M_D>
  const auto q8 =
      "SELECT /*+ overlaps_max_size(1000), overlaps_bucket_threshold(0.3) */ count(*) "
      "FROM does_not_intersect_b as b JOIN does_not_intersect_a as a ON "
      "ST_Intersects(a.poly, b.poly);";
  execSQL(q8, ExecutorDeviceType::CPU);
  ASSERT_EQ(QR::get()->getNumberOfCachedOverlapsHashTablesAndTuningParams(), (size_t)7);
}

class OverlapsJoinHashTableMock : public OverlapsJoinHashTable {
 public:
  struct ExpectedValues {
    size_t entry_count;
    size_t emitted_keys_count;
  };

  static std::shared_ptr<OverlapsJoinHashTableMock> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const int device_count,
      const RegisteredQueryHint& query_hint,
      const std::vector<OverlapsJoinHashTableMock::ExpectedValues>& expected_values) {
    auto hash_join = std::make_shared<OverlapsJoinHashTableMock>(condition,
                                                                 query_infos,
                                                                 memory_level,
                                                                 column_cache,
                                                                 executor,
                                                                 device_count,
                                                                 expected_values);
    hash_join->registerQueryHint(query_hint);
    hash_join->reifyWithLayout(HashType::OneToMany);
    return hash_join;
  }

  OverlapsJoinHashTableMock(const std::shared_ptr<Analyzer::BinOper> condition,
                            const std::vector<InputTableInfo>& query_infos,
                            const Data_Namespace::MemoryLevel memory_level,
                            ColumnCacheMap& column_cache,
                            Executor* executor,
                            const int device_count,
                            const std::vector<ExpectedValues>& expected_values)
      : OverlapsJoinHashTable(
            condition,
            JoinType::INVALID,  // b/c this is mock
            query_infos,
            memory_level,
            column_cache,
            executor,
            HashJoin::normalizeColumnPairs(condition.get(),
                                           *executor->getCatalog(),
                                           executor->getTemporaryTables()),
            device_count,
            EMPTY_QUERY_PLAN,
            {},
            {})
      , expected_values_per_step_(expected_values) {}

 protected:
  void reifyImpl(std::vector<ColumnsForDevice>& columns_per_device,
                 const Fragmenter_Namespace::TableInfo& query_info,
                 const HashType layout,
                 const size_t shard_count,
                 const size_t entry_count,
                 const size_t emitted_keys_count,
                 const bool skip_hashtable_caching,
                 const size_t chosen_max_hashtable_size,
                 const double chosen_bucket_threshold) final {
    EXPECT_LE(step_, expected_values_per_step_.size());
    auto& expected_values = expected_values_per_step_.back();
    EXPECT_EQ(entry_count, expected_values.entry_count);
    EXPECT_EQ(emitted_keys_count, expected_values.emitted_keys_count);
    return;
  }

  // returns entry_count, emitted_keys_count
  std::pair<size_t, size_t> approximateTupleCount(
      const std::vector<double>& bucket_sizes_for_dimension,
      std::vector<ColumnsForDevice>& columns_per_device,
      const size_t chosen_max_hashtable_size,
      const double chosen_bucket_threshold) final {
    auto [entry_count, emitted_keys_count] =
        OverlapsJoinHashTable::approximateTupleCount(bucket_sizes_for_dimension,
                                                     columns_per_device,
                                                     chosen_max_hashtable_size,
                                                     chosen_bucket_threshold);
    return std::make_pair(entry_count, emitted_keys_count);
  }

  // returns entry_count, emitted_keys_count
  std::pair<size_t, size_t> computeHashTableCounts(
      const size_t shard_count,
      const std::vector<double>& bucket_sizes_for_dimension,
      std::vector<ColumnsForDevice>& columns_per_device,
      const size_t chosen_max_hashtable_size,
      const double chosen_bucket_threshold) final {
    auto [entry_count, emitted_keys_count] =
        OverlapsJoinHashTable::computeHashTableCounts(shard_count,
                                                      bucket_sizes_for_dimension,
                                                      columns_per_device,
                                                      chosen_max_hashtable_size,
                                                      chosen_bucket_threshold);
    EXPECT_LT(step_, expected_values_per_step_.size());
    auto& expected_values = expected_values_per_step_[step_];
    EXPECT_EQ(entry_count, expected_values.entry_count);
    EXPECT_EQ(emitted_keys_count, expected_values.emitted_keys_count);
    step_++;
    return std::make_pair(entry_count, emitted_keys_count);
  }

  std::vector<ExpectedValues> expected_values_per_step_;
  size_t step_{0};
};

class BucketSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS bucket_size_poly;");
    QR::get()->runDDLStatement("CREATE TABLE bucket_size_poly (poly MULTIPOLYGON);");
    QR::get()->runSQL(
        R"(INSERT INTO bucket_size_poly VALUES ('MULTIPOLYGON(((0 0, 0 2, 2 0, 2 2)))');)",
        ExecutorDeviceType::CPU);
    QR::get()->runSQL(
        R"(INSERT INTO bucket_size_poly VALUES ('MULTIPOLYGON(((0 0, 0 2, 2 0, 2 2)))');)",
        ExecutorDeviceType::CPU);
    QR::get()->runSQL(
        R"(INSERT INTO bucket_size_poly VALUES ('MULTIPOLYGON(((2 2, 2 4, 4 2, 4 4)))');)",
        ExecutorDeviceType::CPU);
    QR::get()->runSQL(
        R"(INSERT INTO bucket_size_poly VALUES ('MULTIPOLYGON(((0 0, 0 50, 50 0, 50 50)))');)",
        ExecutorDeviceType::CPU);

    QR::get()->runDDLStatement("DROP TABLE IF EXISTS bucket_size_pt;");
    QR::get()->runDDLStatement("CREATE TABLE bucket_size_pt (pt POINT);");
  }

  void TearDown() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS bucket_size_poly;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS bucket_size_pt;");
  }

 public:
  static std::pair<std::shared_ptr<Analyzer::BinOper>, std::vector<InputTableInfo>>
  getOverlapsBuildInfo() {
    auto catalog = QR::get()->getCatalog();
    CHECK(catalog);

    std::vector<InputTableInfo> query_infos;

    const auto pts_td = catalog->getMetadataForTable("bucket_size_pt");
    CHECK(pts_td);
    const auto pts_cd = catalog->getMetadataForColumn(pts_td->tableId, "pt");
    CHECK(pts_cd);
    auto pt_col_var = std::make_shared<Analyzer::ColumnVar>(
        pts_cd->columnType, pts_cd->tableId, pts_cd->columnId, 0);
    query_infos.emplace_back(InputTableInfo{pts_td->tableId, build_table_info({pts_td})});

    const auto poly_td = catalog->getMetadataForTable("bucket_size_poly");
    CHECK(poly_td);
    const auto poly_cd = catalog->getMetadataForColumn(poly_td->tableId, "poly");
    CHECK(poly_cd);
    const auto bounds_cd =
        catalog->getMetadataForColumn(poly_td->tableId, poly_cd->columnId + 4);
    CHECK(bounds_cd && bounds_cd->columnType.is_array());
    auto poly_col_var = std::make_shared<Analyzer::ColumnVar>(
        bounds_cd->columnType, poly_cd->tableId, bounds_cd->columnId, 1);
    query_infos.emplace_back(
        InputTableInfo{poly_td->tableId, build_table_info({poly_td})});

    auto condition = std::make_shared<Analyzer::BinOper>(
        kBOOLEAN, kOVERLAPS, kANY, pt_col_var, poly_col_var);
    return std::make_pair(condition, query_infos);
  }
};

TEST_F(BucketSizeTest, OverlapsTunerEarlyOut) {
  // 2 steps, early out due to increasing keys per bin
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);
  auto executor = QR::get()->getExecutor();
  executor->setCatalog(catalog.get());

  auto [condition, query_infos] = BucketSizeTest::getOverlapsBuildInfo();

  ColumnCacheMap column_cache;
  std::vector<OverlapsJoinHashTableMock::ExpectedValues> expected_values;
  expected_values.emplace_back(
      OverlapsJoinHashTableMock::ExpectedValues{8, 7});  // step 1
  expected_values.emplace_back(
      OverlapsJoinHashTableMock::ExpectedValues{1340, 688});  // step 2
  expected_values.emplace_back(OverlapsJoinHashTableMock::ExpectedValues{
      1340, 688});  // increasing keys per bin, stop at step 2

  auto hash_table =
      OverlapsJoinHashTableMock::getInstance(condition,
                                             query_infos,
                                             Data_Namespace::MemoryLevel::CPU_LEVEL,
                                             column_cache,
                                             executor.get(),
                                             /*device_count=*/1,
                                             RegisteredQueryHint::defaults(),
                                             expected_values);
  CHECK(hash_table);
}

TEST_F(BucketSizeTest, OverlapsTooBig) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);
  auto executor = QR::get()->getExecutor();
  executor->setCatalog(catalog.get());

  auto [condition, query_infos] = BucketSizeTest::getOverlapsBuildInfo();

  ColumnCacheMap column_cache;
  std::vector<OverlapsJoinHashTableMock::ExpectedValues> expected_values;
  // runs 8 back tuner steps after initial size too big failure
  expected_values.emplace_back(
      OverlapsJoinHashTableMock::ExpectedValues{8, 7});  // step 1
  expected_values.emplace_back(
      OverlapsJoinHashTableMock::ExpectedValues{2, 4});  // step 2 (reversal)
  expected_values.emplace_back(OverlapsJoinHashTableMock::ExpectedValues{
      2, 4});  // step 3 (hash table not getting smaller, bails)

  RegisteredQueryHint hint;
  hint.overlaps_max_size = 2;
  hint.registerHint(QueryHint::kOverlapsMaxSize);
  EXPECT_ANY_THROW(
      OverlapsJoinHashTableMock::getInstance(condition,
                                             query_infos,
                                             Data_Namespace::MemoryLevel::CPU_LEVEL,
                                             column_cache,
                                             executor.get(),
                                             /*device_count=*/1,
                                             hint,
                                             expected_values));
}

void populateTablesForVarlenLinearizationTest() {
  {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfgeo;");  // non-null geo col val
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfgeo;");  // non-null geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS mfgeo_n;");  // contains null-valued geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS sfgeo_n;");  // contains null-valued geo col val
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfgeo3;");  // non-null geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS mfgeo3_n;");  // contains null-valued geo col val

    auto table_ddl =
        " (id INTEGER,\n"
        "  gpt GEOMETRY(POINT),\n"
        "  gpt4e GEOMETRY(POINT, 4326) ENCODING COMPRESSED(32),\n"
        "  gpt4n GEOMETRY(POINT, 4326) ENCODING NONE,\n"
        "  gl GEOMETRY(LINESTRING),\n"
        "  gl4e GEOMETRY(LINESTRING, 4326) ENCODING COMPRESSED(32),\n"
        "  gl4n GEOMETRY(LINESTRING, 4326) ENCODING NONE,\n"
        "  gp GEOMETRY(POLYGON),\n"
        "  gp4e GEOMETRY(POLYGON, 4326) ENCODING COMPRESSED(32),\n"
        "  gp4n GEOMETRY(POLYGON, 4326) ENCODING NONE,\n"
        "  gmp GEOMETRY(MULTIPOLYGON),\n"
        "  gmp4e GEOMETRY(MULTIPOLYGON, 4326) ENCODING COMPRESSED(32),\n"
        "  gmp4n GEOMETRY(MULTIPOLYGON, 4326) ENCODING NONE)";

    auto create_table_ddl_gen = [&table_ddl](const std::string& tbl_name,
                                             const bool multi_frag,
                                             const int fragment_size = 2) {
      std::ostringstream oss;
      oss << "CREATE TABLE " << tbl_name << table_ddl;
      if (multi_frag) {
        oss << " WITH (FRAGMENT_SIZE = " << fragment_size << ")";
      }
      oss << ";";
      return oss.str();
    };

    QR::get()->runDDLStatement(create_table_ddl_gen("mfgeo", true, 2));
    QR::get()->runDDLStatement(create_table_ddl_gen("sfgeo", false));
    QR::get()->runDDLStatement(create_table_ddl_gen("mfgeo_n", true, 2));
    QR::get()->runDDLStatement(create_table_ddl_gen("sfgeo_n", false));

    QR::get()->runDDLStatement(create_table_ddl_gen("mfgeo3", true, 3));
    QR::get()->runDDLStatement(create_table_ddl_gen("mfgeo3_n", true, 3));

    std::vector<std::vector<std::string>> input_val_non_nullable;
    input_val_non_nullable.push_back({"0",
                                      "\'POINT(0 0)\'",
                                      "\'LINESTRING(0 0,1 0)\'",
                                      "\'POLYGON((0 0,1 0,1 1,0 1,0 0))\'",
                                      "\'MULTIPOLYGON(((0 0,1 0,1 1,0 1,0 0)))\'"});
    input_val_non_nullable.push_back({"1",
                                      "\'POINT(1 1)\'",
                                      "\'LINESTRING(2 0,4 4)\'",
                                      "\'POLYGON((0 0,2 0,0 2,0 0))\'",
                                      "\'MULTIPOLYGON(((0 0,2 0,0 2,0 0)))\'"});
    input_val_non_nullable.push_back(
        {"2",
         "\'POINT(2 2)\'",
         "\'LINESTRING(1 0,2 2,3 3)\'",
         "\'POLYGON((0 0,2 0,3 0,3 3,0 3,1 0,0 0))\'",
         "\'MULTIPOLYGON(((0 0,2 0,3 0,3 3,0 3,1 0,0 0)))\'"});
    input_val_non_nullable.push_back({"3",
                                      "\'POINT(3 3)\'",
                                      "\'LINESTRING(3 0,6 6,7 7)\'",
                                      "\'POLYGON((0 0,4 0,0 4,0 0))\'",
                                      "\'MULTIPOLYGON(((0 0,4 0,0 4,0 0)))\'"});
    input_val_non_nullable.push_back(
        {"4",
         "\'POINT(4 4)\'",
         "\'LINESTRING(1 0,2 2,3 3)\'",
         "\'POLYGON((0 0,5 0,0 5,0 0))\'",
         "\'MULTIPOLYGON(((0 0,2 0,3 0,3 3,0 3,1 0,0 0)))\'"});
    input_val_non_nullable.push_back({"5",
                                      "\'POINT(5 5)\'",
                                      "\'LINESTRING(6 0, 12 12)\'",
                                      "\'POLYGON((0 0,6 0,0 6,0 0))\'",
                                      "\'MULTIPOLYGON(((0 0,6 0,0 6,0 0)))\'"});
    input_val_non_nullable.push_back(
        {"6",
         "\'POINT(6 6)\'",
         "\'LINESTRING(1 0,2 2,3 3)\'",
         "\'POLYGON((0 0,1 1,3 0,4 1,7 0,0 7,0 4,0 1,0 0))\'",
         "\'MULTIPOLYGON(((0 0,1 1,3 0,4 1,7 0,0 7,0 4,0 1,0 0)))\'"});
    input_val_non_nullable.push_back({"7",
                                      "\'POINT(7 7)\'",
                                      "\'LINESTRING(7 0,14 14,15 15)\'",
                                      "\'POLYGON((0 0,8 0,0 8,0 0))\'",
                                      "\'MULTIPOLYGON(((0 0,8 0,0 8,0 0)))\'"});
    input_val_non_nullable.push_back({"8",
                                      "\'POINT(8 8)\'",
                                      "\'LINESTRING(8 0,16 16)\'",
                                      "\'POLYGON((0 0,9 0,0 9,0 0))\'",
                                      "\'MULTIPOLYGON(((0 0,9 0,0 9,0 0)))\'"});
    input_val_non_nullable.push_back(
        {"9",
         "\'POINT(9 9)\'",
         "\'LINESTRING(9 0,18 18,19 19)\'",
         "\'POLYGON((0 0,5 0,10 0,10 5,10 10,0 10,0 3,0 1,0 0))\'",
         "\'MULTIPOLYGON(((0 0,5 0,10 0,10 5,10 10,0 10,0 3,0 1,0 0)))\'"});

    std::vector<std::vector<std::string>> input_val_nullable;
    input_val_nullable.push_back({"0",
                                  "\'POINT(0 0)\'",
                                  "\'NULL\'",
                                  "\'POLYGON((0 0,1 0,1 1,0 1,0 0))\'",
                                  "\'NULL\'"});
    input_val_nullable.push_back({"1",
                                  "\'POINT(1 1)\'",
                                  "\'LINESTRING(2 0,4 4)\'",
                                  "\'NULL\'",
                                  "\'MULTIPOLYGON(((0 0,2 0,0 2,0 0)))\'"});
    input_val_nullable.push_back({"2",
                                  "\'NULL\'",
                                  "\'LINESTRING(1 0,2 2,3 3)\'",
                                  "\'POLYGON((0 0,2 0,3 0,3 3,0 3,1 0,0 0))\'",
                                  "\'MULTIPOLYGON(((0 0,2 0,3 0,3 3,0 3,1 0,0 0)))\'"});
    input_val_nullable.push_back({"3",
                                  "\'POINT(3 3)\'",
                                  "\'LINESTRING(3 0,6 6,7 7)\'",
                                  "\'POLYGON((0 0,4 0,0 4,0 0))\'",
                                  "\'NULL\'"});
    input_val_nullable.push_back({"4",
                                  "\'POINT(4 4)\'",
                                  "\'LINESTRING(1 0,2 2,3 3)\'",
                                  "\'POLYGON((0 0,5 0,0 5,0 0))\'",
                                  "\'MULTIPOLYGON(((0 0,2 0,3 0,3 3,0 3,1 0,0 0)))\'"});
    input_val_nullable.push_back({"5",
                                  "\'NULL\'",
                                  "\'LINESTRING(6 0, 12 12)\'",
                                  "\'POLYGON((0 0,6 0,0 6,0 0))\'",
                                  "\'MULTIPOLYGON(((0 0,6 0,0 6,0 0)))\'"});
    input_val_nullable.push_back(
        {"6",
         "\'POINT(6 6)\'",
         "\'LINESTRING(1 0,2 2,3 3)\'",
         "\'POLYGON((0 0,1 1,3 0,4 1,7 0,0 7,0 4,0 1,0 0))\'",
         "\'MULTIPOLYGON(((0 0,1 1,3 0,4 1,7 0,0 7,0 4,0 1,0 0)))\'"});
    input_val_nullable.push_back({"7",
                                  "\'POINT(7 7)\'",
                                  "\'NULL\'",
                                  "\'POLYGON((0 0,8 0,0 8,0 0))\'",
                                  "\'MULTIPOLYGON(((0 0,8 0,0 8,0 0)))\'"});
    input_val_nullable.push_back({"8",
                                  "\'POINT(8 8)\'",
                                  "\'LINESTRING(8 0,16 16)\'",
                                  "\'NULL\'",
                                  "\'MULTIPOLYGON(((0 0,9 0,0 9,0 0)))\'"});
    input_val_nullable.push_back(
        {"9",
         "\'POINT(9 9)\'",
         "\'LINESTRING(9 0,18 18,19 19)\'",
         "\'POLYGON((0 0,5 0,10 0,10 5,10 10,0 10,0 3,0 1,0 0))\'",
         "\'MULTIPOLYGON(((0 0,5 0,10 0,10 5,10 10,0 10,0 3,0 1,0 0)))\'"});

    auto insert_stmt_gen = [](const std::vector<std::vector<std::string>>& input_col_vals,
                              const std::string& tbl_name) {
      for (auto& vec : input_col_vals) {
        int type_idx = 0;
        size_t num_rows = 0;
        std::ostringstream oss;
        oss << "INSERT INTO " << tbl_name << " VALUES(";
        std::vector<std::string> vals;
        for (auto& val : vec) {
          switch (type_idx) {
            case 0: {  // ID
              num_rows = 1;
              break;
            }
            case 1:    // POINT
            case 2:    // LINESTRING
            case 3:    // POLYGON
            case 4: {  // MULTIPOLYGON
              num_rows = 3;
              break;
            }
            default:
              break;
          }
          for (size_t i = 0; i < num_rows; i++) {
            vals.push_back(val);
          }
          type_idx++;
        }
        auto val_str = boost::join(vals, ",");
        oss << val_str << ");";
        QR::get()->runSQL(oss.str(), ExecutorDeviceType::CPU);
      }
    };

    insert_stmt_gen(input_val_non_nullable, "mfgeo");
    insert_stmt_gen(input_val_non_nullable, "sfgeo");
    insert_stmt_gen(input_val_nullable, "mfgeo_n");
    insert_stmt_gen(input_val_nullable, "sfgeo_n");

    insert_stmt_gen(input_val_non_nullable, "mfgeo3");
    insert_stmt_gen(input_val_nullable, "mfgeo3_n");

    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfgeo_n_v2;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfgeo_n_v2;");
    QR::get()->runDDLStatement(
        "CREATE TABLE sfgeo_n_v2 (pt GEOMETRY(POINT, 4326), l GEOMETRY(LINESTRING, "
        "4326), p GEOMETRY(POLYGON, 4326), mp GEOMETRY(MULTIPOLYGON, 4326));");
    QR::get()->runDDLStatement(
        "CREATE TABLE mfgeo_n_v2 (pt GEOMETRY(POINT, 4326), l GEOMETRY(LINESTRING, "
        "4326), p GEOMETRY(POLYGON, 4326), mp GEOMETRY(MULTIPOLYGON, 4326)) WITH "
        "(FRAGMENT_SIZE = 5);");

    auto insert_dml_specialized_null_test = [](const std::string& tbl_name) {
      std::vector<std::string> dml_vec;
      auto common_part = "INSERT INTO " + tbl_name + " VALUES (";
      auto null = "NULL,NULL,NULL,NULL";
      auto v = [](int i) { return i % 90; };
      auto gpt = [&v](int i) {
        std::ostringstream oss;
        oss << "\'POINT(" << v(i) << " " << v(i) << ")\'";
        return oss.str();
      };
      auto gl = [&v](int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "\'LINESTRING(0 0," << v(i) << " 0," << v(i + 2) << " " << v(i + 2)
              << "," << v(3 * i) << " " << v(2 * i) << ",0 " << v(i) << ")\'";
        } else if (i % 3 == 2) {
          oss << "\'LINESTRING(" << v(i) << " 0,0 " << v(i) << ")\'";
        } else {
          oss << "\'LINESTRING(0 0," << v(i) << " 0," << v(i) << " " << v(i) << ",0 "
              << v(i) << ")\'";
        }
        return oss.str();
      };
      auto gp = [&v](int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "\'POLYGON((0 0," << v(i) << " 0," << v(i + 2) << " " << v(i + 2) << ","
              << v(3 * i) << " " << v(2 * i) << ",0 " << v(i) << ",0 0))\'";
        } else if (i % 3 == 2) {
          oss << "\'POLYGON((0 0," << v(i) << " 0,0 " << v(i) << ",0 0))\'";
        } else {
          oss << "\'POLYGON((" << v(i) << " 0," << v(i) << " " << v(i) << ",0 " << v(i)
              << "))\'";
        }
        return oss.str();
      };
      auto gmp = [&v](int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "\'MULTIPOLYGON(((0 0," << v(i) << " 0," << v(i + 2) << " " << v(i + 2)
              << "," << v(3 * i) << " " << v(2 * i) << ",0 " << v(i) << ",0 0)))\'";
        } else if (i % 3 == 2) {
          oss << "\'MULTIPOLYGON(((0 0," << v(i) << " 0,0 " << v(i) << ",0 0)))\'";
        } else {
          oss << "\'MULTIPOLYGON(((" << v(i) << " 0," << v(i) << " " << v(i) << ",0 "
              << v(i) << ")))\'";
        }
        return oss.str();
      };
      auto g = [&gpt, &gl, &gp, &gmp, &null, &common_part](const int i, bool null_row) {
        std::ostringstream oss;
        oss << common_part;
        if (null_row) {
          oss << null << ");";
        } else {
          oss << gpt(i) << "," << gl(i) << "," << gp(i) << "," << gmp(i) << ");";
        }
        return oss.str();
      };
      int i = 1;
      // create length-5 bitstring corresponding to 0 ~ 31: 00000, 00001, ... , 11110,
      // 11111 where 1's position indicates a row having null value among five rows we
      // specify a fragment_size as five, so a chunk of bitstring 00010 means its fourth
      // row contains null value
      for (int chunk_idx = 0; chunk_idx < 32; chunk_idx++) {
        std::string str = std::bitset<5>(chunk_idx).to_string();
        for (int row_idx = 0; row_idx < 5; row_idx++, i++) {
          bool null_row = str.at(row_idx) == '1';
          if (null_row) {
            dml_vec.push_back(g(0, true));
          } else {
            dml_vec.push_back(g(i, false));
          }
        }
      }
      for (auto& insert_dml : dml_vec) {
        QR::get()->runSQL(insert_dml, ExecutorDeviceType::CPU);
      }
    };
    insert_dml_specialized_null_test("sfgeo_n_v2");
    insert_dml_specialized_null_test("mfgeo_n_v2");
  }

  {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfgeo_p;");  // non-null geo col val
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfgeo_p;");  // non-null geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS mfgeo_n_p;");  // contains null-valued geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS sfgeo_n_p;");  // contains null-valued geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS mfgeo_n2_p;");  // contains null-valued geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS sfgeo_n2_p;");  // contains null-valued geo col val

    auto table_ddl =
        " (pt GEOMETRY(POINT, 4326),\n"
        "  l GEOMETRY(LINESTRING, 4326),\n"
        "  p GEOMETRY(POLYGON, 4326),\n"
        "  mp GEOMETRY(MULTIPOLYGON, 4326))";

    auto create_table_ddl_gen = [&table_ddl](const std::string& tbl_name,
                                             const bool multi_frag,
                                             const int fragment_size = 100) {
      std::ostringstream oss;
      oss << "CREATE TABLE " << tbl_name << table_ddl;
      if (multi_frag) {
        oss << " WITH (FRAGMENT_SIZE = " << fragment_size << ")";
      }
      oss << ";";
      return oss.str();
    };

    QR::get()->runDDLStatement(create_table_ddl_gen("mfgeo_p", true));
    QR::get()->runDDLStatement(create_table_ddl_gen("sfgeo_p", false));
    QR::get()->runDDLStatement(create_table_ddl_gen("mfgeo_n_p", true));
    QR::get()->runDDLStatement(create_table_ddl_gen("sfgeo_n_p", false));
    QR::get()->runDDLStatement(create_table_ddl_gen("mfgeo_n2_p", true));
    QR::get()->runDDLStatement(create_table_ddl_gen("sfgeo_n2_p", false));

    auto insert_table = [](const std::string& tbl_name,
                           int num_tuples,
                           int frag_size,
                           bool allow_null,
                           bool first_frag_row_null) {
      std::vector<std::string> dml_vec;
      auto common_part = "INSERT INTO " + tbl_name + " VALUES (";
      auto null = "NULL,NULL,NULL,NULL";
      auto v = [](int i) { return i % 90; };
      auto gpt = [&v](int i) {
        std::ostringstream oss;
        oss << "\'POINT(" << v(i) << " " << v(i) << ")\'";
        return oss.str();
      };
      auto gl = [&v](int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "\'LINESTRING(0 0," << v(i) << " 0," << v(i + 2) << " " << v(i + 2)
              << "," << v(3 * i) << " " << v(2 * i) << ",0 " << v(i) << ")\'";
        } else if (i % 3 == 2) {
          oss << "\'LINESTRING(" << v(i) << " 0,0 " << v(i) << ")\'";
        } else {
          oss << "\'LINESTRING(0 0," << v(i) << " 0," << v(i) << " " << v(i) << ",0 "
              << v(i) << ")\'";
        }
        return oss.str();
      };
      auto gp = [&v](int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "\'POLYGON((0 0," << v(i) << " 0," << v(i + 2) << " " << v(i + 2) << ","
              << v(3 * i) << " " << v(2 * i) << ",0 " << v(i) << ",0 0))\'";
        } else if (i % 3 == 2) {
          oss << "\'POLYGON((0 0," << v(i) << " 0,0 " << v(i) << ",0 0))\'";
        } else {
          oss << "\'POLYGON((0 0," << v(i) << " 0," << v(i) << " " << v(i) << ",0 "
              << v(i) << ",0 0))\'";
        }
        return oss.str();
      };
      auto gmp = [&v](int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "\'MULTIPOLYGON(((0 0," << v(i) << " 0," << v(i + 2) << " " << v(i + 2)
              << "," << v(3 * i) << " " << v(2 * i) << ",0 " << v(i) << ",0 0)))\'";
        } else if (i % 3 == 2) {
          oss << "\'MULTIPOLYGON(((0 0," << v(i) << " 0,0 " << v(i) << ",0 0)))\'";
        } else {
          oss << "\'MULTIPOLYGON(((0 0," << v(i) << " 0," << v(i) << " " << v(i) << ",0 "
              << v(i) << ",0 0)))\'";
        }
        return oss.str();
      };
      auto g = [&gpt, &gl, &gp, &gmp, &null, &common_part](const int i, bool null_row) {
        std::ostringstream oss;
        oss << common_part;
        if (null_row) {
          oss << null << ");";
        } else {
          oss << gpt(i) << "," << gl(i) << "," << gp(i) << "," << gmp(i) << ");";
        }
        return oss.str();
      };

      if (allow_null) {
        for (int i = 0; i < num_tuples; i++) {
          if ((first_frag_row_null && (i % frag_size) == 0) || (i % 17 == 5)) {
            dml_vec.push_back(g(0, true));
          } else {
            dml_vec.push_back(g(i, false));
          }
        }
      } else {
        for (int i = 0; i < num_tuples; i++) {
          dml_vec.push_back(g(i, false));
        }
      }
      for (auto& insert_dml : dml_vec) {
        QR::get()->runSQL(insert_dml, ExecutorDeviceType::CPU);
      }
    };
    insert_table("sfgeo_p", 300, 100, false, false);
    insert_table("mfgeo_p", 300, 100, false, false);
    insert_table("sfgeo_n_p", 300, 100, true, true);
    insert_table("mfgeo_n_p", 300, 100, true, true);
    insert_table("sfgeo_n2_p", 300, 100, true, false);
    insert_table("mfgeo_n2_p", 300, 100, true, false);
  }
}

void dropTablesForVarlenLinearizationTest() {
  {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfgeo;");  // non-null geo col val
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfgeo;");  // non-null geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS mfgeo_n;");  // contains null-valued geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS sfgeo_n;");  // contains null-valued geo col val
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfgeo3;");  // non-null geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS mfgeo3_n;");  // contains null-valued geo col val
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfgeo_n_v2;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfgeo_n_v2;");
  }

  {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfgeo_p;");  // non-null geo col val
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfgeo_p;");  // non-null geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS mfgeo_n_p;");  // contains null-valued geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS sfgeo_n_p;");  // contains null-valued geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS mfgeo_n2_p;");  // contains null-valued geo col val
    QR::get()->runDDLStatement(
        "DROP TABLE IF EXISTS sfgeo_n2_p;");  // contains null-valued geo col val
  }
}

class MultiFragGeoOverlapsJoinTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(MultiFragGeoOverlapsJoinTest, Point) {
  // point - point by stwithin
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    std::vector<std::string> cols{"gpt", "gpt4e", "gpt4n"};
    int64_t single_frag_res = 10;
    auto execute_tests = [&](std::vector<std::string>& col_names) {
      for (auto& c : col_names) {
        std::ostringstream mq1, mq2;
        mq1 << "SELECT COUNT(1) FROM mfgeo s, mfgeo r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        mq2 << "SELECT COUNT(1) FROM mfgeo3 s, mfgeo3 r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        auto multi_frag_res1 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
        auto multi_frag_res2 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
        ASSERT_EQ(single_frag_res, multi_frag_res1) << mq1.str();
        ASSERT_EQ(single_frag_res, multi_frag_res2) << mq2.str();
      }
    };
    execute_tests(cols);
  });
}

TEST_F(MultiFragGeoOverlapsJoinTest, Linestring) {
  // linestring - polygon by st_intersect
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    std::vector<std::string> cols{"gl", "gl4e", "gl4n"};
    int64_t single_frag_res = 22;
    auto execute_tests = [&](std::vector<std::string>& col_names) {
      for (auto& c : col_names) {
        std::ostringstream mq1, mq2;
        mq1 << "SELECT COUNT(1) FROM mfgeo s, mfgeo r WHERE ST_INTERSECTS(s." << c
            << ", r." << c << ");";
        mq2 << "SELECT COUNT(1) FROM mfgeo3 s, mfgeo3 r WHERE ST_INTERSECTS(s." << c
            << ", r." << c << ");";
        auto multi_frag_res1 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
        auto multi_frag_res2 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
        ASSERT_EQ(single_frag_res, multi_frag_res1) << mq1.str();
        ASSERT_EQ(single_frag_res, multi_frag_res2) << mq2.str();
      }
    };
    execute_tests(cols);
  });
}

TEST_F(MultiFragGeoOverlapsJoinTest, Polygon) {
  // polygon - point by st_intersects
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    std::vector<std::string> cols{"gp", "gp4e", "gp4n"};
    int64_t single_frag_res = 100;
    auto execute_tests = [&](std::vector<std::string>& col_names) {
      for (auto& c : col_names) {
        std::ostringstream mq1, mq2;
        mq1 << "SELECT COUNT(1) FROM mfgeo s, mfgeo r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        mq2 << "SELECT COUNT(1) FROM mfgeo3 s, mfgeo3 r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        auto multi_frag_res1 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
        auto multi_frag_res2 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
        ASSERT_EQ(single_frag_res, multi_frag_res1) << mq1.str();
        ASSERT_EQ(single_frag_res, multi_frag_res2) << mq2.str();
      }
    };
    execute_tests(cols);
  });
}

TEST_F(MultiFragGeoOverlapsJoinTest, MultiPolygon) {
  // multipolygon - polygon by st_intersects
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    std::vector<std::string> cols{"gmp", "gmp4e", "gmp4n"};
    int64_t single_frag_res = 100;
    auto execute_tests = [&](std::vector<std::string>& col_names) {
      for (auto& c : col_names) {
        std::ostringstream mq1, mq2;
        mq1 << "SELECT COUNT(1) FROM mfgeo s, mfgeo r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        mq2 << "SELECT COUNT(1) FROM mfgeo3 s, mfgeo3 r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        auto multi_frag_res1 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
        auto multi_frag_res2 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
        ASSERT_EQ(single_frag_res, multi_frag_res1) << mq1.str();
        ASSERT_EQ(single_frag_res, multi_frag_res2) << mq2.str();
      }
    };
    execute_tests(cols);
  });
}

TEST_F(MultiFragGeoOverlapsJoinTest, Point_Nullable) {
  // point - point by stwithin
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    std::vector<std::string> cols{"gpt", "gpt4e", "gpt4n"};
    int64_t single_frag_res = 8;
    auto execute_tests = [&](std::vector<std::string>& col_names) {
      for (auto& c : col_names) {
        std::ostringstream mq1, mq2;
        mq1 << "SELECT COUNT(1) FROM mfgeo_n s, mfgeo_n r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        mq2 << "SELECT COUNT(1) FROM mfgeo3_n s, mfgeo3_n r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        auto multi_frag_res1 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
        auto multi_frag_res2 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
        ASSERT_EQ(single_frag_res, multi_frag_res1) << mq1.str();
        ASSERT_EQ(single_frag_res, multi_frag_res2) << mq2.str();
      }
    };
    execute_tests(cols);
  });
}

TEST_F(MultiFragGeoOverlapsJoinTest, Linestring_Nullable) {
  // linestring - polygon by st_intersect
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    std::vector<std::string> cols{"gl", "gl4e", "gl4n"};
    int64_t single_frag_res = 14;
    auto execute_tests = [&](std::vector<std::string>& col_names) {
      for (auto& c : col_names) {
        std::ostringstream mq1, mq2;
        mq1 << "SELECT COUNT(1) FROM mfgeo_n s, mfgeo_n r WHERE ST_INTERSECTS(s." << c
            << ", r." << c << ");";
        mq2 << "SELECT COUNT(1) FROM mfgeo3_n s, mfgeo3_n r WHERE ST_INTERSECTS(s." << c
            << ", r." << c << ");";
        auto multi_frag_res1 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
        auto multi_frag_res2 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
        ASSERT_EQ(single_frag_res, multi_frag_res1) << mq1.str();
        ASSERT_EQ(single_frag_res, multi_frag_res2) << mq2.str();
      }
    };
    execute_tests(cols);
  });
}

TEST_F(MultiFragGeoOverlapsJoinTest, Polygon_Nullable) {
  // polygon - point by st_intersects
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    std::vector<std::string> cols{"gp", "gp4e", "gp4n"};
    int64_t single_frag_res = 64;
    auto execute_tests = [&](std::vector<std::string>& col_names) {
      for (auto& c : col_names) {
        std::ostringstream mq1, mq2;
        mq1 << "SELECT COUNT(1) FROM mfgeo_n s, mfgeo_n r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        mq2 << "SELECT COUNT(1) FROM mfgeo3_n s, mfgeo3_n r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        auto multi_frag_res1 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
        auto multi_frag_res2 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
        ASSERT_EQ(single_frag_res, multi_frag_res1) << mq1.str();
        ASSERT_EQ(single_frag_res, multi_frag_res2) << mq2.str();
      }
    };
    execute_tests(cols);
  });
}

TEST_F(MultiFragGeoOverlapsJoinTest, MultiPolygon_Nullable) {
  // multipolygon - polygon by st_intersects
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    std::vector<std::string> cols{"gmp", "gmp4e", "gmp4n"};
    int64_t single_frag_res = 64;
    auto execute_tests = [&](std::vector<std::string>& col_names) {
      for (auto& c : col_names) {
        std::ostringstream mq1, mq2;
        mq1 << "SELECT COUNT(1) FROM mfgeo_n s, mfgeo_n r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        mq2 << "SELECT COUNT(1) FROM mfgeo3_n s, mfgeo3_n r WHERE ST_INTERSECTS(r." << c
            << ", s." << c << ");";
        auto multi_frag_res1 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
        auto multi_frag_res2 =
            v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
        ASSERT_EQ(single_frag_res, multi_frag_res1) << mq1.str();
        ASSERT_EQ(single_frag_res, multi_frag_res2) << mq2.str();
      }
    };
    execute_tests(cols);
  });
}

TEST_F(MultiFragGeoOverlapsJoinTest, Nullable_Geo_Exhaustive) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    int64_t single_frag_res1 = 114;
    int64_t single_frag_res2 = 5163;
    int64_t single_frag_res3 = 2178;
    int64_t single_frag_res4 = 2178;
    std::ostringstream mq1, mq2, mq3, mq4;
    mq1 << "SELECT COUNT(1) FROM mfgeo_n_v2 r, mfgeo_n_v2 s WHERE ST_INTERSECTS(r.pt, "
           "s.pt);";
    mq2 << "SELECT COUNT(1) FROM mfgeo_n_v2 r, mfgeo_n_v2 s WHERE ST_INTERSECTS(s.p, "
           "r.l);";
    mq3 << "SELECT COUNT(1) FROM mfgeo_n_v2 r, mfgeo_n_v2 s WHERE ST_INTERSECTS(r.p, "
           "s.pt);";
    mq4 << "SELECT COUNT(1) FROM mfgeo_n_v2 r, mfgeo_n_v2 s WHERE ST_INTERSECTS(r.mp, "
           "s.pt);";
    auto multi_frag_res1 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
    auto multi_frag_res2 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
    auto multi_frag_res3 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq3.str(), ctx.device_type));
    auto multi_frag_res4 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq4.str(), ctx.device_type));
    ASSERT_EQ(single_frag_res1, multi_frag_res1) << mq1.str();
    ASSERT_EQ(single_frag_res2, multi_frag_res2) << mq2.str();
    ASSERT_EQ(single_frag_res3, multi_frag_res3) << mq3.str();
    ASSERT_EQ(single_frag_res4, multi_frag_res4) << mq4.str();
  });
}

class ParallelLinearization : public ::testing::Test {
 protected:
  void SetUp() override { g_enable_parallel_linearization = 10; }
  void TearDown() override { g_enable_parallel_linearization = 20000; }
};

TEST_F(ParallelLinearization, GeoJoin) {
  executeAllScenarios([](const ExecutionContext ctx) -> void {
    int64_t single_frag_res1 = 1020;
    int64_t single_frag_res2 = 80940;
    int64_t single_frag_res3 = 38378;
    int64_t single_frag_res4 = 38378;
    std::ostringstream mq1, mq2, mq3, mq4;
    mq1 << "SELECT COUNT(1) FROM mfgeo_p r, mfgeo_p s WHERE ST_INTERSECTS(r.pt, s.pt);";
    mq2 << "SELECT COUNT(1) FROM mfgeo_p r, mfgeo_p s WHERE ST_INTERSECTS(s.p, r.l);";
    mq3 << "SELECT COUNT(1) FROM mfgeo_p r, mfgeo_p s WHERE ST_INTERSECTS(r.p, s.pt);";
    mq4 << "SELECT COUNT(1) FROM mfgeo_p r, mfgeo_p s WHERE ST_INTERSECTS(r.mp, s.pt);";
    auto multi_frag_res1 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
    auto multi_frag_res2 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
    auto multi_frag_res3 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq3.str(), ctx.device_type));
    auto multi_frag_res4 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq4.str(), ctx.device_type));
    ASSERT_EQ(single_frag_res1, multi_frag_res1) << mq1.str();
    ASSERT_EQ(single_frag_res2, multi_frag_res2) << mq2.str();
    ASSERT_EQ(single_frag_res3, multi_frag_res3) << mq3.str();
    ASSERT_EQ(single_frag_res4, multi_frag_res4) << mq4.str();
  });

  executeAllScenarios([](const ExecutionContext ctx) -> void {
    int64_t single_frag_res1 = 895;
    int64_t single_frag_res2 = 70115;
    int64_t single_frag_res3 = 33096;
    int64_t single_frag_res4 = 33096;
    std::ostringstream mq1, mq2, mq3, mq4;
    mq1 << "SELECT COUNT(1) FROM mfgeo_n_p r, mfgeo_n_p s WHERE ST_INTERSECTS(r.pt, "
           "s.pt);";
    mq2 << "SELECT COUNT(1) FROM mfgeo_n_p r, mfgeo_n_p s WHERE ST_INTERSECTS(s.p, r.l);";
    mq3 << "SELECT COUNT(1) FROM mfgeo_n_p r, mfgeo_n_p s WHERE ST_INTERSECTS(r.p, "
           "s.pt);";
    mq4 << "SELECT COUNT(1) FROM mfgeo_n_p r, mfgeo_n_p s WHERE ST_INTERSECTS(r.mp, "
           "s.pt);";
    auto multi_frag_res1 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
    auto multi_frag_res2 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
    auto multi_frag_res3 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq3.str(), ctx.device_type));
    auto multi_frag_res4 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq4.str(), ctx.device_type));
    ASSERT_EQ(single_frag_res1, multi_frag_res1) << mq1.str();
    ASSERT_EQ(single_frag_res2, multi_frag_res2) << mq2.str();
    ASSERT_EQ(single_frag_res3, multi_frag_res3) << mq3.str();
    ASSERT_EQ(single_frag_res4, multi_frag_res4) << mq4.str();
  });

  executeAllScenarios([](const ExecutionContext ctx) -> void {
    int64_t single_frag_res1 = 914;
    int64_t single_frag_res2 = 71556;
    int64_t single_frag_res3 = 33905;
    int64_t single_frag_res4 = 33905;
    std::ostringstream mq1, mq2, mq3, mq4;
    mq1 << "SELECT COUNT(1) FROM mfgeo_n2_p r, mfgeo_n2_p s WHERE ST_INTERSECTS(r.pt, "
           "s.pt);";
    mq2 << "SELECT COUNT(1) FROM mfgeo_n2_p r, mfgeo_n2_p s WHERE ST_INTERSECTS(s.p, "
           "r.l);";
    mq3 << "SELECT COUNT(1) FROM mfgeo_n2_p r, mfgeo_n2_p s WHERE ST_INTERSECTS(r.p, "
           "s.pt);";
    mq4 << "SELECT COUNT(1) FROM mfgeo_n2_p r, mfgeo_n2_p s WHERE ST_INTERSECTS(r.mp, "
           "s.pt);";
    auto multi_frag_res1 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq1.str(), ctx.device_type));
    auto multi_frag_res2 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq2.str(), ctx.device_type));
    auto multi_frag_res3 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq3.str(), ctx.device_type));
    auto multi_frag_res4 =
        v<int64_t>(execSQLWithAllowLoopJoin(mq4.str(), ctx.device_type));
    ASSERT_EQ(single_frag_res1, multi_frag_res1) << mq1.str();
    ASSERT_EQ(single_frag_res2, multi_frag_res2) << mq2.str();
    ASSERT_EQ(single_frag_res3, multi_frag_res3) << mq3.str();
    ASSERT_EQ(single_frag_res4, multi_frag_res4) << mq4.str();
  });
}

namespace range_join {

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

}  // namespace range_join
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
    for (const auto& stmt : range_join::cleanup_stmts) {
      QR::get()->runDDLStatement(stmt);
    }

    for (const auto& stmt : range_join::setup_stmts) {
      QR::get()->runDDLStatement(stmt);
    }

    for (const auto& stmt : range_join::insert_data_stmts) {
      QR::get()->runSQL(stmt, ExecutorDeviceType::CPU);
    }
  }

  static void TearDownTestSuite() {
    for (const auto& stmt : range_join::cleanup_stmts) {
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
  size_t expected_hash_tables{0};

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

  const size_t expected_hash_tables{0};

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
    populateTablesForVarlenLinearizationTest();
    err = RUN_ALL_TESTS();
    dropTablesForVarlenLinearizationTest();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  QR::reset();
  return err;
}
