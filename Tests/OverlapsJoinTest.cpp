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

#include <gtest/gtest.h>
#include <functional>
#include <iostream>
#include <string>

#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/Execute.h"
#include "Shared/ConfigResolve.h"
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

template <typename TEST_BODY>
void executeAllScenarios(TEST_BODY fn) {
  for (const auto overlaps_state : {true, false}) {
    const auto enable_overlaps_hashjoin_state = g_enable_overlaps_hashjoin;
    g_enable_overlaps_hashjoin = overlaps_state;
    ScopeGuard reset_overlaps_state = [&enable_overlaps_hashjoin_state] {
      g_enable_overlaps_hashjoin = enable_overlaps_hashjoin_state;
    };
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      fn(dt);
    }
  }
}

// clang-format off
const auto init_stmts_ddl = {
    R"(drop table if exists does_intersect_a;)",
    R"(drop table if exists does_intersect_b;)",
    R"(drop table if exists does_not_intersect_a;)",
    R"(drop table if exists does_not_intersect_b;)",
    R"(create table does_intersect_a (id int,
                                      poly geometry(polygon, 4326),
                                      pt geometry(point, 4326));
    )",
    R"(create table does_intersect_b (id int,
                                      poly geometry(polygon, 4326),
                                      pt geometry(point, 4326));
    )",
    R"(create table does_not_intersect_a (id int,
                                        poly geometry(polygon, 4326),
                                        pt geometry(point, 4326));
    )",
    R"(create table does_not_intersect_b (id int,
                                        poly geometry(polygon, 4326),
                                        pt geometry(point, 4326));
    )"
};

const auto init_stmts_dml = {
    R"(insert into does_intersect_a
       values (0,
              'polygon((25 25,30 25,30 30,25 30,25 25))',
              'point(22 22)');
    )",
    R"(insert into does_intersect_a 
       values (1,
              'polygon((2 2,10 2,10 10,2 10,2 2))',
              'point(8 8)');
    )",
    R"(insert into does_intersect_a
       values (2,
              'polygon((2 2,10 2,10 10,2 10,2 2))',
              'point(8 8)');
    )",
    R"(insert into does_intersect_b
       values (0,
              'polygon((0 0,30 0,30 0,30 30,0 0))',
              'point(8 8)');
    )",
    R"(insert into does_intersect_b
       values (1,
              'polygon((25 25,30 25,30 30,25 30,25 25))',
              'point(28 28)');
    )",
    R"(insert into does_not_intersect_a
       values (1,
              'polygon((0 0,0 1,1 0,1 1,0 0))',
              'point(0 0)');
    )",
    R"(insert into does_not_intersect_a
       values (1,
              'polygon((0 0,0 1,1 0,1 1,0 0))',
              'point(0 0)');
    )"
};
// clang-format on

class OverlapsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    for (const auto& stmt : init_stmts_ddl) {
      QR::get()->runDDLStatement(stmt);
    }

    for (const auto& stmt : init_stmts_dml) {
      QR::get()->runSQL(stmt, ExecutorDeviceType::CPU);
    }
  }
};

TargetValue execSQL(const std::string& stmt,
                    const ExecutorDeviceType dt,
                    const bool geo_return_geo_tv = true) {
  auto rows = QR::get()->runSQL(stmt, dt);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << stmt;
  return crt_row[0];
}

TEST_F(OverlapsTest, SimplePointInPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    const auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_a WHERE ST_Intersects(poly, pt);";
    ASSERT_EQ(static_cast<int64_t>(2), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, InnerJoinPointInPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_a as a JOIN does_intersect_b as b ON ST_Intersects(a.poly, "
        "b.pt);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));

    sql =
        "SELECT count(*) from does_intersect_b as b JOIN "
        "does_intersect_a as a ON ST_Intersects(a.poly, b.pt);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));
  });
}

// TODO(jclay): we fail when point is given as the first arg.
// this passes in postgis.
TEST_F(OverlapsTest, DISABLED_InnerJoinPolyInPointIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    const auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_a as a JOIN does_intersect_b as b ON ST_Intersects(a.pt, "
        "b.poly);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, InnerJoinPolyPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Intersects(a.poly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, DISABLED_InnerJoinPolyPolyIntersectsTranspose) {
  // Note(jclay): We have a bug somewhere. This should == the result
  // of InnerJoinPolyPolyIntersects, but it does not.
  // have verified the expected value in Postgis
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    const auto sql = R"(SELECT count(*) from does_intersect_a as a
                        JOIN does_intersect_b as b
                        ON ST_Intersects(b.poly, a.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, LeftJoinPolyPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                      LEFT JOIN does_intersect_b as b
                      ON ST_Intersects(a.poly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

// TODO(jclay): Fails currently, I believe this gets fixed in my
// many to many overlaps branch.
TEST_F(OverlapsTest, DISABLED_JoinPolyPointContains) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_a as a JOIN does_intersect_b as b ON ST_Contains(a.poly, b.pt);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));

    sql =
        "SELECT "
        "count(*) from "
        "does_intersect_a as a JOIN does_intersect_b as b ON ST_Contains(a.pt, b.poly);";
    ASSERT_EQ(static_cast<int64_t>(0), v<int64_t>(execSQL(sql, dt)));
  });
}

// Note(jclay): This fails on master on GPU execution with overlaps hash join enabled.
TEST_F(OverlapsTest, PolyPolyDoesNotIntersect) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(execSQL("SELECT count(*) FROM does_not_intersect_a as a "
                                 "JOIN does_not_intersect_b as b "
                                 "ON ST_Intersects(a.poly, b.poly);",
                                 dt)));
  });
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  TestHelpers::init_logger_stderr_only(argc, argv);

  logger::LogOptions log_options(argv[0]);

  log_options.severity_ = logger::Severity::FATAL;
  // log_options.severity_clog_ = logger::DEBUG4;
  log_options.set_options();
  logger::init(log_options);

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
