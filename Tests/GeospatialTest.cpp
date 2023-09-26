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

#include "TestHelpers.h"

#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/StringTransform.h"
#include "Shared/clean_boost_regex.hpp"
#include "Shared/scope.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

const size_t g_num_rows{10};

bool g_keep_data{false};
bool g_all_utm_zones{false};
bool g_aggregator{false};
bool g_hoist_literals{true};

extern size_t g_leaf_count;
extern bool g_cluster;
extern bool g_is_test_env;
extern bool g_allow_cpu_retry;
extern bool g_allow_query_step_cpu_retry;

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

#define SKIP_ALL_ON_AGGREGATOR()                         \
  if (g_aggregator) {                                    \
    LOG(ERROR) << "Tests not valid in distributed mode"; \
    return;                                              \
  }

#define SKIP_ON_AGGREGATOR(EXP) \
  if (!g_aggregator) {          \
    EXP;                        \
  }

#define THROW_ON_AGGREGATOR(EXP) \
  if (!g_aggregator) {           \
    EXP;                         \
  } else {                       \
    EXPECT_ANY_THROW(EXP);       \
  }

#define EXPECT_GPU_THROW(EXP) \
  if (skip_tests(dt)) {       \
    EXPECT_ANY_THROW(EXP);    \
  } else {                    \
    EXP;                      \
  }

namespace {

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type,
                                            const bool allow_loop_joins) {
  return QR::get()->runSQL(query_str, device_type, g_hoist_literals, allow_loop_joins);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, g_hoist_literals, true);
}

TargetValue run_simple_agg(const std::string& query_str,
                           const ExecutorDeviceType device_type,
                           const bool geo_return_geo_tv = true,
                           const bool allow_loop_joins = true) {
  auto rows =
      QR::get()->runSQL(query_str, device_type, g_hoist_literals, allow_loop_joins);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
}

TargetValue get_first_target(const std::string& query_str,
                             const ExecutorDeviceType device_type,
                             const bool geo_return_geo_tv = true) {
  auto rows = run_multiple_agg(query_str, device_type);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_GE(crt_row.size(), size_t(1)) << query_str;
  return crt_row[0];
}

void import_geospatial_test(const bool use_temporary_tables) {
  const std::string geospatial_test("DROP TABLE IF EXISTS geospatial_test;");
  run_ddl_statement(geospatial_test);
  const auto create_ddl = build_create_table_statement(
      R"(id INT, p POINT, p_null POINT, mp MULTIPOINT, mp_null MULTIPOINT, l LINESTRING, l_null LINESTRING, ml MULTILINESTRING, ml2 MULTILINESTRING, ml_null MULTILINESTRING, poly POLYGON, poly_null POLYGON, mpoly MULTIPOLYGON, mpoly2 MULTIPOLYGON, mpoly_null MULTIPOLYGON, gp GEOMETRY(POINT), gp4326 GEOMETRY(POINT,4326) ENCODING COMPRESSED(32), gp4326none GEOMETRY(POINT,4326) ENCODING NONE, gp900913 GEOMETRY(POINT,900913), gmp4326 GEOMETRY(MULTIPOINT,4326), gl4326none GEOMETRY(LINESTRING,4326) ENCODING NONE, gml4326 GEOMETRY(MULTILINESTRING,4326), gpoly4326 GEOMETRY(POLYGON,4326), gpoly900913 GEOMETRY(POLYGON,900913))",
      "geospatial_test",
      {"", 0},
      {},
      2,
      /*use_temporary_tables=*/use_temporary_tables,
      /*deleted_support=*/true,
      /*is_replicated=*/false);
  run_ddl_statement(create_ddl);
  TestHelpers::ValuesGenerator gen("geospatial_test");
  for (size_t i = 0; i < g_num_rows; ++i) {
    const std::string point{"'POINT(" + std::to_string(i) + " " + std::to_string(i) +
                            ")'"};
    const std::string point_null{"'NULL'"};
    const std::string multipoint{
        "'MULTIPOINT(" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        ")'"};
    const std::string multipoint_null{"'NULL'"};
    const std::string linestring{
        "'LINESTRING(" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        ")'"};
    const std::string linestring_null{"'NULL'"};
    const std::string multilinestring{
        "'MULTILINESTRING((" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        "))'"};
    const std::string multilinestring2{
        "'MULTILINESTRING((" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        "), (" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        "))'"};
    const std::string multilinestring_null{"'NULL'"};
    const std::string poly{"'POLYGON((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                           std::to_string(i + 1) + ", 0 0))'"};
    const std::string poly_null{"'NULL'"};
    const std::string mpoly{"'MULTIPOLYGON(((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                            std::to_string(i + 1) + ", 0 0)))'"};
    const std::string mpoly2{"'MULTIPOLYGON(((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                             std::to_string(i + 1) + ", 0 0)), ((0 0, " +
                             std::to_string(i + 1) + " 0, 0 " + std::to_string(i + 1) +
                             ", 0 0)))'"};
    const std::string mpoly_null{"'NULL'"};
    run_multiple_agg(gen(i,
                         point,
                         point_null,
                         multipoint,
                         multipoint_null,
                         linestring,
                         linestring_null,
                         multilinestring,
                         multilinestring2,
                         multilinestring_null,
                         poly,
                         poly_null,
                         mpoly,
                         mpoly2,
                         mpoly_null,
                         point,
                         point,
                         point,
                         point,
                         multipoint,
                         linestring,
                         multilinestring,
                         poly,
                         poly),
                     ExecutorDeviceType::CPU);
  }
}

void import_geospatial_join_test(const bool use_temporary_tables) {
  // Create a single fragment inner table that is half the size of the geospatial_test
  // (outer) table
  const std::string drop_geospatial_test(
      "DROP TABLE IF EXISTS geospatial_inner_join_test;");
  run_ddl_statement(drop_geospatial_test);
  std::string column_definition =
      "id INT, p POINT, l LINESTRING, poly POLYGON, gp4326 GEOMETRY(POLYGON, 4326), "
      "mpoly MULTIPOLYGON";
  auto create_statement =
      build_create_table_statement(column_definition,
                                   "geospatial_inner_join_test",
                                   {"", 0},
                                   {},
                                   20,
                                   /*use_temporary_tables=*/use_temporary_tables,
                                   /*deleted_support=*/true,
                                   g_aggregator);
  run_ddl_statement(create_statement);
  TestHelpers::ValuesGenerator gen("geospatial_inner_join_test");
  for (size_t i = 0; i < g_num_rows; i += 2) {
    const std::string point{"'POINT(" + std::to_string(i) + " " + std::to_string(i) +
                            ")'"};
    const std::string linestring{
        "'LINESTRING(" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        ")'"};
    const std::string poly{"'POLYGON((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                           std::to_string(i + 1) + ", 0 0))'"};
    const std::string mpoly{"'MULTIPOLYGON(((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                            std::to_string(i + 1) + ", 0 0)))'"};
    run_multiple_agg(gen(i, point, linestring, poly, poly, mpoly),
                     ExecutorDeviceType::CPU);
  }
}

void import_geospatial_null_test(const bool use_temporary_tables) {
  const std::string geospatial_null_test("DROP TABLE IF EXISTS geospatial_null_test;");
  run_ddl_statement(geospatial_null_test);
  const auto create_ddl = build_create_table_statement(
      "id INT, p POINT, mp MULTIPOINT, l LINESTRING, ml MULTILINESTRING, "
      "poly POLYGON, mpoly MULTIPOLYGON, gpnotnull GEOMETRY(POINT) NOT NULL, "
      "gp4326 GEOMETRY(POINT,4326) ENCODING COMPRESSED(32), "
      "gp4326none GEOMETRY(POINT,4326) ENCODING NONE, "
      "gp900913 GEOMETRY(POINT,900913), gmp4326 GEOMETRY(MULTIPOINT,4326), "
      "gl4326none GEOMETRY(LINESTRING,4326) ENCODING NONE, "
      "gml4326 GEOMETRY(MULTILINESTRING,4326), gpoly4326 GEOMETRY(POLYGON,4326)",
      "geospatial_null_test",
      {"", 0},
      {},
      2,
      /*use_temporary_tables=*/use_temporary_tables,
      /*deleted_support=*/true,
      /*is_replicated=*/false);
  run_ddl_statement(create_ddl);
  TestHelpers::ValuesGenerator gen("geospatial_null_test");
  for (size_t i = 0; i < g_num_rows; ++i) {
    const std::string point{"'POINT(" + std::to_string(i) + " " + std::to_string(i) +
                            ")'"};
    const std::string multipoint{
        "'MULTIPOINT(" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        ")'"};
    const std::string linestring{
        "'LINESTRING(" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        ")'"};
    const std::string multilinestring{
        "'MULTILINESTRING((" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        "))'"};
    const std::string poly{"'POLYGON((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                           std::to_string(i + 1) + ", 0 0))'"};
    const std::string mpoly{"'MULTIPOLYGON(((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                            std::to_string(i + 1) + ", 0 0)))'"};
    run_multiple_agg(gen(i,
                         (i % 2 == 0) ? "NULL" : point,
                         (i % 2 == 0) ? "NULL" : multipoint,
                         (i == 1) ? "NULL" : linestring,
                         (i == 7) ? "NULL" : multilinestring,
                         (i == 2) ? "'NULL'" : poly,
                         (i == 3) ? "NULL" : mpoly,
                         point,
                         (i == 4) ? "NULL" : point,
                         (i == 5) ? "NULL" : point,
                         (i == 6) ? "NULL" : point,
                         (i == 6) ? "NULL" : multipoint,
                         (i == 7) ? "NULL" : linestring,
                         (i == 1) ? "NULL" : multilinestring,
                         (i == 8) ? "NULL" : poly),
                     ExecutorDeviceType::CPU);
  }
}

void import_geospatial_multi_frag_test(const bool use_temporary_tables) {
  const std::string geospatial_multi_frag_test(
      "DROP TABLE IF EXISTS geospatial_multi_frag_test;");
  run_ddl_statement(geospatial_multi_frag_test);
  const auto create_ddl = build_create_table_statement(
      "pt geometry(point, 4326), pt_none geometry(point, 4326) encoding none, pt_comp "
      "geometry(point, 4326) encoding compressed(32)",
      "geospatial_multi_frag_test",
      {"", 0},
      {},
      2,
      /*use_temporary_tables=*/use_temporary_tables,
      /*deleted_support=*/true,
      /*is_replicated=*/false);
  run_ddl_statement(create_ddl);
  TestHelpers::ValuesGenerator gen("geospatial_multi_frag_test");
  for (size_t i = 0; i < 11; ++i) {
    const std::string point{"'POINT(" + std::to_string(i) + " " + std::to_string(i) +
                            ")'"};
    run_multiple_agg(gen(point, point, point), ExecutorDeviceType::CPU);
  }
  run_multiple_agg("insert into geospatial_multi_frag_test values (NULL, NULL, NULL)",
                   ExecutorDeviceType::CPU);
}

}  // namespace

class GeoSpatialTestTablesFixture : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override { import_geospatial_test(/*with_temporary_tables=*/GetParam()); }

  void TearDown() override {
    if (!GetParam() && !g_keep_data) {
      run_ddl_statement("DROP TABLE IF EXISTS geospatial_test;");
    }
  }
};

TEST_P(GeoSpatialTestTablesFixture, Basics) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test where p IS NOT NULL;", dt)));
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test where poly IS NULL;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(p,p) < 0.1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT count(p) FROM geospatial_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT count(mp) FROM geospatial_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT count(l) FROM geospatial_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT count(ml) FROM geospatial_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT count(poly) FROM geospatial_test;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg("SELECT count(mpoly) FROM geospatial_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Distance('POINT(0 0)', p) < 100.0;",
                                        dt)));
    ASSERT_EQ(
        static_cast<int64_t>(7),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                  "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 9;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(mp,p) <= 2.0;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(4),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                  "WHERE ST_Distance('MULTIPOINT(-1 0, 0 1)', p) < 3.8;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                  "WHERE ST_Distance('MULTIPOINT(-1 0, 0 1)', p) < 1.1;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                  "WHERE ST_Distance(p, 'MULTIPOINT(-1 0, 0 1)') < 2.5;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(5),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(l,p) <= 2.0;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                  "WHERE ST_Distance('LINESTRING(-1 0, 0 1)', p) < 0.8;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                  "WHERE ST_Distance('LINESTRING(-1 0, 0 1)', p) < 1.1;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                  "WHERE ST_Distance(p, 'LINESTRING(-1 0, 0 1)') < 2.5;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(5),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(ml,p) <= 2.0;", dt)));
    // yet unsupported native implementations
    EXPECT_THROW(
        run_simple_agg("SELECT ST_Distance(ml,ml) FROM geospatial_test limit 1;", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_simple_agg("SELECT ST_Distance(ml,poly) FROM geospatial_test limit 1;", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_simple_agg("SELECT ST_Distance(mpoly,ml) FROM geospatial_test limit 1;", dt),
        std::runtime_error);
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test "
                  "WHERE ST_Distance('MULTILINESTRING((-1 0, 0 1))', p) < 0.8;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test "
                  "WHERE ST_Distance('MULTILINESTRING((-1 0, 0 1))', p) < 1.1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test "
                  "WHERE ST_Distance(p, 'MULTILINESTRING((-1 0, 0 1))') < 2.5;",
                  dt)));

    // distance transforms
    EXPECT_EQ(double(0),
              v<double>(run_simple_agg(
                  "SELECT ST_Distance(ST_Transform(gpoly4326, 900913), gp900913) from "
                  "geospatial_test WHERE id = 1;",
                  dt)));
    EXPECT_NEAR(
        (double)472720.79722545284,
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Transform(ST_SetSRID(ST_POINT(ST_XMAX(gpoly4326), ST_YMAX(gpoly4326)), 4326), 900913), ST_Transform(gpoly4326, 900913)) FROM geospatial_test WHERE id = 5;)",
            dt)),
        (double)0.001);

    // SRID mismatch
    EXPECT_THROW(run_simple_agg("SELECT ST_Distance('POINT(0 0)', "
                                "ST_Transform(ST_SetSRID(p, 4326), 900913)) "
                                "FROM geospatial_test limit 1;",
                                dt),
                 std::runtime_error);
    // supported aggs
    {
      std::string query("SELECT id, COUNT(poly) FROM geospatial_test GROUP BY id;");
      auto result = run_multiple_agg(query, dt);
      ASSERT_EQ(result->rowCount(), size_t(g_num_rows));
    }

    // Unsupported aggs
    for (std::string const agg :
         {"AVG", "MIN", "MAX", "SUM", "APPROX_MEDIAN", "MODE", "COUNT_IF"}) {
      auto const query = "SELECT " + agg + "(p) FROM geospatial_test;";
      EXPECT_ANY_THROW(run_simple_agg(query, dt));
    }
    EXPECT_ANY_THROW(run_simple_agg(
        "SELECT COUNT(*) FROM geospatial_test a, geospatial_test b WHERE a.p = b.p;",
        dt));
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT id, MIN(p) FROM geospatial_test GROUP BY id ORDER BY id DESC;", dt));
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT id, MAX(l) FROM geospatial_test GROUP BY id ORDER BY id DESC;", dt));
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT id, SUM(poly) FROM geospatial_test GROUP BY id ORDER BY id DESC;", dt));
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT id, AVG(mpoly) FROM geospatial_test GROUP BY id ORDER BY id DESC;", dt));

    // Select *
    {
      const auto rows =
          run_multiple_agg("SELECT * FROM geospatial_test WHERE id = 1", dt);
      const auto row = rows->getNextRow(false, false);
      ASSERT_EQ(row.size(), size_t(24));
    }

    // Projection (return GeoTargetValue)
    compare_geo_target(run_simple_agg("SELECT p FROM geospatial_test WHERE id = 1;", dt),
                       GeoPointTargetValue({1., 1.}));
    compare_geo_target(run_simple_agg("SELECT mp FROM geospatial_test WHERE id = 1;", dt),
                       GeoMultiPointTargetValue({1., 0., 2., 2., 3., 3.}));
    compare_geo_target(run_simple_agg("SELECT l FROM geospatial_test WHERE id = 1;", dt),
                       GeoLineStringTargetValue({1., 0., 2., 2., 3., 3.}));
    compare_geo_target(run_simple_agg("SELECT ml FROM geospatial_test WHERE id = 1;", dt),
                       GeoMultiLineStringTargetValue({1., 0., 2., 2., 3., 3.}, {3}));
    compare_geo_target(
        run_simple_agg("SELECT poly FROM geospatial_test WHERE id = 1;", dt),
        GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}));
    compare_geo_target(
        run_simple_agg("SELECT mpoly FROM geospatial_test WHERE id = 1;", dt),
        GeoMultiPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}, {1}));

    // Sample() version of above
    THROW_ON_AGGREGATOR(compare_geo_target(
        run_simple_agg("SELECT SAMPLE(p) FROM geospatial_test WHERE id = 1;", dt),
        GeoPointTargetValue({1., 1.})));
    THROW_ON_AGGREGATOR(compare_geo_target(
        run_simple_agg("SELECT SAMPLE(mp) FROM geospatial_test WHERE id = 1;", dt),
        GeoMultiPointTargetValue({1., 0., 2., 2., 3., 3.})));
    THROW_ON_AGGREGATOR(compare_geo_target(
        run_simple_agg("SELECT SAMPLE(l) FROM geospatial_test WHERE id = 1;", dt),
        GeoLineStringTargetValue({1., 0., 2., 2., 3., 3.})));
    THROW_ON_AGGREGATOR(compare_geo_target(
        run_simple_agg("SELECT SAMPLE(ml) FROM geospatial_test WHERE id = 1;", dt),
        GeoMultiLineStringTargetValue({1., 0., 2., 2., 3., 3.}, {3})));
    THROW_ON_AGGREGATOR(compare_geo_target(
        run_simple_agg("SELECT SAMPLE(poly) FROM geospatial_test WHERE id = 1;", dt),
        GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3})));
    THROW_ON_AGGREGATOR(compare_geo_target(
        run_simple_agg("SELECT SAMPLE(mpoly) FROM geospatial_test WHERE id = 1;", dt),
        GeoMultiPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}, {1})));

    // Sample() version of above with GROUP BY
    compare_geo_target(
        run_simple_agg("SELECT SAMPLE(p) FROM geospatial_test WHERE id = 1 GROUP BY id;",
                       dt),
        GeoPointTargetValue({1., 1.}));
    compare_geo_target(
        run_simple_agg("SELECT SAMPLE(mp) FROM geospatial_test WHERE id = 1 GROUP BY id;",
                       dt),
        GeoMultiPointTargetValue({1., 0., 2., 2., 3., 3.}));
    compare_geo_target(
        run_simple_agg("SELECT SAMPLE(l) FROM geospatial_test WHERE id = 1 GROUP BY id;",
                       dt),
        GeoLineStringTargetValue({1., 0., 2., 2., 3., 3.}));
    compare_geo_target(
        run_simple_agg("SELECT SAMPLE(ml) FROM geospatial_test WHERE id = 1 GROUP BY id;",
                       dt),
        GeoMultiLineStringTargetValue({1., 0., 2., 2., 3., 3.}, {3}));
    compare_geo_target(
        run_simple_agg(
            "SELECT SAMPLE(poly) FROM geospatial_test WHERE id = 1 GROUP BY id;", dt),
        GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}));
    compare_geo_target(
        run_simple_agg(
            "SELECT SAMPLE(mpoly) FROM geospatial_test WHERE id = 1 GROUP BY id;", dt),
        GeoMultiPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}, {1}));

    // Sample() with compression
    compare_geo_target(
        run_simple_agg(
            "SELECT SAMPLE(gp4326) FROM geospatial_test WHERE id = 1 GROUP BY id;", dt),
        GeoPointTargetValue({1., 1.}),
        0.01);
    compare_geo_target(
        run_simple_agg(
            "SELECT SAMPLE(gpoly4326) FROM geospatial_test WHERE id = 1 GROUP BY id;",
            dt),
        GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}),
        0.01);

    // Reductions (TODO: It would be nice to have some correctness, but for now we ensure
    // these queries run without crashing)
    EXPECT_NO_THROW(run_simple_agg(
        "SELECT SAMPLE(mpoly) FROM geospatial_test WHERE id > 2 GROUP BY id", dt));
    EXPECT_NO_THROW(run_simple_agg(
        "SELECT SAMPLE(gpoly4326) FROM geospatial_test WHERE id > 2 GROUP BY id", dt));

    // Sample with multiple aggs
    {
      const auto rows = run_multiple_agg(
          "SELECT COUNT(*), SAMPLE(l) FROM geospatial_test WHERE id = 1 GROUP BY id;",
          dt);
      rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
      const auto row = rows->getNextRow(false, false);
      CHECK_EQ(row.size(), size_t(2));
      ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(row[0]));
      compare_geo_target(row[1], GeoLineStringTargetValue({1., 0., 2., 2., 3., 3.}));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT COUNT(*), SAMPLE(poly) FROM geospatial_test WHERE id = 1 GROUP BY id;",
          dt);
      rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
      const auto row = rows->getNextRow(false, false);
      CHECK_EQ(row.size(), size_t(2));
      ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(row[0]));
      compare_geo_target(row[1], GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT COUNT(*), SAMPLE(ST_X(p)), SAMPLE(ST_Y(p)) FROM geospatial_test WHERE "
          "id = 1 GROUP BY id;",
          dt);
      rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
      const auto row = rows->getNextRow(false, false);
      CHECK_EQ(row.size(), size_t(3));
      ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(row[0]));
      ASSERT_EQ(static_cast<double>(1.0), v<double>(row[1]));
      ASSERT_EQ(static_cast<double>(1.0), v<double>(row[2]));
    }

    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                  "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                                  dt)));
    compare_geo_target(
        run_simple_agg("SELECT p FROM geospatial_test WHERE "
                       "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                       dt),
        GeoPointTargetValue({0, 0}));

    compare_geo_target(
        get_first_target("SELECT p, l FROM geospatial_test WHERE "
                         "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                         dt),
        GeoPointTargetValue({0, 0}));
    compare_geo_target(
        get_first_target("SELECT p, ST_Distance(ST_GeomFromText('POINT(0 0)'), p), l "
                         "FROM geospatial_test "
                         "WHERE ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                         dt),
        GeoPointTargetValue({0, 0}));
    compare_geo_target(
        get_first_target("SELECT l, ST_Distance(ST_GeomFromText('POINT(0 0)'), p), p "
                         "FROM geospatial_test "
                         "WHERE ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                         dt),
        GeoLineStringTargetValue({0., 0., 0., 0.}));
    ASSERT_EQ(
        static_cast<double>(0.),
        v<double>(get_first_target("SELECT ST_Distance(ST_GeomFromText('POINT(0 0)'), "
                                   "p), p, l FROM geospatial_test WHERE "
                                   "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                                   dt)));

    compare_geo_target(
        run_simple_agg("SELECT l FROM geospatial_test WHERE "
                       "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                       dt),
        GeoLineStringTargetValue({0., 0., 0., 0.}));
    compare_geo_target(
        run_simple_agg("SELECT l FROM geospatial_test WHERE "
                       "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) BETWEEN 7 AND 8;",
                       dt),
        GeoLineStringTargetValue({5., 0., 10., 10., 11., 11.}));
    compare_geo_target(
        run_simple_agg("SELECT gp4326 FROM geospatial_test WHERE "
                       "ST_Distance(ST_GeomFromText('POINT(0 0)'), "
                       "p) > 1 AND ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 2",
                       dt),
        GeoPointTargetValue({0.9999, 0.9999}),
        0.01);
    // Projection (return WKT strings)
    ASSERT_EQ("POINT (1 1)",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT p FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("MULTIPOINT (1 0,2 2,3 3)",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT mp FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("LINESTRING (1 0,2 2,3 3)",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT l FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("MULTILINESTRING ((1 0,2 2,3 3))",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT ml FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("POLYGON ((0 0,2 0,0 2,0 0))",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT poly FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("MULTIPOLYGON (((0 0,2 0,0 2,0 0)))",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT mpoly FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("LINESTRING (5 0,10 10,11 11)",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT l FROM geospatial_test WHERE "
                  "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) BETWEEN 7 AND 8;",
                  dt,
                  false))));
    ASSERT_EQ("LINESTRING (0 0,0 0)",
              boost::get<std::string>(v<NullableString>(
                  get_first_target("SELECT l, p FROM geospatial_test WHERE "
                                   "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                                   dt,
                                   false))));
    ASSERT_EQ("POINT (0 0)",
              boost::get<std::string>(v<NullableString>(
                  get_first_target("SELECT p, l FROM geospatial_test WHERE "
                                   "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                                   dt,
                                   false))));

    // more distance
    ASSERT_NEAR(
        static_cast<double>(26.87005768),
        v<double>(run_simple_agg(
            R"(SELECT Max(ST_MaxDistance(l, 'POINT(0 0)')) FROM geospatial_test;)", dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(14.142135),
                v<double>(run_simple_agg(
                    R"(SELECT Max(ST_MaxDistance(p, l)) FROM geospatial_test;)", dt)),
                static_cast<double>(0.01));

    // point equals
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                        "WHERE ST_Equals('POINT(2 2)', p);",
                                        dt)));
    // precise comparisons for uncompressed points
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                        "WHERE ST_Equals('POINT(2.000000002 2)', p);",
                                        dt)));
    // 4326 geo literals are compressed by default, check equality with uncompressed col
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test "
                  "WHERE ST_Equals(ST_GeomFromText('POINT(2 2)', 4326), gp4326none);",
                  dt)));
    // spatial equality of same points stored in compressed and uncompressed columns
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                        "WHERE ST_Equals(gp4326, gp4326none);",
                                        dt)));

    // intersects
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Intersects(p,p);", dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(
                  run_simple_agg("SELECT count(*) FROM geospatial_test "
                                 "WHERE ST_Intersects(p, ST_GeomFromText('POINT(0 0)'));",
                                 dt)));
    ASSERT_EQ(static_cast<int64_t>(6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(p, ST_GeomFromText('LINESTRING(0 0, 5 5)'));",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(p, ST_GeomFromText('LINESTRING(0 0, 15 15)'));",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Intersects(l, ST_GeomFromText('LINESTRING(0.5 0.5, 6.5 0.5)'));",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Intersects(poly, ST_GeomFromText('LINESTRING(0 4.5, 7 0.5)'));",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Intersects(mpoly, ST_GeomFromText('LINESTRING(0 4.5, 7 0.5)'));",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test "
                                  "WHERE ST_Intersects(l, ST_GeomFromText('POLYGON((0.5 "
                                  "0.5, 6.5 0.5, 3 0.1))'));",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test "
                                  "WHERE ST_Intersects(poly, ST_GeomFromText('POLYGON((0 "
                                  "4.5, 7 0.5, 10 10))'));",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test "
                                  "WHERE ST_Intersects(mpoly, "
                                  "ST_GeomFromText('POLYGON((0 4.5, 7 0.5, 10 10))'));",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(l, ST_GeomFromText('MULTIPOLYGON(((0.5 0.5, 6.5 "
                  "0.5, 3 0.1)))'));",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(poly, ST_GeomFromText('MULTIPOLYGON(((0 4.5, 7 "
                  "0.5, 10 10)))'));",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(mpoly, ST_GeomFromText('MULTIPOLYGON(((0 4.5, 7 "
                  "0.5, 10 10)))'));",
                  dt)));
    // Pip running on double coords (point literal blocking switch to compressed coords)
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Intersects(gpoly4326, ST_GeomFromText('POINT(0.1 0.1)', 4326));)",
            dt)));
    // Pip running on compressed coords (centroid is dynamically compressed)
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Intersects(gpoly4326, ST_SetSRID(ST_Centroid(gpoly4326),4326));)",
            dt)));
    // Pip running on compressed coords (arguments are swapped for consistency)
    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Intersects(gp4326,gpoly4326);)",
            dt)));
    // Pip running on compressed coords (centroid arg is proactively dynamically
    // compressed, additionally arguments are swapped for consistency)
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Intersects(ST_SetSRID(ST_Centroid(gpoly4326),4326),gpoly4326);)",
            dt)));

    // disjoint
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Disjoint(p,p);", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows - 1),
        v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test "
                                  "WHERE ST_Disjoint(p, ST_GeomFromText('POINT(0 0)'));",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows - 6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Disjoint(p, ST_GeomFromText('LINESTRING(0 0, 5 5)'));",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Disjoint(p, ST_GeomFromText('LINESTRING(0 0, 15 15)'));",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows - 6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Disjoint(l, ST_GeomFromText('LINESTRING(0.5 0.5, 6.5 0.5)'));",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows - 6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Disjoint(poly, ST_GeomFromText('LINESTRING(0 4.5, 7 0.5)'));",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows - 6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Disjoint(mpoly, ST_GeomFromText('LINESTRING(0 4.5, 7 0.5)'));",
            dt)));
    // Disjoint runs as negated Intersects, compressed geometries will result in a switch
    // to pip running on compressed coords (arguments are swapped for consistency)
    ASSERT_EQ(
        static_cast<int64_t>(8),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Disjoint(gp4326,gpoly4326);)",
            dt)));

    // contains, within
    // ST_Contains, ST_Within
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(p,p);", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE "
                  "ST_Contains('POINT(0 0)', p) OR ST_Contains('POINT(1 1)', p);",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains('POINT(0 0)', p);",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(gp4326none, "
                  "ST_GeomFromText('POINT(1 1)', 4326));",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains('POINT(0 0)', l);",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Within('POINT(10.5 10.5)', l);",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(p,l);", dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(l,p);", dt)));

    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Contains(poly, 'POINT(-1 -1)');",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Contains(poly, 'POINT(0.1 0.1)');",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Within('POINT(0.1 0.1)', poly);",
                                        dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(gpoly4326, ST_GeomFromText('POINT(0.1 0.1)', 4326));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(7),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_DWithin(l, 'POINT(-1 -1)', 8.0);)",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_DFullyWithin(l, 'POINT(-1 -1)', 8.0);",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(5),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_DWithin(poly, 'POINT(5 5)', 3.0);",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(6),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_DWithin(poly, 'POINT(5 5)', id);",
                                        dt)));

    // accessors
    SKIP_ON_AGGREGATOR(ASSERT_NEAR(
        static_cast<double>(5),
        v<double>(run_simple_agg(
            "SELECT ST_XMax(p) from geospatial_test order by id limit 1 offset 5;", dt)),
        static_cast<double>(0.0)));
    SKIP_ON_AGGREGATOR(ASSERT_NEAR(
        static_cast<double>(1.0),
        v<double>(run_simple_agg(
            "SELECT ST_YMin(gp4326) from geospatial_test limit 1 offset 1;", dt)),
        static_cast<double>(0.001)));
    ASSERT_NEAR(
        static_cast<double>(2 * 7 + 1),
        v<double>(run_simple_agg(
            "SELECT ST_XMax(l) from geospatial_test order by id limit 1 offset 7;", dt)),
        static_cast<double>(0.0));
    ASSERT_NEAR(
        static_cast<double>(2 + 1),
        v<double>(run_simple_agg(
            "SELECT ST_YMax(mpoly) from geospatial_test order by id limit 1 offset 2;",
            dt)),
        static_cast<double>(0.0));

    // Check linestring indexing on ST_Contains(LINESTRING,LINESTRING) and ST_Distance
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(l,ST_StartPoint(l));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(l,ST_EndPoint(l));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(ST_PointN(l,1),ST_EndPoint(l));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(l,ST_StartPoint(l))=0.0;)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(ST_EndPoint(l),l)=0.0;)",
            dt)));

    // Point geometries/geographies, literals in different spatial references, transforms
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance('POINT(0 0)', gp) < 100.0;)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(4),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(ST_GeogFromText('POINT(0 0)', 4326), CastToGeography(gp4326)) < 500000.0;)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(4),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(ST_GeomFromText('POINT(0 0)', 900913), gp900913) < 5.0;)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(4),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(ST_Transform(ST_GeomFromText('POINT(0 0)', 4326), 900913), ST_Transform(gp4326, 900913)) < 500000.0;)",
            dt)));
    ASSERT_DOUBLE_EQ(
        static_cast<double>(111319.4841946785),
        v<double>(run_simple_agg(
            R"(SELECT conv_4326_900913_x(ST_X(gp4326)) FROM geospatial_test WHERE id = 1;)",
            dt)));
    // Check that geography casts are registered in geo operators
    ASSERT_NEAR(
        static_cast<double>(157293.74),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(CastToGeography(gp4326), ST_GeogFromText('POINT(1 1)',4326)) from geospatial_test WHERE id = 0;)",
            dt)),
        static_cast<double>(0.01));

    // ST_NRings
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NRings(poly) from geospatial_test limit 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NRings(mpoly) from geospatial_test limit 1;", dt)));
    // null
    ASSERT_EQ(inline_int_null_value<int>(),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NRings(poly_null) from geospatial_test limit 1;", dt)));
    ASSERT_EQ(inline_int_null_value<int>(),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NRings(mpoly_null) from geospatial_test limit 1;", dt)));

    // ST_NPoints
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NPoints(p) from geospatial_test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg("SELECT ST_NPoints(l) FROM geospatial_test ORDER "
                                        "BY ST_NPoints(l) DESC LIMIT 1;",
                                        dt)));
    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg("SELECT ST_NPoints(ml) FROM geospatial_test ORDER "
                                  "BY ST_NPoints(l) DESC LIMIT 1;",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg("SELECT ST_NPoints(poly) FROM geospatial_test "
                                        "ORDER BY ST_NPoints(l) DESC LIMIT 1;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg("SELECT ST_NPoints(mpoly) FROM geospatial_test "
                                        "ORDER BY ST_NPoints(l) DESC LIMIT 1;",
                                        dt)));
    // null
    // for a POINT, this still returns 1 even if the point value is NULL
    // @TODO check the required behavior here and fix separately if required
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NPoints(p_null) from geospatial_test limit 1", dt)));
    ASSERT_EQ(inline_int_null_value<int>(),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NPoints(mp_null) from geospatial_test limit 1", dt)));
    ASSERT_EQ(inline_int_null_value<int>(),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NPoints(l_null) from geospatial_test limit 1", dt)));
    ASSERT_EQ(inline_int_null_value<int>(),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NPoints(ml_null) from geospatial_test limit 1", dt)));
    ASSERT_EQ(inline_int_null_value<int>(),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NPoints(poly_null) from geospatial_test limit 1", dt)));
    ASSERT_EQ(inline_int_null_value<int>(),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NPoints(mpoly_null) from geospatial_test limit 1", dt)));

    // ST_SRID, ST_SetSRID
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(
                  run_simple_agg("SELECT ST_SRID(p) from geospatial_test limit 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(4326),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_SRID(gp4326) from geospatial_test limit 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(900913),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_SRID(gp900913) from geospatial_test limit 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(4326),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_SRID(ST_GeogFromText('POINT(-118.243683 34.052235)', 4326)) "
                  "from geospatial_test limit 1;",
                  dt)));

    // SQLw/out geo support
    EXPECT_THROW(run_multiple_agg("SELECT count(distinct p) FROM geospatial_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT approx_count_distinct(p) FROM geospatial_test;", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT avg(p) FROM geospatial_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT p, count(*) FROM geospatial_test GROUP BY p;", dt),
        std::runtime_error);

    // ST_contains + n_rings + constructors
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(poly, ST_Point(0.1 + ST_NRings(poly)/10.0, 0.1));)",
            dt)));

    // perimeter and area
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_Perimeter(p) FROM geospatial_test WHERE id = 4;)", dt));
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_Perimeter(l) FROM geospatial_test WHERE id = 4;)", dt));
    ASSERT_NEAR(
        v<double>(run_simple_agg(
            R"(SELECT ST_Perimeter(poly) FROM geospatial_test WHERE id = 4;)", dt)),
        double(17.071067811865476),
        double(10e-5));
    ASSERT_NEAR(
        v<double>(run_simple_agg(
            R"(SELECT ST_Perimeter(mpoly) FROM geospatial_test WHERE id = 4;)", dt)),
        double(17.071067811865476),
        double(10e-5));
    ASSERT_NEAR(
        v<double>(run_simple_agg(
            R"(SELECT ST_Perimeter(gpoly4326) FROM geospatial_test WHERE id = 4;)", dt)),
        double(17.07106773237212),
        double(10e-5));

    EXPECT_ANY_THROW(
        run_simple_agg(R"(SELECT ST_Area(p) FROM geospatial_test WHERE id = 4;)", dt));
    EXPECT_ANY_THROW(
        run_simple_agg(R"(SELECT ST_Area(l) FROM geospatial_test WHERE id = 4;)", dt));
    ASSERT_NEAR(v<double>(run_simple_agg(
                    R"(SELECT ST_Area(poly) FROM geospatial_test WHERE id = 4;)", dt)),
                double(12.5),
                double(10e-5));
    ASSERT_NEAR(v<double>(run_simple_agg(
                    R"(SELECT ST_Area(mpoly) FROM geospatial_test WHERE id = 4;)", dt)),
                double(12.5),
                double(10e-5));
    ASSERT_NEAR(
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(gpoly4326) FROM geospatial_test WHERE id = 4;)", dt)),
        double(12.5),
        double(10e-5));
    // Same area projected to web mercator - square meters
    ASSERT_NEAR(
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Transform(gpoly4326,900913)) FROM geospatial_test WHERE id = 4;)",
            dt)),
        double(155097342153.4868),
        double(0.01));

    // centroid
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_test where ST_Distance(gp4326,ST_Centroid(gmp4326)) < 3;)",
            dt)));
    ASSERT_NEAR(
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Centroid(gpoly4326),'POINT (1.6666666 1.66666666)') FROM geospatial_test WHERE id = 4;)",
            dt)),
        double(0.0),
        double(10e-5));
    // web mercator centroid
    ASSERT_NEAR(
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Centroid(ST_Transform(gpoly4326,900913)), 'POINT (185532.482988 185768.418973)') FROM geospatial_test WHERE id = 4;)",
            dt)),
        double(0.0),
        double(10e-5));
    // make sure ST_Centroid picks up its argument's srid
    // TODO: make ST_Distance and other geo operators complain about srid mismatches
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DWithin(ST_Centroid(gpoly4326), ST_GeomFromText('POINT (1.6666666 1.66666666)',4326),0.001) FROM geospatial_test WHERE id = 4;)",
            dt)));
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_DWithin(ST_Centroid(gpoly4326), 'POINT (1.6666666 1.66666666)',0.001) FROM geospatial_test WHERE id = 4;)",
        dt));

    // order by (unsupported)
    EXPECT_ANY_THROW(run_multiple_agg("SELECT p FROM geospatial_test ORDER BY p;", dt));
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT poly, l, id FROM geospatial_test ORDER BY id, poly;", dt));

    // ST_IntersectsBox to join two tables
    // todo: support boolean predicate operator to determine geospatial relationship based
    // on bounding box
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT ST_IntersectsBox(poly, mpoly) FROM geospatial_test", dt));
    // geo operator with non-geo column
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT COUNT(1) FROM geospatial_test R, geospatial_test S WHERE "
        "ST_IntersectsBox(R.poly, S.id)",
        dt));
    ASSERT_EQ(static_cast<int64_t>(100),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(1) FROM geospatial_test R, geospatial_test S WHERE "
                  "ST_IntersectsBox(R.poly, S.mpoly)",
                  dt)));

    // ST_NumGeometries
    // pass (actual count)
    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_NumGeometries(mp) FROM geospatial_test WHERE id = 1;)", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_NumGeometries(ml2) FROM geospatial_test WHERE id = 1;)", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  R"(SELECT ST_NumGeometries(mpoly2) FROM geospatial_test WHERE id = 1;)",
                  dt)));
    // pass (non-MULTI geo, return 1, per PostGIS)
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_NumGeometries(p) FROM geospatial_test WHERE id = 1;)", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_NumGeometries(l) FROM geospatial_test WHERE id = 1;)", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_NumGeometries(poly) FROM geospatial_test WHERE id = 1;)", dt)));
    // pass (null)
    // @TODO investigate why a NULL POINT does not report as NULL
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT ST_NumGeometries(p_null) from geospatial_test WHERE id = 1", dt)));
    ASSERT_EQ(
        inline_int_null_value<int>(),
        v<int64_t>(run_simple_agg(
            "SELECT ST_NumGeometries(mp_null) from geospatial_test WHERE id = 1", dt)));
    ASSERT_EQ(
        inline_int_null_value<int>(),
        v<int64_t>(run_simple_agg(
            "SELECT ST_NumGeometries(l_null) from geospatial_test WHERE id = 1", dt)));
    ASSERT_EQ(
        inline_int_null_value<int>(),
        v<int64_t>(run_simple_agg(
            "SELECT ST_NumGeometries(ml_null) from geospatial_test WHERE id = 1", dt)));
    ASSERT_EQ(
        inline_int_null_value<int>(),
        v<int64_t>(run_simple_agg(
            "SELECT ST_NumGeometries(poly_null) from geospatial_test WHERE id = 1", dt)));
    ASSERT_EQ(inline_int_null_value<int>(),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NumGeometries(mpoly_null) from geospatial_test WHERE id = 1",
                  dt)));
    // fail (invalid input)
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT ST_NumGeometries(id) FROM geospatial_test WHERE id = 1;", dt));
    EXPECT_ANY_THROW(run_multiple_agg(
        "SELECT ST_NumGeometries(42) FROM geospatial_test WHERE id = 1;", dt));
    // fail for now
    // @TODO add support for literals
    EXPECT_ANY_THROW(
        run_multiple_agg("SELECT ST_NumGeometries('POLYGON((0 0, 1 0, 1 1, 0 1))') FROM "
                         "geospatial_test WHERE id = 1;",
                         dt));
  }
}

TEST_P(GeoSpatialTestTablesFixture, Constructors) {
  constexpr double EPS = 1e-8;  // non-zero double values should match to about 8 sigfigs
  double expected;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    {
      auto rows = run_multiple_agg(
          R"(SELECT ST_Point(id, id), id, ST_Point(id + 1, id + 2) FROM geospatial_test WHERE id < 2 ORDER BY 2;)",
          dt);
      rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
      EXPECT_EQ(rows->rowCount(), size_t(2));

      auto process_row = [&](const int64_t id_for_row) {
        auto row = rows->getNextRow(false, false);
        EXPECT_EQ(row.size(), size_t(3));
        const auto id = v<int64_t>(row[1]);
        EXPECT_EQ(id, id_for_row);
        const auto first_geo_tv = boost::get<GeoTargetValue>(row[0]);
        CHECK(first_geo_tv);
        const auto first_pt = boost::get<GeoPointTargetValue>(*first_geo_tv);
        CHECK(first_pt.coords->data());
        const double* first_pt_array = first_pt.coords->data();
        EXPECT_EQ(first_pt_array[0], static_cast<double>(id));
        EXPECT_EQ(first_pt_array[1], static_cast<double>(id));

        const auto second_geo_tv = boost::get<GeoTargetValue>(row[2]);
        CHECK(second_geo_tv);
        const auto second_pt = boost::get<GeoPointTargetValue>(*second_geo_tv);
        CHECK(second_pt.coords->data());
        const double* second_pt_array = second_pt.coords->data();
        EXPECT_EQ(second_pt_array[0], static_cast<double>(id + 1));
        EXPECT_EQ(second_pt_array[1], static_cast<double>(id + 2));
      };

      process_row(0);
      process_row(1);
    }

    {
      // multi-frag iteration check
      auto rows = run_multiple_agg(
          R"(SELECT id, ST_Point(id, id) FROM geospatial_test WHERE id > 2 ORDER BY 1;)",
          dt);
      rows->setGeoReturnType(ResultSet::GeoReturnType::WktString);
      EXPECT_EQ(rows->rowCount(), size_t(7));

      auto process_row = [&](const int64_t id_for_row) {
        auto row = rows->getNextRow(false, false);
        EXPECT_EQ(row.size(), size_t(2));
        const auto id = v<int64_t>(row[0]);
        EXPECT_EQ(id, id_for_row + 3);  // offset by 3 from filter
        const auto wkt_str = boost::get<std::string>(v<NullableString>(row[1]));
        EXPECT_EQ(wkt_str,
                  "POINT (" + std::to_string(id) + " " + std::to_string(id) + ")");
      };
      for (size_t i = 0; i < 7; i++) {
        process_row(i);
      }
    }

    EXPECT_EQ(
        "POINT (2 2)",
        boost::get<std::string>(v<NullableString>(run_simple_agg(
            "SELECT ST_Point(id,id) FROM geospatial_test WHERE id = 2;", dt, false))));
    EXPECT_EQ(
        "POINT (2 2)",
        boost::get<std::string>(v<NullableString>(run_simple_agg(
            "SELECT ST_SetSRID(ST_Point(id,id),4326) FROM geospatial_test WHERE id = 2;",
            dt,
            false))));
    EXPECT_EQ(
        double(2),
        v<double>(run_simple_agg(
            "SELECT ST_X(ST_Point(id, id)) FROM geospatial_test WHERE id = 2;", dt)));
    EXPECT_EQ(
        double(3),
        v<double>(run_simple_agg(
            "SELECT ST_Y(ST_Point(id, id + 1)) FROM geospatial_test WHERE id = 2;", dt)));
    EXPECT_EQ(
        inline_fp_null_value<double>(),
        v<double>(run_simple_agg(
            "SELECT ST_Y(ST_Point(id, null)) FROM geospatial_test WHERE id = 2;", dt)));
    EXPECT_NEAR(222638.981586547,
                v<double>(run_simple_agg(
                    "SELECT ST_X(ST_Transform(ST_SetSRID(ST_Point(id,id),4326), 900913)) "
                    "FROM geospatial_test WHERE id = 2;",
                    dt,
                    false)),
                10e-8);
    EXPECT_NEAR(222684.208505543,
                v<double>(run_simple_agg(
                    "SELECT ST_Y(ST_Transform(ST_SetSRID(ST_Point(id,id),4326), 900913)) "
                    "FROM geospatial_test WHERE id = 2;",
                    dt,
                    false)),
                10e-8);
    EXPECT_NEAR(222638.977750596,
                v<double>(run_simple_agg("SELECT ST_X(ST_Transform(gp4326, 900913)) FROM "
                                         "geospatial_test WHERE id = 2;",
                                         dt,
                                         false)),
                10e-8);
    EXPECT_NEAR(222684.204667253,
                v<double>(run_simple_agg("SELECT ST_Y(ST_Transform(gp4326, 900913)) FROM "
                                         "geospatial_test WHERE id = 2;",
                                         dt,
                                         false)),
                10e-8);
    SKIP_ON_AGGREGATOR({
      // ensure transforms run on GPU. transforms use math functions which need to be
      // specialized for GPU
      if (dt == ExecutorDeviceType::GPU) {
        const auto query_explain_result = QR::get()->runSelectQuery(
            R"(SELECT ST_Transform(gp4326, 900913) FROM geospatial_test WHERE id = 2;)",
            dt,
            /*hoist_literals=*/true,
            /*allow_loop_joins=*/false,
            /*just_explain=*/true);
        const auto explain_result = query_explain_result->getRows();
        EXPECT_EQ(size_t(1), explain_result->rowCount());
        const auto crt_row = explain_result->getNextRow(true, true);
        EXPECT_EQ(size_t(1), crt_row.size());
        const auto explain_str = boost::get<std::string>(v<NullableString>(crt_row[0]));
        EXPECT_NE(explain_str.find("IR for the GPU:"), std::string::npos) << explain_str;
      }
    });
    EXPECT_DOUBLE_EQ(
        222638.97775059601,
        v<double>(run_simple_agg(
            R"(SELECT ST_X(ST_Transform(gp4326, 900913)) FROM geospatial_test WHERE id = 2;)",
            dt,
            false)));
    double const expected_x = 1.7966305682390428e-05;
    EXPECT_NEAR(
        expected_x,
        v<double>(run_simple_agg(
            R"(SELECT ST_X(ST_Transform(gp900913, 4326)) FROM geospatial_test WHERE id = 2;)",
            dt,
            false)),
        expected_x * EPS);
    double const expected_y = 1.7966305681601786e-05;
    EXPECT_NEAR(
        expected_y,
        v<double>(run_simple_agg(
            R"(SELECT ST_Y(ST_Transform(gp900913, 4326)) FROM geospatial_test WHERE id = 2;)",
            dt,
            false)),
        expected_y * EPS);
    // "POINT (0.000017966305682 0.000017966305682)"
    std::string const point = boost::get<std::string>(v<NullableString>(run_simple_agg(
        R"(SELECT ST_Transform(gp900913, 4326) FROM geospatial_test WHERE id = 2;)",
        dt,
        false)));
    boost::cmatch md;  // match data
    boost::regex const regex("^POINT \\((.+?) (.+?)\\)$");
    EXPECT_TRUE(boost::regex_match(point.c_str(), md, regex)) << point;
    char* end;
    expected = strtod(md[1].first, &end);   // Parse x-coordinate into a double value
    EXPECT_EQ(end, md[1].second) << point;  // Confirm no extra characters
    EXPECT_NEAR(expected, expected_x, expected * EPS) << point;
    expected = strtod(md[2].first, &end);   // Parse y-coordinate into a double value
    EXPECT_EQ(end, md[2].second) << point;  // Confirm no extra characters
    EXPECT_NEAR(expected, expected_y, expected * EPS) << point;
    SKIP_ON_AGGREGATOR({
      // ensure transforms run on GPU. transforms use math functions which need to be
      // specialized for GPU
      if (dt == ExecutorDeviceType::GPU) {
        const auto query_explain_result = QR::get()->runSelectQuery(
            R"(SELECT ST_Transform(gp900913, 4326) FROM geospatial_test WHERE id = 2;)",
            dt,
            /*hoist_literals=*/true,
            /*allow_loop_joins=*/false,
            /*just_explain=*/true);
        const auto explain_result = query_explain_result->getRows();
        EXPECT_EQ(size_t(1), explain_result->rowCount());
        const auto crt_row = explain_result->getNextRow(true, true);
        EXPECT_EQ(size_t(1), crt_row.size());
        const auto explain_str = boost::get<std::string>(v<NullableString>(crt_row[0]));
        EXPECT_NE(explain_str.find("IR for the GPU:"), std::string::npos) << explain_str;
      }
    });
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_Transform(gpoly900913, 4326) FROM geospatial_test WHERE id = 2;)",
        dt));
  }
}

TEST_P(GeoSpatialTestTablesFixture, LLVMOptimization) {
  SKIP_ALL_ON_AGGREGATOR();

  ScopeGuard reset_explain_type = [] {
    QR::get()->setExplainType(ExecutorExplainType::Default);
  };
  QR::get()->setExplainType(ExecutorExplainType::Optimized);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    auto get_explain_result = [](const std::string& query, const ExecutorDeviceType dt) {
      const auto query_explain_result =
          QR::get()->runSelectQuery(query,
                                    dt,
                                    /*hoist_literals=*/true,
                                    /*allow_loop_joins=*/false,
                                    /*just_explain=*/true);
      const auto explain_result = query_explain_result->getRows();
      EXPECT_EQ(size_t(1), explain_result->rowCount());
      const auto crt_row = explain_result->getNextRow(true, true);
      EXPECT_EQ(size_t(1), crt_row.size());
      return boost::get<std::string>(v<NullableString>(crt_row[0]));
    };

    // expect the x decompression code to be absent in optimized IR
    std::string explain = get_explain_result(
        "SELECT ST_Y(ST_Transform(gp4326, 900913)) from geospatial_test;", dt);
    EXPECT_EQ(explain.find("decompress_x_coord_geoint"), std::string::npos) << explain;

    // expect the y decompression code to be absent in optimized IR
    explain = get_explain_result(
        "SELECT ST_X(ST_Transform(gp4326, 900913)) from geospatial_test;", dt);
    EXPECT_EQ(explain.find("decompress_y_coord_geoint"), std::string::npos) << explain;

    // expect both decompression codes to be present
    explain = get_explain_result(
        "SELECT ST_X(ST_Transform(gp4326, 900913)), ST_Y(ST_Transform(gp4326, 900913)) "
        "from geospatial_test;",
        dt);
    EXPECT_NE(explain.find("decompress_y_coord_geoint"), std::string::npos) << explain;
    explain = get_explain_result(
        "SELECT ST_X(ST_Transform(gp4326, 900913)), ST_Y(ST_Transform(gp4326, 900913)) "
        "from geospatial_test;",
        dt);
    EXPECT_NE(explain.find("decompress_y_coord_geoint"), std::string::npos) << explain;
  }
}

TEST_P(GeoSpatialTestTablesFixture, AsText) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    if (dt == ExecutorDeviceType::GPU) {
      LOG(WARNING) << "ST_AsText is a CPU-specific UDF";
    } else {
      std::vector<std::string> col_names{
          "p",       "p_null",     "mp",      "mp_null",   "l",          "l_null",
          "ml",      "ml2",        "ml_null", "poly",      "poly_null",  "mpoly",
          "mpoly2",  "mpoly_null", "gp",      "gp4326",    "gp4326none", "gp900913",
          "gmp4326", "gl4326none", "gml4326", "gpoly4326", "gpoly900913"};

      std::vector<std::string> fns{"ST_AsText", "ST_AsWkt"};

      for (std::string& fn : fns) {
        for (std::string& col : col_names) {
          std::string query =
              "SELECT " + fn + "(" + col + "), " + col + " FROM geospatial_test;";
          const auto rows = run_multiple_agg(query, dt);
          size_t row_count = rows->rowCount();
          for (size_t i = 0; i < row_count; i++) {
            auto row = rows->getNextRow(true, false);
            NullableString ns = TestHelpers::v<NullableString>(row[0]);

            if (std::string* str = boost::get<std::string>(&ns)) {
              std::string expected =
                  boost::get<std::string>(TestHelpers::v<NullableString>(row[1]));
              ASSERT_EQ(expected, *str);
            } else {
              void* val = boost::get<void*>(TestHelpers::v<NullableString>(row[1]));
              ASSERT_EQ(val, nullptr);
            }
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(GeoSpatialTablesTests,
                         GeoSpatialTestTablesFixture,
                         ::testing::Values(true, false));

class GeoSpatialNullTablesFixture : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    import_geospatial_null_test(/*with_temporary_tables=*/GetParam());
  }

  void TearDown() override {
    if (!GetParam() && !g_keep_data) {
      run_ddl_statement("DROP TABLE IF EXISTS geospatial_null_test;");
    }
  }
};

TEST_P(GeoSpatialNullTablesFixture, GeoWithNulls) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_null_test where p IS NOT NULL;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_null_test where p IS NULL;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows / 2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_null_test WHERE ST_Distance(p,p) < 0.1;",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows / 2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_null_test WHERE ST_Distance(p,p) IS NULL;",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_null_test WHERE "
                                        "ST_Distance(l,gpnotnull) IS NULL;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT count(gpnotnull) FROM geospatial_null_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT count(ST_X(p)) FROM geospatial_null_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows - 1),
              v<int64_t>(run_simple_agg(
                  "SELECT count(ST_X(gp4326)) FROM geospatial_null_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_null_test WHERE "
                                        "ST_Distance('POINT(0 0)', p) < 100.0;",
                                        dt)));
    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_null_test WHERE "
                                  "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 9;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_null_test WHERE ST_Distance(p,mp) < 2.0;",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_null_test WHERE ST_Distance(p,l) < 2.0;",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_null_test WHERE ST_Distance(p,ml) < 2.0;",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_null_test WHERE "
                                        "ST_Distance(p,gpnotnull) >= 0.0;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_null_test WHERE "
                                        "ST_Distance(gp4326,gp4326none) IS NULL;",
                                        dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_null_test "
                                  "WHERE ST_Distance('MULTIPOINT(-1 0, 0 1)', p) < 6.0;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_null_test "
                                  "WHERE ST_Distance('LINESTRING(-1 0, 0 1)', p) < 6.0;",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_null_test "
                  "WHERE ST_Distance('MULTILINESTRING((-1 0, 0 1))', p) < 6.0;",
                  dt)));

    ASSERT_EQ("POINT (1 1)",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT p FROM geospatial_null_test WHERE id = 1;", dt, false))));
    auto p = v<NullableString>(
        run_simple_agg("SELECT p FROM geospatial_null_test WHERE id = 2;", dt, false));
    auto p_v = boost::get<void*>(&p);
    auto p_s = boost::get<std::string>(&p);
    ASSERT_TRUE(p_v && *p_v == nullptr && !p_s);
    p = v<NullableString>(
        run_simple_agg("SELECT poly FROM geospatial_null_test WHERE id = 2;", dt, false));
    p_v = boost::get<void*>(&p);
    p_s = boost::get<std::string>(&p);
    ASSERT_TRUE(p_v && *p_v == nullptr && !p_s);
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  R"(SELECT ST_Contains(poly,p) FROM geospatial_null_test WHERE id=1;)",
                  dt,
                  false)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT ST_Contains(poly,p) IS NULL FROM geospatial_null_test WHERE id=2;",
            dt,
            false)));
  }
}

TEST_P(GeoSpatialNullTablesFixture, Constructors) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    auto nullcheck_result = [](auto p) {
      auto p_v = boost::get<void*>(&p);
      auto p_s = boost::get<std::string>(&p);
      EXPECT_TRUE(p_v && *p_v == nullptr && !p_s);
    };

    nullcheck_result(v<NullableString>(run_simple_agg(
        R"(SELECT ST_Transform(gp4326, 900913) FROM geospatial_null_test WHERE id = 4;)",
        dt,
        false)));
    nullcheck_result(v<NullableString>(run_simple_agg(
        R"(SELECT ST_Transform(gp4326none, 900913) FROM geospatial_null_test WHERE id = 5;)",
        dt,
        false)));
    nullcheck_result(v<NullableString>(run_simple_agg(
        R"(SELECT ST_Transform(gp900913, 4326) FROM geospatial_null_test WHERE id = 6;)",
        dt,
        false)));
  }
}

TEST_P(GeoSpatialNullTablesFixture, LazyFetch) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_ALL_ON_AGGREGATOR();
    SKIP_NO_GPU();
    std::vector<std::string> col_names{"p", "l", "poly", "mpoly"};
    for (auto& col_name : col_names) {
      std::string query =
          "SELECT b." + col_name +
          " FROM geospatial_null_test a INNER JOIN geospatial_null_test b ON "
          "ST_Intersects(ST_SetSRID(b.mpoly, 4326), a.gp4326) WHERE a.id = 1;";
      const auto query_res = QR::get()->runSQL(query, dt);
      EXPECT_EQ(size_t(8), query_res->rowCount());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(GeospatialNullTests,
                         GeoSpatialNullTablesFixture,
                         ::testing::Values(true, false));

TEST(GeoSpatial, Math) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // ST_Distance
    ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('LINESTRING(-2 2, 2 2)', 'LINESTRING(4 2, 4 3)');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('LINESTRING(-2 2, 2 2, 2 0)', 'LINESTRING(4 0, 0 -4, -4 0, 0 4)');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.31),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('LINESTRING(-2 2, 2 2, 2 0)', 'LINESTRING(4 0, 0 -4, -4 0, 0 5)');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(3.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_GeomFromText('POINT(5 -1)'), ST_GeomFromText('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))'));)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_GeomFromText('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))'), ST_GeomFromText('POINT(0.5 0.5)'));)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.5),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_GeomFromText('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2), (1 1, -1 1, -1 -1, 1 -1, 1 1))'), ST_GeomFromText('POINT(0.5 0.5)'));)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_GeomFromText('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))'), ST_GeomFromText('LINESTRING(0.5 0.5, 0.7 0.75, -0.3 -0.3, -0.82 0.12, 0.3 0.64)'));)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.18),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_GeomFromText('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2), (1 1, -1 1, -1 -1, 1 -1, 1 1))'), ST_GeomFromText('LINESTRING(0.5 0.5, 0.7 0.75, -0.3 -0.3, -0.82 0.12, 0.3 0.64)'));)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))', 'POLYGON((0.5 0.5, -0.5 0.5, -0.5 -0.5, 0.5 -0.5, 0.5 0.5))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.5),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2), (1 1, -1 1, -1 -1, 1 -1, 1 1))', 'POLYGON((0.5 0.5, -0.5 0.5, -0.5 -0.5, 0.5 -0.5, 0.5 0.5))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2), (1 1, -1 1, -1 -1, 1 -1, 1 1))', 'POLYGON((4 2, 5 2, 5 3, 4 3, 4 2))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(1.4142),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POLYGON((0 0, 4 0, 4 4, 2 5, 0 4, 0 0), (1 1, 1 3, 2 4, 3 3, 3 1, 1 1))', 'POLYGON((5 5, 8 2, 8 4, 5 5))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POLYGON((0 0, 4 0, 4 4, 2 5, 0 4, 0 0), (1 1, 1 3, 2 4, 3 3, 3 1, 1 1))','POLYGON((3.5 3.5, 8 2, 8 4, 3.5 3.5))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POLYGON((0 0, 4 0, 4 4, 2 5, 0 4, 0 0), (1 1, 1 3, 2 4, 3 3, 3 1, 1 1))', 'POLYGON((8 2, 8 4, 2 2, 8 2))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 'POINT(4 2)');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 'LINESTRING(4 2, 5 3)');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('LINESTRING(4 2, 5 3)', 'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POLYGON((4 2, 5 3, 4 3))', 'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 'POLYGON((4 2, 5 3, 4 3))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(1.4142),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 'MULTIPOLYGON(((4 2, 5 3, 4 3)), ((3 3, 4 3, 3 4)))');)",
            dt)),
        static_cast<double>(0.01));

    ASSERT_NEAR(
        static_cast<double>(25.4558441),
        v<double>(run_simple_agg(
            R"(SELECT ST_MaxDistance('POINT(1 1)', 'LINESTRING (9 0,18 18,19 19)');)",
            dt)),
        static_cast<double>(0.01));

    // Geodesic distance between Paris and LA geographic points: ~9105km
    ASSERT_NEAR(
        static_cast<double>(9105643.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_GeogFromText('POINT(-118.4079 33.9434)', 4326), ST_GeogFromText('POINT(2.5559 49.0083)', 4326));)",
            dt)),
        static_cast<double>(10000.0));
    // Geodesic distance between Paris and LA geometry points cast as geography points:
    // ~9105km
    ASSERT_NEAR(
        static_cast<double>(9105643.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(CastToGeography(ST_GeomFromText('POINT(-118.4079 33.9434)', 4326)), cast (ST_GeomFromText('POINT(2.5559 49.0083)', 4326) as geography));)",
            dt)),
        static_cast<double>(10000.0));
    // Cartesian distance between Paris and LA calculated from wgs84 degrees
    ASSERT_NEAR(
        static_cast<double>(121.89),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_GeomFromText('POINT(-118.4079 33.9434)', 4326), ST_GeomFromText('POINT(2.5559 49.0083)', 4326));)",
            dt)),
        static_cast<double>(1.0));
    ASSERT_NEAR(
        static_cast<double>(121.89),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(-118.4079 33.9434)', 'POINT(2.5559 49.0083)');)",
            dt)),
        static_cast<double>(1.0));
    // Cartesian distance between Paris and LA wgs84 coords transformed to web merc
    ASSERT_NEAR(
        static_cast<double>(13653148.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Transform(ST_GeomFromText('POINT(-118.4079 33.9434)', 4326), 900913), ST_Transform(ST_GeomFromText('POINT(2.5559 49.0083)', 4326), 900913));)",
            dt)),
        static_cast<double>(10000.0));

    // ST_Length
    // Cartesian length of a planar path
    ASSERT_NEAR(static_cast<double>(5.65685),
                v<double>(run_simple_agg(
                    R"(SELECT ST_Length('LINESTRING(1 0, 0 1, -1 0, 0 -1, 1 0)');)", dt)),
                static_cast<double>(0.0001));
    // Geodesic length of a geographic path, in meters
    ASSERT_NEAR(
        static_cast<double>(617121.626),
        v<double>(run_simple_agg(
            R"(SELECT ST_Length(CAST (ST_GeomFromText('LINESTRING(-76.6168198439371 39.9703199555959, -80.5189990254673 40.6493554919257, -82.5189990254673 42.6493554919257)', 4326) as GEOGRAPHY));)",
            dt)),
        static_cast<double>(0.01));
    // Cartesian length of a planar multi path
    ASSERT_NEAR(
        static_cast<double>(6.65685),
        v<double>(run_simple_agg(
            R"(SELECT ST_Length('MULTILINESTRING((1 0, 0 1, -1 0, 0 -1, 1 0),(2 2,2 3))');)",
            dt)),
        static_cast<double>(0.0001));

    // ST_Perimeter
    // Cartesian perimeter of a planar polygon
    ASSERT_NEAR(
        static_cast<double>(5.65685),
        v<double>(run_simple_agg(
            R"(SELECT ST_Perimeter('POLYGON((1 0, 0 1, -1 0, 0 -1, 1 0),(0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0))');)",
            dt)),
        static_cast<double>(0.0001));
    // Geodesic perimeter of a polygon geography, in meters
    ASSERT_NEAR(
        static_cast<double>(1193066.02892),
        v<double>(run_simple_agg(
            R"(SELECT ST_Perimeter(ST_GeogFromText('POLYGON((-76.6168198439371 39.9703199555959, -80.5189990254673 40.6493554919257, -82.5189990254673 42.6493554919257, -76.6168198439371 39.9703199555959))', 4326));)",
            dt)),
        static_cast<double>(0.01));
    // Cartesian perimeter of a planar multipolygon
    ASSERT_NEAR(
        static_cast<double>(4 * 1.41421 + 4 * 2.82842),
        v<double>(run_simple_agg(
            R"(SELECT ST_Perimeter('MULTIPOLYGON(((1 0, 0 1, -1 0, 0 -1, 1 0), (0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0)), ((2 0, 0 2, -2 0, 0 -2, 2 0), (0.2 0, 0 0.2, -0.2 0, 0 -0.2, 0.2 0)))');)",
            dt)),
        static_cast<double>(0.0001));
    // Geodesic perimeter of a polygon geography, in meters
    ASSERT_NEAR(
        static_cast<double>(1193066.02892 + 1055903.62342 + 907463.55601),
        v<double>(run_simple_agg(
            R"(SELECT ST_Perimeter(ST_GeogFromText('MULTIPOLYGON(((-76.6168198439371 39.9703199555959, -80.5189990254673 40.6493554919257, -82.5189990254673 42.6493554919257, -76.6168198439371 39.9703199555959)), ((-66.6168198439371 49.9703199555959, -70.5189990254673 50.6493554919257, -72.5189990254673 52.6493554919257, -66.6168198439371 49.9703199555959)), ((-56.6168198439371 59.9703199555959, -60.5189990254673 60.6493554919257, -62.5189990254673 62.6493554919257, -56.6168198439371 59.9703199555959)))', 4326));)",
            dt)),
        static_cast<double>(0.01));

    // ST_Area
    // Area of a planar polygon
    ASSERT_NEAR(
        static_cast<double>(2.0 - 0.02),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area('POLYGON((1 0, 0 1, -1 0, 0 -1, 1 0),(0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0))');)",
            dt)),
        static_cast<double>(0.0001));
    // Area of a planar multipolygon
    ASSERT_NEAR(
        static_cast<double>(2.0 - 0.02 + 8.0 - 0.08),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area('MULTIPOLYGON(((1 0, 0 1, -1 0, 0 -1, 1 0), (0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0)), ((2 0, 0 2, -2 0, 0 -2, 2 0), (0.2 0, 0 0.2, -0.2 0, 0 -0.2, 0.2 0)))');)",
            dt)),
        static_cast<double>(0.0001));

    // ST_Equals
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  R"(SELECT ST_Equals('POINT(1 1)', 'POINT(1 1)');)", dt)));
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  R"(SELECT ST_Equals('POINT(1 1)', 'POINT(1.00000001 1)');)", dt)));

    // ST_Intersects
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POINT(0.9 0.9)'), ST_GeomFromText('POINT(1.1 1.1)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POINT(1 1)'), ST_GeomFromText('LINESTRING(2 0, 0 2, -2 0, 0 -2)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('LINESTRING(2 0, 0 2, -2 0, 0 -2)'), ST_GeomFromText('POINT(1 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POINT(1 1)'), ST_GeomFromText('POLYGON((0 0, 1 0, 0 1, 0 0))'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))'), ST_GeomFromText('POINT(1 1)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POINT(1 1)'), ST_GeomFromText('MULTIPOLYGON(((5 5, 6 6, 5 6)), ((0 0, 1 0, 0 1, 0 0)))'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('MULTIPOLYGON(((0 0, 2 0, 2 2, 0 2, 0 0)), ((5 5, 6 6, 5 6)))'), ST_GeomFromText('POINT(1 1)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('LINESTRING(1 1, 0.5 1.5, 2 4)'), ST_GeomFromText('LINESTRING(2 0, 0 2, -2 0, 0 -2)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('LINESTRING(1 1, 0.5 1.5, 1.5 1, 1.5 1.5)'), ST_GeomFromText('LINESTRING(1 0, 0 1, -1 0, 0 -1)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('LINESTRING(3 3, 3 2, 2.1 2.1)'), ST_GeomFromText('POLYGON((2 2, 0 1, -2 2, -2 0, 2 0, 2 2))'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POLYGON((2 2, 0 1, -2 2, -2 0, 2 0, 2 2))'), ST_GeomFromText('LINESTRING(3 3, 3 2, 2 2)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('LINESTRING(3 3, 3 2, 2.1 2.1)'), ST_GeomFromText('MULTIPOLYGON(((5 5, 6 6, 5 6)), ((2 2, 0 1, -2 2, -2 0, 2 0, 2 2)))'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('MULTIPOLYGON(((2 2, 0 1, -2 2, -2 0, 2 0, 2 2)), ((5 5, 6 6, 5 6)))'), ST_GeomFromText('LINESTRING(3 3, 3 2, 2 2)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POLYGON((-118.66313066279504 44.533565793694436,-115.28301791070872 44.533565793694436,-115.28301791070872 46.49961643537853,-118.66313066279504 46.49961643537853,-118.66313066279504 44.533565793694436))'),ST_GeomFromText('LINESTRING (-118.526348964556 45.6369689645418,-118.568716970537 45.552529965319,-118.604668964913 45.5192699867856,-118.700612922525 45.4517749629224)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POLYGON((-165.27254008488316 60.286744877866084,-164.279755308478 60.286744877866084, -164.279755308478 60.818880025426154,-165.27254008488316 60.818880025426154))', 4326),ST_GeomFromText('MULTIPOLYGON (((-165.273152946156 60.5488599839382,-165.244307548387 60.4963022239955,-165.23881195357 60.4964759808483,-165.234271979534 60.4961199595109,-165.23165799921 60.496354988076,-165.229399998313 60.4973489979735,-165.225239975948 60.4977589987674,-165.217958113746 60.4974514248303,-165.21276192051 60.4972319866052)))',4326));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POLYGON((-9.838404039411898 50.55533029518068, -2.310857889588476 50.55533029518068, -2.310857889588476 53.61604635210904, -9.838404039411898 53.61604635210904, -9.838404039411898 50.55533029518068))', 4326), ST_GeomFromText('LINESTRING (-9.54855228287566 51.7461543817754,-9.54461588968738 51.7447587529871,-9.54434548949094 51.7369761558887)', 4326));)",
            dt)));

    // ST_Disjoint
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Disjoint(ST_GeomFromText('POLYGON((2 2, 0 1, -2 2, -2 0, 2 0, 2 2))'), ST_GeomFromText('LINESTRING(3 3, 3 2, 2 2)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Disjoint(ST_GeomFromText('LINESTRING(3 3, 3 2, 2.1 2.1)'), ST_GeomFromText('MULTIPOLYGON(((5 5, 6 6, 5 6)), ((2 2, 0 1, -2 2, -2 0, 2 0, 2 2)))'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Disjoint(ST_GeomFromText('POLYGON((3 3, 3 2, 2.1 2.1))'), ST_GeomFromText('MULTIPOLYGON(((5 5, 6 6, 5 6)), ((2 2, 0 1, -2 2, -2 0, 2 0, 2 2)))'));)",
            dt)));

    // ST_Contains
    ASSERT_EQ(
        static_cast<int64_t>(1),  // polygon containing a point
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0))'), ST_GeomFromText('POINT(0 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(
            0),  // same polygon but with a hole in the middle that the point falls into
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0),(1 0, 0 1, -1 0, 0 -1, 1 0))', 'POINT(0.1 0.1)');)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // back to true if we combine the holed polygon with one
                                  // more in a multipolygon
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains('MULTIPOLYGON(((2 0, 0 2, -2 0, 0 -2, 2 0),(1 0, 0 1, -1 0, 0 -1, 1 0)), ((2 0, 0 2, -2 0, 0 -2, 1 -2, 2 -1)))', 'POINT(0.1 0.1)');)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // last query but for 4326 objects
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('MULTIPOLYGON(((2 0, 0 2, -2 0, 0 -2, 2 0),(1 0, 0 1, -1 0, 0 -1, 1 0)), ((2 0, 0 2, -2 0, 0 -2, 1 -2, 2 -1)))', 4326), ST_GeomFromText('POINT(0.1 0.1)', 4326));)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(0),  // point in polygon, on left edge
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 -1, 2 1, 0 1, 0 -1))'), ST_GeomFromText('POINT(0 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),  // point in polygon, on right edge
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 -1, 2 1, 0 1, 0 -1))'), ST_GeomFromText('POINT(1 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // point in polygon, touch+leave
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 5 2, 0 2, -1 0))'), ST_GeomFromText('POINT(0 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // point in polygon, touch+overlay+leave
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 4 0, 5 2, 0 2, -1 0))'), ST_GeomFromText('POINT(0 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // point in polygon, touch+cross
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 4 -1, 5 2, 0 2, -1 0))'), ST_GeomFromText('POINT(0 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // point in polygon, touch+overlay+cross
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 4 0, 4.5 -1, 5 2, 0 2, -1 0))'), ST_GeomFromText('POINT(0 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),  // point in polygon, check yray redundancy
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 5 2, 0 2, -1 0))'), ST_GeomFromText('POINT(2 0)'));)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(1),  // polygon containing linestring
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0))'), ST_GeomFromText('LINESTRING(1 0, 0 1, -1 0, 0 -1, 1 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),  // polygon containing only a part of linestring
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0))'), ST_GeomFromText('LINESTRING(1 0, 0 1, -1 0, 0 -1, 3 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),    // polygon containing linestring vertices
        v<int64_t>(run_simple_agg(  // but not all of linestring's segments
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((2 2, 0 1, -2 2, -2 0, 2 0, 2 2))'), ST_GeomFromText('LINESTRING(1.5 1.5, -1.5 1.5, 0 0.5, 1.5 1.5)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // polygon containing another polygon
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0))'), ST_GeomFromText('POLYGON((1 0, 0 1, -1 0, 0 -1, 1 0))'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // multipolygon containing linestring
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('MULTIPOLYGON(((3 3, 4 3, 4 4)), ((2 0, 0 2, -2 0, 0 -2, 2 0)))'), ST_GeomFromText('LINESTRING(1 0, 0 1, -1 0, 0 -1, 1 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),    // multipolygon containing linestring vertices
        v<int64_t>(run_simple_agg(  // but not all of linestring's segments
            R"(SELECT ST_Contains(ST_GeomFromText('MULTIPOLYGON(((2 2, 0 1, -2 2, -2 0, 2 0, 2 2)), ((3 3, 4 3, 4 4)))'), ST_GeomFromText('LINESTRING(1.5 1.5, -1.5 1.5, 0 0.5, 1.5 1.5)'));)",
            dt)));
    // Tolerance
    ASSERT_EQ(
        static_cast<int64_t>(1),  // point containing an extremely close point
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POINT(2.1100000001 -1.7229999999)'), ST_GeomFromText('POINT(2.11 -1.723)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),  // point not containing a very close point
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POINT(2.11 -1.723)'),ST_GeomFromText('POINT(2.1100001 -1.7229999)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // linestring containing an extremely close point
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('LINESTRING(1 -1.0000000001, 3 -1.0000000001)'), ST_GeomFromText('POINT(0.9999999992 -1)'));)",
            dt)));

    // ST_Contains doesn't yet support MULTILINESTRING
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0))'), ST_GeomFromText('MULTILINESTRING((1 0, 0 1, -1 0), (0 -1, 1 0))'));)",
        dt));

    // Postgis compatibility
    ASSERT_EQ(
        static_cast<int64_t>(0),  // point on vertex of polygon
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'), ST_GeomFromText('POINT(0 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // point within polygon
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'), ST_GeomFromText('POINT(5 5)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),  // point outside polygon
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'), ST_GeomFromText('POINT(-1 0)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),  // point on edge of polygon
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'), ST_GeomFromText('POINT(0 5)'));)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),  // point in line with polygon edge
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Contains(ST_GeomFromText('POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))'), ST_GeomFromText('POINT(0 12)'));)",
            dt)));

    // ST_DWithin, ST_DFullyWithin
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DWithin('POLYGON((4 2, 5 3, 4 3))', 'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 3.0);)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DWithin('MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 'POLYGON((4 2, 5 3, 4 3))', 3.0);)",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DFullyWithin('POINT(1 1)', 'LINESTRING (9 0,18 18,19 19)', 26.0) AND NOT ST_DFullyWithin('LINESTRING (9 0,18 18,19 19)', 'POINT(1 1)', 25.0);)",
            dt)));

    // Check if Paris and LA are within a 9500km geodesic distance
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DWithin(ST_GeogFromText('POINT(-118.4079 33.9434)', 4326), ST_GeogFromText('POINT(2.5559 49.0083)', 4326), 9500000.0);)",
            dt)));
    // .. though not within 9000km
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DWithin(ST_GeogFromText('POINT(-118.4079 33.9434)', 4326), ST_GeogFromText('POINT(2.5559 49.0083)', 4326), 9000000.0);)",
            dt)));
    // Make sure geodesic form of ST_DWithin rejects non-POINT GEOGRAPHYs
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_DWithin(ST_GeogFromText('POLYGON((-118.4079 33.9434, -119.4079 32.9434, -117.4079 34.9434))', 4326), ST_GeogFromText('POINT(2.5559 49.0083)', 4326), 9000000.0);)",
        dt));

    // ST_DWithin optimization to trim irrelevant heads and tails of very big linestrings
    // Discarding very big linestring if its every segment is too far from buffered bbox
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DWithin('LINESTRING(0 5, 1 5, 2 5, 3 5, 4 5, 5 5, 5 4, 5 3, 5 2, 5 1, 4 0)', 'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 1);)",
            dt)));
    // Trimming very big linestring just to a portion that might be within distance,
    // but distance calc shows that it's actually not
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DWithin('LINESTRING(0 5, 1 5, 2 5, 3 4, 4 3, 5 2, 5 1, 5 0)', 'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 1.9);)",
            dt)));
    // Trimming very big linestring just to a portion that might be within distance
    // and distance calc confirms that it actually is
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DWithin('LINESTRING(0 5, 1 5, 2 5, 3 4, 4 3, 5 2, 5 1, 5 0)', 'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1))', 2.15);)",
            dt)));
    // Trimming very big linestring just to a portion that might be within distance,
    // with the linestring's tail landing inside the buffered bbox. Only head is trimmed
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_DWithin('LINESTRING(0 5, 1 5, 2 5, 3 5, 4 5, 5 5, 5 4, 5 3, 5 2, 5 1, 4 0)', 'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 2.1);)",
            dt)));

    // Coord accessors
    ASSERT_NEAR(
        static_cast<double>(-118.4079),
        v<double>(run_simple_agg(R"(SELECT ST_X('POINT(-118.4079 33.9434)');)", dt)),
        static_cast<double>(0.0));
    ASSERT_NEAR(
        static_cast<double>(33.9434),
        v<double>(run_simple_agg(
            R"(SELECT ST_Y(ST_GeomFromText('POINT(-118.4079 33.9434)', 4326));)", dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(4021204.558),
        v<double>(run_simple_agg(
            R"(SELECT ST_Y(ST_Transform(ST_GeomFromText('POINT(-118.4079 33.9434)', 4326), 900913));)",
            dt)),
        static_cast<double>(0.01));

    ASSERT_NEAR(
        static_cast<double>(-118.4079),
        v<double>(run_simple_agg(R"(SELECT ST_XMax('POINT(-118.4079 33.9434)');)", dt)),
        static_cast<double>(0.0));
    ASSERT_NEAR(
        static_cast<double>(3960189.382),
        v<double>(run_simple_agg(
            R"(SELECT ST_YMax('MULTIPOLYGON (((-13201820.2402333 3957482.147359,-13189665.9329505 3960189.38265416,-13176924.0813953 3949756.56479131)))');)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(4021204.558),
        v<double>(run_simple_agg(
            R"(SELECT ST_YMin(ST_Transform(ST_GeomFromText('LINESTRING (-118.4079 33.9434, 2.5559 49.0083)', 4326), 900913));)",
            dt)),
        static_cast<double>(0.01));

    // Point accessors, Linestring indexing
    ASSERT_NEAR(
        static_cast<double>(34.274647),
        v<double>(run_simple_agg(
            R"(SELECT ST_Y(ST_PointN(ST_GeomFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, -122.446747 37.733795)', 4326), 2));)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(-122.446747),
        v<double>(run_simple_agg(
            R"(SELECT ST_X(ST_EndPoint(ST_GeomFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, -122.446747 37.733795)', 4326)));)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(  // TODO: postgis has this at 557422.59741475
        static_cast<double>(
            557637.3711),  // geodesic distance between first and end points: LA - SF trip
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_PointN(ST_GeogFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, -122.446747 37.733795)', 4326), 1), ST_EndPoint(ST_GeogFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, -122.446747 37.733795)', 4326)));)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(
            5.587),  // cartesian distance in degrees, same points: LA - SF trip
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_PointN(ST_GeomFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, -122.446747 37.733795)', 4326), 1), ST_EndPoint(ST_GeomFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, -122.446747 37.733795)', 4326)));)",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(689217.783),  // cartesian distance between merc-transformed
                                          // first and end points
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_StartPoint(ST_Transform(ST_GeomFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, -122.446747 37.733795)', 4326), 900913)), ST_EndPoint(ST_Transform(ST_GeomFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, -122.446747 37.733795)', 4326), 900913)));)",
            dt)),
        static_cast<double>(0.01));
    // Linestring: check that runaway indices are controlled
    ASSERT_NEAR(
        static_cast<double>(inline_fp_null_value<double>()),  // return null
        v<double>(run_simple_agg(
            R"(SELECT ST_X(ST_PointN(ST_GeomFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, -122.446747 37.733795)', 4326), 1000000));)",
            dt)),
        static_cast<double>(0.01));

    // Test some exceptions
    // Point coord accessor used on a non-POINT, in this case unindexed LINESTRING
    // (missing ST_POINTN)
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_Y(ST_GeogFromText('LINESTRING(-118.243683 34.052235, -119.229034 34.274647)', 4326));)",
        dt));
    // Two accessors in a row
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_X(ST_Y(ST_GeogFromText('POINT(-118.243683 34.052235)', 4326)));)",
        dt));
    // Coord order reversed, longitude value is out of latitude range
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_Y(ST_GeogFromText('POINT(34.052235 -118.243683)', 4326));)", dt));
    // Linestring accessor on a non-LINESTRING
    EXPECT_ANY_THROW(run_simple_agg(
        R"(SELECT ST_X(ST_ENDPOINT('POINT(-118.243683 34.052235)'));)", dt));

    // Geodesic distance between Paris and LA: ~9105km
    ASSERT_NEAR(
        static_cast<double>(9105643.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(CastToGeography(ST_SetSRID(ST_GeomFromText('POINT(-118.4079 33.9434)'), 4326)), CastToGeography(ST_SetSRID(ST_GeomFromText('POINT(2.5559 49.0083)'), 4326)));)",
            dt)),
        static_cast<double>(10000.0));

    // ST_Point geo constructor
    ASSERT_NEAR(static_cast<double>(1.4142135),
                v<double>(run_simple_agg(
                    R"(SELECT ST_Distance(ST_Point(0,0), ST_Point(1,1));)", dt)),
                static_cast<double>(0.00001));
    // Cartesian distance between Paris and LA, point constructors
    ASSERT_NEAR(
        static_cast<double>(13653148.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Transform(ST_SetSRID(ST_Point(-118.4079, 33.9434), 4326), 900913), ST_Transform(ST_SetSRID(ST_Point(2.5559, 49.0083), 4326), 900913));)",
            dt)),
        static_cast<double>(10000.0));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Intersects(ST_GeomFromText('POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))'), ST_Point(1.1 - 0.1, 3.0 - 1.0 ));)",
            dt)));

    // ST_Centroid geo constructor
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(1 1)', ST_Centroid('POINT(1 1)'));)", dt)),
        static_cast<double>(0.00001));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(1 1)', ST_Centroid('MULTIPOINT(0 0, 2 0, 2 2, 0 2)'));)",
            dt)),
        static_cast<double>(0.00001));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(0.8 0.8)', ST_Centroid('MULTIPOINT(0 0, 2 0, 2 2, 0 2, 0 0)'));)",
            dt)),
        static_cast<double>(0.00001));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(-6.0 40.5)', ST_Centroid('LINESTRING(-20 35, 8 46)'));)",
            dt)),
        static_cast<double>(0.00001));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(1.3333333 1)', ST_Centroid('LINESTRING(0 0, 2 0, 2 2, 0 2)'));)",
            dt)),
        static_cast<double>(0.00001));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(1 1)', ST_Centroid('LINESTRING(0 0, 2 0, 2 2, 0 2, 0 0)'));)",
            dt)),
        static_cast<double>(0.00001));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(1 1)', ST_Centroid('POLYGON((0 0, 2 0, 2 2, 0 2))'));)",
            dt)),
        static_cast<double>(0.00001));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(10.9291 50.68245)', ST_Centroid('POLYGON((10.9099 50.6917,10.9483 50.6917,10.9483 50.6732,10.9099 50.6732,10.9099 50.6917))'));)",
            dt)),
        static_cast<double>(0.0001));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(0.166666666 0.933333333)', ST_Centroid('MULTIPOLYGON(((1 0,2 1,2 0,1 0)),((-1 -1,2 2,-1 2,-1 -1)))'));)",
            dt)),
        static_cast<double>(0.00001));
    // Degenerate input geometries triggering fall backs to linestring and point centroids
    // zero-area, non-zero-length: fall back to linestring centroid
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(1.585786 1.0)', ST_Centroid('MULTIPOLYGON(((0 0, 2 2, 0 2, 2 0, 0 0)),((3 0, 3 2, 3 1, 3 0)))'));)",
            dt)),
        static_cast<double>(0.0001));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(1.0 1.0)', ST_Centroid('MULTIPOLYGON(((0 0, 1 0, 2 0)),((0 2, 1 2, 2 2)))'));)",
            dt)),
        static_cast<double>(0.0001));
    // zero-area, zero-length: point centroid
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(1.5 1.5)', ST_Centroid('MULTIPOLYGON(((0 0, 0 0, 0 0, 0 0)),((3 3, 3 3, 3 3, 3 3)))'));)",
            dt)),
        static_cast<double>(0.0001));
    // zero-area, non-zero-length: linestring centroid
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(1.0 1.0)', ST_Centroid('POLYGON((0 0, 2 2, 0 2, 2 0, 0 0))'));)",
            dt)),
        static_cast<double>(0.0001));
    // zero-area, zero-length: point centroid
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(3.0 3.0)', ST_Centroid('POLYGON((3 3, 3 3, 3 3, 3 3))'));)",
            dt)),
        static_cast<double>(0.0001));
    // zero-length: fallback to point centroid
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(0 89)', ST_CENTROID('LINESTRING(0 89, 0 89, 0 89, 0 89)'));)",
            dt)),
        static_cast<double>(0.0001));
  }
}

TEST(GeoSpatial, Projections) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_EQ("POINT (2 2)",
              boost::get<std::string>(v<NullableString>(
                  run_simple_agg("SELECT ST_GeomFromText('POINT(2 2)');", dt, false))));
    ASSERT_EQ("POINT (2 2)",
              boost::get<std::string>(
                  v<NullableString>(run_simple_agg("SELECT ST_Point(2,2);", dt, false))));

    // Projection of a900913 literal transformed to 4326
    ASSERT_EQ(
        R"(POLYGON ((-157.552903293424 70.5298902655625,-157.540578542995 70.5298902655625,-157.540578542995 70.5316394430267,-157.552903293424 70.5316394430267,-157.552903293424 70.5298902655625)))",
        boost::get<std::string>(v<NullableString>(run_simple_agg(
            R"(SELECT ST_Transform(ST_GeomFromText('POLYGON(( -17538708.976320468 11243413.985557318, -17537336.990984567 11243413.985557318, -17537336.990984567 11243998.200894408, -17538708.976320468 11243998.200894408, -17538708.976320468 11243413.985557318))', 900913), 4326);)",
            dt,
            false))));
    // Projection of a 4326 literal transformed to 900913
    ASSERT_EQ(
        R"(POLYGON ((-17538708.9676258 11243413.9770419,-17537336.9826839 11243413.9770419,-17537336.9826839 11243998.1869375,-17538708.9676258 11243998.1869375,-17538708.9676258 11243413.9770419)))",
        boost::get<std::string>(v<NullableString>(run_simple_agg(
            R"(SELECT ST_Transform(ST_GeomFromText('POLYGON ((-157.552903293424 70.5298902655625,-157.540578542995 70.5298902655625,-157.540578542995 70.5316394430267,-157.552903293424 70.5316394430267,-157.552903293424 70.5298902655625))', 4326), 900913);)",
            dt,
            false))));

    // unsupported transform projections
    EXPECT_ANY_THROW(run_multiple_agg(
        R"(SELECT ST_Transform(mpoly, 900913) FROM geospatial_test;)", dt));

    EXPECT_ANY_THROW(run_multiple_agg(
        R"(SELECT ST_Transform(gpoly4326, 900913) FROM geospatial_test;)", dt));
  }
}

class GeoSpatialTempTables : public ::testing::Test {
 protected:
  void SetUp() override { import_geospatial_test(/*with_temporary_tables=*/false); }

  void TearDown() override {
    if (!g_keep_data) {
      run_ddl_statement("DROP TABLE IF EXISTS geospatial_test;");
    }
  }
};

TEST_F(GeoSpatialTempTables, Geos) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Currently not supporting cpu retry in distributed, just throwing in gpu mode
    if (g_aggregator && dt == ExecutorDeviceType::GPU) {
      LOG(WARNING) << "Skipping Geos tests on distributed GPU";
      continue;
    }

#ifdef ENABLE_GEOS
    // geos-backed ST functions:
    // Measuring ST_Area of geometry generated by geos-backed function to disregard
    // coordinate ordering chosen by geos and also to test interoperability with
    // natively supported ST functions
    // poly id=2: POLYGON (((0 0,3 0,0 3,0 0))
    // ST_Intersection with poly: MULTIPOLYGON (((1 2,1 1,2 1,1 2)))
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(0.5),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Intersection(poly, 'POLYGON((1 1,3 1,3 3,1 3,1 1))')) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.00001)));
    // ST_Union with poly: MULTIPOLYGON (((2 1,3 1,3 3,1 3,1 2,0 3,0 0,3 0,2 1)))
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(8.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Union(poly, 'POLYGON((1 1,3 1,3 3,1 3,1 1))')) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.00001)));
    // ST_Difference with poly:  MULTIPOLYGON (((2 1,1 1,1 2,0 3,0 0,3 0,2 1)))
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(4.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Difference(poly, 'POLYGON((1 1,3 1,3 3,1 3,1 1))')) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.00001)));
    // ST_Buffer of poly, 0 width: MULTIPOLYGON (((0 0,3 0,0 3,0 0)))
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(4.5),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Buffer(poly, 0.0)) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.00001)));
    // ST_Buffer of poly, 0.1 width: huge rounded MULTIPOLYGON wrapped around poly
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(5.539),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Buffer(poly, 0.1)) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.05)));
    // ST_Buffer on a point, 1.0 width: almost a circle, with area close to Pi
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(3.14159),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Buffer(p, 1.0)) FROM geospatial_test WHERE id = 3;)",
            dt)),
        static_cast<double>(0.03)));
    // ST_Buffer on a point, 1.0 width: distance to buffer
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Buffer(p, 1.0), 'POINT(0 3)') FROM geospatial_test WHERE id = 3;)",
            dt)),
        static_cast<double>(0.03)));
    // ST_Buffer on a multipoint, 1.0 width: basically circles around each point
    // should be 3PI, but slightly less because not true circles
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(9.364),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Buffer('MULTIPOINT(0 0, 10 0, 10 10)', 1.0));)", dt)),
        static_cast<double>(0.03)));
    // should also be 3PI, but slightly less again because two circles touch and merge
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(8.803),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Buffer(mp, 1.0)) FROM geospatial_test WHERE id = 3;)",
            dt)),
        static_cast<double>(0.03)));
    // ST_Buffer on a linestring, 1.0 width: two 10-unit segments
    // each segment is buffered by ~2x10 wide stretch (2 * 2 * 10) plus circular areas
    // around mid- and endpoints
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(42.9018),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Buffer('LINESTRING(0 0, 10 0, 10 10)', 1.0)) FROM geospatial_test WHERE id = 3;)",
            dt)),
        static_cast<double>(0.03)));
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(42.9018),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Buffer('MULTILINESTRING((0 0, 10 0, 10 10))', 1.0)) FROM geospatial_test WHERE id = 3;)",
            dt)),
        static_cast<double>(0.03)));
    // ST_ConcaveHull
#if (GEOS_VERSION_MAJOR > 3) || (GEOS_VERSION_MAJOR == 3 && GEOS_VERSION_MINOR >= 11)
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(22565.6485058608),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_ConcaveHull(ST_GeomFromText('MULTILINESTRING((106 164,30 112,74 70,82 112,130 94,130 62,122 40,156 32,162 76,172 88),(132 178,134 148,128 136,96 128,132 108,150 130,170 142,174 110,156 96,158 90,158 88),(22 64,66 28,94 38,94 68,114 76,112 30,132 10,168 18,178 34,186 52,184 74,190 100,190 122,182 148,178 170,176 184,156 164,146 178,132 186,92 182,56 158,36 150,62 150,76 128,88 118))'),0.99));)",
            dt)),
        static_cast<double>(0.03)));
#else
    // geo operators can't deal with geo operator output transforms yet
    EXPECT_THROW(
        run_simple_agg(
            R"(SELECT ST_Area(ST_ConcaveHull('LINESTRING(0 0, 10 0, 10 10)', 1.0));)",
            dt),
        std::runtime_error);
#endif
    // ST_ConvexHull
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_Area(ST_ConvexHull(ST_GeomFromText('MULTILINESTRING((100 190,10 8),(150 10, 20 30),(50 5, 150 30, 50 10, 10 10))'))) = ST_Area('POLYGON((50 5,10 8,10 10,100 190,150 30,150 10,50 5))');)",
            dt))));  // ConvexHull = POLYGON((50 5,10 8,10 10,100 190,150 30,150 10,50 5))
    // ST_IsValid
    EXPECT_GPU_THROW(
        ASSERT_EQ(static_cast<int64_t>(1),
                  v<int64_t>(run_simple_agg(
                      R"(SELECT ST_IsValid(poly) from geospatial_test limit 1;)", dt))));
    // ST_IsValid: invalid: self-intersecting poly
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_IsValid('POLYGON((0 0,1 1,1 0,0 1,0 0))') from geospatial_test limit 1;)",
            dt))));
    // ST_IsValid: invalid: intersecting polys in a multipolygon
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_IsValid('MULTIPOLYGON(((1 1,3 1,3 3,1 3)),((2 2,2 4,4 4,4 2)))') from geospatial_test limit 1;)",
            dt))));
    // geos-backed ST_Equals for non-point geometries
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT ST_Equals('POLYGON((0 0, 1 1, 1 0))', 'POLYGON((0 0, 1 0, 1 1))');",
            dt))));
    // Different, spatially unequal geometries
    EXPECT_GPU_THROW(
        ASSERT_EQ(static_cast<int64_t>(0),
                  v<int64_t>(run_simple_agg(
                      "SELECT ST_Equals('LINESTRING(0 0, 1 1)', 'POINT(0 0)');", dt))));
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            "SELECT ST_Equals('POLYGON((0 0, 1 1, 1 0))', 'POLYGON((0 0, 1 1, 0 1))');",
            dt))));
    // Different but spatially equal geometries
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT ST_Equals('LINESTRING(0 0, 1 1)', 'LINESTRING(1 1, 0 0)');", dt))));
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg("SELECT ST_Equals('POLYGON((0 0, 2 0, 2 2, 0 2))', "
                                  "'MULTIPOLYGON(((0 1, 0 2, 2 2, 2 0, 0 0)))');",
                                  dt))));
    EXPECT_GPU_THROW(
        ASSERT_EQ(static_cast<int64_t>(1),
                  v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test "
                                            "WHERE ST_Equals(l, 'LINESTRING(2 0, 4 4)');",
                                            dt))));
    // confirm geos recognizes equality of 4326 column and a geo literal, both compressed
    EXPECT_GPU_THROW(
        ASSERT_EQ(static_cast<int64_t>(1),
                  v<int64_t>(run_simple_agg(
                      "SELECT count(*) FROM geospatial_test WHERE ST_Equals(gpoly4326, "
                      "ST_GeomFromText('POLYGON ((0 0,4 0.0,0.0 4,0 0))', 4326));",
                      dt))));
    // same as above but add two extra vertices to the geo literal without changing shape
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test WHERE ST_Equals(gpoly4326, "
            "ST_GeomFromText('POLYGON ((0 0,2 0,4 0.0,0.0 4,0 2,0 0))', 4326));",
            dt))));
    // giving geos a tolerance margin to recognize spatial equality of
    // an uncompressed geo stored in 4326 column and a compressed geo literal
    EXPECT_GPU_THROW(
        ASSERT_EQ(static_cast<int64_t>(1),
                  v<int64_t>(run_simple_agg(
                      "SELECT count(*) FROM geospatial_test WHERE ST_Equals(gl4326none, "
                      "ST_GeomFromText('LINESTRING (4 0,8 8)', 4326))",
                      dt))));
    // geos-backed ST_Union(MULTIPOLYGON,MULTIPOLYGON)
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(14.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Union('MULTIPOLYGON(((0 0,2 0,2 2,0 2)),((4 4,6 4,6 6,4 6)))', 'MULTIPOLYGON(((1 1,3 1,3 3,1 3,1 1)),((5 5,7 5,7 7,5 7)))')) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.001)));
    // geos-backed ST_Intersection(MULTIPOLYGON,MULTIPOLYGON)
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Intersection('MULTIPOLYGON(((0 0,2 0,2 2,0 2)),((4 4,6 4,6 6,4 6)))', 'MULTIPOLYGON(((1 1,3 1,3 3,1 3,1 1)),((5 5,7 5,7 7,5 7)))')) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.001)));
    // geos-backed ST_Intersection(POLYGON,MULTIPOLYGON)
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(3.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Intersection('POLYGON((2 2,2 6,7 6,7 2))', 'MULTIPOLYGON(((1 1,3 1,3 3,1 3,1 1)),((5 5,7 5,7 7,5 7)))')) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.001)));
    // geos-backed ST_Intersection(POLYGON,MULTIPOLYGON) returning a POINT
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(2.828427),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance('POINT(0 0)',ST_Intersection('POLYGON((2 2,2 6,7 6,7 2))', 'MULTIPOLYGON(((1 1,2 1,2 2,1 2,1 1)))')) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.001)));
    // geos-backed ST_Intersection returning GEOMETRYCOLLECTION EMPTY
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Intersection('POLYGON((3 3,3 6,7 6,7 3))', 'MULTIPOLYGON(((1 1,2 1,2 2,1 2,1 1)))')) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.001)));
    // geos-backed ST_IsEmpty on ST_Intersection returning GEOMETRYCOLLECTION EMPTY
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_IsEmpty(ST_Intersection('POLYGON((3 3,3 6,7 6,7 3))', 'MULTIPOLYGON(((1 1,2 1,2 2,1 2,1 1)))')) FROM geospatial_test WHERE id = 2;)",
            dt))));
    // geos-backed ST_IsEmpty on ST_Intersection returning non-empty geo
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_IsEmpty(ST_Intersection('POLYGON((3 3,3 6,7 6,7 3))', 'MULTIPOLYGON(((1 1,4 1,4 4,1 4,1 1)))')) FROM geospatial_test WHERE id = 2;)",
            dt))));
    // geos runtime support for geometry decompression
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(4.5),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Buffer(gpoly4326, 0.0)) FROM geospatial_test WHERE id = 2;)",
            dt)),
        static_cast<double>(0.00001)));
    // geos runtime support for any gdal-recognized transforms on geos call inputs
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(409421478.75976562),
        v<double>(run_simple_agg(
            R"(SELECT ST_Area(ST_Buffer(ST_Transform(ST_GeomFromText('POLYGON((-118.240356 34.04880299999999,-118.64035599999998 34.04880300000001,-118.440356 34.24880300000001))',4326), 26945), 1.0));)",
            dt)),
        static_cast<double>(0.00001)));
    // expect throw for now: geos call output transforms can be sunk into geos runtime but
    // geo operators can't deal with it yet
    EXPECT_THROW(
        // geos runtime support for any gdal-recognized transforms on geos call outputs
        ASSERT_NEAR(
            static_cast<double>(409421494.3899536),
            v<double>(run_simple_agg(
                R"(SELECT ST_Area(ST_Transform(ST_Buffer(ST_GeomFromText('POLYGON((-118.240356 34.04880299999999,-118.64035599999998 34.04880300000001,-118.440356 34.24880300000001))',4326), 1.0), 26945));)",
                dt)),
            static_cast<double>(0.00001)),
        std::runtime_error);
    EXPECT_THROW(
        // geos runtime support for both input and output geo transforms (gdal-backed)
        ASSERT_NEAR(
            static_cast<double>(1756.549591064453),
            v<double>(run_simple_agg(
                R"(SELECT ST_Area(ST_Transform(ST_Buffer(ST_Transform(ST_GeomFromText('POLYGON((-71.11603599316368 42.37469906933211,-71.11600627260486 42.37479327587576,-71.11582940503467 42.37476302224121,-71.11582340452516 42.37478309974037,-71.11570078841396 42.37476310907647,-71.11565279759817 42.37492120281317,-71.11577467489042 42.374941582218895,-71.11576735791459 42.374966813944184,-71.11631216001115 42.37505880035607,-71.11631985924761 42.37503569400519,-71.11641211477945 42.37505132899332,-71.11646061071951 42.37489401310859,-71.11636318099954 42.37487692897568,-71.11636960854412 42.37485520073258,-71.11618998476843 42.37482420784997,-71.11621803803246 42.37472943072518,-71.11603599316368 42.37469906933211))',4326), 26919), 1.0), 26986));)",
                dt)),
            static_cast<double>(0.00001)),
        std::runtime_error);
    EXPECT_THROW(
        // geos runtime support for both input and output geo transforms (gdal-backed),
        // case of geos noop call, it's short-circuited leaving in place just transforms
        ASSERT_NEAR(
            static_cast<double>(1558.806243896484),
            v<double>(run_simple_agg(
                R"(SELECT ST_Area(ST_Transform(ST_Buffer(ST_Transform(ST_GeomFromText('POLYGON((-71.11603599316368 42.37469906933211,-71.11600627260486 42.37479327587576,-71.11582940503467 42.37476302224121,-71.11582340452516 42.37478309974037,-71.11570078841396 42.37476310907647,-71.11565279759817 42.37492120281317,-71.11577467489042 42.374941582218895,-71.11576735791459 42.374966813944184,-71.11631216001115 42.37505880035607,-71.11631985924761 42.37503569400519,-71.11641211477945 42.37505132899332,-71.11646061071951 42.37489401310859,-71.11636318099954 42.37487692897568,-71.11636960854412 42.37485520073258,-71.11618998476843 42.37482420784997,-71.11621803803246 42.37472943072518,-71.11603599316368 42.37469906933211))',4326), 26919), 0.0), 26986));)",
                dt)),
            static_cast<double>(0.00001)),
        std::runtime_error);
    // geos runtime support for input transforms (gdal-backed) of geo columns,
    // also can be used for projection of gdal-transformed constructed geometries, e.g.
    // SELECT ST_Buffer(ST_Transform(gpoly4326, 900913),0) from geospatial_test;
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(37106.49473665067),
        v<double>(run_simple_agg(
            R"(SELECT ST_X(ST_Centroid(ST_Buffer(ST_Transform(gpoly4326, 900913),0))) from geospatial_test limit 1;)",
            dt)),
        static_cast<double>(0.00001)));
    // geo operators can't deal with geo operator output transforms yet
    EXPECT_THROW(
        ASSERT_NEAR(
            static_cast<double>(37106.49473665067),
            v<double>(run_simple_agg(
                R"(SELECT ST_X(ST_Centroid(ST_Transform(ST_Buffer(gpoly4326,0),900913))) from geospatial_test limit 1;)",
                dt)),
            static_cast<double>(0.00001)),
        std::runtime_error);
    // Handling geos returning a MULTIPOINT
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(0.9),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Union('POINT(2 1)', 'POINT(3 0)'), 'POINT(2 0.1)');)",
            dt)),
        static_cast<double>(0.00001)));
    // Handling geos returning a LINESTRING
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(0.8062257740),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Union('LINESTRING(2 1, 3 1)', 'LINESTRING(3 1, 4 1, 3 0)'), 'POINT(2.2 0.1)');)",
            dt)),
        static_cast<double>(0.00001)));
    // Handling geos returning a MULTILINESTRING
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(0.9),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Union('LINESTRING(2 1, 3 1)', 'LINESTRING(3 -1, 2 -1)'), 'POINT(2 0.1)');)",
            dt)),
        static_cast<double>(0.00001)));
    // Handling geos returning a GEOMETRYCOLLECTION
    EXPECT_GPU_THROW(ASSERT_NEAR(
        static_cast<double>(0.9),
        v<double>(run_simple_agg(
            R"(SELECT ST_Distance(ST_Union('LINESTRING(2 1, 3 1)', 'POINT(2 -1)'), 'POINT(2 0.1)');)",
            dt)),
        static_cast<double>(0.00001)));
    // ST_IsValid: geos validation of SRID-carrying geometries
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_IsValid(gpoly4326) FROM geospatial_test limit 1;)", dt))));
#ifdef DISABLE_THIS_TEST_UNTIL_WE_CAN_FIND_OUT_WHY_ITS_FAILING
    // @TODO(se) this test triggers the exception at GeoIR.cpp:194
    // the output SRID is 900913, but the input SRID is 0
    // disabling this test until we can work out why
    // geos runtime support for input geo transforms
    EXPECT_GPU_THROW(ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT ST_IsEmpty(ST_Transform(gpoly4326, 900913)) FROM geospatial_test limit 1;)",
            dt))));
#endif
    // geos runtime doesn't yet support geometry columns in temporary tables
    EXPECT_THROW(run_simple_agg("SELECT ST_Intersection(SAMPLE(poly), SAMPLE(mpoly)) "
                                "FROM geospatial_test limit 1;",
                                dt),
                 std::runtime_error);
#else
    // geos disabled, expect throws
    EXPECT_THROW(
        run_simple_agg(
            "SELECT ST_Area(ST_Intersection(poly, 'POLYGON((1 1,3 1,3 3,1 3,1 1))')) "
            "FROM geospatial_test WHERE id = 2;",
            dt),
        std::runtime_error);
    EXPECT_THROW(
        run_simple_agg(
            "SELECT ST_Area(ST_Difference(poly, 'POLYGON((1 1,3 1,3 3,1 3,1 1))')) "
            "FROM geospatial_test WHERE id = 2;",
            dt),
        std::runtime_error);
    EXPECT_THROW(
        run_simple_agg("SELECT ST_IsValid(poly) from geospatial_test limit 1;", dt),
        std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT ST_Area(ST_Buffer(poly, 0.1)) "
                                "FROM geospatial_test WHERE id = 2;",
                                dt),
                 std::runtime_error);
#endif
  }
}

class GeoSpatialJoinTablesFixture : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    import_geospatial_test(/*with_temporary_tables=*/GetParam());
    import_geospatial_join_test(/*with_temporary_tables=*/GetParam());
  }

  void TearDown() override {
    if (!GetParam() && !g_keep_data) {
      run_ddl_statement("DROP TABLE IF EXISTS geospatial_test;");
      run_ddl_statement("DROP TABLE IF EXISTS geospatial_inner_join_test;");
    }
  }
};

TEST_P(GeoSpatialJoinTablesFixture, GeoJoins) {
  const auto enable_bbox_intersect_state = g_enable_bbox_intersect_hashjoin;
  g_enable_bbox_intersect_hashjoin = false;
  ScopeGuard reset_state = [&enable_bbox_intersect_state] {
    g_enable_bbox_intersect_hashjoin = enable_bbox_intersect_state;
  };

  // Test loop joins
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  "SELECT a.id FROM geospatial_test a JOIN geospatial_inner_join_test "
                  "b ON ST_Intersects(b.poly, a.poly) ORDER BY a.id;",
                  dt)));

    ASSERT_NO_THROW(run_simple_agg(
        "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test "
        "b ON ST_Contains(b.poly, a.p);",
        dt,
        true,     // geo_return_geo_tv
        false));  // allow_loop_joins

    SKIP_ON_AGGREGATOR(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test "
            "b ON ST_Contains(b.poly, a.p) WHERE b.id = 2;",
            dt,
            true,       // geo_return_geo_tv
            false))));  // allow_loop_joins

    const auto trivial_loop_join_state = g_trivial_loop_join_threshold;
    g_trivial_loop_join_threshold = 1;
    ScopeGuard reset_loop_join_state = [&trivial_loop_join_state] {
      g_trivial_loop_join_threshold = trivial_loop_join_state;
    };

    SKIP_ON_AGGREGATOR(EXPECT_THROW(
        run_multiple_agg(
            "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test "
            "b ON ST_Contains(b.poly, a.p);",
            dt,
            false),
        std::runtime_error));

    // Geometry projection not supported for outer joins
    SKIP_ON_AGGREGATOR(EXPECT_THROW(
        run_multiple_agg(
            "SELECT b.poly FROM geospatial_test a LEFT JOIN geospatial_inner_join_test "
            "b ON ST_Contains(b.poly, a.p);",
            dt,
            false),
        std::runtime_error));
  }

  g_enable_bbox_intersect_hashjoin = true;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // distance joins
    EXPECT_EQ(
        int64_t(26),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) from geospatial_test a, geospatial_inner_join_test b  WHERE ST_Distance(gl4326none, ST_SetSRID(ST_Point(b.id, b.id), 4326)) > 3;)",
            dt)));
    EXPECT_EQ(
        int64_t(20),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) from geospatial_test a, geospatial_inner_join_test b  WHERE ST_Distance(a.gpoly4326, ST_SetSRID(ST_Point(b.id, b.id), 4326)) > 3;)",
            dt)));
    EXPECT_EQ(
        int64_t(20),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) from geospatial_test a, geospatial_inner_join_test b  WHERE ST_Distance(ST_SetSRID(a.mpoly, 4326), ST_SetSRID(ST_Point(b.id, b.id), 4326)) > 3;)",
            dt)));

    // Test query rewrite for simple project
    ASSERT_NO_THROW(run_simple_agg(
        R"(SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);)",
        dt));

    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            R"(SELECT a.id FROM geospatial_test a JOIN geospatial_inner_join_test b ON ST_Intersects(b.poly, a.poly) ORDER BY a.id;)",
            dt)));

    SKIP_ON_AGGREGATOR(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p) WHERE b.id = 2 ORDER BY 1;)",
            dt))));

    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p) WHERE b.id = 4)",
            dt)));
    // re-run to test hash join cache (currently CPU only)
    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p) WHERE b.id = 4;)",
            dt)));

    // with compression
    SKIP_ON_AGGREGATOR(ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            R"(SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Contains(ST_SetSRID(b.poly, 4326), a.gp4326) WHERE b.id = 2 ORDER BY 1;)",
            dt))));

    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Contains(ST_SetSRID(b.poly, 4326), a.gp4326) WHERE b.id = 4;)",
            dt)));

    // enable table reordering, disable loop joins
    const bool table_reordering_state = g_from_table_reordering;
    ScopeGuard table_reordering_reset = [table_reordering_state] {
      g_from_table_reordering = table_reordering_state;
    };
    g_from_table_reordering = true;
    const auto trivial_loop_join_state = g_trivial_loop_join_threshold;
    g_trivial_loop_join_threshold = 1;
    ScopeGuard reset_loop_join_state = [&trivial_loop_join_state] {
      g_trivial_loop_join_threshold = trivial_loop_join_state;
    };

    // constructed point
    ASSERT_EQ(
        static_cast<int64_t>(10),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_test a LEFT JOIN geospatial_inner_join_test b ON ST_Contains(ST_SetSRID(ST_Point(a.id, a.id), 4326), b.gp4326);)",
            dt)));

    EXPECT_NO_THROW(run_multiple_agg(
        R"(SELECT a.id FROM geospatial_test a LEFT JOIN geospatial_inner_join_test b ON ST_Contains(ST_SetSRID(ST_Point(a.id, a.id), 4326), b.gp4326);)",
        dt));

    ASSERT_EQ(
        static_cast<int64_t>(15),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Intersects(ST_SetSRID(ST_Point(a.id, a.id), 4326), b.gp4326);)",
            dt)));
    // contains w/ centroid
    ASSERT_EQ(
        static_cast<int64_t>(35),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test a, geospatial_inner_join_test b WHERE ST_Contains(a.mpoly, ST_Centroid(b.mpoly));)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(36),
        v<int64_t>(run_simple_agg(
            R"(SELECT COUNT(*) FROM geospatial_test a, geospatial_inner_join_test b WHERE ST_Contains(a.gpoly4326, ST_Centroid(b.gp4326));)",
            dt)));
  }
}

INSTANTIATE_TEST_SUITE_P(GeospatialJoinTests,
                         GeoSpatialJoinTablesFixture,
                         ::testing::Values(true, false));

class GeoSpatialMultiFragTestTablesFixture : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    import_geospatial_multi_frag_test(/*with_temporary_tables=*/GetParam());
  }

  void TearDown() override {
    if (!GetParam() && !g_keep_data) {
      run_ddl_statement("DROP TABLE IF EXISTS geospatial_multi_frag_test;");
    }
  }
};

TEST_P(GeoSpatialMultiFragTestTablesFixture, LoopJoin) {
  SKIP_ALL_ON_AGGREGATOR();  // TODO(adb): investigate different result in distributed

  const auto enable_bbox_intersect_state = g_enable_bbox_intersect_hashjoin;
  g_enable_bbox_intersect_hashjoin = false;
  ScopeGuard reset_state = [&enable_bbox_intersect_state] {
    g_enable_bbox_intersect_hashjoin = enable_bbox_intersect_state;
  };

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_EQ(
        static_cast<int64_t>(109),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt, t2.pt) < 10;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(109),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt_none, t2.pt_none) < 10;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(109),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt_comp, t2.pt_comp) < 10;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(65),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt, t2.pt) < 5;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(65),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt_none, t2.pt_none) < 5;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(65),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt_comp, t2.pt_comp) < 5;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(11),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt, t2.pt) < 1;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(11),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt_none, t2.pt_none) < 1;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(11),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt_comp, t2.pt_comp) < 1;)",
            dt)));

    // valid rows: { pt(1 1), pt(2 2), ..., pt(10 10) }
    // invalid rows: { pt(0 0), pt(null null) }
    // expected rows in the resultset:
    // row 1 ~ 10:  zero          | 10 valid rows
    // row 11 ~ 20: 10 valid rows | zero
    // row 21:      zero          | null
    // row 22:      null          | zero
    // row 23:      null          | null
    // total 23 rows
    ASSERT_EQ(
        static_cast<int64_t>(23),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt, t2.pt) is null;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(23),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt_none, t2.pt_none) is null;)",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(23),
        v<int64_t>(run_simple_agg(
            R"(SELECT count(*) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2 WHERE ST_DISTANCE(t1.pt_comp, t2.pt_comp) is null;)",
            dt)));
    ASSERT_NEAR(
        static_cast<double>(14.14213561714551),
        v<double>(run_simple_agg(
            R"(SELECT MAX(ST_DISTANCE(t1.pt, t2.pt)) FROM geospatial_multi_frag_test t1, geospatial_multi_frag_test t2;)",
            dt)),
        static_cast<double>(0.01));
  }
}

// For each of the 120 UTM (curvi-)rectangular zones, test 4326 <-> UTM transformations
// on each of the 4 corners and the center point along the equator.
TEST(GeoSpatial, UTMTransform) {
  unsigned const skip = g_all_utm_zones ? 1 : 30;
  constexpr double eps = 1e-10;
  struct Point {
    double x;
    double y;
  };
  auto const query = [](char dim, Point p, auto from, auto to) {
    std::ostringstream oss;
    oss.precision(17);
    oss << "SELECT ST_" << dim << "(ST_Transform(ST_SetSRID(ST_Point(" << p.x << ','
        << p.y << "), " << from << "), " << to << "));";
    return oss.str();
  };
  auto const transform_point = [&](Point p, auto from, auto to, auto dt) {
    return Point{v<double>(run_simple_agg(query('X', p, from, to), dt, false)),
                 v<double>(run_simple_agg(query('Y', p, from, to), dt, false))};
  };
  // Given (lon,lat) and (utm_x,utm_y,srid) test transformations in both directions.
  // Triangulate results with the 900913 srid.
  auto run_tests = [&](Point wgs, Point utm, unsigned utm_srid, auto dt) {
    Point const wgs_utm = transform_point(wgs, 4326, utm_srid, dt);
    ASSERT_NEAR(utm.x, wgs_utm.x, eps * std::fabs(utm.x));
    ASSERT_NEAR(utm.y, wgs_utm.y, eps * std::fabs(utm.y));

    Point const utm_wgs = transform_point(utm, utm_srid, 4326, dt);
    ASSERT_NEAR(wgs.x, utm_wgs.x, wgs.x ? eps * std::fabs(wgs.x) : 1e-14);
    ASSERT_NEAR(wgs.y, utm_wgs.y, eps * std::fabs(wgs.y));

    Point const web = transform_point(wgs, 4326, 900913, dt);
    Point const utm_web = transform_point(utm, utm_srid, 900913, dt);
    ASSERT_NEAR(web.x, utm_web.x, web.x ? eps * std::fabs(web.x) : 1e-9);
    ASSERT_NEAR(web.y, utm_web.y, eps * std::fabs(web.y));

    Point const wgs_web_utm = transform_point(web, 900913, utm_srid, dt);
    ASSERT_NEAR(utm.x, wgs_web_utm.x, eps * std::fabs(utm.x));
    ASSERT_NEAR(utm.y, wgs_web_utm.y, utm.y ? eps * std::fabs(utm.y) : 1e-8);
  };
  for (auto const dt : {ExecutorDeviceType::GPU, ExecutorDeviceType::CPU}) {
    SKIP_NO_GPU();
    for (bool const is_south : {false, true}) {
      for (unsigned zone = 1; zone <= 60; zone += skip) {
        unsigned const utm_srid = 32600 + is_south * 100 + zone;
        int const x = ((zone - 1u) % 60u) * 6 - 177;  // [-177, 177]
        double const E0 = 500e3;                      // UTM False easting
        double const N0 = is_south ? 10e6 : 0;        // UTM False northing
        // Test values for each zone's equatorial/meridian point to/from UTM coordinates.
        run_tests({x + 0., 0.}, {E0, N0}, utm_srid, dt);
        // Test UTM zone boundary points along equator.
        constexpr double x0 = 333978.55691946047591;
        run_tests({x - 3., 0.}, {E0 - x0, N0}, utm_srid, dt);
        run_tests({x + 3., 0.}, {E0 + x0, N0}, utm_srid, dt);
        if (is_south) {
          // Test points along southern boundary of each UTM zone.
          constexpr double x80s = 58132.215132799166895;
          constexpr double y80s = 1116915.0440516974777;
          run_tests({x - 3., -80.}, {E0 - x80s, y80s}, utm_srid, dt);
          run_tests({x + 3., -80.}, {E0 + x80s, y80s}, utm_srid, dt);
        } else {
          // Test points along northern boundary of each UTM zone.
          constexpr double x84n = 34994.655061136436416;
          constexpr double y84n = 9329005.1824474334717;
          run_tests({x - 3., 84.}, {E0 - x84n, y84n}, utm_srid, dt);
          run_tests({x + 3., 84.}, {E0 + x84n, y84n}, utm_srid, dt);
        }
      }
    }
  }
}

// Disabled because:
//  * CPU throws an exception, GPU returns NaN,
//    and distributed and render modes get these confused.
TEST(GeoSpatial, DISABLED_UTMTransformCoords) {
  constexpr double eps = 1e-10;
  // Length of 1-degree arc along equator centered at longitude=3 according to UTM.
  constexpr double one_degree_in_meters = 111276.3876347362;
  char const* const query =
      "SELECT ST_LENGTH(ST_TRANSFORM(ST_GeogFromText('LINESTRING(2.5 0, 3.5 0)', 4326), "
      "32601));";
  for (auto const dt : {ExecutorDeviceType::GPU, ExecutorDeviceType::CPU}) {
    SKIP_NO_GPU();
    ASSERT_NEAR(one_degree_in_meters, v<double>(run_simple_agg(query, dt, false)), eps);
  }
}

INSTANTIATE_TEST_SUITE_P(GeospatialMultiFragExecutionTests,
                         GeoSpatialMultiFragTestTablesFixture,
                         ::testing::Values(true, false));

class GeoSpatialConditionalExpressionTest : public ExecutorDeviceParameterizedTest {
 public:
  static void SetUpTestSuite() { import_geospatial_test(false); }

  static void TearDownTestSuite() {
    if (!g_keep_data) {
      run_ddl_statement("DROP TABLE IF EXISTS geospatial_test;");
    }
  }

  void queryAndAssertGeoExprError(const std::string& query) {
    try {
      run_multiple_agg(query, device_type_);
      FAIL() << "An exception should have been thrown for this test case";
    } catch (const TDBException& e) {
      assertGeoExprErrorMessage(e.error_msg);
    } catch (const std::exception& e) {
      assertGeoExprErrorMessage(e.what());
    }
  }

  void assertGeoExprErrorMessage(const std::string& error_message) {
    ASSERT_TRUE(
        error_message.find("Geospatial column projections are currently not supported in "
                           "conditional expressions.") != std::string::npos);
  }

  bool gpusPresent() override { return QR::get()->gpusPresent(); }
};

TEST_P(GeoSpatialConditionalExpressionTest, Coalesce) {
  queryAndAssertGeoExprError("SELECT coalesce(p_null, p) FROM geospatial_test;");
}

TEST_P(GeoSpatialConditionalExpressionTest, CaseExpr) {
  queryAndAssertGeoExprError(
      "SELECT CASE WHEN MOD(id, 2) = 1 THEN p ELSE p_null END FROM geospatial_test;");
}

TEST_P(GeoSpatialConditionalExpressionTest, CaseExprNullWhenCondition) {
  queryAndAssertGeoExprError(
      "SELECT CASE WHEN MOD(id, 2) = 1 THEN NULL ELSE p END FROM geospatial_test;");
}

TEST_P(GeoSpatialConditionalExpressionTest, CaseExprNullElseCondition) {
  queryAndAssertGeoExprError(
      "SELECT CASE WHEN MOD(id, 2) = 1 THEN p ELSE NULL END FROM geospatial_test;");
}

TEST_P(GeoSpatialConditionalExpressionTest, CaseExprNullBetweenThenAndElseCondition) {
  queryAndAssertGeoExprError(
      "SELECT CASE WHEN MOD(id, 2) = 1 THEN p WHEN id = 1 THEN NULL ELSE p_null END FROM "
      "geospatial_test;");
}

INSTANTIATE_TEST_SUITE_P(CpuAndGpuExecutorDevices,
                         GeoSpatialConditionalExpressionTest,
                         ::testing::Values(ExecutorDeviceType::CPU,
                                           ExecutorDeviceType::GPU),
                         ::testing::PrintToStringParamName());

TEST(GeoSpatial, ProjectGeoColAfterLeftJoin) {
  auto drop_tables_ddl = []() {
    run_ddl_statement("DROP TABLE IF EXISTS LFJ;");
    run_ddl_statement("DROP TABLE IF EXISTS AllGeoTypes;");
  };
  drop_tables_ddl();
  run_ddl_statement("CREATE TABLE LFJ (v INT);");
  std::string all_geo_types_ddl =
      "CREATE TABLE AllGeoTypes (\n"
      "v int,\n"
      "pt point,\n"
      "gpt geometry(point),\n"
      "gpt4 geometry(point, 4326),\n"
      "gpt4e geometry(point, 4326) encoding compressed (32),\n"
      "gpt4n geometry(point, 4326) encoding none,\n"
      "gpt9 geometry(point, 900913),\n"
      "mpt multipoint,\n"
      "gmpt geometry(multipoint),\n"
      "gmpt4 geometry(multipoint, 4326),\n"
      "gmpt4e geometry(multipoint, 4326) encoding compressed (32),\n"
      "gmpt4n geometry(multipoint, 4326) encoding none,\n"
      "gmpt9 geometry(multipoint, 900913),\n"
      "l linestring,\n"
      "gl geometry(linestring),\n"
      "gl4 geometry(linestring, 4326),\n"
      "gl4e geometry(linestring, 4326) encoding compressed (32),\n"
      "gl4n geometry(linestring, 4326) encoding none,\n"
      "gl9 geometry(linestring, 900913),\n"
      "gl9n geometry(linestring, 900913) encoding none,\n"
      "ml multilinestring,\n"
      "gml geometry(multilinestring),\n"
      "gml4 geometry(multilinestring, 4326),\n"
      "gml4e geometry(multilinestring, 4326) encoding compressed (32),\n"
      "gml4n geometry(multilinestring, 4326) encoding none,\n"
      "gml9 geometry(multilinestring, 900913),\n"
      "gml9n geometry(multilinestring, 900913) encoding none,\n"
      "p polygon,\n"
      "gp geometry(polygon),\n"
      "gp4 geometry(polygon, 4326),\n"
      "gp4e geometry(polygon, 4326) encoding compressed (32),\n"
      "gp4n geometry(polygon, 4326) encoding none,\n"
      "gp9 geometry(polygon, 900913),\n"
      "gp9n geometry(polygon, 900913) encoding none,\n"
      "mp multipolygon,\n"
      "gmp geometry(multipolygon),\n"
      "gmp4 geometry(multipolygon, 4326),\n"
      "gmp4e geometry(multipolygon, 4326) encoding compressed (32),\n"
      "gmp4n geometry(multipolygon, 4326) encoding none,\n"
      "gmp9 geometry(multipolygon, 900913),\n"
      "gmp9n geometry(multipolygon, 900913) encoding none)";
  std::string replicated_dec{!g_cluster ? "" : " WITH(PARTITIONS='REPLICATED');"};
  all_geo_types_ddl += replicated_dec;
  run_ddl_statement(all_geo_types_ddl);

  run_multiple_agg("INSERT INTO LFJ VALUES (1);", ExecutorDeviceType::CPU);
  run_multiple_agg("INSERT INTO LFJ VALUES (2);", ExecutorDeviceType::CPU);
  run_multiple_agg(
      "insert into AllGeoTypes values (\n"
      "1,\n"
      "'POINT(3 3)', 'POINT(3 3)', 'POINT(3 3)', 'POINT(3 3)', 'POINT(3 3)', 'POINT(3 "
      "3)',\n"
      "'MULTIPOINT(3 3, 5 5)', 'MULTIPOINT(3 3, 5 5)', 'MULTIPOINT(3 3, 5 5)', "
      "'MULTIPOINT(3 3, 5 5)', 'MULTIPOINT(3 3, 5 5)', 'MULTIPOINT(3 3, 5 5)',\n"
      "'LINESTRING(3 0, 6 6, 7 7)', 'LINESTRING(3 0, 6 6, 7 7)', 'LINESTRING(3 0, 6 6, 7 "
      "7)', 'LINESTRING(3 0, 6 6, 7 7)', 'LINESTRING(3 0, 6 6, 7 7)', 'LINESTRING(3 0, 6 "
      "6, 7 7)', 'LINESTRING(3 0, 6 6, 7 7)',\n"
      "'MULTILINESTRING((3 0, 6 6, 7 7),(3.2 4.2,5.1 5.2))', 'MULTILINESTRING((3 0, 6 6, "
      "7 7),(3.2 4.2,5.1 5.2))', 'MULTILINESTRING((3 0, 6 6, 7 7),(3.2 4.2,5.1 5.2))', "
      "'MULTILINESTRING((3 0, 6 6, 7 7),(3.2 4.2,5.1 5.2))', 'MULTILINESTRING((3 0, 6 6, "
      "7 7),(3.2 4.2,5.1 5.2))', 'MULTILINESTRING((3 0, 6 6, 7 7),(3.2 4.2,5.1 5.2))', "
      "'MULTILINESTRING((3 0, 6 6, 7 7),(3.2 4.2,5.1 5.2))',\n"
      "'POLYGON((0 0, 4 0, 0 4, 0 0))', 'POLYGON((0 0, 4 0, 0 4, 0 0))', 'POLYGON((0 0, "
      "4 0, 0 4, 0 0))', 'POLYGON((0 0, 4 0, 0 4, 0 0))', 'POLYGON((0 0, 4 0, 0 4, 0 "
      "0))', 'POLYGON((0 0, 4 0, 0 4, 0 0))', 'POLYGON((0 0, 4 0, 0 4, 0 0))',\n"
      "'MULTIPOLYGON(((0 0, 4 0, 0 4, 0 0)))', 'MULTIPOLYGON(((0 0, 4 0, 0 4, 0 0)))', "
      "'MULTIPOLYGON(((0 0, 4 0, 0 4, 0 0)))', 'MULTIPOLYGON(((0 0, 4 0, 0 4, 0 0)))', "
      "'MULTIPOLYGON(((0 0, 4 0, 0 4, 0 0)))', 'MULTIPOLYGON(((0 0, 4 0, 0 4, 0 0)))', "
      "'MULTIPOLYGON(((0 0, 4 0, 0 4, 0 0)))'\n"
      ");",
      ExecutorDeviceType::CPU);

  auto query1_gen = [](const std::string& col_name) {
    std::ostringstream oss;
    oss << "SELECT COUNT(1) FROM LFJ R LEFT JOIN AllGeoTypes S ON R.v = S.v WHERE S."
        << col_name << " IS NULL";
    return oss.str();
  };

  auto query2_gen = [](const std::string& col_name) {
    std::ostringstream oss;
    oss << "SELECT R.v, S." << col_name
        << " FROM LFJ R LEFT JOIN AllGeoTypes S ON R.v = S.v";
    return oss.str();
  };

  for (std::string col_name :
       {"pt",    "gpt",    "gpt4",   "gpt4e", "gpt4n", "gpt9",  "mpt",   "gmpt",
        "gmpt4", "gmpt4e", "gmpt4n", "gmpt9", "l",     "gl",    "gl4",   "gl4e",
        "gl4n",  "gl9",    "gl9n",   "ml",    "gml",   "gml4",  "gml4n", "gml4e",
        "gml9",  "gml9n",  "p",      "gp",    "gp4",   "gp4e",  "gp4n",  "gp9",
        "gp9n",  "mp",     "gmp",    "gmp4",  "gmp4n", "gmp4e", "gmp9",  "gmp9n"}) {
    for (auto const dt : {ExecutorDeviceType::GPU, ExecutorDeviceType::CPU}) {
      SKIP_NO_GPU();
      {
        EXPECT_EQ(static_cast<int64_t>(1),
                  v<int64_t>(run_simple_agg(query1_gen(col_name), dt)));
        const auto rows = run_multiple_agg(query2_gen(col_name), dt);
        EXPECT_EQ(rows->rowCount(), (size_t)2);
        const auto row1 = rows->getNextRow(false, false);
        const auto row2 = rows->getNextRow(false, false);
        EXPECT_EQ(row1.size(), size_t(2));
        EXPECT_EQ(row2.size(), size_t(2));
      }
    }
  }

  if (!g_keep_data) {
    drop_tables_ddl();
  }
}

TEST_P(GeoSpatialTestTablesFixture, Cast) {
  if (g_aggregator) {
    GTEST_SKIP() << "TDBException(error_msg=Cast to dictionary-encoded string type "
                    "not supported for distributed queries)";
  }
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    char const* query =
        "SELECT CAST(ST_X(p) AS TEXT)='5.000000' FROM geospatial_test WHERE ST_X(p)=5;";
    ASSERT_EQ(int64_t(1), v<int64_t>(run_simple_agg(query, dt)));
  }
}

using TestParamType =
    std::tuple<ExecutorDeviceType, std::string, std::string, std::string>;
class GeoFunctionsParamTypeTest : public ::testing::TestWithParam<TestParamType> {
 public:
  static void SetUpTestSuite() {
    std::string create_table_statement;
    for (const auto& column_name_and_type : column_names_and_types_) {
      auto column_name = column_name_and_type[0], column_type = column_name_and_type[1];
      if (create_table_statement.empty()) {
        create_table_statement =
            "CREATE TABLE geo_table (" + column_name + " " + column_type;
      } else {
        create_table_statement += "," + column_name + " " + column_type;
      }
    }
    create_table_statement += ");";
    run_ddl_statement(create_table_statement);

    run_multiple_agg(
        "INSERT INTO geo_table VALUES "
        "  ('POINT(2 1)', 'MULTIPOINT(2 1)', 'LINESTRING(1 1, 2 1, 3 1)', "
        "   'MULTILINESTRING((1 1, 2 1, 3 1))', 'POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))', "
        "   'MULTIPOLYGON(((0 0, 4 0, 4 4, 0 4, 0 0)))'), "
        "  ('POINT(0 0)', 'MULTIPOINT(10 10)', 'LINESTRING(50 50, 60 50, 70 50)', "
        "   'MULTILINESTRING((50 100, 60 100, 70 100))', "
        "   'POLYGON((50 150, 60 150, 60 200, 50 200, 50 150))', "
        "   'MULTIPOLYGON(((50 300, 60 300, 60 350, 50 350, 50 300)))');",
        ExecutorDeviceType::CPU);
  }

  static void TearDownTestSuite() { run_ddl_statement("DROP TABLE geo_table;"); }

  void SetUp() override {
    auto device_type = getDeviceType();
#ifdef HAVE_CUDA
    bool skip_test =
        (device_type == ExecutorDeviceType::GPU && !QR::get()->gpusPresent());
#else
    bool skip_test = (device_type == ExecutorDeviceType::GPU);
#endif
    if (skip_test) {
      GTEST_SKIP() << "Unsupported device type: " << device_type;
    }
  }

  std::string getGeoPredicate() {
    const auto& [_, function_name, left_operand_type, right_operand_type] = GetParam();
    auto predicate = function_name + "(" + getColumnName(left_operand_type) + "," +
                     getColumnName(right_operand_type) + ")";
    if (function_name == "ST_DISTANCE") {
      predicate += " < 1.0";
    }
    return predicate;
  }

  int64_t getResultCount() {
    int64_t count;
    if (mapContainsParam(true_result_mapping_)) {
      count = 2;
    } else if (mapContainsParam(false_result_mapping_)) {
      count = 0;
    } else {
      // Table rows were created in such a way that most queries should return a count
      // of 1.
      count = 1;
    }
    return count;
  }

  using ParamMapType =
      std::map<std::string, std::map<std::string, std::vector<std::string>>>;
  bool mapContainsParam(const ParamMapType& param_map) {
    const auto& [_, function_name, left_operand_type, right_operand_type] = GetParam();
    if (param_map.count(function_name) > 0) {
      const auto& type_map = shared::get_from_map(param_map, function_name);
      if (type_map.count(left_operand_type) > 0) {
        auto right_operand_types = shared::get_from_map(type_map, left_operand_type);
        return shared::contains(right_operand_types, right_operand_type);
      }
    }
    return false;
  }

  std::string getColumnName(const std::string& column_type) {
    for (const auto& column_name_and_type : column_names_and_types_) {
      if (column_name_and_type[1] == column_type) {
        return column_name_and_type[0];
      }
    }
    UNREACHABLE() << "Could not find column name for type: " << column_type;
    return {};
  }

  ExecutorDeviceType getDeviceType() {
    return std::get<0>(GetParam());
  }

  static std::vector<std::string> getGeoColumnTypes() {
    std::vector<std::string> column_types;
    for (const auto& column_name_and_type : column_names_and_types_) {
      column_types.emplace_back(column_name_and_type[1]);
    }
    return column_types;
  }

  static std::string printTestParams(
      const ::testing::TestParamInfo<TestParamType>& info) {
    const auto& param = info.param;
    std::stringstream ss;
    ss << std::get<0>(param) << "_" << std::get<1>(param) << "_" << std::get<2>(param)
       << "_" << std::get<3>(param);
    return ss.str();
  }

  static inline const std::vector<std::vector<std::string>> column_names_and_types_{
      {"pt1", "POINT"},
      {"pt2", "MULTIPOINT"},
      {"ls1", "LINESTRING"},
      {"ls2", "MULTILINESTRING"},
      {"py1", "POLYGON"},
      {"py2", "MULTIPOLYGON"}};

  // Use cases/geo type permutations that are currently supported.
  static inline const ParamMapType supported_mapping_{
      {"ST_CONTAINS",
       {{"POINT", {"POINT", "LINESTRING", "POLYGON"}},
        {"LINESTRING", {"POINT", "POLYGON"}},
        {"POLYGON", {"POINT", "LINESTRING", "POLYGON"}},
        {"MULTIPOLYGON", {"POINT", "LINESTRING"}}}},
      {"ST_WITHIN",
       {{"POINT", {"POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"LINESTRING", {"POINT", "POLYGON", "MULTIPOLYGON"}},
        {"POLYGON", {"POINT", "LINESTRING", "POLYGON"}}}},
      {"ST_DISTANCE",
       {{"POINT",
         {"POINT",
          "MULTIPOINT",
          "LINESTRING",
          "MULTILINESTRING",
          "POLYGON",
          "MULTIPOLYGON"}},
        {"MULTIPOINT", {"POINT", "MULTIPOINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"LINESTRING", {"POINT", "MULTIPOINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"MULTILINESTRING", {"POINT"}},
        {"POLYGON", {"POINT", "MULTIPOINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"MULTIPOLYGON",
         {"POINT", "MULTIPOINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}}}},
      {"ST_DISJOINT",
       {{"POINT", {"POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"LINESTRING", {"POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"POLYGON", {"POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"MULTIPOLYGON", {"POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}}}},
      {"ST_INTERSECTS",
       {{"POINT", {"POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"LINESTRING", {"POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"POLYGON", {"POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}},
        {"MULTIPOLYGON", {"POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON"}}}}};

  // Use cases that always return true.
  static inline const ParamMapType true_result_mapping_{
      {"ST_CONTAINS", {{"POINT", {"POINT"}}}},  // Always true for the same point.
      {"ST_WITHIN", {{"POINT", {"POINT"}}}},    // Always true for the same point.
      {"ST_DISTANCE",  // Same values will always have a distance of 0.
       {{"POINT", {"POINT"}},
        {"MULTIPOINT", {"MULTIPOINT"}},
        {"LINESTRING", {"LINESTRING"}},
        {"MULTILINESTRING", {"MULTILINESTRING"}},
        {"POLYGON", {"POLYGON"}},
        {"MULTIPOLYGON", {"MULTIPOLYGON"}}}},
      {"ST_INTERSECTS",  // Values always intersect with themselves.
       {{"POINT", {"POINT"}},
        {"LINESTRING", {"LINESTRING"}},
        {"POLYGON", {"POLYGON"}},
        {"MULTIPOLYGON", {"MULTIPOLYGON"}}}}};

  // Use cases that always return false.
  static inline const ParamMapType false_result_mapping_{
      {"ST_CONTAINS",  // Smaller geo types cannot contain larger geo types.
       {{"POINT", {"LINESTRING", "POLYGON"}},
        {"LINESTRING", {"POLYGON"}},
        {"POLYGON", {"POLYGON"}}}},  // TODO: This case should return true
      {"ST_WITHIN",                  // Smaller geo types cannot contain larger geo types.
       {{"LINESTRING", {"POINT"}},
        {"POLYGON", {"POINT", "LINESTRING", "POLYGON"}}}},  // TODO: POLYGON/POLYGON case
                                                            // should return true
      {"ST_DISJOINT",  // Values cannot be disjoint with themselves.
       {{"POINT", {"POINT"}},
        {"LINESTRING", {"LINESTRING"}},
        {"POLYGON", {"POLYGON"}},
        {"MULTIPOLYGON", {"MULTIPOLYGON"}}}}};
};

class SupportedGeoFunctionsParamTypeTest : public GeoFunctionsParamTypeTest {
 public:
  static std::vector<TestParamType> getTestParams() {
    std::vector<TestParamType> result;
    for (auto device_type : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      for (const auto& [function_name, type_map] : supported_mapping_) {
        for (const auto& [type, values] : type_map) {
          for (const auto& value : values) {
            result.emplace_back(device_type, function_name, type, value);
          }
        }
      }
    }
    return result;
  }
};

TEST_P(SupportedGeoFunctionsParamTypeTest, AllSupportedTypes) {
  auto query = "SELECT count(*) FROM geo_table WHERE " + getGeoPredicate() + ";";
  ASSERT_EQ(getResultCount(), v<int64_t>(run_simple_agg(query, getDeviceType())));
}

INSTANTIATE_TEST_SUITE_P(
    OperatorsAndColumnTypes,
    SupportedGeoFunctionsParamTypeTest,
    ::testing::ValuesIn(SupportedGeoFunctionsParamTypeTest::getTestParams()),
    GeoFunctionsParamTypeTest::printTestParams);

class UnsupportedGeoFunctionsParamTypeTest : public GeoFunctionsParamTypeTest {
 public:
  static std::vector<TestParamType> getTestParams() {
    auto column_types = getGeoColumnTypes();
    std::vector<TestParamType> result;
    for (auto device_type : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      for (const auto& [function_name, type_map] : supported_mapping_) {
        for (const auto& left_operand_type : column_types) {
          if (type_map.find(left_operand_type) == type_map.end()) {
            // Include all geo column types for the right operand in the test, if the geo
            // function does not support this column type.
            for (const auto& right_operand_type : column_types) {
              result.emplace_back(
                  device_type, function_name, left_operand_type, right_operand_type);
            }
          } else {
            const auto& right_operand_types =
                shared::get_from_map(type_map, left_operand_type);
            for (const auto& right_operand_type : column_types) {
              // Include geo column types that are missing from the supported types list.
              if (!shared::contains(right_operand_types, right_operand_type)) {
                result.emplace_back(
                    device_type, function_name, left_operand_type, right_operand_type);
              }
            }
          }
        }
      }
    }
    return result;
  }
};

TEST_P(UnsupportedGeoFunctionsParamTypeTest, AllUnsupportedTypes) {
  try {
    auto query = "SELECT count(*) FROM geo_table WHERE " + getGeoPredicate() + ";";
    run_simple_agg(query, getDeviceType());
    FAIL() << "An exception should have been thrown for this test case. Query: " << query;
  } catch (const std::runtime_error& e) {
    const auto& [_, function_name, left_operand_type, right_operand_type] = GetParam();
    // In certain cases, ST_WITHIN is converted to ST_CONTAINS, and ST_DISJOINT is
    // converted to ST_INTERSECT. See RelAlgTranslator::translateBinaryGeoFunction.
    std::string executed_function_name;
    if (function_name == "ST_WITHIN") {
      executed_function_name = "ST_CONTAINS";
    } else if (function_name == "ST_DISJOINT") {
      executed_function_name = "ST_INTERSECTS";
    } else {
      executed_function_name = function_name;
    }
    auto error_regex =
        "(" + (".*Function\\s" + executed_function_name + ".*not\\ssupported.*") + "|" +
        (".*" + function_name +
         "\\scurrently\\sdoesn't\\ssupport\\sthis\\sargument\\scombination.*") +
        ")";
    std::string actual_error{e.what()};
    ASSERT_TRUE(
        std::regex_match(actual_error, std::regex{error_regex, std::regex::icase}))
        << "Error: " << actual_error << ", Regex: " << error_regex;
  }
}

INSTANTIATE_TEST_SUITE_P(
    OperatorsAndColumnTypes,
    UnsupportedGeoFunctionsParamTypeTest,
    ::testing::ValuesIn(UnsupportedGeoFunctionsParamTypeTest::getTestParams()),
    GeoFunctionsParamTypeTest::printTestParams);

int main(int argc, char** argv) {
  g_is_test_env = true;

  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("keep-data", "Don't drop tables at the end of the tests");
  desc.add_options()("all-utm-zones", "Test all 120 UTM zones");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  logger::init(log_options);

  if (vm.count("keep-data")) {
    g_keep_data = true;
  }
  g_all_utm_zones = vm.count("all-utm-zones");

  // disable CPU retry to catch illegal code generation on GPU
  g_allow_cpu_retry = false;
  g_allow_query_step_cpu_retry = false;

  QR::init(BASE_PATH);

  int err{0};
  try {
    ScopeGuard resetFlag = [orig = g_allow_cpu_retry] {
      // sample on varlen col on multi-fragmented table is now
      // punt to cpu, so we turn cpu_retry on to properly perform the test
      g_allow_cpu_retry = orig;
    };
    g_allow_cpu_retry = true;
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  // TODO: drop tables
  QR::reset();
  return err;
}
