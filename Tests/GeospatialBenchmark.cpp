/*
 * Copyright 2021 OmniSci, Inc.
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

#include <benchmark/benchmark.h>

#include "../QueryRunner/QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

std::once_flag setup_flag;
void global_setup() {
  TestHelpers::init_logger_stderr_only();
  QR::init(BASE_PATH);
}

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(
      query_str, device_type, /*hoist_literals=*/true, /*allow_loop_joins=*/true);
}

TargetValue run_simple_agg(const std::string& query_str,
                           const ExecutorDeviceType device_type) {
  auto rows = QR::get()->runSQL(query_str, device_type, /*allow_loop_joins=*/true);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
}

class GeospatialMathFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    std::call_once(setup_flag, global_setup);
  }

  void TearDown(const ::benchmark::State& state) override {
    // noop
  }
};

void run_distance_math_benchmark(const ExecutorDeviceType dt) {
  // point point
  run_multiple_agg(R"(SELECT ST_Distance('POINT(0 0)', 'POINT(2 2)');)", dt);
  // point point geodesic
  run_multiple_agg(
      R"(SELECT ST_Distance(ST_GeogFromText('POINT(0 0)', 4326), ST_GeogFromText('POINT(2 2)', 4326));)",
      dt);
  // point linestring
  run_multiple_agg(R"(SELECT ST_Distance('POINT(0 0)', 'LINESTRING(-2 2, 2 2, 2 0)');)",
                   dt);
  // point polygon (inside)
  run_multiple_agg(
      R"(SELECT ST_Distance('POINT(0 0)', 'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))');)",
      dt);
  // point multipolygon (outside)
  run_multiple_agg(
      R"(SELECT ST_Distance('POINT(-4 -4)', 'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))');)",
      dt);
  // linestring linestring
  run_multiple_agg(
      R"(SELECT ST_Distance('LINESTRING(-2 2, 2 2, 2 0)', 'LINESTRING(4 0, 0 -4, -4 0, 0 5)');)",
      dt);
  // linestring polygon (inside)
  run_multiple_agg(
      R"(SELECT ST_Distance('LINESTRING(-2 2, 2 2, 2 0)', 'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))');)",
      dt);
  // linestring multipolygon
  run_multiple_agg(
      R"(SELECT ST_Distance('LINESTRING(-8 8, -7 7, -6 3)', 'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))');)",
      dt);
  // polygon polygon
  run_multiple_agg(
      R"(SELECT ST_Distance('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))', 'POLYGON((0.5 0.5, -0.5 0.5, -0.5 -0.5, 0.5 -0.5, 0.5 0.5))');)",
      dt);
  // polygon multipolygon
  run_multiple_agg(
      R"(SELECT ST_Distance('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))', 'MULTIPOLYGON(((4 2, 5 3, 4 3)), ((3 3, 4 3, 3 4)))');)",
      dt);
  // multipoly multipoly
  run_multiple_agg(
      R"(SELECT ST_Distance('MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), ((1 1, -1 1, -1 -1, 1 -1, 1 1)))', 'MULTIPOLYGON(((4 2, 5 3, 4 3)), ((3 3, 4 3, 3 4)))');)",
      dt);
}

BENCHMARK_DEFINE_F(GeospatialMathFixture, GeospatialDistanceCPU)
(benchmark::State& state) {
  // warmup: we want to measure JIT performance, not data load
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
  run_distance_math_benchmark(ExecutorDeviceType::CPU);

  for (auto _ : state) {
    run_distance_math_benchmark(ExecutorDeviceType::CPU);
  }
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
}

BENCHMARK_DEFINE_F(GeospatialMathFixture, GeospatialDistanceGPU)
(benchmark::State& state) {
  // warmup: we want to measure JIT performance, not data load
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
  run_distance_math_benchmark(ExecutorDeviceType::GPU);

  for (auto _ : state) {
    run_distance_math_benchmark(ExecutorDeviceType::GPU);
  }
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
}

BENCHMARK_REGISTER_F(GeospatialMathFixture, GeospatialDistanceCPU)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(GeospatialMathFixture, GeospatialDistanceGPU)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// TODO: de-dupe w/ GeospatialTest
void import_geospatial_test(const bool use_temporary_tables, const size_t num_rows) {
  const std::string geospatial_test("DROP TABLE IF EXISTS geospatial_test;");
  run_ddl_statement(geospatial_test);
  const auto create_ddl = TestHelpers::build_create_table_statement(
      R"(id INT, p POINT, l LINESTRING, poly POLYGON, mpoly MULTIPOLYGON, gp GEOMETRY(POINT), gp4326 GEOMETRY(POINT,4326) ENCODING COMPRESSED(32), gp4326none GEOMETRY(POINT,4326) ENCODING NONE, gp900913 GEOMETRY(POINT,900913), gl4326none GEOMETRY(LINESTRING,4326) ENCODING NONE, gpoly4326 GEOMETRY(POLYGON,4326), gpoly900913 GEOMETRY(POLYGON,900913))",
      "geospatial_test",
      {},
      2,
      /*use_temporary_tables=*/use_temporary_tables,
      /*is_replicated=*/false);
  run_ddl_statement(create_ddl);
  TestHelpers::ValuesGenerator gen("geospatial_test");
  for (size_t i = 0; i < num_rows; ++i) {
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
    run_multiple_agg(gen(i,
                         point,
                         linestring,
                         poly,
                         mpoly,
                         point,
                         point,
                         point,
                         point,
                         linestring,
                         poly,
                         poly),
                     ExecutorDeviceType::CPU);
  }
}
class GeospatialTableFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    std::call_once(setup_flag, global_setup);

    // TODO: perhaps using temporary tables is a way to drop the overhead of disk w/ the
    // same accessors used in codegen?
    import_geospatial_test(/*use_temporary_tables=*/false, /*num_rows=*/35);
  }

  void TearDown(const ::benchmark::State& state) override {
    run_ddl_statement("DROP TABLE IF EXISTS geospatial_test");
  }
};

void run_distance_table_benchmark(const ExecutorDeviceType dt) {
  // point point
  run_multiple_agg(R"(SELECT ST_Distance(p, p) FROM geospatial_test;)", dt);
  // point point geodesic
  run_multiple_agg(R"(SELECT ST_Distance(gp4326, gp4326) FROM geospatial_test;)", dt);
  // point linestring
  run_multiple_agg(R"(SELECT ST_Distance(p, l) FROM geospatial_test;)", dt);
  // point polygon
  run_multiple_agg(R"(SELECT ST_Distance(p, poly) FROM geospatial_test;)", dt);
  // point polygon geodesic
  run_multiple_agg(R"(SELECT ST_Distance(gp4326, gpoly4326) FROM geospatial_test;)", dt);
  // point multipolygon
  run_multiple_agg(R"(SELECT ST_Distance(p, mpoly) FROM geospatial_test;)", dt);
  // linestring linestring
  run_multiple_agg(R"(SELECT ST_Distance(l ,l) FROM geospatial_test;)", dt);
  // linestring polygon
  run_multiple_agg(R"(SELECT ST_Distance(l, poly) FROM geospatial_test;)", dt);
  // linestring multipolygon
  run_multiple_agg(R"(SELECT ST_Distance(l, mpoly) FROM geospatial_test;)", dt);
  // polygon polygon
  run_multiple_agg(R"(SELECT ST_Distance(poly, poly) FROM geospatial_test;)", dt);
  // polygon multipolygon
  run_multiple_agg(R"(SELECT ST_Distance(poly, mpoly) FROM geospatial_test;)", dt);
  // multipoly multipoly
  run_multiple_agg(R"(SELECT ST_Distance(mpoly, mpoly) FROM geospatial_test;)", dt);
}

BENCHMARK_DEFINE_F(GeospatialTableFixture, GeospatialDistanceCPU)
(benchmark::State& state) {
  // warmup: we want to measure JIT performance, not data load
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
  run_distance_table_benchmark(ExecutorDeviceType::CPU);

  for (auto _ : state) {
    run_distance_table_benchmark(ExecutorDeviceType::CPU);
  }
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
}

BENCHMARK_DEFINE_F(GeospatialTableFixture, GeospatialDistanceGPU)
(benchmark::State& state) {
  // warmup: we want to measure JIT performance, not data load
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
  run_distance_table_benchmark(ExecutorDeviceType::GPU);

  for (auto _ : state) {
    run_distance_table_benchmark(ExecutorDeviceType::GPU);
  }
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
}

BENCHMARK_REGISTER_F(GeospatialTableFixture, GeospatialDistanceCPU)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(GeospatialTableFixture, GeospatialDistanceGPU)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(GeospatialTableFixture, TransformPointCPU)(benchmark::State& state) {
  // warmup: we want to measure JIT performance, not data load
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
  run_multiple_agg(
      R"(SELECT ST_X(ST_Transform(gp4326, 900913)), ST_Y(ST_Transform(gp4326, 900913)) from geospatial_test;)",
      ExecutorDeviceType::CPU);

  for (auto _ : state) {
    run_multiple_agg(
        R"(SELECT ST_X(ST_Transform(gp4326, 900913)), ST_Y(ST_Transform(gp4326, 900913)) from geospatial_test;)",
        ExecutorDeviceType::CPU);
  }
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
}

BENCHMARK_DEFINE_F(GeospatialTableFixture, TransformPointGPU)(benchmark::State& state) {
  // warmup: we want to measure JIT performance, not data load
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
  run_multiple_agg(
      R"(SELECT ST_X(ST_Transform(gp4326, 900913)), ST_Y(ST_Transform(gp4326, 900913)) from geospatial_test;)",
      ExecutorDeviceType::GPU);

  for (auto _ : state) {
    run_multiple_agg(
        R"(SELECT ST_X(ST_Transform(gp4326, 900913)), ST_Y(ST_Transform(gp4326, 900913)) from geospatial_test;)",
        ExecutorDeviceType::GPU);
  }
  QR::get()->clearCpuMemory();
  QR::get()->clearGpuMemory();
}

BENCHMARK_REGISTER_F(GeospatialTableFixture, TransformPointCPU)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(GeospatialTableFixture, TransformPointGPU)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
