/*
 * Copyright 2020 OmniSci, Inc.
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
#include <iostream>
#include <mutex>
#include <thread>

#include "../ImportExport/Importer.h"
#include "../Logger/Logger.h"
#include "../QueryEngine/ResultSet.h"
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

class UpdateFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    std::call_once(setup_flag, global_setup);

    run_ddl_statement("DROP TABLE IF EXISTS update_bench_1;");
    run_ddl_statement(
        "CREATE TABLE update_bench_1 (x INT, y DOUBLE, z BOOLEAN, str TEXT ENCODING "
        "DICT(16)) WITH (FRAGMENT_SIZE=100);");

    std::array<std::string, 15> str_values{"foo",
                                           "bar",
                                           "baz",
                                           "hello",
                                           "world",
                                           "the",
                                           "quick",
                                           "brown",
                                           "fox",
                                           "jumped",
                                           "over",
                                           "lazy",
                                           "dog",
                                           "today",
                                           "tomorrow"};

    // TODO(adb): cleanup this interface a bit, as we're half in parser land, half in
    // direct API lands
    auto cat = QR::get()->getCatalog();
    const auto td = cat->getMetadataForTable("update_bench_1");
    CHECK(td);
    auto loader = QR::get()->getLoader(td);
    CHECK(loader);

    auto col_descs = loader->get_column_descs();
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
    for (auto cd : col_descs) {
      import_buffers.push_back(std::unique_ptr<import_export::TypedImportBuffer>(
          new import_export::TypedImportBuffer(cd, loader->getStringDict(cd))));
    }

    for (int64_t i = 0; i < state.range(0); i++) {
      std::vector<std::string> values{
          std::to_string(i % 10000),
          std::to_string(1.1 * (i % 10000)),
          i % 2 == 0 ? std::string{"true"} : std::string{"false"},
          str_values[i % str_values.size()]};
      size_t index = 0;
      for (auto cd : col_descs) {
        CHECK_LT(index, values.size());
        CHECK_LT(index, import_buffers.size());
        import_buffers[index]->add_value(cd,
                                         values[index],
                                         /*is_null=*/false,
                                         import_export::CopyParams());
        index++;
      }
    }

    loader->load(import_buffers, state.range(0), nullptr);

    // make sure we're warmed up
    run_multiple_agg("SELECT * FROM update_bench_1;", ExecutorDeviceType::CPU);
    CHECK_EQ(static_cast<int64_t>(state.range(0)),
             TestHelpers::v<int64_t>(run_simple_agg(
                 "SELECT COUNT(*) FROM update_bench_1;", ExecutorDeviceType::CPU)));
  }

  void TearDown(const ::benchmark::State& state) override {
    run_ddl_statement("DROP TABLE IF EXISTS update_bench_1;");
  }
};

//! Run a single column, scalar update query where roughly one value per fragment passes
//! the filter, scaling up the number of fragments
BENCHMARK_DEFINE_F(UpdateFixture, ScalarUpdateTest)(benchmark::State& state) {
  for (auto _ : state) {
    run_multiple_agg("UPDATE update_bench_1 SET x = 100 WHERE x % 100 = 0;",
                     ExecutorDeviceType::CPU);
  }
}

BENCHMARK_REGISTER_F(UpdateFixture, ScalarUpdateTest)
    ->Range(100, 1000000)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

//! Run a multi column, scalar update query where roughly one value per fragment passes
//! the filter, scaling up the number of fragments
BENCHMARK_DEFINE_F(UpdateFixture, ScalarMultiColumnUpdateTest)
(benchmark::State& state) {
  for (auto _ : state) {
    run_multiple_agg("UPDATE update_bench_1 SET x = 100, y = 2.1 WHERE x % 100 = 0;",
                     ExecutorDeviceType::CPU);
  }
}

BENCHMARK_REGISTER_F(UpdateFixture, ScalarMultiColumnUpdateTest)
    ->Range(100, 1000000)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

//! Run a single column, dictionary encoded string update query where roughly one value
//! per fragment passes the filter, scaling up the number of fragments
BENCHMARK_DEFINE_F(UpdateFixture, DictEncodedStrUpdateTest)(benchmark::State& state) {
  for (auto _ : state) {
    run_multiple_agg("UPDATE update_bench_1 SET str = 'hi' WHERE x % 100 = 0;",
                     ExecutorDeviceType::CPU);
  }
}

BENCHMARK_REGISTER_F(UpdateFixture, DictEncodedStrUpdateTest)
    ->Range(100, 1000000)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

class TempTableUpdateFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    std::call_once(setup_flag, global_setup);

    run_ddl_statement("DROP TABLE IF EXISTS update_bench_1;");
    run_ddl_statement(
        "CREATE TEMPORARY TABLE update_bench_1 (x INT, y DOUBLE, z BOOLEAN, str TEXT "
        "ENCODING "
        "DICT(16)) WITH (FRAGMENT_SIZE=100);");

    std::array<std::string, 15> str_values{"foo",
                                           "bar",
                                           "baz",
                                           "hello",
                                           "world",
                                           "the",
                                           "quick",
                                           "brown",
                                           "fox",
                                           "jumped",
                                           "over",
                                           "lazy",
                                           "dog",
                                           "today",
                                           "tomorrow"};

    // TODO(adb): cleanup this interface a bit, as we're half in parser land, half in
    // direct API lands
    auto cat = QR::get()->getCatalog();
    const auto td = cat->getMetadataForTable("update_bench_1");
    CHECK(td);
    auto loader = QR::get()->getLoader(td);
    CHECK(loader);

    auto col_descs = loader->get_column_descs();
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
    for (auto cd : col_descs) {
      import_buffers.push_back(std::unique_ptr<import_export::TypedImportBuffer>(
          new import_export::TypedImportBuffer(cd, loader->getStringDict(cd))));
    }

    for (int64_t i = 0; i < state.range(0); i++) {
      std::vector<std::string> values{
          std::to_string(i % 10000),
          std::to_string(1.1 * (i % 10000)),
          i % 2 == 0 ? std::string{"true"} : std::string{"false"},
          str_values[i % str_values.size()]};
      size_t index = 0;
      for (auto cd : col_descs) {
        CHECK_LT(index, values.size());
        CHECK_LT(index, import_buffers.size());
        import_buffers[index]->add_value(cd,
                                         values[index],
                                         /*is_null=*/false,
                                         import_export::CopyParams());
        index++;
      }
    }

    loader->load(import_buffers, state.range(0), nullptr);

    // make sure we're warmed up
    run_multiple_agg("SELECT * FROM update_bench_1;", ExecutorDeviceType::CPU);
    CHECK_EQ(static_cast<int64_t>(state.range(0)),
             TestHelpers::v<int64_t>(run_simple_agg(
                 "SELECT COUNT(*) FROM update_bench_1;", ExecutorDeviceType::CPU)));
  }

  void TearDown(const ::benchmark::State& state) override {
    run_ddl_statement("DROP TABLE IF EXISTS update_bench_1;");
  }
};

//! Run a single column, scalar update query where roughly one value per fragment passes
//! the filter, scaling up the number of fragments
BENCHMARK_DEFINE_F(TempTableUpdateFixture, ScalarUpdateTest)(benchmark::State& state) {
  for (auto _ : state) {
    run_multiple_agg("UPDATE update_bench_1 SET x = 100 WHERE x % 100 = 0;",
                     ExecutorDeviceType::CPU);
  }
}

BENCHMARK_REGISTER_F(TempTableUpdateFixture, ScalarUpdateTest)
    ->Range(100, 500000)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
