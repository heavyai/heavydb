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

std::once_flag setup_flag;
void global_setup() {
  TestHelpers::init_logger_stderr_only();
  QR::init(BASE_PATH);

  boost::filesystem::path lz4_data_path{
      "../../Tests/OneDALBenchmarkDataFiles/florida_parcels_2020.dump.lz4"};

  if (!boost::filesystem::exists(lz4_data_path)) {
    throw std::runtime_error("florida_parcels data not found at " +
                             boost::filesystem::canonical(lz4_data_path).string());
  }

  run_ddl_statement("DROP TABLE IF EXISTS florida_parcels_2020;");
  run_ddl_statement("RESTORE TABLE florida_parcels_2020 FROM '" +
                    boost::filesystem::canonical(lz4_data_path).string() +
                    "' WITH (COMPRESSION='lz4');");

  // make sure we're warmed up
  run_multiple_agg("SELECT * FROM florida_parcels_2020 LIMIT 10000;",
                   ExecutorDeviceType::CPU);
}

class DalFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    std::call_once(setup_flag, global_setup);
  }
};

//! Run KMeans clustering for OneDAL
BENCHMARK_DEFINE_F(DalFixture, OneDALKMeansClustering)(benchmark::State& state) {
  for (auto _ : state) {
    run_multiple_agg(
        "SELECT * FROM TABLE(KMEANS(data => CURSOR(SELECT PARCELID as id, LAT_DD, "
        "LONG_DD "
        "FROM florida_parcels_2020), num_clusters => " +
            std::to_string(state.range(0)) + ", num_iterations => " +
            std::to_string(state.range(1)) +
            ", preferred_ml_framework => 'ONEDAL')) ORDER BY id;",
        ExecutorDeviceType::CPU);
  }
}

//! Run KMeans clustering for OneAPI
BENCHMARK_DEFINE_F(DalFixture, OneAPIKMeansClustering)(benchmark::State& state) {
  for (auto _ : state) {
    run_multiple_agg(
        "SELECT * FROM TABLE(KMEANS(data => CURSOR(SELECT PARCELID as id, LAT_DD, "
        "LONG_DD "
        "FROM florida_parcels_2020), num_clusters => " +
            std::to_string(state.range(0)) + ", num_iterations => " +
            std::to_string(state.range(1)) +
            ", preferred_ml_framework => 'ONEAPI')) ORDER BY id;",
        ExecutorDeviceType::CPU);
  }
}

//! Run DBScan clustering for OneDAL
BENCHMARK_DEFINE_F(DalFixture, OneDALDBScanClustering)(benchmark::State& state) {
  for (auto _ : state) {
    run_multiple_agg(
        "SELECT * FROM TABLE(DBSCAN(data => CURSOR(SELECT PARCELID as id, LAT_DD, "
        "LONG_DD "
        "FROM florida_parcels_2020 LIMIT 1000000), min_observations => " +
            std::to_string(state.range(0)) +
            ", epsilon => 0.5"
            ", preferred_ml_framework => 'ONEDAL')) ORDER BY id;",
        ExecutorDeviceType::CPU);
  }
}

//! Run DBScan clustering for OneAPI
BENCHMARK_DEFINE_F(DalFixture, OneAPIDBScanClustering)(benchmark::State& state) {
  for (auto _ : state) {
    run_multiple_agg(
        "SELECT * FROM TABLE(DBSCAN(data => CURSOR(SELECT PARCELID as id, LAT_DD, "
        "LONG_DD FROM florida_parcels_2020 LIMIT 1000000), min_observations => " +
            std::to_string(state.range(0)) +
            ", epsilon => 0.5, preferred_ml_framework => 'ONEAPI')) ORDER BY id;",
        ExecutorDeviceType::CPU);
  }
}

//! Run PCA for OneDAL
BENCHMARK_DEFINE_F(DalFixture, OneDALPrincipalComponentAnalysis)
(benchmark::State& state) {
  for (auto _ : state) {
    std::string fit_query =
        "SELECT * FROM TABLE(PCA_FIT(data => CURSOR(SELECT LAT_DD, LONG_DD, "
        "Shape_Length, "
        "Shape_Area FROM florida_parcels_2020), model_name => 'PCA_MODEL_ONEDAL', "
        "preferred_ml_framework => 'ONEDAL'));";
    std::string project_subquery =
        "PCA_PROJECT('PCA_MODEL_ONEDAL', LAT_DD, LONG_DD, Shape_Length, Shape_Area, ";
    std::string project_query =
        "SELECT LAT_DD, LONG_DD, Shape_Length, Shape_Area, " + project_subquery +
        "1) AS pca_1, " + project_subquery + "2) AS pca_2, " + project_subquery +
        "3) AS pca_3, " + project_subquery + "4) AS pca_4 FROM florida_parcels_2020;";
    run_multiple_agg(fit_query, ExecutorDeviceType::CPU);
    run_multiple_agg(project_query, ExecutorDeviceType::CPU);
  }
}

//! Run PCA for OneAPI
BENCHMARK_DEFINE_F(DalFixture, OneAPIPrincipalComponentAnalysis)
(benchmark::State& state) {
  for (auto _ : state) {
    std::string fit_query =
        "SELECT * FROM TABLE(PCA_FIT(data => CURSOR(SELECT LAT_DD, LONG_DD, "
        "Shape_Length, "
        "Shape_Area FROM florida_parcels_2020), model_name => 'PCA_MODEL_ONEAPI', "
        "preferred_ml_framework => 'ONEAPI'));";
    std::string project_subquery =
        "PCA_PROJECT('PCA_MODEL_ONEAPI', LAT_DD, LONG_DD, Shape_Length, Shape_Area, ";
    std::string project_query =
        "SELECT LAT_DD, LONG_DD, Shape_Length, Shape_Area, " + project_subquery +
        "1) AS pca_1, " + project_subquery + "2) AS pca_2, " + project_subquery +
        "3) AS pca_3, " + project_subquery + "4) AS pca_4 FROM florida_parcels_2020;";
    run_multiple_agg(fit_query, ExecutorDeviceType::CPU);
    run_multiple_agg(project_query, ExecutorDeviceType::CPU);
  }
}

//! Run Linear Regression for OneDAL
BENCHMARK_DEFINE_F(DalFixture, OneDALLinearReg)(benchmark::State& state) {
  for (auto _ : state) {
    std::string fit_query =
        "SELECT * FROM TABLE(LINEAR_REG_FIT(model_name => 'LINEAR_REG_ONEDAL', data => "
        "CURSOR(SELECT CAST(SALEPRC1 AS DOUBLE), OSTATE, LAT_DD, LONG_DD, Shape_Length, "
        "Shape_Area FROM florida_parcels_2020), preferred_ml_framework => 'ONEDAL', "
        "cat_top_k => 10, cat_min_fraction => 0.0001));";
    std::string predict_query =
        "SELECT ML_PREDICT('LINEAR_REG_ONEDAL', OSTATE, LAT_DD, LONG_DD, Shape_Length, "
        "Shape_Area) FROM florida_parcels_2020;";
    run_multiple_agg(fit_query, ExecutorDeviceType::CPU);
    run_multiple_agg(predict_query, ExecutorDeviceType::CPU);
  }
}

//! Run Linear Regression for OneAPI
BENCHMARK_DEFINE_F(DalFixture, OneAPILinearReg)(benchmark::State& state) {
  for (auto _ : state) {
    std::string fit_query =
        "SELECT * FROM TABLE(LINEAR_REG_FIT(model_name => 'LINEAR_REG_ONEAPI', data => "
        "CURSOR(SELECT CAST(SALEPRC1 AS DOUBLE), OSTATE, LAT_DD, LONG_DD, Shape_Length, "
        "Shape_Area FROM florida_parcels_2020), preferred_ml_framework => 'ONEAPI', "
        "cat_top_k => 10, cat_min_fraction => 0.0001));";
    std::string predict_query =
        "SELECT ML_PREDICT('LINEAR_REG_ONEAPI', OSTATE, LAT_DD, LONG_DD, Shape_Length, "
        "Shape_Area) FROM florida_parcels_2020;";
    run_multiple_agg(fit_query, ExecutorDeviceType::CPU);
    run_multiple_agg(predict_query, ExecutorDeviceType::CPU);
  }
}

//! Run Random Forest Regression for OneDAL
BENCHMARK_DEFINE_F(DalFixture, OneDALRandomForest)(benchmark::State& state) {
  for (auto _ : state) {
    std::string fit_query =
        "SELECT * FROM TABLE(RANDOM_FOREST_REG_FIT(model_name => 'RANDOM_FOREST_ONEDAL', "
        "data => CURSOR(SELECT CAST(SALEPRC1 AS DOUBLE), OSTATE, LAT_DD, LONG_DD, "
        "Shape_Length, Shape_Area FROM florida_parcels_2020), preferred_ml_framework => "
        "'ONEDAL', cat_top_k => 10, cat_min_fraction => 0.0001));";
    std::string predict_query =
        "SELECT ML_PREDICT('RANDOM_FOREST_ONEDAL', OSTATE, LAT_DD, LONG_DD, "
        "Shape_Length, Shape_Area) FROM florida_parcels_2020;";
    run_multiple_agg(fit_query, ExecutorDeviceType::CPU);
    run_multiple_agg(predict_query, ExecutorDeviceType::CPU);
  }
}

//! Run Random Forest Regression for OneAPI
BENCHMARK_DEFINE_F(DalFixture, OneAPIRandomForest)(benchmark::State& state) {
  for (auto _ : state) {
    std::string fit_query =
        "SELECT * FROM TABLE(RANDOM_FOREST_REG_FIT(model_name => 'RANDOM_FOREST_ONEAPI', "
        "data => CURSOR(SELECT CAST(SALEPRC1 AS DOUBLE), OSTATE, LAT_DD, LONG_DD, "
        "Shape_Length, Shape_Area FROM florida_parcels_2020), preferred_ml_framework => "
        "'ONEAPI', cat_top_k => 10, cat_min_fraction => 0.0001));";
    std::string predict_query =
        "SELECT ML_PREDICT('RANDOM_FOREST_ONEAPI', OSTATE, LAT_DD, LONG_DD, "
        "Shape_Length, Shape_Area) FROM florida_parcels_2020;";
    run_multiple_agg(fit_query, ExecutorDeviceType::CPU);
    run_multiple_agg(predict_query, ExecutorDeviceType::CPU);
  }
}

BENCHMARK_REGISTER_F(DalFixture, OneDALKMeansClustering)
    ->Args({3, 10})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DalFixture, OneDALDBScanClustering)
    ->Args({10})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DalFixture, OneDALPrincipalComponentAnalysis)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DalFixture, OneDALLinearReg)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DalFixture, OneDALRandomForest)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DalFixture, OneAPIKMeansClustering)
    ->Args({3, 10})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DalFixture, OneAPIDBScanClustering)
    ->Args({10})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DalFixture, OneAPIPrincipalComponentAnalysis)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DalFixture, OneAPILinearReg)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(DalFixture, OneAPIRandomForest)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();