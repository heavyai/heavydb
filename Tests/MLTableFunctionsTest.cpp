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

#include "../QueryEngine/Execute.h"
#include "../QueryRunner/QueryRunner.h"
#include "QueryEngine/TableFunctions/TableFunctionManager.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

extern bool g_enable_table_functions;
using namespace TestHelpers;

namespace {

inline void run_ddl_statement(const std::string& stmt) {
  QR::get()->runDDLStatement(stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, false, false);
}

TargetValue run_simple_agg(const std::string& query_str,
                           const ExecutorDeviceType device_type,
                           const bool geo_return_geo_tv = true,
                           const bool allow_loop_joins = true) {
  auto rows = QR::get()->runSQL(query_str, device_type, allow_loop_joins);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
}

}  // namespace

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

const char* iris_schema = R"(
    CREATE TABLE ml_iris(
      id INT,
      sepal_length_cm FLOAT,
      sepal_width_cm FLOAT,
      petal_length_cm FLOAT,
      petal_width_cm FLOAT,
      species TEXT);
)";

class SystemTFs : public ::testing::Test {
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS ml_iris;"));

    ASSERT_NO_THROW(run_ddl_statement(iris_schema));
    const std::string import_file{"iris.csv"};
    const auto load_str = std::string("COPY ml_iris FROM '") +
                          "../../Tests/Import/datafiles/ml/" + import_file +
                          "' WITH (header='true');";
    ASSERT_NO_THROW(run_ddl_statement(load_str));
  }
  void TearDown() override { ASSERT_NO_THROW(run_ddl_statement("DROP TABLE ml_iris;")); }
};

std::string generate_cursor_query(const std::string& table_name,
                                  const std::string& id_column,
                                  const std::vector<std::string>& feature_columns,
                                  const std::string& numeric_data_type) {
  std::ostringstream oss;
  oss << "CURSOR(SELECT " << id_column << " AS id";
  for (const auto& feature_column : feature_columns) {
    oss << ", CAST(" << feature_column << " AS " << numeric_data_type << ") AS "
        << feature_column;
  }
  oss << " FROM " << table_name << ")";
  return oss.str();
}

std::string generate_kmeans_query(const std::string& cursor_query,
                                  const int32_t num_clusters,
                                  const int32_t num_iterations,
                                  const std::string& init_type = "",
                                  const bool add_select_star = true) {
  std::ostringstream oss;
  if (add_select_star) {
    oss << "SELECT * FROM ";
  }
  oss << "TABLE(KMEANS("
      << "data => " << cursor_query << ", num_clusters => " << num_clusters
      << ", num_iterations => " << num_iterations;
  if (!init_type.empty()) {
    oss << ", init_type => " << init_type;
  }
  if (add_select_star) {
    oss << ")) ORDER BY id ASC";
  }
  return oss.str();
}

std::string generate_query(const std::string& algo_name,
                           const std::vector<std::pair<std::string, std::string>>& args,
                           const std::vector<std::string>& order_by_cols,
                           const bool make_args_named) {
  std::ostringstream oss;
  const bool project_data = !order_by_cols.empty();
  if (project_data) {
    oss << "SELECT * FROM ";
  }
  oss << "TABLE(" << algo_name << "(";
  bool is_first_arg = true;
  for (const auto& arg : args) {
    if (!is_first_arg) {
      oss << ", ";
    }
    is_first_arg = false;
    if (make_args_named) {
      oss << arg.first << " => ";
    }
    oss << arg.second;
  }
  oss << "))";
  if (project_data) {
    oss << " ORDER BY ";
    is_first_arg = true;
    for (const auto& order_by_col : order_by_cols) {
      if (!is_first_arg) {
        oss << ", ";
      }
      is_first_arg = false;
      oss << order_by_col;
    }
    oss << " ASC NULLS LAST;";
  }
  return oss.str();
}

std::string generate_unsupervised_classifier_precision_query(
    const std::string& algo_name,
    const std::vector<std::pair<std::string, std::string>>& args,
    const std::string& data_table_name,
    const std::string& data_table_id_col,
    const std::string& data_table_class_col,
    const bool make_args_named = true) {
  const auto classifier_query = generate_query(algo_name, args, {}, make_args_named);
  std::ostringstream oss;
  oss << "SELECT CAST(SUM(n) AS DOUBLE) / SUM(total_in_class) "
      << "FROM(SELECT class, cluster_id, n, total_in_class, perc_in_cluster "
      << "FROM(SELECT class, cluster_id, n, total_in_class, perc_in_cluster, "
      << "ROW_NUMBER() OVER (PARTITION BY class ORDER BY perc_in_cluster DESC) AS "
         "cluster_rank "
      << "FROM(SELECT class, cluster_id, COUNT(*) as n, SUM(COUNT(*)) OVER (PARTITION BY "
         "class) "
      << "AS total_in_class, CAST(COUNT(*) AS DOUBLE) / SUM(COUNT(*)) OVER (PARTITION BY "
         "class) "
      << "AS perc_in_cluster "
      << "FROM(SELECT " << data_table_class_col << " AS class, cluster_id "
      << "FROM " << data_table_name << ", " << classifier_query << " classifier_query "
      << "WHERE " << data_table_name << "." << data_table_id_col
      << " = classifier_query.id) "
      << "GROUP BY class, cluster_id)) "
      << "WHERE cluster_rank = 1);";
  return oss.str();
}

double calc_classification_accuracy(const std::string& data_table_name,
                                    const std::string& classification_query) {
  std::ostringstream oss;
  return 0;
}

TEST_F(SystemTFs, KMeans_Missing_Args) {
  const auto cursor_query = generate_cursor_query(
      "ml_iris",
      "id",
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
      "float");

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Missing args
    {
      for (bool make_args_named : {false, true}) {
        EXPECT_ANY_THROW(run_multiple_agg(
            generate_query("KMEANS",
                           {{"data", cursor_query}, {"num_clusters", "3"}},
                           {"id"},
                           make_args_named),
            dt));

        EXPECT_ANY_THROW(run_multiple_agg(
            generate_query("KMEANS",
                           {{"data", cursor_query}, {"num_iterations", "10"}},
                           {"id"},
                           make_args_named),
            dt));

        EXPECT_ANY_THROW(
            run_multiple_agg(generate_query("KMEANS",
                                            {
                                                {"data", cursor_query},
                                                {"num_clusters", "3"},
                                                {"init_type", "'DETERMINISTIC'"},
                                            },
                                            {"id"},
                                            make_args_named),
                             dt));

        EXPECT_ANY_THROW(run_multiple_agg(
            generate_query("KMEANS",
                           {{"num_clusters", "3"}, {"num_iterations", "10"}},
                           {"id"},
                           make_args_named),
            dt));
      }
    }
  }
}

TEST_F(SystemTFs, KMeansInvalidType) {
  for (auto numeric_data_type : {"int", "bigint"}) {
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      const auto cursor_query = generate_cursor_query(
          "ml_iris",
          "id",
          {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
          numeric_data_type);
      EXPECT_ANY_THROW(run_multiple_agg(
          generate_query(
              "KMEANS",
              {{"data", cursor_query}, {"num_clusters", "3"}, {"num_iterations", "10"}},
              {"id"},
              true /* make_args_named */),
          dt));
    }
  }
}

TEST_F(SystemTFs, KMeansInvalidArgs) {
  const auto numeric_data_type{"float"};
  const auto cursor_query = generate_cursor_query(
      "ml_iris",
      "id",
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
      numeric_data_type);
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Should throw if num clusters <= 0
    {
      EXPECT_THROW(run_multiple_agg(generate_query("KMEANS",
                                                   {{"data", cursor_query},
                                                    {"num_clusters", "0"},
                                                    {"num_iterations", "10"}},
                                                   {"id"},
                                                   true /* make_args_named */),
                                    dt),
                   UserTableFunctionError);
      EXPECT_THROW(run_multiple_agg(generate_query("KMEANS",
                                                   {{"data", cursor_query},
                                                    {"num_clusters", "-1"},
                                                    {"num_iterations", "10"}},
                                                   {"id"},
                                                   true /* make_args_named */),
                                    dt),
                   UserTableFunctionError);
      // There are only 150 observations/rows in the iris dataset
      // We should throw if we ask for more clusters than that (here 300)
      EXPECT_THROW(run_multiple_agg(generate_query("KMEANS",
                                                   {{"data", cursor_query},
                                                    {"num_clusters", "300"},
                                                    {"num_iterations", "10"}},
                                                   {"id"},
                                                   true /* make_args_named */),
                                    dt),
                   UserTableFunctionError);
    }
    // Should throw if num_iterations <= 0
    {
      EXPECT_THROW(run_multiple_agg(generate_query("KMEANS",
                                                   {{"data", cursor_query},
                                                    {"num_clusters", "3"},
                                                    {"num_iterations", "0"}},
                                                   {"id"},
                                                   true /* make_args_named */),
                                    dt),
                   UserTableFunctionError);
      EXPECT_THROW(run_multiple_agg(generate_query("KMEANS",
                                                   {{"data", cursor_query},
                                                    {"num_clusters", "3"},
                                                    {"num_iterations", "-1"}},
                                                   {"id"},
                                                   true /* make_args_named */),
                                    dt),
                   UserTableFunctionError);
    }
    // Should throw if kmeans init type is invalid
    {
      EXPECT_THROW(run_multiple_agg(generate_query("KMEANS",
                                                   {{"data", cursor_query},
                                                    {"num_clusters", "3"},
                                                    {"num_iterations", "10"},
                                                    {"init_type", "'foo_bar'"}},
                                                   {"id"},
                                                   true /* make_args_named */),
                                    dt),
                   UserTableFunctionError);
    }
  }
}

TEST_F(SystemTFs, KMeansNumClusters) {
  for (auto numeric_data_type : {"float", "double"}) {
    const auto cursor_query = generate_cursor_query(
        "ml_iris",
        "id",
        {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
        numeric_data_type);
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      const auto num_obs = TestHelpers::v<int64_t>(
          run_simple_agg("SELECT COUNT(DISTINCT id) FROM ml_iris;", dt));
      for (int32_t num_clusters = 1; num_clusters < 10; ++num_clusters) {
        std::ostringstream oss;
        oss << "SELECT COUNT(DISTINCT id), COUNT(DISTINCT cluster_id) FROM TABLE(KMEANS("
            << "data => " << cursor_query << ", num_clusters => " << num_clusters
            << ", num_iterations => 10));";
        const std::string query = oss.str();
        const auto rows = run_multiple_agg(query, dt);
        EXPECT_EQ(rows->rowCount(), size_t(1));
        EXPECT_EQ(rows->colCount(), size_t(2));
        auto crt_row = rows->getNextRow(true, true);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), num_obs);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[1]), num_clusters);
      }
    }
  }
}

TEST_F(SystemTFs, KMeansPrecision) {
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{"id"};
  const std::vector<std::string> feature_cols{
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"}};
  const std::string class_col{"species"};
  for (auto numeric_data_type : {"float", "double"}) {
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      const size_t num_classes = TestHelpers::v<int64_t>(run_simple_agg(
          "SELECT COUNT(DISTINCT " + class_col + ") FROM " + data_table_name + ";", dt));
      const auto cursor_query =
          generate_cursor_query(data_table_name, id_col, feature_cols, numeric_data_type);
      const auto precision_query = generate_unsupervised_classifier_precision_query(
          "KMEANS",
          {{"data", cursor_query},
           {"num_clusters", std::to_string(num_classes)},
           {"num_iterations", "10"},
           {"init_type", "'DEFAULT'"}},
          data_table_name,
          id_col,
          class_col,
          true);
      EXPECT_GE(TestHelpers::v<double>(run_simple_agg(precision_query, dt)), 0.8);
    }
  }
}

// Hitting issues with the dbscan preflight require checks
// causing a crash at launch time, so disabling until can troubleshoot

TEST_F(SystemTFs, DISABLED_DBSCANInvalidArgs) {
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{"id"};
  const std::vector<std::string> feature_cols{
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"}};
  const std::string class_col{"species"};

  const auto numeric_data_type{"float"};
  const auto cursor_query =
      generate_cursor_query(data_table_name, id_col, feature_cols, numeric_data_type);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Should throw if epsilon <= 0
    EXPECT_THROW(run_multiple_agg(generate_query("DBSCAN",
                                                 {{"data", cursor_query},
                                                  {"epsilon", "-1.0"},
                                                  {"min_observations", "10"}},
                                                 {"id"},
                                                 true /* make_args_named */),
                                  dt),
                 UserTableFunctionError);

    EXPECT_THROW(run_multiple_agg(generate_query("DBSCAN",
                                                 {{"data", cursor_query},
                                                  {"epsilon", "0.0"},
                                                  {"min_observations", "10"}},
                                                 {"id"},
                                                 true /* make_args_named */),
                                  dt),
                 UserTableFunctionError);

    // Should throw if min_observations <= 0
    EXPECT_THROW(run_multiple_agg(generate_query("DBSCAN",
                                                 {{"data", cursor_query},
                                                  {"epsilon", "0.5"},
                                                  {"min_observations", "-10"}},
                                                 {"id"},
                                                 true /* make_args_named */),
                                  dt),
                 UserTableFunctionError);

    EXPECT_THROW(run_multiple_agg(generate_query("DBSCAN",
                                                 {{"data", cursor_query},
                                                  {"epsilon", "0.5"},
                                                  {"min_observations", "0"}},
                                                 {"id"},
                                                 true /* make_args_named */),
                                  dt),
                 UserTableFunctionError);
  }
}

TEST_F(SystemTFs, DBSCAN) {
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{"id"};
  const std::vector<std::string> feature_cols{
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"}};
  const std::string class_col{"species"};

  for (auto numeric_data_type : {"float", "double"}) {
    const auto cursor_query = generate_cursor_query(
        "ml_iris",
        "id",
        {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
        numeric_data_type);
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      const auto num_obs = TestHelpers::v<int64_t>(
          run_simple_agg("SELECT COUNT(DISTINCT id) FROM " + data_table_name + ";", dt));
      const auto dbscan_partial_query = generate_query(
          "DBSCAN",
          {{"data", cursor_query}, {"epsilon", "0.5"}, {"min_observations", "10"}},
          {},
          true /* make_args_named */);
      const auto query =
          "SELECT COUNT(DISTINCT id), COUNT(DISTINCT cluster_id) "
          " FROM " +
          dbscan_partial_query + ";";
      const auto rows = run_multiple_agg(query, dt);
      EXPECT_EQ(rows->rowCount(), size_t(1));
      EXPECT_EQ(rows->colCount(), size_t(2));
      auto crt_row = rows->getNextRow(true, true);
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), num_obs);
      EXPECT_GT(TestHelpers::v<int64_t>(crt_row[1]), 1);
    }
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  // Table function support must be enabled before initialized the query runner
  // environment
  g_enable_table_functions = true;
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