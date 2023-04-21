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
#include "QueryEngine/TableFunctions/SystemFunctions/os/ML/MLModel.h"
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
extern bool g_enable_ml_functions;
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

const char* craigslist_f150s_schema = R"(
  CREATE TABLE craigslist_f150s (
    region TEXT ENCODING DICT(16),
    state TEXT ENCODING DICT(8),
    price INTEGER,
    year_ SMALLINT,
    condition_ TEXT ENCODING DICT(8),
    cylinders TEXT ENCODING DICT(8),
    fuel TEXT ENCODING DICT(8),
    odometer FLOAT,
    title_status TEXT ENCODING DICT(8),
    transmission TEXT ENCODING DICT(8),
    drive TEXT ENCODING DICT(8),
    paint_color TEXT ENCODING DICT(8),
    posting_date TIMESTAMP(0) ENCODING FIXED(32));
)";

class MLTableFunctionsTest : public ::testing::Test {
 public:
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

class MLCreateModelTest : public ::testing::Test {
 public:
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

class MLRegressionFunctionsTest : public testing::TestWithParam<MLModelType> {
 public:
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

class MLCategoricalRegressionFunctionsTest : public testing::TestWithParam<MLModelType> {
 public:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS craigslist_f150s;"));

    ASSERT_NO_THROW(run_ddl_statement(craigslist_f150s_schema));
    const std::string import_file{"craigslist_f150s.csv.gz"};
    const auto load_str = std::string("COPY craigslist_f150s FROM '") +
                          "../../Tests/Import/datafiles/ml/" + import_file +
                          "' WITH (header='true');";
    ASSERT_NO_THROW(run_ddl_statement(load_str));
  }
  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("DROP TABLE craigslist_f150s;"));
  }
};

std::string generate_cursor_query(const std::string& table_name,
                                  const std::string& id_column,
                                  const std::vector<std::string>& feature_columns,
                                  const std::string& filter,
                                  const std::string& numeric_data_type,
                                  const bool add_cursor = true,
                                  const double sample_frac = 0.0,
                                  const bool add_casts = true,
                                  const bool use_alias = true) {
  std::ostringstream oss;
  if (add_cursor) {
    oss << "CURSOR(";
  }
  oss << "SELECT ";
  if (!id_column.empty()) {
    oss << id_column << " AS id, ";
  }
  bool is_first_col = true;
  size_t feature_idx = 0;
  for (const auto& feature_column : feature_columns) {
    if (!is_first_col) {
      oss << ", ";
    } else {
      is_first_col = false;
    }
    if (add_casts) {
      oss << "CAST(";
    }
    oss << feature_column;
    if (add_casts) {
      oss << " AS " << numeric_data_type << ")";
    }
    if (use_alias) {
      const std::string alias_name = "var" + std::to_string(feature_idx++);
      oss << " AS " << alias_name;
    }
  }
  oss << " FROM " << table_name;
  if (!filter.empty() || sample_frac > 0.0) {
    oss << " WHERE " << filter;
    if (sample_frac > 0.0) {
      if (!filter.empty()) {
        oss << " AND ";
      }
      oss << "SAMPLE_RATIO(" << sample_frac << ")";
    }
  }
  if (add_cursor) {
    oss << ")";
  }
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

std::string generate_create_model_query(
    const std::string& algo_name,
    const std::string& model_name,
    const std::string& query,
    const std::vector<std::pair<std::string, std::string>>& args,
    const bool create_or_replace) {
  std::ostringstream oss;
  oss << "CREATE ";
  if (create_or_replace) {
    oss << "OR REPLACE ";
  }
  oss << "MODEL " << model_name << " OF TYPE " << algo_name << " AS " << query;
  if (args.size()) {
    oss << " WITH (";
    bool is_first_arg = true;
    for (const auto& arg : args) {
      if (!is_first_arg) {
        oss << ", ";
      }
      is_first_arg = false;
      oss << arg.first << " = " << arg.second;
    }
    oss << ");";
  } else {
    oss << ";";
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

std::vector<std::string> get_supported_ml_frameworks() {
  const std::string query =
      "SELECT ml_framework FROM "
      "TABLE(supported_ml_frameworks()) WHERE is_available = TRUE ORDER BY ml_framework "
      "DESC;";
  const auto rows = run_multiple_agg(query, ExecutorDeviceType::CPU);
  std::vector<std::string> supported_ml_frameworks;
  for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
    auto crt_row = rows->getNextRow(true, true);
    supported_ml_frameworks.emplace_back(
        std::string("'") +
        boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[0])) +
        std::string("'"));
  }
  return supported_ml_frameworks;
}

TEST_F(MLTableFunctionsTest, SupportedMLFrameworks) {
  const std::vector<std::string> expected_ml_frameworks = {"onedal", "mlpack"};
  std::vector<bool> expected_is_available;
  std::vector<bool> expected_is_default;
  bool found_ml_framework = false;
#ifdef HAVE_ONEDAL
  expected_is_available.emplace_back(true);
  expected_is_default.emplace_back(!found_ml_framework);
  found_ml_framework = true;
#else
  expected_is_available.emplace_back(false);
  expected_is_default.emplace_back(false);
#endif

#ifdef HAVE_MLPACK
  expected_is_available.emplace_back(true);
  expected_is_default.emplace_back(!found_ml_framework);
  found_ml_framework = true;
#else
  expected_is_available.emplace_back(false);
  expected_is_default.emplace_back(false);
#endif

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    const std::string query =
        "SELECT ml_framework, is_available, is_default FROM "
        "TABLE(supported_ml_frameworks()) ORDER BY ml_framework DESC;";
    const auto rows = run_multiple_agg(query, dt);
    const size_t num_rows = rows->rowCount();
    EXPECT_EQ(num_rows, size_t(2));
    EXPECT_EQ(rows->colCount(), size_t(3));
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
      auto crt_row = rows->getNextRow(true, true);
      const auto ml_framework =
          boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[0]));
      EXPECT_EQ(ml_framework, expected_ml_frameworks[row_idx]);
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[1]),
                expected_is_available[row_idx] ? 1 : 0);
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[2]),
                expected_is_default[row_idx] ? 1 : 0);
    }
  }
}

TEST_F(MLTableFunctionsTest, KMeansMissingArgs) {
  const auto cursor_query = generate_cursor_query(
      "ml_iris",
      "id",
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
      "",
      "float");

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Missing args
    {
      for (bool make_args_named : {false, true}) {
        if (make_args_named) {
          EXPECT_NO_THROW(run_multiple_agg(
              generate_query("KMEANS",
                             {{"data", cursor_query}, {"num_clusters", "3"}},
                             {"id"},
                             make_args_named),
              dt));
        } else {
          EXPECT_ANY_THROW(run_multiple_agg(
              generate_query("KMEANS",
                             {{"data", cursor_query}, {"num_clusters", "3"}},
                             {"id"},
                             make_args_named),
              dt));
        }

        EXPECT_ANY_THROW(run_multiple_agg(
            generate_query("KMEANS",
                           {{"data", cursor_query}, {"num_iterations", "10"}},
                           {"id"},
                           make_args_named),
            dt));

        if (make_args_named) {
          EXPECT_NO_THROW(run_multiple_agg(generate_query("KMEANS",
                                                          {
                                                              {"data", cursor_query},
                                                              {"num_clusters", "3"},
                                                              {"init_type", "'DEFAULT'"},
                                                          },
                                                          {"id"},
                                                          make_args_named),
                                           dt));
        } else {
          EXPECT_ANY_THROW(run_multiple_agg(generate_query("KMEANS",
                                                           {
                                                               {"data", cursor_query},
                                                               {"num_clusters", "3"},
                                                               {"init_type", "'DEFAULT'"},
                                                           },
                                                           {"id"},
                                                           make_args_named),
                                            dt));
        }

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

TEST_F(MLTableFunctionsTest, KMeansInvalidType) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (auto numeric_data_type : {"int", "bigint"}) {
      const auto cursor_query = generate_cursor_query(
          "ml_iris",
          "id",
          {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
          "",
          numeric_data_type);
      EXPECT_NO_THROW(run_multiple_agg(
          generate_query(
              "KMEANS",
              {{"data", cursor_query}, {"num_clusters", "3"}, {"num_iterations", "10"}},
              {"id"},
              true /* make_args_named */),
          dt));
    }
    for (auto numeric_data_type : {"text"}) {
      // We do not auto-cast from TEXT to numeric types, so this will fail
      const auto cursor_query = generate_cursor_query(
          "ml_iris",
          "id",
          {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
          "",
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

TEST_F(MLTableFunctionsTest, KMeansInvalidArgs) {
  const auto numeric_data_type{"float"};
  const auto cursor_query = generate_cursor_query(
      "ml_iris",
      "id",
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
      "",
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

TEST_F(MLTableFunctionsTest, KMeansNumClusters) {
  for (auto numeric_data_type : {"float", "double"}) {
    const auto cursor_query = generate_cursor_query(
        "ml_iris",
        "id",
        {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
        "",
        numeric_data_type);
    const auto supported_ml_frameworks = get_supported_ml_frameworks();
    for (auto& ml_framework : supported_ml_frameworks) {
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        const auto num_obs = TestHelpers::v<int64_t>(
            run_simple_agg("SELECT COUNT(DISTINCT id) FROM ml_iris;", dt));
        for (int32_t num_clusters = 1; num_clusters < 10; ++num_clusters) {
          std::ostringstream oss;
          oss << "SELECT COUNT(DISTINCT id), COUNT(DISTINCT cluster_id) FROM "
                 "TABLE(KMEANS("
              << "data => " << cursor_query << ", num_clusters => " << num_clusters
              << ", num_iterations => 10, init_type => 'DEFAULT', "
              << "preferred_ml_framework => " << ml_framework << "));";
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
}

TEST_F(MLTableFunctionsTest, KMeansPrecision) {
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{"id"};
  const std::vector<std::string> feature_cols{
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"}};
  const std::string class_col{"species"};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  for (auto& ml_framework : supported_ml_frameworks) {
    for (auto numeric_data_type : {"DOUBLE"}) {
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        const size_t num_classes = TestHelpers::v<int64_t>(run_simple_agg(
            "SELECT COUNT(DISTINCT " + class_col + ") FROM " + data_table_name + ";",
            dt));
        const auto cursor_query = generate_cursor_query(
            data_table_name, id_col, feature_cols, "", numeric_data_type);
        const auto precision_query = generate_unsupervised_classifier_precision_query(
            "KMEANS",
            {{"data", cursor_query},
             {"num_clusters", std::to_string(num_classes)},
             {"num_iterations", "10"},
             {"init_type", "'DEFAULT'"},
             {"preferred_ml_framework", ml_framework}},
            data_table_name,
            id_col,
            class_col,
            true);
        EXPECT_GE(TestHelpers::v<double>(run_simple_agg(precision_query, dt)), 0.8);
      }
    }
  }
}

// Hitting issues with the dbscan preflight require checks
// causing a crash at launch time, so disabling until can troubleshoot

TEST_F(MLTableFunctionsTest, DBSCANInvalidArgs) {
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{"id"};
  const std::vector<std::string> feature_cols{
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"}};
  const std::string class_col{"species"};

  const auto numeric_data_type{"float"};
  const auto cursor_query =
      generate_cursor_query(data_table_name, id_col, feature_cols, "", numeric_data_type);

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

TEST_F(MLTableFunctionsTest, DBSCAN) {
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{"id"};
  const std::vector<std::string> feature_cols{
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"}};
  const std::string class_col{"species"};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  for (auto& ml_framework : supported_ml_frameworks) {
    for (auto numeric_data_type : {"DOUBLE"}) {
      const auto cursor_query = generate_cursor_query(
          "ml_iris",
          "id",
          {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
          "",
          numeric_data_type);
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        const auto num_obs = TestHelpers::v<int64_t>(run_simple_agg(
            "SELECT COUNT(DISTINCT id) FROM " + data_table_name + ";", dt));
        const auto dbscan_partial_query =
            generate_query("DBSCAN",
                           {{"data", cursor_query},
                            {"epsilon", "0.5"},
                            {"min_observations", "10"},
                            {"preferred_ml_framework", ml_framework}},
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
}

TEST_F(MLTableFunctionsTest, PCA) {
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{"id"};
  const std::string model_name{"PCA_MODEL"};
  const std::string quoted_model_name{"'" + model_name + "'"};
  const std::vector<std::string> feature_cols{
      {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"}};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  for (auto& ml_framework : supported_ml_frameworks) {
    if (ml_framework != "'onedal'") {
      continue;
    }
    for (auto numeric_data_type : {"DOUBLE"}) {
      const auto cursor_query = generate_cursor_query(
          "ml_iris",
          "",
          {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
          "",
          numeric_data_type);
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        const auto pca_partial_query =
            generate_query("PCA_FIT",
                           {{"model_name", quoted_model_name},
                            {"data", cursor_query},
                            {"preferred_ml_framework", ml_framework}},
                           {},
                           true /* make_args_named */);
        const auto query = "SELECT * FROM " + pca_partial_query + ";";
        const auto fit_rows = run_multiple_agg(query, dt);
        EXPECT_EQ(fit_rows->rowCount(), size_t(1));
        EXPECT_EQ(fit_rows->colCount(), size_t(1));
        auto model_row = fit_rows->getNextRow(true, true);
        EXPECT_EQ(boost::get<std::string>(TestHelpers::v<NullableString>(model_row[0])),
                  model_name);
        const auto model = g_ml_models.getModel(model_name);
        const auto pca_model = std::dynamic_pointer_cast<PcaModel>(model);
        const auto model_type = pca_model->getModelType();
        EXPECT_EQ(model_type, MLModelType::PCA);
        const auto num_features = pca_model->getNumFeatures();
        constexpr int64_t expected_num_features = 4;
        EXPECT_EQ(num_features, expected_num_features);
        const auto& eigenvalues = pca_model->getEigenvalues();
        const auto& eigenvectors = pca_model->getEigenvectors();
        const std::vector<double> expected_eigenvalues{
            2.91082, 0.921221, 0.147353, 0.0206075};
        const std::vector<std::vector<double>> expected_eigenvectors{
            {0.581254, 0.565611, 0.522371, -0.263355},
            {0.0210948, 0.065416, 0.372318, 0.925557},
            {-0.140893, -0.633801, 0.721017, -0.242033},
            {0.801154, -0.523547, -0.261995, 0.124135}};
        EXPECT_EQ(eigenvalues.size(), size_t(expected_num_features));
        for (int64_t feature_idx = 0; feature_idx < expected_num_features;
             ++feature_idx) {
          EXPECT_NEAR(eigenvalues[feature_idx], expected_eigenvalues[feature_idx], 1e-2);
          EXPECT_EQ(eigenvectors[feature_idx].size(), size_t(expected_num_features));
          for (int64_t ev_idx = 0; ev_idx < expected_num_features; ++ev_idx) {
            EXPECT_NEAR(eigenvectors[feature_idx][ev_idx],
                        expected_eigenvectors[feature_idx][ev_idx],
                        1e-2);
          }
        }
      }
    }
  }
}

TEST_P(MLRegressionFunctionsTest, REG_MODEL_FIT_NO_ROWS) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_name{model_type_str + "_MODEL"};
  const std::string quoted_model_name{"'" + model_type_str + "_MODEL'"};
  const std::string model_fit_func{model_type_str + "_FIT"};
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{""};  // No id column allowed for Fit calls
  const std::string label_column{"petal_length_cm"};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  const bool make_args_named{true};
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    for (bool use_create_syntax : {false, true}) {
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        for (std::string numeric_data_type : {"DOUBLE"}) {
          const auto cursor_query = generate_cursor_query(
              data_table_name,
              id_col,
              {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
              "SAMPLE_RATIO(0.0)",
              numeric_data_type,
              !use_create_syntax);
          const auto fit_query =
              use_create_syntax
                  ? generate_create_model_query(
                        model_type_str,
                        model_name,
                        cursor_query,
                        {{"preferred_ml_framework", ml_framework}},
                        true)
                  : generate_query(model_fit_func,
                                   {{"model_name", quoted_model_name},
                                    {"data", cursor_query},
                                    {"preferred_ml_framework", ml_framework}},
                                   {"model_name"},
                                   make_args_named);

          if (!use_create_syntax) {
            EXPECT_ANY_THROW(run_multiple_agg(fit_query, dt));
          } else {
            EXPECT_ANY_THROW(run_ddl_statement(fit_query));
          }
        }
      }
    }
  }
}

TEST_P(MLRegressionFunctionsTest, REG_MODEL_FIT) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_name{model_type_str + "_MODEL"};
  const std::string quoted_model_name{"'" + model_type_str + "_MODEL'"};
  const std::string model_fit_func{model_type_str + "_FIT"};
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{""};  // No id column allowed for Fit calls
  const std::string label_column{"petal_length_cm"};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  const bool make_args_named{true};
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    for (bool use_create_syntax : {false, true}) {
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        for (std::string numeric_data_type : {"DOUBLE"}) {
          // Synthetic data, also test nulls
          {
            const std::string query =
                "SELECT CAST(CASE WHEN MOD(generate_series, 10) = 0 THEN NULL "
                "ELSE generate_series * 0.001 * 2.0 + 10.0 END AS " +
                numeric_data_type +
                ") AS var0, "
                "CAST(generate_series * 0.001 AS " +
                numeric_data_type + ") AS var1 FROM TABLE(generate_series(0, 99999))";
            const std::string cursor_query = "CURSOR(" + query + ")";
            const auto fit_query =
                use_create_syntax
                    ? generate_create_model_query(
                          model_type_str,
                          model_name,
                          query,
                          {{"preferred_ml_framework", ml_framework}},
                          true)
                    : generate_query(model_fit_func,
                                     {{"model_name", quoted_model_name},
                                      {"data", cursor_query},
                                      {"preferred_ml_framework", ml_framework}},
                                     {"model_name"},
                                     make_args_named);

            if (!use_create_syntax) {
              const auto fit_rows = run_multiple_agg(fit_query, dt);
              EXPECT_EQ(fit_rows->rowCount(), 1UL);
              EXPECT_EQ(fit_rows->colCount(), 1UL);
              auto model_row = fit_rows->getNextRow(true, true);
              EXPECT_EQ(
                  boost::get<std::string>(TestHelpers::v<NullableString>(model_row[0])),
                  model_name);
            } else {
              run_ddl_statement(fit_query);
            }

            if (model_type == MLModelType::LINEAR_REG) {
              const auto coef_query = generate_query("LINEAR_REG_COEFS",
                                                     {{"model_name", quoted_model_name}},
                                                     {"coef_idx, sub_coef_idx"},
                                                     make_args_named);
              const auto coef_rows = run_multiple_agg(coef_query, dt);
              constexpr double allowed_epsilon{0.1};
              const std::vector<double> expected_coefs = {10.0, 2.0};
              const int64_t num_rows = coef_rows->rowCount();
              EXPECT_EQ(num_rows, 2L);
              EXPECT_EQ(coef_rows->colCount(), size_t(5));
              for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                auto crt_row = coef_rows->getNextRow(true, true);
                EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), row_idx);
                if (use_create_syntax) {
                  const auto feature =
                      boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[1]));
                  const std::string expected_feature =
                      row_idx == 0 ? "intercept" : "var" + std::to_string(row_idx);
                  EXPECT_EQ(feature, expected_feature);
                }
                EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[2]), 1L);
                const double actual_coef = TestHelpers::v<double>(crt_row[4]);
                EXPECT_GE(actual_coef, expected_coefs[row_idx] - allowed_epsilon);
                EXPECT_LE(actual_coef, expected_coefs[row_idx] + allowed_epsilon);
              }
            } else if (model_type == MLModelType::RANDOM_FOREST_REG) {
              const auto var_importance_query =
                  generate_query("RANDOM_FOREST_REG_VAR_IMPORTANCE",
                                 {{"model_name", quoted_model_name}},
                                 {"feature_id, sub_feature_id"},
                                 make_args_named);
              const auto var_importance_rows = run_multiple_agg(var_importance_query, dt);

              constexpr double allowed_epsilon_frac{0.02};
              const std::vector<double> expected_var_importances = {4996.12};
              const int64_t num_rows = var_importance_rows->rowCount();
              EXPECT_EQ(num_rows, 1L);
              EXPECT_EQ(var_importance_rows->colCount(), size_t(5));
              for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                auto crt_row = var_importance_rows->getNextRow(true, true);
                // Feature id
                EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), row_idx + 1);
                if (use_create_syntax) {
                  const auto feature =
                      boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[1]));
                  const std::string expected_feature =
                      "var" + std::to_string(row_idx + 1);
                  EXPECT_EQ(feature, expected_feature);
                }
                // Sub-feature id
                // We have no categorical one-hot encoded features, so each sub-id
                // will be 1
                EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[2]), 1L);
                const double actual_var_importance = TestHelpers::v<double>(crt_row[4]);
                EXPECT_GE(actual_var_importance,
                          expected_var_importances[row_idx] -
                              allowed_epsilon_frac * expected_var_importances[row_idx]);
                EXPECT_LE(actual_var_importance,
                          expected_var_importances[row_idx] +
                              allowed_epsilon_frac * expected_var_importances[row_idx]);
              }
            }
          }
          {
            // 3 independent variables
            const auto cursor_query = generate_cursor_query(data_table_name,
                                                            id_col,
                                                            {"petal_length_cm",
                                                             "petal_width_cm",
                                                             "sepal_length_cm",
                                                             "sepal_width_cm"},
                                                            "",
                                                            numeric_data_type,
                                                            !use_create_syntax);
            const auto fit_query =
                use_create_syntax
                    ? generate_create_model_query(
                          model_type_str,
                          model_name,
                          cursor_query,
                          {{"preferred_ml_framework", ml_framework}},
                          true)
                    : generate_query(model_fit_func,
                                     {{"model_name", quoted_model_name},
                                      {"data", cursor_query},
                                      {"preferred_ml_framework", ml_framework}},
                                     {"model_name"},
                                     make_args_named);

            if (!use_create_syntax) {
              const auto fit_rows = run_multiple_agg(fit_query, dt);
              EXPECT_EQ(fit_rows->rowCount(), 1UL);
              EXPECT_EQ(fit_rows->colCount(), 1UL);
              auto model_row = fit_rows->getNextRow(true, true);
              EXPECT_EQ(
                  boost::get<std::string>(TestHelpers::v<NullableString>(model_row[0])),
                  model_name);
            } else {
              run_ddl_statement(fit_query);
            }

            if (model_type == MLModelType::LINEAR_REG) {
              const auto coef_query = generate_query("LINEAR_REG_COEFS",
                                                     {{"model_name", quoted_model_name}},
                                                     {"coef_idx, sub_coef_idx"},
                                                     make_args_named);
              const auto coef_rows = run_multiple_agg(coef_query, dt);

              constexpr double allowed_epsilon{0.01};
              const std::vector<double> expected_coefs = {
                  -0.252664, 1.44572, 0.730363, -0.651394};
              const int64_t num_rows = coef_rows->rowCount();
              EXPECT_EQ(num_rows, 4L);
              EXPECT_EQ(coef_rows->colCount(), size_t(5));
              for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                auto crt_row = coef_rows->getNextRow(true, true);
                EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), row_idx);
                if (use_create_syntax) {
                  const auto feature =
                      boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[1]));
                  const std::string expected_feature =
                      row_idx == 0 ? "intercept" : "var" + std::to_string(row_idx);
                  EXPECT_EQ(feature, expected_feature);
                }
                EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[2]), 1L);
                const double actual_coef = TestHelpers::v<double>(crt_row[4]);
                EXPECT_GE(actual_coef, expected_coefs[row_idx] - allowed_epsilon);
                EXPECT_LE(actual_coef, expected_coefs[row_idx] + allowed_epsilon);
              }
            } else if (model_type == MLModelType::RANDOM_FOREST_REG) {
              const auto var_importance_query =
                  generate_query("RANDOM_FOREST_REG_VAR_IMPORTANCE",
                                 {{"model_name", quoted_model_name}},
                                 {"feature_id, sub_feature_id"},
                                 make_args_named);
              const auto var_importance_rows = run_multiple_agg(var_importance_query, dt);
              constexpr double allowed_epsilon_frac{1.0};
              const std::vector<double> expected_var_importances = {
                  1.9378, 2.5813, 0.89731};
              const int64_t num_rows = var_importance_rows->rowCount();
              EXPECT_EQ(num_rows, 3L);
              EXPECT_EQ(var_importance_rows->colCount(), size_t(5));
              for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                auto crt_row = var_importance_rows->getNextRow(true, true);
                // Feature id
                EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), row_idx + 1);
                if (use_create_syntax) {
                  const auto feature =
                      boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[1]));
                  const std::string expected_feature =
                      "var" + std::to_string(row_idx + 1);
                  EXPECT_EQ(feature, expected_feature);
                }
                // Sub-feature id
                EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[2]), 1L);
                const double actual_var_importance = TestHelpers::v<double>(crt_row[4]);
                EXPECT_GE(actual_var_importance,
                          expected_var_importances[row_idx] -
                              allowed_epsilon_frac * expected_var_importances[row_idx]);
                EXPECT_LE(actual_var_importance,
                          expected_var_importances[row_idx] +
                              allowed_epsilon_frac * expected_var_importances[row_idx]);
              }
            }
          }
        }
      }
    }
  }
}

TEST_P(MLRegressionFunctionsTest, MLRegPredict) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_name{model_type_str + "_MODEL"};
  const std::string model_fit_func{model_type_str + "_FIT"};

  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      for (bool make_args_named : {false, true}) {
        for (std::string numeric_data_type : {"DOUBLE"}) {
          // Synthetic data
          const std::string data_table_name("TABLE(generate_series(0, 99999))");
          const std::string id_col("generate_series");
          const std::string feature_col("generate_series * 0.001");

          const auto data_query = generate_cursor_query(
              data_table_name, id_col, {feature_col}, "", numeric_data_type);
          const std::string model_query =
              std::string(
                  "CURSOR(SELECT * FROM "
                  "TABLE(" +
                  model_fit_func + "(model_name =>'" + model_name +
                  "', data "
                  "=> CURSOR(SELECT CAST(CASE WHEN MOD(generate_series, 10) = 0 THEN "
                  "NULL ELSE "
                  "generate_series * 0.001 * 2.0 + 10.0 END AS ") +
              numeric_data_type + ") AS y, CAST(generate_series * 0.001 AS " +
              numeric_data_type +
              ") AS x "
              "FROM TABLE(generate_series(0, 99999))))))";

          const auto query = generate_query("ML_REG_PREDICT",
                                            {{"model_name", model_query},
                                             {"data", data_query},
                                             {"preferred_ml_framework", ml_framework}},
                                            {"id"},
                                            make_args_named);
          const auto rows = run_multiple_agg(query, dt);
          const int64_t num_rows = rows->rowCount();
          EXPECT_EQ(num_rows, 100000L);
          EXPECT_EQ(rows->colCount(), size_t(2));
          // Tree models can be much less accurate than linear regression in this case,
          // so give tree models large epsilon.
          const double allowed_epsilon =
              model_type == MLModelType::LINEAR_REG ? 0.02 : 20.0;
          for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            auto crt_row = rows->getNextRow(true, true);
            // id col
            EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), row_idx);
            const double predicted_val = (numeric_data_type == "FLOAT")
                                             ? TestHelpers::v<float>(crt_row[1])
                                             : TestHelpers::v<double>(crt_row[1]);
            const double expected_predicted_val = row_idx * 0.001 * 2.0 + 10.0;
            EXPECT_GE(predicted_val, expected_predicted_val - allowed_epsilon);
            EXPECT_LE(predicted_val, expected_predicted_val + allowed_epsilon);
          }
        }
      }
    }
  }
}

TEST_P(MLRegressionFunctionsTest, R2_SCORE) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_fit_func{model_type_str + "_FIT"};
  const std::string model_name{model_type_str + "_MODEL"};
  const std::string quoted_model_name{"'" + model_type_str + "_MODEL'"};
  const std::string data_table_name{"ml_iris"};
  const std::string id_col{""};  // No id column allowed for Fit calls
  const bool make_args_named{true};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      for (std::string numeric_data_type : {"DOUBLE"}) {
        const auto train_cursor_query = generate_cursor_query(
            data_table_name,
            id_col,
            {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
            "SAMPLE_RATIO(0.7)",
            numeric_data_type);
        const auto fit_query = generate_query(model_fit_func,
                                              {{"model_name", quoted_model_name},
                                               {"data", train_cursor_query},
                                               {"preferred_ml_framework", ml_framework}},
                                              {"model_name"},
                                              make_args_named);
        const auto fit_rows = run_multiple_agg(fit_query, dt);
        EXPECT_EQ(fit_rows->rowCount(), 1UL);
        EXPECT_EQ(fit_rows->colCount(), 1UL);
        auto model_row = fit_rows->getNextRow(true, true);
        EXPECT_EQ(boost::get<std::string>(TestHelpers::v<NullableString>(model_row[0])),
                  model_name);

        const auto test_cursor_query = generate_cursor_query(
            data_table_name,
            id_col,
            {"petal_length_cm", "petal_width_cm", "sepal_length_cm", "sepal_width_cm"},
            "SAMPLE_RATIO(0.3)",
            numeric_data_type);
        const auto r2_query = generate_query(
            "R2_SCORE",
            {{"model_name", quoted_model_name}, {"data", test_cursor_query}},
            {"r2"},
            make_args_named);
        const auto r2_rows = run_multiple_agg(r2_query, dt);
        EXPECT_EQ(r2_rows->rowCount(), 1UL);
        EXPECT_EQ(r2_rows->colCount(), 1UL);
        auto r2_row = r2_rows->getNextRow(true, true);
        const double actual_r2 = TestHelpers::v<double>(r2_row[0]);
        const double expected_min_r2{0.95};
        EXPECT_GE(actual_r2, expected_min_r2);
      }
    }
  }
}

TEST_P(MLRegressionFunctionsTest, ML_PREDICT_MISSING_MODEL) {
  const std::string query = std::string(
      "SELECT ML_PREDICT('missing_model', petal_length_cm, petal_width_cm) as prediction "
      "FROM ml_iris;");
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg(query, dt), std::runtime_error);
  }
}

TEST_P(MLRegressionFunctionsTest, ML_PREDICT_WRONG_NUM_REGRESSORS) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_fit_func{model_type_str + "_FIT"};
  const std::string model_name("IRIS_" + model_type_str + "_MODEL");
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    const std::string train_query("SELECT * FROM TABLE(" + model_fit_func +
                                  "(model_name =>'" + model_name +
                                  "', "
                                  "preferred_ml_framework=>" +
                                  ml_framework +
                                  ", data=>CURSOR(select "
                                  "petal_length_cm, petal_width_cm FROM ml_iris)));");
    const std::string predict_query("SELECT ML_PREDICT('" + model_name +
                                    "', petal_width_cm, "
                                    "sepal_width_cm) as prediction FROM ml_iris;");
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      EXPECT_NO_THROW(run_multiple_agg(train_query, dt));
      EXPECT_THROW(run_multiple_agg(predict_query, dt), std::runtime_error);
    }
  }
}

TEST_P(MLRegressionFunctionsTest, ML_PREDICT) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_fit_func{model_type_str + "_FIT"};
  const std::string model_name{"IRIS_" + model_type_str + "_MODEL"};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    const std::string train_query(
        "SELECT * FROM TABLE(" + model_fit_func + "(model_name =>'" + model_name +
        "', "
        "preferred_ml_framework=>" +
        ml_framework +
        ", data=>CURSOR(select "
        "petal_length_cm, petal_width_cm, sepal_length_cm, sepal_width_cm FROM "
        "ml_iris)));");
    const std::string row_wise_predict_query(
        "SELECT AVG(ML_PREDICT('" + model_name +
        "', petal_width_cm, "
        "sepal_length_cm, sepal_width_cm)) FROM ml_iris;");
    const std::string tf_predict_query(
        "SELECT AVG(prediction) FROM TABLE(ML_REG_PREDICT(model_name =>'" + model_name +
        "', "
        "preferred_ml_framework=>" +
        ml_framework +
        ", "
        "data=>CURSOR(SELECT rowid, petal_width_cm, sepal_length_cm, sepal_width_cm FROM "
        "ml_iris)));");
    const double allowed_epsilon{0.1};
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      EXPECT_NO_THROW(run_multiple_agg(train_query, dt));
      const auto row_wise_prediction_avg =
          TestHelpers::v<double>(run_simple_agg(row_wise_predict_query, dt));
      const auto tf_prediction_avg =
          TestHelpers::v<double>(run_simple_agg(tf_predict_query, dt));
      EXPECT_GE(row_wise_prediction_avg, tf_prediction_avg - allowed_epsilon);
      EXPECT_LE(row_wise_prediction_avg, tf_prediction_avg + allowed_epsilon);
    }
  }
}

TEST_P(MLRegressionFunctionsTest, ML_PREDICT_NULLS) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_fit_func{model_type_str + "_FIT"};
  const std::string model_name("IRIS_" + model_type_str + "_MODEL");
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    const std::string train_query(
        "SELECT * FROM TABLE(" + model_fit_func + "(model_name =>'" + model_name +
        "', "
        "preferred_ml_framework=>" +
        ml_framework +
        ", data=>CURSOR(select "
        "petal_length_cm, petal_width_cm, sepal_length_cm, sepal_width_cm FROM "
        "ml_iris)));");
    const std::string row_wise_predict_query(
        "SELECT AVG(ML_PREDICT('" + model_name +
        "', petal_width_cm, "
        "CASE WHEN MOD(rowid, 2) = 0 THEN sepal_length_cm ELSE NULL END, "
        "sepal_width_cm)) "
        "AS avg_result,"
        "COUNT(ML_PREDICT('" +
        model_name +
        "', petal_width_cm, CASE WHEN MOD(rowid, 2) = "
        "0 "
        "THEN sepal_length_cm ELSE NULL END, sepal_width_cm)) AS num_not_null FROM "
        "ml_iris;");
    const std::string tf_predict_query(
        "SELECT AVG(prediction) as avg_result, COUNT(prediction) AS num_not_null FROM "
        "TABLE( "
        "ML_REG_PREDICT(model_name =>'" +
        model_name +
        "', "
        "preferred_ml_framework=>" +
        ml_framework +
        ", "
        "data=>CURSOR(SELECT rowid, petal_width_cm, CASE WHEN MOD(rowid, 2) = 0 THEN "
        "sepal_length_cm ELSE NULL END, sepal_width_cm FROM "
        "ml_iris)));");
    const double allowed_epsilon{0.01};
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      EXPECT_NO_THROW(run_multiple_agg(train_query, dt));
      const auto row_wise_prediction_result =
          run_multiple_agg(row_wise_predict_query, dt);
      EXPECT_EQ(row_wise_prediction_result->rowCount(), size_t(1));
      EXPECT_EQ(row_wise_prediction_result->colCount(), size_t(2));
      const auto tf_prediction_result = run_multiple_agg(tf_predict_query, dt);
      EXPECT_EQ(tf_prediction_result->rowCount(), size_t(1));
      EXPECT_EQ(tf_prediction_result->colCount(), size_t(2));
      auto row_wise_row = row_wise_prediction_result->getNextRow(true, true);
      auto tf_row = tf_prediction_result->getNextRow(true, true);
      const double row_wise_avg = TestHelpers::v<double>(row_wise_row[0]);
      const double tf_avg = TestHelpers::v<double>(tf_row[0]);
      EXPECT_GE(row_wise_avg, tf_avg - allowed_epsilon);
      EXPECT_LE(row_wise_avg, tf_avg + allowed_epsilon);
      const int64_t row_wise_non_null_count = TestHelpers::v<int64_t>(row_wise_row[1]);
      const int64_t tf_non_null_count = TestHelpers::v<int64_t>(tf_row[1]);
      EXPECT_EQ(row_wise_non_null_count, tf_non_null_count);
      EXPECT_EQ(row_wise_non_null_count, int64_t(75));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(MLRegressionTests,
                         MLRegressionFunctionsTest,
                         ::testing::Values(MLModelType::LINEAR_REG,
                                           MLModelType::DECISION_TREE_REG,
                                           MLModelType::GBT_REG,
                                           MLModelType::RANDOM_FOREST_REG));

TEST_P(MLCategoricalRegressionFunctionsTest, ML_PREDICT_CATEGORICAL_FEATURES_MISORDERED) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_fit_func{model_type_str + "_FIT"};
  const std::string model_name("CRAIGSLIST_" + model_type_str + "_MODEL");
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    const std::string train_query("SELECT * FROM TABLE(" + model_fit_func +
                                  "(model_name =>'" + model_name +
                                  "', "
                                  "preferred_ml_framework=>" +
                                  ml_framework +
                                  ", data=>CURSOR(select "
                                  "price, state, odometer FROM craigslist_f150s)));");
    // Note we put the TEXT state column after the odometer column, which is illegal
    // (categorical predictors must come first)
    const std::string predict_query("SELECT ML_PREDICT('" + model_name +
                                    "', odometer, state) as prediction "
                                    "FROM craigslist_f150s;");
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      EXPECT_NO_THROW(run_multiple_agg(train_query, dt));
      EXPECT_ANY_THROW(run_multiple_agg(predict_query, dt));
    }
  }
}

TEST_P(MLCategoricalRegressionFunctionsTest, REG_MODEL_FIT) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_name{model_type_str + "_MODEL"};
  const std::string quoted_model_name{"'" + model_type_str + "_MODEL'"};
  const std::string model_fit_func{model_type_str + "_FIT"};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  // const double allowed_epsilon{0.1};
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    for (bool use_create_syntax : {false, true}) {
      for (std::string numeric_data_type : {"DOUBLE"}) {
        const auto data_query = generate_cursor_query("craigslist_f150s",
                                                      "",
                                                      {"price",
                                                       "state",
                                                       "title_status",
                                                       "paint_color",
                                                       "condition_",
                                                       "year_",
                                                       "odometer"},
                                                      "",
                                                      numeric_data_type,
                                                      !use_create_syntax,
                                                      0.5,
                                                      false,
                                                      false);
        const auto fit_query =
            use_create_syntax
                ? generate_create_model_query(model_type_str,
                                              model_name,
                                              data_query,
                                              {{"preferred_ml_framework", ml_framework},
                                               {"cat_top_k", "10"},
                                               {"cat_min_fraction", "0.001"}},
                                              true)
                : generate_query(model_fit_func,
                                 {{"model_name", quoted_model_name},
                                  {"data", data_query},
                                  {"preferred_ml_framework", ml_framework},
                                  {"cat_top_k", "10"},
                                  {"cat_min_fraction", "0.001"}},
                                 {"model_name"},
                                 true);

        for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
          SKIP_NO_GPU();
          if (!use_create_syntax) {
            const auto fit_rows = run_multiple_agg(fit_query, dt);
            EXPECT_EQ(fit_rows->rowCount(), 1UL);
            EXPECT_EQ(fit_rows->colCount(), 1UL);
            auto model_row = fit_rows->getNextRow(true, true);
            EXPECT_EQ(
                boost::get<std::string>(TestHelpers::v<NullableString>(model_row[0])),
                model_name);
          } else {
            run_ddl_statement(fit_query);
          }
          if (model_type == MLModelType::LINEAR_REG) {
            const std::vector<int64_t> expected_feature_ids = {
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6};
            const std::vector<int64_t> expected_sub_feature_ids = {
                1, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 1, 2, 3, 4, 5,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,  2, 3, 4, 1, 1};

            const std::vector<double> expected_coefs = {
                -2681553.0554, 1725.9846, 782.4798,   1086.2600,  -2629.4910, -1294.4318,
                2935.9767,     3032.3311, 524.7911,   -482.2400,  4439.0142,  5996.3565,
                273.6281,      656.8885,  8127.7868,  22739.9185, -107.2775,  1322.8270,
                2172.7193,     486.7899,  1696.4142,  1930.9537,  1834.2931,  227.2314,
                243.9149,      6250.7301, -2171.6164, -86.0133,   480.2044,   -1779.6998,
                1340.5951,     -0.0046};

            const std::vector<std::string> expected_features = {
                "intercept",    "state",        "state",        "state",
                "state",        "state",        "state",        "state",
                "state",        "state",        "state",        "title_status",
                "title_status", "title_status", "title_status", "title_status",
                "paint_color",  "paint_color",  "paint_color",  "paint_color",
                "paint_color",  "paint_color",  "paint_color",  "paint_color",
                "paint_color",  "paint_color",  "condition_",   "condition_",
                "condition_",   "condition_",   "year_",        "odometer"};
            const std::vector<std::string> expected_sub_features = {
                "",        "fl",      "ca",        "tx",      "mi",       "or",
                "wa",      "id",      "nc",        "oh",      "mt",       "clean",
                "rebuilt", "salvage", "lien",      "missing", "white",    "black",
                "red",     "silver",  "blue",      "grey",    "brown",    "green",
                "custom",  "orange",  "excellent", "good",    "like new", "fair",
                "",        ""};
            const auto coef_query = generate_query("LINEAR_REG_COEFS",
                                                   {{"model_name", quoted_model_name}},
                                                   {"coef_idx, sub_coef_idx"},
                                                   true /* make_args_named */);
            const auto coef_rows = run_multiple_agg(coef_query, dt);
            const int64_t num_rows = coef_rows->rowCount();
            EXPECT_EQ(num_rows, static_cast<int64_t>(expected_sub_features.size()));
            EXPECT_EQ(coef_rows->colCount(), size_t(5));
            constexpr double allowed_epsilon{0.1};
            for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
              auto crt_row = coef_rows->getNextRow(true, true);
              EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]),
                        expected_feature_ids[row_idx]);
              if (use_create_syntax) {
                const auto feature =
                    boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[1]));
                EXPECT_EQ(feature, expected_features[row_idx]);
              }
              EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[2]),
                        expected_sub_feature_ids[row_idx]);
              if (!expected_sub_features[row_idx].empty()) {
                const auto sub_coef =
                    boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[3]));
                EXPECT_EQ(sub_coef, expected_sub_features[row_idx]);
              }
              const double actual_coef = TestHelpers::v<double>(crt_row[4]);
              EXPECT_GE(actual_coef, expected_coefs[row_idx] - allowed_epsilon);
              EXPECT_LE(actual_coef, expected_coefs[row_idx] + allowed_epsilon);
            }
          } else if (model_type == MLModelType::RANDOM_FOREST_REG) {
            const std::vector<std::string> expected_features = {
                "state",        "state",        "state",        "state",
                "state",        "state",        "state",        "state",
                "state",        "state",        "title_status", "title_status",
                "title_status", "title_status", "title_status", "paint_color",
                "paint_color",  "paint_color",  "paint_color",  "paint_color",
                "paint_color",  "paint_color",  "paint_color",  "paint_color",
                "paint_color",  "condition_",   "condition_",   "condition_",
                "condition_",   "year_",        "odometer"};
            const std::vector<std::string> expected_sub_features = {
                "fl",      "ca",        "tx",      "mi",       "or",    "wa",
                "id",      "nc",        "oh",      "mt",       "clean", "rebuilt",
                "salvage", "lien",      "missing", "white",    "black", "red",
                "silver",  "blue",      "grey",    "brown",    "green", "custom",
                "orange",  "excellent", "good",    "like new", "fair",  "",
                ""};

            const std::vector<int64_t> expected_feature_ids = {
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6};
            const std::vector<int64_t> expected_sub_feature_ids = {
                1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 1, 2, 3, 4, 5, 1,
                2, 3, 4, 5, 6, 7, 8, 9, 10, 1,  2, 3, 4, 1, 1};

            const auto var_importance_query =
                generate_query("RANDOM_FOREST_REG_VAR_IMPORTANCE",
                               {{"model_name", quoted_model_name}},
                               {"feature_id, sub_feature_id"},
                               true /* make_args_named */);
            const auto var_importance_rows = run_multiple_agg(var_importance_query, dt);
            const int64_t num_rows = var_importance_rows->rowCount();
            EXPECT_EQ(num_rows, static_cast<int64_t>(expected_sub_features.size()));
            EXPECT_EQ(var_importance_rows->colCount(), size_t(5));
            for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
              auto crt_row = var_importance_rows->getNextRow(true, true);
              // Feature id
              EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]),
                        expected_feature_ids[row_idx]);
              if (use_create_syntax) {
                const auto feature =
                    boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[1]));
                EXPECT_EQ(feature, expected_features[row_idx]);
              }
              EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[2]),
                        expected_sub_feature_ids[row_idx]);
              if (!expected_sub_features[row_idx].empty()) {
                const auto sub_coef =
                    boost::get<std::string>(TestHelpers::v<NullableString>(crt_row[3]));
                EXPECT_EQ(sub_coef, expected_sub_features[row_idx]);
              }
              const double actual_var_importance = TestHelpers::v<double>(crt_row[4]);
              // Variable importances change wildly between runs on this dataset, so
              // let's do the minimum sanity check until we can make them deterministic
              EXPECT_GE(actual_var_importance, (double)0);
            }
          }
        }
      }
    }
  }
}

TEST_P(MLCategoricalRegressionFunctionsTest, ML_PREDICT) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_name{model_type_str + "_MODEL"};
  const std::string model_fit_func{model_type_str + "_FIT"};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  const double allowed_epsilon{0.1};
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    for (std::string numeric_data_type : {"DOUBLE"}) {
      const std::string train_query(
          "SELECT * FROM TABLE(" + model_fit_func + "(model_name =>'" + model_name +
          "', "
          "preferred_ml_framework=>" +
          ml_framework +
          ", data=>CURSOR(select  price, state, title_status, paint_color, "
          "condition_, year_, odometer FROM craigslist_f150s WHERE SAMPLE_RATIO(0.5)), "
          " cat_top_k=>50, cat_min_fraction=>0.001));");
      const std::string row_wise_predict_query(
          "SELECT AVG(ML_PREDICT('" + model_name +
          "', state, title_status, paint_color, condition_, year_, odometer)) "
          "FROM craigslist_f150s WHERE NOT SAMPLE_RATIO(0.5);");
      const std::string tf_predict_query(
          "SELECT AVG(prediction) FROM TABLE(ML_REG_PREDICT(model_name =>'" + model_name +
          "', "
          "preferred_ml_framework=>" +
          ml_framework +
          ", "
          "data=>CURSOR(SELECT rowid, state, title_status, paint_color, condition_, "
          "year_, odometer FROM craigslist_f150s WHERE NOT SAMPLE_RATIO(0.5))));");

      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        EXPECT_NO_THROW(run_multiple_agg(train_query, dt));
        const auto row_wise_prediction_avg =
            TestHelpers::v<double>(run_simple_agg(row_wise_predict_query, dt));
        const auto tf_prediction_avg =
            TestHelpers::v<double>(run_simple_agg(tf_predict_query, dt));
        EXPECT_GE(row_wise_prediction_avg, tf_prediction_avg - allowed_epsilon);
        EXPECT_LE(row_wise_prediction_avg, tf_prediction_avg + allowed_epsilon);
      }
    }
  }
}

TEST_P(MLCategoricalRegressionFunctionsTest, R2_SCORE) {
  const auto model_type = GetParam();
  const auto model_type_str = get_ml_model_type_str(model_type);
  const std::string model_name{model_type_str + "_MODEL"};
  const std::string model_fit_func{model_type_str + "_FIT"};
  const auto supported_ml_frameworks = get_supported_ml_frameworks();
  const double allowed_epsilon{0.01};
  for (auto& ml_framework : supported_ml_frameworks) {
    if (model_type != MLModelType::LINEAR_REG && ml_framework == "'mlpack'") {
      continue;
    }
    for (std::string numeric_data_type : {"DOUBLE"}) {
      const std::string train_query(
          "SELECT * FROM TABLE(" + model_fit_func + "(model_name =>'" + model_name +
          "', "
          "preferred_ml_framework=>" +
          ml_framework +
          ", data=>CURSOR(select  price, state, title_status, paint_color, "
          "condition_, year_, odometer FROM craigslist_f150s WHERE SAMPLE_RATIO(0.5)), "
          " cat_top_k=>50, cat_min_fraction=>0.001));");
      const std::string row_wise_r2_query(
          "SELECT 1.0 - (SUM(POWER(price - ML_PREDICT('" + model_name +
          "', state, title_status, paint_color, condition_, year_, odometer), 2)) "
          "/ SUM(POWER(price - (SELECT AVG(price) FROM craigslist_f150s WHERE NOT "
          "SAMPLE_RATIO(0.5)), 2))) FROM "
          "craigslist_f150s WHERE NOT "
          "SAMPLE_RATIO(0.5) AND price IS NOT NULL "
          "AND year_ IS NOT NULL AND odometer IS NOT NULL;");
      const std::string tf_r2_query(
          "SELECT r2 FROM TABLE(R2_SCORE(model_name =>'" + model_name +
          "', data=>CURSOR(SELECT price, state, title_status, paint_color, condition_, "
          "year_, odometer FROM craigslist_f150s WHERE NOT SAMPLE_RATIO(0.5))));");

      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        EXPECT_NO_THROW(run_multiple_agg(train_query, dt));
        const auto row_wise_r2 =
            TestHelpers::v<double>(run_simple_agg(row_wise_r2_query, dt));
        const auto tf_r2 = TestHelpers::v<double>(run_simple_agg(tf_r2_query, dt));
        EXPECT_GE(row_wise_r2, 0.0);
        EXPECT_LE(row_wise_r2, 1.0);
        EXPECT_GE(row_wise_r2, tf_r2 - allowed_epsilon);
        EXPECT_LE(row_wise_r2, tf_r2 + allowed_epsilon);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(MLCategoricalRegressionTests,
                         MLCategoricalRegressionFunctionsTest,
                         ::testing::Values(MLModelType::LINEAR_REG,
                                           MLModelType::DECISION_TREE_REG,
                                           MLModelType::GBT_REG,
                                           MLModelType::RANDOM_FOREST_REG));

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  // Table function support must be enabled before initialized the query runner
  // environment
  g_enable_table_functions = true;
  g_enable_ml_functions = true;
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
