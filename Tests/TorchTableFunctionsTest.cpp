/*
 * Copyright 2023 HEAVY.AI, Inc.
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
#include "RuntimeLibManager/RuntimeLibManager.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

extern bool g_enable_table_functions;
extern bool g_enable_dev_table_functions;

using namespace TestHelpers;

namespace {

inline void run_ddl_statement(const std::string& stmt) {
  QR::get()->runDDLStatement(stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, false, false);
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

class TorchTFs : public ::testing::Test {
  void SetUp() override {
    if (RuntimeLibManager::is_libtorch_loaded()) {
      ASSERT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS torch_test;"));
      ASSERT_NO_THROW(run_ddl_statement("CREATE TABLE torch_test(x BIGINT, d DOUBLE);"));
      ASSERT_NO_THROW(run_ddl_statement("INSERT INTO torch_test VALUES(1, 1.0);"));
    }
  }
  void TearDown() override {
    if (RuntimeLibManager::is_libtorch_loaded()) {
      ASSERT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS torch_test;"));
    }
  }
};

#ifdef HAVE_TORCH_TFS
TEST_F(TorchTFs, TestFunctions) {
  if (!RuntimeLibManager::is_libtorch_loaded()) {
    FAIL() << "Attempted to run LibTorch tests, but LibTorch was not properly loaded at "
              "runtime!";
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // test that torch table functions exist/were loaded properly
    {
      if (dt == ExecutorDeviceType::CPU) {
        EXPECT_NO_THROW(
            run_multiple_agg("SELECT * FROM TABLE(tf_test_runtime_torch(CURSOR(SELECT x "
                             "from torch_test)));",
                             dt));
        EXPECT_NO_THROW(run_multiple_agg(
            "SELECT * FROM TABLE(tf_test_runtime_torch_template(CURSOR(SELECT x from "
            "torch_test)));",
            dt));
        EXPECT_NO_THROW(run_multiple_agg(
            "SELECT * FROM TABLE(tf_test_runtime_torch_template(CURSOR(SELECT d from "
            "torch_test)));",
            dt));
      }
    }

    // test training an actual model
    {
      constexpr double allowed_epsilon{0.15};
      // generate a table with 1 billion random (x, x^2, x^3, x^4) entries
      const std::string saved_model_filename = "test.pt";
      const std::string data_gen_query =
          "SELECT output, POWER(output, 2), POWER(output, 3), POWER(output, 4) FROM "
          "TABLE(tf_test_torch_generate_random_column(10000000))";
      // train a neural network to fit a 4-degree polynomial, with batch size 32, using
      // GPU if available, and save it to torchscript file "test.pt"
      const std::string train_query =
          "SELECT * from TABLE(tf_test_torch_regression(CURSOR(" + data_gen_query +
          "), 32, " + (dt == ExecutorDeviceType::CPU ? "false" : "true") + ", true, '" +
          saved_model_filename + "'));";
      const auto rows = run_multiple_agg(train_query, dt);
      for (size_t idx = 0; idx < rows->rowCount(); idx += 2) {
        auto fit_coef_row = rows->getNextRow(true, true);
        auto expected_coef_row = rows->getNextRow(true, true);
        const double fit_coef = TestHelpers::v<double>(fit_coef_row[0]);
        const double expected_coef = TestHelpers::v<double>(expected_coef_row[0]);
        EXPECT_GE(fit_coef, expected_coef - allowed_epsilon);
        EXPECT_LE(fit_coef, expected_coef + allowed_epsilon);

        // model_filename should be non-empty if save_model is true
        const std::string invalid_train_query2 =
            "SELECT * FROM TABLE(tf_test_torch_regression(CURSOR(" + data_gen_query +
            "), 32, " + (dt == ExecutorDeviceType::CPU ? "false" : "true") +
            ", true, ''))";
        EXPECT_THROW(run_multiple_agg(invalid_train_query2, dt), UserTableFunctionError);
      }
      {
        // load the previously created/saved torchscript model
        EXPECT_NO_THROW(
            run_multiple_agg("SELECT * from TABLE(tf_test_torch_load_model('" +
                                 saved_model_filename + "'));",
                             dt));

        // attempt to load invalid model name, expect throw
        EXPECT_THROW(
            run_multiple_agg(
                "SELECT * FROM TABLE(tf_test_torch_load_model('_bogus_model.pt'));", dt),
            UserTableFunctionError);
      }
    }
  }
}
#endif

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  // Table function support must be enabled before initialized the query runner
  // environment
  g_enable_table_functions = true;
  g_enable_dev_table_functions = true;
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