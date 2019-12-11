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

#include <string>
#include <vector>

#include "../QueryEngine/Execute.h"
#include "../QueryRunner/QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

bool g_keep_data{false};

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

inline void run_ddl_statement(const std::string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(
      query_str, device_type, /*hoist_literals=*/true, /*allow_loop_joins=*/true);
}

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

const char* array_ext_ops_schema = R"(
    CREATE TABLE array_ext_ops_test (
        i64 BIGINT,
        i32 INT,
        i16 SMALLINT,
        i8 TINYINT,
        i1 BOOLEAN,
        str TEXT ENCODING DICT(32),
        arri64 BIGINT[],
        arri32 INT[],
        arri16 SMALLINT[],
        arri8 TINYINT[],
        arri1 BOOLEAN[],
        arrstr TEXT[]
    );
)";

class ArrayExtOpsEnv : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS array_ext_ops_test;"));

    ASSERT_NO_THROW(run_ddl_statement(array_ext_ops_schema));
    ValuesGenerator gen("array_ext_ops_test");
    run_multiple_agg(gen(3,
                         3,
                         3,
                         3,
                         "'true'",
                         "'c'",
                         "{1, 2}",
                         "{1, 2}",
                         "{1, 2}",
                         "{1, 2}",
                         "{'true', 'false'}",
                         "{'a', 'b'}"),
                     ExecutorDeviceType::CPU);
    run_multiple_agg(
        gen(1, 1, 1, 1, "'false'", "'a'", "{}", "{}", "{}", "{}", "{}", "{}"),
        ExecutorDeviceType::CPU);
    run_multiple_agg(gen(0,
                         0,
                         0,
                         0,
                         "'false'",
                         "'a'",
                         "{-1}",
                         "{-1}",
                         "{-1}",
                         "{-1}",
                         "{'true'}",
                         "{'z'}"),
                     ExecutorDeviceType::CPU);
    run_multiple_agg(
        gen(0, 0, 0, 0, "'false'", "'a'", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL"),
        ExecutorDeviceType::CPU);
    run_multiple_agg(gen("NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "{4, 5}",
                         "{4, 5}",
                         "{4, 5}",
                         "{4, 5}",
                         "{'false', 'true'}",
                         "{'d', 'e'}"),
                     ExecutorDeviceType::CPU);
    run_multiple_agg(gen("NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL"),
                     ExecutorDeviceType::CPU);
  }

  void TearDown() override {
    if (!g_keep_data) {
      ASSERT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS array_ext_ops_test;"));
    }
  }
};

TEST_F(ArrayExtOpsEnv, ArrayAppend) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    const auto rows =
        run_multiple_agg("SELECT array_append(arri64, i64) FROM array_ext_ops_test;", dt);
    ASSERT_EQ(rows->rowCount(), size_t(6));

    auto check_row_result = [](const auto& crt_row, const auto& expected) {
      compare_array(crt_row[0], expected);
    };

    check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{1, 2, 3});
    check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{1});
    check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{-1, 0});
    check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{0});
    check_row_result(rows->getNextRow(true, true),
                     std::vector<int64_t>{4, 5, inline_int_null_value<int64_t>()});
    check_row_result(rows->getNextRow(true, true),
                     std::vector<int64_t>{inline_int_null_value<int64_t>()});
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("keep-data", "Don't drop tables at the end of the tests");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("keep-data")) {
    g_keep_data = true;
  }

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
