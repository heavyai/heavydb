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

extern bool g_is_test_env;

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
        d DOUBLE,
        f FLOAT,
        i1 BOOLEAN,
        str TEXT ENCODING DICT(32),
        arri64 BIGINT[],
        arri32 INT[],
        arri16 SMALLINT[],
        arri8 TINYINT[],
        arrd DOUBLE[],
        arrf FLOAT[],
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
                         3,
                         3,
                         "'true'",
                         "'c'",
                         "{1, 2}",
                         "{1, 2}",
                         "{1, 2}",
                         "{1, 2}",
                         "{1, 2}",
                         "{1, 2}",
                         "{'true', 'false'}",
                         "{'a', 'b'}"),
                     ExecutorDeviceType::CPU);
    run_multiple_agg(gen(1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         "'false'",
                         "'a'",
                         "{}",
                         "{}",
                         "{}",
                         "{}",
                         "{}",
                         "{}",
                         "{}",
                         "{}"),
                     ExecutorDeviceType::CPU);
    run_multiple_agg(gen(0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         "'false'",
                         "'a'",
                         "{-1}",
                         "{-1}",
                         "{-1}",
                         "{-1}",
                         "{-1}",
                         "{-1}",
                         "{'true'}",
                         "{'z'}"),
                     ExecutorDeviceType::CPU);
    run_multiple_agg(gen(0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         "'false'",
                         "'a'",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL"),
                     ExecutorDeviceType::CPU);
    run_multiple_agg(gen("NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "NULL",
                         "{4, 5}",
                         "{4, 5}",
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

TEST_F(ArrayExtOpsEnv, ArrayAppendInteger) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    auto check_row_result = [](const auto& crt_row, const auto& expected) {
      compare_array(crt_row[0], expected);
    };

    auto check_entire_integer_result = [&check_row_result](const auto& rows,
                                                           const int64_t null_sentinel) {
      ASSERT_EQ(rows->rowCount(), size_t(6));

      check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{1, 2, 3});
      check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{1});
      check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{-1, 0});
      check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{0});
      check_row_result(rows->getNextRow(true, true),
                       std::vector<int64_t>{4, 5, null_sentinel});
      check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{null_sentinel});
    };

    // i64
    {
      const auto rows = run_multiple_agg(
          "SELECT array_append(arri64, i64) FROM array_ext_ops_test;", dt);
      check_entire_integer_result(rows, inline_int_null_value<int64_t>());
    }

    // i32
    {
      const auto rows = run_multiple_agg(
          "SELECT array_append(arri32, i32) FROM array_ext_ops_test;", dt);
      check_entire_integer_result(rows, inline_int_null_value<int32_t>());
    }

    // i16
    {
      const auto rows = run_multiple_agg(
          "SELECT array_append(arri16, i16) FROM array_ext_ops_test;", dt);
      check_entire_integer_result(rows, inline_int_null_value<int16_t>());
    }

    // i8
    {
      const auto rows =
          run_multiple_agg("SELECT array_append(arri8, i8) FROM array_ext_ops_test;", dt);
      check_entire_integer_result(rows, inline_int_null_value<int8_t>());
    }

    // upcast
    {
      const auto rows = run_multiple_agg(
          "SELECT array_append(arri64, i8) FROM array_ext_ops_test;", dt);
      check_entire_integer_result(rows, inline_int_null_value<int64_t>());
    }
  }
}

/* 22 Oct 20 MAT Disabling this test as currently boolean arrays
 * are broken and we need to fix the undelying array and then barray_append
 */
TEST_F(ArrayExtOpsEnv, DISABLED_ArrayAppendBool) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto check_row_result = [](const auto& crt_row, const auto& expected) {
      compare_array(crt_row[0], expected);
    };

    auto check_entire_bool_result = [&check_row_result](const auto& rows,
                                                        const int64_t null_sentinel) {
      ASSERT_EQ(rows->rowCount(), size_t(6));

      check_row_result(rows->getNextRow(true, true),
                       std::vector<int64_t>{true, false, true});
      check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{false});
      check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{true, false});
      check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{false});
      check_row_result(rows->getNextRow(true, true),
                       std::vector<int64_t>{false, true, null_sentinel});
      check_row_result(rows->getNextRow(true, true), std::vector<int64_t>{null_sentinel});
    };

    // bool
    {
      const auto rows = run_multiple_agg(
          "SELECT barray_append(arri1, i1) FROM array_ext_ops_test;", dt);
      check_entire_bool_result(rows, inline_int_null_value<int8_t>());
    }
  }
}

TEST_F(ArrayExtOpsEnv, ArrayAppendDouble) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto check_row_result = [](const auto& crt_row, const auto& expected) {
      compare_array(crt_row[0], expected);
    };
    auto check_entire_double_result = [&check_row_result](const auto& rows) {
      ASSERT_EQ(rows->rowCount(), size_t(6));

      check_row_result(rows->getNextRow(true, true), std::vector<double>{1, 2, 3});
      check_row_result(rows->getNextRow(true, true), std::vector<double>{1});
      check_row_result(rows->getNextRow(true, true), std::vector<double>{-1, 0});
      check_row_result(rows->getNextRow(true, true), std::vector<double>{0});
      check_row_result(rows->getNextRow(true, true),
                       std::vector<double>{4, 5, inline_fp_null_value<double>()});
      check_row_result(rows->getNextRow(true, true),
                       std::vector<double>{inline_fp_null_value<double>()});
    };

    // double
    {
      const auto rows =
          run_multiple_agg("SELECT array_append(arrd, d) FROM array_ext_ops_test;", dt);
      check_entire_double_result(rows);
    }
  }
}

TEST_F(ArrayExtOpsEnv, ArrayAppendFloat) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    auto check_row_result = [](const auto& crt_row, const auto& expected) {
      compare_array(crt_row[0], expected);
    };
    auto check_entire_float_result = [&check_row_result](const auto& rows) {
      ASSERT_EQ(rows->rowCount(), size_t(6));

      check_row_result(rows->getNextRow(true, true), std::vector<float>{1, 2, 3});
      check_row_result(rows->getNextRow(true, true), std::vector<float>{1});
      check_row_result(rows->getNextRow(true, true), std::vector<float>{-1, 0});
      check_row_result(rows->getNextRow(true, true), std::vector<float>{0});
      check_row_result(rows->getNextRow(true, true),
                       std::vector<float>{4, 5, inline_fp_null_value<float>()});
      check_row_result(rows->getNextRow(true, true),
                       std::vector<float>{inline_fp_null_value<float>()});
    };

    // float
    {
      const auto rows =
          run_multiple_agg("SELECT array_append(arrf, f) FROM array_ext_ops_test;", dt);
      check_entire_float_result(rows);
    }
  }
}

TEST_F(ArrayExtOpsEnv, ArrayAppendDowncast) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // unsupported downcast
    {
      EXPECT_ANY_THROW(run_multiple_agg(
          "SELECT array_append(arri32, i64) FROM array_ext_ops_test;", dt));
    }
  }
}

class FixedEncodedArrayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i64_i32;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i64_i16;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i64_i8;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i32_i8;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i32_i16;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i16_i8;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_dt64_dt32;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_dt64_dt16;");
    run_ddl_statement("DROP TABLE IF EXISTS farr_dt64_dt32;");
    run_ddl_statement("DROP TABLE IF EXISTS farr_dt64_dt16;");
  }

  void TearDown() override {
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i64_i32;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i64_i16;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i64_i8;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i32_i8;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i32_i16;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_i16_i8;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_dt64_dt32;");
    run_ddl_statement("DROP TABLE IF EXISTS vfarr_dt64_dt16;");
    run_ddl_statement("DROP TABLE IF EXISTS farr_dt64_dt32;");
    run_ddl_statement("DROP TABLE IF EXISTS farr_dt64_dt16;");
  }
};

TEST_F(FixedEncodedArrayTest, ExceptionTest) {
  // Check whether we throw exception for the below cases instead of crashes
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE varr_i64_i32 (val BIGINT[] ENCODING FIXED(32));"));
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE varr_i64_i16 (val BIGINT[] ENCODING FIXED(16));"));
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE varr_i64_i8 (val BIGINT[] ENCODING FIXED(8));"));
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE varr_i32_i8 (val INT[] ENCODING FIXED(16));"));
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE varr_i32_i16 (val INT[] ENCODING FIXED(8));"));
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE varr_i16_i8 (val SMALLINT[] ENCODING FIXED(8));"));
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE varr_dt64_dt32 (val DATE[] ENCODING FIXED(32));"));
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE varr_dt64_dt16 (val DATE[] ENCODING FIXED(16));"));
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE farr_dt64_dt32 (val DATE[1] ENCODING FIXED(32));"));
  ASSERT_ANY_THROW(
      run_ddl_statement("CREATE TABLE farr_dt64_dt16 (val DATE[1] ENCODING FIXED(16));"));
}

int main(int argc, char** argv) {
  g_is_test_env = true;

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
