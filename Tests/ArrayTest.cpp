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

TargetValue execSQLWithAllowLoopJoin(const std::string& stmt,
                                     const ExecutorDeviceType dt,
                                     const bool geo_return_geo_tv = true) {
  auto rows = QR::get()->runSQL(stmt, dt, true, true);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << stmt;
  return crt_row[0];
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
  void SetUp() override { dropTestTables(); }

  void TearDown() override { dropTestTables(); }

 private:
  void dropTestTables() {
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

class TinyIntArrayImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP TABLE IF EXISTS tinyint_arr;");
    run_ddl_statement("CREATE TABLE tinyint_arr (ti tinyint[]);");
  }

  void TearDown() override { run_ddl_statement("DROP TABLE IF EXISTS tinyint_arr;"); }
};

TEST_F(TinyIntArrayImportTest, TinyIntImportBugTest) {
  ASSERT_NO_THROW(QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({1});",
                                    ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(QR::get()->runSQL("INSERT INTO tinyint_arr VALUES (NULL);",
                                    ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({1});",
                                    ExecutorDeviceType::CPU));

  TearDown();
  SetUp();
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({1});",
                                    ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(QR::get()->runSQL("INSERT INTO tinyint_arr VALUES (NULL);",
                                    ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({1});",
                                    ExecutorDeviceType::CPU));

  TearDown();
  SetUp();
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({1});",
                                    ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(QR::get()->runSQL("INSERT INTO tinyint_arr VALUES (NULL);",
                                    ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(
      QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({});", ExecutorDeviceType::CPU));
  ASSERT_NO_THROW(QR::get()->runSQL("INSERT INTO tinyint_arr VALUES ({1});",
                                    ExecutorDeviceType::CPU));
}

class MultiFragArrayJoinTest : public ::testing::Test {
 protected:
  void SetUp() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr_n;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr_n;");
    std::vector<std::string> integer_type_cols{"tiv tinyint[]",
                                               "tif tinyint[3]",
                                               "siv smallint[]",
                                               "sif smallint[3]",
                                               "intv int[]",
                                               "intf int[3]",
                                               "biv bigint[]",
                                               "bif bigint[3]"};
    std::vector<std::string> floating_type_cols{"dv double[]",
                                                "df double[3]",
                                                "dcv decimal(18,6)[]",
                                                "dcf decimal(18,6)[3]",
                                                "fv float[]",
                                                "ff float[3]"};
    std::vector<std::string> date_and_time_type_cols{"dtv date[]",
                                                     "dtf date[3]",
                                                     "tv time[]",
                                                     "tf time[3]",
                                                     "tsv timestamp[]",
                                                     "tsf timestamp[3]"};
    std::vector<std::string> text_type_cols{"tx text",
                                            "txe4 text encoding dict(32)",
                                            "txe2 text encoding dict(16)",
                                            "txe1 text encoding dict(8)",
                                            "txn text encoding none",
                                            "txv text[]",
                                            "txve text[] encoding dict (32)"};
    std::vector<std::string> boolean_type_cols{"boolv boolean[]", "boolf boolean[3]"};
    auto create_table_ddl_gen = [&integer_type_cols,
                                 &floating_type_cols,
                                 &date_and_time_type_cols,
                                 &text_type_cols,
                                 &boolean_type_cols](
                                    const std::string& tbl_name,
                                    const bool multi_frag,
                                    const bool has_integer_types = true,
                                    const bool has_floting_types = true,
                                    const bool has_date_and_time_types = true,
                                    const bool has_text_types = true,
                                    const bool has_boolean_type = true) {
      std::vector<std::string> cols;
      if (has_integer_types) {
        std::copy(
            integer_type_cols.begin(), integer_type_cols.end(), std::back_inserter(cols));
      }
      if (has_floting_types) {
        std::copy(floating_type_cols.begin(),
                  floating_type_cols.end(),
                  std::back_inserter(cols));
      }
      if (has_date_and_time_types) {
        std::copy(date_and_time_type_cols.begin(),
                  date_and_time_type_cols.end(),
                  std::back_inserter(cols));
      }
      if (has_text_types) {
        std::copy(text_type_cols.begin(), text_type_cols.end(), std::back_inserter(cols));
      }
      if (has_boolean_type) {
        std::copy(
            boolean_type_cols.begin(), boolean_type_cols.end(), std::back_inserter(cols));
      }
      auto table_cols_ddl = boost::join(cols, ",");
      auto table_ddl = "(" + table_cols_ddl;
      std::ostringstream oss;
      oss << "CREATE TABLE " << tbl_name << table_ddl;
      if (has_text_types &&
          (tbl_name.compare("mfarr") == 0 || tbl_name.compare("sfarr") == 0)) {
        // exclude these cols for multi-frag nullable case to avoid not accepted null
        // value issue (i.e., RelAlgExecutor::2833)
        oss << ", txf text[3], txfe text[3] encoding dict (32)";
      };
      oss << ")";
      if (multi_frag) {
        oss << " WITH (FRAGMENT_SIZE = 5)";
      }
      oss << ";";
      return oss.str();
    };

    // this test case works on chunk 0 ~ chunk 32 where each chunk contains five rows
    // and for nullable case, we manipulate how null rows are located in each chunk
    // here, start_chunk_idx and end_chunk_idx control which null patterns we inserted
    // into a test table i.e., start_chunk_idx ~ end_chunk_idx --> 0 ~ 2
    // --> conrresponding table contains three null pattern: 00000, 00001, 000010
    // where 1's row contains nulled row, i.e., 00010 --> fourth row is null row
    // if we use frag size to be five, using 2^5 = 32 patterns can test every possible
    // chunk's row status to see our linearization logic works well under such various
    // fragments
    auto insert_dml_gen = [](const std::string& tbl_name,
                             const bool allow_null,
                             int start_chunk_idx = 0,
                             int end_chunk_idx = 32,
                             const bool has_integer_types = true,
                             const bool has_floting_types = true,
                             const bool has_date_and_time_types = true,
                             const bool has_text_types = true,
                             const bool has_boolean_type = true) {
      std::ostringstream oss;
      oss << "INSERT INTO " << tbl_name << " VALUES(";
      auto gi = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{" << i + 1 << "}";
        } else if (i % 3 == 2) {
          oss << "{" << i << "," << i + 2 << "}";
        } else {
          oss << "{" << i + 1 << "," << i + 2 << "," << i + 3 << "}";
        }
        return oss.str();
      };
      auto gf = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{" << i << "." << i << "}";
        } else if (i % 3 == 2) {
          oss << "{" << i << "." << i << "," << i * 2 << "." << i * 2 << "}";
        } else {
          oss << "{" << i + 1 << "." << i + 1 << "," << i + 2 << "." << i + 2 << ","
              << i + 3 << "." << i + 3 << "}";
        }
        return oss.str();
      };
      auto gdt = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{\'01/01/2021\'}";
        } else if (i % 3 == 2) {
          oss << "{\'01/01/2021\',\'01/02/2021\'}";
        } else {
          oss << "{\'01/01/2021\',\'01/02/2021\',\'01/03/2021\'}";
        }
        return oss.str();
      };
      auto gt = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{\'01:01:01\'}";
        } else if (i % 3 == 2) {
          oss << "{\'01:01:01\',\'01:01:02\'}";
        } else {
          oss << "{\'01:01:01\',\'01:01:02\',\'01:01:03\'}";
        }
        return oss.str();
      };
      auto gts = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{\'01/01/2021 01:01:01\'}";
        } else if (i % 3 == 2) {
          oss << "{\'01/01/2021 01:01:01\',\'01/01/2021 01:01:02\'}";
        } else {
          oss << "{\'01/01/2021 01:01:01\',\'01/01/2021 01:01:02\', \'01/01/2021 "
                 "01:01:03\'}";
        }
        return oss.str();
      };
      auto gtx = [](const int i, bool array) {
        std::ostringstream oss;
        if (array) {
          if (i % 3 == 1) {
            oss << "{\'" << i << "\'}";
          } else if (i % 3 == 2) {
            oss << "{\'" << i << "\',\'" << i << i << "\'}";
          } else {
            oss << "{\'" << i << "\',\'" << i << i << "\',\'" << i << i << i << "\'}";
          }
        } else {
          if (i % 3 == 1) {
            oss << "\'" << i << "\'";
          } else if (i % 3 == 2) {
            oss << "\'" << i << i << "\'";
          } else {
            oss << "\'" << i << i << i << "\'";
          }
        }
        return oss.str();
      };
      auto gbool = [](const int i) {
        std::ostringstream oss;
        auto bool_val = i % 3 == 0 ? "\'true\'" : "\'false\'";
        if (i % 3 == 1) {
          oss << "{" << bool_val << "}";
        } else if (i % 3 == 2) {
          oss << "{" << bool_val << "," << bool_val << "}";
        } else {
          oss << "{" << bool_val << "," << bool_val << "," << bool_val << "}";
        }
        return oss.str();
      };
      size_t col_count = 0;
      if (has_integer_types) {
        col_count += 8;
      }
      if (has_floting_types) {
        col_count += 6;
      }
      if (has_date_and_time_types) {
        col_count += 6;
      }
      if (has_text_types) {
        col_count += 7;
      }
      if (has_boolean_type) {
        col_count += 2;
      }
      if (has_text_types &&
          (tbl_name.compare("mfarr") == 0 || tbl_name.compare("sfarr") == 0)) {
        col_count += 2;
      }
      std::vector<std::string> null_vec(col_count, "NULL");
      auto null_str = boost::join(null_vec, ",");
      auto g = [&tbl_name,
                &gi,
                &gf,
                &gt,
                &gdt,
                &gts,
                &gtx,
                &gbool,
                &null_str,
                &has_integer_types,
                &has_floting_types,
                &has_date_and_time_types,
                &has_text_types,
                &has_boolean_type](const int i, int null_row) {
        std::ostringstream oss;
        std::vector<std::string> col_vals;
        oss << "INSERT INTO " << tbl_name << " VALUES(";
        if (null_row) {
          oss << null_str;
        } else {
          if (has_integer_types) {
            col_vals.push_back(gi(i % 120));
            col_vals.push_back(gi(3));
            col_vals.push_back(gi(i));
            col_vals.push_back(gi(3));
            col_vals.push_back(gi(i));
            col_vals.push_back(gi(3));
            col_vals.push_back(gi(i));
            col_vals.push_back(gi(3));
          }
          if (has_floting_types) {
            col_vals.push_back(gf(i));
            col_vals.push_back(gf(3));
            col_vals.push_back(gf(i));
            col_vals.push_back(gf(3));
            col_vals.push_back(gf(i));
            col_vals.push_back(gf(3));
          }
          if (has_date_and_time_types) {
            col_vals.push_back(gdt(i));
            col_vals.push_back(gdt(3));
            col_vals.push_back(gt(i));
            col_vals.push_back(gt(3));
            col_vals.push_back(gts(i));
            col_vals.push_back(gts(3));
          }
          if (has_text_types) {
            col_vals.push_back(gtx(i, false));
            col_vals.push_back(gtx(i, false));
            col_vals.push_back(gtx(i, false));
            col_vals.push_back(gtx(i, false));
            col_vals.push_back(gtx(i, false));
            col_vals.push_back(gtx(i, true));
            col_vals.push_back(gtx(i, true));
          }
          if (has_boolean_type) {
            col_vals.push_back(gbool(i));
            col_vals.push_back(gbool(3));
          }
          if (has_text_types &&
              (tbl_name.compare("mfarr") == 0 || tbl_name.compare("sfarr") == 0)) {
            col_vals.push_back(gtx(3, true));
            col_vals.push_back(gtx(3, true));
          }
        }
        oss << boost::join(col_vals, ",");
        oss << ");";
        return oss.str();
      };
      std::vector<std::string> insert_dml_vec;
      if (allow_null) {
        int i = 1;
        for (int chunk_idx = start_chunk_idx; chunk_idx < end_chunk_idx; chunk_idx++) {
          std::string str = std::bitset<5>(chunk_idx).to_string();
          for (int row_idx = 0; row_idx < 5; row_idx++, i++) {
            bool null_row = str.at(row_idx) == '1';
            if (null_row) {
              insert_dml_vec.push_back(g(0, true));
            } else {
              insert_dml_vec.push_back(g(i, false));
            }
          }
        }
      } else {
        for (auto chunk_idx = start_chunk_idx; chunk_idx < end_chunk_idx; chunk_idx++) {
          for (auto row_idx = 0; row_idx < 5; row_idx++) {
            auto i = (chunk_idx * 5) + row_idx;
            insert_dml_vec.push_back(g(i, false));
          }
        }
      }
      for (auto& dml : insert_dml_vec) {
        QR::get()->runSQL(dml, ExecutorDeviceType::CPU);
      }
      col_count = 0;
    };
    QR::get()->runDDLStatement(create_table_ddl_gen("sfarr", false));
    QR::get()->runDDLStatement(create_table_ddl_gen("mfarr", true));
    QR::get()->runDDLStatement(create_table_ddl_gen("sfarr_n", false));
    QR::get()->runDDLStatement(create_table_ddl_gen("mfarr_n", true));

    insert_dml_gen("sfarr", false);
    insert_dml_gen("mfarr", false);
    insert_dml_gen("sfarr_n", true);
    insert_dml_gen("mfarr_n", true);
  }

  void TearDown() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr_n;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr_n;");
  }
};

TEST_F(MultiFragArrayJoinTest, IndexedArrayJoin) {
  // 1. non-nullable array
  {
    std::vector<std::string> integer_types{
        "tiv", "tif", "siv", "sif", "intv", "intf", "biv", "bif"};
    std::vector<std::string> float_types{"dv", "df", "dcv", "dcf", "fv", "ff"};
    std::vector<std::string> date_types{"dtv", "dtf"};
    std::vector<std::string> time_types{"tv", "tf"};
    std::vector<std::string> timestamp_types{"tsv", "tsf"};
    std::vector<std::string> non_encoded_text_types{"tx", "txe4", "txe2", "txe1", "txn"};
    std::vector<std::string> text_array_types{"txv", "txve", "txf", "txfe"};
    std::vector<std::string> bool_types{"boolf", "boolv"};
    auto run_tests = [](const std::vector<std::string> col_types) {
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        for (const auto& col_name : col_types) {
          std::ostringstream sq, q1;
          auto join_cond = "s." + col_name + "[1] = r." + col_name + "[1];";
          if (col_name.compare("tx") == 0 || col_name.compare("txe4") == 0 ||
              col_name.compare("txe2") == 0 || col_name.compare("txe1") == 0 ||
              col_name.compare("txn") == 0) {
            join_cond = "s." + col_name + " = r." + col_name + ";";
          }
          sq << "SELECT COUNT(1) FROM sfarr s, sfarr r WHERE " << join_cond;
          q1 << "SELECT COUNT(1) FROM mfarr r, mfarr s WHERE " << join_cond;
          auto single_frag_res = v<int64_t>(execSQLWithAllowLoopJoin(sq.str(), dt));
          auto res1 = v<int64_t>(execSQLWithAllowLoopJoin(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    run_tests(date_types);
    run_tests(time_types);
    run_tests(timestamp_types);
    run_tests(non_encoded_text_types);
    run_tests(text_array_types);
    run_tests(bool_types);
  }

  // 2. nullable array
  {
    std::vector<std::string> integer_types{
        "tiv", "tif", "siv", "sif", "intv", "intf", "biv", "bif"};
    std::vector<std::string> float_types{"dv", "df", "dcv", "dcf", "fv", "ff"};
    std::vector<std::string> date_types{"dtv", "dtf"};
    std::vector<std::string> time_types{"tv", "tf"};
    std::vector<std::string> timestamp_types{"tsv", "tsf"};
    std::vector<std::string> non_encoded_text_types{"tx", "txe4", "txe2", "txe1", "txn"};
    std::vector<std::string> text_array_types{
        "txv", "txve"};  // skip txf text[3] / txfe text[3] encoding dict (32)
    std::vector<std::string> bool_types{"boolf", "boolv"};
    auto run_tests = [](const std::vector<std::string> col_types) {
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        for (const auto& col_name : col_types) {
          std::ostringstream sq, q1;
          auto join_cond = "s." + col_name + "[1] = r." + col_name + "[1];";
          if (col_name.compare("tx") == 0 || col_name.compare("txe4") == 0 ||
              col_name.compare("txe2") == 0 || col_name.compare("txe1") == 0 ||
              col_name.compare("txn") == 0) {
            join_cond = "s." + col_name + " = r." + col_name + ";";
          }
          sq << "SELECT COUNT(1) FROM sfarr_n s, sfarr_n r WHERE " << join_cond;
          q1 << "SELECT COUNT(1) FROM mfarr_n r, mfarr_n s WHERE " << join_cond;
          auto single_frag_res = v<int64_t>(execSQLWithAllowLoopJoin(sq.str(), dt));
          auto res1 = v<int64_t>(execSQLWithAllowLoopJoin(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    run_tests(date_types);
    run_tests(time_types);
    run_tests(timestamp_types);
    run_tests(non_encoded_text_types);
    run_tests(text_array_types);
    run_tests(bool_types);
  }
}

class MultiFragArrayParallelLinearizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr_n;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr_n;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr_n2;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr_n2;");
    g_enable_parallel_linearization = 10;

    std::vector<std::string> cols{"tiv tinyint[]",
                                  "siv smallint[]",
                                  "intv int[]",
                                  "biv bigint[]",
                                  "dv double[]",
                                  "dcv decimal(18,6)[]",
                                  "fv float[]",
                                  "dtv date[]",
                                  "tv time[]",
                                  "tsv timestamp[]",
                                  "txv text[]",
                                  "txve text[] encoding dict (32)",
                                  "boolv boolean[]"};
    auto table_cols_ddl = boost::join(cols, ",");
    auto create_table_ddl_gen = [&table_cols_ddl](const std::string& tbl_name,
                                                  const bool multi_frag) {
      auto table_ddl = "(" + table_cols_ddl;
      std::ostringstream oss;
      oss << "CREATE TABLE " << tbl_name << table_ddl << ")";
      if (multi_frag) {
        oss << " WITH (FRAGMENT_SIZE = 5)";
      }
      oss << ";";
      return oss.str();
    };

    QR::get()->runDDLStatement(create_table_ddl_gen("sfarr", false));
    QR::get()->runDDLStatement(create_table_ddl_gen("mfarr", true));
    QR::get()->runDDLStatement(create_table_ddl_gen("sfarr_n", false));
    QR::get()->runDDLStatement(create_table_ddl_gen("mfarr_n", true));
    QR::get()->runDDLStatement(create_table_ddl_gen("sfarr_n2", false));
    QR::get()->runDDLStatement(create_table_ddl_gen("mfarr_n2", true));

    auto insert_table = [](const std::string& tbl_name,
                           int num_tuples,
                           int frag_size,
                           bool allow_null,
                           bool first_frag_row_null) {
      std::ostringstream oss;
      oss << "INSERT INTO " << tbl_name << " VALUES(";
      auto gi = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{" << i + 1 << "}";
        } else if (i % 3 == 2) {
          oss << "{" << i << "," << i + 2 << "}";
        } else {
          oss << "{" << i + 1 << "," << i + 2 << "," << i + 3 << "}";
        }
        return oss.str();
      };
      auto gf = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{" << i << "." << i << "}";
        } else if (i % 3 == 2) {
          oss << "{" << i << "." << i << "," << i * 2 << "." << i * 2 << "}";
        } else {
          oss << "{" << i + 1 << "." << i + 1 << "," << i + 2 << "." << i + 2 << ","
              << i + 3 << "." << i + 3 << "}";
        }
        return oss.str();
      };
      auto gdt = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{\'01/01/2021\'}";
        } else if (i % 3 == 2) {
          oss << "{\'01/01/2021\',\'01/02/2021\'}";
        } else {
          oss << "{\'01/01/2021\',\'01/02/2021\',\'01/03/2021\'}";
        }
        return oss.str();
      };
      auto gt = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{\'01:01:01\'}";
        } else if (i % 3 == 2) {
          oss << "{\'01:01:01\',\'01:01:02\'}";
        } else {
          oss << "{\'01:01:01\',\'01:01:02\',\'01:01:03\'}";
        }
        return oss.str();
      };
      auto gts = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{\'01/01/2021 01:01:01\'}";
        } else if (i % 3 == 2) {
          oss << "{\'01/01/2021 01:01:01\',\'01/01/2021 01:01:02\'}";
        } else {
          oss << "{\'01/01/2021 01:01:01\',\'01/01/2021 01:01:02\', \'01/01/2021 "
                 "01:01:03\'}";
        }
        return oss.str();
      };
      auto gtx = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "{\'" << i << "\'}";
        } else if (i % 3 == 2) {
          oss << "{\'" << i << "\',\'" << i << i << "\'}";
        } else {
          oss << "{\'" << i << "\',\'" << i << i << "\',\'" << i << i << i << "\'}";
        }
        return oss.str();
      };
      auto gbool = [](const int i) {
        std::ostringstream oss;
        auto bool_val = i % 3 == 0 ? "\'true\'" : "\'false\'";
        if (i % 3 == 1) {
          oss << "{" << bool_val << "}";
        } else if (i % 3 == 2) {
          oss << "{" << bool_val << "," << bool_val << "}";
        } else {
          oss << "{" << bool_val << "," << bool_val << "," << bool_val << "}";
        }
        return oss.str();
      };
      size_t col_count = 13;
      std::vector<std::string> null_vec(col_count, "NULL");
      auto null_str = boost::join(null_vec, ",");
      auto g = [&tbl_name, &null_str, &gi, &gf, &gt, &gdt, &gts, &gtx, &gbool](
                   const int i, int null_row) {
        std::ostringstream oss;
        std::vector<std::string> col_vals;
        oss << "INSERT INTO " << tbl_name << " VALUES(";
        if (null_row) {
          oss << null_str;
        } else {
          auto i_r = gi(i);
          auto i_f = gf(i);
          auto i_tx = gtx(i);
          col_vals.push_back(gi(i % 120));
          col_vals.push_back(i_r);
          col_vals.push_back(i_r);
          col_vals.push_back(i_r);
          col_vals.push_back(i_f);
          col_vals.push_back(i_f);
          col_vals.push_back(i_f);
          col_vals.push_back(gdt(i));
          col_vals.push_back(gt(i));
          col_vals.push_back(gts(i));
          col_vals.push_back(i_tx);
          col_vals.push_back(i_tx);
          col_vals.push_back(gbool(i));
        }
        oss << boost::join(col_vals, ",");
        oss << ");";
        return oss.str();
      };
      std::vector<std::string> insert_dml_vec;
      if (allow_null) {
        for (int i = 0; i < num_tuples; i++) {
          if ((first_frag_row_null && (i % frag_size) == 0) || (i % 17 == 5)) {
            insert_dml_vec.push_back(g(0, true));
          } else {
            insert_dml_vec.push_back(g(i, false));
          }
        }
      } else {
        for (int i = 0; i < num_tuples; i++) {
          insert_dml_vec.push_back(g(i, false));
        }
      }

      for (auto& dml : insert_dml_vec) {
        QR::get()->runSQL(dml, ExecutorDeviceType::CPU);
      }
    };

    insert_table("sfarr", 300, 100, false, false);
    insert_table("mfarr", 300, 100, false, false);
    insert_table("sfarr_n", 300, 100, true, true);
    insert_table("mfarr_n", 300, 100, true, true);
    insert_table("sfarr_n2", 300, 100, true, false);
    insert_table("mfarr_n2", 300, 100, true, false);
  }

  void TearDown() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr_n;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr_n;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS sfarr_n2;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS mfarr_n2;");
    g_enable_parallel_linearization = 20000;
  }
};

TEST_F(MultiFragArrayParallelLinearizationTest, IndexedArrayJoin) {
  std::vector<std::string> integer_types{"tiv", "siv", "intv", "biv"};
  std::vector<std::string> float_types{"dv", "dcv", "fv"};
  std::vector<std::string> date_types{"dtv"};
  std::vector<std::string> time_types{"tv"};
  std::vector<std::string> timestamp_types{"tsv"};
  std::vector<std::string> text_array_types{"txv", "txve"};
  std::vector<std::string> bool_types{"boolv"};

  // case 1. non_null
  {
    auto run_tests = [](const std::vector<std::string> col_types) {
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        for (const auto& col_name : col_types) {
          std::ostringstream sq, q1;
          auto join_cond = "s." + col_name + "[1] = r." + col_name + "[1];";
          sq << "SELECT COUNT(1) FROM sfarr s, sfarr r WHERE " << join_cond;
          q1 << "SELECT COUNT(1) FROM mfarr r, mfarr s WHERE " << join_cond;
          auto single_frag_res = v<int64_t>(execSQLWithAllowLoopJoin(sq.str(), dt));
          auto res1 = v<int64_t>(execSQLWithAllowLoopJoin(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    run_tests(date_types);
    run_tests(time_types);
    run_tests(timestamp_types);
    run_tests(text_array_types);
    run_tests(bool_types);
  }

  // case 2-a. nullable having first row of each frag is null row
  {
    auto run_tests = [](const std::vector<std::string> col_types) {
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        for (const auto& col_name : col_types) {
          std::ostringstream sq, q1;
          auto join_cond = "s." + col_name + "[1] = r." + col_name + "[1];";
          sq << "SELECT COUNT(1) FROM sfarr_n s, sfarr_n r WHERE " << join_cond;
          q1 << "SELECT COUNT(1) FROM mfarr_n r, mfarr_n s WHERE " << join_cond;
          auto single_frag_res = v<int64_t>(execSQLWithAllowLoopJoin(sq.str(), dt));
          auto res1 = v<int64_t>(execSQLWithAllowLoopJoin(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    run_tests(date_types);
    run_tests(time_types);
    run_tests(timestamp_types);
    run_tests(text_array_types);
    run_tests(bool_types);
  }

  // case 2-b. nullable having first row of each frag is non-null row
  {
    auto run_tests = [](const std::vector<std::string> col_types) {
      for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
        SKIP_NO_GPU();
        for (const auto& col_name : col_types) {
          std::ostringstream sq, q1;
          auto join_cond = "s." + col_name + "[1] = r." + col_name + "[1];";
          sq << "SELECT COUNT(1) FROM sfarr_n2 s, sfarr_n2 r WHERE " << join_cond;
          q1 << "SELECT COUNT(1) FROM mfarr_n2 r, mfarr_n2 s WHERE " << join_cond;
          auto single_frag_res = v<int64_t>(execSQLWithAllowLoopJoin(sq.str(), dt));
          auto res1 = v<int64_t>(execSQLWithAllowLoopJoin(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    run_tests(date_types);
    run_tests(time_types);
    run_tests(timestamp_types);
    run_tests(text_array_types);
    run_tests(bool_types);
  }
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
