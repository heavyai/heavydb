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

#include "ArrowSQLRunner.h"
#include "TestHelpers.h"

#include <string>
#include <vector>

#include "../QueryEngine/Execute.h"

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

extern bool g_is_test_env;
extern bool g_enable_parallel_linearization;

using namespace TestHelpers;
using namespace TestHelpers::ArrowSQLRunner;

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

class ArrayExtOpsEnv : public ::testing::Test {
 protected:
  void SetUp() override {
    createTable("array_ext_ops_test",
                {{"i64", SQLTypeInfo(kBIGINT)},
                 {"i32", SQLTypeInfo(kINT)},
                 {"i16", SQLTypeInfo(kSMALLINT)},
                 {"i8", SQLTypeInfo(kTINYINT)},
                 {"d", SQLTypeInfo(kDOUBLE)},
                 {"f", SQLTypeInfo(kFLOAT)},
                 {"i1", SQLTypeInfo(kBOOLEAN)},
                 {"str", dictType()},
                 {"arri64", arrayType(kBIGINT)},
                 {"arri32", arrayType(kINT)},
                 {"arri16", arrayType(kSMALLINT)},
                 {"arri8", arrayType(kTINYINT)},
                 {"arrd", arrayType(kDOUBLE)},
                 {"arrf", arrayType(kFLOAT)},
                 {"arri1", arrayType(kBOOLEAN)},
                 {"arrstr", arrayType(kTEXT)}});

    insertJsonValues(
        "array_ext_ops_test",
        R"___({"i64": 3, "i32": 3, "i16": 3, "i8": 3, "d": 3, "f": 3, "i1": true, "str": "c", "arri64": [1, 2], "arri32": [1, 2], "arri16": [1, 2], "arri8": [1, 2], "arrd": [1, 2], "arrf": [1, 2], "arri1": [true, false], "arrstr": ["a", "b"]}
{"i64": 1, "i32": 1, "i16": 1, "i8": 1, "d": 1, "f": 1, "i1": false, "str": "a", "arri64": [], "arri32": [], "arri16": [], "arri8": [], "arrd": [], "arrf": [], "arri1": [], "arrstr": []}
{"i64": 0, "i32": 0, "i16": 0, "i8": 0, "d": 0, "f": 0, "i1": false, "str": "a", "arri64": [-1], "arri32": [-1], "arri16": [-1], "arri8": [-1], "arrd": [-1], "arrf": [-1], "arri1": [true], "arrstr": ["z"]}
{"i64": 0, "i32": 0, "i16": 0, "i8": 0, "d": 0, "f": 0, "i1": false, "str": "a", "arri64": null, "arri32": null, "arri16": null, "arri8": null, "arrd": null, "arrf": null, "arri1": null, "arrstr": null}
{"i64": null, "i32": null, "i16": null, "i8": null, "d": null, "f": null, "i1": null, "str": null, "arri64": [4, 5], "arri32": [4, 5], "arri16": [4, 5], "arri8": [4, 5], "arrd": [4, 5], "arrf": [4, 5], "arri1": [false, true], "arrstr": ["d", "e"]}
{"i64": null, "i32": null, "i16": null, "i8": null, "d": null, "f": null, "i1": null, "str": null, "arri64": null, "arri32": null, "arri16": null, "arri8": null, "arrd": null, "arrf": null, "arri1": null, "arrstr": null})___");
  }

  void TearDown() override { dropTable("array_ext_ops_test"); }
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

TEST_F(ArrayExtOpsEnv, ArrayAppendBool) {
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

class TinyIntArrayImportTest : public ::testing::Test {
 protected:
  void SetUp() override { createTable("tinyint_arr", {{"ti", arrayType(kTINYINT)}}); }

  void TearDown() override { dropTable("tinyint_arr"); }
};

TEST_F(TinyIntArrayImportTest, TinyIntImportBugTest) {
  insertJsonValues("tinyint_arr", "{\"ti\": [1]}");
  insertJsonValues("tinyint_arr", "{\"ti\": null}");
  insertJsonValues("tinyint_arr", "{\"ti\": [1]}");

  TearDown();
  SetUp();
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": [1]}");
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": [null]}");
  insertJsonValues("tinyint_arr", "{\"ti\": [1]}");

  TearDown();
  SetUp();
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": [1]}");
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": [null]}");
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": []}");
  insertJsonValues("tinyint_arr", "{\"ti\": [1]}");
}

class MultiFragArrayJoinTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<ArrowStorage::ColumnDescription> integer_type_cols{
        {"tiv", arrayType(kTINYINT)},
        {"tif", arrayType(kTINYINT, 3)},
        {"siv", arrayType(kSMALLINT)},
        {"sif", arrayType(kSMALLINT, 3)},
        {"intv", arrayType(kINT)},
        {"intf", arrayType(kINT, 3)},
        {"biv", arrayType(kBIGINT)},
        {"bif", arrayType(kBIGINT, 3)}};
    std::vector<ArrowStorage::ColumnDescription> floating_type_cols{
        {"dv", arrayType(kDOUBLE)},
        {"df", arrayType(kDOUBLE, 3)},
        {"dcv", decimalArrayType(18, 6)},
        {"dcf", decimalArrayType(18, 6, 3)},
        {"fv", arrayType(kFLOAT)},
        {"ff", arrayType(kFLOAT, 3)}};
    std::vector<ArrowStorage::ColumnDescription> date_and_time_type_cols{
        {"dtv", arrayType(kDATE)},
        {"dtf", arrayType(kDATE, 3)},
        {"tv", arrayType(kTIME)},
        {"tf", arrayType(kTIME, 3)},
        {"tsv", arrayType(kTIMESTAMP)},
        {"tsf", arrayType(kTIMESTAMP, 3)}};
    std::vector<ArrowStorage::ColumnDescription> text_type_cols{
        {"tx", dictType()},
        {"txe4", dictType(4)},
        {"txe2", dictType(2)},
        {"txe1", dictType(1)},
        {"txn", SQLTypeInfo(kTEXT)},
        {"txv", arrayType(kTEXT)},
        {"txve", arrayType(kTEXT)}};
    std::vector<ArrowStorage::ColumnDescription> boolean_type_cols{
        {"boolv", arrayType(kBOOLEAN)}, {"boolf", arrayType(kBOOLEAN, 3)}};
    // TODO: Enable date and time when JSON parser for corresponding arrays is available
    auto create_table = [&integer_type_cols,
                         &floating_type_cols,
                         &date_and_time_type_cols,
                         &text_type_cols,
                         &boolean_type_cols](const std::string& tbl_name,
                                             const bool multi_frag,
                                             const bool has_integer_types = true,
                                             const bool has_floting_types = true,
                                             const bool has_date_and_time_types = false,
                                             const bool has_text_types = true,
                                             const bool has_boolean_type = true) {
      std::vector<ArrowStorage::ColumnDescription> cols;
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
      if (has_text_types &&
          (tbl_name.compare("mfarr") == 0 || tbl_name.compare("sfarr") == 0)) {
        // exclude these cols for multi-frag nullable case to avoid not accepted null
        // value issue (i.e., RelAlgExecutor::2833)
        cols.push_back({"txf", arrayType(kTEXT, 3)});
        cols.push_back({"txfe", arrayType(kTEXT, 3)});
      };

      createTable(tbl_name, cols, {multi_frag ? 5ULL : 32000000ULL});
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
    auto insert_values = [](const std::string& tbl_name,
                            const bool allow_null,
                            int start_chunk_idx = 0,
                            int end_chunk_idx = 32,
                            const bool has_integer_types = true,
                            const bool has_floting_types = true,
                            const bool has_date_and_time_types = false,
                            const bool has_text_types = true,
                            const bool has_boolean_type = true) {
      auto gi = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[" << i + 1 << "]";
        } else if (i % 3 == 2) {
          oss << "[" << i << "," << i + 2 << "]";
        } else {
          oss << "[" << i + 1 << "," << i + 2 << "," << i + 3 << "]";
        }
        return oss.str();
      };
      auto gf = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[" << i << "." << i << "]";
        } else if (i % 3 == 2) {
          oss << "[" << i << "." << i << "," << i * 2 << "." << i * 2 << "]";
        } else {
          oss << "[" << i + 1 << "." << i + 1 << "," << i + 2 << "." << i + 2 << ","
              << i + 3 << "." << i + 3 << "]";
        }
        return oss.str();
      };
      auto gdt = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[\"01/01/2021\"]";
        } else if (i % 3 == 2) {
          oss << "[\"01/01/2021\",\"01/02/2021\"]";
        } else {
          oss << "[\"01/01/2021\",\"01/02/2021\",\"01/03/2021\"]";
        }
        return oss.str();
      };
      auto gt = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[\"01:01:01\"]";
        } else if (i % 3 == 2) {
          oss << "[\"01:01:01\",\"01:01:02\"]";
        } else {
          oss << "[\"01:01:01\",\"01:01:02\",\"01:01:03\"]";
        }
        return oss.str();
      };
      auto gts = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[\"01/01/2021 01:01:01\"]";
        } else if (i % 3 == 2) {
          oss << "[\"01/01/2021 01:01:01\",\"01/01/2021 01:01:02\"]";
        } else {
          oss << "[\"01/01/2021 01:01:01\",\"01/01/2021 01:01:02\", \"01/01/2021 "
                 "01:01:03\"]";
        }
        return oss.str();
      };
      auto gtx = [](const int i, bool array) {
        std::ostringstream oss;
        if (array) {
          if (i % 3 == 1) {
            oss << "[\"" << i << "\"]";
          } else if (i % 3 == 2) {
            oss << "[\"" << i << "\",\"" << i << i << "\"]";
          } else {
            oss << "[\"" << i << "\",\"" << i << i << "\",\"" << i << i << i << "\"]";
          }
        } else {
          if (i % 3 == 1) {
            oss << "\"" << i << "\"";
          } else if (i % 3 == 2) {
            oss << "\"" << i << i << "\"";
          } else {
            oss << "\"" << i << i << i << "\"";
          }
        }
        return oss.str();
      };
      auto gbool = [](const int i) {
        std::ostringstream oss;
        auto bool_val = i % 3 == 0 ? "true" : "false";
        if (i % 3 == 1) {
          oss << "[" << bool_val << "]";
        } else if (i % 3 == 2) {
          oss << "[" << bool_val << "," << bool_val << "]";
        } else {
          oss << "[" << bool_val << "," << bool_val << "," << bool_val << "]";
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
      auto insert_json_values = [&tbl_name,
                                 &gi,
                                 &gf,
                                 &gt,
                                 &gdt,
                                 &gts,
                                 &gtx,
                                 &gbool,
                                 &has_integer_types,
                                 &has_floting_types,
                                 &has_date_and_time_types,
                                 &has_text_types,
                                 &has_boolean_type](const int i, int null_row) {
        std::ostringstream oss;
        bool first = true;
        oss << "{";
        if (null_row) {
          auto all_cols = getStorage()->listColumns(
              *getStorage()->getTableInfo(TEST_DB_ID, tbl_name));
          for (auto& col : all_cols) {
            if (col->is_rowid) {
              continue;
            }

            if (first) {
              first = false;
            } else {
              oss << ", ";
            }
            oss << "\"" << col->name << "\": null";
          }
        } else {
          if (has_integer_types) {
            oss << "\"tiv\": " << gi(i % 120) << ", \"tif\": " << gi(3)
                << ", \"siv\": " << gi(i) << ", \"sif\": " << gi(3)
                << ", \"intv\": " << gi(i) << ", \"intf\": " << gi(3)
                << ", \"biv\": " << gi(i) << ", \"bif\": " << gi(3);
            first = false;
          }
          if (has_floting_types) {
            oss << (first ? "" : ", ");
            oss << "\"dv\": " << gf(i) << ", \"df\": " << gf(3) << ", \"dcv\": " << gf(i)
                << ", \"dcf\": " << gf(3) << ", \"fv\": " << gf(i)
                << ", \"ff\": " << gf(3);
            first = false;
          }
          if (has_date_and_time_types) {
            oss << (first ? "" : ", ");
            oss << "\"dtv\": " << gdt(i) << ", \"dtf\": " << gdt(3)
                << ", \"tv\": " << gt(i) << ", \"tv\": " << gt(3)
                << ", \"tsv\": " << gts(i) << ", \"tsf\": " << gts(3);
            first = false;
          }
          if (has_text_types) {
            oss << (first ? "" : ", ");
            oss << "\"tx\": " << gtx(i, false) << ", \"txe4\": " << gtx(i, false)
                << ", \"txe2\": " << gtx(i, false) << ", \"txe1\": " << gtx(i, false)
                << ", \"txn\": " << gtx(i, false) << ", \"txv\": " << gtx(i, true)
                << ", \"txve\": " << gtx(i, true);
            first = false;
          }
          if (has_boolean_type) {
            oss << (first ? "" : ", ");
            oss << "\"boolv\": " << gbool(i) << ", \"boolf\": " << gbool(3);
            first = false;
          }
          if (has_text_types &&
              (tbl_name.compare("mfarr") == 0 || tbl_name.compare("sfarr") == 0)) {
            oss << (first ? "" : ", ");
            oss << "\"txf\": " << gtx(3, true) << ", \"txfe\": " << gtx(3, true);
            first = false;
          };
        }
        oss << "}";

        insertJsonValues(tbl_name, oss.str());
      };
      if (allow_null) {
        int i = 1;
        for (int chunk_idx = start_chunk_idx; chunk_idx < end_chunk_idx; chunk_idx++) {
          std::string str = std::bitset<5>(chunk_idx).to_string();
          for (int row_idx = 0; row_idx < 5; row_idx++, i++) {
            bool null_row = str.at(row_idx) == '1';
            if (null_row) {
              insert_json_values(0, true);
            } else {
              insert_json_values(i, false);
            }
          }
        }
      } else {
        for (auto chunk_idx = start_chunk_idx; chunk_idx < end_chunk_idx; chunk_idx++) {
          for (auto row_idx = 0; row_idx < 5; row_idx++) {
            auto i = (chunk_idx * 5) + row_idx;
            insert_json_values(i, false);
          }
        }
      }
      col_count = 0;
    };

    create_table("sfarr", false);
    create_table("mfarr", true);
    create_table("sfarr_n", false);
    create_table("mfarr_n", true);

    insert_values("sfarr", false);
    insert_values("mfarr", false);
    insert_values("sfarr_n", true);
    insert_values("mfarr_n", true);
  }

  void TearDown() override {
    dropTable("sfarr");
    dropTable("mfarr");
    dropTable("sfarr_n");
    dropTable("mfarr_n");
  }
};

TEST_F(MultiFragArrayJoinTest, Dump) {
  run_simple_agg("SELECT COUNT(1) FROM sfarr r, sfarr s WHERE s.siv[1] = r.siv[1];",
                 ExecutorDeviceType::CPU);
  run_simple_agg("SELECT COUNT(1) FROM mfarr r, mfarr s WHERE s.siv[1] = r.siv[1];",
                 ExecutorDeviceType::CPU);
}

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
          auto single_frag_res = v<int64_t>(run_simple_agg(sq.str(), dt));
          auto res1 = v<int64_t>(run_simple_agg(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    // TODO: Enable when JSON parser for datetime arrays is available
    // run_tests(date_types);
    // run_tests(time_types);
    // run_tests(timestamp_types);
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
          auto single_frag_res = v<int64_t>(run_simple_agg(sq.str(), dt));
          auto res1 = v<int64_t>(run_simple_agg(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    // run_tests(date_types);
    // run_tests(time_types);
    // run_tests(timestamp_types);
    run_tests(non_encoded_text_types);
    run_tests(text_array_types);
    run_tests(bool_types);
  }
}

class MultiFragArrayParallelLinearizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_enable_parallel_linearization = 10;

    std::vector<ArrowStorage::ColumnDescription> cols{
        {"tiv", arrayType(kTINYINT)},
        {"siv", arrayType(kSMALLINT)},
        {"intv", arrayType(kINT)},
        {"biv", arrayType(kBIGINT)},
        {"dv", arrayType(kDOUBLE)},
        {"dcv", decimalArrayType(18, 6)},
        {"fv", arrayType(kFLOAT)},
        // TODO: enable when datetime arrays are supported in JSON parser
        //{"dtv", arrayType(kDATE)},
        //{"tv", arrayType(kTIME)},
        //{"tsv", arrayType(kTIMESTAMP)},
        {"txv", arrayType(kTEXT)},
        {"txve", arrayType(kTEXT)},
        {"boolv", arrayType(kBOOLEAN)}};
    auto create_table = [&cols](const std::string& tbl_name, const bool multi_frag) {
      createTable(tbl_name, cols, {multi_frag ? 5ULL : 32000000ULL});
    };

    create_table("sfarr", false);
    create_table("mfarr", true);
    create_table("sfarr_n", false);
    create_table("mfarr_n", true);
    create_table("sfarr_n2", false);
    create_table("mfarr_n2", true);

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
          oss << "[" << i + 1 << "]";
        } else if (i % 3 == 2) {
          oss << "[" << i << "," << i + 2 << "]";
        } else {
          oss << "[" << i + 1 << "," << i + 2 << "," << i + 3 << "]";
        }
        return oss.str();
      };
      auto gf = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[" << i << "." << i << "]";
        } else if (i % 3 == 2) {
          oss << "[" << i << "." << i << "," << i * 2 << "." << i * 2 << "]";
        } else {
          oss << "[" << i + 1 << "." << i + 1 << "," << i + 2 << "." << i + 2 << ","
              << i + 3 << "." << i + 3 << "]";
        }
        return oss.str();
      };
      auto gdt = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[\"01/01/2021\"]";
        } else if (i % 3 == 2) {
          oss << "[\"01/01/2021\",\"01/02/2021\"]";
        } else {
          oss << "[\"01/01/2021\",\"01/02/2021\",\"01/03/2021\"]";
        }
        return oss.str();
      };
      auto gt = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[\"01:01:01\"]";
        } else if (i % 3 == 2) {
          oss << "[\"01:01:01\",\"01:01:02\"]";
        } else {
          oss << "[\"01:01:01\",\"01:01:02\",\"01:01:03\"]";
        }
        return oss.str();
      };
      auto gts = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[\"01/01/2021 01:01:01\"}";
        } else if (i % 3 == 2) {
          oss << "[\"01/01/2021 01:01:01\",\"01/01/2021 01:01:02\"}";
        } else {
          oss << "[\"01/01/2021 01:01:01\",\"01/01/2021 01:01:02\", \"01/01/2021 "
                 "01:01:03\"}";
        }
        return oss.str();
      };
      auto gtx = [](const int i) {
        std::ostringstream oss;
        if (i % 3 == 1) {
          oss << "[\"" << i << "\"]";
        } else if (i % 3 == 2) {
          oss << "[\"" << i << "\",\"" << i << i << "\"]";
        } else {
          oss << "[\"" << i << "\",\"" << i << i << "\",\"" << i << i << i << "\"]";
        }
        return oss.str();
      };
      auto gbool = [](const int i) {
        std::ostringstream oss;
        auto bool_val = i % 3 == 0 ? "true" : "false";
        if (i % 3 == 1) {
          oss << "[" << bool_val << "]";
        } else if (i % 3 == 2) {
          oss << "[" << bool_val << "," << bool_val << "]";
        } else {
          oss << "[" << bool_val << "," << bool_val << "," << bool_val << "]";
        }
        return oss.str();
      };
      auto insertValues = [&tbl_name, &gi, &gf, &gt, &gdt, &gts, &gtx, &gbool](
                              const int i, int null_row) {
        std::ostringstream oss;
        if (null_row) {
          oss << R"___({"tiv": null, "siv": null, "intv": null, "biv": null, "dv": null, "dcv": null, "fv": null, "txv": null, "txve": null, "boolv": null})___";
        } else {
          auto i_r = gi(i);
          auto i_f = gf(i);
          auto i_tx = gtx(i);
          oss << "{\"tiv\": " << gi(i % 120) << ", \"siv\": " << i_r
              << ", \"intv\": " << i_r << ", \"biv\": " << i_r << ", \"dv\": " << i_f
              << ", \"dcv\": " << i_f << ", \"fv\": "
              << i_f
              // TODO: enable when datetime arrays are supported in JSON parser
              // << ", \"dtv\": " << gdt(i) << ", \"tv\": " << gt(i) << ", \"tsv\": " <<
              // gts(i)
              << ", \"txv\": " << i_tx << ", \"txve\": " << i_tx
              << ", \"boolv\": " << gbool(i) << "}" << std::endl;
        }
        insertJsonValues(tbl_name, oss.str());
      };
      if (allow_null) {
        for (int i = 0; i < num_tuples; i++) {
          if ((first_frag_row_null && (i % frag_size) == 0) || (i % 17 == 5)) {
            insertValues(0, true);
          } else {
            insertValues(i, false);
          }
        }
      } else {
        for (int i = 0; i < num_tuples; i++) {
          insertValues(i, false);
        }
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
    dropTable("sfarr");
    dropTable("mfarr");
    dropTable("sfarr_n");
    dropTable("mfarr_n");
    dropTable("sfarr_n2");
    dropTable("mfarr_n2");
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
          auto single_frag_res = v<int64_t>(run_simple_agg(sq.str(), dt));
          auto res1 = v<int64_t>(run_simple_agg(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    // TODO: enable when datetime arrays are supported in JSON parser
    // run_tests(date_types);
    // run_tests(time_types);
    // run_tests(timestamp_types);
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
          auto single_frag_res = v<int64_t>(run_simple_agg(sq.str(), dt));
          auto res1 = v<int64_t>(run_simple_agg(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    // TODO: enable when datetime arrays are supported in JSON parser
    // run_tests(date_types);
    // run_tests(time_types);
    // run_tests(timestamp_types);
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
          auto single_frag_res = v<int64_t>(run_simple_agg(sq.str(), dt));
          auto res1 = v<int64_t>(run_simple_agg(q1.str(), dt));
          ASSERT_EQ(single_frag_res, res1) << q1.str();
        }
      }
    };
    run_tests(integer_types);
    run_tests(float_types);
    // TODO: enable when datetime arrays are supported in JSON parser
    // run_tests(date_types);
    // run_tests(time_types);
    // run_tests(timestamp_types);
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

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  init();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  reset();
  return err;
}
