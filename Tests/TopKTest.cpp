
/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../ImportExport/Importer.h"
#include "../Parser/parser.h"
#include "../QueryEngine/ArrowResultSet.h"
#include "../QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "../QueryEngine/Execute.h"
#include "../QueryRunner/QueryRunner.h"
#include "../Shared/DateTimeParser.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <sstream>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#ifdef _WIN32
#define timegm _mkgmtime
#endif

using namespace std;

using QR = QueryRunner::QueryRunner;
namespace {

inline void run_ddl_statement(const std::string query_str) {
  QR::get()->runDDLStatement(query_str);
}

std::shared_ptr<ResultSet> run_multiple_agg(const string& query_str,
                                            const ExecutorDeviceType device_type,
                                            const bool allow_loop_joins) {
  return QR::get()->runSQL(query_str, device_type, true, allow_loop_joins);
}

std::shared_ptr<ResultSet> run_multiple_agg(const string& query_str,
                                            const ExecutorDeviceType device_type) {
  return run_multiple_agg(query_str, device_type, true);
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !QR::get()->gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

bool approx_eq(const double v, const double target, const double eps = 0.01) {
  const auto v_u64 = *reinterpret_cast<const uint64_t*>(may_alias_ptr(&v));
  const auto target_u64 = *reinterpret_cast<const uint64_t*>(may_alias_ptr(&target));
  return v_u64 == target_u64 || (target - eps < v && v < target + eps);
}

class SQLiteComparator {
 public:
  SQLiteComparator() : connector_("sqliteTestDB", "") {}

  void query(const std::string& query_string) { connector_.query(query_string); }

  void compare(const std::string& query_string, const ExecutorDeviceType device_type) {
    const auto omnisci_results = run_multiple_agg(query_string, device_type);
    compare_impl(omnisci_results.get(), query_string, device_type, false);
  }

  void compare_arrow_output(const std::string& query_string,
                            const std::string& sqlite_query_string,
                            const ExecutorDeviceType device_type) {
    const auto results = QR::get()->runSQL(query_string, device_type, true, true);
    const auto arrow_omnisci_results = result_set_arrow_loopback(nullptr, results);
    compare_impl(
        arrow_omnisci_results.get(), sqlite_query_string, device_type, false, true);
  }

  void compare(const std::string& query_string,
               const std::string& sqlite_query_string,
               const ExecutorDeviceType device_type) {
    const auto omnisci_results = run_multiple_agg(query_string, device_type);
    compare_impl(omnisci_results.get(), sqlite_query_string, device_type, false);
  }

  // added to deal with time shift for now testing
  void compare_timstamp_approx(const std::string& query_string,
                               const ExecutorDeviceType device_type) {
    const auto omnisci_results = run_multiple_agg(query_string, device_type);
    compare_impl(omnisci_results.get(), query_string, device_type, true);
  }

 private:
  template <class RESULT_SET>
  void compare_impl(const RESULT_SET* omnisci_results,
                    const std::string& sqlite_query_string,
                    const ExecutorDeviceType device_type,
                    const bool timestamp_approx,
                    const bool is_arrow = false) {
    auto const errmsg = ExecutorDeviceType::CPU == device_type
                            ? "CPU: " + sqlite_query_string
                            : "GPU: " + sqlite_query_string;
    connector_.query(sqlite_query_string);
    ASSERT_EQ(connector_.getNumRows(), omnisci_results->rowCount()) << errmsg;
    const int num_rows{static_cast<int>(connector_.getNumRows())};
    if (omnisci_results->definitelyHasNoRows()) {
      ASSERT_EQ(0, num_rows) << errmsg;
      return;
    }
    if (!num_rows) {
      return;
    }
    CHECK_EQ(connector_.getNumCols(), omnisci_results->colCount()) << errmsg;
    const int num_cols{static_cast<int>(connector_.getNumCols())};
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const auto crt_row = omnisci_results->getNextRow(true, true);
      CHECK(!crt_row.empty()) << errmsg;
      CHECK_EQ(static_cast<size_t>(num_cols), crt_row.size()) << errmsg;
      for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        const auto ref_col_type = connector_.columnTypes[col_idx];
        const auto omnisci_variant = crt_row[col_idx];
        const auto scalar_omnisci_variant =
            boost::get<ScalarTargetValue>(&omnisci_variant);
        CHECK(scalar_omnisci_variant) << errmsg;
        auto omnisci_ti = omnisci_results->getColType(col_idx);
        const auto omnisci_type = omnisci_ti.get_type();
        checkTypeConsistency(ref_col_type, omnisci_ti);
        const bool ref_is_null = connector_.isNull(row_idx, col_idx);
        switch (omnisci_type) {
          case kTINYINT:
          case kSMALLINT:
          case kINT:
          case kBIGINT: {
            const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
            ASSERT_NE(nullptr, omnisci_as_int_p);
            const auto omnisci_val = *omnisci_as_int_p;
            if (ref_is_null) {
              ASSERT_EQ(inline_int_null_val(omnisci_ti), omnisci_val) << errmsg;
            } else {
              const auto ref_val = connector_.getData<int64_t>(row_idx, col_idx);
              ASSERT_EQ(ref_val, omnisci_val) << errmsg;
            }
            break;
          }
          case kTEXT:
          case kCHAR:
          case kVARCHAR: {
            const auto omnisci_as_str_p =
                boost::get<NullableString>(scalar_omnisci_variant);
            ASSERT_NE(nullptr, omnisci_as_str_p) << errmsg;
            const auto omnisci_str_notnull = boost::get<std::string>(omnisci_as_str_p);
            const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
            if (omnisci_str_notnull) {
              const auto omnisci_val = *omnisci_str_notnull;
              ASSERT_EQ(ref_val, omnisci_val) << errmsg;
            } else {
              // not null but no data, so val is empty string
              const auto omnisci_val = "";
              ASSERT_EQ(ref_val, omnisci_val) << errmsg;
            }
            break;
          }
          case kNUMERIC:
          case kDECIMAL:
          case kDOUBLE: {
            const auto omnisci_as_double_p = boost::get<double>(scalar_omnisci_variant);
            ASSERT_NE(nullptr, omnisci_as_double_p) << errmsg;
            const auto omnisci_val = *omnisci_as_double_p;
            if (ref_is_null) {
              ASSERT_EQ(inline_fp_null_val(SQLTypeInfo(kDOUBLE, false)), omnisci_val)
                  << errmsg;
            } else {
              const auto ref_val = connector_.getData<double>(row_idx, col_idx);
              ASSERT_TRUE(approx_eq(ref_val, omnisci_val)) << errmsg;
            }
            break;
          }
          case kFLOAT: {
            const auto omnisci_as_float_p = boost::get<float>(scalar_omnisci_variant);
            ASSERT_NE(nullptr, omnisci_as_float_p) << errmsg;
            const auto omnisci_val = *omnisci_as_float_p;
            if (ref_is_null) {
              ASSERT_EQ(inline_fp_null_val(SQLTypeInfo(kFLOAT, false)), omnisci_val)
                  << errmsg;
            } else {
              const auto ref_val = connector_.getData<float>(row_idx, col_idx);
              ASSERT_TRUE(approx_eq(ref_val, omnisci_val)) << errmsg;
            }
            break;
          }
          case kTIMESTAMP:
          case kDATE: {
            const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
            const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
            CHECK(omnisci_as_int_p);
            const auto omnisci_val = *omnisci_as_int_p;
            time_t nsec = 0;
            const int dimen = omnisci_ti.get_dimension();

            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(omnisci_ti), omnisci_val) << errmsg;
            } else {
              auto temp_ref_value = dateTimeParseOptional<kTIMESTAMP>(ref_val, dimen);
              if (!temp_ref_value) {
                temp_ref_value = dateTimeParseOptional<kDATE>(ref_val, dimen);
              }
              CHECK(temp_ref_value) << ref_val;
              nsec = temp_ref_value.value();
              if (timestamp_approx) {
                // approximate result give 10 second lee way
                ASSERT_NEAR(*omnisci_as_int_p, nsec, dimen > 0 ? 10 * pow(10, dimen) : 10)
                    << errmsg;
              } else {
                struct tm tm_struct {
                  0
                };
#ifdef _WIN32
                auto ret_code = gmtime_s(&tm_struct, &nsec);
                CHECK(ret_code == 0) << "Error code returned " << ret_code;
#else
                gmtime_r(&nsec, &tm_struct);
#endif
                if (is_arrow && omnisci_type == kDATE) {
                  ASSERT_EQ(*omnisci_as_int_p, timegm(&tm_struct) * kMilliSecsPerSec)
                      << errmsg;
                } else {
                  ASSERT_EQ(*omnisci_as_int_p, dimen > 0 ? nsec : timegm(&tm_struct))
                      << errmsg;
                }
              }
            }
            break;
          }
          case kBOOLEAN: {
            const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
            CHECK(omnisci_as_int_p) << errmsg;
            const auto omnisci_val = *omnisci_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(omnisci_ti), omnisci_val) << errmsg;
            } else {
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              if (ref_val == "t") {
                ASSERT_EQ(1, *omnisci_as_int_p) << errmsg;
              } else {
                CHECK_EQ("f", ref_val) << errmsg;
                ASSERT_EQ(0, *omnisci_as_int_p) << errmsg;
              }
            }
            break;
          }
          case kTIME: {
            const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
            CHECK(omnisci_as_int_p) << errmsg;
            const auto omnisci_val = *omnisci_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(omnisci_ti), omnisci_val) << errmsg;
            } else {
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              std::vector<std::string> time_tokens;
              boost::split(time_tokens, ref_val, boost::is_any_of(":"));
              ASSERT_EQ(size_t(3), time_tokens.size()) << errmsg;
              ASSERT_EQ(boost::lexical_cast<int64_t>(time_tokens[0]) * 3600 +
                            boost::lexical_cast<int64_t>(time_tokens[1]) * 60 +
                            boost::lexical_cast<int64_t>(time_tokens[2]),
                        *omnisci_as_int_p)
                  << errmsg;
            }
            break;
          }
          default:
            CHECK(false) << errmsg;
        }
      }
    }
  }

 private:
  static void checkTypeConsistency(const int ref_col_type,
                                   const SQLTypeInfo& omnisci_ti) {
    if (ref_col_type == SQLITE_NULL) {
      // TODO(alex): re-enable the check that omnisci_ti is nullable,
      //             got invalidated because of outer joins
      return;
    }
    if (omnisci_ti.is_integer()) {
      CHECK_EQ(SQLITE_INTEGER, ref_col_type);
    } else if (omnisci_ti.is_fp() || omnisci_ti.is_decimal()) {
      CHECK(ref_col_type == SQLITE_FLOAT || ref_col_type == SQLITE_INTEGER);
    } else {
      CHECK_EQ(SQLITE_TEXT, ref_col_type);
    }
  }

  SqliteConnector connector_;
};

SQLiteComparator g_sqlite_comparator;

void c(const std::string& query_string,
       const std::string& sqlite_query_string,
       const ExecutorDeviceType device_type) {
  g_sqlite_comparator.compare(query_string, sqlite_query_string, device_type);
}
}  // namespace

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

namespace {

int create_and_populate_tables() {
  try {
    // Drop old table
    const std::string drop_old_test{"DROP TABLE IF EXISTS tdata;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);

    // Create a table
    const std::string create_test_table{
        "CREATE TABLE tdata (id SMALLINT, b BOOLEAN, i INT, bi BIGINT, n DECIMAL(10, 2), "
        "f FLOAT, t TEXT, tt TIME, d "
        "DATE, ts TIMESTAMP, vc VARCHAR(15));"};
    run_ddl_statement(create_test_table);
    g_sqlite_comparator.query(create_test_table);

    // Insert data into the table
    std::vector<std::string> data_col_value_list;
    data_col_value_list.emplace_back(
        "1, 't', 23, 2349923, 111.1, 1.1, 'SFO', '15:13:14', '1999-09-09', '2014-12-13 "
        "22:23:15', 'paris'");
    data_col_value_list.emplace_back(
        "2, 'f', null, -973273, 7263.11, 87.1, null, '20:05:00', '2017-12-12', "
        "'2017-12-12 20:05:00', 'toronto'");
    data_col_value_list.emplace_back(
        "3, 'f', 702, 87395, 333.5, null, 'YVR', '11:11:11', '2010-01-01', '2010-01-02 "
        "04:11:45', 'vancouver'");
    data_col_value_list.emplace_back(
        "4, null, 864, 100001, null, 9.9, 'SJC', null, '2015-05-05', '2010-05-05 "
        "05:15:55', 'london'");
    data_col_value_list.emplace_back(
        "5, 'f', 333, 112233, 99.9, 9.9, 'ABQ', '22:22:22', '2015-05-05', '2010-05-05 "
        "05:15:55', 'new york'");
    data_col_value_list.emplace_back(
        "6, 't', -3, 18, 765.8, 2.2, 'YYZ', '00:00:01', null, '2009-01-08 12:13:14', "
        "null");
    data_col_value_list.emplace_back(
        "7, 'f', -9873, 3789, 789.3, 4.7, 'DCA', '11:22:33', '2001-02-03', '2005-04-03 "
        "15:16:17', 'rio de janerio'");
    data_col_value_list.emplace_back(
        "8, 't', 12, 4321, 83.9, 1.2, 'DXB', '21:20:10', null, '2007-12-01 23:22:21', "
        "'dubai'");
    data_col_value_list.emplace_back(
        "9, 't', 48, null, 83.9, 1.2, 'BWI', '09:08:07', '2001-09-11', null, "
        "'washington'");
    data_col_value_list.emplace_back(
        "10, 'f', 99, 777, 77.7, 7.7, 'LLBG', '07:07:07', '2017-07-07', '2017-07-07 "
        "07:07:07', 'Tel Aviv'");

    for (const auto& data_col_values : data_col_value_list) {
      std::string insert_query = "INSERT INTO tdata VALUES(" + data_col_values + ");";
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
    }
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test'";
    return -EEXIST;
  }

  return 0;
}

void drop_tables() {
  const std::string drop_test_inner{"DROP TABLE tdata;"};
  run_ddl_statement(drop_test_inner);
  g_sqlite_comparator.query(drop_test_inner);
}
}  // namespace

TEST(Select, TopK_LIMIT_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i NULLS FIRST LIMIT 5;",
      "SELECT i FROM tdata ORDER BY i ASC LIMIT 5;",
      dt);
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 5;",
      "SELECT b FROM tdata ORDER BY b LIMIT 5;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 5;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 5;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 5;",
      "SELECT n FROM tdata ORDER BY n LIMIT 5;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 5;",
      "SELECT f FROM tdata ORDER BY f LIMIT 5;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 5;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 5;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 5;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 5;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 5;",
      "SELECT d FROM tdata ORDER BY d LIMIT 5;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 5;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 5;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 5;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 5;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 5;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 5;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 5;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 5;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 5;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 5;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 5;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 5;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 5;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 5;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_GreaterThan_TotalOfDataRows_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 11;",
      "SELECT b FROM tdata ORDER BY b LIMIT 11;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 11;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 11;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 11;",
      "SELECT n FROM tdata ORDER BY n LIMIT 11;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 11;",
      "SELECT f FROM tdata ORDER BY f LIMIT 11;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 11;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 11;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 11;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 11;",
      "SELECT d FROM tdata ORDER BY d LIMIT 11;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_GreaterThan_TotalOfDataRows_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 11;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 11;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 11;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 11;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 11;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 11;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 11;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 11;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 11;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 11;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 11;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 11;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 11;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 11;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 11;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_OFFSET_TopHalf_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT i FROM tdata ORDER BY i LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT b FROM tdata ORDER BY b LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT n FROM tdata ORDER BY n LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT f FROM tdata ORDER BY f LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 5 OFFSET 0;",
      "SELECT d FROM tdata ORDER BY d LIMIT 5 OFFSET 0;",
      dt);
  }
}

TEST(Select, TopK_LIMIT_OFFSET_TopHalf_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 5 OFFSET 0;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5 OFFSET 0;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 5 OFFSET 0;xxx",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_BottomHalf_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT i FROM tdata ORDER BY i LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT b FROM tdata ORDER BY b LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT n FROM tdata ORDER BY n LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT f FROM tdata ORDER BY f LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 5 OFFSET 5;",
      "SELECT d FROM tdata ORDER BY d LIMIT 5 OFFSET 5;",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_BottomHalf_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 5 OFFSET 5;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5 OFFSET 5;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 5 OFFSET 5;",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_GreaterThan_TotalOfDataRows_AscendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT i FROM tdata ORDER BY i LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT b FROM tdata ORDER BY b NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT b FROM tdata ORDER BY b LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT bi FROM tdata ORDER BY bi LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT n FROM tdata ORDER BY n NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT n FROM tdata ORDER BY n LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT f FROM tdata ORDER BY f NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT f FROM tdata ORDER BY f LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT tt FROM tdata ORDER BY tt LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT ts FROM tdata ORDER BY ts LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY d NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY d LIMIT 5 OFFSET 11;",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_GreaterThan_TotalOfDataRows_DescendSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT i FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT b FROM tdata ORDER BY b DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT b FROM tdata ORDER BY b DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT bi FROM tdata ORDER BY bi DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT bi FROM tdata ORDER BY bi DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT n FROM tdata ORDER BY n DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT n FROM tdata ORDER BY n DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT f FROM tdata ORDER BY f DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT f FROM tdata ORDER BY f DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT tt FROM tdata ORDER BY tt DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT tt FROM tdata ORDER BY tt DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT ts FROM tdata ORDER BY ts DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT ts FROM tdata ORDER BY ts DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY d DESC LIMIT 5 OFFSET 11;",
      dt);
  }
}

TEST(Select, DISABLED_TopK_LIMIT_OFFSET_DifferentOrders) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT i,d FROM tdata ORDER BY d DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY d DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT i,d FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT i,d FROM tdata ORDER BY i,d DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY i,d DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY i DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY i DESC LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY i,d DESC NULLS LAST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY i,d DESC LIMIT 5 OFFSET 11;",
      dt);

    c("SELECT i,d FROM tdata ORDER BY d NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY d LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT i,d FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY i LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT i,d FROM tdata ORDER BY i,d NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT i,d FROM tdata ORDER BY i,d LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY i NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY i LIMIT 5 OFFSET 11;",
      dt);
    c("SELECT d FROM tdata ORDER BY i,d NULLS FIRST LIMIT 5 OFFSET 11;",
      "SELECT d FROM tdata ORDER BY i,d LIMIT 5 OFFSET 11;",
      dt);
  }
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  QR::init(BASE_PATH);

  int err{0};
  err = create_and_populate_tables();
  if (err) {
    return err;
  }

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  drop_tables();
  QR::reset();
  return err;
}
