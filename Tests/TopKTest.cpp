/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../Shared/DateTimeParser.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "SqlTestFixture.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <sstream>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std;

namespace {

inline void run_ddl_statement(const std::string& query_str) {
  SqlTestFixture::sql(query_str);
}

std::shared_ptr<ThinResultSet> run_multiple_agg(const string& query_str,
                                                const TExecuteMode::type device_type) {
  return SqlTestFixture::run_multiple_agg(query_str, device_type);
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

bool skip_tests(const TExecuteMode::type device_type) {
  return device_type == TExecuteMode::GPU && !SqlTestFixture::gpusPresent();
}

bool approx_eq(const double v, const double target, const double eps = 0.01) {
  uint64_t v_u64, target_u64;
  std::memcpy(&v_u64, &v, sizeof(v));
  std::memcpy(&target_u64, &target, sizeof(target));
  return v_u64 == target_u64 || (target - eps < v && v < target + eps);
}

// The value ThinResultSet reports for a NULL of the given integer-backed type.
int64_t int_null_val(const TDatumType::type type) {
  switch (type) {
    case TDatumType::TINYINT:
    case TDatumType::BOOL:
      return ThinResultSet::kNullTinyint;
    case TDatumType::SMALLINT:
      return ThinResultSet::kNullSmallint;
    case TDatumType::INT:
      return ThinResultSet::kNullInt;
    default:  // BIGINT, TIME, TIMESTAMP, DATE
      return ThinResultSet::kNullBigint;
  }
}

class SQLiteComparator {
 public:
  SQLiteComparator() : connector_("sqliteTestDB", "") {}

  void query(const std::string& query_string) { connector_.query(query_string); }

  void compare(const std::string& query_string, const TExecuteMode::type device_type) {
    const auto omnisci_results = run_multiple_agg(query_string, device_type);
    compare_impl(omnisci_results.get(), query_string, device_type, false);
  }

  void compare(const std::string& query_string,
               const std::string& sqlite_query_string,
               const TExecuteMode::type device_type) {
    const auto omnisci_results = run_multiple_agg(query_string, device_type);
    compare_impl(omnisci_results.get(), sqlite_query_string, device_type, false);
  }

  // added to deal with time shift for now testing
  void compare_timstamp_approx(const std::string& query_string,
                               const TExecuteMode::type device_type) {
    const auto omnisci_results = run_multiple_agg(query_string, device_type);
    compare_impl(omnisci_results.get(), query_string, device_type, true);
  }

 private:
  void compare_impl(ThinResultSet* omnisci_results,
                    const std::string& sqlite_query_string,
                    const TExecuteMode::type device_type,
                    const bool timestamp_approx) {
    auto const errmsg = TExecuteMode::CPU == device_type
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
        const auto& omnisci_ti = omnisci_results->getColType(col_idx).col_type;
        const auto omnisci_type = omnisci_ti.type;
        checkTypeConsistency(ref_col_type, omnisci_type);
        const bool ref_is_null = connector_.isNull(row_idx, col_idx);
        switch (omnisci_type) {
          case TDatumType::TINYINT:
          case TDatumType::SMALLINT:
          case TDatumType::INT:
          case TDatumType::BIGINT: {
            const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
            ASSERT_NE(nullptr, omnisci_as_int_p);
            const auto omnisci_val = *omnisci_as_int_p;
            if (ref_is_null) {
              ASSERT_EQ(int_null_val(omnisci_type), omnisci_val) << errmsg;
            } else {
              const auto ref_val = connector_.getData<int64_t>(row_idx, col_idx);
              ASSERT_EQ(ref_val, omnisci_val) << errmsg;
            }
            break;
          }
          case TDatumType::STR: {
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
          case TDatumType::DECIMAL:
          case TDatumType::DOUBLE: {
            const auto omnisci_as_double_p = boost::get<double>(scalar_omnisci_variant);
            ASSERT_NE(nullptr, omnisci_as_double_p) << errmsg;
            const auto omnisci_val = *omnisci_as_double_p;
            if (ref_is_null) {
              ASSERT_EQ(ThinResultSet::kNullDouble, omnisci_val) << errmsg;
            } else {
              const auto ref_val = connector_.getData<double>(row_idx, col_idx);
              ASSERT_TRUE(approx_eq(ref_val, omnisci_val)) << errmsg;
            }
            break;
          }
          case TDatumType::FLOAT: {
            const auto omnisci_as_float_p = boost::get<float>(scalar_omnisci_variant);
            ASSERT_NE(nullptr, omnisci_as_float_p) << errmsg;
            const auto omnisci_val = *omnisci_as_float_p;
            if (ref_is_null) {
              ASSERT_EQ(ThinResultSet::kNullFloat, omnisci_val) << errmsg;
            } else {
              const auto ref_val = connector_.getData<float>(row_idx, col_idx);
              ASSERT_TRUE(approx_eq(ref_val, omnisci_val)) << errmsg;
            }
            break;
          }
          case TDatumType::TIMESTAMP:
          case TDatumType::DATE: {
            const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
            const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
            CHECK(omnisci_as_int_p);
            const auto omnisci_val = *omnisci_as_int_p;
            time_t nsec = 0;
            const int dimen = omnisci_ti.precision;

            if (ref_is_null) {
              CHECK_EQ(int_null_val(omnisci_type), omnisci_val) << errmsg;
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
                gmtime_r(&nsec, &tm_struct);
                ASSERT_EQ(*omnisci_as_int_p, dimen > 0 ? nsec : timegm(&tm_struct))
                    << errmsg;
              }
            }
            break;
          }
          case TDatumType::BOOL: {
            const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
            CHECK(omnisci_as_int_p) << errmsg;
            const auto omnisci_val = *omnisci_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(int_null_val(omnisci_type), omnisci_val) << errmsg;
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
          case TDatumType::TIME: {
            const auto omnisci_as_int_p = boost::get<int64_t>(scalar_omnisci_variant);
            CHECK(omnisci_as_int_p) << errmsg;
            const auto omnisci_val = *omnisci_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(int_null_val(omnisci_type), omnisci_val) << errmsg;
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

  static void checkTypeConsistency(const int ref_col_type,
                                   const TDatumType::type omnisci_type) {
    if (ref_col_type == SQLITE_NULL) {
      // TODO(alex): re-enable the check that omnisci_ti is nullable,
      //             got invalidated because of outer joins
      return;
    }
    if (omnisci_type == TDatumType::TINYINT || omnisci_type == TDatumType::SMALLINT ||
        omnisci_type == TDatumType::INT || omnisci_type == TDatumType::BIGINT) {
      CHECK_EQ(SQLITE_INTEGER, ref_col_type);
    } else if (omnisci_type == TDatumType::FLOAT || omnisci_type == TDatumType::DOUBLE ||
               omnisci_type == TDatumType::DECIMAL) {
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
       const TExecuteMode::type device_type) {
  g_sqlite_comparator.compare(query_string, sqlite_query_string, device_type);
}
}  // namespace

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == TExecuteMode::GPU);                          \
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
      run_multiple_agg(insert_query, TExecuteMode::CPU);
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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
  for (auto dt : {TExecuteMode::CPU, TExecuteMode::GPU}) {
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

  int err{0};
  try {
    SqlTestFixture::initConnection(argc, argv);
    err = create_and_populate_tables();
    if (err) {
      return err;
    }
    err = RUN_ALL_TESTS();
    drop_tables();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = 1;
  }
  return err;
}
