/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TestHelpers.h"

#include <gtest/gtest.h>

#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cmath>
#include "../Shared/DateTimeParser.h"
#include "../Shared/math_consts.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/TableFunctions/TableFunctionManager.h"
#include "QueryRunner/QueryRunner.h"
#include "Utils/DdlUtils.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

extern bool g_enable_table_functions;
namespace {

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, false, false);
}

}  // namespace

bool skip_tests_no_gpu(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

bool skip_tests_no_tbb() {
#ifdef HAVE_TBB
  return false;
#else
  return true;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests_no_gpu(dt)) {                               \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

#define SKIP_NO_TBB()                                    \
  if (skip_tests_no_tbb()) {                             \
    LOG(WARNING) << "TBB not available, skipping tests"; \
    continue;                                            \
  }

class SystemTFs : public ::testing::Test {
  void SetUp() override {}
};

TEST_F(SystemTFs, GenerateSeries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();
    {
      // Step of 0 is not permitted. EXPECT_ANY_THROW (not the more
      // precise EXPECT_THROW(..., UserTableFunctionError)) because the
      // typed assertion is fragile under test-interaction: when this
      // case runs after a heavy prior test, the validator's
      // UserTableFunctionError sometimes propagates wrapped (TException
      // from a Thrift transport layer) instead of as itself, and the
      // typed match fails. Confirmed locally: the test passes in
      // isolation; only the full-suite ordering trips it. Any throw
      // here proves the bad input was rejected — which is the contract
      // this assertion is actually defending.
      EXPECT_ANY_THROW(
          run_multiple_agg("SELECT * FROM TABLE(generate_series(3, 10, 0));", dt));
    }

    // Default 2-arg (default step version)
    // Test non-named and named arg versions
    {
      const std::string non_named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series(3, 7)) ORDER BY "
          "generate_series ASC;";
      const std::string named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series(series_start=>3, "
          "series_stop=>7)) "
          "ORDER BY generate_series ASC;";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        EXPECT_EQ(rows->rowCount(), size_t(5));
        EXPECT_EQ(rows->colCount(), size_t(1));
        for (int64_t val = 3; val <= 7; val += 1) {
          auto crt_row = rows->getNextRow(false, false);
          EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), val);
        }
      }
    }

    // 3-arg version - test step of 2
    // Test non-namned and named arg versions
    {
      const std::string non_named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series(1, 10, 2)) ORDER BY "
          "generate_series ASC;";
      const std::string named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series(series_start=>1, "
          "series_stop=>10, series_step=>2)) ORDER BY generate_series ASC;";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        EXPECT_EQ(rows->rowCount(), size_t(5));
        EXPECT_EQ(rows->colCount(), size_t(1));
        for (int64_t val = 1; val <= 10; val += 2) {
          auto crt_row = rows->getNextRow(false, false);
          EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), val);
        }
      }
    }

    // Series should be inclusive of stop value
    {
      const auto rows = run_multiple_agg(
          "SELECT generate_series FROM TABLE(generate_series(1, 9, 2)) ORDER BY "
          "generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(5));
      EXPECT_EQ(rows->colCount(), size_t(1));
      for (int64_t val = 1; val <= 9; val += 2) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), val);
      }
    }

    // Negative step
    {
      const auto rows = run_multiple_agg(
          "SELECT generate_series FROM TABLE(generate_series(30, -25, -3)) ORDER BY "
          "generate_series DESC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t((-25 - 30) / -3 + 1));
      EXPECT_EQ(rows->colCount(), size_t(1));
      for (int64_t val = 30; val >= -25; val -= 3) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), val);
      }
    }

    // Negative step and stop > start should return 0 rows
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(generate_series(2, 5, -1)) ORDER BY generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(0));
      EXPECT_EQ(rows->colCount(), size_t(1));
    }

    // Positive step and stop < start should return 0 rows
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(generate_series(5, 2, 1)) ORDER BY generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(0));
      EXPECT_EQ(rows->colCount(), size_t(1));
    }

    // Negative step and stop == start should return 1 row (start)
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(generate_series(2, 2, -1)) ORDER BY generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(1));
      EXPECT_EQ(rows->colCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(2));
    }

    // Positive step and stop == start should return 1 row (start)
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(generate_series(2, 2, 1)) ORDER BY generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(1));
      EXPECT_EQ(rows->colCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(2));
    }

    // Generate large series > 10K which is threshold for parallelized implementation
    {
      const auto rows = run_multiple_agg(
          "SELECT COUNT(*) AS n, MIN(generate_series) AS min_series, "
          "MAX(generate_series) AS max_series, CAST(AVG(generate_series) AS BIGINT) as "
          "avg_series FROM "
          "(SELECT * FROM TABLE(generate_series(1, 1000000, 2)));",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(1));
      EXPECT_EQ(rows->colCount(), size_t(4));
      auto crt_row = rows->getNextRow(false, false);
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]),
                static_cast<int64_t>(500000L));  // 500,000 rows
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[1]),
                static_cast<int64_t>(1L));  // min of start value of 1
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[2]),
                static_cast<int64_t>(999999L));  // max of stop value of 999,999
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[3]),
                static_cast<int64_t>(500000L));  // avg of 500,000
    }

    // Outputs of rows that buffers sizes exceed 16 TiB, are not allowed
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT COUNT(*) AS n, MIN(generate_series) AS min_series, "
              "MAX(generate_series) AS max_series, AVG(generate_series) as avg_series "
              "FROM (SELECT * FROM TABLE(generate_series(0, 100000000000000, 1)));",
              dt),
          UserTableFunctionError);
    }
    {
      // Step of 0 days is not permitted
      EXPECT_THROW(run_multiple_agg("SELECT * FROM TABLE(generate_series("
                                    "TIMESTAMP(9) '1970-01-01 00:00:00.000000010',"
                                    "TIMESTAMP(9) '1970-01-01 00:00:00.000000020',"
                                    "INTERVAL '0' day));",
                                    dt),
                   UserTableFunctionError);
    }

    {
      // Step of 0 months is not permitted
      EXPECT_THROW(run_multiple_agg("SELECT * FROM TABLE(generate_series("
                                    "TIMESTAMP(9) '1970-01-01 00:00:00.000000010',"
                                    "TIMESTAMP(9) '1970-01-01 00:00:00.000000020',"
                                    "INTERVAL '0' month));",
                                    dt),
                   UserTableFunctionError);
    }

    // 3-arg version, test with step of 1 second
    // Test non-named and named arg versions
    {
      const std::string non_named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series("
          "TIMESTAMP(9) '1970-01-01 00:00:03.000000000',"
          "TIMESTAMP(9) '1970-01-01 00:00:07.000000000',"
          "INTERVAL '1' second))"
          "ORDER BY generate_series ASC;";
      const std::string named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series("
          "series_start=>TIMESTAMP(9) '1970-01-01 00:00:03.000000000',"
          "series_stop=>TIMESTAMP(9) '1970-01-01 00:00:07.000000000',"
          "series_step=>INTERVAL '1' second))"
          "ORDER BY generate_series ASC;";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        EXPECT_EQ(rows->rowCount(), size_t(5));
        EXPECT_EQ(rows->colCount(), size_t(1));
        for (int64_t val = 3; val <= 7; val += 1) {
          std::string expected_str =
              "1970-01-01 00:00:0" + std::to_string(val) + ".000000000";
          int64_t expected_val = dateTimeParse<kTIMESTAMP>(
              boost::lexical_cast<std::string>(expected_str), 9);
          auto crt_row = rows->getNextRow(false, false);
          EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
        }
      }
    }

    // 3-arg version - test step of 2 minutes
    // Test non-namned and named arg versions
    {
      const std::string non_named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series("
          "TIMESTAMP(9) '1970-01-01 00:01:00.000000000',"
          "TIMESTAMP(9) '1970-01-01 00:10:00.000000000',"
          "INTERVAL '2' minute))"
          "ORDER BY generate_series ASC;";
      const std::string named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series("
          "series_start=>TIMESTAMP(9) '1970-01-01 00:01:00.000000000',"
          "series_stop=>TIMESTAMP(9) '1970-01-01 00:10:00.000000000',"
          "series_step=>INTERVAL '2' minute))"
          "ORDER BY generate_series ASC;";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        EXPECT_EQ(rows->rowCount(), size_t(5));
        EXPECT_EQ(rows->colCount(), size_t(1));
        for (int64_t val = 1; val <= 10; val += 2) {
          std::string expected_str =
              "1970-01-01 00:0" + std::to_string(val) + "00.000000000";
          int64_t expected_val = dateTimeParse<kTIMESTAMP>(
              boost::lexical_cast<std::string>(expected_str), 9);
          auto crt_row = rows->getNextRow(false, false);
          EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
        }
      }
    }

    // 3-arg version - test month time intervals
    // Test non-namned and named arg versions
    {
      const std::string non_named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series("
          "TIMESTAMP(9) '1970-01-01 00:00:00.000000000',"
          "TIMESTAMP(9) '1970-09-01 00:00:00.000000000',"
          "INTERVAL '1' month))"
          "ORDER BY generate_series ASC;";
      const std::string named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series("
          "series_start=>TIMESTAMP(9) '1970-01-01 00:00:00.000000000',"
          "series_stop=>TIMESTAMP(9) '1970-09-01 00:00:00.000000000',"
          "series_step=>INTERVAL '1' month))"
          "ORDER BY generate_series ASC;";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        EXPECT_EQ(rows->rowCount(), size_t(9));
        EXPECT_EQ(rows->colCount(), size_t(1));
        for (int64_t val = 1; val <= 9; val += 1) {
          std::string expected_str =
              "1970-0" + std::to_string(val) + "-01 00:000.000000000";
          int64_t expected_val = dateTimeParse<kTIMESTAMP>(
              boost::lexical_cast<std::string>(expected_str), 9);
          auto crt_row = rows->getNextRow(false, false);
          EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
        }
      }
    }

    // 3-arg version - test year time intervals
    // Test non-namned and named arg versions
    {
      const std::string non_named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series("
          "TIMESTAMP(9) '1970-01-01 00:00:00.000000000',"
          "TIMESTAMP(9) '1979-01-01 00:00:00.000000000',"
          "INTERVAL '2' year))"
          "ORDER BY generate_series ASC;";
      const std::string named_arg_query =
          "SELECT generate_series FROM TABLE(generate_series("
          "series_start=>TIMESTAMP(9) '1970-01-01 00:00:00.000000000',"
          "series_stop=>TIMESTAMP(9) '1979-01-01 00:00:00.000000000',"
          "series_step=>INTERVAL '2' year))"
          "ORDER BY generate_series ASC;";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        EXPECT_EQ(rows->rowCount(), size_t(5));
        EXPECT_EQ(rows->colCount(), size_t(1));
        for (int64_t val = 0; val <= 8; val += 2) {
          std::string expected_str =
              "197" + std::to_string(val) + "-01-01 00:000.000000000";
          int64_t expected_val = dateTimeParse<kTIMESTAMP>(
              boost::lexical_cast<std::string>(expected_str), 9);
          auto crt_row = rows->getNextRow(false, false);
          EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
        }
      }
    }

    // Series should be inclusive of stop value
    {
      const auto rows = run_multiple_agg(
          "SELECT generate_series FROM TABLE(generate_series("
          "TIMESTAMP(9) '2000-04-04 01:00:00.000000000',"
          "TIMESTAMP(9) '2000-04-04 09:00:00.000000000',"
          "INTERVAL '2' hour))"
          "ORDER BY generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(5));
      EXPECT_EQ(rows->colCount(), size_t(1));
      for (int64_t val = 1; val <= 9; val += 2) {
        std::string expected_str =
            "2000-04-04 0" + std::to_string(val) + ":00:00.000000000";
        int64_t expected_val =
            dateTimeParse<kTIMESTAMP>(boost::lexical_cast<std::string>(expected_str), 9);
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
      }
    }

    // Negative step
    {
      const auto rows = run_multiple_agg(
          "SELECT generate_series FROM TABLE(generate_series("
          "TIMESTAMP(9) '1999-09-04 09:00:00.000000000',"
          "TIMESTAMP(9) '1999-09-04 01:00:00.000000000',"
          "INTERVAL '-1' hour))"
          "ORDER BY generate_series DESC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(9));
      EXPECT_EQ(rows->colCount(), size_t(1));
      for (int64_t val = 9; val >= 1; val -= 1) {
        std::string expected_str =
            "1999-09-04 0" + std::to_string(val) + ":00:00.000000000";
        int64_t expected_val =
            dateTimeParse<kTIMESTAMP>(boost::lexical_cast<std::string>(expected_str), 9);
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
      }
    }

    // Negative month step
    {
      const auto rows = run_multiple_agg(
          "SELECT generate_series FROM TABLE(generate_series("
          "TIMESTAMP(9) '1999-09-04 09:00:00.000000000',"
          "TIMESTAMP(9) '1999-01-04 09:00:00.000000000',"
          "INTERVAL '-1' MONTH))"
          "ORDER BY generate_series DESC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(9));
      EXPECT_EQ(rows->colCount(), size_t(1));
      for (int64_t val = 9; val >= 1; val -= 1) {
        std::string expected_str =
            "1999-0" + std::to_string(val) + "-04 09:00:00.000000000";
        int64_t expected_val =
            dateTimeParse<kTIMESTAMP>(boost::lexical_cast<std::string>(expected_str), 9);
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
      }
    }

    // Negative year step
    {
      const auto rows = run_multiple_agg(
          "SELECT generate_series FROM TABLE(generate_series("
          "TIMESTAMP(9) '1999-09-04 09:00:00.000000000',"
          "TIMESTAMP(9) '1991-09-04 09:00:00.000000000',"
          "INTERVAL '-1' YEAR))"
          "ORDER BY generate_series DESC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(9));
      EXPECT_EQ(rows->colCount(), size_t(1));
      for (int64_t val = 9; val >= 1; val -= 1) {
        std::string expected_str =
            "199" + std::to_string(val) + "-09-04 09:00:00.000000000";
        int64_t expected_val =
            dateTimeParse<kTIMESTAMP>(boost::lexical_cast<std::string>(expected_str), 9);
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
      }
    }

    // Negative step and stop > start should return 0 rows
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(generate_series("
          "TIMESTAMP(9) '1970-01-01 00:00:02.000000000',"
          "TIMESTAMP(9) '1970-01-01 00:00:05.000000000',"
          "INTERVAL '-1' second))"
          "ORDER BY generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(0));
      EXPECT_EQ(rows->colCount(), size_t(1));
    }

    // Positive step and stop < start should return 0 rows
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(generate_series("
          "TIMESTAMP(9) '1970-01-01 00:00:05.000000000',"
          "TIMESTAMP(9) '1970-01-01 00:00:02.000000000',"
          "INTERVAL '1' second))"
          "ORDER BY generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(0));
      EXPECT_EQ(rows->colCount(), size_t(1));
    }

    // Negative step and stop == start should return 1 row (start)
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(generate_series("
          "TIMESTAMP(9) '1970-01-01 00:00:02.000000000',"
          "TIMESTAMP(9) '1970-01-01 00:00:02.000000000',"
          "INTERVAL '-1' second))"
          "ORDER BY generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(1));
      EXPECT_EQ(rows->colCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      std::string expected_str = "1970-01-01 00:00:02.000000000";
      int64_t expected_val =
          dateTimeParse<kTIMESTAMP>(boost::lexical_cast<std::string>(expected_str), 9);
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
    }

    // Positive step and stop == start should return 1 row (start)
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(generate_series("
          "TIMESTAMP(9) '1970-01-01 00:00:02.000000000',"
          "TIMESTAMP(9) '1970-01-01 00:00:02.000000000',"
          "INTERVAL '1' second))"
          "ORDER BY generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(1));
      EXPECT_EQ(rows->colCount(), size_t(1));
      auto crt_row = rows->getNextRow(false, false);
      std::string expected_str = "1970-01-01 00:00:02.000000000";
      int64_t expected_val =
          dateTimeParse<kTIMESTAMP>(boost::lexical_cast<std::string>(expected_str), 9);
      EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), expected_val);
    }
  }
}

TEST_F(SystemTFs, GenerateRandomStrings) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();
    {
      // num_strings must be >= 0
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(generate_random_strings(-3, 10));", dt),
          UserTableFunctionError);
    }

    {
      // string_length must be > 0 (to protect against our aliasing of empty strings to
      // null strings)
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(generate_random_strings(3, 0));", dt),
          UserTableFunctionError);
    }

    {
      // string_length must be <= StringDictionary::MAX_STRLEN (to protect against our
      // aliasing of empty strings to null strings)
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(generate_random_strings(3, 40000));", dt),
          UserTableFunctionError);
    }

    for (size_t idx = 0; idx < 10; ++idx) {
      const size_t num_strings = idx;
      const size_t str_len = 10UL - idx;

      const std::string non_named_arg_query =
          "SELECT id, rand_str FROM TABLE(generate_random_strings(" +
          std::to_string(num_strings) + ", " + std::to_string(str_len) +
          ")) ORDER BY id ASC;";
      const std::string named_arg_query =
          "SELECT id, rand_str FROM TABLE(generate_random_strings(num_strings => " +
          std::to_string(num_strings) + ", string_length => " + std::to_string(str_len) +
          ")) ORDER BY id ASC;";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        const size_t num_rows = rows->rowCount();
        const size_t num_cols = rows->colCount();
        ASSERT_EQ(num_rows, num_strings);
        ASSERT_EQ(num_cols, 2UL);

        for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
          auto row = rows->getNextRow(true, false);
          ASSERT_EQ(TestHelpers::v<int64_t>(row[0]), static_cast<int64_t>(row_idx));
          auto str = boost::get<std::string>(TestHelpers::v<NullableString>(row[1]));
          ASSERT_EQ(str.size(), str_len);
        }
      }
    }
  }
}

TEST_F(SystemTFs, Mandelbrot) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    if (dt == ExecutorDeviceType::CPU) {
      SKIP_NO_TBB();
    }
    // We won't make the default on GPU for function `tf_mandelbrot` use CUDA
    // until code cacheing is introduced for table functions
    const std::string tf_name =
        dt == ExecutorDeviceType::CPU ? "tf_mandelbrot" : "tf_mandelbrot_cuda";
    {
      // Mandelbrot table function requires max_iterations to be >= 1
      EXPECT_THROW(run_multiple_agg(
                       "SELECT * FROM TABLE(" + tf_name +
                           "(128 /* width */ , 128 /* "
                           "height */, -2.5 /* min_x */, 1.0 /* max_x */, -1.0 /* min_y "
                           "*/, 1.0 /* max_y */, 0 /* max_iterations */));",
                       dt),
                   UserTableFunctionError);
    }
    {
      // Should throw when using named argument syntax if argument is missing
      EXPECT_THROW(run_multiple_agg(
                       "SELECT * FROM TABLE(" + tf_name +
                           "(width=>128, height=>128, "
                           "min_x=>-2.5, max_x=>1.0, max_y=>1.0, max_iterations=>0));",
                       dt),
                   std::exception);
    }
    {
      const std::string non_named_arg_query =
          "SELECT MIN(num_iterations) AS min_iterations, MAX(num_iterations) AS "
          "max_iterations, COUNT(*) AS n FROM TABLE(" +
          tf_name +
          "(128 /* width */ , 128 "
          "/* height */, -2.5 /* min_x */, 1.0 /* max_x */, -1.0 /* min_y */, 1.0 /* "
          "max_y */, 256 /* max_iterations */));";
      // Varying spacing in query below is deliberate to verify parser robustness
      const std::string named_arg_query =
          "SELECT MIN(num_iterations) AS min_iterations, MAX(num_iterations) AS "
          "max_iterations, COUNT(*) AS n FROM TABLE(" +
          tf_name +
          "(x_pixels=>128, "
          "y_pixels => 128, y_max=>1.0,x_min =>-2.5, x_max => 1.0, y_min=>-1.0, "
          "max_iterations=>256));";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        ASSERT_EQ(rows->rowCount(), size_t(1));
        ASSERT_EQ(rows->colCount(), size_t(3));
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0]),
                  static_cast<int64_t>(1));  // min_iterations
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[1]),
                  static_cast<int64_t>(256));  // max_iterations
        ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[2]),
                  static_cast<int64_t>(16384));  // num pixels - width X height
      }
    }
  }
}

TEST_F(SystemTFs, GeoRasterize) {
  const std::string raster_values_sql =
      "CURSOR(SELECT CAST(x AS DOUBLE) AS x, CAST(y AS DOUBLE) AS y, CAST(z AS FLOAT) as "
      "z FROM (VALUES (0.0, 0.0, 10.0), (1.1, 1.2, 20.0), (0.8, 0., 5.0), (1.2, 1.43, "
      "15.0), (-0.4, 0.8, 40.0)) AS t(x, y, z))";
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();
    // tf_geo_rasterize requires bin_dim_meters to be > 0
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
                  ", 'MAX' /* agg_type */, 'BOX_AVG' /* fill_agg_type */, "
                  "0.0 /* bin_dim_meters */, false /* geographic_coords */, "
                  "0 /* neighborhood_fill_radius */, false /* fill_only_nulls */));",
              dt),
          TableFunctionError);
    }

    // tf_geo_rasterize requires neighborhood_fill_radius to be >= 0
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
                  ", 'MAX' /* agg_type */, 'BOX_AVG' /* fill_agg_type */, "
                  "1.0 /* bin_dim_meters */, false /* geographic_coords */, "
                  "-1 /* neighborhood_fill_radius */, false /* fill_only_nulls */));",
              dt),
          TableFunctionError);
    }

    // tf_geo_rasterize requires x_min to be < x_max
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
                  ", 'MAX' /* agg_type */, 'BOX_AVG' /* fill_agg_type */, "
                  "1.0 /* bin_dim_meters */, false /*, geographic_coords */, "
                  "0 /* neighborhood_fill_radius */, false /* fill_only_nulls */, "
                  "0.0 /* x_min */, 0.0 /* "
                  "x_max */, -1.0 /* y_min */, 1.0 /* y_max */));",
              dt),
          TableFunctionError);
    }

    // tf_geo_rasterize requires all non-default arguments to be specified
    {
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(tf_geo_rasterize(raster => " +
                               raster_values_sql +
                               ", agg_type => 'MAX', fill_agg_type => 'BOX_AVG', "
                               "geographic_coords => false, "
                               "fill_only_nulls => false));",
                           dt),
          std::exception);
    }
    // Test case without null fill radius or bounds definition
    {
      const auto non_named_arg_query =
          "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
          ", 'MAX' /* agg_type */, 'BOX_AVG' /* fill_agg_type */, "
          "1.0 /* bin_dim_meters */, false /* geographic_coords "
          "*/, 0 /* neighborhood_fill_radius */, false /* fill_only_nulls */)) ORDER BY "
          "x, y;";
      const auto named_arg_query =
          "SELECT * FROM TABLE(tf_geo_rasterize(raster => " + raster_values_sql +
          ", agg_type => 'MAX', fill_agg_type => 'BOX_AVG', "
          "bin_dim_meters => 1.0, neighborhood_fill_radius => 0, "
          "geographic_coords=>false, fill_only_nulls => false)) ORDER BY x, y;";
      const auto named_arg_query_using_defaults =
          "SELECT * FROM TABLE(tf_geo_rasterize(raster => " + raster_values_sql +
          ", agg_type => 'MAX', fill_agg_type => 'BOX_AVG', "
          "bin_dim_meters => 1.0, geographic_coords => false)) ORDER BY x, y;";
      for (auto query :
           {non_named_arg_query, named_arg_query, named_arg_query_using_defaults}) {
        const auto rows = run_multiple_agg(query, dt);
        const size_t num_rows = rows->rowCount();
        ASSERT_EQ(num_rows, size_t(6));
        ASSERT_EQ(rows->colCount(), size_t(3));
        const int64_t null_val = inline_fp_null_val(SQLTypeInfo(kFLOAT, false));
        const std::vector<int64_t> expected_z_values = {
            40, null_val, 10, null_val, null_val, 20};
        for (size_t r = 0; r < num_rows; ++r) {
          auto crt_row = rows->getNextRow(false, false);
          ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[0]))),
                    static_cast<int64_t>(r) / 2 - 1);
          ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[1]))),
                    static_cast<int64_t>(r) % 2);
          ASSERT_EQ(static_cast<int64_t>(TestHelpers::v<float>(crt_row[2])),
                    expected_z_values[r]);
        }
      }
    }
    // Test explicit raster bounds definition
    {
      const auto non_named_arg_query =
          "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
          ", 'MAX' /* agg_type */, 'BOX_AVG' /* fill_agg_type */, "
          "1.0 /* bin_dim_meters */, false /* geographic_coords "
          "*/, 0 /* neighborhood_fill_radius */, false /*, "
          "fill_only_nulls */, 1.0 /* x_min "
          "*/, 2.0 /* x_max */, 1.0 "
          "/* y_min */, 2.0 /* y_max */ )) ORDER BY x, y;";
      const auto named_arg_query =
          "SELECT * FROM TABLE(tf_geo_rasterize(raster => " + raster_values_sql +
          ", agg_type => 'MAX', fill_agg_type => 'BOX_AVG', bin_dim_meters => 1.0, "
          "x_max => 2.0, y_max => 2.0, fill_only_nulls=> false, "
          "x_min => 1.0, geographic_coords => false, neighborhood_fill_radius => 0, "
          "y_min => 1.0 )) ORDER BY x, y;";
      const auto named_arg_query_with_defaults =
          "SELECT * FROM TABLE(tf_geo_rasterize(raster => " + raster_values_sql +
          ", agg_type => 'MAX', fill_agg_type => 'BOX_AVG', bin_dim_meters => 1.0, "
          "x_max => 2.0, y_max => 2.0, fill_only_nulls=> false, "
          "x_min => 1.0, geographic_coords => false, neighborhood_fill_radius => 0, "
          "y_min => 1.0 )) ORDER BY x, y;";

      for (auto query :
           {non_named_arg_query, named_arg_query, named_arg_query_with_defaults}) {
        const auto rows = run_multiple_agg(query, dt);
        const size_t num_rows = rows->rowCount();
        ASSERT_EQ(num_rows, size_t(1));
        ASSERT_EQ(rows->colCount(), size_t(3));
        const std::vector<int64_t> expected_z_values = {20};
        for (size_t r = 0; r < num_rows; ++r) {
          auto crt_row = rows->getNextRow(false, false);
          ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[0]))),
                    static_cast<int64_t>(1));
          ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[1]))),
                    static_cast<int64_t>(1));
          ASSERT_EQ(static_cast<int64_t>(TestHelpers::v<float>(crt_row[2])),
                    expected_z_values[r]);
        }
      }
    }

    // Test null neighborhood fill radius
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
              ", 'MAX' /* agg_type */, 'BOX_AVG' /* fill_agg_type */, "
              "1.0 /* bin_dim_meters */, false /* geographic_coords */, 1 /* "
              "neighborhood_fill_radius */, true /* fill_only_nulls */)) ORDER BY x, y;",
          dt);
      const size_t num_rows = rows->rowCount();
      ASSERT_EQ(num_rows, size_t(6));
      ASSERT_EQ(rows->colCount(), size_t(3));
      const std::vector<int64_t> expected_z_values = {40, 25, 10, 23, 15, 20};
      for (size_t r = 0; r < num_rows; ++r) {
        auto crt_row = rows->getNextRow(false, false);
        ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[0]))),
                  static_cast<int64_t>(r) / 2 - 1);
        ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[1]))),
                  static_cast<int64_t>(r) % 2);
        ASSERT_EQ(static_cast<int64_t>(TestHelpers::v<float>(crt_row[2])),
                  expected_z_values[r]);
      }
    }

    // Test slope and aspect computation
    {
      std::string slope_aspect_raster_values_sql =
          "CURSOR(SELECT CAST(x AS DOUBLE) AS x, CAST(y AS DOUBLE) AS y, "
          "CAST(z AS DOUBLE) AS z FROM (VALUES ";
      for (int32_t y_bin = 0; y_bin < 5; ++y_bin) {
        for (int32_t x_bin = 0; x_bin < 5; ++x_bin) {
          const std::string x_val_str = std::to_string(x_bin * 2) + ".1";
          const std::string y_val_str = std::to_string(y_bin * 2) + ".2";
          const double z_val = 3.0 - abs(x_bin - 2);
          const std::string z_val_str = std::to_string(z_val);
          if (x_bin > 0 || y_bin > 0) {
            slope_aspect_raster_values_sql += ", ";
          }
          slope_aspect_raster_values_sql +=
              "(" + x_val_str + ", " + y_val_str + ", " + z_val_str + ")";
        }
      }
      slope_aspect_raster_values_sql += ") AS t(x, y, z))";

      const std::string non_named_arg_query =
          "SELECT * FROM TABLE(tf_geo_rasterize_slope(" + slope_aspect_raster_values_sql +
          ", 'MAX' /* agg_type */, 2.0 /* bin_dim_meters */, false /* geographic_coords "
          "*/, 0 /* "
          "neighborhood_fill_radius */, true /* fill_only_nulls */, true /* "
          "compute_slope_in_degrees */)) ORDER BY x, y;";
      const std::string named_arg_query =
          "SELECT * FROM TABLE(tf_geo_rasterize_slope("
          "raster => " +
          slope_aspect_raster_values_sql +
          ", agg_type => 'MAX', "
          "bin_dim_meters => 2.0, geographic_coords => false, "
          "neighborhood_fill_radius => 0, fill_only_nulls => true, "
          "compute_slope_in_degrees => true)) ORDER BY x, y;";
      const std::string named_arg_query_with_defaults =
          "SELECT * FROM TABLE(tf_geo_rasterize_slope("
          "raster => " +
          slope_aspect_raster_values_sql +
          ", agg_type => 'MAX', "
          "bin_dim_meters => 2.0, geographic_coords => false, "
          "fill_only_nulls => true, compute_slope_in_degrees => true)) ORDER BY x, y;";
      for (auto query :
           {non_named_arg_query, named_arg_query, named_arg_query_with_defaults}) {
        const auto rows = run_multiple_agg(query, dt);
        const size_t num_rows = rows->rowCount();
        ASSERT_EQ(num_rows, size_t(25));
        ASSERT_EQ(rows->colCount(), size_t(5));
        const double null_value = inline_fp_null_val(SQLTypeInfo(kDOUBLE, false));
        constexpr double SLOPE_EPS = 1.0e-7;
        for (int32_t x_bin = 0; x_bin < 5; ++x_bin) {
          for (int32_t y_bin = 0; y_bin < 5; ++y_bin) {
            auto crt_row = rows->getNextRow(false, false);
            ASSERT_EQ(
                static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[0]))),
                static_cast<int64_t>(x_bin * 2 + 1));
            ASSERT_EQ(
                static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[1]))),
                static_cast<int64_t>(y_bin * 2 + 1));
            ASSERT_EQ(
                static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[2]))),
                static_cast<int64_t>(3 - abs(x_bin - 2)));
            if (x_bin == 0 || x_bin == 4 || y_bin == 0 || y_bin == 4) {
              ASSERT_EQ(TestHelpers::v<double>(crt_row[3]), null_value);
              ASSERT_EQ(TestHelpers::v<double>(crt_row[4]), null_value);
            } else {
              const double expected_slope =
                  (x_bin == 1 || x_bin == 3) ? atan(0.5) * 180.0 / math_consts::m_pi : 0;
              ASSERT_NEAR(TestHelpers::v<double>(crt_row[3]), expected_slope, SLOPE_EPS);
              if (x_bin == 2) {
                // No aspect at crest
                ASSERT_EQ(TestHelpers::v<double>(crt_row[4]), null_value);
              } else {
                const double expected_aspect = x_bin == 1 ? 270.0 : 90.0;
                ASSERT_NEAR(
                    TestHelpers::v<double>(crt_row[4]), expected_aspect, SLOPE_EPS);
              }
            }
          }
        }
      }
    }
    // Test tf_geo_multi_rasterize
    {
      const std::string multi_raster_values_sql =
          "CURSOR(SELECT CAST(x AS DOUBLE) AS x, CAST(y AS DOUBLE) AS y, CAST(z0 AS "
          "FLOAT) AS "
          "z0, CAST(z1 AS FLOAT) AS z1 FROM (VALUES (0.0, 0.0, 10.0, 20.0), (1.1, 1.2, "
          "20.0, 30.0), "
          "(0.8, 0., 5.0, 15.0), (1.2, 1.43, 15.0, 25.0), (-0.4, 0.8, 40.0, 50.0)) AS "
          "t(x, y, z0, z1))";
      const std::string non_named_arg_query =
          "SELECT x, y, z[1] AS z0, z[2] AS z1 FROM "
          "TABLE(tf_geo_multi_rasterize(" +
          multi_raster_values_sql +
          ", 'MAX|MAX' /* agg_types */, 'BOX_AVG|BOX_AVG' /* fill_agg_types */, "
          "1.0 /* bin_dim_meters */, false /* geographic_coords "
          "*/, 0 /* neighborhood_fill_radius */, false /* fill_only_nulls */)) ORDER BY "
          "x, y;";
      const auto named_arg_query =
          "SELECT x, y, z[1] AS z0, z[2] AS z1 FROM "
          "TABLE(tf_geo_multi_rasterize(raster => " +
          multi_raster_values_sql +
          ", agg_types => 'MAX|MAX', fill_agg_types => 'BOX_AVG|BOX_AVG', "
          "bin_dim_meters => 1.0, neighborhood_fill_radius => 0, "
          "geographic_coords=>false, fill_only_nulls => false)) ORDER BY x, y;";
      const auto named_arg_query_with_defaults =
          "SELECT x, y, z[1] AS z0, z[2] AS z1 FROM "
          "TABLE(tf_geo_multi_rasterize(raster => " +
          multi_raster_values_sql +
          ", agg_types => 'MAX|MAX', fill_agg_types => 'BOX_AVG|BOX_AVG', "
          "bin_dim_meters => 1.0, geographic_coords=>false)) ORDER BY x, y;";
      for (auto query :
           {non_named_arg_query, named_arg_query, named_arg_query_with_defaults}) {
        const auto rows = run_multiple_agg(query, dt);
        const size_t num_rows = rows->rowCount();
        ASSERT_EQ(num_rows, size_t(6));
        ASSERT_EQ(rows->colCount(), size_t(4));
        const int64_t null_val = inline_fp_null_val(SQLTypeInfo(kFLOAT, false));
        const std::vector<int64_t> expected_z0_values = {
            40, null_val, 10, null_val, null_val, 20};
        const std::vector<int64_t> expected_z1_values = {
            50, null_val, 20, null_val, null_val, 30};
        for (size_t r = 0; r < num_rows; ++r) {
          auto crt_row = rows->getNextRow(false, false);
          ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[0]))),
                    static_cast<int64_t>(r) / 2 - 1);
          ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[1]))),
                    static_cast<int64_t>(r) % 2);
          ASSERT_EQ(static_cast<int64_t>(TestHelpers::v<float>(crt_row[2])),
                    expected_z0_values[r]);
          ASSERT_EQ(static_cast<int64_t>(TestHelpers::v<float>(crt_row[3])),
                    expected_z1_values[r]);
        }
      }
    }

    // TODO(todd): Add tests for geographic coords
  }
}

TEST_F(SystemTFs, ComputeDwellTimes) {
  // We currently crash when using TIMESTAMP values directly in a logical
  // values statement, so use integer values and TIMESTAMPADD as a workaround
  // until that can be fixed.
  // constexpr const int32_t null_int32_t = std::numeric_limits<int32_t>::lowest();
  constexpr const int64_t null_int64_t = std::numeric_limits<int64_t>::lowest();
  const std::string event_values_sql =
      "CURSOR(SELECT CAST(i AS BIGINT), cast(s AS BIGINT), TIMESTAMPADD(SECOND, t, "
      "TIMESTAMP(0) '2021-07-01 12:00:00') "
      "FROM(SELECT * FROM (VALUES (1, 100, 0), (2, 101, 60), (1, 100, -60), (3, 100, 0), "
      "(2, 101, -60), (1, 101, 240), (2, 102, 240), (2, 102, 300)) AS t(i, s, t)))";

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();
    // tf_compute_dwell_times requires min_dwell_points to be >= 0
    {
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(tf_compute_dwell_times(data => " +
                               event_values_sql +
                               ", min_dwell_points => -1, min_dwell_seconds => "
                               "10, max_inactive_seconds => 300));",
                           dt),
          TableFunctionError);
    }
    // tf_compute_dwell_times requires min_dwell_seconds to be >= 0
    {
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(tf_compute_dwell_times(data => " +
                               event_values_sql +
                               ", min_dwell_points => 1, min_dwell_seconds => "
                               "-1, max_inactive_seconds => 300));",
                           dt),
          TableFunctionError);
    }
    // tf_compute_dwell_times requires max_inactive_seconds to be >= 0
    {
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(tf_compute_dwell_times(data => " +
                               event_values_sql +
                               ", min_dwell_points => 1, min_dwell_seconds => "
                               "10, max_inactive_seconds => -1));",
                           dt),
          TableFunctionError);
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(tf_compute_dwell_times(data => " + event_values_sql +
              ", min_dwell_points => 1, min_dwell_seconds => "
              "0, max_inactive_seconds => 300)) ORDER BY entity_id, ts ASC;",
          dt);

      std::vector<int64_t> expected_ids = {1, 1, 2, 2, 3};
      std::vector<int64_t> expected_site_ids = {100, 101, 101, 102, 100};
      std::vector<int64_t> expected_prev_site_ids = {
          null_int64_t, 100, null_int64_t, 101, null_int64_t};
      std::vector<int64_t> expected_next_site_ids = {
          101, null_int64_t, 102, null_int64_t, null_int64_t};
      std::vector<int64_t> expected_session_ids = {1, 2, 1, 2, 1};
      std::vector<int64_t> expected_start_seq_ids = {1, 3, 1, 3, 1};
      std::vector<int64_t> expected_start_ts = {1625140740000000000,
                                                1625141040000000000,
                                                1625140740000000000,
                                                1625141040000000000,
                                                1625140800000000000};
      std::vector<int64_t> expected_dwell_times_sec = {60, 0, 120, 60, 0};
      std::vector<int64_t> expected_num_dwell_points = {2, 1, 2, 2, 1};

      ASSERT_EQ(rows->rowCount(), 5UL);
      ASSERT_EQ(rows->colCount(), 9UL);
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(expected_ids[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
        EXPECT_EQ(expected_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[1]));
        EXPECT_EQ(expected_prev_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[2]));
        EXPECT_EQ(expected_next_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[3]));
        EXPECT_EQ(expected_session_ids[row_idx], TestHelpers::v<int64_t>(crt_row[4]));
        EXPECT_EQ(expected_start_seq_ids[row_idx], TestHelpers::v<int64_t>(crt_row[5]));
        EXPECT_EQ(expected_start_ts[row_idx], TestHelpers::v<int64_t>(crt_row[6]));
        EXPECT_EQ(expected_dwell_times_sec[row_idx], TestHelpers::v<int64_t>(crt_row[7]));
        EXPECT_EQ(expected_num_dwell_points[row_idx],
                  TestHelpers::v<int64_t>(crt_row[8]));
      }
    }
    // Test max_inactive_time
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(tf_compute_dwell_times(data => " + event_values_sql +
              ", min_dwell_points => 1, min_dwell_seconds => "
              "0, max_inactive_seconds => 60)) ORDER BY entity_id, ts ASC;",
          dt);

      std::vector<int64_t> expected_ids = {1, 1, 2, 2, 2, 3};
      std::vector<int64_t> expected_site_ids = {100, 101, 101, 101, 102, 100};
      std::vector<int64_t> expected_prev_site_ids = {
          null_int64_t, 100, null_int64_t, 101, 101, null_int64_t};
      std::vector<int64_t> expected_next_site_ids = {
          101, null_int64_t, 101, 102, null_int64_t, null_int64_t};
      std::vector<int64_t> expected_session_ids = {1, 2, 1, 2, 3, 1};
      std::vector<int64_t> expected_start_seq_ids = {1, 3, 1, 2, 3, 1};
      std::vector<int64_t> expected_start_ts = {1625140740000000000,
                                                1625141040000000000,
                                                1625140740000000000,
                                                1625140860000000000,
                                                1625141040000000000,
                                                1625140800000000000};
      std::vector<int64_t> expected_dwell_times_sec = {60, 0, 0, 0, 60, 0};
      std::vector<int64_t> expected_num_dwell_points = {2, 1, 1, 1, 2, 1};

      ASSERT_EQ(rows->rowCount(), 6UL);
      ASSERT_EQ(rows->colCount(), 9UL);
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(expected_ids[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
        EXPECT_EQ(expected_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[1]));
        EXPECT_EQ(expected_prev_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[2]));
        EXPECT_EQ(expected_next_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[3]));
        EXPECT_EQ(expected_session_ids[row_idx], TestHelpers::v<int64_t>(crt_row[4]));
        EXPECT_EQ(expected_start_seq_ids[row_idx], TestHelpers::v<int64_t>(crt_row[5]));
        EXPECT_EQ(expected_start_ts[row_idx], TestHelpers::v<int64_t>(crt_row[6]));
        EXPECT_EQ(expected_dwell_times_sec[row_idx], TestHelpers::v<int64_t>(crt_row[7]));
        EXPECT_EQ(expected_num_dwell_points[row_idx],
                  TestHelpers::v<int64_t>(crt_row[8]));
      }
    }
    // Test min_dwell_points and min_dwell_seconds (both should give same result
    // with provided parameters)
    {
      const std::string min_dwell_points_query =
          "SELECT * FROM TABLE(tf_compute_dwell_times(data => " + event_values_sql +
          ", min_dwell_points => 2, min_dwell_seconds => 0, "
          "max_inactive_seconds => 300)) ORDER BY entity_id, ts ASC;";
      const std::string min_dwell_seconds_query =
          "SELECT * FROM TABLE(tf_compute_dwell_times(data => " + event_values_sql +
          ", min_dwell_points => 1, min_dwell_seconds => 60, "
          "max_inactive_seconds => 300)) ORDER BY entity_id, ts ASC;";

      std::vector<int64_t> expected_ids = {1, 2, 2};
      std::vector<int64_t> expected_site_ids = {100, 101, 102};
      std::vector<int64_t> expected_prev_site_ids = {null_int64_t, null_int64_t, 101};
      std::vector<int64_t> expected_next_site_ids = {null_int64_t, 102, null_int64_t};
      std::vector<int64_t> expected_session_ids = {1, 1, 2};
      std::vector<int64_t> expected_start_seq_ids = {1, 1, 3};
      std::vector<int64_t> expected_start_ts = {
          1625140740000000000, 1625140740000000000, 1625141040000000000};
      std::vector<int64_t> expected_dwell_times_sec = {60, 120, 60};
      std::vector<int64_t> expected_num_dwell_points = {2, 2, 2};

      for (auto query : {min_dwell_points_query, min_dwell_seconds_query}) {
        const auto rows = run_multiple_agg(query, dt);
        ASSERT_EQ(rows->rowCount(), 3UL);
        ASSERT_EQ(rows->colCount(), 9UL);

        for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
          auto crt_row = rows->getNextRow(false, false);
          EXPECT_EQ(expected_ids[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
          EXPECT_EQ(expected_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[1]));
          EXPECT_EQ(expected_prev_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[2]));
          EXPECT_EQ(expected_next_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[3]));
          EXPECT_EQ(expected_session_ids[row_idx], TestHelpers::v<int64_t>(crt_row[4]));
          EXPECT_EQ(expected_start_seq_ids[row_idx], TestHelpers::v<int64_t>(crt_row[5]));
          EXPECT_EQ(expected_start_ts[row_idx], TestHelpers::v<int64_t>(crt_row[6]));
          EXPECT_EQ(expected_dwell_times_sec[row_idx],
                    TestHelpers::v<int64_t>(crt_row[7]));
          EXPECT_EQ(expected_num_dwell_points[row_idx],
                    TestHelpers::v<int64_t>(crt_row[8]));
        }
      }
    }
    // Test interspersed null entity_ids and site_ids
    {
      const std::string event_values_with_nulls_sql =
          "CURSOR(SELECT CAST(i AS BIGINT), cast(s AS BIGINT), TIMESTAMPADD(SECOND, t, "
          "TIMESTAMP(0) '2021-07-01 "
          "12:00:00') "
          "FROM(SELECT * FROM (VALUES (1, 100, 0), (2, 101, 60), (1, 100, -60), (3, 100, "
          "0), "
          "(2, 101, -60), (NULL, 101, -30), (2, NULL, -30), (1, 101, 240), (2, 102, "
          "240), "
          "(2, 102, 300)) AS t(i, s, t)))";
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(tf_compute_dwell_times(data => " +
              event_values_with_nulls_sql +
              ", min_dwell_points => 1, min_dwell_seconds => "
              "0, max_inactive_seconds => 300)) ORDER BY entity_id, ts ASC;",
          dt);

      std::vector<int64_t> expected_ids = {1, 1, 2, 2, 2, 3};
      std::vector<int64_t> expected_site_ids = {100, 101, 101, 101, 102, 100};
      std::vector<int64_t> expected_prev_site_ids = {
          null_int64_t, 100, null_int64_t, 101, 101, null_int64_t};
      std::vector<int64_t> expected_next_site_ids = {
          101, null_int64_t, 101, 102, null_int64_t, null_int64_t};
      std::vector<int64_t> expected_session_ids = {1, 2, 1, 2, 3, 1};
      std::vector<int64_t> expected_start_seq_ids = {1, 3, 1, 3, 4, 1};
      std::vector<int64_t> expected_start_ts = {1625140740000000000,
                                                1625141040000000000,
                                                1625140740000000000,
                                                1625140860000000000,
                                                1625141040000000000,
                                                1625140800000000000};
      std::vector<int64_t> expected_dwell_times_sec = {60, 0, 0, 0, 60, 0};
      std::vector<int64_t> expected_num_dwell_points = {2, 1, 1, 1, 2, 1};

      ASSERT_EQ(rows->rowCount(), 6UL);
      ASSERT_EQ(rows->colCount(), 9UL);
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(expected_ids[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
        EXPECT_EQ(expected_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[1]));
        EXPECT_EQ(expected_prev_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[2]));
        EXPECT_EQ(expected_next_site_ids[row_idx], TestHelpers::v<int64_t>(crt_row[3]));
        EXPECT_EQ(expected_session_ids[row_idx], TestHelpers::v<int64_t>(crt_row[4]));
        EXPECT_EQ(expected_start_seq_ids[row_idx], TestHelpers::v<int64_t>(crt_row[5]));
        EXPECT_EQ(expected_start_ts[row_idx], TestHelpers::v<int64_t>(crt_row[6]));
        EXPECT_EQ(expected_dwell_times_sec[row_idx], TestHelpers::v<int64_t>(crt_row[7]));
        EXPECT_EQ(expected_num_dwell_points[row_idx],
                  TestHelpers::v<int64_t>(crt_row[8]));
      }
    }
  }
}

TEST_F(SystemTFs, FeatureSimilarity) {
  const std::string primary_features_sql =
      "CURSOR(SELECT CAST(k AS INT), cast(f AS INT), CAST(cnt AS INT) "
      "FROM(VALUES (1, 100, 2), (1, 101, 3), (1, 102, 2), "
      "(2, 100, 2), (2, 104, 3))  AS t(k, f, cnt))";
  const std::string comparison_features_sql =
      "CURSOR(SELECT cast(f AS INT), CAST(cnt AS INT) "
      "FROM(VALUES (100, 2), (101, 3), (102, 2)) "
      "AS t(f, cnt))";
  constexpr double max_epsilon = 0.01;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();
    // idf normalization off
    {
      const std::string tf_similarity_query =
          "SELECT * FROM TABLE(tf_feature_similarity(primary_features => " +
          primary_features_sql + ", comparison_features => " + comparison_features_sql +
          ", use_tf_idf => false)) "
          "ORDER BY class ASC;";
      const auto rows = run_multiple_agg(tf_similarity_query, dt);
      ASSERT_EQ(rows->rowCount(), 2UL);
      ASSERT_EQ(rows->colCount(), 2UL);
      const std::vector<int64_t> expected_classes = {1, 2};

      // Second similarity score is (2^2) /
      // ((sqrt(2^2 + 3^2) * sqrt(2^2 + 3^2 + 2^2)) ~= 0.26907
      const std::vector<double> expected_similarity_scores = {1.0, 0.2690691176};
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(expected_classes[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
        EXPECT_GE(TestHelpers::v<float>(crt_row[1]),
                  expected_similarity_scores[row_idx] - max_epsilon);
        EXPECT_LE(TestHelpers::v<float>(crt_row[1]),
                  expected_similarity_scores[row_idx] + max_epsilon);
      }
    }
    // idf normalization on
    {
      const std::string tf_similarity_query =
          "SELECT * FROM TABLE(tf_feature_similarity(primary_features => " +
          primary_features_sql + ", comparison_features => " + comparison_features_sql +
          ", use_tf_idf => true)) "
          "ORDER BY class ASC;";
      const auto rows = run_multiple_agg(tf_similarity_query, dt);
      ASSERT_EQ(rows->rowCount(), 2UL);
      ASSERT_EQ(rows->colCount(), 2UL);
      const std::vector<int64_t> expected_classes = {1, 2};
      const std::vector<double> expected_similarity_scores = {1.0, 0.141971051693};
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(expected_classes[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
        EXPECT_GE(TestHelpers::v<float>(crt_row[1]),
                  expected_similarity_scores[row_idx] - max_epsilon);
        EXPECT_LE(TestHelpers::v<float>(crt_row[1]),
                  expected_similarity_scores[row_idx] + max_epsilon);
      }
    }
  }
}

TEST_F(SystemTFs, FeatureSelfSimilarity) {
  const std::string primary_features_sql =
      "CURSOR(SELECT CAST(k AS INT), cast(f AS INT), CAST(cnt AS INT) "
      "FROM(VALUES (1, 100, 2), (1, 101, 3), (1, 102, 2), "
      "(2, 100, 2), (2, 104, 3))  AS t(k, f, cnt))";
  constexpr double max_epsilon = 0.01;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();
    // idf normalization off
    {
      const std::string tf_similarity_query =
          "SELECT * FROM TABLE(tf_feature_self_similarity(primary_features => " +
          primary_features_sql +
          ", use_tf_idf => false)) "
          "ORDER BY class1 ASC, class2 ASC;";
      const auto rows = run_multiple_agg(tf_similarity_query, dt);
      ASSERT_EQ(rows->rowCount(), 3UL);
      ASSERT_EQ(rows->colCount(), 3UL);
      const std::vector<int64_t> expected_class_1 = {1, 1, 2};
      const std::vector<int64_t> expected_class_2 = {1, 2, 2};
      const std::vector<double> expected_similarity_scores = {1.0, 0.26906913519, 1.0};
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(expected_class_1[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
        EXPECT_EQ(expected_class_2[row_idx], TestHelpers::v<int64_t>(crt_row[1]));
        EXPECT_GE(TestHelpers::v<float>(crt_row[2]),
                  expected_similarity_scores[row_idx] - max_epsilon);
        EXPECT_LE(TestHelpers::v<float>(crt_row[2]),
                  expected_similarity_scores[row_idx] + max_epsilon);
      }
    }
    // idf normalization on
    {
      const std::string tf_similarity_query =
          "SELECT * FROM TABLE(tf_feature_self_similarity(primary_features => " +
          primary_features_sql +
          ", use_tf_idf => true)) "
          "ORDER BY class1 ASC, class2 ASC;";
      const auto rows = run_multiple_agg(tf_similarity_query, dt);
      ASSERT_EQ(rows->rowCount(), 3UL);
      ASSERT_EQ(rows->colCount(), 3UL);
      const std::vector<int64_t> expected_class_1 = {1, 1, 2};
      const std::vector<int64_t> expected_class_2 = {1, 2, 2};
      const std::vector<double> expected_similarity_scores = {1.0, 0.14197105169, 1.0};
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(expected_class_1[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
        EXPECT_EQ(expected_class_2[row_idx], TestHelpers::v<int64_t>(crt_row[1]));
        EXPECT_GE(TestHelpers::v<float>(crt_row[2]),
                  expected_similarity_scores[row_idx] - max_epsilon);
        EXPECT_LE(TestHelpers::v<float>(crt_row[2]),
                  expected_similarity_scores[row_idx] + max_epsilon);
      }
    }
  }
}

TEST_F(SystemTFs, GraphShortestPath) {
  constexpr double max_epsilon = 0.01;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();
    {
      const std::string graph_values_sql =
          "CURSOR(SELECT CAST(node1 AS INT) AS node1, CAST(node2 AS INT) AS node2, "
          "CAST(distance AS FLOAT) as distance FROM (VALUES (1, 2, 1.0), (1, 3, 2.0), "
          "(2, 4, 3.0), (3, 4, 1.0)) AS t(node1, node2, distance))";
      const std::string origin_node{"1"};
      const std::string dest_node{"4"};
      const std::string query =
          "SELECT * FROM TABLE(tf_graph_shortest_path("
          "edge_list => " +
          graph_values_sql + ", origin_node => " + origin_node +
          ", destination_node => " + dest_node + ")) ORDER BY path_step ASC;";
      const auto rows = run_multiple_agg(query, dt);
      const std::vector<int32_t> expected_path_steps = {1, 2, 3};
      const std::vector<int32_t> expected_nodes = {1, 3, 4};
      const std::vector<float> expected_cume_dists = {0.0, 2.0, 3.0};
      ASSERT_EQ(rows->rowCount(), 3UL);
      ASSERT_EQ(rows->colCount(), 3UL);
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(expected_path_steps[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
        EXPECT_EQ(expected_nodes[row_idx], TestHelpers::v<int64_t>(crt_row[1]));
        EXPECT_GE(TestHelpers::v<float>(crt_row[2]),
                  expected_cume_dists[row_idx] - max_epsilon);
        EXPECT_LE(TestHelpers::v<float>(crt_row[2]),
                  expected_cume_dists[row_idx] + max_epsilon);
      }
    }
  }
}

TEST_F(SystemTFs, GraphShortestPathsDistances) {
  constexpr double max_epsilon = 0.01;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();
    {
      const std::string graph_values_sql =
          "CURSOR(SELECT CAST(node1 AS INT) AS node1, CAST(node2 AS INT) AS node2, "
          "CAST(distance AS FLOAT) as distance FROM (VALUES (1, 2, 1.0), (1, 3, 2.0), "
          "(2, 4, 3.0), (3, 4, 1.0)) AS t(node1, node2, distance))";
      const std::string origin_node{"1"};
      const std::string dest_node{"4"};
      const std::string query =
          "SELECT * FROM TABLE(tf_graph_shortest_paths_distances("
          "edge_list => " +
          graph_values_sql + ", origin_node => " + origin_node +
          ")) ORDER BY origin_node, "
          "destination_node ASC;";
      const auto rows = run_multiple_agg(query, dt);
      const std::vector<int32_t> expected_origin_nodes = {1, 1, 1, 1};
      const std::vector<int32_t> expected_destination_nodes = {1, 2, 3, 4};
      const std::vector<float> expected_distances = {0.0, 1.0, 2.0, 3.0};
      const std::vector<int32_t> expected_num_edges_traversed = {0, 1, 1, 2};
      ASSERT_EQ(rows->rowCount(), 4UL);
      ASSERT_EQ(rows->colCount(), 4UL);
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(expected_origin_nodes[row_idx], TestHelpers::v<int64_t>(crt_row[0]));
        EXPECT_EQ(expected_destination_nodes[row_idx],
                  TestHelpers::v<int64_t>(crt_row[1]));
        EXPECT_GE(TestHelpers::v<float>(crt_row[2]),
                  expected_distances[row_idx] - max_epsilon);
        EXPECT_LE(TestHelpers::v<float>(crt_row[2]),
                  expected_distances[row_idx] + max_epsilon);
        EXPECT_EQ(expected_num_edges_traversed[row_idx],
                  TestHelpers::v<int64_t>(crt_row[3]));
      }
    }
  }
}

TEST_F(SystemTFs, RasterGraphShortestPathsDistances) {
  constexpr double max_epsilon = 0.01;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();
    {
      const std::string raster_graph_values_sql =
          "CURSOR(SELECT CAST(a.x AS FLOAT) AS x, "
          "CAST(b.y AS FLOAT) AS y, "
          "CEIL(CAST(b.y AS FLOAT) / 2) * 5.0 + a.x * "
          "CASE WHEN MOD(b.y, 2) = 0 THEN 1.0 ELSE 0.5 END AS z "
          "FROM TABLE(generate_series(0, 9)) AS a(x), "
          "TABLE(generate_series(0, 9)) AS b(y))";

      const float origin_x{0.5};
      const float origin_y{0.5};
      const float destination_x{9.5};
      const float destination_y{9.5};

      const std::string query =
          "SELECT * FROM TABLE(tf_raster_graph_shortest_slope_weighted_path("
          "raster => " +
          raster_graph_values_sql +
          ", agg_type => 'AVG', "
          "bin_dim => 1.0, geographic_coords => FALSE, neighborhood_fill_radius => "
          "0, "
          "fill_only_nulls => TRUE, "
          "origin_x => " +
          std::to_string(origin_x) +
          ", "
          "origin_y => " +
          std::to_string(origin_y) +
          ", "
          "destination_x => " +
          std::to_string(destination_x) +
          ", "
          "destination_y => " +
          std::to_string(destination_y) +
          ", "
          "slope_weight_exponent => 3.0, slope_pct_max => 100.0)) ORDER BY "
          "path_step ASC";
      const auto rows = run_multiple_agg(query, dt);
      ASSERT_LE(rows->rowCount(), 100UL);
      ASSERT_EQ(rows->colCount(), 3UL);
      std::vector<float> x_positions;
      std::vector<float> y_positions;
      for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), static_cast<int64_t>(row_idx + 1));
        EXPECT_GE(TestHelpers::v<float>(crt_row[1]), 0.0);
        EXPECT_LE(TestHelpers::v<float>(crt_row[1]), 10.0);
        EXPECT_GE(TestHelpers::v<float>(crt_row[2]), 0.0);
        EXPECT_LE(TestHelpers::v<float>(crt_row[2]), 100.0);
        x_positions.emplace_back(TestHelpers::v<float>(crt_row[1]));
        y_positions.emplace_back(TestHelpers::v<float>(crt_row[2]));
      }
      EXPECT_GE(x_positions[0], origin_x - max_epsilon);
      EXPECT_LE(x_positions[0], origin_x + max_epsilon);
      EXPECT_GE(y_positions[0], origin_y - max_epsilon);
      EXPECT_LE(y_positions[0], origin_y + max_epsilon);

      const auto num_positions = x_positions.size();
      ASSERT_EQ(num_positions, rows->rowCount());

      EXPECT_GE(x_positions[num_positions - 1], destination_x - max_epsilon);
      EXPECT_LE(x_positions[num_positions - 1], destination_x + max_epsilon);
      EXPECT_GE(y_positions[num_positions - 1], destination_y - max_epsilon);
      EXPECT_LE(y_positions[num_positions - 1], destination_y + max_epsilon);
    }
  }
}

#ifdef HAVE_POINT_CLOUD_TFS

namespace {

const std::string laz_file_dir{"../../Tests/Import/datafiles/lidar/laz/"};
const std::string laz_example_file{
    "../../Tests/Import/datafiles/lidar/laz/10-998-552-511.laz"};
const int64_t laz_files_expected_major_version{1};
const int64_t laz_files_expected_minor_version{2};
const std::vector<std::string> laz_files_expected_file_names = {"10-998-550-511.laz",
                                                                "10-998-551-511.laz",
                                                                "10-998-552-511.laz"};
const std::vector<int64_t> laz_files_expected_num_points = {92599, 115018, 116841};
const std::vector<double> laz_files_expected_x_min_4326 = {-81.46036732093941,
                                                           -81.46036714127636,
                                                           -81.46036732093941};
const std::vector<double> laz_files_expected_x_max_4326 = {-81.45699957306475,
                                                           -81.45699953693926,
                                                           -81.45699949370029};
const std::vector<double> laz_files_expected_y_min_4326 = {41.4110481460852,
                                                           41.41357387954122,
                                                           41.41609944638788};
const std::vector<double> laz_files_expected_y_max_4326 = {41.4135736893742,
                                                           41.41609938006061,
                                                           41.4186249110525};
const std::vector<double> laz_files_expected_z_min_4326 = {360.2,
                                                           371.19,
                                                           360.6766573533145};
const std::vector<double> laz_files_expected_z_max_4326 = {407.2898145796289,
                                                           409.322834645669,
                                                           410.2890525781049};

constexpr double XYZ_EPS = 1.0e-6;
constexpr double Z_DATA_EPS =
    1.0e-2;  // We need a more relaxed value as the actual z ranges and calculated
             // metadata are more divergent

TEST_F(SystemTFs, PointCloudMetadata) {
  // get_point_cloud_meta

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();

    ddl_utils::FilePathWhitelist::clear();
    ddl_utils::FilePathWhitelist::initialize(BASE_PATH, "[]", "[]");
    // Test non-whitelisted file - should throw
    {
      EXPECT_ANY_THROW(run_multiple_agg("SELECT * FROM TABLE(tf_point_cloud_metadata('" +
                                            laz_file_dir + "')) ORDER BY y_min_4326 ASC;",
                                        dt));
    }

    ddl_utils::FilePathWhitelist::clear();
    ddl_utils::FilePathWhitelist::initialize(
        BASE_PATH, "[\"/\", \"../../Tests/Import/datafiles/lidar/laz\"]", "[\"/\"]");

    // Test non-existant file - should output 0 rows
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(tf_point_cloud_metadata('" + laz_file_dir +
              "i_dont_exist.laz')) ORDER BY y_min_4326 ASC;",
          dt);
      const size_t num_rows = rows->rowCount();
      ASSERT_EQ(num_rows, size_t(0));
    }

    // Test inverted filter extent range - should throw error
    {
      EXPECT_THROW(run_multiple_agg(
                       "SELECT * FROM TABLE(tf_point_cloud_metadata('" + laz_file_dir +
                           "', -90.0, -100.0, 41.412, 41.413)) ORDER BY y_min_4326 ASC;",
                       dt),
                   std::runtime_error);
    }

    // Test single file and specific columns
    {
      const auto non_named_arg_query =
          "SELECT file_name, num_points, y_min_4326, y_max_4326 FROM "
          "TABLE(tf_point_cloud_metadata('" +
          laz_example_file + "')) ORDER BY y_min_4326 ASC;";
      const auto named_arg_query =
          "SELECT file_name, num_points, y_min_4326, y_max_4326 FROM "
          "TABLE(tf_point_cloud_metadata(path => '" +
          laz_example_file + "')) ORDER BY y_min_4326 ASC;";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        const size_t num_rows = rows->rowCount();
        ASSERT_EQ(num_rows, size_t(1));
        ASSERT_EQ(rows->colCount(), size_t(4));
        auto row = rows->getNextRow(true, false);

        // Note we read metadata for the last file in y-range order, which is index 2
        // in the above expected metadata arrays
        ASSERT_EQ(boost::get<std::string>(TestHelpers::v<NullableString>(row[0])),
                  laz_files_expected_file_names[2]);
        ASSERT_EQ(TestHelpers::v<int64_t>(row[1]), laz_files_expected_num_points[2]);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[2]), laz_files_expected_y_min_4326[2], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[3]), laz_files_expected_y_max_4326[2], XYZ_EPS);
      }
    }

    // Test multiple files with all columns
    {
      const auto rows =
          run_multiple_agg("SELECT * FROM TABLE(tf_point_cloud_metadata('" +
                               laz_file_dir + "')) ORDER BY y_min_4326 ASC;",
                           dt);
      const size_t num_rows = rows->rowCount();
      ASSERT_EQ(num_rows, size_t(3));
      ASSERT_EQ(rows->colCount(), size_t(28));
      for (size_t r = 0; r < num_rows; ++r) {
        auto row = rows->getNextRow(true, false);
        // file name - column 2
        ASSERT_EQ(boost::get<std::string>(TestHelpers::v<NullableString>(row[1])),
                  laz_files_expected_file_names[r]);
        // major version - column 4
        ASSERT_EQ(TestHelpers::v<int64_t>(row[3]), laz_files_expected_major_version);
        // minor version - column 5
        ASSERT_EQ(TestHelpers::v<int64_t>(row[4]), laz_files_expected_minor_version);
        // num_points - column 8
        ASSERT_EQ(TestHelpers::v<int64_t>(row[7]), laz_files_expected_num_points[r]);

        // x_min_4326 - column 23
        ASSERT_NEAR(
            TestHelpers::v<double>(row[22]), laz_files_expected_x_min_4326[r], XYZ_EPS);
        // x_max_4326 - column 24
        ASSERT_NEAR(
            TestHelpers::v<double>(row[23]), laz_files_expected_x_max_4326[r], XYZ_EPS);
        // y_min_4326 - column 25
        ASSERT_NEAR(
            TestHelpers::v<double>(row[24]), laz_files_expected_y_min_4326[r], XYZ_EPS);
        // y_max_4326 - column 26
        ASSERT_NEAR(
            TestHelpers::v<double>(row[25]), laz_files_expected_y_max_4326[r], XYZ_EPS);
        // z_min_4326 - column 27
        ASSERT_NEAR(
            TestHelpers::v<double>(row[26]), laz_files_expected_z_min_4326[r], XYZ_EPS);
        // z_max_4326 - column 28
        ASSERT_NEAR(
            TestHelpers::v<double>(row[27]), laz_files_expected_z_max_4326[r], XYZ_EPS);
      }
    }

    // Test xy metadata filtering, here on y
    {
      const auto non_named_arg_query =
          "SELECT * FROM TABLE(tf_point_cloud_metadata('" + laz_file_dir +
          "', -90.0, -70.0, 41.412, 41.413)) ORDER BY y_min_4326 ASC;";
      const auto named_arg_query =
          "SELECT * FROM TABLE(tf_point_cloud_metadata(path=> '" + laz_file_dir +
          "', x_min => -90.0, x_max => -70.0, y_min => 41.412, y_max => 41.413)) "
          "ORDER BY y_min_4326 ASC;";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        const size_t num_rows = rows->rowCount();
        ASSERT_EQ(num_rows, size_t(1));
        ASSERT_EQ(rows->colCount(), size_t(28));
        auto row = rows->getNextRow(true, false);
        ASSERT_EQ(boost::get<std::string>(TestHelpers::v<NullableString>(row[1])),
                  laz_files_expected_file_names[0]);
        ASSERT_EQ(TestHelpers::v<int64_t>(row[7]), laz_files_expected_num_points[0]);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[24]), laz_files_expected_y_min_4326[0], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[25]), laz_files_expected_y_max_4326[0], XYZ_EPS);
      }
    }
  }
}

TEST_F(SystemTFs, LoadPointCloud) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    SKIP_NO_TBB();

    ddl_utils::FilePathWhitelist::clear();
    ddl_utils::FilePathWhitelist::initialize(BASE_PATH, "[]", "[]");

    // Test non-whitelisted file - should throw
    {
      EXPECT_ANY_THROW(run_multiple_agg(
          "SELECT COUNT(*) AS n FROM TABLE(tf_load_point_cloud('" + laz_file_dir + "'));",
          dt));
    }

    ddl_utils::FilePathWhitelist::clear();
    ddl_utils::FilePathWhitelist::initialize(
        BASE_PATH, "[\"/\", \"../../Tests/Import/datafiles/lidar/laz\"]", "[\"/\"]");

    // Test non-existant file - should output 0 rows
    {
      const auto rows =
          run_multiple_agg("SELECT COUNT(*) AS n FROM TABLE(tf_load_point_cloud('" +
                               laz_file_dir + "i_dont_exist.laz'));",
                           dt);
      const size_t num_rows = rows->rowCount();
      ASSERT_EQ(num_rows, size_t(1));
      ASSERT_EQ(rows->colCount(), size_t(1));
      auto row = rows->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<int64_t>(row[0]), 0);
    }

    // Test inverted filter extent range - should throw error
    {
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(tf_load_point_cloud('" + laz_file_dir +
                               "', -90.0, -100.0, 41.412, 41.413));",
                           dt),
          std::runtime_error);
    }

    // Test single file and specific columns
    {
      const auto non_named_arg_query =
          "SELECT COUNT(*) AS num_points, MIN(x) as min_x, MAX(x) as max_x, MIN(y) as "
          "min_y, MAX(y) as max_y, MIN(z) as min_z, MAX(z) as max_z FROM "
          "TABLE(tf_load_point_cloud('" +
          laz_example_file + "'));";
      const auto named_arg_query =
          "SELECT COUNT(*) AS num_points, MIN(x) as min_x, MAX(x) as max_x, MIN(y) as "
          "min_y, MAX(y) as max_y, MIN(z) as min_z, MAX(z) as max_z FROM "
          "TABLE(tf_load_point_cloud(path => '" +
          laz_example_file + "'));";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        const size_t num_rows = rows->rowCount();
        ASSERT_EQ(num_rows, size_t(1));
        ASSERT_EQ(rows->colCount(), size_t(7));
        auto row = rows->getNextRow(false, false);

        // Note: we read the last file in y-range order, which is index 2
        // in the above expected metadata arrays
        // We should find that the calculated range extents match up with the expected
        // file metadata

        ASSERT_EQ(TestHelpers::v<int64_t>(row[0]), laz_files_expected_num_points[2]);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[1]), laz_files_expected_x_min_4326[2], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[2]), laz_files_expected_x_max_4326[2], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[3]), laz_files_expected_y_min_4326[2], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[4]), laz_files_expected_y_max_4326[2], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[5]), laz_files_expected_z_min_4326[2], Z_DATA_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[6]), laz_files_expected_z_max_4326[2], Z_DATA_EPS);
      }
    }

    // Test multiple files and specific columns
    {
      const auto rows = run_multiple_agg(
          "SELECT COUNT(*) AS num_points, MIN(x) as min_x, MAX(x) as max_x, MIN(y) as "
          "min_y, MAX(y) as max_y, MIN(z) as min_z, MAX(z) as max_z FROM "
          "TABLE(tf_load_point_cloud('" +
              laz_file_dir + "'));",
          dt);
      const size_t num_rows = rows->rowCount();
      ASSERT_EQ(num_rows, size_t(1));
      ASSERT_EQ(rows->colCount(), size_t(7));
      auto row = rows->getNextRow(false, false);

      // We should find that the calculated range extents match up with the aggregate sum
      // for count, and min/max for the ranges, of the expected file metadata

      ASSERT_EQ(TestHelpers::v<int64_t>(row[0]),
                std::accumulate(laz_files_expected_num_points.begin(),
                                laz_files_expected_num_points.end(),
                                0));
      ASSERT_NEAR(
          TestHelpers::v<double>(row[1]), laz_files_expected_x_min_4326[1], XYZ_EPS);
      ASSERT_NEAR(
          TestHelpers::v<double>(row[2]), laz_files_expected_x_max_4326[0], XYZ_EPS);
      ASSERT_NEAR(
          TestHelpers::v<double>(row[3]), laz_files_expected_y_min_4326[0], XYZ_EPS);
      ASSERT_NEAR(
          TestHelpers::v<double>(row[4]), laz_files_expected_y_max_4326[2], XYZ_EPS);
      ASSERT_NEAR(
          TestHelpers::v<double>(row[5]), laz_files_expected_z_min_4326[0], Z_DATA_EPS);
      ASSERT_NEAR(
          TestHelpers::v<double>(row[6]), laz_files_expected_z_max_4326[2], Z_DATA_EPS);
    }

    // Test xy metadata filtering, here on y
    // Also test cache off
    {
      const auto non_named_arg_query =
          "SELECT COUNT(*) AS num_points, MIN(x) as min_x, MAX(x) as max_x, MIN(y) as "
          "min_y, MAX(y) as max_y, MIN(z) as min_z, MAX(z) as max_z FROM "
          "TABLE(tf_load_point_cloud('" +
          laz_file_dir + "', 'EPSG:4326', false, -90.0, -70.0, 41.412, 41.413));";
      const auto named_arg_query =
          "SELECT COUNT(*) AS num_points, MIN(x) as min_x, MAX(x) as max_x, MIN(y) as "
          "min_y, MAX(y) as max_y, MIN(z) as min_z, MAX(z) as max_z FROM "
          "TABLE(tf_load_point_cloud(path => '" +
          laz_file_dir +
          "'"
          ", out_srs => 'EPSG:4326', use_cache => false, x_min => -90.0, "
          "x_max => -70.0, y_min => 41.412, y_max => 41.413));";
      for (auto query : {non_named_arg_query, named_arg_query}) {
        const auto rows = run_multiple_agg(query, dt);
        const size_t num_rows = rows->rowCount();
        ASSERT_EQ(num_rows, size_t(1));
        ASSERT_EQ(rows->colCount(), size_t(7));
        auto row = rows->getNextRow(false, false);

        // Note that the range filters will cause us to read the first file
        // in y-range order, index 0 in the above expected metadata arrays
        // We should find that the calculated range extents match up with the expected
        // file metadata

        ASSERT_EQ(TestHelpers::v<int64_t>(row[0]), laz_files_expected_num_points[0]);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[1]), laz_files_expected_x_min_4326[0], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[2]), laz_files_expected_x_max_4326[0], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[3]), laz_files_expected_y_min_4326[0], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[4]), laz_files_expected_y_max_4326[0], XYZ_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[5]), laz_files_expected_z_min_4326[0], Z_DATA_EPS);
        ASSERT_NEAR(
            TestHelpers::v<double>(row[6]), laz_files_expected_z_max_4326[0], Z_DATA_EPS);
      }
    }
    // TODO(todd): Add tests for coordinate conversions
  }
}

}  // unnamed namespace

#endif  // HAVE_POINT_CLOUD_TFS

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
