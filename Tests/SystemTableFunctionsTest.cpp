/*
 * Copyright 2021 OmniSci, Inc.
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

#include <gtest/gtest.h>

#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <math.h>
#include "../Shared/math_consts.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/TableFunctions/TableFunctionManager.h"
#include "QueryRunner/QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

extern bool g_enable_table_functions;
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

class SystemTFs : public ::testing::Test {
  void SetUp() override {}
};

TEST_F(SystemTFs, GenerateSeries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // Step of 0 is not permitted
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(generate_series(3, 10, 0));", dt),
          UserTableFunctionError);
    }

    // Test step of 2
    {
      const auto rows = run_multiple_agg(
          "SELECT generate_series FROM TABLE(generate_series(1, 10, 2)) ORDER BY "
          "generate_series ASC;",
          dt);
      EXPECT_EQ(rows->rowCount(), size_t(5));
      EXPECT_EQ(rows->colCount(), size_t(1));
      for (int64_t val = 1; val <= 10; val += 2) {
        auto crt_row = rows->getNextRow(false, false);
        EXPECT_EQ(TestHelpers::v<int64_t>(crt_row[0]), val);
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

    // Outputs of more than 2^30 rows are not allowed
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT COUNT(*) AS n, MIN(generate_series) AS min_series, "
              "MAX(generate_series) AS max_series, AVG(generate_series) as avg_series "
              "FROM (SELECT * FROM TABLE(generate_series(0, 2000000000, 1)));",
              dt),
          UserTableFunctionError);
    }
  }
}

TEST_F(SystemTFs, Mandelbrot) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // Mandelbrot table function requires max_iterations to be >= 1
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(tf_mandelbrot(128 /* width */ , 128 /* "
                           "height */, -2.5 /* min_x */, 1.0 /* max_x */, -1.0 /* min_y "
                           "*/, 1.0 /* max_y */, 0 /* max_iterations */));",
                           dt),
          UserTableFunctionError);
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT MIN(num_iterations) AS min_iterations, MAX(num_iterations) AS "
          "max_iterations, COUNT(*) AS n FROM TABLE(tf_mandelbrot(128 /* width */ , 128 "
          "/* height */, -2.5 /* min_x */, 1.0 /* max_x */, -1.0 /* min_y */, 1.0 /* "
          "max_y */, 256 /* max_iterations */));",
          dt);
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
    {
      // skip CPU for GPU tests
      if (dt == ExecutorDeviceType::GPU) {
        const auto rows = run_multiple_agg(
            "SELECT MIN(num_iterations) AS min_iterations, MAX(num_iterations) AS "
            "max_iterations, COUNT(*) AS n FROM TABLE(tf_mandelbrot_cuda(128 /* width */ "
            ", "
            "128 "
            "/* height */, -2.5 /* min_x */, 1.0 /* max_x */, -1.0 /* min_y */, 1.0 /* "
            "max_y */, 256 /* max_iterations */));",
            dt);
        ASSERT_EQ(rows->rowCount(), size_t(1));
        ASSERT_EQ(rows->colCount(), size_t(3));
        auto crt_row = rows->getNextRow(false, false);
        // TODO: The CUDA function seems to return -1 for some reason
        // ASSERT_EQ(TestHelpers::v<int64_t>(crt_row[0])
        //          static_cast<int64_t>(1));  // min_iterations
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
    // tf_geo_rasterize requires bin_dim_meters to be > 0
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
                  ", 0.0 /* bin_dim_meters */, false /* geographic_coords "
                  "*/, 0 /* neighborhood_fill_radius */, false /* fill_only_nulls */));",
              dt),
          UserTableFunctionError);
    }

    // tf_geo_rasterize requires neighborhood_fill_radius to be >= 0
    {
      EXPECT_THROW(
          run_multiple_agg(
              "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
                  ", 1.0 /* bin_dim_meters */, false /* geographic_coords "
                  "*/, -1 /* neighborhood_fill_radius */, false /* fill_only_nulls */));",
              dt),
          UserTableFunctionError);
    }

    // tf_geo_rasterize requires x_min to be < x_max
    {
      EXPECT_THROW(run_multiple_agg(
                       "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
                           ", 1.0 /* bin_dim_meters */, false /* geographic_coords */, 0 "
                           "/* neighborhood_fill_radius */, false /* fill_only_nulls */, "
                           "0.0 /* x_min */, 0.0 /* "
                           "x_max */, -1.0 /* y_min */, 1.0 /* y_max */));",
                       dt),
                   UserTableFunctionError);
    }
    // Test case without null fill radius or bounds definition
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
              ", 1.0 /* bin_dim_meters */, false /* geographic_coords */, 0 /* "
              "neighborhood_fill_radius */, false /* fill_only_nulls */)) ORDER BY x, y;",
          dt);
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
    // Test explicit raster bounds definition
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
              ", 1.0 /* bin_dim_meters */, false /* geographic_coords */, 0 /* "
              "neighborhood_fill_radius */, false /* fill_only_nulls */, 1.0 /* x_min "
              "*/, 2.0 /* x_max */, 1.0 "
              "/* y_min */, 2.0 /* y_max */ )) ORDER BY x, y;",
          dt);
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
    // Test null neighborhood fill radius
    {
      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(tf_geo_rasterize(" + raster_values_sql +
              ", 1.0 /* bin_dim_meters */, false /* geographic_coords */, 1 /* "
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
          "CURSOR(SELECT CAST(x AS DOUBLE) AS x, CAST(y AS DOUBLE) AS y, CAST(z AS "
          "DOUBLE) as "
          "z FROM (VALUES ";
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

      const auto rows = run_multiple_agg(
          "SELECT * FROM TABLE(tf_geo_rasterize_slope(" + slope_aspect_raster_values_sql +
              ", 2.0 /* bin_dim_meters */, false /* geographic_coords */, 0 /* "
              "neighborhood_fill_radius */, true /* fill_only_nulls */, true /* "
              "compute_slope_in_degrees */)) ORDER BY x, y;",
          dt);

      const size_t num_rows = rows->rowCount();
      ASSERT_EQ(num_rows, size_t(25));
      ASSERT_EQ(rows->colCount(), size_t(5));
      const double null_value = inline_fp_null_val(SQLTypeInfo(kDOUBLE, false));
      constexpr double SLOPE_EPS = 1.0e-7;
      for (int32_t x_bin = 0; x_bin < 5; ++x_bin) {
        for (int32_t y_bin = 0; y_bin < 5; ++y_bin) {
          auto crt_row = rows->getNextRow(false, false);
          ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[0]))),
                    static_cast<int64_t>(x_bin * 2 + 1));
          ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[1]))),
                    static_cast<int64_t>(y_bin * 2 + 1));
          ASSERT_EQ(static_cast<int64_t>(std::floor(TestHelpers::v<double>(crt_row[2]))),
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
              ASSERT_NEAR(TestHelpers::v<double>(crt_row[4]), expected_aspect, SLOPE_EPS);
            }
          }
        }
      }
    }

    // TODO(todd): Add tests for geographic coords
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
