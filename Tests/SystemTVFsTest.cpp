/*
 * Copyright 2019 OmniSci, Inc.
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

#ifdef HAVE_SYSTEM_TVFS

#include "TestHelpers.h"

#include <gtest/gtest.h>

#include "QueryEngine/ResultSet.h"
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

class SystemTVFs : public ::testing::Test {
  void SetUp() override {}
};

TEST_F(SystemTVFs, Mandelbrot) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // Mandelbrot table function requires max_iterations to be >= 1
      EXPECT_THROW(
          run_multiple_agg("SELECT * FROM TABLE(tvf_mandelbrot(128 /* width */ , 128 /* "
                           "height */, -2.5 /* min_x */, 1.0 /* max_x */, -1.0 /* min_y "
                           "*/, 1.0 /* max_y */, 0 /* max_iterations */));",
                           dt),
          std::runtime_error);
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT MIN(num_iterations) AS min_iterations, MAX(num_iterations) AS "
          "max_iterations, COUNT(*) AS n FROM TABLE(tvf_mandelbrot(128 /* width */ , 128 "
          "/* height */, -2.5 /* min_x */, 1.0 /* max_x */, -1.0 /* min_y */, 1.0 /* "
          "max_y */, 256 /* max_iterations */));",
          dt);
      ASSERT_EQ(rows->rowCount(), size_t(1));
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

#endif  // HAVE_SYSTEM_TVFS
