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

#include <gtest/gtest.h>

#include "Catalog/Catalog.h"
#include "Catalog/DBObject.h"
#include "DBHandlerTestHelpers.h"
#include "QueryEngine/Execute.h"
#include "QueryRunner/QueryRunner.h"

namespace po = boost::program_options;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace Catalog_Namespace;

extern bool g_enable_table_functions;
extern bool g_enable_overlaps_hashjoin;

using QR = QueryRunner::QueryRunner;

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

bool approx_eq(const double v, const double target, const double eps = 0.01) {
  const auto v_u64 = *reinterpret_cast<const uint64_t*>(may_alias_ptr(&v));
  const auto target_u64 = *reinterpret_cast<const uint64_t*>(may_alias_ptr(&target));
  return v_u64 == target_u64 || (target - eps < v && v < target + eps);
}

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

std::shared_ptr<ResultSet> run_query(const std::string& query_str,
                                     const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, true, true);
}

void createTable() {
  QR::get()->runDDLStatement(
      "CREATE TABLE SQL_HINT_DUMMY(key int, ts1 timestamp(0) encoding fixed(32), ts2 "
      "timestamp(0) encoding fixed(32), str1 TEXT ENCODING DICT(16));");
  QR::get()->runDDLStatement(
      "CREATE TABLE geospatial_test(id INT, p POINT, l LINESTRING, poly POLYGON);");
  QR::get()->runDDLStatement(
      "CREATE TABLE geospatial_inner_join_test(id INT, p POINT, l LINESTRING, poly "
      "POLYGON);");
}

void dropTable() {
  QR::get()->runDDLStatement("DROP TABLE IF EXISTS SQL_HINT_DUMMY;");
  QR::get()->runDDLStatement("DROP TABLE IF EXISTS geospatial_test;");
  QR::get()->runDDLStatement("DROP TABLE IF EXISTS geospatial_inner_join_test;");
}

TEST(kCpuMode, ForceToCPUMode) {
  const auto query_with_cpu_mode_hint = "SELECT /*+ cpu_mode */ * FROM SQL_HINT_DUMMY";
  const auto query_without_cpu_mode_hint = "SELECT * FROM SQL_HINT_DUMMY";
  if (QR::get()->gpusPresent()) {
    auto query_hints = QR::get()->getParsedQueryHint(query_with_cpu_mode_hint);
    const bool cpu_mode_enabled = query_hints.isHintRegistered(QueryHint::kCpuMode);
    EXPECT_TRUE(cpu_mode_enabled);
    query_hints = QR::get()->getParsedQueryHint(query_without_cpu_mode_hint);
    EXPECT_FALSE(query_hints.isAnyQueryHintDelivered());
  }
}

TEST(QueryHint, CheckQueryHintForOverlapsJoin) {
  const auto overlaps_join_status_backup = g_enable_overlaps_hashjoin;
  g_enable_overlaps_hashjoin = true;
  ScopeGuard reset_loop_join_state = [&overlaps_join_status_backup] {
    g_enable_overlaps_hashjoin = overlaps_join_status_backup;
  };

  {
    const auto q1 =
        "SELECT /*+ overlaps_bucket_threshold(0.718) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q1_hints = QR::get()->getParsedQueryHint(q1);
    EXPECT_TRUE(q1_hints.isHintRegistered(QueryHint::kOverlapsBucketThreshold) &&
                approx_eq(q1_hints.overlaps_bucket_threshold, 0.718));
  }
  {
    const auto q2 =
        "SELECT /*+ overlaps_max_size(2021) */ a.id FROM geospatial_test a INNER JOIN "
        "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q2_hints = QR::get()->getParsedQueryHint(q2);
    EXPECT_TRUE(q2_hints.isHintRegistered(QueryHint::kOverlapsMaxSize) &&
                (q2_hints.overlaps_max_size == 2021));
  }

  {
    const auto q3 =
        "SELECT /*+ overlaps_bucket_threshold(0.718), overlaps_max_size(2021) */ a.id "
        "FROM "
        "geospatial_test a INNER JOIN geospatial_inner_join_test b ON "
        "ST_Contains(b.poly, "
        "a.p);";
    auto q3_hints = QR::get()->getParsedQueryHint(q3);
    EXPECT_TRUE(q3_hints.isHintRegistered(QueryHint::kOverlapsBucketThreshold) &&
                q3_hints.isHintRegistered(QueryHint::kOverlapsMaxSize) &&
                (q3_hints.overlaps_max_size == 2021) &&
                approx_eq(q3_hints.overlaps_bucket_threshold, 0.718));
  }

  {
    const auto query =
        R"(SELECT /*+ overlaps_allow_gpu_build */ a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);)";
    const auto hints = QR::get()->getParsedQueryHint(query);
    EXPECT_TRUE(hints.isHintRegistered(QueryHint::kOverlapsAllowGpuBuild));
    EXPECT_TRUE(hints.overlaps_allow_gpu_build);
  }
  {
    const auto q4 =
        "SELECT /*+ overlaps_bucket_threshold(0.1) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q4_hints = QR::get()->getParsedQueryHint(q4);
    EXPECT_TRUE(q4_hints.isHintRegistered(QueryHint::kOverlapsBucketThreshold) &&
                approx_eq(q4_hints.overlaps_bucket_threshold, 0.1));
  }
  {
    const auto q5 =
        "SELECT /*+ overlaps_keys_per_bin(0.1) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q5_hints = QR::get()->getParsedQueryHint(q5);
    EXPECT_TRUE(q5_hints.isHintRegistered(QueryHint::kOverlapsKeysPerBin) &&
                approx_eq(q5_hints.overlaps_keys_per_bin, 0.1));
  }
  {
    const auto q6 =
        "SELECT /*+ overlaps_keys_per_bin(19980909.01) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q6_hints = QR::get()->getParsedQueryHint(q6);
    EXPECT_TRUE(q6_hints.isHintRegistered(QueryHint::kOverlapsKeysPerBin) &&
                approx_eq(q6_hints.overlaps_keys_per_bin, 19980909.01));
  }

  {
    const auto query_without_hint =
        "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON "
        "ST_Contains(b.poly, a.p);";
    auto query_without_hint_res = QR::get()->getParsedQueryHint(query_without_hint);
    EXPECT_TRUE(!query_without_hint_res.isAnyQueryHintDelivered());
  }

  {
    const auto wrong_q1 =
        "SELECT /*+ overlaps_bucket_threshold(-0.718) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto wrong_q1_hints = QR::get()->getParsedQueryHint(wrong_q1);
    EXPECT_TRUE(!wrong_q1_hints.isHintRegistered(QueryHint::kOverlapsBucketThreshold));
  }

  {
    const auto wrong_q2 =
        "SELECT /*+ overlaps_bucket_threshold(91.718) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto wrong_q2_hints = QR::get()->getParsedQueryHint(wrong_q2);
    EXPECT_TRUE(!wrong_q2_hints.isHintRegistered(QueryHint::kOverlapsBucketThreshold));
  }

  {
    const auto wrong_q3 =
        "SELECT /*+ overlaps_max_size(-2021) */ a.id FROM geospatial_test a INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto wrong_q3_hints = QR::get()->getParsedQueryHint(wrong_q3);
    EXPECT_TRUE(!wrong_q3_hints.isHintRegistered(QueryHint::kOverlapsMaxSize));
  }
  {
    const auto wrong_q4 =
        "SELECT /*+ overlaps_keys_per_bin(-0.1) */ a.id FROM geospatial_test a INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto wrong_q4_hints = QR::get()->getParsedQueryHint(wrong_q4);
    EXPECT_TRUE(!wrong_q4_hints.isHintRegistered(QueryHint::kOverlapsKeysPerBin));
  }
  {
    // overlaps_keys_per_bin needs to below then DOUBLE_MAX
    auto double_max = std::to_string(std::numeric_limits<double>::max());
    const auto wrong_q5 =
        "SELECT /*+ overlaps_keys_per_bin(" + double_max +
        ") */ a.id "
        "FROM geospatial_test a INNER JOIN geospatial_inner_join_test b "
        "ON ST_Contains(b.poly, a.p);";
    auto wrong_q5_hints = QR::get()->getParsedQueryHint(wrong_q5);
    EXPECT_TRUE(!wrong_q5_hints.isHintRegistered(QueryHint::kOverlapsKeysPerBin));
  }
}

TEST(QueryHint, checkQueryLayoutHintWithEnablingColumnarOutput) {
  const auto enable_columnar_output = g_enable_columnar_output;
  g_enable_columnar_output = true;
  ScopeGuard reset_columnar_output = [&enable_columnar_output] {
    g_enable_columnar_output = enable_columnar_output;
  };

  const auto q1 = "SELECT /*+ columnar_output */ * FROM SQL_HINT_DUMMY";
  const auto q2 = "SELECT /*+ rowwise_output */ * FROM SQL_HINT_DUMMY";
  const auto q3 = "SELECT /*+ columnar_output, rowwise_output */ * FROM SQL_HINT_DUMMY";
  const auto q4 = "SELECT /*+ rowwise_output, columnar_output */ * FROM SQL_HINT_DUMMY";
  const auto q5 =
      "SELECT /*+ rowwise_output, columnar_output, rowwise_output */ * FROM "
      "SQL_HINT_DUMMY";
  const auto q6 = "SELECT /*+ rowwise_output, rowwise_output */ * FROM SQL_HINT_DUMMY";
  const auto q7 = "SELECT /*+ columnar_output, columnar_output */ * FROM SQL_HINT_DUMMY";
  {
    auto query_hints = QR::get()->getParsedQueryHint(q1);
    auto hint_enabled = query_hints.isHintRegistered(QueryHint::kColumnarOutput);
    EXPECT_FALSE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q2);
    auto hint_enabled = query_hints.isHintRegistered(QueryHint::kRowwiseOutput);
    EXPECT_TRUE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q3);
    auto hint_enabled = query_hints.isAnyQueryHintDelivered();
    EXPECT_FALSE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q4);
    auto hint_enabled = query_hints.isAnyQueryHintDelivered();
    EXPECT_FALSE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q5);
    auto hint_enabled = query_hints.isAnyQueryHintDelivered();
    EXPECT_FALSE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q6);
    auto hint_enabled = query_hints.isHintRegistered(QueryHint::kRowwiseOutput);
    EXPECT_TRUE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q7);
    auto hint_enabled = query_hints.isHintRegistered(QueryHint::kColumnarOutput);
    EXPECT_FALSE(hint_enabled);
  }
}

TEST(QueryHint, checkQueryLayoutHintWithoutEnablingColumnarOutput) {
  const auto enable_columnar_output = g_enable_columnar_output;
  g_enable_columnar_output = false;
  ScopeGuard reset_columnar_output = [&enable_columnar_output] {
    g_enable_columnar_output = enable_columnar_output;
  };
  const auto q1 = "SELECT /*+ columnar_output */ * FROM SQL_HINT_DUMMY";
  const auto q2 = "SELECT /*+ rowwise_output */ * FROM SQL_HINT_DUMMY";
  const auto q3 = "SELECT /*+ columnar_output, rowwise_output */ * FROM SQL_HINT_DUMMY";
  const auto q4 = "SELECT /*+ rowwise_output, columnar_output */ * FROM SQL_HINT_DUMMY";
  const auto q5 =
      "SELECT /*+ rowwise_output, columnar_output, rowwise_output */ * FROM "
      "SQL_HINT_DUMMY";
  const auto q6 = "SELECT /*+ rowwise_output, rowwise_output */ * FROM SQL_HINT_DUMMY";
  const auto q7 = "SELECT /*+ columnar_output, columnar_output */ * FROM SQL_HINT_DUMMY";
  {
    auto query_hints = QR::get()->getParsedQueryHint(q1);
    auto hint_enabled = query_hints.isHintRegistered(QueryHint::kColumnarOutput);
    EXPECT_TRUE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q2);
    auto hint_enabled = query_hints.isHintRegistered(QueryHint::kRowwiseOutput);
    EXPECT_FALSE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q3);
    auto hint_enabled = query_hints.isAnyQueryHintDelivered();
    EXPECT_FALSE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q4);
    auto hint_enabled = query_hints.isAnyQueryHintDelivered();
    EXPECT_FALSE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q5);
    auto hint_enabled = query_hints.isAnyQueryHintDelivered();
    EXPECT_FALSE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q6);
    auto hint_enabled = query_hints.isHintRegistered(QueryHint::kRowwiseOutput);
    EXPECT_FALSE(hint_enabled);
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q7);
    auto hint_enabled = query_hints.isHintRegistered(QueryHint::kColumnarOutput);
    EXPECT_TRUE(hint_enabled);
  }
}

TEST(QueryHint, UDF) {
  const auto enable_columnar_output = g_enable_columnar_output;
  g_enable_columnar_output = false;
  ScopeGuard reset_columnar_output = [&enable_columnar_output] {
    g_enable_columnar_output = enable_columnar_output;
  };

  const auto q1 =
      "SELECT out0 FROM TABLE(get_max_with_row_offset(cursor(SELECT /*+ columnar_output "
      "*/ key FROM SQL_HINT_DUMMY)));";
  const auto q2 =
      "SELECT out0 FROM TABLE(get_max_with_row_offset(cursor(SELECT /*+ columnar_output, "
      "cpu_mode */ key FROM SQL_HINT_DUMMY)));";
  {
    auto query_hints = QR::get()->getParsedQueryHints(q1);
    EXPECT_TRUE(query_hints);
    EXPECT_EQ(query_hints->size(), static_cast<size_t>(1));
    EXPECT_TRUE(
        query_hints->begin()->second.isHintRegistered(QueryHint::kColumnarOutput));
  }
  {
    auto query_hints = QR::get()->getParsedQueryHints(q2);
    EXPECT_TRUE(query_hints);
    EXPECT_EQ(query_hints->size(), static_cast<size_t>(1));
    EXPECT_TRUE(
        query_hints->begin()->second.isHintRegistered(QueryHint::kColumnarOutput));
    EXPECT_TRUE(query_hints->begin()->second.isHintRegistered(QueryHint::kCpuMode));
  }
}

TEST(QueryHint, checkPerQueryBlockHint) {
  const auto enable_columnar_output = g_enable_columnar_output;
  g_enable_columnar_output = false;
  ScopeGuard reset_columnar_output = [&enable_columnar_output] {
    g_enable_columnar_output = enable_columnar_output;
  };

  const auto q1 =
      "SELECT /*+ cpu_mode */ T2.k FROM SQL_HINT_DUMMY T1, (SELECT /*+ columnar_output "
      "*/ key as k FROM SQL_HINT_DUMMY WHERE key = 1) T2 WHERE T1.key = T2.k;";
  const auto q2 =
      "SELECT /*+ cpu_mode */ out0 FROM TABLE(get_max_with_row_offset(cursor(SELECT /*+ "
      "columnar_output */ key FROM SQL_HINT_DUMMY)));";
  // to recognize query hint for a specific query block, we need more complex hint getter
  // func in QR but for test, it is enough to check the functionality in brute-force
  // manner
  auto check_registered_hint =
      [](std::unordered_map<size_t, RegisteredQueryHint>& hints) {
        bool find_columnar_hint = false;
        bool find_cpu_mode_hint = false;
        CHECK(hints.size() == static_cast<size_t>(2));
        for (auto& hint : hints) {
          if (hint.second.isHintRegistered(QueryHint::kColumnarOutput)) {
            find_columnar_hint = true;
            EXPECT_FALSE(hint.second.isHintRegistered(QueryHint::kCpuMode));
            continue;
          }
          if (hint.second.isHintRegistered(QueryHint::kCpuMode)) {
            find_cpu_mode_hint = true;
            EXPECT_FALSE(hint.second.isHintRegistered(QueryHint::kColumnarOutput));
            continue;
          }
        }
        EXPECT_TRUE(find_columnar_hint);
        EXPECT_TRUE(find_cpu_mode_hint);
      };
  {
    auto query_hints = QR::get()->getParsedQueryHints(q1);
    EXPECT_TRUE(query_hints);
    check_registered_hint(query_hints.value());
  }
  {
    auto query_hints = QR::get()->getParsedQueryHints(q2);
    EXPECT_TRUE(query_hints);
    check_registered_hint(query_hints.value());
  }
}

TEST(QueryHint, WindowFunction) {
  const auto enable_columnar_output = g_enable_columnar_output;
  g_enable_columnar_output = false;
  ScopeGuard reset_columnar_output = [&enable_columnar_output] {
    g_enable_columnar_output = enable_columnar_output;
  };

  const auto q1 =
      "SELECT /*+ columnar_output */ str1, timestampdiff(minute, lag(ts1) over "
      "(partition by str1 order by ts1), ts2) as m_el FROM SQL_HINT_DUMMY;";
  {
    auto query_hints = QR::get()->getParsedQueryHints(q1);
    EXPECT_TRUE(query_hints);
    for (auto& kv : *query_hints) {
      auto query_hint = kv.second;
      EXPECT_TRUE(query_hint.isHintRegistered(QueryHint::kColumnarOutput));
    }
  }
  const auto q2 =
      "SELECT /*+ columnar_output */ count(1) FROM (SELECT /*+ columnar_output */ str1, "
      "timestampdiff(minute, lag(ts1) over (partition by str1 order by ts1), ts2) as "
      "m_el FROM SQL_HINT_DUMMY) T1 WHERE T1.m_el < 30;";
  {
    auto query_hints = QR::get()->getParsedQueryHints(q2);
    EXPECT_TRUE(query_hints);
    for (auto& kv : *query_hints) {
      auto query_hint = kv.second;
      EXPECT_TRUE(query_hint.isHintRegistered(QueryHint::kColumnarOutput));
    }
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  g_enable_table_functions = true;
  QR::init(BASE_PATH);
  int err{0};

  try {
    createTable();
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  dropTable();
  QR::reset();
  return err;
}
