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
#include "DataMgr/DataMgr.h"
#include "QueryEngine/Execute.h"
#include "QueryRunner/QueryRunner.h"

namespace po = boost::program_options;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace Catalog_Namespace;

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

TEST(CPU_MODE, ForceToCPUMode) {
  const auto create_table_ddl = "CREATE TABLE SQL_HINT_DUMMY(key int)";
  const auto drop_table_ddl = "DROP TABLE IF EXISTS SQL_HINT_DUMMY";
  const auto query_with_cpu_mode_hint = "SELECT /*+ cpu_mode */ * FROM SQL_HINT_DUMMY";
  const auto query_without_cpu_mode_hint = "SELECT * FROM SQL_HINT_DUMMY";
  QR::get()->runDDLStatement(drop_table_ddl);
  QR::get()->runDDLStatement(create_table_ddl);
  if (QR::get()->gpusPresent()) {
    auto query_hints = QR::get()->getParsedQueryHint(query_with_cpu_mode_hint);
    const bool cpu_mode_enabled = query_hints.isHintRegistered("cpu_mode");
    CHECK(cpu_mode_enabled);
    query_hints = QR::get()->getParsedQueryHint(query_without_cpu_mode_hint);
    CHECK(!query_hints.isAnyQueryHintDelivered());
  }
  QR::get()->runDDLStatement(drop_table_ddl);
}

TEST(OVERLAPS_JOIN_PARAM, Check_Overlaps_Join_Hint) {
  const auto overlaps_join_status_backup = g_enable_overlaps_hashjoin;
  g_enable_overlaps_hashjoin = true;
  ScopeGuard reset_loop_join_state = [&overlaps_join_status_backup] {
    g_enable_overlaps_hashjoin = overlaps_join_status_backup;
  };

  const auto drop_table_ddl_1 = "DROP TABLE IF EXISTS geospatial_test";
  const auto drop_table_ddl_2 = "DROP TABLE IF EXISTS geospatial_inner_join_test";
  const auto create_table_ddl_1 =
      "CREATE TABLE geospatial_test(id INT, p POINT, l LINESTRING, poly POLYGON);";
  const auto create_table_ddl_2 =
      "CREATE TABLE geospatial_inner_join_test(id INT, p POINT, l LINESTRING, poly "
      "POLYGON);";

  QR::get()->runDDLStatement(drop_table_ddl_1);
  QR::get()->runDDLStatement(drop_table_ddl_2);
  QR::get()->runDDLStatement(create_table_ddl_1);
  QR::get()->runDDLStatement(create_table_ddl_2);

  ScopeGuard cleanup = [&] {
    QR::get()->runDDLStatement(drop_table_ddl_1);
    QR::get()->runDDLStatement(drop_table_ddl_2);
  };

  {
    const auto q1 =
        "SELECT /*+ overlaps_bucket_threshold(0.718) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q1_hints = QR::get()->getParsedQueryHint(q1);
    EXPECT_TRUE(q1_hints.isHintRegistered("overlaps_bucket_threshold") &&
                approx_eq(q1_hints.overlaps_bucket_threshold, 0.718));
  }
  {
    const auto q2 =
        "SELECT /*+ overlaps_max_size(2021) */ a.id FROM geospatial_test a INNER JOIN "
        "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q2_hints = QR::get()->getParsedQueryHint(q2);
    EXPECT_TRUE(q2_hints.isHintRegistered("overlaps_max_size") &&
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
    EXPECT_TRUE(q3_hints.isHintRegistered("overlaps_max_size") &&
                q3_hints.isHintRegistered("overlaps_bucket_threshold") &&
                (q3_hints.overlaps_max_size == 2021) &&
                approx_eq(q3_hints.overlaps_bucket_threshold, 0.718));
  }

  {
    const auto query =
        R"(SELECT /*+ overlaps_allow_gpu_build */ a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);)";
    const auto hints = QR::get()->getParsedQueryHint(query);
    EXPECT_TRUE(hints.isHintRegistered("overlaps_allow_gpu_build"));
    EXPECT_TRUE(hints.overlaps_allow_gpu_build);
  }
  {
    const auto q4 =
        "SELECT /*+ overlaps_bucket_threshold(0.1) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q1_hints = QR::get()->getParsedQueryHint(q4);
    EXPECT_TRUE(q1_hints.isHintRegistered("overlaps_bucket_threshold") &&
                approx_eq(q1_hints.overlaps_bucket_threshold, 0.1));
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
    EXPECT_TRUE(!wrong_q1_hints.isHintRegistered("overlaps_bucket_threshold"));
  }

  {
    const auto wrong_q2 =
        "SELECT /*+ overlaps_bucket_threshold(91.718) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto wrong_q2_hints = QR::get()->getParsedQueryHint(wrong_q2);
    EXPECT_TRUE(!wrong_q2_hints.isHintRegistered("overlaps_bucket_threshold"));
  }

  {
    const auto wrong_q3 =
        "SELECT /*+ overlaps_max_size(-2021) */ a.id FROM geospatial_test a INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto wrong_q3_hints = QR::get()->getParsedQueryHint(wrong_q3);
    EXPECT_TRUE(!wrong_q3_hints.isHintRegistered("overlaps_max_size"));
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

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
