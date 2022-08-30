/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include "QueryEngine/Execute.h"
#include "QueryEngine/QueryHint.h"
#include "QueryEngine/RelAlgDagSerializer/Serializer.h"
#include "QueryRunner/QueryRunner.h"

namespace po = boost::program_options;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace Catalog_Namespace;

extern bool g_enable_table_functions;
extern bool g_enable_test_table_functions;

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

constexpr double EPS = 1e-10;

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

std::shared_ptr<ResultSet> run_query(const std::string& query_str,
                                     const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, true, true);
}

bool is_hint_registered(
    std::unordered_map<size_t, std::unordered_map<unsigned, RegisteredQueryHint>>& hints,
    QueryHint expected_hint) {
  for (const auto& kv : hints) {
    for (auto& kv2 : kv.second) {
      if (kv2.second.isHintRegistered(expected_hint)) {
        return true;
      }
    }
  }
  return false;
}

bool is_hint_globally_registered(const RegisteredQueryHint& hints,
                                 QueryHint expected_hint) {
  return hints.isHintRegistered(expected_hint);
}

struct QueryHintInfo {
  QueryHint hint;
  bool is_global;
};

bool is_all_hints_are_registered(RelAlgDag* rel_alg_dag,
                                 const std::vector<QueryHintInfo>& expected_hints) {
  return std::all_of(expected_hints.begin(),
                     expected_hints.end(),
                     [&rel_alg_dag](const QueryHintInfo& hint_info) {
                       if (hint_info.is_global) {
                         return is_hint_globally_registered(rel_alg_dag->getGlobalHints(),
                                                            hint_info.hint);
                       } else {
                         return is_hint_registered(rel_alg_dag->getQueryHints(),
                                                   hint_info.hint);
                       }
                     });
}

bool check_serialized_rel_alg_dag(const std::string& query_str,
                                  const std::vector<QueryHintInfo>& expected_hints) {
  auto rel_alg_dag = QR::get()->getRelAlgDag(query_str);
  CHECK(rel_alg_dag);
  CHECK(is_all_hints_are_registered(rel_alg_dag.get(), expected_hints));
  const auto serialized_dag = Serializer::serializeRelAlgDag(*rel_alg_dag);
  auto restored_rel_alg_dag =
      Serializer::deserializeRelAlgDag(*QR::get()->getCatalog(), serialized_dag);
  CHECK(restored_rel_alg_dag);
  return is_all_hints_are_registered(restored_rel_alg_dag.get(), expected_hints);
}

std::pair<std::shared_ptr<HashTable>, std::optional<RegisteredQueryHint>>
getCachedHashTable(std::set<QueryPlanHash>& already_visited,
                   CacheItemType cache_item_type) {
  auto cached_ht = QR::get()->getCachedHashtableWithoutCacheKey(
      already_visited, cache_item_type, 0 /* CPU_DEVICE_IDENTIFIER*/);
  auto cache_key = std::get<0>(cached_ht);
  already_visited.insert(cache_key);
  return std::make_pair(std::get<1>(cached_ht),
                        std::get<2>(cached_ht)->registered_query_hint);
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
  QR::get()->runDDLStatement(
      "CREATE TABLE complex_windowing(str text encoding dict(32), ts timestamp(0), lat "
      "float, lon float);");
  QR::get()->runDDLStatement("CREATE TABLE subquery_test (x int, y int)");
  QR::get()->runDDLStatement("CREATE TABLE JOIN_HINT_TEST (v INT, v2 INT);");
}

void populateTable() {
  std::vector<std::string> geospatial_test_data{
      "0,'POINT (0 0)','LINESTRING (0 0,0 0)','POLYGON ((0 0,1 0,0 1,0 0))'",
      "1,'POINT (1 1)','LINESTRING (1 0,2 2,3 3)','POLYGON ((0 0,2 0,0 2,0 0))'",
      "2,'POINT (2 2)','LINESTRING (2 0,4 4)','POLYGON ((0 0,3 0,0 3,0 0))'",
      "3,'POINT (3 3)','LINESTRING (3 0,6 6,7 7)','POLYGON ((0 0,4 0,0 4,0 0))'",
      "4,'POINT (4 4)','LINESTRING (4 0,8 8)','POLYGON ((0 0,5 0,0 5,0 0))'",
      "5,'POINT (5 5)','LINESTRING (5 0,10 10,11 11)','POLYGON ((0 0,6 0,0 6,0 0))'",
      "6,'POINT (6 6)','LINESTRING (6 0,12 12)','POLYGON ((0 0,7 0,0 7,0 0))'",
      "7,'POINT (7 7)','LINESTRING (7 0,14 14,15 15)','POLYGON ((0 0,8 0,0 8,0 0))'",
      "8,'POINT (8 8)','LINESTRING (8 0,16 16)','POLYGON ((0 0,9 0,0 9,0 0))'",
      "9,'POINT (9 9)','LINESTRING (9 0,18 18,19 19)','POLYGON ((0 0,10 0,0 10,0 0))'"};

  for (const auto& data : geospatial_test_data) {
    const auto data_str = "INSERT INTO geospatial_test VALUES(" + data + ");";
    run_query(data_str, ExecutorDeviceType::CPU);
  }

  std::vector<std::string> geospatial_inner_test_data{
      "0,'POINT (0 0)','LINESTRING (0 0,0 0)','POLYGON ((0 0,1 0,0 1,0 0))'",
      "2,'POINT (2 2)','LINESTRING (2 0,4 4)','POLYGON ((0 0,3 0,0 3,0 0))'",
      "4,'POINT (4 4)','LINESTRING (4 0,8 8)','POLYGON ((0 0,5 0,0 5,0 0))'",
      "6,'POINT (6 6)','LINESTRING (6 0,12 12)','POLYGON ((0 0,7 0,0 7,0 0))'",
      "8,'POINT (8 8)','LINESTRING (8 0,16 16)','POLYGON ((0 0,9 0,0 9,0 0))'"};

  for (const auto& data : geospatial_inner_test_data) {
    const auto data_str = "INSERT INTO geospatial_inner_join_test VALUES(" + data + ");";
    run_query(data_str, ExecutorDeviceType::CPU);
  }

  std::vector<std::string> complex_windowing_test_data{
      "\'N712SW\',\'2008-01-03 22:11:00\',38.94453,-77.45581",
      "\'N772SW\',\'2008-01-03 10:02:00\',38.94453,-77.45581",
      "\'N428WN\',\'2008-01-03 08:04:00\',39.71733,-86.29439",
      "\'N612SW\',\'2008-01-03 10:54:00\',39.71733,-86.29439",
      "\'N689SW\',\'2008-01-03 06:52:00\',39.71733,-86.29439",
      "\'N648SW\',\'2008-01-03 16:39:00\',39.71733,-86.29439",
      "\'N690SW\',\'2008-01-03 09:16:00\',39.71733,-86.29439",
      "\'N334SW\',\'2008-01-03 18:45:00\',39.71733,-86.29439",
      "\'N286WN\',\'2008-01-03 16:40:00\',39.71733,-86.29439",
      "\'N778SW\',\'2008-01-03 09:40:00\',39.71733,-86.29439",
  };
  for (const auto& data : complex_windowing_test_data) {
    const auto data_str = "INSERT INTO complex_windowing VALUES(" + data + ");";
    run_query(data_str, ExecutorDeviceType::CPU);
  }

  run_query("INSERT INTO subquery_test VALUES (1, 1)", ExecutorDeviceType::CPU);

  for (int i = 1; i < 5; i++) {
    const auto val = ::toString(1);
    run_query("INSERT INTO JOIN_HINT_TEST VALUES (" + val + ", " + val + ");",
              ExecutorDeviceType::CPU);
  }
}

void dropTable() {
  QR::get()->runDDLStatement("DROP TABLE IF EXISTS SQL_HINT_DUMMY;");
  QR::get()->runDDLStatement("DROP TABLE IF EXISTS geospatial_test;");
  QR::get()->runDDLStatement("DROP TABLE IF EXISTS geospatial_inner_join_test;");
  QR::get()->runDDLStatement("DROP TABLE IF EXISTS complex_windowing;");
  QR::get()->runDDLStatement("DROP TABLE IF EXISTS subquery_test");
  QR::get()->runDDLStatement("DROP TABLE IF EXISTS JOIN_HINT_TEST;");
}

TEST(QueryHint, ForceToCPUMode) {
  const auto query_with_cpu_mode_hint = "SELECT /*+ cpu_mode */ * FROM SQL_HINT_DUMMY";
  const auto query_without_cpu_mode_hint = "SELECT * FROM SQL_HINT_DUMMY";
  if (QR::get()->gpusPresent()) {
    auto query_hints = QR::get()->getParsedQueryHint(query_with_cpu_mode_hint);
    const bool cpu_mode_enabled = query_hints.isHintRegistered(QueryHint::kCpuMode);
    EXPECT_TRUE(cpu_mode_enabled);
    query_hints = QR::get()->getParsedQueryHint(query_without_cpu_mode_hint);
    EXPECT_FALSE(query_hints.isAnyQueryHintDelivered());
  }
  EXPECT_TRUE(check_serialized_rel_alg_dag(query_with_cpu_mode_hint,
                                           {{QueryHint::kCpuMode, false}}));
}

TEST(QueryHint, QueryHintForOverlapsJoin) {
  ScopeGuard reset_loop_join_state = [orig_overlaps_hash_join =
                                          g_enable_overlaps_hashjoin] {
    g_enable_overlaps_hashjoin = orig_overlaps_hash_join;
  };
  g_enable_overlaps_hashjoin = true;

  {
    const auto q1 =
        "SELECT /*+ overlaps_bucket_threshold(0.718) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q1_hints = QR::get()->getParsedQueryHint(q1);
    EXPECT_TRUE(q1_hints.isHintRegistered(QueryHint::kOverlapsBucketThreshold));
    EXPECT_NEAR(0.718, q1_hints.overlaps_bucket_threshold, EPS * 0.718);
    EXPECT_TRUE(
        check_serialized_rel_alg_dag(q1, {{QueryHint::kOverlapsBucketThreshold, false}}));
  }
  {
    const auto q2 =
        "SELECT /*+ overlaps_max_size(2021) */ a.id FROM geospatial_test a INNER JOIN "
        "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q2_hints = QR::get()->getParsedQueryHint(q2);
    EXPECT_TRUE(q2_hints.isHintRegistered(QueryHint::kOverlapsMaxSize) &&
                q2_hints.overlaps_max_size == 2021);
    EXPECT_TRUE(check_serialized_rel_alg_dag(q2, {{QueryHint::kOverlapsMaxSize, false}}));
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
                q3_hints.overlaps_max_size == 2021);
    EXPECT_NEAR(0.718, q3_hints.overlaps_bucket_threshold, EPS * 0.718);
    EXPECT_TRUE(
        check_serialized_rel_alg_dag(q3,
                                     {{QueryHint::kOverlapsBucketThreshold, false},
                                      {QueryHint::kOverlapsMaxSize, false}}));
  }

  {
    const auto query =
        R"(SELECT /*+ overlaps_allow_gpu_build */ a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);)";
    const auto hints = QR::get()->getParsedQueryHint(query);
    EXPECT_TRUE(hints.isHintRegistered(QueryHint::kOverlapsAllowGpuBuild));
    EXPECT_TRUE(hints.overlaps_allow_gpu_build);
    EXPECT_TRUE(check_serialized_rel_alg_dag(
        query, {{QueryHint::kOverlapsAllowGpuBuild, false}}));
  }
  {
    const auto q4 =
        "SELECT /*+ overlaps_bucket_threshold(0.1) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q4_hints = QR::get()->getParsedQueryHint(q4);
    EXPECT_TRUE(q4_hints.isHintRegistered(QueryHint::kOverlapsBucketThreshold));
    EXPECT_NEAR(0.1, q4_hints.overlaps_bucket_threshold, EPS * 0.1);
  }
  {
    const auto q5 =
        "SELECT /*+ overlaps_keys_per_bin(0.1) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q5_hints = QR::get()->getParsedQueryHint(q5);
    EXPECT_TRUE(q5_hints.isHintRegistered(QueryHint::kOverlapsKeysPerBin));
    EXPECT_NEAR(0.1, q5_hints.overlaps_keys_per_bin, EPS * 0.1);
    EXPECT_TRUE(
        check_serialized_rel_alg_dag(q5, {{QueryHint::kOverlapsKeysPerBin, false}}));
  }
  {
    const auto q6 =
        "SELECT /*+ overlaps_keys_per_bin(19980909.01) */ a.id FROM geospatial_test a "
        "INNER "
        "JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p);";
    auto q6_hints = QR::get()->getParsedQueryHint(q6);
    EXPECT_TRUE(q6_hints.isHintRegistered(QueryHint::kOverlapsKeysPerBin));
    EXPECT_NEAR(19980909.01, q6_hints.overlaps_keys_per_bin, EPS * 19980909.01);
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

TEST(QueryHint, QueryLayoutHintWithEnablingColumnarOutput) {
  ScopeGuard reset_columnar_output = [orig_columnar_output = g_enable_columnar_output] {
    g_enable_columnar_output = orig_columnar_output;
  };
  g_enable_columnar_output = true;

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
    EXPECT_TRUE(check_serialized_rel_alg_dag(q2, {{QueryHint::kRowwiseOutput, false}}));
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
    EXPECT_TRUE(check_serialized_rel_alg_dag(q6, {{QueryHint::kRowwiseOutput, false}}));
  }

  {
    auto query_hints = QR::get()->getParsedQueryHint(q7);
    auto hint_enabled = query_hints.isHintRegistered(QueryHint::kColumnarOutput);
    EXPECT_FALSE(hint_enabled);
  }
}

TEST(QueryHint, QueryLayoutHintWithoutEnablingColumnarOutput) {
  ScopeGuard reset_columnar_output = [orig_columnar_output = g_enable_columnar_output] {
    g_enable_columnar_output = orig_columnar_output;
  };
  g_enable_columnar_output = false;

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
    EXPECT_TRUE(check_serialized_rel_alg_dag(q1, {{QueryHint::kColumnarOutput, false}}));
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
    EXPECT_TRUE(check_serialized_rel_alg_dag(q7, {{QueryHint::kColumnarOutput, false}}));
  }
}

TEST(QueryHint, UDF) {
  ScopeGuard reset_columnar_output = [orig_columnar_output = g_enable_columnar_output] {
    g_enable_columnar_output = orig_columnar_output;
  };
  g_enable_columnar_output = false;

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
    EXPECT_TRUE(query_hints->begin()->second.begin()->second.isHintRegistered(
        QueryHint::kColumnarOutput));
    EXPECT_TRUE(check_serialized_rel_alg_dag(q1, {{QueryHint::kColumnarOutput, false}}));
  }
  {
    auto query_hints = QR::get()->getParsedQueryHints(q2);
    EXPECT_TRUE(query_hints);
    EXPECT_EQ(query_hints->size(), static_cast<size_t>(1));
    EXPECT_TRUE(query_hints->begin()->second.begin()->second.isHintRegistered(
        QueryHint::kColumnarOutput));
    EXPECT_TRUE(query_hints->begin()->second.begin()->second.isHintRegistered(
        QueryHint::kCpuMode));
    EXPECT_TRUE(check_serialized_rel_alg_dag(
        q2, {{QueryHint::kCpuMode, false}, {QueryHint::kColumnarOutput, false}}));
  }
}

TEST(QueryHint, PerQueryBlockHint) {
  ScopeGuard reset_columnar_output = [orig_columnar_output = g_enable_columnar_output] {
    g_enable_columnar_output = orig_columnar_output;
  };
  g_enable_columnar_output = false;

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
      [](std::unordered_map<size_t, std::unordered_map<unsigned, RegisteredQueryHint>>&
             hints) {
        bool find_columnar_hint = false;
        bool find_cpu_mode_hint = false;
        CHECK(hints.size() == static_cast<size_t>(2));
        for (auto& kv : hints) {
          for (auto& kv2 : kv.second) {
            auto hint = kv2.second;
            if (hint.isHintRegistered(QueryHint::kColumnarOutput)) {
              find_columnar_hint = true;
              EXPECT_FALSE(hint.isHintRegistered(QueryHint::kCpuMode));
              continue;
            }
            if (hint.isHintRegistered(QueryHint::kCpuMode)) {
              find_cpu_mode_hint = true;
              EXPECT_FALSE(hint.isHintRegistered(QueryHint::kColumnarOutput));
              continue;
            }
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
  ScopeGuard reset_columnar_output = [orig_columnar_output = g_enable_columnar_output] {
    g_enable_columnar_output = orig_columnar_output;
  };
  g_enable_columnar_output = false;

  const auto q1 =
      "SELECT /*+ columnar_output */ str1, timestampdiff(minute, lag(ts1) over "
      "(partition by str1 order by ts1), ts2) as m_el FROM SQL_HINT_DUMMY;";
  {
    auto query_hints = QR::get()->getParsedQueryHints(q1);
    EXPECT_TRUE(query_hints);
    for (auto& kv : *query_hints) {
      auto& query_hint_map = kv.second;
      for (auto& kv2 : query_hint_map) {
        auto query_hint = kv2.second;
        EXPECT_TRUE(query_hint.isHintRegistered(QueryHint::kColumnarOutput));
      }
    }
    EXPECT_TRUE(check_serialized_rel_alg_dag(q1, {{QueryHint::kColumnarOutput, false}}));
  }
  const auto q2 =
      "SELECT /*+ columnar_output */ count(1) FROM (SELECT /*+ columnar_output */ str1, "
      "timestampdiff(minute, lag(ts1) over (partition by str1 order by ts1), ts2) as "
      "m_el FROM SQL_HINT_DUMMY) T1 WHERE T1.m_el < 30;";
  {
    auto query_hints = QR::get()->getParsedQueryHints(q2);
    EXPECT_TRUE(query_hints);
    for (auto& kv : *query_hints) {
      auto& query_hint_map = kv.second;
      for (auto& kv2 : query_hint_map) {
        auto query_hint = kv2.second;
        EXPECT_TRUE(query_hint.isHintRegistered(QueryHint::kColumnarOutput));
      }
    }
    EXPECT_TRUE(check_serialized_rel_alg_dag(q2, {{QueryHint::kColumnarOutput, false}}));
  }
  const auto q3 =
      "select /*+ columnar_output */ *, 1 * v1 / (v2 + 0.01) as v3 from (select /*+ "
      "cpu_mode */ str, ts, lat, lon, distance_in_meters( lag(lon) over ( partition by "
      "str order by ts ), lag(lat) over ( partition by str order by ts ), lon, lat ) as "
      "v1, timestampdiff( second, lag(ts) over ( partition by str order by ts ), ts ) as "
      "v2 from complex_windowing) order by v3;";
  EXPECT_EQ(QR::get()->runSQL(q3, ExecutorDeviceType::CPU)->colCount(),
            static_cast<size_t>(7));
  const auto q4 =
      "select /*+ g_cpu_mode */ *, 1 * v1 / (v2 + 0.01) as v3 from (select str, ts, lat, "
      "lon, distance_in_meters( lag(lon) over ( partition by str order by ts ), lag(lat) "
      "over ( partition by str order by ts ), lon, lat ) as v1, timestampdiff( second, "
      "lag(ts) over ( partition by str order by ts ), ts ) as v2 from complex_windowing) "
      "order by v3;";
  EXPECT_EQ(QR::get()->runSQL(q4, ExecutorDeviceType::CPU)->colCount(),
            static_cast<size_t>(7));
  const auto q5 =
      "select /*+ cpu_mode */ *, 1 * v1 / (v2 + 0.01) as v3 from (select str, ts, lat, "
      "lon, distance_in_meters( lag(lon) over ( partition by str order by ts ), lag(lat) "
      "over ( partition by str order by ts ), lon, lat ) as v1, timestampdiff( second, "
      "lag(ts) over ( partition by str order by ts ), ts ) as v2 from complex_windowing) "
      "order by v3;";
  EXPECT_EQ(QR::get()->runSQL(q5, ExecutorDeviceType::CPU)->colCount(),
            static_cast<size_t>(7));
}

TEST(QueryHint, GlobalHint_OverlapsJoinHashtable) {
  ScopeGuard reset_loop_join_state = [orig_overlaps_hash_join =
                                          g_enable_overlaps_hashjoin] {
    g_enable_overlaps_hashjoin = orig_overlaps_hash_join;
  };
  g_enable_overlaps_hashjoin = true;

  // testing global query hint for overlaps join is tricky since we apply all registered
  // hint during hashtable building time, so it's hard to get the result at that time
  // instead by exploiting cached hashtable we can check whether hints are registered &
  // applied correctly in indirect manner

  // q1 and q2: global query hint registered to the main query block
  const auto q1 =
      "SELECT /*+ g_overlaps_no_cache */ t1.ID FROM (SELECT a.id FROM geospatial_test a "
      "INNER JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p)) T1;";
  {
    auto res = run_query(q1, ExecutorDeviceType::CPU);
    auto numCachedOverlapsHashTable = QR::get()->getNumberOfCachedItem(
        QueryRunner::CacheItemStatus::ALL, CacheItemType::OVERLAPS_HT);
    EXPECT_EQ(numCachedOverlapsHashTable, static_cast<size_t>(0));
    EXPECT_TRUE(check_serialized_rel_alg_dag(q1, {{QueryHint::kOverlapsNoCache, true}}));
  }

  if (QR::get()->gpusPresent()) {
    const auto q2 =
        "SELECT /*+ g_overlaps_allow_gpu_build */ t1.ID FROM (SELECT a.id FROM "
        "geospatial_test a INNER JOIN geospatial_inner_join_test b ON "
        "ST_Contains(b.poly, a.p)) T1;";
    auto res = run_query(q2, ExecutorDeviceType::GPU);
    auto numCachedOverlapsHashTable = QR::get()->getNumberOfCachedItem(
        QueryRunner::CacheItemStatus::ALL, CacheItemType::OVERLAPS_HT);
    EXPECT_EQ(numCachedOverlapsHashTable, static_cast<size_t>(0));
    EXPECT_TRUE(
        check_serialized_rel_alg_dag(q2, {{QueryHint::kOverlapsAllowGpuBuild, true}}));
  }

  // q3 and q4: two (e.g., multiple) subqueries and we disallow to put hashtable to cache
  // for one of them so we should have just a single overlaps join hashtable with
  // registered global hint
  std::set<QueryPlanHash> visited_hashtable_key;
  const auto q3 =
      "SELECT /*+ g_overlaps_max_size(7777) */ t1.ID, t2.ID FROM \n"
      "(SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON "
      "ST_Contains(b.poly, a.p)) T1, \n"
      "(SELECT /*+ overlaps_no_cache */ a.id FROM geospatial_test a INNER JOIN "
      "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p)) T2 \n"
      "WHERE t1.ID = t2.ID;";
  {
    auto res = run_query(q3, ExecutorDeviceType::CPU);
    auto cached_ht_info =
        getCachedHashTable(visited_hashtable_key, CacheItemType::OVERLAPS_HT);
    auto query_hint = cached_ht_info.second;
    EXPECT_TRUE(query_hint.has_value());
    EXPECT_EQ(query_hint->overlaps_max_size, static_cast<size_t>(7777));
    auto numCachedOverlapsHashTable = QR::get()->getNumberOfCachedItem(
        QueryRunner::CacheItemStatus::ALL, CacheItemType::OVERLAPS_HT);
    EXPECT_EQ(numCachedOverlapsHashTable, static_cast<size_t>(1));
    QR::get()->clearCpuMemory();
    visited_hashtable_key.clear();
    EXPECT_TRUE(check_serialized_rel_alg_dag(
        q3, {{QueryHint::kOverlapsNoCache, false}, {QueryHint::kOverlapsMaxSize, true}}));
  }

  if (QR::get()->gpusPresent()) {
    const auto q4 =
        "SELECT /*+ g_overlaps_bucket_threshold(0.718) */ t1.ID, t2.ID FROM \n"
        "(SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test b ON "
        "ST_Contains(b.poly, a.p)) T1,\n"
        "(SELECT /*+ overlaps_allow_gpu_build */ a.id FROM geospatial_test a INNER JOIN "
        "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p)) T2\n"
        "WHERE t1.ID = t2.ID;";
    auto res = run_query(q4, ExecutorDeviceType::GPU);
    auto cached_ht_info =
        getCachedHashTable(visited_hashtable_key, CacheItemType::OVERLAPS_HT);
    auto query_hint = cached_ht_info.second;
    EXPECT_TRUE(query_hint.has_value());
    EXPECT_NEAR(0.718, query_hint->overlaps_bucket_threshold, EPS * 0.718);
    auto numCachedOverlapsHashTable = QR::get()->getNumberOfCachedItem(
        QueryRunner::CacheItemStatus::ALL, CacheItemType::OVERLAPS_HT);
    EXPECT_EQ(numCachedOverlapsHashTable, static_cast<size_t>(1));
    QR::get()->clearCpuMemory();
    visited_hashtable_key.clear();
    EXPECT_TRUE(
        check_serialized_rel_alg_dag(q4,
                                     {{QueryHint::kOverlapsAllowGpuBuild, false},
                                      {QueryHint::kOverlapsBucketThreshold, true}}));
  }

  // q5, q6 and q7: a subquery block which is allowed to interact with hashtable cache
  // should have the info related to both global and local query hint(s)
  const auto q5 =
      "SELECT /*+ g_overlaps_keys_per_bin(0.1) */ t1.ID, t2.ID FROM \n"
      "(SELECT /*+ overlaps_max_size(7777) */ a.id FROM geospatial_test a INNER JOIN "
      "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p)) T1,\n"
      "(SELECT /*+ overlaps_no_cache */ a.id FROM geospatial_test a INNER JOIN "
      "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p)) T2\n"
      "WHERE t1.ID = t2.ID;";
  {
    auto res = run_query(q5, ExecutorDeviceType::CPU);
    auto cached_ht_info =
        getCachedHashTable(visited_hashtable_key, CacheItemType::OVERLAPS_HT);
    auto query_hint = cached_ht_info.second;
    EXPECT_TRUE(query_hint.has_value());
    EXPECT_NEAR(0.1, query_hint->overlaps_keys_per_bin, EPS * 0.1);
    EXPECT_EQ(query_hint->overlaps_max_size, static_cast<size_t>(7777));
    auto numCachedOverlapsHashTable = QR::get()->getNumberOfCachedItem(
        QueryRunner::CacheItemStatus::ALL, CacheItemType::OVERLAPS_HT);
    EXPECT_EQ(numCachedOverlapsHashTable, static_cast<size_t>(1));
    QR::get()->clearCpuMemory();
    visited_hashtable_key.clear();
    EXPECT_TRUE(check_serialized_rel_alg_dag(q5,
                                             {{QueryHint::kOverlapsNoCache, false},
                                              {QueryHint::kOverlapsKeysPerBin, true},
                                              {QueryHint::kOverlapsMaxSize, false}}));
  }

  const auto q6 =
      "SELECT /*+ g_overlaps_keys_per_bin(0.1) */ t1.ID, t2.ID FROM \n"
      "(SELECT /*+ g_overlaps_bucket_threshold(0.718) */ a.id FROM geospatial_test a "
      "INNER JOIN geospatial_inner_join_test b ON ST_Contains(b.poly, a.p)) T1,\n"
      "(SELECT /*+ overlaps_no_cache */ a.id FROM geospatial_test a INNER JOIN "
      "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p)) T2\n"
      "WHERE t1.ID = t2.ID;";
  {
    auto res = run_query(q6, ExecutorDeviceType::CPU);
    auto cached_ht_info =
        getCachedHashTable(visited_hashtable_key, CacheItemType::OVERLAPS_HT);
    auto query_hint = cached_ht_info.second;
    EXPECT_TRUE(query_hint.has_value());
    EXPECT_NEAR(0.1, query_hint->overlaps_keys_per_bin, EPS * 0.1);
    EXPECT_NEAR(0.718, query_hint->overlaps_bucket_threshold, EPS * 0.718);
    auto numCachedOverlapsHashTable = QR::get()->getNumberOfCachedItem(
        QueryRunner::CacheItemStatus::ALL, CacheItemType::OVERLAPS_HT);
    EXPECT_EQ(numCachedOverlapsHashTable, static_cast<size_t>(1));
    QR::get()->clearCpuMemory();
    visited_hashtable_key.clear();
    EXPECT_TRUE(
        check_serialized_rel_alg_dag(q6,
                                     {{QueryHint::kOverlapsNoCache, false},
                                      {QueryHint::kOverlapsKeysPerBin, true},
                                      {QueryHint::kOverlapsBucketThreshold, true}}));
  }

  const auto q7 =
      "SELECT /*+ g_overlaps_max_size(7777) */ t1.ID, t2.ID FROM \n"
      "(SELECT /*+ overlaps_keys_per_bin(0.1) */ a.id FROM geospatial_test a INNER JOIN "
      "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p)) T1,\n"
      "(SELECT /*+ overlaps_no_cache */ a.id FROM geospatial_test a INNER JOIN "
      "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p)) T2\n"
      "WHERE t1.ID = t2.ID;";
  {
    auto res = run_query(q7, ExecutorDeviceType::CPU);
    auto cached_ht_info =
        getCachedHashTable(visited_hashtable_key, CacheItemType::OVERLAPS_HT);
    auto query_hint = cached_ht_info.second;
    EXPECT_TRUE(query_hint.has_value());
    EXPECT_NEAR(0.1, query_hint->overlaps_keys_per_bin, EPS * 0.1);
    EXPECT_EQ(query_hint->overlaps_max_size, static_cast<size_t>(7777));
    auto numCachedOverlapsHashTable = QR::get()->getNumberOfCachedItem(
        QueryRunner::CacheItemStatus::ALL, CacheItemType::OVERLAPS_HT);
    EXPECT_EQ(numCachedOverlapsHashTable, static_cast<size_t>(1));
    QR::get()->clearCpuMemory();
    visited_hashtable_key.clear();
    EXPECT_TRUE(check_serialized_rel_alg_dag(q7,
                                             {{QueryHint::kOverlapsNoCache, false},
                                              {QueryHint::kOverlapsKeysPerBin, false},
                                              {QueryHint::kOverlapsMaxSize, true}}));
  }
}

TEST(QueryHint, GlobalHint_ResultsetLayoutAndCPUMode) {
  ScopeGuard reset_columnar_output = [orig_columnar_output = g_enable_columnar_output] {
    g_enable_columnar_output = orig_columnar_output;
  };
  g_enable_columnar_output = false;

  // check whether we can see the enabled global hint in the outer query block
  const auto q1 =
      "SELECT T2.k FROM SQL_HINT_DUMMY T1, (SELECT /*+ g_cpu_mode */ key as k FROM "
      "SQL_HINT_DUMMY WHERE key = 1) T2 WHERE T1.key = T2.k;";
  {
    auto global_query_hints = QR::get()->getParsedGlobalQueryHints(q1);
    CHECK(global_query_hints);
    EXPECT_TRUE(global_query_hints->isHintRegistered(QueryHint::kCpuMode));
    EXPECT_TRUE(check_serialized_rel_alg_dag(q1, {{QueryHint::kCpuMode, true}}));
  }

  // check whether inner query has enabled cpu hint
  const auto q2 =
      "SELECT /*+ g_cpu_mode */ T2.k FROM SQL_HINT_DUMMY T1, (SELECT key as k FROM "
      "SQL_HINT_DUMMY WHERE key = 1) T2 WHERE T1.key = T2.k;";
  {
    auto global_query_hints = QR::get()->getParsedGlobalQueryHints(q2);
    EXPECT_TRUE(global_query_hints);
    EXPECT_TRUE(global_query_hints->isHintRegistered(QueryHint::kCpuMode));
    EXPECT_TRUE(check_serialized_rel_alg_dag(q2, {{QueryHint::kCpuMode, true}}));
  }

  // check whether we can see not only cpu mode hint but also global columnar output hint
  const auto q3 =
      "SELECT /*+ cpu_mode */ out0 FROM TABLE(get_max_with_row_offset(cursor(SELECT /*+ "
      "g_columnar_output */ key FROM SQL_HINT_DUMMY)));";
  {
    auto query_hints = QR::get()->getParsedQueryHints(q3);
    EXPECT_TRUE(query_hints);
    bool found_local_hint = false;
    for (auto& kv : *query_hints) {
      for (auto& kv2 : kv.second) {
        auto& hint = kv2.second;
        if (hint.isAnyQueryHintDelivered()) {
          if (hint.isHintRegistered(QueryHint::kCpuMode)) {
            found_local_hint = true;
          }
        }
      }
    }
    EXPECT_TRUE(found_local_hint);
    auto global_query_hints = QR::get()->getParsedGlobalQueryHints(q3);
    EXPECT_TRUE(global_query_hints);
    EXPECT_TRUE(global_query_hints->isHintRegistered(QueryHint::kColumnarOutput));
    EXPECT_TRUE(check_serialized_rel_alg_dag(
        q3, {{QueryHint::kCpuMode, false}, {QueryHint::kColumnarOutput, true}}));
  }

  const auto q4 =
      "SELECT /*+ columnar_output */ out0 FROM "
      "TABLE(get_max_with_row_offset(cursor(SELECT /*+ g_rowwise_output */ key FROM "
      "SQL_HINT_DUMMY)));";
  {
    auto query_hints = QR::get()->getParsedQueryHints(q4);
    EXPECT_TRUE(query_hints);
    bool found_local_hint = false;
    for (auto& kv : *query_hints) {
      for (auto& kv2 : kv.second) {
        auto& hint = kv2.second;
        if (hint.isAnyQueryHintDelivered()) {
          if (hint.isHintRegistered(QueryHint::kColumnarOutput)) {
            found_local_hint = true;
          }
        }
      }
    }
    EXPECT_TRUE(found_local_hint);
    auto global_query_hints = QR::get()->getParsedGlobalQueryHints(q4);
    EXPECT_TRUE(global_query_hints);
    EXPECT_FALSE(global_query_hints->isHintRegistered(QueryHint::kRowwiseOutput));
    EXPECT_TRUE(check_serialized_rel_alg_dag(q4, {{QueryHint::kColumnarOutput, false}}));
  }

  // we disable columnar output so rowwise global hint is ignored too
  // thus we expect to see the enabled global columnar output hint
  const auto q5 =
      "SELECT /*+ g_rowwise_output */ out0 FROM "
      "TABLE(get_max_with_row_offset(cursor(SELECT /*+ g_columnar_output */ key FROM "
      "SQL_HINT_DUMMY)));";
  {
    auto query_hints = QR::get()->getParsedQueryHints(q5);
    EXPECT_TRUE(query_hints);
    bool columnar_enabled_local = false;
    bool rowwise_enabled_local = false;
    for (auto& kv : *query_hints) {
      for (auto& kv2 : kv.second) {
        auto& hint = kv2.second;
        if (hint.isAnyQueryHintDelivered()) {
          if (hint.isHintRegistered(QueryHint::kColumnarOutput)) {
            columnar_enabled_local = true;
          }
          if (hint.isHintRegistered(QueryHint::kRowwiseOutput)) {
            rowwise_enabled_local = true;
          }
        }
      }
    }
    EXPECT_TRUE(columnar_enabled_local);
    EXPECT_FALSE(rowwise_enabled_local);
    auto global_query_hints = QR::get()->getParsedGlobalQueryHints(q5);
    EXPECT_TRUE(global_query_hints);
    EXPECT_TRUE(global_query_hints->isHintRegistered(QueryHint::kColumnarOutput));
    EXPECT_FALSE(global_query_hints->isHintRegistered(QueryHint::kRowwiseOutput));
    EXPECT_TRUE(check_serialized_rel_alg_dag(q5, {{QueryHint::kColumnarOutput, true}}));
  }

  ScopeGuard reset_resultset_recycler_state =
      [orig_data_recycler = g_enable_data_recycler,
       orig_resultset_recycler = g_use_query_resultset_cache] {
        g_enable_data_recycler = orig_data_recycler,
        g_use_query_resultset_cache = orig_resultset_recycler;
      };
  g_enable_data_recycler = true;
  g_use_query_resultset_cache = true;

  // check the resultset hint for table function
  const auto q6 =
      "SELECT /*+ keep_table_function_result */ out0 FROM "
      "TABLE(get_max_with_row_offset(cursor(SELECT key FROM SQL_HINT_DUMMY)));";
  {
    auto global_query_hints = QR::get()->getParsedGlobalQueryHints(q6);
    EXPECT_TRUE(global_query_hints);
    EXPECT_TRUE(global_query_hints->isHintRegistered(QueryHint::kKeepTableFuncResult));
    EXPECT_TRUE(
        check_serialized_rel_alg_dag(q6, {{QueryHint::kKeepTableFuncResult, true}}));
  }

  // check the resultset hint for table function
  const auto q7 =
      "SELECT out0 FROM TABLE(get_max_with_row_offset(cursor(SELECT /*+ "
      "keep_table_function_result */ key FROM SQL_HINT_DUMMY)));";
  {
    auto global_query_hints = QR::get()->getParsedGlobalQueryHints(q7);
    EXPECT_TRUE(global_query_hints);
    EXPECT_TRUE(global_query_hints->isHintRegistered(QueryHint::kKeepTableFuncResult));
    EXPECT_TRUE(
        check_serialized_rel_alg_dag(q7, {{QueryHint::kKeepTableFuncResult, true}}));
  }
}

TEST(QueryHint, CudaBlockAndGridSizes) {
  if (QR::get()->gpusPresent()) {
    const auto unitary_executor = QR::get()->getExecutor();
    const auto default_grid_size = unitary_executor->gridSize();
    const auto default_block_size = unitary_executor->blockSize();
    auto query_gen = [](bool is_block_size_hint, std::string val) {
      std::string hint_name =
          is_block_size_hint ? "cuda_block_size" : "cuda_grid_size_multiplier";
      auto hint = hint_name + "(" + val + ")";
      return "SELECT /*+ " + hint + " */ COUNT(1) FROM SQL_HINT_DUMMY";
    };

    auto query_hint1 = QR::get()->getParsedQueryHint(query_gen(true, "10"));
    EXPECT_TRUE(query_hint1.isHintRegistered(QueryHint::kCudaBlockSize));
    EXPECT_TRUE(check_serialized_rel_alg_dag(query_gen(true, "10"),
                                             {{QueryHint::kCudaBlockSize, false}}));
    auto res1 = QR::get()->runSQL(query_gen(true, "10"), ExecutorDeviceType::GPU);
    CHECK_EQ(res1->getBlockSize(), (unsigned)10);
    res1 = QR::get()->runSQL(query_gen(true, "31"), ExecutorDeviceType::GPU);
    CHECK_EQ(res1->getBlockSize(), (unsigned)31);
    res1 = QR::get()->runSQL(query_gen(true, "32"), ExecutorDeviceType::GPU);
    CHECK_EQ(res1->getBlockSize(), (unsigned)32);
    res1 = QR::get()->runSQL(query_gen(true, "33"), ExecutorDeviceType::GPU);
    CHECK_EQ(res1->getBlockSize(), (unsigned)64);
    // check whether we use default grid and block sizes if no query hint is provided
    res1 = QR::get()->runSQL("SELECT * FROM SQL_HINT_DUMMY", ExecutorDeviceType::GPU);
    CHECK_EQ(res1->getBlockSize(), default_block_size);
    CHECK_EQ(res1->getGridSize(), default_grid_size);

    auto query_hint2 = QR::get()->getParsedQueryHint(query_gen(true, "-10"));
    EXPECT_TRUE(!query_hint2.isHintRegistered(QueryHint::kCudaBlockSize));
    auto query_hint3 = QR::get()->getParsedQueryHint(query_gen(true, "1.11"));
    EXPECT_TRUE(query_hint3.isHintRegistered(QueryHint::kCudaBlockSize));
    auto res2 = QR::get()->runSQL(query_gen(true, "1.11"), ExecutorDeviceType::GPU);
    CHECK_EQ(res2->getBlockSize(), (unsigned)1);
    auto query_hint4 = QR::get()->getParsedQueryHint(query_gen(true, "0"));
    EXPECT_TRUE(!query_hint4.isHintRegistered(QueryHint::kCudaBlockSize));
    auto query_hint5 = QR::get()->getParsedQueryHint(query_gen(true, "1026"));
    EXPECT_TRUE(!query_hint5.isHintRegistered(QueryHint::kCudaBlockSize));

    auto query_hint6 = QR::get()->getParsedQueryHint(query_gen(false, "4.44"));
    EXPECT_TRUE(query_hint6.isHintRegistered(QueryHint::kCudaGridSize));
    EXPECT_TRUE(check_serialized_rel_alg_dag(query_gen(false, "4.44"),
                                             {{QueryHint::kCudaGridSize, false}}));
    auto res3 = QR::get()->runSQL(query_gen(false, "4.44"), ExecutorDeviceType::GPU);
    CHECK_NE(res3->getGridSize(), default_grid_size);
    auto query_hint7 = QR::get()->getParsedQueryHint(query_gen(false, "-4.44"));
    EXPECT_TRUE(!query_hint7.isHintRegistered(QueryHint::kCudaGridSize));
    auto query_hint8 = QR::get()->getParsedQueryHint(query_gen(false, "0"));
    EXPECT_TRUE(!query_hint8.isHintRegistered(QueryHint::kCudaGridSize));
    auto query_hint9 = QR::get()->getParsedQueryHint(query_gen(false, "1026"));
    EXPECT_TRUE(!query_hint9.isHintRegistered(QueryHint::kCudaGridSize));

    auto query_hint10 = QR::get()->getParsedQueryHint(
        "SELECT /*+ cuda_opt_block_and_grid_sizes */ COUNT(1) FROM SQL_HINT_DUMMY");
    EXPECT_TRUE(query_hint10.isHintRegistered(QueryHint::kOptCudaBlockAndGridSizes));
    EXPECT_TRUE(check_serialized_rel_alg_dag(
        "SELECT /*+ cuda_opt_block_and_grid_sizes */ COUNT(1) FROM SQL_HINT_DUMMY",
        {{QueryHint::kOptCudaBlockAndGridSizes, false}}));

    auto query_hint11 = QR::get()->getParsedGlobalQueryHints(
        "SELECT /*+ g_cuda_opt_block_and_grid_sizes */ COUNT(1) FROM SQL_HINT_DUMMY");
    EXPECT_TRUE(query_hint11->isHintRegistered(QueryHint::kOptCudaBlockAndGridSizes));
    EXPECT_TRUE(check_serialized_rel_alg_dag(
        "SELECT /*+ g_cuda_opt_block_and_grid_sizes */ COUNT(1) FROM SQL_HINT_DUMMY",
        {{QueryHint::kOptCudaBlockAndGridSizes, true}}));

    auto query_hint12 = QR::get()->getParsedQueryHint(
        "SELECT /*+ cuda_opt_block_and_grid_sizes, cuda_block_size(512) */ COUNT(1) FROM "
        "SQL_HINT_DUMMY");
    EXPECT_TRUE(query_hint12.isHintRegistered(QueryHint::kOptCudaBlockAndGridSizes));
    EXPECT_TRUE(query_hint12.isHintRegistered(QueryHint::kCudaBlockSize));
    EXPECT_TRUE(check_serialized_rel_alg_dag(
        "SELECT /*+ cuda_opt_block_and_grid_sizes, cuda_block_size(512) */ COUNT(1) FROM "
        "SQL_HINT_DUMMY",
        {{QueryHint::kCudaBlockSize, false},
         {QueryHint::kOptCudaBlockAndGridSizes, false}}));

    // grid_size should be greater or equal to one regardless of query hint value
    auto res4 = QR::get()->runSQL(query_gen(false, "0.000001"), ExecutorDeviceType::GPU);
    EXPECT_EQ(res4->getGridSize(), (unsigned)1);
    auto num_sms =
        QR::get()->getCatalog()->getDataMgr().getCudaMgr()->getMinNumMPsForAllDevices();
    EXPECT_GE(std::ceil(res4->getGridSize() / (double)num_sms), (unsigned)1);
  }
}

TEST(QueryHint, Watchdog) {
  ScopeGuard reset_flag = [orig = g_enable_watchdog] { g_enable_watchdog = orig; };
  const auto watchdog_on_query = "SELECT /*+ watchdog */ * FROM SQL_HINT_DUMMY";
  const auto watchdog_off_query = "SELECT /*+ watchdog_off */ * FROM SQL_HINT_DUMMY";
  {
    g_enable_watchdog = false;
    auto watchdog_on = QR::get()->getParsedQueryHint(watchdog_on_query);
    EXPECT_TRUE(watchdog_on.isHintRegistered(QueryHint::kWatchdog));
    EXPECT_TRUE(
        check_serialized_rel_alg_dag(watchdog_on_query, {{QueryHint::kWatchdog, false}}));
    auto watchdog_off = QR::get()->getParsedQueryHint(watchdog_off_query);
    EXPECT_TRUE(!watchdog_off.isHintRegistered(QueryHint::kWatchdogOff));
  }
  {
    g_enable_watchdog = true;
    auto watchdog_on = QR::get()->getParsedQueryHint(watchdog_on_query);
    EXPECT_TRUE(!watchdog_on.isHintRegistered(QueryHint::kWatchdog));
    auto watchdog_off = QR::get()->getParsedQueryHint(watchdog_off_query);
    EXPECT_TRUE(watchdog_off.isHintRegistered(QueryHint::kWatchdogOff));
    EXPECT_TRUE(check_serialized_rel_alg_dag(watchdog_off_query,
                                             {{QueryHint::kWatchdogOff, false}}));
  }
}

TEST(QueryHint, DynamicWatchdog) {
  ScopeGuard reset_flag = [orig_flag = g_enable_dynamic_watchdog,
                           orig_val = g_dynamic_watchdog_time_limit] {
    g_enable_dynamic_watchdog = orig_flag;
    g_dynamic_watchdog_time_limit = orig_val;
  };
  const auto dynamic_watchdog_on_query =
      "SELECT /*+ dynamic_watchdog */ * FROM SQL_HINT_DUMMY";
  const auto dynamic_watchdog_on_with_timelimit_query =
      "SELECT /*+ dynamic_watchdog, query_time_limit(1000) */ * FROM SQL_HINT_DUMMY";
  const auto dynamic_watchdog_off_query =
      "SELECT /*+ dynamic_watchdog_off */ * FROM SQL_HINT_DUMMY";
  const auto timelimit_query =
      "SELECT /*+ query_time_limit(1000) */ * FROM SQL_HINT_DUMMY";
  const auto wrong_timelimit_query =
      "SELECT /*+ query_time_limit(-1000) */ * FROM SQL_HINT_DUMMY";
  const auto wrong_timelimit_query2 =
      "SELECT /*+ query_time_limit(0) */ * FROM SQL_HINT_DUMMY";
  {
    g_enable_dynamic_watchdog = false;
    auto watchdog_on = QR::get()->getParsedQueryHint(dynamic_watchdog_on_query);
    EXPECT_TRUE(watchdog_on.isHintRegistered(QueryHint::kDynamicWatchdog));
    EXPECT_TRUE(check_serialized_rel_alg_dag(dynamic_watchdog_on_query,
                                             {{QueryHint::kDynamicWatchdog, false}}));
    auto watchdog_off = QR::get()->getParsedQueryHint(dynamic_watchdog_off_query);
    EXPECT_TRUE(!watchdog_off.isHintRegistered(QueryHint::kDynamicWatchdogOff));
    auto dynamic_watchdog_on_with_timelimit =
        QR::get()->getParsedQueryHint(dynamic_watchdog_on_with_timelimit_query);
    EXPECT_TRUE(
        dynamic_watchdog_on_with_timelimit.isHintRegistered(QueryHint::kDynamicWatchdog));
    EXPECT_TRUE(
        dynamic_watchdog_on_with_timelimit.isHintRegistered(QueryHint::kQueryTimeLimit));
    EXPECT_TRUE(check_serialized_rel_alg_dag(
        dynamic_watchdog_on_with_timelimit_query,
        {{QueryHint::kDynamicWatchdog, false}, {QueryHint::kQueryTimeLimit, false}}));
    auto timelimit = QR::get()->getParsedQueryHint(timelimit_query);
    EXPECT_TRUE(timelimit.isHintRegistered(QueryHint::kQueryTimeLimit));
    auto wrong_timelimit1 = QR::get()->getParsedQueryHint(wrong_timelimit_query);
    EXPECT_TRUE(!wrong_timelimit1.isHintRegistered(QueryHint::kQueryTimeLimit));
    auto wrong_timelimit2 = QR::get()->getParsedQueryHint(wrong_timelimit_query2);
    EXPECT_TRUE(!wrong_timelimit2.isHintRegistered(QueryHint::kQueryTimeLimit));
  }
  {
    g_enable_dynamic_watchdog = true;
    auto watchdog_on = QR::get()->getParsedQueryHint(dynamic_watchdog_on_query);
    EXPECT_TRUE(!watchdog_on.isHintRegistered(QueryHint::kDynamicWatchdog));
    auto watchdog_off = QR::get()->getParsedQueryHint(dynamic_watchdog_off_query);
    EXPECT_TRUE(watchdog_off.isHintRegistered(QueryHint::kDynamicWatchdogOff));
    EXPECT_TRUE(check_serialized_rel_alg_dag(dynamic_watchdog_off_query,
                                             {{QueryHint::kDynamicWatchdogOff, false}}));
    auto dynamic_watchdog_on_with_timelimit =
        QR::get()->getParsedQueryHint(dynamic_watchdog_on_with_timelimit_query);
    EXPECT_TRUE(!dynamic_watchdog_on_with_timelimit.isHintRegistered(
        QueryHint::kDynamicWatchdog));
    EXPECT_TRUE(
        dynamic_watchdog_on_with_timelimit.isHintRegistered(QueryHint::kQueryTimeLimit));
    auto timelimit = QR::get()->getParsedQueryHint(timelimit_query);
    EXPECT_TRUE(timelimit.isHintRegistered(QueryHint::kQueryTimeLimit));
    auto wrong_timelimit1 = QR::get()->getParsedQueryHint(wrong_timelimit_query);
    EXPECT_TRUE(!wrong_timelimit1.isHintRegistered(QueryHint::kQueryTimeLimit));
    auto wrong_timelimit2 = QR::get()->getParsedQueryHint(wrong_timelimit_query2);
    EXPECT_TRUE(!wrong_timelimit2.isHintRegistered(QueryHint::kQueryTimeLimit));
  }
}

TEST(QueryHint, LoopJoin) {
  ScopeGuard reset_flag = [orig_flag = g_trivial_loop_join_threshold] {
    g_trivial_loop_join_threshold = orig_flag;
  };
  const auto loop_join_on_query =
      "SELECT /*+ allow_loop_join */ COUNT(1) FROM JOIN_HINT_TEST R, JOIN_HINT_TEST S;";
  auto loop_join_on = QR::get()->getParsedQueryHint(loop_join_on_query);
  EXPECT_TRUE(loop_join_on.isHintRegistered(QueryHint::kAllowLoopJoin));
  EXPECT_NO_THROW(
      QR::get()->runSQL(loop_join_on_query, ExecutorDeviceType::CPU, false, false));

  const auto loop_join_off_query =
      "SELECT /*+ disable_loop_join */ COUNT(1) FROM JOIN_HINT_TEST R, JOIN_HINT_TEST S;";
  auto loop_join_off = QR::get()->getParsedQueryHint(loop_join_off_query);
  EXPECT_TRUE(loop_join_off.isHintRegistered(QueryHint::kDisableLoopJoin));
  EXPECT_ANY_THROW(
      QR::get()->runSQL(loop_join_off_query, ExecutorDeviceType::CPU, false, true));

  const auto loop_join_limit_query =
      "SELECT /*+ loop_join_inner_table_max_num_rows(3) */ COUNT(1) FROM JOIN_HINT_TEST "
      "R, JOIN_HINT_TEST S;";
  auto loop_join_limit = QR::get()->getParsedQueryHint(loop_join_limit_query);
  EXPECT_TRUE(loop_join_limit.isHintRegistered(QueryHint::kLoopJoinInnerTableMaxNumRows));
  EXPECT_ANY_THROW(
      QR::get()->runSQL(loop_join_limit_query, ExecutorDeviceType::CPU, false, true));

  const auto loop_join_limit_query2 =
      "SELECT /*+ loop_join_inner_table_max_num_rows(999999) */ COUNT(1) FROM "
      "JOIN_HINT_TEST "
      "R, JOIN_HINT_TEST S;";
  auto loop_join_limit2 = QR::get()->getParsedQueryHint(loop_join_limit_query2);
  EXPECT_TRUE(
      loop_join_limit2.isHintRegistered(QueryHint::kLoopJoinInnerTableMaxNumRows));
  EXPECT_NO_THROW(
      QR::get()->runSQL(loop_join_limit_query2, ExecutorDeviceType::CPU, false, true));

  const auto wrong_query_hint_query =
      "SELECT /*+ loop_join_inner_table_max_num_rows(-1) */ COUNT(1) FROM JOIN_HINT_TEST "
      "R, JOIN_HINT_TEST S;";
  auto wrong_query_hint = QR::get()->getParsedQueryHint(wrong_query_hint_query);
  EXPECT_TRUE(
      !wrong_query_hint.isHintRegistered(QueryHint::kLoopJoinInnerTableMaxNumRows));
}

TEST(QueryHint, JoinHashTableSize) {
  const auto perfect_hash_limit_query =
      "SELECT /*+ max_join_hashtable_size(1) */ COUNT(1) FROM JOIN_HINT_TEST R, "
      "JOIN_HINT_TEST S WHERE R.v = S.v;";
  auto perfect_hash_limit = QR::get()->getParsedQueryHint(perfect_hash_limit_query);
  EXPECT_TRUE(perfect_hash_limit.isHintRegistered(QueryHint::kMaxJoinHashTableSize));
  EXPECT_ANY_THROW(
      QR::get()->runSQL(perfect_hash_limit_query, ExecutorDeviceType::CPU, false, true));

  const auto baseline_hash_limit_query =
      "SELECT /*+ max_join_hashtable_size(1) */ COUNT(1) FROM JOIN_HINT_TEST R, "
      "JOIN_HINT_TEST S WHERE R.v = S.v AND R.v2 = S.v2;";
  auto baseline_hash_limit = QR::get()->getParsedQueryHint(baseline_hash_limit_query);
  EXPECT_TRUE(baseline_hash_limit.isHintRegistered(QueryHint::kMaxJoinHashTableSize));
  EXPECT_ANY_THROW(
      QR::get()->runSQL(baseline_hash_limit_query, ExecutorDeviceType::CPU, false, true));

  const auto overlaps_join_limit_query =
      "SELECT /*+ max_join_hashtable_size(1) */ COUNT(1) FROM geospatial_test R, "
      "geospatial_test S WHERE ST_DISTANCE(R.p, S.p) < 0.1;";
  auto overlaps_join_limit = QR::get()->getParsedQueryHint(overlaps_join_limit_query);
  EXPECT_TRUE(overlaps_join_limit.isHintRegistered(QueryHint::kMaxJoinHashTableSize));
  EXPECT_ANY_THROW(
      QR::get()->runSQL(overlaps_join_limit_query, ExecutorDeviceType::CPU, false, true));

  const auto wrong_query_hint_query =
      "SELECT /*+ max_join_hashtable_size(-1) */ COUNT(1) FROM JOIN_HINT_TEST R, "
      "JOIN_HINT_TEST S WHERE R.v = S.v;";
  auto wrong_query_hint = QR::get()->getParsedQueryHint(wrong_query_hint_query);
  EXPECT_TRUE(!wrong_query_hint.isHintRegistered(QueryHint::kMaxJoinHashTableSize));
}

TEST(QueryHint, Subquery) {
  {
    std::string q1 =
        "select /*+ g_cpu_mode */ count(1) from subquery_test where x in (select x from "
        "subquery_test "
        "where y = (select y from subquery_test where rowid = 1));";
    const auto rel_alg_dag = QR::get()->getRelAlgDag(q1);
    EXPECT_TRUE(rel_alg_dag->getGlobalHints().isAnyQueryHintDelivered());
  }
  {
    std::string q2 =
        "select count(1) from subquery_test where x in (select /*+ g_cpu_mode */ x from "
        "subquery_test "
        "where y = (select y from subquery_test where rowid = 1));";
    const auto rel_alg_dag = QR::get()->getRelAlgDag(q2);
    EXPECT_TRUE(rel_alg_dag->getGlobalHints().isAnyQueryHintDelivered());
  }
  {
    std::string q3 =
        "select count(1) from subquery_test where x in (select x from subquery_test "
        "where y = (select /*+ "
        "g_cpu_mode */ y from subquery_test where rowid = 1));";
    const auto rel_alg_dag = QR::get()->getRelAlgDag(q3);
    EXPECT_TRUE(rel_alg_dag->getGlobalHints().isAnyQueryHintDelivered());
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  g_enable_table_functions = true;
  g_enable_dev_table_functions = true;
  QR::init(BASE_PATH);
  int err{0};

  try {
    dropTable();
    createTable();
    populateTable();
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  dropTable();
  QR::reset();
  return err;
}
