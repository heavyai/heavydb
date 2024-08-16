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

#include "Logger/Logger.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/DataRecycler/HashtableRecycler.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/QueryPlanDagCache.h"
#include "QueryEngine/QueryPlanDagExtractor.h"
#include "QueryEngine/Visitors/SQLOperatorDetector.h"
#include "QueryRunner/QueryRunner.h"

#include <gtest/gtest.h>
#include <boost/algorithm/string/join.hpp>

#include <exception>
#include <future>
#include <random>
#include <stdexcept>

extern bool g_is_test_env;
extern bool g_enable_table_functions;
extern bool g_enable_dev_table_functions;

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

bool g_cpu_only{false};

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

namespace {

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

TargetValue run_simple_query(const std::string& query_str,
                             const ExecutorDeviceType device_type,
                             const bool geo_return_geo_tv = true,
                             const bool allow_loop_joins = true) {
  auto rows = QR::get()->runSQL(query_str, device_type, allow_loop_joins);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
}

void drop_tables_for_bbox_intersect() {
  const auto cleanup_stmts = {R"(drop table if exists bbox_intersect_t11;)",
                              R"(drop table if exists bbox_intersect_t12;)",
                              R"(drop table if exists bbox_intersect_t13;)",
                              R"(drop table if exists bbox_intersect_t2;)",
                              R"(drop table if exists bbox_intersect_t3;)",
                              R"(drop table if exists bbox_intersect_t4;)"};

  for (const auto& stmt : cleanup_stmts) {
    QR::get()->runDDLStatement(stmt);
  }
}

void create_table_for_bbox_intersect() {
  const auto init_stmts_ddl = {
      R"(create table bbox_intersect_t11 (id int,
                        poly geometry(polygon, 4326),
                        mpoly geometry(multipolygon, 4326),
                        pt geometry(point, 4326));
    )",
      R"(create table bbox_intersect_t12 (id int,
                        poly geometry(polygon, 4326),
                        mpoly geometry(multipolygon, 4326),
                        pt geometry(point, 4326));
    )",
      R"(create table bbox_intersect_t13 (id int,
                        poly geometry(polygon, 4326),
                        mpoly geometry(multipolygon, 4326),
                        pt geometry(point, 4326));
    )",
      R"(create table bbox_intersect_t2 (id int,
                        poly geometry(polygon, 4326),
                        mpoly geometry(multipolygon, 4326),
                        pt geometry(point, 4326));
    )",
      R"(create table bbox_intersect_t3 (id int,
                        poly geometry(polygon, 4326),
                        mpoly geometry(multipolygon, 4326),
                        pt geometry(point, 4326));
    )",
      R"(create table bbox_intersect_t4 (id int,
                        poly geometry(polygon, 4326),
                        mpoly geometry(multipolygon, 4326),
                        pt geometry(point, 4326));
    )"};

  for (const auto& stmt : init_stmts_ddl) {
    QR::get()->runDDLStatement(stmt);
  }
}

void insert_dml_for_bbox_intersect() {
  std::string value_str =
      "(0,'polygon((20 20,30 25,30 30,25 30,20 20))','multipolygon(((20 20,30 25,30 "
      "30,25 30,20 2)))','point(22 22)');";
  std::string bbox_intersect_val1 =
      " values (0,'polygon((20 20,30 25,30 30,25 30,20 "
      "20))','multipolygon(((20 20,30 25,30 30,25 30,20 2)))','point(22 22)')";
  std::string bbox_intersect_val2 =
      " values (1,'polygon((2 2,10 2,10 10,2 10,2 2))', "
      "'multipolygon(((2 2,10 2,10 10,2 10,2 2)))', 'point(8 8)')";
  auto insert_stmt = [&value_str](const std::string& tbl_name) {
    return "INSERT INTO " + tbl_name + " VALUES " + value_str;
  };
  std::vector<std::string> tbl_names1{
      "bbox_intersect_t11", "bbox_intersect_t12", "bbox_intersect_t13"};
  std::vector<std::string> tbl_names2{
      "bbox_intersect_t2", "bbox_intersect_t3", "bbox_intersect_t4"};
  for (const std::string& tbl_name : tbl_names1) {
    QR::get()->runSQL(insert_stmt(tbl_name), ExecutorDeviceType::CPU);
  }
  for (const std::string& tbl_name : tbl_names2) {
    QR::get()->runSQL("insert into " + tbl_name + bbox_intersect_val1,
                      ExecutorDeviceType::CPU);
    QR::get()->runSQL("insert into " + tbl_name + bbox_intersect_val2,
                      ExecutorDeviceType::CPU);
  }
}

int drop_table() {
  try {
    run_ddl_statement("DROP TABLE IF EXISTS T1;");
    run_ddl_statement("DROP TABLE IF EXISTS T2;");
    run_ddl_statement("DROP TABLE IF EXISTS T3;");
    run_ddl_statement("DROP TABLE IF EXISTS T4;");
    run_ddl_statement("DROP TABLE IF EXISTS TF_TEST;");
    drop_tables_for_bbox_intersect();
  } catch (...) {
    LOG(ERROR) << "Failed to drop table";
    return -1;
  }
  return 0;
}

int create_and_populate_table() {
  try {
    drop_table();
    const auto table_ddl = "(x int, y int, z text encoding dict);";
    auto create_table_ddl = [&table_ddl](const std::string& tbl_name) {
      return "CREATE TABLE " + tbl_name + table_ddl;
    };
    run_ddl_statement(create_table_ddl("T1"));
    run_ddl_statement(create_table_ddl("T2"));
    run_ddl_statement(create_table_ddl("T3"));
    run_ddl_statement(create_table_ddl("T4"));
    const auto data_insertion = [](const std::string& tbl_name) {
      auto insert_dml = "INSERT INTO " + tbl_name + " VALUES(";
      std::vector<std::string> value_vec = {"1, 1, '1'", "2, 1, '2'", "3, 1, '3'"};
      for (auto& v : value_vec) {
        QR::get()->runSQL(insert_dml + v + ");", ExecutorDeviceType::CPU);
      }
    };
    data_insertion("T1");
    data_insertion("T2");
    QR::get()->runSQL("INSERT INTO T2 VALUES(4,2,'4');", ExecutorDeviceType::CPU);
    data_insertion("T3");
    QR::get()->runSQL("INSERT INTO T3 VALUES(4,2,'4');", ExecutorDeviceType::CPU);
    QR::get()->runSQL("INSERT INTO T3 VALUES(5,2,'5');", ExecutorDeviceType::CPU);
    data_insertion("T4");
    QR::get()->runSQL("INSERT INTO T4 VALUES(4,2,'4');", ExecutorDeviceType::CPU);
    QR::get()->runSQL("INSERT INTO T4 VALUES(5,2,'5');", ExecutorDeviceType::CPU);
    QR::get()->runSQL("INSERT INTO T4 VALUES(6,2,'6');", ExecutorDeviceType::CPU);

    run_ddl_statement("CREATE TABLE TF_TEST(d DOUBLE, d2 DOUBLE);");
    auto insert_dml = "INSERT INTO TF_TEST VALUES(";
    for (int i = 0; i < 20; i++) {
      double dv = i + (0.1 * i);
      auto v = std::to_string(dv);
      QR::get()->runSQL(insert_dml + v + ", " + v + ");", ExecutorDeviceType::CPU);
    }

    create_table_for_bbox_intersect();
    insert_dml_for_bbox_intersect();

  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table";
    return -1;
  }
  return 0;
}

std::shared_ptr<CacheItemMetric> getCachedHashTableMetric(
    std::set<QueryPlanHash>& already_visited,
    CacheItemType cache_item_type) {
  auto cached_ht = QR::get()->getCachedHashtableWithoutCacheKey(
      already_visited, cache_item_type, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  auto cache_key = std::get<0>(cached_ht);
  already_visited.insert(cache_key);
  return QR::get()->getCacheItemMetric(
      cache_key, cache_item_type, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
}

struct BBoxIntersectCachedHTAndMetaInfo {
  QueryPlanHash key;
  std::shared_ptr<HashTable> cached_ht;
  std::optional<HashtableCacheMetaInfo> cached_ht_metainfo;
  std::shared_ptr<CacheItemMetric> cached_metric;
  std::optional<AutoTunerMetaInfo> cached_tuning_info;
};

BBoxIntersectCachedHTAndMetaInfo
getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
    std::set<QueryPlanHash>& already_visited) {
  auto cached_ht = QR::get()->getCachedHashtableWithoutCacheKey(
      already_visited,
      CacheItemType::BBOX_INTERSECT_HT,
      DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  auto cache_key = std::get<0>(cached_ht);
  already_visited.insert(cache_key);
  auto ht_metric = QR::get()->getCacheItemMetric(cache_key,
                                                 CacheItemType::BBOX_INTERSECT_HT,
                                                 DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  auto tuning_param_cache =
      BoundingBoxIntersectJoinHashTable::getBoundingBoxIntersectTuningParamCache();
  auto tuning_param =
      tuning_param_cache->getItemFromCache(cache_key,
                                           CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM,
                                           DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  return {
      cache_key, std::get<1>(cached_ht), std::get<2>(cached_ht), ht_metric, tuning_param};
}

}  // namespace

TEST(DataRecycler, QueryPlanDagExtractor_Simple_Project_Query) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  auto q1_str = "SELECT x FROM T1 ORDER BY x;";
  auto q1_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q1_str);
  EXPECT_TRUE(q1_query_info.left_deep_trees_id.empty());
  auto q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(), executor);
  // 1. a sort node becomes a root (dag_rel_id = 0)
  // 2. a project node becomes a child of the sort node (dag_rel_id = 1)
  // 3. a scan node (the leaf of the query plan) becomes a child of the project node
  EXPECT_TRUE(q1_plan_dag.extracted_dag.compare("0|1|2|") == 0);

  auto q2_str = "SELECT x FROM T1;";
  auto q2_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q2_str);
  EXPECT_TRUE(q2_query_info.left_deep_trees_id.empty());
  auto q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(), executor);
  // q2 is the same as q1 except sort node
  EXPECT_TRUE(q2_plan_dag.extracted_dag.compare("1|2|") == 0);

  auto q3_str = "SELECT x FROM T1 GROUP BY x;";
  auto q3_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q3_str);
  EXPECT_TRUE(q3_query_info.left_deep_trees_id.empty());
  auto q3_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q3_query_info.root_node.get(), executor);
  // compound node becomes the root (dag_rel_id = 3), and the scan node
  // (that is the same node as both q1 and q2) is the leaf of the query plan
  EXPECT_TRUE(q3_plan_dag.extracted_dag.compare("3|2|") == 0);

  auto q4_str = "SELECT x FROM T1 GROUP BY x ORDER BY x;";
  auto q4_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q4_str);
  EXPECT_TRUE(q4_query_info.left_deep_trees_id.empty());
  auto q4_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q4_query_info.root_node.get(), executor);
  // this sort node has different input compared with that of q1
  // so we assign the new dag_rel_id (4) to the sort node
  EXPECT_TRUE(q4_plan_dag.extracted_dag.compare("4|3|2|") == 0);

  auto q1_dup_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(), executor);
  EXPECT_TRUE(q1_dup_plan_dag.extracted_dag.compare("0|1|2|") == 0);

  auto q4_dup_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q4_query_info.root_node.get(), executor);
  EXPECT_TRUE(q4_dup_plan_dag.extracted_dag.compare("4|3|2|") == 0);
}

TEST(DataRecycler, QueryPlanDagExtractor_Heavy_IN_clause) {
  // we do not extract query plan dag where at least one rel node
  // containing a heavy IN-expr w.r.t its value list, i.e., |value list| > 20
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();

  auto create_query_having_IN_expr = [](const std::string tbl_name,
                                        const std::string agg_col_name,
                                        const int value_list_size) {
    // assume value_list_size > 0
    std::vector<std::string> value_list;
    for (int i = 0; i < value_list_size; i++) {
      value_list.push_back(std::to_string(i));
    }
    return "SELECT COUNT(" + agg_col_name + ") FROM " + tbl_name + " WHERE " +
           agg_col_name + " IN (" + boost::algorithm::join(value_list, ",") + ");";
  };

  // so we can extract q1's dag
  auto q1_str = create_query_having_IN_expr("T1", "x", 20);
  auto q1_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q1_str);
  EXPECT_TRUE(q1_query_info.left_deep_trees_id.empty());
  auto q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(), executor);
  EXPECT_EQ(q1_plan_dag.contain_not_supported_rel_node, false);
  // but we skip to extract a DAG for q2 since it contains IN-expr having 21 elems in its
  // value list

  auto q2_str = create_query_having_IN_expr("T1", "x", 21);
  auto q2_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q2_str);
  EXPECT_TRUE(q2_query_info.left_deep_trees_id.empty());
  auto q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(), executor);
  EXPECT_EQ(q2_plan_dag.contain_not_supported_rel_node, true);
}

TEST(DataRecycler, QueryPlanDagExtractor_Join_Query) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();

  auto q1_str = "SELECT T1.x FROM T1, T2 WHERE T1.x = T2.x;";
  auto q1_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q1_str);
  EXPECT_TRUE(q1_query_info.left_deep_trees_id.size() == 1);
  auto q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(), executor);

  auto q2_str = "SELECT T1.x FROM T1 JOIN T2 ON T1.x = T2.x;";
  auto q2_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q2_str);
  EXPECT_TRUE(q2_query_info.left_deep_trees_id.size() == 1);
  auto q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(), executor);

  EXPECT_TRUE(q1_plan_dag.extracted_dag.compare(q2_plan_dag.extracted_dag) != 0);

  auto q3_str = "SELECT T1.x FROM T1, T2 WHERE T1.x = T2.x and T2.y = T1.y;";
  auto q3_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q3_str);
  EXPECT_TRUE(q3_query_info.left_deep_trees_id.size() == 1);
  auto q3_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q3_query_info.root_node.get(), executor);

  auto q4_str = "SELECT T1.x FROM T1 JOIN T2 ON T1.x = T2.x and T1.y = T2.y;";
  auto q4_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q4_str);
  EXPECT_TRUE(q4_query_info.left_deep_trees_id.size() == 1);
  auto q4_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q4_query_info.root_node.get(), executor);

  EXPECT_TRUE(q3_plan_dag.extracted_dag.compare(q4_plan_dag.extracted_dag) != 0);

  auto q5_str = "SELECT T1.x FROM T1 JOIN T2 ON T1.y = T2.y and T1.x = T2.x;";
  auto q5_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q5_str);
  EXPECT_TRUE(q5_query_info.left_deep_trees_id.size() == 1);
  auto q5_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q5_query_info.root_node.get(), executor);
  EXPECT_TRUE(q3_plan_dag.extracted_dag.compare(q5_plan_dag.extracted_dag) != 0);

  std::unordered_set<std::string> query_plan_dag_hash;
  std::vector<std::string> queries;
  queries.emplace_back(
      "SELECT COUNT(1) FROM T1 LEFT JOIN T2 ON T1.y <= T2.y WHERE T1.x = T2.x");
  queries.emplace_back(
      "SELECT COUNT(1) FROM T1 LEFT JOIN T2 ON T1.y = T2.y WHERE T1.x = T2.x");
  queries.emplace_back(
      "SELECT COUNT(1) FROM T1 INNER JOIN T2 ON T1.y = T2.y WHERE T1.x = T2.x");
  queries.emplace_back(
      "SELECT COUNT(1) FROM T1 INNER JOIN T2 ON T1.y <= T2.y WHERE T1.x = T2.x");
  for (const auto& sql : queries) {
    auto query_info = QR::get()->getQueryInfoForDataRecyclerTest(sql);
    auto dag =
        QueryPlanDagExtractor::extractQueryPlanDag(query_info.root_node.get(), executor);
    query_plan_dag_hash.insert(dag.extracted_dag);
  }
  // check whether we correctly extract query plan DAG for outer join having loop-join
  EXPECT_EQ(query_plan_dag_hash.size(), queries.size());
}

TEST(DataRecycler, QueryPlanDagExtractor_TableFunction) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();

  auto q1_str = "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM TF_TEST), 1));";
  auto res1 = QR::get()->runSQL(q1_str, ExecutorDeviceType::CPU, false, false);
  auto q1_plan_dag = executor->getLatestQueryPlanDagExtracted();
  EXPECT_TRUE(q1_plan_dag.compare(EMPTY_QUERY_PLAN) != 0);

  auto q2_str = "SELECT out0 FROM TABLE(row_copier(cursor(SELECT d FROM TF_TEST), 2));";
  auto res2 = QR::get()->runSQL(q2_str, ExecutorDeviceType::CPU, false, false);
  auto q2_plan_dag = executor->getLatestQueryPlanDagExtracted();
  EXPECT_TRUE(q2_plan_dag.compare(EMPTY_QUERY_PLAN) != 0);
  EXPECT_TRUE(q2_plan_dag.compare(q1_plan_dag) != 0);

  auto q3_str =
      "SELECT out0 FROM TABLE(row_adder(1, cursor(SELECT d, d2 FROM TF_TEST)));";
  auto res3 = QR::get()->runSQL(q3_str, ExecutorDeviceType::CPU, false, false);
  auto q3_plan_dag = executor->getLatestQueryPlanDagExtracted();
  EXPECT_TRUE(q3_plan_dag.compare(EMPTY_QUERY_PLAN) != 0);
  EXPECT_TRUE(q3_plan_dag.compare(q2_plan_dag) != 0);

  auto q4_str =
      "SELECT out0, out1 FROM TABLE(row_addsub(1, cursor(SELECT d, d2 FROM TF_TEST)))";
  auto res4 = QR::get()->runSQL(q4_str, ExecutorDeviceType::CPU, false, false);
  auto q4_plan_dag = executor->getLatestQueryPlanDagExtracted();
  EXPECT_TRUE(q4_plan_dag.compare(EMPTY_QUERY_PLAN) != 0);
  EXPECT_TRUE(q4_plan_dag.compare(q3_plan_dag) != 0);

  auto q5_str = q1_str;
  auto res5 = QR::get()->runSQL(q5_str, ExecutorDeviceType::CPU, false, false);
  auto q5_plan_dag = executor->getLatestQueryPlanDagExtracted();
  EXPECT_TRUE(q5_plan_dag.compare(EMPTY_QUERY_PLAN) != 0);
  EXPECT_TRUE(q5_plan_dag.compare(q1_plan_dag) == 0);
}

TEST(DataRecycler, Update_QueryPlanDagHash_After_DeadColumnElimination) {
  auto drop_tables = []() {
    run_ddl_statement("DROP TABLE IF EXISTS flights_qe1179;");
    run_ddl_statement("DROP TABLE IF EXISTS airports_qe1179;");
    run_ddl_statement("DROP TABLE IF EXISTS usa_states_qe1179;");
  };
  ScopeGuard drop_table_after_test = [drop_tables]() { drop_tables(); };
  drop_tables();
  run_ddl_statement(
      "CREATE TABLE flights_qe1179 (\n"
      "  carrier_name TEXT ENCODING DICT(8),\n"
      "  tail_num TEXT ENCODING DICT(32),\n"
      "  origin_airport_id SMALLINT);");
  run_ddl_statement(
      "CREATE TABLE airports_qe1179 (\n"
      "  airport_id SMALLINT,\n"
      "  airport_state_name TEXT ENCODING DICT(32));");
  run_ddl_statement(
      "CREATE TABLE usa_states_qe1179 (\n"
      "  STATE_NAME TEXT ENCODING DICT(32),\n"
      "  geom GEOMETRY(MULTIPOLYGON, 4326) ENCODING COMPRESSED(32));");
  std::string common_subq{
      "SELECT T1.tail_num, T1.carrier_name, T2.airport_state_name FROM flights_qe1179 AS "
      "T1 LEFT JOIN airports_qe1179 AS T2 ON T1.origin_airport_id = T2.airport_id"};
  std::string q1_where{
      "WHERE ((airport_state_name in (SELECT STATE_NAME from usa_states_qe1179 WHERE "
      "ST_XMax(geom) >= -172.976569702))) GROUP BY dimension0 ORDER BY measure0 desc "
      "NULLS LAST LIMIT 500;"};
  std::string q2_where{
      "WHERE ((airport_state_name in (SELECT STATE_NAME from usa_states_qe1179 WHERE "
      "ST_XMax(geom) >= -172.976569702)));"};
  std::string q1_project{"count(*) AS measure0, carrier_name AS dimension0"};
  std::string q2_project{"APPROX_COUNT_DISTINCT(tail_num) AS val"};
  std::ostringstream q1_oss;
  q1_oss << "SELECT " << q1_project << " FROM (" << common_subq << ") " << q1_where;
  std::ostringstream q2_oss;
  q2_oss << "SELECT " << q2_project << " FROM (" << common_subq << ") " << q2_where;
  auto executor = QR::get()->getExecutor().get();
  auto q1_ra_dag = QR::get()->getRelAlgDag(q1_oss.str());
  auto q1_ed_seq = RaExecutionSequence(&q1_ra_dag->getRootNode(), executor, false);
  CHECK_EQ(q1_ed_seq.size(), static_cast<size_t>(3));
  auto q1_project_hash_val = q1_ed_seq.getDescriptor(1)->getBody()->toHash();
  ASSERT_NE(q1_project_hash_val, EMPTY_HASHED_PLAN_DAG_KEY);
  auto q2_ra_dag = QR::get()->getRelAlgDag(q2_oss.str());
  auto q2_ed_seq = RaExecutionSequence(&q2_ra_dag->getRootNode(), executor, false);
  CHECK_EQ(q2_ed_seq.size(), static_cast<size_t>(3));
  auto q2_project_hash_val = q2_ed_seq.getDescriptor(1)->getBody()->toHash();
  ASSERT_NE(q1_project_hash_val, EMPTY_HASHED_PLAN_DAG_KEY);
  ASSERT_NE(q1_project_hash_val, q2_project_hash_val);
}

TEST(DataRecycler, DAG_Cache_Size_Management) {
  // test if DAG cache becomes full
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  // get query info for DAG cache test in advance
  auto& DAG_CACHE = executor->getQueryPlanDagCache();

  auto original_DAG_cache_max_size = MAX_NODE_CACHE_SIZE;
  ScopeGuard reset_dag_state = [&original_DAG_cache_max_size, &DAG_CACHE] {
    DAG_CACHE.setNodeMapMaxSize(original_DAG_cache_max_size);
  };

  auto q1_str = "SELECT x FROM T1 ORDER BY x;";
  auto q2_str = "SELECT y FROM T1;";
  auto q3_str =
      "SELECT T2.y, COUNT(T1.x) FROM T1, T2 WHERE T1.y = T2.y and T1.x = T2.x GROUP BY "
      "T2.y;";
  auto q1_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q1_str);
  auto q2_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q2_str);
  auto q3_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q3_str);
  DAG_CACHE.clearQueryPlanCache();

  // test: when DAG cache becomes full, it should skip the following query and clear the
  // cached plan
  DAG_CACHE.setNodeMapMaxSize(48);
  auto q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(), executor);
  // 1. a sort node becomes a root (dag_rel_id = 0)
  // 2. a project node becomes a child of the sort node (dag_rel_id = 1)
  // 3. a scan node (the leaf of the query plan) becomes a child of the project node
  EXPECT_TRUE(q1_plan_dag.extracted_dag.compare("0|1|2|") == 0);
  // 3 unique REL nodes in the cache --> 3 * 2 * 8 = 48
  EXPECT_EQ(DAG_CACHE.getCurrentNodeMapSize(), 48u);
  auto q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(), executor);
  // we set the DAG cache size be 48, so when we try to cache the q2, it becomes full
  // so it skips to extract DAG plan of this query and also clear the cache itself
  EXPECT_TRUE(q2_plan_dag.extracted_dag.compare("") == 0);
  EXPECT_EQ(DAG_CACHE.getCurrentNodeMapSize(), 0u);
  DAG_CACHE.clearQueryPlanCache();

  // test: when a query size is too large, we skip caching the query
  auto q3_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q3_query_info.root_node.get(), executor);
  // q3 has more than three nodes, so its size is beyond the limit of the DAG cache (48)
  // so we cannot keep it to our DAG cache
  EXPECT_EQ(DAG_CACHE.getCurrentNodeMapSize(), 0u);

  // test: increase the cache size that is enough to hold both q1 and q2
  DAG_CACHE.setNodeMapMaxSize(80);
  auto new_q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(), executor);
  EXPECT_TRUE(new_q1_plan_dag.extracted_dag.compare("0|1|2|") == 0);
  auto new_q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(), executor);
  EXPECT_TRUE(new_q2_plan_dag.extracted_dag.compare("3|2|") == 0);
  EXPECT_GE(DAG_CACHE.getCurrentNodeMapSize(), 48u);
}

TEST(DataRecycler, Hashtable_For_BBox_Intersect_Cache_Maintanence) {
  ScopeGuard reset_state =
      [orig_bbox_hashjoin_state = g_enable_bbox_intersect_hashjoin,
       orig_hashjoin_many_to_many_state = g_enable_hashjoin_many_to_many,
       orig_trivial_loop_join_threshold = g_trivial_loop_join_threshold,
       orig_table_reordering_state = g_from_table_reordering] {
        g_enable_bbox_intersect_hashjoin = orig_bbox_hashjoin_state;
        g_enable_hashjoin_many_to_many = orig_hashjoin_many_to_many_state;
        g_trivial_loop_join_threshold = orig_trivial_loop_join_threshold;
        g_from_table_reordering = orig_table_reordering_state;
      };
  g_enable_bbox_intersect_hashjoin = true;
  g_enable_hashjoin_many_to_many = true;
  g_trivial_loop_join_threshold = 1;
  // we need to disable table reordering to control our logic
  g_from_table_reordering = false;

  std::set<QueryPlanHash> visited_hashtable_key;

  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  auto clearCaches = [&executor, &visited_hashtable_key] {
    executor->clearMemory(MemoryLevel::CPU_LEVEL);
    executor->getQueryPlanDagCache().clearQueryPlanCache();
    visited_hashtable_key.clear();
  };

  for (auto dt : {ExecutorDeviceType::CPU}) {
    // currently we do not support hashtable caching for GPU

    // hashtables of t11, t12, t13: 208 bytes
    // hashtable of t2: 416 bytes
    {
      // test1. cache hashtable of t11 and then reuse it correctly?
      clearCaches();

      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q1_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      auto ht1_ref_count_v1 = q1_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count_v1);
      EXPECT_EQ(static_cast<size_t>(208), q1_ht_metrics->getMemSize());
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      auto ht1_ref_count_v2 = q1_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v1, ht1_ref_count_v2);
      auto q2 =
          R"(SELECT count(*) from bbox_intersect_t13 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto ht1_ref_count_v3 = q1_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v2, ht1_ref_count_v3);
    }

    {
      // test2. cache t11 and t12
      clearCaches();
      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t11 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));

      auto q2 =
          R"(SELECT count(*) from bbox_intersect_t11 as b JOIN bbox_intersect_t12 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
    }

    {
      // test3. set hashtable cache size as 420 bytes,
      // and try to cache t11's hashtable and then that of t2's
      // so we finally have t's only since sizeof(t11) + sizeof(t2) > 420
      // and so we need to remove t11's to cache t2's
      // (to check we disallow having more hashtables beyond its capacity)
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      BoundingBoxIntersectJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::BBOX_INTERSECT_HT, 420);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        BoundingBoxIntersectJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::BBOX_INTERSECT_HT, original_total_cache_size);
      };
      clearCaches();

      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q1_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      auto ht1_ref_count = q1_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count);
      EXPECT_EQ(static_cast<size_t>(208), q1_ht_metrics->getMemSize());

      auto q2 =
          R"(SELECT count(*) from bbox_intersect_t2 as b JOIN bbox_intersect_t2 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q2_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      auto ht2_ref_count = q2_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht2_ref_count);
      EXPECT_EQ(static_cast<size_t>(416), q2_ht_metrics->getMemSize());
    }

    {
      // test4. set hashtable cache size as 500 bytes, and
      // cache t11 and t12 (so total 416 bytes) and then try to cache t2
      // and check whether cache only has t2 (remove t11 and t12 to make a room for t2)
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      BoundingBoxIntersectJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::BBOX_INTERSECT_HT, 500);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        BoundingBoxIntersectJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::BBOX_INTERSECT_HT, original_total_cache_size);
      };
      clearCaches();

      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q1_ht_dag_info = getCachedHashTableMetric(visited_hashtable_key,
                                                     CacheItemType::BBOX_INTERSECT_HT);

      auto q2 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t12 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q2_ht_dag_info = getCachedHashTableMetric(visited_hashtable_key,
                                                     CacheItemType::BBOX_INTERSECT_HT);

      auto q3 =
          R"(SELECT count(*) from bbox_intersect_t2 as b JOIN bbox_intersect_t2 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q3_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      auto ht2_ref_count = q3_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht2_ref_count);
      EXPECT_EQ(static_cast<size_t>(416), q3_ht_metrics->getMemSize());
    }

    {
      // test5. set hashtable cache size as 650 bytes, and
      // we try to cache t11, t12 and t2's hashtables
      // here we make t11 to be more frequently reused one than t12
      // and try to cache t2.
      // if our cache maintenance works correctly, we should remove t12 since it is
      // less frequently reused one
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      BoundingBoxIntersectJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::BBOX_INTERSECT_HT, 650);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        BoundingBoxIntersectJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::BBOX_INTERSECT_HT, original_total_cache_size);
      };
      clearCaches();

      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q1_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      auto ht1_ref_count = q1_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count);
      EXPECT_EQ(static_cast<size_t>(208), q1_ht_metrics->getMemSize());

      auto q2 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t12 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      auto q2_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);

      auto q3 =
          R"(SELECT count(*) from bbox_intersect_t2 as b JOIN bbox_intersect_t2 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q3, dt)));
      auto q3_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      EXPECT_LT(static_cast<size_t>(1), q1_ht_metrics->getRefCount());
      EXPECT_EQ(static_cast<size_t>(1), q3_ht_metrics->getRefCount());
      EXPECT_EQ(static_cast<size_t>(416), q3_ht_metrics->getMemSize());
    }

    {
      // test 6. set per_hashtable_size_limit to be 250
      // and try to cache t11, t12 and t2
      // due to the per item limit, we can cache t11 and t12 but not t2
      const auto original_per_max_hashtable_size = g_max_cacheable_hashtable_size_bytes;
      BoundingBoxIntersectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
          CacheItemType::BBOX_INTERSECT_HT, 250);
      ScopeGuard reset_cache_status = [&original_per_max_hashtable_size] {
        BoundingBoxIntersectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
            CacheItemType::BBOX_INTERSECT_HT, original_per_max_hashtable_size);
      };
      clearCaches();

      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q1_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);

      auto q2 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t12 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q2_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);

      auto q3 =
          R"(SELECT count(*) from bbox_intersect_t2 as b JOIN bbox_intersect_t2 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q3_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      auto current_cache_size =
          BoundingBoxIntersectJoinHashTable::getHashTableCache()
              ->getCurrentCacheSizeForDevice(CacheItemType::BBOX_INTERSECT_HT,
                                             DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      EXPECT_EQ(static_cast<size_t>(416), current_cache_size);
    }

    {
      clearCaches();
      std::unordered_set<QueryPlanHash> key_set;
      // Test 7. check whether we can recycle hash table for bbox intersect correctly for
      // st_contain between mpoly and st_point
      auto q1 =
          R"(SELECT count(1) FROM bbox_intersect_t11 t2, bbox_intersect_t2 t1 WHERE st_contains(t1.mpoly, t2.pt);)";
      auto q2 =
          R"(SELECT count(1) FROM bbox_intersect_t12 t2, bbox_intersect_t3 t1 WHERE st_contains(t1.mpoly, t2.pt);)";
      auto q3 =
          R"(SELECT count(1) FROM bbox_intersect_t13 t2, bbox_intersect_t4 t1 WHERE st_contains(t1.mpoly, t2.pt);)";
      run_simple_query(q1, dt);
      auto q1_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      auto q1_key = q1_ht_metrics->getQueryPlanHash();
      key_set.insert(q1_key);
      run_simple_query(q2, dt);
      auto q2_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      auto q2_key = q2_ht_metrics->getQueryPlanHash();
      key_set.insert(q2_key);
      run_simple_query(q3, dt);
      auto q3_ht_metrics = getCachedHashTableMetric(visited_hashtable_key,
                                                    CacheItemType::BBOX_INTERSECT_HT);
      auto q3_key = q3_ht_metrics->getQueryPlanHash();
      key_set.insert(q3_key);
      EXPECT_EQ(static_cast<size_t>(3),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      EXPECT_EQ(static_cast<size_t>(3), key_set.size());
      run_simple_query(q2, dt);
      run_simple_query(q2, dt);
      run_simple_query(q3, dt);
      run_simple_query(q3, dt);
      run_simple_query(q3, dt);
      run_simple_query(q3, dt);

      auto q1_ref_cnt = q1_ht_metrics->getRefCount();
      auto q2_ref_cnt = q2_ht_metrics->getRefCount();
      auto q3_ref_cnt = q3_ht_metrics->getRefCount();
      EXPECT_LT(q1_ref_cnt, q2_ref_cnt);
      EXPECT_LT(q2_ref_cnt, q3_ref_cnt);
    }
  }
}

TEST(DataRecycler, Hashtable_For_BBox_Intersect_Reuse_Per_Parameter) {
  ScopeGuard reset_state =
      [orig_bbox_intersect_state = g_enable_bbox_intersect_hashjoin,
       orig_hashjoin_many_to_many_state = g_enable_hashjoin_many_to_many,
       orig_trivial_loop_join_threshold = g_trivial_loop_join_threshold] {
        g_enable_bbox_intersect_hashjoin = orig_bbox_intersect_state;
        g_enable_hashjoin_many_to_many = orig_hashjoin_many_to_many_state;
        g_trivial_loop_join_threshold = orig_trivial_loop_join_threshold;
      };
  g_enable_bbox_intersect_hashjoin = true;
  g_enable_hashjoin_many_to_many = true;
  g_trivial_loop_join_threshold = 1;
  std::set<QueryPlanHash> visited_hashtable_key;

  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  auto clearCaches = [&executor, &visited_hashtable_key] {
    executor->clearMemory(MemoryLevel::CPU_LEVEL);
    executor->getQueryPlanDagCache().clearQueryPlanCache();
    visited_hashtable_key.clear();
  };

  auto compareBucketDims = [](const std::vector<double>& rhs,
                              const std::vector<double>& lhs) {
    if (rhs.size() != lhs.size()) {
      return false;
    }
    for (size_t i = 0; i < rhs.size(); ++i) {
      if (rhs[i] != lhs[i]) {
        return false;
      }
    }
    return true;
  };

  auto compareHTParams = [&compareBucketDims](
                             const std::optional<BoundingBoxIntersectMetaInfo> rhs,
                             const std::optional<BoundingBoxIntersectMetaInfo> lhs) {
    return rhs.has_value() && lhs.has_value() &&
           rhs->bbox_intersect_max_table_size_bytes ==
               lhs->bbox_intersect_max_table_size_bytes &&
           rhs->bbox_intersect_bucket_threshold == lhs->bbox_intersect_bucket_threshold &&
           compareBucketDims(rhs->bucket_sizes, lhs->bucket_sizes);
  };

  for (auto dt : {ExecutorDeviceType::CPU}) {
    // currently we do not support hashtable caching for GPU
    // hashtables of t11, t12, t13: 208 bytes
    // hashtable of t2: 416 bytes
    // note that we do not compute bbox-intersect join hashtable params if given sql query
    // contains bucket_threshold

    // test1. run q1 with different bbox-intersect tuning param hint
    // to see whether hashtable recycler utilizes the latest param
    {
      clearCaches();
      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(
          static_cast<size_t>(2),
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));
      auto q1_ht_and_metainfo =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      auto q1_ht_metainfo = q1_ht_and_metainfo.cached_ht_metainfo;
      EXPECT_TRUE(q1_ht_metainfo.has_value());
      EXPECT_TRUE(q1_ht_metainfo->bbox_intersect_meta_info.has_value());
      auto q1_tuning_param = q1_ht_and_metainfo.cached_tuning_info;
      EXPECT_EQ(static_cast<size_t>(1), q1_ht_and_metainfo.cached_metric->getRefCount());
      EXPECT_EQ(static_cast<size_t>(208), q1_ht_and_metainfo.cached_metric->getMemSize());

      auto q1_v2 =
          R"(SELECT /*+ bbox_intersect_bucket_threshold(0.718) */ count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1_v2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      EXPECT_EQ(
          static_cast<size_t>(3),
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));
      auto q1_v2_ht_and_metainfo =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      auto q1_v2_ht_metainfo = q1_v2_ht_and_metainfo.cached_ht_metainfo;
      EXPECT_TRUE(q1_v2_ht_metainfo.has_value());
      EXPECT_TRUE(q1_v2_ht_metainfo->bbox_intersect_meta_info.has_value());
      auto q1_v2_tuning_param = q1_v2_ht_and_metainfo.cached_tuning_info;
      // we do not cache the tuning param if we give a related sql hint
      EXPECT_TRUE(!q1_v2_tuning_param.has_value());
      // due to the hint the same query has different hashtable params
      EXPECT_TRUE(!compareHTParams(q1_ht_metainfo->bbox_intersect_meta_info,
                                   q1_v2_ht_metainfo->bbox_intersect_meta_info));
      auto q1_v3 =
          R"(SELECT /*+ bbox_intersect_bucket_threshold(0.909), bbox_intersect_max_size(2021) */ count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1_v3, dt)));
      EXPECT_EQ(static_cast<size_t>(3),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      EXPECT_EQ(
          static_cast<size_t>(4),
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));

      auto q1_v3_ht_and_metainfo =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      auto q1_v3_ht_metainfo = q1_v3_ht_and_metainfo.cached_ht_metainfo;
      EXPECT_TRUE(q1_v3_ht_metainfo.has_value());
      EXPECT_TRUE(q1_v3_ht_metainfo->bbox_intersect_meta_info.has_value());
      auto q1_v3_tuning_param = q1_v3_ht_and_metainfo.cached_tuning_info;
      // we do not cache the tuning param if we give a related sql hint
      EXPECT_TRUE(!q1_v3_tuning_param.has_value());
      // due to the changes in the hint the same query has different hashtable params
      EXPECT_TRUE(!compareHTParams(q1_v2_ht_metainfo->bbox_intersect_meta_info,
                                   q1_v3_ht_metainfo->bbox_intersect_meta_info));
    }

    // test2. run q1 and then run q2 having different bbox_intersect
    // ht params to see whether we keep the latest q2's bbox_intersect ht
    {
      clearCaches();
      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(
          static_cast<size_t>(2),
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));
      auto q1_ht_and_metainfo =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      auto q1_ht_metainfo = q1_ht_and_metainfo.cached_ht_metainfo;
      EXPECT_TRUE(q1_ht_metainfo.has_value());
      EXPECT_TRUE(q1_ht_metainfo->bbox_intersect_meta_info.has_value());
      auto q1_tuning_param = q1_ht_and_metainfo.cached_tuning_info;
      EXPECT_EQ(static_cast<size_t>(1), q1_ht_and_metainfo.cached_metric->getRefCount());
      EXPECT_EQ(static_cast<size_t>(208), q1_ht_and_metainfo.cached_metric->getMemSize());

      auto q2 =
          R"(SELECT count(*) from bbox_intersect_t13 as b JOIN bbox_intersect_t12 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(
          static_cast<size_t>(4),
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));
      auto q2_ht_and_metainfo =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      auto q2_ht_metainfo = q2_ht_and_metainfo.cached_ht_metainfo;
      EXPECT_TRUE(q2_ht_metainfo.has_value());
      EXPECT_TRUE(q2_ht_metainfo->bbox_intersect_meta_info.has_value());
      auto q2_tuning_param = q2_ht_and_metainfo.cached_tuning_info;
      EXPECT_TRUE(q2_tuning_param.has_value());

      auto q2_v2 =
          R"(SELECT /*+ bbox_intersect_max_size(2021) */ count(*) from bbox_intersect_t13 as b JOIN bbox_intersect_t12 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q2_v2, dt)));
      EXPECT_EQ(
          static_cast<size_t>(6),
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));
      auto q2_v2_ht_and_metainfo =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      auto q2_v2_ht_metainfo = q2_v2_ht_and_metainfo.cached_ht_metainfo;
      EXPECT_TRUE(q2_v2_ht_metainfo.has_value());
      EXPECT_TRUE(q2_v2_ht_metainfo->bbox_intersect_meta_info.has_value());
      auto q2_v2_tuning_param = q2_v2_ht_and_metainfo.cached_tuning_info;
      // we compute hashtable param when we give max_hashtable size hint
      EXPECT_TRUE(q2_v2_tuning_param.has_value());
      // we should have different meta info due to the updated ht when executing q2_v2
      EXPECT_TRUE(!compareHTParams(q2_ht_metainfo->bbox_intersect_meta_info,
                                   q2_v2_ht_metainfo->bbox_intersect_meta_info));
    }

    // test3. run q1 and then run q2 but make cache has limited space to
    // see whether we invalidate ht cache but keep auto tuner param cache
    {
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      BoundingBoxIntersectJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::BBOX_INTERSECT_HT, 250);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        BoundingBoxIntersectJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::BBOX_INTERSECT_HT, original_total_cache_size);
      };
      clearCaches();
      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(
          static_cast<size_t>(2),
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));
      auto q1_ht_and_metainfo =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      auto q1_ht_metainfo = q1_ht_and_metainfo.cached_ht_metainfo;
      EXPECT_TRUE(q1_ht_metainfo.has_value());
      EXPECT_TRUE(q1_ht_metainfo->bbox_intersect_meta_info.has_value());
      auto q1_tuning_param = q1_ht_and_metainfo.cached_tuning_info;
      EXPECT_EQ(static_cast<size_t>(1), q1_ht_and_metainfo.cached_metric->getRefCount());
      EXPECT_EQ(static_cast<size_t>(208), q1_ht_and_metainfo.cached_metric->getMemSize());
      const auto q1_hash_table_cache_key = *visited_hashtable_key.begin();
      CHECK_NE(q1_hash_table_cache_key, EMPTY_HASHED_PLAN_DAG_KEY);

      auto q2 =
          R"(SELECT count(*) from bbox_intersect_t2 as b JOIN bbox_intersect_t2 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(
          static_cast<size_t>(3),
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q2_ht_and_metainfo =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      CHECK(!q2_ht_and_metainfo.cached_ht);
    }

    // test4. run q1 and then run q2 but make cache ignore
    // the q2's ht due to per-ht max limit then we should have
    // q1's ht and its auto tuner param in the cache
    {
      clearCaches();
      const auto original_max_cache_size = g_max_cacheable_hashtable_size_bytes;
      BoundingBoxIntersectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
          CacheItemType::BBOX_INTERSECT_HT, 250);
      ScopeGuard reset_cache_status = [&original_max_cache_size] {
        BoundingBoxIntersectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
            CacheItemType::BBOX_INTERSECT_HT, original_max_cache_size);
      };
      clearCaches();
      auto q1 =
          R"(SELECT count(*) from bbox_intersect_t12 as b JOIN bbox_intersect_t11 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(
          static_cast<size_t>(2),
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));
      auto q1_ht_and_metainfo =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      auto q1_ht_metainfo = q1_ht_and_metainfo.cached_ht_metainfo;
      EXPECT_TRUE(q1_ht_metainfo.has_value());
      EXPECT_TRUE(q1_ht_metainfo->bbox_intersect_meta_info.has_value());
      auto q1_tuning_param = q1_ht_and_metainfo.cached_tuning_info;
      EXPECT_EQ(static_cast<size_t>(1), q1_ht_and_metainfo.cached_metric->getRefCount());
      EXPECT_EQ(static_cast<size_t>(208), q1_ht_and_metainfo.cached_metric->getMemSize());
      visited_hashtable_key.clear();

      auto q2 =
          R"(SELECT count(*) from bbox_intersect_t2 as b JOIN bbox_intersect_t2 as a ON ST_Intersects(a.poly, b.pt);)";
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(
          static_cast<size_t>(3),  // hashtable: q1, auto tuner param: q1 and q2
          QR::get()->getNumberOfCachedItem(
              QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BBOX_INTERSECT_HT));
      auto q1_ht_and_metainfo_v2 =
          getCachedHashTableForBoundingBoxIntersectWithItsTuningParam(
              visited_hashtable_key);
      // but we skip to cache ht of q2 and this means we still have that of q1
      EXPECT_EQ(static_cast<size_t>(1),
                q1_ht_and_metainfo_v2.cached_metric->getRefCount());
      EXPECT_EQ(static_cast<size_t>(208),
                q1_ht_and_metainfo_v2.cached_metric->getMemSize());
      auto q1_ht_metainfo_v2 = q1_ht_and_metainfo_v2.cached_ht_metainfo;
      EXPECT_TRUE(q1_ht_metainfo_v2.has_value());
      EXPECT_TRUE(q1_ht_metainfo_v2->bbox_intersect_meta_info.has_value());
      auto q1_tuning_param_v2 = q1_ht_and_metainfo_v2.cached_tuning_info;
      EXPECT_TRUE(q1_tuning_param_v2.has_value());
      EXPECT_TRUE(compareHTParams(
          q1_ht_and_metainfo.cached_ht_metainfo->bbox_intersect_meta_info,
          q1_ht_and_metainfo_v2.cached_ht_metainfo->bbox_intersect_meta_info));
    }
  }
}

TEST(DataRecycler, Perfect_Hashtable_Cache_Maintanence) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  std::set<QueryPlanHash> visited_hashtable_key;
  auto clearCaches = [&executor, &visited_hashtable_key] {
    executor->clearMemory(MemoryLevel::CPU_LEVEL);
    executor->getQueryPlanDagCache().clearQueryPlanCache();
    visited_hashtable_key.clear();
  };

  // hashtables for t1, t2, t3 and t4
  // column x: unique key, like t1: {1,2,3} and t4: {1,2,3,4,5,6}
  // column y: dup. key, like t1: {1,1,1} and t2: {1,1,1,2,2,2}
  // query on a single col: deal with perfect hashtable
  // query on both cols x and y: deal with baseline hashtable
  // hashtable size info
  // t1.x = 12 / t1.y = 20 / t1.x and t1.y = 72
  // t2.x = 16 / t2.y = 32 / t2.x and t2.y = 96
  // t3.x = 20 / t3.y = 36 / t3.x and t3.y = 120
  // t4.x = 24 / t4.y = 40 / t4.x and t4.y = 144

  for (auto dt : {ExecutorDeviceType::CPU}) {
    {
      // test1. cache hashtable of t1.x and then reuse it correctly?
      clearCaches();
      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q1_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      auto ht1_ref_count_v1 = q1_perfect_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count_v1);
      EXPECT_EQ(static_cast<size_t>(12), q1_perfect_ht_metrics->getMemSize());
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto ht1_ref_count_v2 = q1_perfect_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v1, ht1_ref_count_v2);
      auto q2 = "SELECT count(*) from t1, t3 where t1.x = t3.x;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto ht1_ref_count_v3 = q1_perfect_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v2, ht1_ref_count_v3);
    }

    {
      // test2. cache t1.x and t1.y
      clearCaches();

      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q1_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      EXPECT_EQ(static_cast<size_t>(1), q1_perfect_ht_metrics->getRefCount());
      EXPECT_EQ(static_cast<size_t>(12), q1_perfect_ht_metrics->getMemSize());
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));

      auto q2 = "SELECT count(*) from t1, t2 where t1.y = t2.y;";
      EXPECT_EQ(static_cast<int64_t>(9), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q2_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      EXPECT_EQ(static_cast<size_t>(1), q2_perfect_ht_metrics->getRefCount());
      EXPECT_EQ(static_cast<size_t>(20), q2_perfect_ht_metrics->getMemSize());
    }

    {
      // test3. set hashtable cache size as 30 bytes,
      // and try to cache t1.x's hashtable (12 bytes) and then that of t4.x's (24 bytes)
      // since sizeof(t1.x) + sizeof(t4.x) > 30 we need to remove t1.x's to cache t4.x's
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      PerfectJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::PERFECT_HT, 30);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        PerfectJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::PERFECT_HT, original_total_cache_size);
      };
      clearCaches();

      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q1_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      EXPECT_EQ(static_cast<size_t>(1), q1_perfect_ht_metrics->getRefCount());
      EXPECT_EQ(static_cast<size_t>(12), q1_perfect_ht_metrics->getMemSize());

      auto q2 = "SELECT count(*) from t4 a, t4 b where a.x = b.x;";
      EXPECT_EQ(static_cast<int64_t>(6), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q2_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      EXPECT_EQ(static_cast<size_t>(1), q2_perfect_ht_metrics->getRefCount());
      EXPECT_EQ(static_cast<size_t>(24), q2_perfect_ht_metrics->getMemSize());
      auto current_cache_size =
          PerfectJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
              CacheItemType::PERFECT_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      EXPECT_EQ(static_cast<size_t>(24), current_cache_size);
    }

    {
      // test4. set hashtable cache size as 30 bytes, and
      // cache t1.x and t2.x (so total 14 * 2 = 28 bytes) and then try to cache t4.x's
      // and check whether cache only has t4.x's (remove t1.x's and t2.x's to make a room
      // for t4.x's)
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      PerfectJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::PERFECT_HT, 30);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        PerfectJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::PERFECT_HT, original_total_cache_size);
      };
      clearCaches();

      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q1_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      EXPECT_EQ(static_cast<size_t>(1), q1_perfect_ht_metrics->getRefCount());
      EXPECT_EQ(static_cast<size_t>(12), q1_perfect_ht_metrics->getMemSize());

      auto q2 = "SELECT count(*) from t2 a, t2 b where a.x = b.x;";
      EXPECT_EQ(static_cast<int64_t>(4), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q2_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);

      auto q3 = "SELECT count(*) from t4 a, t4 b where a.x = b.x;";
      EXPECT_EQ(static_cast<int64_t>(6), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q3_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      EXPECT_EQ(static_cast<size_t>(24), q3_perfect_ht_metrics->getMemSize());
      EXPECT_EQ(static_cast<size_t>(1), q3_perfect_ht_metrics->getRefCount());
      auto current_cache_size =
          PerfectJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
              CacheItemType::PERFECT_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      EXPECT_EQ(static_cast<size_t>(24), current_cache_size);
    }

    {
      // test5. set hashtable cache size as 40 bytes, and
      // we try to cache t1.x, t2.x and t3.x's hashtables
      // here we make t1.x's to be more frequently reused one than t2.x's
      // and try to cache t3.x's.
      // if our cache maintenance works correctly, we should remove t2.x's since it is
      // less frequently reused one
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      PerfectJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::PERFECT_HT, 40);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        PerfectJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::PERFECT_HT, original_total_cache_size);
      };
      clearCaches();

      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q1_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      EXPECT_EQ(static_cast<size_t>(1), q1_perfect_ht_metrics->getRefCount());
      EXPECT_EQ(static_cast<size_t>(12), q1_perfect_ht_metrics->getMemSize());

      auto q2 = "SELECT count(*) from t2 a, t2 b where a.x = b.x;";
      EXPECT_EQ(static_cast<int64_t>(4), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));

      auto q3 = "SELECT count(*) from t3 a, t3 b where a.x = b.x;";
      EXPECT_EQ(static_cast<int64_t>(5), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q3_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      EXPECT_EQ(q1_perfect_ht_metrics->getMemSize(), static_cast<size_t>(12));
      EXPECT_GT(q1_perfect_ht_metrics->getRefCount(), static_cast<size_t>(1));
      EXPECT_EQ(q3_perfect_ht_metrics->getMemSize(), static_cast<size_t>(20));
      EXPECT_EQ(q3_perfect_ht_metrics->getRefCount(), static_cast<size_t>(1));
      auto current_cache_size =
          PerfectJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
              CacheItemType::PERFECT_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      EXPECT_EQ(static_cast<size_t>(32), current_cache_size);
    }

    {
      // test 6. set per_hashtable_size_limit to be 18
      // and try to cache t1.x, t2.x and t3.x
      // due to the per item limit, we can cache t1.x's and t2.x's but not t3.x's
      const auto original_per_max_hashtable_size = g_max_cacheable_hashtable_size_bytes;
      PerfectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
          CacheItemType::PERFECT_HT, 18);
      ScopeGuard reset_cache_status = [&original_per_max_hashtable_size] {
        PerfectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
            CacheItemType::PERFECT_HT, original_per_max_hashtable_size);
      };
      clearCaches();

      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      auto q1_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);

      auto q2 = "SELECT count(*) from t2 a, t2 b where a.x = b.x;";
      EXPECT_EQ(static_cast<int64_t>(4), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      EXPECT_EQ(static_cast<int64_t>(4), v<int64_t>(run_simple_query(q2, dt)));
      auto q2_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);

      auto q3 = "SELECT count(*) from t3 a, t3 b where a.x = b.x;";
      EXPECT_EQ(static_cast<int64_t>(5), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q3_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      auto current_cache_size =
          PerfectJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
              CacheItemType::PERFECT_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      EXPECT_EQ(static_cast<size_t>(28), current_cache_size);
      if (!q1_perfect_ht_metrics || !q1_perfect_ht_metrics || q3_perfect_ht_metrics) {
        EXPECT_TRUE(false);
      }
    }

    {
      // test 7. a join query using synthetically generated table
      clearCaches();
      auto q1 =
          "with cte as (select s from table(generate_series(0, 10, 2)) as t(s)) select "
          "count(1) from table(generate_series(0, 10, 1)) as ft(s), cte where ft.s = "
          "cte.s;";
      EXPECT_EQ(static_cast<int64_t>(6), v<int64_t>(run_simple_query(q1, dt)));
      auto q2 =
          "with cte as (select s from table(generate_series(0, 10, 3)) as t(s)) select "
          "count(1) from table(generate_series(0, 10, 1)) as ft(s), cte where ft.s = "
          "cte.s;";
      EXPECT_EQ(static_cast<int64_t>(4), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                                 CacheItemType::PERFECT_HT));

      // check whether hash tables generated from q1 and q2 are invalidated
      auto q3 = "SELECT count(*) from t3 a, t3 b where a.x = b.x;";
      EXPECT_EQ(static_cast<int64_t>(5), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(3),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                                 CacheItemType::PERFECT_HT));

      // there is no rows in t3 matching the filter condition
      QR::get()->runSQL("update t3 set x = 1 where x > 999999999999;", dt);
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                                 CacheItemType::PERFECT_HT));
    }
  }
}

TEST(DataRecycler, Baseline_Hashtable_Cache_Maintanence) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  std::set<QueryPlanHash> visited_hashtable_key;
  auto clearCaches = [&executor, &visited_hashtable_key] {
    executor->clearMemory(MemoryLevel::CPU_LEVEL);
    executor->getQueryPlanDagCache().clearQueryPlanCache();
    visited_hashtable_key.clear();
  };

  // hashtables for t1, t2, t3 and t4
  // column x: unique key, like t1: {1,2,3} and t4: {1,2,3,4,5,6}
  // column y: dup. key, like t1: {1,1,1} and t2: {1,1,1,2,2,2}
  // query on a single col: deal with perfect hashtable
  // query on both cols x and y: deal with baseline hashtable
  // hashtable size info
  // t1.x = 12 / t1.y = 20 / t1.x and t1.y = 72
  // t2.x = 16 / t2.y = 32 / t2.x and t2.y = 96
  // t3.x = 20 / t3.y = 36 / t3.x and t3.y = 120
  // t4.x = 24 / t4.y = 40 / t4.x and t4.y = 144
  for (auto dt : {ExecutorDeviceType::CPU}) {
    // currently we do not support hashtable caching for GPU
    {
      // test1. cache hashtable of t1 and then reuse it correctly?
      clearCaches();
      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x and t1.y = t2.y;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q1_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      auto ht1_ref_count_v1 = q1_baseline_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count_v1);
      EXPECT_EQ(static_cast<size_t>(72), q1_baseline_ht_metrics->getMemSize());
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto ht1_ref_count_v2 = q1_baseline_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v1, ht1_ref_count_v2);
      auto q2 = "SELECT count(*) from t1, t3 where t1.x = t3.x and t1.y = t3.y;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto ht1_ref_count_v3 = q1_baseline_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v2, ht1_ref_count_v3);
    }

    {
      // test2. cache t1 and t2
      clearCaches();
      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x and t1.y = t2.y;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q1_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      auto ht1_ref_count_v1 = q1_baseline_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count_v1);
      EXPECT_EQ(static_cast<size_t>(72), q1_baseline_ht_metrics->getMemSize());

      auto q2 = "SELECT count(*) from t2, t3 where t2.x = t3.x and t2.y = t3.y;";
      EXPECT_EQ(static_cast<int64_t>(4), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
    }

    {
      // test3. set hashtable cache size as 150 bytes,
      // and try to cache t1's hashtable (72 bytes) and then that of t4's (144 bytes)
      // since sizeof(t1) + sizeof(t4) > 150 we need to remove t1's to cache t4's
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      BaselineJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::BASELINE_HT, 150);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        BaselineJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::BASELINE_HT, original_total_cache_size);
      };
      clearCaches();

      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x and t1.y = t2.y;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q1_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      auto ht1_ref_count = q1_baseline_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count);
      EXPECT_EQ(static_cast<size_t>(72), q1_baseline_ht_metrics->getMemSize());

      auto q2 = "SELECT count(*) from t4 a, t4 b where a.x = b.x and a.y = b.y;";
      EXPECT_EQ(static_cast<int64_t>(6), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q4_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      auto ht4_ref_count = q4_baseline_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht4_ref_count);
      EXPECT_EQ(static_cast<size_t>(144), q4_baseline_ht_metrics->getMemSize());
      auto current_cache_size =
          BaselineJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
              CacheItemType::BASELINE_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      EXPECT_EQ(static_cast<size_t>(144), current_cache_size);
    }

    {
      // test4. set hashtable cache size as 180 bytes, and
      // cache t1 and t2 (so total 72+96 = 168 bytes) and then try to cache t4's
      // and check whether cache only has t4's (remove t1.x's and t2.x's to make a room
      // for t4's)
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      BaselineJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::BASELINE_HT, 180);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        BaselineJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::BASELINE_HT, original_total_cache_size);
      };
      clearCaches();

      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x and t1.y = t2.y;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q1_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      auto ht1_ref_count = q1_baseline_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count);
      EXPECT_EQ(static_cast<size_t>(72), q1_baseline_ht_metrics->getMemSize());

      auto q2 = "SELECT count(*) from t2 a, t2 b where a.x = b.x and a.y = b.y;";
      EXPECT_EQ(static_cast<int64_t>(4), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));

      auto q3 = "SELECT count(*) from t4 a, t4 b where a.x = b.x and a.y = b.y;";
      EXPECT_EQ(static_cast<int64_t>(6), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q3_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      auto ht4_ref_count = q3_baseline_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht4_ref_count);
      EXPECT_EQ(static_cast<size_t>(144), q3_baseline_ht_metrics->getMemSize());
      auto current_cache_size =
          BaselineJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
              CacheItemType::BASELINE_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      EXPECT_EQ(static_cast<size_t>(144), current_cache_size);
    }

    {
      // test5. set hashtable cache size as 200 bytes, and
      // we try to cache t1 (72 bytes), t2 (96 bytes) and t3's hashtable (120 bytes),
      // respectively here we make t1's to be more frequently reused one than t2's and try
      // to cache t3's. in this case we should remove at least one hashtable btw. t1 and
      // t2 since sizeof(t1) + sizeof(t2) + sizeof(t3) ? 72 + 96 + 120 > 200 if our cache
      // maintenance works correctly, we should remove t2's since it is less frequently
      // reused one (so 72 + 120 = 192 < 200)
      const auto original_total_cache_size = g_hashtable_cache_total_bytes;
      BaselineJoinHashTable::getHashTableCache()->setTotalCacheSize(
          CacheItemType::BASELINE_HT, 200);
      ScopeGuard reset_cache_status = [&original_total_cache_size] {
        BaselineJoinHashTable::getHashTableCache()->setTotalCacheSize(
            CacheItemType::BASELINE_HT, original_total_cache_size);
      };
      clearCaches();

      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x and t1.y = t2.y;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q1_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      EXPECT_EQ(static_cast<size_t>(1), q1_baseline_ht_metrics->getRefCount());
      EXPECT_EQ(static_cast<size_t>(72), q1_baseline_ht_metrics->getMemSize());
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));

      auto q2 = "SELECT count(*) from t2 a, t2 b where a.x = b.x and a.y = b.y;";
      EXPECT_EQ(static_cast<int64_t>(4), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));

      auto q3 = "SELECT count(*) from t3 a, t3 b where a.x = b.x and a.y = b.y;";
      EXPECT_EQ(static_cast<int64_t>(5), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q3_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      EXPECT_EQ(q3_baseline_ht_metrics->getMemSize(), static_cast<size_t>(120));
      auto current_cache_size =
          BaselineJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
              CacheItemType::BASELINE_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      EXPECT_EQ(static_cast<size_t>(192), current_cache_size);
    }

    {
      // test 6. set per_hashtable_size_limit to be 100
      // and try to cache t1 (72 bytes), t2 (96 bytes) and t3 (120 bytes)
      // due to the per item limit, we can cache t1's and t2's but not t3's
      const auto original_per_max_hashtable_size = g_max_cacheable_hashtable_size_bytes;
      BaselineJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
          CacheItemType::BASELINE_HT, 100);
      ScopeGuard reset_cache_status = [&original_per_max_hashtable_size] {
        BaselineJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
            CacheItemType::BASELINE_HT, original_per_max_hashtable_size);
      };
      clearCaches();

      auto q1 = "SELECT count(*) from t1, t2 where t1.x = t2.x and t1.y = t2.y;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));

      auto q2 = "SELECT count(*) from t2 a, t2 b where a.x = b.x and a.y = b.y;";
      EXPECT_EQ(static_cast<int64_t>(4), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));

      auto q3 = "SELECT count(*) from t3 a, t3 b where a.x = b.x and a.y = b.y;";
      EXPECT_EQ(static_cast<int64_t>(5), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto current_cache_size =
          BaselineJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
              CacheItemType::BASELINE_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      EXPECT_EQ(static_cast<size_t>(168), current_cache_size);
    }
  }
}

TEST(DataRecycler, Hashtable_From_Subqueries) {
  // todo (yoonmin): revisit here if we support skipping hashtable building based on
  // consideration of filter quals
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  std::set<QueryPlanHash> visited_hashtable_key;
  auto clearCaches = [&executor, &visited_hashtable_key] {
    executor->clearMemory(MemoryLevel::CPU_LEVEL);
    executor->getQueryPlanDagCache().clearQueryPlanCache();
    visited_hashtable_key.clear();
  };

  for (auto dt : {ExecutorDeviceType::CPU}) {
    // currently we do not support hashtable caching for GPU
    {
      // test1. perfect hashtable
      clearCaches();
      auto q1 =
          "SELECT count(*) from t1, (select x from t2 where x < 3) tt2 where t1.x = "
          "tt2.x;";
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q1_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      auto ht1_ref_count_v1 = q1_perfect_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count_v1);
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto ht1_ref_count_v2 = q1_perfect_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v1, ht1_ref_count_v2);

      auto q2 =
          "SELECT count(*) from t1, (select x from t2 where x < 2) tt2 where t1.x = "
          "tt2.x;";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto q2_perfect_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::PERFECT_HT);
      auto ht1_ref_count_v3 = q2_perfect_ht_metrics->getRefCount();
      EXPECT_GT(ht1_ref_count_v2, ht1_ref_count_v3);

      auto q3 =
          "SELECT count(*) from (select x from t1) tt1, (select x from t2 where x < 2) "
          "tt2 where tt1.x = tt2.x;";
      EXPECT_EQ(static_cast<int64_t>(1), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      auto ht1_ref_count_v4 = q2_perfect_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v3, ht1_ref_count_v4);
    }

    {
      // test2. baseline hashtable
      clearCaches();
      auto q1 =
          "SELECT count(*) from t1, (select x,y from t2 where x < 4) tt2 where t1.x = "
          "tt2.x and t1.y = tt2.y;";
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q1_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      auto ht1_ref_count_v1 = q1_baseline_ht_metrics->getRefCount();
      EXPECT_EQ(static_cast<size_t>(1), ht1_ref_count_v1);
      EXPECT_EQ(static_cast<int64_t>(3), v<int64_t>(run_simple_query(q1, dt)));
      EXPECT_EQ(static_cast<size_t>(1),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto ht1_ref_count_v2 = q1_baseline_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v1, ht1_ref_count_v2);

      auto q2 =
          "SELECT count(*) from t1, (select x, y from t3 where x < 3) tt3 where t1.x = "
          "tt3.x and t1.y = tt3.y;";
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q2, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto q2_baseline_ht_metrics =
          getCachedHashTableMetric(visited_hashtable_key, CacheItemType::BASELINE_HT);
      auto ht1_ref_count_v3 = q2_baseline_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v3, ht1_ref_count_v2);

      auto q3 =
          "SELECT count(*) from (select x, y from t1 where x < 3) tt1, (select x, y from "
          "t3 where x < 3) tt3 where tt1.x = tt3.x and tt1.y = tt3.y;";
      EXPECT_EQ(static_cast<int64_t>(2), v<int64_t>(run_simple_query(q3, dt)));
      EXPECT_EQ(static_cast<size_t>(2),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::BASELINE_HT));
      auto ht1_ref_count_v4 = q2_baseline_ht_metrics->getRefCount();
      EXPECT_LT(ht1_ref_count_v3, ht1_ref_count_v4);
    }

    {
      // test 3. changing resultset's ordering
      clearCaches();
      auto q1 =
          "WITH tt2 AS (SELECT z, approx_count_distinct(x) * 1.0 AS xx FROM t2 GROUP BY "
          "z) SELECT tt2.z, tt2.xx FROM t4, tt2 WHERE (t4.z = tt2.z) ORDER BY tt2.z;";
      auto q2 =
          "WITH tt2 AS (SELECT z, x FROM t2 limit 2) SELECT tt2.z, tt2.x FROM t4, tt2 "
          "WHERE (t4.z = tt2.z) ORDER BY tt2.z;";
      for (int i = 0; i < 5; ++i) {
        QR::get()->runSQL(q1, dt);
        QR::get()->runSQL(q2, dt);
      }
      // we have to skip hashtable caching for the above query
      EXPECT_EQ(static_cast<size_t>(0),
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
    }
  }
}

TEST(DataRecycler, Empty_Hashtable) {
  run_ddl_statement("DROP TABLE IF EXISTS T5;");
  run_ddl_statement("DROP TABLE IF EXISTS T6;");
  run_ddl_statement("CREATE TABLE T5 (c1 INT, c2 INT);");
  run_ddl_statement("CREATE TABLE T6 (c1 INT, c2 INT);");
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  auto clearCaches = [&executor](ExecutorDeviceType dt) {
    auto memory_level =
        dt == ExecutorDeviceType::CPU ? MemoryLevel::CPU_LEVEL : MemoryLevel::GPU_LEVEL;
    executor->clearMemory(memory_level);
    executor->getQueryPlanDagCache().clearQueryPlanCache();
  };
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    QR::get()->runSQL("SELECT COUNT(1) FROM T5 INNER JOIN T6 ON (T5.c1 = T6.c1);", dt);
    clearCaches(dt);
    QR::get()->runSQL(
        "SELECT COUNT(1) FROM T5 INNER JOIN T6 ON (T5.c1 = T6.c1 AND T5.c2 = T6.c2);",
        dt);
    clearCaches(dt);
  }
}

TEST(DataRecycler, Hashtable_For_Dict_Encoded_Column) {
  run_ddl_statement("DROP TABLE IF EXISTS TT1;");
  run_ddl_statement("DROP TABLE IF EXISTS TT2;");
  run_ddl_statement("CREATE TABLE TT1 (c1 TEXT ENCODING DICT(32), id1 INT);");
  run_ddl_statement("CREATE TABLE TT2 (c2 TEXT ENCODING DICT(32), id2 INT);");
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  auto clear_caches = [&executor](ExecutorDeviceType dt) {
    auto memory_level =
        dt == ExecutorDeviceType::CPU ? MemoryLevel::CPU_LEVEL : MemoryLevel::GPU_LEVEL;
    executor->clearMemory(memory_level);
    executor->getQueryPlanDagCache().clearQueryPlanCache();
  };

  auto data_loader = [](const std::string& table_name, int num_rows) {
    for (int i = 1; i <= num_rows; ++i) {
      auto val = ::toString(i);
      auto insert_stmt = "INSERT INTO " + table_name + " VALUES (" + val + ", \'" + val +
                         "\'"
                         ");";
      QR::get()->runSQL(insert_stmt, ExecutorDeviceType::CPU);
    }
  };

  data_loader("TT1", 10);
  data_loader("TT2", 20);

  std::string q1a{"SELECT count(1) FROM TT1 WHERE c1 IN (SELECT c2 FROM TT2);"};
  std::string q1b{
      "SELECT count(1) FROM TT1, (SELECT c2 FROM TT2 GROUP BY 1) T2 WHERE c1 = c2;"};
  auto q1 = std::make_pair(q1a, q1b);

  std::string q2a{
      "SELECT count(1) FROM TT1 WHERE c1 IN (SELECT c2 FROM TT2 WHERE id2 < 15);"};
  std::string q2b{
      "SELECT count(1) FROM TT1, (SELECT c2 FROM TT2 WHERE id2 < 15  GROUP BY 1) T2 "
      "WHERE c1 = c2;"};
  auto q2 = std::make_pair(q2a, q2b);

  std::string q3a{
      "SELECT count(1) FROM TT1 WHERE c1 IN (SELECT c2 FROM TT2 WHERE id2 < 5);"};
  std::string q3b{
      "SELECT count(1) FROM TT1, (SELECT c2 FROM TT2 WHERE id2 < 5  GROUP BY 1) T2 WHERE "
      "c1 = c2;"};
  auto q3 = std::make_pair(q3a, q3b);

  std::string q4a{"SELECT count(1) FROM TT2 WHERE c2 IN (SELECT c1 FROM TT1);"};
  std::string q4b{
      "SELECT count(1) FROM TT2, (SELECT c1 FROM TT1  GROUP BY 1) T1 WHERE c1 = c2;"};
  auto q4 = std::make_pair(q4a, q4b);

  std::string q5a{
      "SELECT count(1) FROM TT2 WHERE c2 IN (SELECT c1 FROM TT1 WHERE id1 < 6);"};
  std::string q5b{
      "SELECT count(1) FROM TT2, (SELECT c1 FROM TT1 WHERE id1 < 6  GROUP BY 1) T1 WHERE "
      "c1 = c2;"};
  auto q5 = std::make_pair(q5a, q5b);

  std::string q6a{
      "SELECT count(1) FROM TT2 WHERE c2 IN (SELECT c1 FROM TT1 WHERE id1 < 3);"};
  std::string q6b{
      "SELECT count(1) FROM TT2, (SELECT c1 FROM TT1 WHERE id1 < 3  GROUP BY 1) T1 WHERE "
      "c1 = c2;"};
  auto q6 = std::make_pair(q6a, q6b);

  std::string q7a{"SELECT count(1) FROM TT2, TT1 WHERE c2 = c1;"};
  std::string q7b{"SELECT count(1) FROM TT1, TT2 WHERE c1 = c2;"};
  auto q7 = std::make_pair(q7a, q7b);

  std::string q8a{"SELECT count(1) FROM TT2, TT1 WHERE c2 = c1 AND id1 < 6;"};
  std::string q8b{"SELECT count(1) FROM TT1, TT2 WHERE c1 = c2 AND id2 < 15;"};
  auto q8 = std::make_pair(q8a, q8b);

  std::string q9a{"SELECT count(1) FROM TT2, TT1 WHERE c2 = c1 AND id1 < 3;"};
  std::string q9b{"SELECT count(1) FROM TT1, TT2 WHERE c1 = c2 AND id2 < 5;"};
  auto q9 = std::make_pair(q9a, q9b);

  auto case1 = std::make_pair(q1a, q7a);
  auto case2 = std::make_pair(q1b, q7a);
  auto case3 = std::make_pair(q1a, q7b);
  auto case4 = std::make_pair(q1b, q7b);

  auto check_query = [](const std::string& query, bool expected) {
    auto root_node = QR::get()->getRootNodeFromParsedQuery(query);
    auto has_in_expr = SQLOperatorDetector::detect(root_node.get(), SQLOps::kIN);
    EXPECT_EQ(has_in_expr, expected);
  };

  auto perform_test = [&clear_caches, &check_query](
                          const auto queries, size_t expected_num_cached_hashtable) {
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      QR::get()->runSQL(queries.first, dt);
      QR::get()->runSQL(queries.second, dt);
      check_query(queries.first, false);
      check_query(queries.second, false);
      EXPECT_EQ(expected_num_cached_hashtable,
                QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                                 CacheItemType::PERFECT_HT));
      clear_caches(ExecutorDeviceType::CPU);
    }
  };

  auto execute_random_query_test = [&clear_caches](auto& queries,
                                                   size_t expected_num_cached_hashtable) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(queries.begin(), queries.end(), g);
    for (const auto& query : queries) {
      QR::get()->runSQL(query, ExecutorDeviceType::CPU);
    }
    EXPECT_EQ(expected_num_cached_hashtable,
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT));
    clear_caches(ExecutorDeviceType::CPU);
  };

  std::vector<std::string> queries_case1 = {
      q1a, q1b, q2a, q2b, q3a, q3b, q4a, q4b, q5a, q5b, q6a, q6b};
  std::vector<std::string> queries_case2 = {q7a, q7b, q8a, q8b, q9a, q9b};

  ScopeGuard reset = [orig = g_from_table_reordering] { g_from_table_reordering = orig; };

  // 1. disable from-table-reordering
  // this means the same join query with different table listing order in FROM clause
  // affects the cache key computation
  // for table involving subqueries, we expect explicit subquery, e.g., SELECT ... FROM
  // ..., (SELECT ...) and implicit subquery (per query planner), e.g., SELECT ... FROM
  // ... WHERE ... IN (SELECT ...) have different cache key even if their query semantic
  // is the same since their plan is different, e.g., decorrelation per query planner adds
  // de-duplication logic
  g_from_table_reordering = false;
  clear_caches(ExecutorDeviceType::CPU);
  for (const auto& test_case : {q1, q2, q3, q4, q5, q6}) {
    perform_test(test_case, static_cast<size_t>(1));
  }
  for (const auto& test_case : {case1, case2, case3, case4}) {
    perform_test(test_case, static_cast<size_t>(2));
  }
  for (const auto& test_case : {q7, q8, q9}) {
    perform_test(test_case, static_cast<size_t>(2));
  }
  execute_random_query_test(queries_case1, 6);
  execute_random_query_test(queries_case2, 2);

  // 2. enable from-table-reordering
  // if the table cardinality and a join qual are the same, we have the same cache key
  // regardless of table listing order in FROM clause
  //
  g_from_table_reordering = true;
  clear_caches(ExecutorDeviceType::CPU);
  for (const auto& test_case : {q1, q2, q3, q4, q5, q6}) {
    perform_test(test_case, static_cast<size_t>(1));
  }
  for (const auto& test_case : {case1, case2, case3, case4}) {
    perform_test(test_case, static_cast<size_t>(2));
  }
  for (const auto& test_case : {q7, q8, q9}) {
    perform_test(test_case, static_cast<size_t>(1));
  }
  execute_random_query_test(queries_case1, 6);
  execute_random_query_test(queries_case2, 1);
}

TEST(DataRecycler, SimpleInsertion) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  executor->clearMemory(MemoryLevel::CPU_LEVEL);
  executor->getQueryPlanDagCache().clearQueryPlanCache();

  auto drop_tables = [] {
    run_ddl_statement("DROP TABLE IF EXISTS tf1;");
    run_ddl_statement("DROP TABLE IF EXISTS tf2;");
    run_ddl_statement("DROP TABLE IF EXISTS tb1;");
    run_ddl_statement("DROP TABLE IF EXISTS tb2;");
  };

  auto prepare_tables = [] {
    run_ddl_statement("CREATE TABLE tf1 (x int);");
    run_ddl_statement("CREATE TABLE tf2 (x int);");
    for (int i = 1; i <= 4; i++) {
      if (i < 3) {
        QR::get()->runSQL("insert into tf1 values (" + ::toString(i) + ")",
                          ExecutorDeviceType::CPU);
      }
      QR::get()->runSQL("insert into tf2 values (" + ::toString(i) + ")",
                        ExecutorDeviceType::CPU);
    }

    run_ddl_statement("CREATE TABLE tb1 (x int, y int);");
    run_ddl_statement("CREATE TABLE tb2 (x int, y int);");
    for (int i = 1; i <= 4; i++) {
      if (i < 3) {
        auto insert_tb1 =
            "insert into tb1 values (" + ::toString(i) + "," + ::toString(i) + ")";
        QR::get()->runSQL(insert_tb1, ExecutorDeviceType::CPU);
        QR::get()->runSQL(insert_tb1, ExecutorDeviceType::CPU);
      }
      auto insert_tb2 =
          "insert into tb2 values (" + ::toString(i) + "," + ::toString(i) + ")";
      QR::get()->runSQL(insert_tb2, ExecutorDeviceType::CPU);
      QR::get()->runSQL(insert_tb2, ExecutorDeviceType::CPU);
    }
  };

  for (auto dt : {ExecutorDeviceType::CPU}) {
    drop_tables();
    prepare_tables();

    // perfect join hashtable
    auto perfect_ht_join_query = "select count(1) from tf1, tf2 where tf1.x = tf2.x;";
    QR::get()->runSQL(perfect_ht_join_query, dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT));
    QR::get()->runSQL("update tf1 set x = 4 where x = 1;", dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT));

    QR::get()->runSQL(perfect_ht_join_query, dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT));

    QR::get()->runSQL("insert into tf1 values (3);", dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT));
    auto perfect_ht_res =
        v<int64_t>(run_simple_query(perfect_ht_join_query, dt, false, false));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<int64_t>(3), perfect_ht_res);

    // baseline join hashtable
    auto baseline_ht_join_query =
        "select count(1) from tb1, tb2 where tb1.x = tb2.x and tb1.y = tb2.y;";
    QR::get()->runSQL(baseline_ht_join_query, dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::BASELINE_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::BASELINE_HT));

    QR::get()->runSQL("update tb1 set x = 4 where x = 1;", dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::BASELINE_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::BASELINE_HT));

    QR::get()->runSQL(baseline_ht_join_query, dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::BASELINE_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::BASELINE_HT));

    QR::get()->runSQL("insert into tb1 values (3, 3);", dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::BASELINE_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::BASELINE_HT));

    auto baseline_ht_res =
        v<int64_t>(run_simple_query(baseline_ht_join_query, dt, false, false));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::BASELINE_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::BASELINE_HT));
    EXPECT_EQ(static_cast<int64_t>(6), baseline_ht_res);
  }
  drop_tables();
}

TEST(DataRecycler, Lazy_Cache_Invalidation) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  auto clearCache = [&executor] {
    executor->clearMemory(MemoryLevel::CPU_LEVEL);
    executor->getQueryPlanDagCache().clearQueryPlanCache();
  };

  auto drop_tables_for_string_joins = [] {
    run_ddl_statement("DROP TABLE IF EXISTS tns1;");
    run_ddl_statement("DROP TABLE IF EXISTS tns2;");
    run_ddl_statement("DROP TABLE IF EXISTS ts1;");
    run_ddl_statement("DROP TABLE IF EXISTS ts2;");
    run_ddl_statement("DROP TABLE IF EXISTS ts3;");
    run_ddl_statement("DROP TABLE IF EXISTS ts4;");
  };
  auto drop_tables = [] {
    run_ddl_statement("DROP TABLE IF EXISTS tf1;");
    run_ddl_statement("DROP TABLE IF EXISTS tf2;");
    run_ddl_statement("DROP TABLE IF EXISTS tf3;");
    run_ddl_statement("DROP TABLE IF EXISTS tf4;");

    run_ddl_statement("DROP TABLE IF EXISTS tb1;");
    run_ddl_statement("DROP TABLE IF EXISTS tb2;");
    run_ddl_statement("DROP TABLE IF EXISTS tb3;");
    run_ddl_statement("DROP TABLE IF EXISTS tb4;");
  };

  auto prepare_tables_for_string_joins = [] {
    run_ddl_statement("CREATE TABLE ts1 (x text encoding dict);");
    run_ddl_statement(
        "CREATE TABLE ts2 (x text, shared dictionary (x) references ts1(x));");
    run_ddl_statement("CREATE TABLE ts3 (x text encoding dict);");
    run_ddl_statement(
        "CREATE TABLE ts4 (x text, shared dictionary (x) references ts3(x));");
    run_ddl_statement("CREATE TABLE tns1 (x text encoding dict);");
    run_ddl_statement("CREATE TABLE tns2 (x text encoding dict);");

    for (int i = 1; i <= 10; i++) {
      if (i < 3) {
        QR::get()->runSQL("insert into ts1 values (" + ::toString(i) + ");",
                          ExecutorDeviceType::CPU);
      }
      if (i < 5) {
        QR::get()->runSQL("insert into ts2 values (" + ::toString(i) + ");",
                          ExecutorDeviceType::CPU);
        QR::get()->runSQL("insert into tns1 values (" + ::toString(i) + ");",
                          ExecutorDeviceType::CPU);
      }
      if (i < 8) {
        QR::get()->runSQL("insert into ts4 values (" + ::toString(i) + ");",
                          ExecutorDeviceType::CPU);
      }
      QR::get()->runSQL("insert into ts3 values (" + ::toString(i) + ");",
                        ExecutorDeviceType::CPU);
      QR::get()->runSQL("insert into tns2 values (" + ::toString(i) + ");",
                        ExecutorDeviceType::CPU);
    }
  };

  auto prepare_tables = [] {
    run_ddl_statement("CREATE TABLE tf1 (x int);");
    run_ddl_statement("CREATE TABLE tf2 (x int);");
    run_ddl_statement("CREATE TABLE tf3 (x int);");
    run_ddl_statement("CREATE TABLE tf4 (x int);");
    for (int i = 1; i <= 10; i++) {
      if (i < 3) {
        QR::get()->runSQL("insert into tf1 values (" + ::toString(i) + ");",
                          ExecutorDeviceType::CPU);
      }
      if (i < 5) {
        QR::get()->runSQL("insert into tf2 values (" + ::toString(i) + ");",
                          ExecutorDeviceType::CPU);
      }
      if (i < 8) {
        QR::get()->runSQL("insert into tf3 values (" + ::toString(i) + ");",
                          ExecutorDeviceType::CPU);
      }
      QR::get()->runSQL("insert into tf4 values (" + ::toString(i) + ");",
                        ExecutorDeviceType::CPU);
    }

    run_ddl_statement("CREATE TABLE tb1 (x int, y int);");
    run_ddl_statement("CREATE TABLE tb2 (x int, y int);");
    run_ddl_statement("CREATE TABLE tb3 (x int, y int);");
    run_ddl_statement("CREATE TABLE tb4 (x int, y int);");
    for (int i = 1; i <= 20; i++) {
      int val = i % 10;
      if (i < 5) {
        auto insert_tb1 =
            "insert into tb1 values (" + ::toString(val) + "," + ::toString(val) + ");";
        QR::get()->runSQL(insert_tb1, ExecutorDeviceType::CPU);
      }
      if (i < 10) {
        auto insert_tb1 =
            "insert into tb2 values (" + ::toString(val) + "," + ::toString(val) + ");";
        QR::get()->runSQL(insert_tb1, ExecutorDeviceType::CPU);
      }
      if (i < 15) {
        auto insert_tb1 =
            "insert into tb3 values (" + ::toString(val) + "," + ::toString(val) + ");";
        QR::get()->runSQL(insert_tb1, ExecutorDeviceType::CPU);
      }
      auto insert_tb2 =
          "insert into tb4 values (" + ::toString(val) + "," + ::toString(val) + ");";
      QR::get()->runSQL(insert_tb2, ExecutorDeviceType::CPU);
    }
  };

  auto gen_perfect_hash_query =
      [](const std::string& inner_table_name,
         const std::string& outer_table_name) -> const std::string {
    return "SELECT COUNT(1) FROM " + inner_table_name + " t1, " + outer_table_name +
           " t2 WHERE t1.x = t2.x;";
  };

  auto gen_baseline_hash_query =
      [](const std::string& inner_table_name,
         const std::string& outer_table_name) -> const std::string {
    return "SELECT COUNT(1) FROM " + inner_table_name + " t1, " + outer_table_name +
           " t2 WHERE t1.x = t2.x and t1.y = t2.y;";
  };

  auto gen_update_query = [](const std::string& targe_table_name) -> const std::string {
    return "UPDATE " + targe_table_name + " SET x = 1988 WHERE x = 1;";
  };

  auto gen_drop_query = [](const std::string& targe_table_name) -> const std::string {
    return "DROP TABLE IF EXISTS " + targe_table_name + ";";
  };

  for (auto dt : {ExecutorDeviceType::CPU}) {
    drop_tables();
    prepare_tables();
    clearCache();

    // execute perfect hash join query, hashtable is built on tf1
    QR::get()->runSQL(gen_perfect_hash_query("tf1", "tf2"), dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));

    // execute baseline hash join query, hashtable is built on tb1
    QR::get()->runSQL(gen_baseline_hash_query("tb1", "tb2"), dt);
    // perfect hashtable cache remains the same
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::BASELINE_HT));

    // execute perfect hash join query, hashtable is built on tf2
    QR::get()->runSQL(gen_perfect_hash_query("tf2", "tf3"), dt);
    // perfect hashtable cache has two items: tf1 and tf2
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));

    // execute baseline hash join query, hashtable is built on tb2
    QR::get()->runSQL(gen_baseline_hash_query("tb2", "tb3"), dt);
    // baseline hashtable cache has two items: tb1 and tb2
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::BASELINE_HT));

    // execute perfect hash join query, hashtable is built on tb2
    QR::get()->runSQL(gen_perfect_hash_query("tb2", "tb3"), dt);
    // perfect hashtable cache has three items: tf1, tf2 and tb2
    EXPECT_EQ(static_cast<size_t>(3),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    // baseline hashtable cache remains the same: tb1 and tb2
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::BASELINE_HT));

    // execute update query on tb2
    QR::get()->runSQL(gen_update_query("tb2"), dt);
    // perfect hashtable cache has two clean items: tf1 and tf2
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    // perfect hashtable cache has one dirty item: tb2
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT));
    // baseline hashtable cache has one dirty item: tb2
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::BASELINE_HT));
    // baseline hashtable cache has total two cached item: tb1 (clean) and tb2 (dirty)
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::BASELINE_HT));

    // execute perfect hash join query, hashtable is built on tb2 and dirty item is
    // removed
    QR::get()->runSQL(gen_perfect_hash_query("tb2", "tb3"), dt);
    // perfect hashtable cache has three clean items: tf1, tf2 and tb2
    EXPECT_EQ(static_cast<size_t>(3),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(0),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(3),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT));
    // hashtable on tb2 is still dirty in the baseline hashtable cache
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::BASELINE_HT));

    // execute baseline hash join query, hashtable is built on tb2 and dirty item is
    // removed
    QR::get()->runSQL(gen_baseline_hash_query("tb2", "tb3"), dt);
    // baseline hashtable cache has two clean items: tb1 and tb2
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::BASELINE_HT));
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::BASELINE_HT));

    // execute perfect hash join query, hashtable is built on tb3
    QR::get()->runSQL(gen_perfect_hash_query("tf3", "tf4"), dt);
    // perfect hashtable cache has four clean items: tf1, tf2, tb2 and tf3
    EXPECT_EQ(static_cast<size_t>(4),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));

    // execute baseline hash join query, hashtable is built on tb3
    QR::get()->runSQL(gen_baseline_hash_query("tb3", "tb4"), dt);
    // baseline hashtable cache has three clean items: tb1, tb2 and tf3
    EXPECT_EQ(static_cast<size_t>(3),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::BASELINE_HT));

    // execute drop table query on tf4, but nothing is changed since we have no
    // cached item on tf4
    run_ddl_statement(gen_drop_query("tf4"));
    EXPECT_EQ(static_cast<size_t>(4),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    // execute drop table query on tb4, but nothing is changed since we have no
    // cached item on tb4
    run_ddl_statement(gen_drop_query("tb4"));
    EXPECT_EQ(static_cast<size_t>(3),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::BASELINE_HT));

    clearCache();

    // string (with dictionaries)
    drop_tables_for_string_joins();
    prepare_tables_for_string_joins();

    // built a hashtable on ts1
    QR::get()->runSQL(gen_perfect_hash_query("ts1", "ts2"), dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    // drop table ts1, hashtable on it will be marked dirty
    run_ddl_statement(gen_drop_query("ts1"));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT));

    // built a hashtable on ts4 (|ts4| < |ts3|)
    QR::get()->runSQL(gen_perfect_hash_query("ts4", "ts3"), dt);
    // we only have a single clean cached item: ts4
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));

    QR::get()->runSQL(gen_perfect_hash_query("ts3", "ts2"), dt);
    // we have a two clean cached items: ts4 and ts2
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    // we still have a dirty cached item: ts1
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT));

    // drop table ts2
    run_ddl_statement(gen_drop_query("ts2"));
    // ts3 is built based on join qual with ts2, so its cached item will be marked dirty
    // b/c they share dictionaries
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(3),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT));

    QR::get()->runSQL(gen_perfect_hash_query("tns1", "tns2"), dt);
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(4),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT));

    clearCache();
    drop_tables_for_string_joins();
    prepare_tables_for_string_joins();

    // join between dict encoded columns, ts1 and ts2: share dict1, ts3 and ts4: shared
    // dict2 hashtable based on qual ts1-ts2 is cached, dict1 is referenced
    QR::get()->runSQL(gen_perfect_hash_query("ts1", "ts2"), dt);
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    // hashtable based on qual ts1-ts3 is cached, dict2 of ts3 is referenced
    QR::get()->runSQL(gen_perfect_hash_query("ts1", "ts3"), dt);
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));

    // here, ts4 shares dictionary with ts3, so the hashtable for the qual ts1-ts4 is the
    // same as ts1-ts3, e.g., use dict2 and |ts1| < |ts2| and |ts1| < |ts4|, but we
    // separate them so hashtable for ts1-ts4 is classified as differently as that of
    // ts1-ts3 in hashtable recycler
    QR::get()->runSQL(gen_perfect_hash_query("ts1", "ts4"), dt);
    // hashtables for 1) ts1-ts2, 2) ts1-ts3 and 3) ts1-ts4
    EXPECT_EQ(static_cast<size_t>(3),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));

    // drop table ts3, so cached hashtable of ts1-ts3 is marked dirty
    run_ddl_statement(gen_drop_query("ts3"));
    // remaining two cached items are in clean state
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT));

    // drop table ts2, so cached hashtable of ts2-ts3 is also marked dirty
    run_ddl_statement(gen_drop_query("ts2"));
    // now all cached hashtables are marked dirty and no clean hashtable exists
    EXPECT_EQ(static_cast<size_t>(1),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
    EXPECT_EQ(static_cast<size_t>(2),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT));

    drop_tables_for_string_joins();
    prepare_tables_for_string_joins();

    // extreme case
    QR::get()->runSQL(gen_perfect_hash_query("ts1", "ts2"), dt);
    QR::get()->runSQL(gen_perfect_hash_query("ts1", "ts3"), dt);
    QR::get()->runSQL(gen_perfect_hash_query("ts1", "tns1"), dt);
    QR::get()->runSQL(gen_perfect_hash_query("ts1", "tns2"), dt);
    run_ddl_statement(gen_drop_query("ts1"));
    EXPECT_EQ(static_cast<size_t>(0),
              QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT));
  }
  drop_tables();
  drop_tables_for_string_joins();
}

TEST(DataRecycler, MetricTrackerTest) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto& resultset_recycler_holder = executor->getResultSetRecyclerHolder();
  auto resultset_recycler = resultset_recycler_holder.getResultSetRecycler();
  CHECK(resultset_recycler);
  auto& metric_tracker = resultset_recycler->getResultSetRecyclerMetricTracker();

  ScopeGuard reset_status = [orig_max_size = g_max_cacheable_query_resultset_size_bytes,
                             orig_total_size = g_query_resultset_cache_total_bytes,
                             &metric_tracker] {
    metric_tracker.setMaxCacheItemSize(orig_max_size);
    metric_tracker.setTotalCacheSize(orig_total_size);
  };

  metric_tracker.setTotalCacheSize(20);
  metric_tracker.setMaxCacheItemSize(15);

  QueryPlanHash dummy_key{1};
  DeviceIdentifier dummy_id{DataRecyclerUtil::CPU_DEVICE_IDENTIFIER};

  auto add_dummy_item_to_cache =
      [&dummy_key, &dummy_id, &metric_tracker](size_t dummy_size) {
        metric_tracker.putNewCacheItemMetric(dummy_key++, dummy_id, dummy_size, 0);
      };

  // case 1. add new item to an empty cache
  {
    add_dummy_item_to_cache(1);
    EXPECT_EQ(*metric_tracker.getCurrentCacheSize(dummy_id), 1UL);

    add_dummy_item_to_cache(2);
    EXPECT_EQ(*metric_tracker.getCurrentCacheSize(dummy_id), 3UL);

    metric_tracker.clearCacheMetricTracker();
  }

  // case 2. try to add item when its size is larger than size limitation
  // a) larger than per-item maximum limit
  {
    CacheAvailability ca1 = metric_tracker.canAddItem(dummy_id, 16);
    // b) larger than total cache size
    CacheAvailability ca2 = metric_tracker.canAddItem(dummy_id, 21);
    EXPECT_EQ(ca1, CacheAvailability::UNAVAILABLE);
    EXPECT_EQ(ca2, CacheAvailability::UNAVAILABLE);

    metric_tracker.clearCacheMetricTracker();
  }

  // case 3. complicated cases
  {
    // fill caches, current cache size = 15 (=1+2+3+4+5)
    for (size_t sz = 1; sz <= 5; sz++) {
      add_dummy_item_to_cache(sz);
    }
    EXPECT_EQ(metric_tracker.getCurrentCacheSize(dummy_id), 15UL);
    // a) total cache size will be 11
    CacheAvailability ca1 = metric_tracker.canAddItem(dummy_id, 1);
    EXPECT_EQ(ca1, CacheAvailability::AVAILABLE);
    // b) total cache size will be 20
    CacheAvailability ca2 = metric_tracker.canAddItem(dummy_id, 5);
    EXPECT_EQ(ca2, CacheAvailability::AVAILABLE);
    // c) need to cleanup few items to add the item of size 10
    CacheAvailability ca3 = metric_tracker.canAddItem(dummy_id, 10);
    EXPECT_EQ(ca3, CacheAvailability::AVAILABLE_AFTER_CLEANUP);
    // d) impossible since it is larger than per-max table cached item limit
    CacheAvailability ca4 = metric_tracker.canAddItem(dummy_id, 16);
    EXPECT_EQ(ca4, CacheAvailability::UNAVAILABLE);
  }
}

TEST(DataRecycler, LargeHashTable) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  executor->clearMemory(MemoryLevel::CPU_LEVEL);
  executor->getQueryPlanDagCache().clearQueryPlanCache();
  auto drop_tables = [] {
    run_ddl_statement("DROP TABLE IF EXISTS ct1;");
    run_ddl_statement("DROP TABLE IF EXISTS ct2;");
  };
  drop_tables();
  auto const dt = ExecutorDeviceType::CPU;
  run_ddl_statement("CREATE TABLE ct1 (v int);");
  run_ddl_statement("CREATE TABLE ct2 (v int);");
  QR::get()->runSQL("INSERT INTO ct1 VALUES (1);", dt);
  QR::get()->runSQL("INSERT INTO ct1 VALUES (1000000000);", dt);
  QR::get()->runSQL("INSERT INTO ct2 VALUES (2);", dt);
  QR::get()->runSQL("INSERT INTO ct2 VALUES (1000000000);", dt);
  ScopeGuard reset = [orig1 = g_hashtable_cache_total_bytes,
                      orig2 = g_max_cacheable_hashtable_size_bytes] {
    PerfectJoinHashTable::getHashTableCache()->setTotalCacheSize(
        CacheItemType::PERFECT_HT, orig1);
    PerfectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
        CacheItemType::PERFECT_HT, orig2);
  };
  auto q = "select count(1) from ct1, ct2 where ct1.v = ct2.v;";
  QR::get()->runSQL(q, dt);
  EXPECT_EQ(static_cast<size_t>(0),
            QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT));
  PerfectJoinHashTable::getHashTableCache()->setMaxCacheItemSize(
      CacheItemType::PERFECT_HT, 8000000000);
  PerfectJoinHashTable::getHashTableCache()->setTotalCacheSize(CacheItemType::PERFECT_HT,
                                                               8000000000);
  QR::get()->runSQL(q, dt);
  EXPECT_EQ(static_cast<size_t>(1),
            QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT));
  drop_tables();
}

class HashtableRecyclerTest : public ::testing::Test {
 protected:
  constexpr static int num_gpus_ = 1;

  HashtableRecyclerTest()
      : hashtable_recycler_(CacheItemType::BBOX_INTERSECT_HT, num_gpus_), meta_info_{} {
    meta_info_.bbox_intersect_meta_info = BoundingBoxIntersectMetaInfo{};
  }

  HashtableRecycler hashtable_recycler_;

  constexpr static size_t item_size_ = 1024;
  constexpr static size_t compute_time_ = 100;
  constexpr static CacheItemType item_type_ = CacheItemType::BBOX_INTERSECT_HT;
  constexpr static DeviceIdentifier device_identifier_ =
      DataRecyclerUtil::CPU_DEVICE_IDENTIFIER;

  // Actual values seen during test run in QE-1250 description.
  BaselineHashTableEntryInfo hash_table_entry_info_{
      26818,                // num_hash_entries_ = entry_count
      16076,                // num_keys_ = emitted_keys_count
      sizeof(int32_t),      // rowid_size_in_bytes_
      2,                    // num_join_keys_ = getKeyComponentCount()
      8,                    // join_key_size_in_byte_ = getKeyComponentWidth()
      HashType::OneToMany,  // layout
      false};               // for_window_framing_

  HashtableCacheMetaInfo meta_info_;
};

TEST_F(HashtableRecyclerTest, PutAndGetCacheItem) {
  constexpr QueryPlanHash key = 1234;
  auto hash_table = std::make_shared<BaselineHashTable>(
      MemoryLevel::CPU_LEVEL, hash_table_entry_info_, nullptr, -1);

  hashtable_recycler_.putItemToCache(key,
                                     hash_table,
                                     item_type_,
                                     device_identifier_,
                                     item_size_,
                                     compute_time_,
                                     meta_info_);

  auto retrieved_item = hashtable_recycler_.getItemFromCache(
      key, item_type_, device_identifier_, meta_info_);
  EXPECT_EQ(retrieved_item.get(), hash_table.get());
}

TEST_F(HashtableRecyclerTest, PutTwoItemsWithSameKey) {
  constexpr QueryPlanHash key = 1234;
  auto hash_table = std::make_shared<BaselineHashTable>(
      MemoryLevel::CPU_LEVEL, hash_table_entry_info_, nullptr, -1);

  hashtable_recycler_.putItemToCache(key,
                                     hash_table,
                                     item_type_,
                                     device_identifier_,
                                     item_size_,
                                     compute_time_,
                                     meta_info_);

  auto hash_table_2 = std::make_shared<BaselineHashTable>(
      MemoryLevel::CPU_LEVEL, hash_table_entry_info_, nullptr, -1);
  HashtableCacheMetaInfo meta_info_2{};
  meta_info_2.bbox_intersect_meta_info = BoundingBoxIntersectMetaInfo{2};
  hashtable_recycler_.putItemToCache(key,
                                     hash_table_2,
                                     item_type_,
                                     device_identifier_,
                                     item_size_,
                                     compute_time_,
                                     meta_info_2);

  auto retrieved_item = hashtable_recycler_.getItemFromCache(
      key, item_type_, device_identifier_, meta_info_);
  EXPECT_EQ(retrieved_item.get(), hash_table.get());

  auto retrieved_item_2 = hashtable_recycler_.getItemFromCache(
      key, item_type_, device_identifier_, meta_info_2);
  EXPECT_EQ(retrieved_item_2.get(), hash_table_2.get());
}

TEST_F(HashtableRecyclerTest, PutTenItemsWithSameKey) {
  constexpr QueryPlanHash key = 1234;

  // Add 10 items to cache
  constexpr size_t N = 10;
  std::array<void*, N> hash_tables;
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(
        i, hashtable_recycler_.getCurrentNumCachedItems(item_type_, device_identifier_));
    auto hash_table = std::make_shared<BaselineHashTable>(
        MemoryLevel::CPU_LEVEL, hash_table_entry_info_, nullptr, -1);
    hash_tables[i] = hash_table.get();
    HashtableCacheMetaInfo meta_info{};
    meta_info.bbox_intersect_meta_info = BoundingBoxIntersectMetaInfo{i};
    hashtable_recycler_.putItemToCache(key,
                                       hash_table,
                                       item_type_,
                                       device_identifier_,
                                       item_size_,
                                       compute_time_,
                                       meta_info);
  }
  EXPECT_EQ(N,
            hashtable_recycler_.getCurrentNumCachedItems(item_type_, device_identifier_));

  // Verify all hash table pointers are distinct
  std::unordered_set<void*> const distinct_values(hash_tables.begin(), hash_tables.end());
  EXPECT_EQ(N, distinct_values.size());

  // Add same 10 meta_info again, and verify size does not change.
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(
        N, hashtable_recycler_.getCurrentNumCachedItems(item_type_, device_identifier_));
    auto hash_table = std::make_shared<BaselineHashTable>(
        MemoryLevel::CPU_LEVEL, hash_table_entry_info_, nullptr, -1);
    HashtableCacheMetaInfo meta_info{};
    meta_info.bbox_intersect_meta_info = BoundingBoxIntersectMetaInfo{i};
    hashtable_recycler_.putItemToCache(key,
                                       hash_table,
                                       item_type_,
                                       device_identifier_,
                                       item_size_,
                                       compute_time_,
                                       meta_info);
  }
  EXPECT_EQ(N,
            hashtable_recycler_.getCurrentNumCachedItems(item_type_, device_identifier_));

  // Verify correct items are retrieved.
  for (size_t i = 0; i < N; ++i) {
    auto hash_table = std::make_shared<BaselineHashTable>(
        MemoryLevel::CPU_LEVEL, hash_table_entry_info_, nullptr, -1);
    HashtableCacheMetaInfo meta_info{};
    meta_info.bbox_intersect_meta_info = BoundingBoxIntersectMetaInfo{i};
    auto retrieved_item = hashtable_recycler_.getItemFromCache(
        key, item_type_, device_identifier_, meta_info);
    EXPECT_EQ(hash_tables[i], retrieved_item.get());
  }
}

TEST_F(HashtableRecyclerTest, ClearCache) {
  constexpr QueryPlanHash key = 91011;
  auto hash_table = std::make_shared<BaselineHashTable>(
      MemoryLevel::CPU_LEVEL, hash_table_entry_info_, nullptr, -1);

  hashtable_recycler_.putItemToCache(key,
                                     hash_table,
                                     item_type_,
                                     device_identifier_,
                                     item_size_,
                                     compute_time_,
                                     meta_info_);
  hashtable_recycler_.clearCache();

  auto retrieved_item = hashtable_recycler_.getItemFromCache(
      key, item_type_, device_identifier_, meta_info_);
  EXPECT_EQ(retrieved_item, nullptr);
}

TEST_F(HashtableRecyclerTest, MarkCachedItemAsDirty) {
  constexpr QueryPlanHash key = 121314;
  auto hash_table = std::make_shared<BaselineHashTable>(
      MemoryLevel::CPU_LEVEL, hash_table_entry_info_, nullptr, -1);
  std::unordered_set<QueryPlanHash> key_set = {key};

  hashtable_recycler_.putItemToCache(
      key, hash_table, item_type_, device_identifier_, item_size_, compute_time_);
  hashtable_recycler_.markCachedItemAsDirty(0, key_set, item_type_, device_identifier_);

  auto retrieved_item =
      hashtable_recycler_.getItemFromCache(key, item_type_, device_identifier_);
  EXPECT_EQ(retrieved_item, nullptr);
}

TEST_F(HashtableRecyclerTest, GetCachedHashtableWithoutCacheKey) {
  constexpr QueryPlanHash key = 151617;
  auto hash_table = std::make_shared<BaselineHashTable>(
      MemoryLevel::CPU_LEVEL, hash_table_entry_info_, nullptr, -1);
  std::set<size_t> visited;

  hashtable_recycler_.putItemToCache(
      key, hash_table, item_type_, device_identifier_, item_size_, compute_time_);

  auto [retrieved_key, retrieved_item, meta_info] =
      hashtable_recycler_.getCachedHashtableWithoutCacheKey(
          visited, item_type_, device_identifier_);

  EXPECT_EQ(retrieved_item, hash_table);
  EXPECT_EQ(retrieved_key, key);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  TestHelpers::init_logger_stderr_only(argc, argv);

  ScopeGuard reset_flag_state = [orig_table_function = g_enable_table_functions,
                                 orig_dev_table_function = g_enable_dev_table_functions,
                                 orig_test_env = g_is_test_env] {
    g_enable_table_functions = orig_table_function;
    g_enable_dev_table_functions = orig_dev_table_function;
    g_is_test_env = orig_test_env;
  };
  g_enable_table_functions = true;
  g_enable_dev_table_functions = true;
  g_is_test_env = true;

  QR::init(BASE_PATH);

  int err{0};
  try {
    err = drop_table();
    err = create_and_populate_table();
    err = RUN_ALL_TESTS();
    err = drop_table();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  g_is_test_env = false;
  QR::reset();
  return err;
}
