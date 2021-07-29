/*
 * Copyright 2021, OmniSci, Inc.
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
#include "QueryEngine/Execute.h"
#include "QueryEngine/QueryPlanDagExtractor.h"

#include "QueryRunner/QueryRunner.h"

#include <gtest/gtest.h>
#include <boost/algorithm/string/join.hpp>

#include <exception>
#include <future>
#include <stdexcept>

using QR = QueryRunner::QueryRunner;

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

namespace {

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

int create_and_populate_table() {
  try {
    run_ddl_statement("DROP TABLE IF EXISTS T1;");
    run_ddl_statement("DROP TABLE IF EXISTS T2;");
    run_ddl_statement("DROP TABLE IF EXISTS T3;");
    run_ddl_statement("DROP TABLE IF EXISTS T4;");
    const auto table_ddl =
        "(x int not null, w tinyint, y int, z smallint, t bigint, b boolean, f float, ff "
        "float, fn float, d double, dn double, str varchar(10), "
        "null_str text encoding dict, fixed_str text encoding dict(16), fixed_null_str "
        "text encoding dict(16), real_str text encoding none, shared_dict text, m "
        "timestamp(0), m_3 timestamp(3), m_6 timestamp(6), m_9 timestamp(9), n time(0), "
        "o date, o1 date encoding fixed(16), o2 date encoding fixed(32), fx int encoding "
        "fixed(16), dd decimal(10, 2), dd_notnull decimal(10, 2) not null, ss text "
        "encoding dict, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not "
        "null, smallint_nulls smallint, bn boolean not null);";
    auto create_table_ddl = [&table_ddl](const std::string& tbl_name) {
      return "CREATE TABLE " + tbl_name + table_ddl;
    };
    const std::string row1{
        "VALUES(7, -8, 42, 101, 1001, 't', 1.1, 1.1, null, 2.2, null, "
        "'foo', null, 'foo', null, "
        "'real_foo', 'foo',"
        "'2014-12-13 22:23:15', '2014-12-13 22:23:15.323', '1999-07-11 "
        "14:02:53.874533', "
        "'2006-04-26 "
        "03:49:04.607435125', "
        "'15:13:14', '1999-09-09', '1999-09-09', '1999-09-09', 9, 111.1, 111.1, "
        "'fish', "
        "null, "
        "2147483647, -2147483648, null, -1, 32767, 't');"};
    const std::string row2{
        "VALUES(8, -7, 43, -78, 1002, 'f', 1.2, 101.2, -101.2, 2.4, "
        "-2002.4, 'bar', null, 'bar', null, "
        "'real_bar', NULL, '2014-12-13 22:23:15', '2014-12-13 22:23:15.323', "
        "'2014-12-13 "
        "22:23:15.874533', "
        "'2014-12-13 22:23:15.607435763', '15:13:14', NULL, NULL, NULL, NULL, 222.2, "
        "222.2, "
        "null, null, null, "
        "-2147483647, "
        "9223372036854775807, -9223372036854775808, null, 'f');"};
    const std::string row3{
        "VALUES(7, -7, 43, 102, 1002, null, 1.3, 1000.3, -1000.3, "
        "2.6, "
        "-220.6, 'baz', null, null, null, "
        "'real_baz', 'baz', '2014-12-14 22:23:15', '2014-12-14 22:23:15.750', "
        "'2014-12-14 22:23:15.437321', "
        "'2014-12-14 22:23:15.934567401', '15:13:14', '1999-09-09', '1999-09-09', "
        "'1999-09-09', 11, "
        "333.3, 333.3, "
        "'boat', null, 1, "
        "-1, 1, -9223372036854775808, 1, 't');"};
    const auto insert_values = [&row1, &row2, &row3](const std::string& tbl_name) {
      auto insert_row1_stmt = "INSERT INTO " + tbl_name + " " + row1;
      auto insert_row2_stmt = "INSERT INTO " + tbl_name + " " + row3;
      auto insert_row3_stmt = "INSERT INTO " + tbl_name + " " + row2;
      QR::get()->runSQL(insert_row1_stmt, ExecutorDeviceType::CPU, false, false);
      QR::get()->runSQL(insert_row2_stmt, ExecutorDeviceType::CPU, false, false);
      QR::get()->runSQL(insert_row3_stmt, ExecutorDeviceType::CPU, false, false);
    };
    run_ddl_statement(create_table_ddl("T1"));
    run_ddl_statement(create_table_ddl("T2"));
    run_ddl_statement(create_table_ddl("T3"));
    run_ddl_statement(create_table_ddl("T4"));
    insert_values("T1");
    insert_values("T2");
    insert_values("T3");
    insert_values("T4");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table";
    return -1;
  }
  return 0;
}

int drop_table() {
  try {
    run_ddl_statement("DROP TABLE IF EXISTS T1;");
    run_ddl_statement("DROP TABLE IF EXISTS T2;");
    run_ddl_statement("DROP TABLE IF EXISTS T3;");
    run_ddl_statement("DROP TABLE IF EXISTS T4;");
  } catch (...) {
    LOG(ERROR) << "Failed to drop table";
    return -1;
  }
  return 0;
}

}  // namespace

TEST(DataRecycler, QueryPlanDagExtractor_Simple_Project_Query) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  auto q1_str = "SELECT x FROM T1 ORDER BY x;";
  auto q1_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q1_str);
  CHECK(q1_query_info.left_deep_trees_id.empty());
  auto q1_rel_alg_translator = QR::get()->getRelAlgTranslator(q1_str, executor);
  auto q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q1_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q1_rel_alg_translator);
  // 1. a sort node becomes a root (dag_rel_id = 0)
  // 2. a project node becomes a child of the sort node (dag_rel_id = 1)
  // 3. a scan node (the leaf of the query plan) becomes a child of the project node
  CHECK(q1_plan_dag.extracted_dag.compare("0|1|2|") == 0);

  auto q2_str = "SELECT x FROM T1;";
  auto q2_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q2_str);
  CHECK(q2_query_info.left_deep_trees_id.empty());
  auto q2_rel_alg_translator = QR::get()->getRelAlgTranslator(q2_str, executor);
  auto q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q2_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q2_rel_alg_translator);
  // q2 is the same as q1 except sort node
  CHECK(q2_plan_dag.extracted_dag.compare("1|2|") == 0);

  auto q3_str = "SELECT x FROM T1 GROUP BY x;";
  auto q3_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q3_str);
  CHECK(q3_query_info.left_deep_trees_id.empty());
  auto q3_rel_alg_translator = QR::get()->getRelAlgTranslator(q3_str, executor);
  auto q3_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q3_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q3_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q3_rel_alg_translator);
  // compound node becomes the root (dag_rel_id = 3), and the scan node
  // (that is the same node as both q1 and q2) is the leaf of the query plan
  CHECK(q3_plan_dag.extracted_dag.compare("3|2|") == 0);

  auto q4_str = "SELECT x FROM T1 GROUP BY x ORDER BY x;";
  auto q4_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q4_str);
  CHECK(q4_query_info.left_deep_trees_id.empty());
  auto q4_rel_alg_translator = QR::get()->getRelAlgTranslator(q4_str, executor);
  auto q4_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q4_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q4_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q4_rel_alg_translator);
  // this sort node has different input compared with that of q1
  // so we assign the new dag_rel_id (4) to the sort node
  CHECK(q4_plan_dag.extracted_dag.compare("4|3|2|") == 0);

  auto q1_dup_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q1_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q1_rel_alg_translator);
  CHECK(q1_dup_plan_dag.extracted_dag.compare("0|1|2|") == 0);

  auto q4_dup_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q4_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q4_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q4_rel_alg_translator);
  CHECK(q4_dup_plan_dag.extracted_dag.compare("4|3|2|") == 0);
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
  CHECK(q1_query_info.left_deep_trees_id.empty());
  auto rel_alg_translator_for_q1 = QR::get()->getRelAlgTranslator(q1_str, executor);
  auto q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q1_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *rel_alg_translator_for_q1);
  CHECK_EQ(q1_plan_dag.contain_not_supported_rel_node, false);
  // but we skip to extract a DAG for q2 since it contains IN-expr having 21 elems in its
  // value list

  auto q2_str = create_query_having_IN_expr("T1", "x", 21);
  auto q2_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q2_str);
  CHECK(q2_query_info.left_deep_trees_id.empty());
  auto rel_alg_translator_for_q2 = QR::get()->getRelAlgTranslator(q2_str, executor);
  auto q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q2_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *rel_alg_translator_for_q2);
  CHECK_EQ(q2_plan_dag.contain_not_supported_rel_node, true);
}

TEST(DataRecycler, QueryPlanDagExtractor_Join_Query) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();

  auto q1_str = "SELECT T1.x FROM T1, T2 WHERE T1.x = T2.x;";
  auto q1_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q1_str);
  CHECK(q1_query_info.left_deep_trees_id.size() == 1);
  auto q1_rel_alg_translator = QR::get()->getRelAlgTranslator(q1_str, executor);
  auto q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 q1_query_info.left_deep_trees_id[0],
                                                 q1_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q1_rel_alg_translator);

  auto q2_str = "SELECT T1.x FROM T1 JOIN T2 ON T1.x = T2.x;";
  auto q2_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q2_str);
  CHECK(q2_query_info.left_deep_trees_id.size() == 1);
  auto q2_rel_alg_translator = QR::get()->getRelAlgTranslator(q2_str, executor);
  auto q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 q2_query_info.left_deep_trees_id[0],
                                                 q2_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q2_rel_alg_translator);

  CHECK(q1_plan_dag.extracted_dag.compare(q2_plan_dag.extracted_dag) == 0);

  auto q3_str = "SELECT T1.x FROM T1, T2 WHERE T1.x = T2.x and T2.y = T1.y;";
  auto q3_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q3_str);
  CHECK(q3_query_info.left_deep_trees_id.size() == 1);
  auto q3_rel_alg_translator = QR::get()->getRelAlgTranslator(q3_str, executor);
  auto q3_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q3_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 q3_query_info.left_deep_trees_id[0],
                                                 q3_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q3_rel_alg_translator);

  auto q4_str = "SELECT T1.x FROM T1 JOIN T2 ON T1.x = T2.x and T1.y = T2.y;";
  auto q4_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q4_str);
  CHECK(q4_query_info.left_deep_trees_id.size() == 1);
  auto q4_rel_alg_translator = QR::get()->getRelAlgTranslator(q4_str, executor);
  auto q4_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q4_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 q4_query_info.left_deep_trees_id[0],
                                                 q4_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q4_rel_alg_translator);

  CHECK(q3_plan_dag.extracted_dag.compare(q4_plan_dag.extracted_dag) != 0);

  auto q5_str = "SELECT T1.x FROM T1 JOIN T2 ON T1.y = T2.y and T1.x = T2.x;";
  auto q5_query_info = QR::get()->getQueryInfoForDataRecyclerTest(q5_str);
  CHECK(q5_query_info.left_deep_trees_id.size() == 1);
  auto q5_rel_alg_translator = QR::get()->getRelAlgTranslator(q5_str, executor);
  auto q5_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q5_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 q5_query_info.left_deep_trees_id[0],
                                                 q5_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q5_rel_alg_translator);
  CHECK(q3_plan_dag.extracted_dag.compare(q5_plan_dag.extracted_dag) != 0);
}

TEST(DataRecycler, DAG_Cache_Size_Management) {
  // test if DAG cache becomes full
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  // get query info for DAG cache test in advance
  auto& DAG_CACHE = executor->getQueryPlanDagCache();
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
  auto q1_rel_alg_translator = QR::get()->getRelAlgTranslator(q1_str, executor);
  auto q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q1_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q1_rel_alg_translator);
  // 1. a sort node becomes a root (dag_rel_id = 0)
  // 2. a project node becomes a child of the sort node (dag_rel_id = 1)
  // 3. a scan node (the leaf of the query plan) becomes a child of the project node
  CHECK(q1_plan_dag.extracted_dag.compare("0|1|2|") == 0);
  // 3 unique REL nodes in the cache --> 3 * 2 * 8 = 48
  CHECK_EQ(DAG_CACHE.getCurrentNodeMapSize(), 48u);
  auto q2_rel_alg_translator = QR::get()->getRelAlgTranslator(q2_str, executor);
  auto q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q2_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q2_rel_alg_translator);
  // we set the DAG cache size be 48, so when we try to cache the q2, it becomes full
  // so it skips to extract DAG plan of this query and also clear the cache itself
  CHECK(q2_plan_dag.extracted_dag.compare("") == 0);
  // 2 unique REL nodes in the cache --> 2 * 2 * 8 = 32
  CHECK_EQ(DAG_CACHE.getCurrentNodeMapSize(), 0u);
  DAG_CACHE.clearQueryPlanCache();

  // test: when a query size is too large, we skip caching the query
  auto q3_rel_alg_translator = QR::get()->getRelAlgTranslator(q3_str, executor);
  auto q3_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q3_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 q3_query_info.left_deep_trees_id[0],
                                                 q3_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q3_rel_alg_translator);
  // q3 has more than three nodes, so its size is beyond the limit of the DAG cache (48)
  // so we cannot keep it to our DAG cache
  CHECK_EQ(DAG_CACHE.getCurrentNodeMapSize(), 0u);

  // test: increase the cache size that is enough to hold both q1 and q2
  DAG_CACHE.setNodeMapMaxSize(80);
  auto new_q1_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q1_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q1_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q1_rel_alg_translator);
  CHECK(new_q1_plan_dag.extracted_dag.compare("0|1|2|") == 0);
  auto new_q2_plan_dag =
      QueryPlanDagExtractor::extractQueryPlanDag(q2_query_info.root_node.get(),
                                                 *executor->getCatalog(),
                                                 std::nullopt,
                                                 q2_query_info.left_deep_trees_info,
                                                 {},
                                                 executor,
                                                 *q2_rel_alg_translator);
  CHECK(new_q2_plan_dag.extracted_dag.compare("3|2|") == 0);
  CHECK_GE(DAG_CACHE.getCurrentNodeMapSize(), 48u);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  TestHelpers::init_logger_stderr_only(argc, argv);

  QR::init(BASE_PATH);

  int err{0};
  try {
    err = create_and_populate_table();
    err = RUN_ALL_TESTS();
    err = drop_table();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
