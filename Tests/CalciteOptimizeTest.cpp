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
#include <boost/filesystem/operations.hpp>
#include <csignal>
#include <thread>
#include <tuple>

#include "../Catalog/Catalog.h"
#include "../Catalog/DBObject.h"
#include "../DataMgr/DataMgr.h"
#include "../Parser/parser.h"
#include "../QueryRunner/QueryRunner.h"
#include "ThriftHandler/QueryState.h"
#include "gen-cpp/CalciteServer.h"

#include "Shared/Restriction.h"

using QR = QueryRunner::QueryRunner;

namespace {

std::shared_ptr<Calcite> g_calcite;

inline void run_ddl_statement(const std::string& query) {
  QR::get()->runDDLStatement(query);
}

}  // namespace

struct ViewObject : testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP VIEW IF EXISTS view_view_table1;");
    run_ddl_statement("DROP VIEW IF EXISTS view_table1;");
    run_ddl_statement("DROP TABLE IF EXISTS table1");
    run_ddl_statement("DROP VIEW IF EXISTS attribute_view");
    run_ddl_statement("DROP VIEW IF EXISTS shape_view");
    run_ddl_statement("DROP TABLE IF EXISTS shape_table");
    run_ddl_statement("DROP TABLE IF EXISTS attribute_table");
    run_ddl_statement("DROP VIEW IF EXISTS attribute_shape_view");
    run_ddl_statement("DROP VIEW IF EXISTS left_join_3tables");

    run_ddl_statement("CREATE TABLE table1(i1 integer, i2 integer);");
    run_ddl_statement("CREATE VIEW view_table1 AS SELECT i1, i2 FROM table1;");
    run_ddl_statement("CREATE VIEW view_view_table1 AS SELECT i1, i2 FROM view_table1;");
    run_ddl_statement("CREATE TABLE shape_table (block_group_id INT)");
    run_ddl_statement(
        "CREATE TABLE attribute_table( block_group_id INT, segment_name TEXT ENCODING "
        "DICT(8), segment_type TEXT ENCODING DICT(8), agg_column TEXT ENCODING DICT(8))");
    run_ddl_statement(
        "CREATE VIEW attribute_view AS select "
        "rowid,block_group_id,segment_name,segment_type,agg_column from attribute_table");
    run_ddl_statement(
        "CREATE VIEW shape_view AS select rowid, block_group_id from shape_table");
    run_ddl_statement(
        R"(CREATE VIEW attribute_shape_view AS SELECT * FROM attribute_table INNER JOIN shape_table ON attribute_table.block_group_id = shape_table.block_group_id;)");
    run_ddl_statement(
        R"(CREATE VIEW left_join_3tables AS SELECT i1 FROM table1 LEFT JOIN attribute_shape_view ON table1.i1 = attribute_shape_view.block_group_id;)");
  }

  void TearDown() override {
    run_ddl_statement("DROP VIEW view_view_table1;");
    run_ddl_statement("DROP VIEW view_table1;");
    run_ddl_statement("DROP TABLE table1");
    run_ddl_statement("DROP VIEW attribute_view");
    run_ddl_statement("DROP VIEW shape_view");
    run_ddl_statement("DROP TABLE shape_table");
    run_ddl_statement("DROP TABLE attribute_table");
    run_ddl_statement("DROP VIEW attribute_shape_view");
    run_ddl_statement("DROP VIEW left_join_3tables");
  }
};

TEST_F(ViewObject, BasicTest) {
  auto session = QR::get()->getSession();
  CHECK(session);

  auto qs1 = QR::create_query_state(session, "select i1 from table1");
  TPlanResult tresult = g_calcite->process(
      qs1->createQueryStateProxy(), qs1->getQueryStr(), {}, true, false, false, true);

  auto qs2 = QR::create_query_state(session, "select i1 from view_view_table1");
  TPlanResult vresult = g_calcite->process(
      qs2->createQueryStateProxy(), qs2->getQueryStr(), {}, true, false, false, true);

  EXPECT_EQ(vresult.plan_result, tresult.plan_result);

  auto qs3 = QR::create_query_state(session, "select i1 from view_view_table1");
  TPlanResult ovresult = g_calcite->process(
      qs3->createQueryStateProxy(), qs3->getQueryStr(), {}, true, false, true, true);

  EXPECT_EQ(ovresult.plan_result, tresult.plan_result);

  auto qs4 = QR::create_query_state(
      session,
      R"(SELECT shape_table.rowid FROM shape_table, attribute_table WHERE shape_table.block_group_id = attribute_table.block_group_id)");
  TPlanResult tab_result = g_calcite->process(
      qs4->createQueryStateProxy(), qs4->getQueryStr(), {}, true, false, true, true);

  auto qs5 = QR::create_query_state(
      session,
      R"(SELECT shape_view.rowid FROM shape_view, attribute_view WHERE shape_view.block_group_id = attribute_view.block_group_id)");
  TPlanResult view_result = g_calcite->process(
      qs5->createQueryStateProxy(), qs5->getQueryStr(), {}, true, false, true, true);
  EXPECT_EQ(tab_result.plan_result, view_result.plan_result);
}

TEST_F(ViewObject, Joins) {
  auto session = QR::get()->getSession();
  CHECK(session);

  {
    auto qs1 = QR::create_query_state(
        session,
        R"(SELECT i1 FROM table1 LEFT JOIN attribute_shape_view ON table1.i1 = attribute_shape_view.block_group_id)");
    TPlanResult tresult = g_calcite->process(
        qs1->createQueryStateProxy(), qs1->getQueryStr(), {}, true, false, true, true);

    auto qs2 = QR::create_query_state(session, "SELECT i1 FROM left_join_3tables");
    TPlanResult vresult = g_calcite->process(
        qs2->createQueryStateProxy(), qs2->getQueryStr(), {}, true, false, true, true);

    EXPECT_EQ(vresult.plan_result, tresult.plan_result);
  }
}

TEST_F(ViewObject, Restrict) {
  auto session = QR::get()->getSession();
  CHECK(session);

  {
    auto qs1 = QR::create_query_state(
        session,
        R"(SELECT segment_name FROM attribute_table where segment_name = 'ab' or segment_name = 'ac')");
    TPlanResult tresult = g_calcite->process(
        qs1->createQueryStateProxy(), qs1->getQueryStr(), {}, true, false, true, true);

    std::string col = "segment_name";
    std::vector<std::string> values = {"ab", "ac"};
    auto rest = std::make_shared<Restriction>(col, values);
    session->set_restriction(rest);

    auto qs2 =
        QR::create_query_state(session, "SELECT segment_name FROM attribute_table");
    TPlanResult resResult = g_calcite->process(
        qs2->createQueryStateProxy(), qs2->getQueryStr(), {}, true, false, true, true);

    EXPECT_EQ(tresult.plan_result, resResult.plan_result);

    auto qs3 = QR::create_query_state(session, R"(select i1 from table1 where i1 = 1)");
    TPlanResult riResult = g_calcite->process(
        qs3->createQueryStateProxy(), qs3->getQueryStr(), {}, true, false, true, true);

    col = "i1";
    values = {"1"};
    rest = std::make_shared<Restriction>(col, values);
    session->set_restriction(rest);

    auto qs4 = QR::create_query_state(session, "select i1 from table1");
    TPlanResult rrResult = g_calcite->process(
        qs4->createQueryStateProxy(), qs4->getQueryStr(), {}, true, false, true, true);

    EXPECT_EQ(riResult.plan_result, rrResult.plan_result);
  }
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  QR::init(BASE_PATH);
  g_calcite = QR::get()->getCatalog()->getCalciteMgr();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
