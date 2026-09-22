/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <boost/filesystem/operations.hpp>
#include <csignal>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "../Catalog/Catalog.h"
#include "../Catalog/DBObject.h"
#include "../DataMgr/DataMgr.h"
#include "../QueryRunner/QueryRunner.h"
#include "ThriftHandler/QueryParsing.h"
#include "ThriftHandler/QueryState.h"
#include "gen-cpp/CalciteServer.h"

using QR = QueryRunner::QueryRunner;

namespace {

std::shared_ptr<Calcite> g_calcite;

inline void run_ddl_statement(const std::string& query) {
  QR::get()->runDDLStatement(query);
}

std::string rel_alg_plan(const std::string& query,
                         bool enable_experimental_query_rewrites = true,
                         bool trust_unenforced_table_constraints = true) {
  auto session = QR::get()->getSession();
  CHECK(session);

  auto calciteQueryParsingOption =
      g_calcite->getCalciteQueryParsingOption(true, false, false);
  auto calciteOptimizationOption =
      g_calcite->getCalciteOptimizationOption(false,
                                              false,
                                              {},
                                              enable_experimental_query_rewrites,
                                              trust_unenforced_table_constraints);
  auto query_state = QR::create_query_state(session, query);
  auto result = query_parsing::process_and_check_access_privileges(
      g_calcite.get(),
      query_state->createQueryStateProxy(),
      query_state->getQueryStr(),
      calciteQueryParsingOption,
      calciteOptimizationOption);
  return result.plan_result;
}

std::string calcite_explain_plan(const std::string& query,
                                 bool enable_experimental_query_rewrites = true,
                                 bool trust_unenforced_table_constraints = true) {
  auto session = QR::get()->getSession();
  CHECK(session);

  auto calciteQueryParsingOption =
      g_calcite->getCalciteQueryParsingOption(true, true, false);
  auto calciteOptimizationOption =
      g_calcite->getCalciteOptimizationOption(false,
                                              false,
                                              {},
                                              enable_experimental_query_rewrites,
                                              trust_unenforced_table_constraints);
  auto query_state = QR::create_query_state(session, query);
  auto result = query_parsing::process_and_check_access_privileges(
      g_calcite.get(),
      query_state->createQueryStateProxy(),
      query_state->getQueryStr(),
      calciteQueryParsingOption,
      calciteOptimizationOption);
  return result.plan_result;
}

size_t count_occurrences(const std::string& text, const std::string& needle) {
  CHECK(!needle.empty());
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

size_t count_aggregate_calls_with_operands(const rapidjson::Value& node,
                                           const std::string& aggregate_name) {
  size_t count{0};
  if (node.IsObject()) {
    const auto aggregate_it = node.FindMember("agg");
    const auto operands_it = node.FindMember("operands");
    if (aggregate_it != node.MemberEnd() && aggregate_it->value.IsString() &&
        aggregate_name == aggregate_it->value.GetString() &&
        operands_it != node.MemberEnd() && operands_it->value.IsArray() &&
        !operands_it->value.Empty()) {
      ++count;
    }
    for (auto member = node.MemberBegin(); member != node.MemberEnd(); ++member) {
      count += count_aggregate_calls_with_operands(member->value, aggregate_name);
    }
  } else if (node.IsArray()) {
    for (const auto& value : node.GetArray()) {
      count += count_aggregate_calls_with_operands(value, aggregate_name);
    }
  }
  return count;
}

void expect_plan_order(const std::string& plan,
                       const std::string& before,
                       const std::string& after) {
  const auto before_pos = plan.find(before);
  ASSERT_NE(std::string::npos, before_pos) << plan;
  const auto after_pos = plan.find(after);
  ASSERT_NE(std::string::npos, after_pos) << plan;
  EXPECT_LT(before_pos, after_pos) << plan;
}

void expect_join_type(const std::string& plan, const std::string& join_type) {
  EXPECT_NE(std::string::npos, plan.find("joinType=[" + join_type + "]")) << plan;
}

const std::string TEST_USER{"test_user"};
const std::string TEST_PASS{"test_pass"};
const std::string TEST_DB{"test_db"};

void canonicalize_literals(rapidjson::Value& node) {
  if (node.IsObject()) {
    auto literal_it = node.FindMember("literal");
    if (literal_it != node.MemberEnd() && literal_it->value.IsString()) {
      // Calcite 1.41 adds explicit type metadata to literal JSON. These access
      // policy tests compare plan shape, so normalize that metadata away.
      node.RemoveMember("target_type");
      node.RemoveMember("scale");
      node.RemoveMember("precision");
      node.RemoveMember("type_scale");
      node.RemoveMember("type_precision");
    }
    for (auto itr = node.MemberBegin(); itr != node.MemberEnd(); ++itr) {
      canonicalize_literals(itr->value);
    }
  } else if (node.IsArray()) {
    for (auto& elem : node.GetArray()) {
      canonicalize_literals(elem);
    }
  }
}

std::string normalize_plan_result(const std::string& plan) {
  rapidjson::Document doc;
  doc.Parse(plan.c_str());
  if (doc.HasParseError()) {
    return plan;
  }
  canonicalize_literals(doc);
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);
  return buffer.GetString();
}

void expect_equivalent_plan(const std::string& lhs, const std::string& rhs) {
  EXPECT_EQ(normalize_plan_result(lhs), normalize_plan_result(rhs));
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

  auto calciteQueryParsingOption =
      g_calcite->getCalciteQueryParsingOption(true, false, false);
  auto calciteOptimizationOption =
      g_calcite->getCalciteOptimizationOption(false, false, {});

  auto qs1 = QR::create_query_state(session, "select i1 from table1");
  TPlanResult tresult =
      query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                         qs1->createQueryStateProxy(),
                                                         qs1->getQueryStr(),
                                                         calciteQueryParsingOption,
                                                         calciteOptimizationOption);

  auto qs2 = QR::create_query_state(session, "select i1 from view_view_table1");
  TPlanResult vresult =
      query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                         qs2->createQueryStateProxy(),
                                                         qs2->getQueryStr(),
                                                         calciteQueryParsingOption,
                                                         calciteOptimizationOption);

  EXPECT_EQ(vresult.plan_result, tresult.plan_result);

  calciteOptimizationOption.is_view_optimize = true;
  auto qs3 = QR::create_query_state(session, "select i1 from view_view_table1");
  TPlanResult ovresult =
      query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                         qs3->createQueryStateProxy(),
                                                         qs3->getQueryStr(),
                                                         calciteQueryParsingOption,
                                                         calciteOptimizationOption);

  EXPECT_EQ(ovresult.plan_result, tresult.plan_result);

  auto qs4 = QR::create_query_state(
      session,
      R"(SELECT shape_table.rowid FROM shape_table, attribute_table WHERE shape_table.block_group_id = attribute_table.block_group_id)");
  TPlanResult tab_result =
      query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                         qs4->createQueryStateProxy(),
                                                         qs4->getQueryStr(),
                                                         calciteQueryParsingOption,
                                                         calciteOptimizationOption);

  auto qs5 = QR::create_query_state(
      session,
      R"(SELECT shape_view.rowid FROM shape_view, attribute_view WHERE shape_view.block_group_id = attribute_view.block_group_id)");
  TPlanResult view_result =
      query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                         qs5->createQueryStateProxy(),
                                                         qs5->getQueryStr(),
                                                         calciteQueryParsingOption,
                                                         calciteOptimizationOption);
  EXPECT_EQ(tab_result.plan_result, view_result.plan_result);
}

TEST_F(ViewObject, Joins) {
  auto session = QR::get()->getSession();
  CHECK(session);

  auto calciteQueryParsingOption =
      g_calcite->getCalciteQueryParsingOption(true, false, false);
  auto calciteOptimizationOption =
      g_calcite->getCalciteOptimizationOption(true, false, {});

  {
    auto qs1 = QR::create_query_state(
        session,
        R"(SELECT i1 FROM table1 LEFT JOIN attribute_shape_view ON table1.i1 = attribute_shape_view.block_group_id)");
    TPlanResult tresult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs1->createQueryStateProxy(),
                                                           qs1->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    auto qs2 = QR::create_query_state(session, "SELECT i1 FROM left_join_3tables");
    TPlanResult vresult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs2->createQueryStateProxy(),
                                                           qs2->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    EXPECT_EQ(vresult.plan_result, tresult.plan_result);
  }
}

TEST_F(ViewObject, RestrictLegacy) {
  auto session = QR::get()->getSession();
  CHECK(session);

  auto calciteQueryParsingOption =
      g_calcite->getCalciteQueryParsingOption(true, false, false);
  auto calciteOptimizationOption =
      g_calcite->getCalciteOptimizationOption(true, false, {});

  {
    Catalog_Namespace::Catalog const& cat = *session->get_catalog_ptr();

    auto qs1 = QR::create_query_state(
        session,
        R"(SELECT segment_name FROM attribute_table where segment_name = 'ab' or segment_name = 'ac')");
    TPlanResult tresult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs1->createQueryStateProxy(),
                                                           qs1->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    Catalog_Namespace::SysCatalog::instance().createLegacyPolicyInMemory(
        cat, "segment_name", TEST_USER, {"'ab'", "'ac'"});

    auto qs2 =
        QR::create_query_state(session, "SELECT segment_name FROM attribute_table");
    TPlanResult resResult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs2->createQueryStateProxy(),
                                                           qs2->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    Catalog_Namespace::SysCatalog::instance().dropLegacyPolicyInMemory(cat, TEST_USER);

    expect_equivalent_plan(tresult.plan_result, resResult.plan_result);

    auto qs3 = QR::create_query_state(session, R"(select i1 from table1 where i1 = 1)");
    TPlanResult riResult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs3->createQueryStateProxy(),
                                                           qs3->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    Catalog_Namespace::SysCatalog::instance().createLegacyPolicyInMemory(
        cat, "i1", TEST_USER, {"1"});

    auto qs4 = QR::create_query_state(session, "select i1 from table1");
    TPlanResult rrResult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs4->createQueryStateProxy(),
                                                           qs4->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    Catalog_Namespace::SysCatalog::instance().dropLegacyPolicyInMemory(cat, TEST_USER);

    expect_equivalent_plan(riResult.plan_result, rrResult.plan_result);
  }
}

TEST_F(ViewObject, Restrict) {
  auto session = QR::get()->getSession();
  CHECK(session);

  auto calciteQueryParsingOption =
      g_calcite->getCalciteQueryParsingOption(true, false, false);
  auto calciteOptimizationOption =
      g_calcite->getCalciteOptimizationOption(true, false, {});

  {
    Catalog_Namespace::Catalog const& cat = *session->get_catalog_ptr();

    auto qs1 = QR::create_query_state(
        session,
        R"(SELECT segment_name FROM attribute_table where segment_name = 'ab' or segment_name = 'ac')");
    TPlanResult tresult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs1->createQueryStateProxy(),
                                                           qs1->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    Catalog_Namespace::SysCatalog::instance().createPolicy(
        cat, {"attribute_table", "segment_name"}, TEST_USER, {"'ab'", "'ac'"});

    auto qs2 =
        QR::create_query_state(session, "SELECT segment_name FROM attribute_table");
    TPlanResult resResult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs2->createQueryStateProxy(),
                                                           qs2->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    Catalog_Namespace::SysCatalog::instance().dropPolicy(
        cat, {"attribute_table", "segment_name"}, TEST_USER);

    expect_equivalent_plan(tresult.plan_result, resResult.plan_result);

    auto qs3 = QR::create_query_state(session, R"(select i1 from table1 where i1 = 1)");
    TPlanResult riResult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs3->createQueryStateProxy(),
                                                           qs3->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    Catalog_Namespace::SysCatalog::instance().createPolicy(
        cat, {"table1", "i1"}, TEST_USER, {"1"});

    auto qs4 = QR::create_query_state(session, "select i1 from table1");
    TPlanResult rrResult =
        query_parsing::process_and_check_access_privileges(g_calcite.get(),
                                                           qs4->createQueryStateProxy(),
                                                           qs4->getQueryStr(),
                                                           calciteQueryParsingOption,
                                                           calciteOptimizationOption);

    Catalog_Namespace::SysCatalog::instance().dropPolicy(
        cat, {"table1", "i1"}, TEST_USER);

    expect_equivalent_plan(riResult.plan_result, rrResult.plan_result);
  }
}

struct PlannerRuleCoverage : testing::Test {
 protected:
  void SetUp() override {
    dropTables();
    run_ddl_statement(
        "CREATE TABLE planner_customer_pk("
        "c_custkey INTEGER NOT NULL, "
        "c_name TEXT ENCODING DICT(32), "
        "CONSTRAINT planner_customer_pk_key PRIMARY KEY (c_custkey));");
    run_ddl_statement(
        "CREATE TABLE planner_customer_no_key("
        "c_custkey INTEGER NOT NULL, "
        "c_name TEXT ENCODING DICT(32));");
    run_ddl_statement(
        "CREATE TABLE planner_customer_composite_key("
        "c_custkey INTEGER NOT NULL, "
        "c_name TEXT NOT NULL ENCODING DICT(32), "
        "CONSTRAINT planner_customer_composite_key_unique "
        "UNIQUE (c_custkey, c_name));");
    run_ddl_statement(
        "CREATE TABLE planner_orders("
        "o_orderkey INTEGER, "
        "o_custkey INTEGER, "
        "o_comment TEXT ENCODING DICT(32));");
    run_ddl_statement(
        "CREATE TABLE planner_orders_fk("
        "o_orderkey INTEGER, "
        "o_custkey INTEGER, "
        "o_comment TEXT ENCODING DICT(32), "
        "CONSTRAINT planner_orders_customer_fk FOREIGN KEY (o_custkey) "
        "REFERENCES planner_customer_pk(c_custkey));");
    run_ddl_statement(
        "CREATE TABLE planner_orders_not_null("
        "o_orderkey INTEGER NOT NULL, "
        "o_custkey INTEGER NOT NULL, "
        "o_comment TEXT ENCODING DICT(32));");
    run_ddl_statement(
        "CREATE TABLE planner_lineitem("
        "l_orderkey INTEGER, "
        "l_suppkey INTEGER, "
        "l_receiptdate INTEGER, "
        "l_commitdate INTEGER);");
    run_ddl_statement("CREATE TABLE planner_count_distinct(k INTEGER, v INTEGER);");
    run_ddl_statement(
        "CREATE TABLE planner_count_distinct_not_null(k INTEGER, v INTEGER NOT NULL);");
    run_ddl_statement(
        "CREATE TABLE planner_count_distinct_array(k INTEGER, arr INTEGER[]);");
  }

  void TearDown() override { dropTables(); }

  static void dropTables() {
    run_ddl_statement("DROP TABLE IF EXISTS planner_count_distinct_array;");
    run_ddl_statement("DROP TABLE IF EXISTS planner_count_distinct_not_null;");
    run_ddl_statement("DROP TABLE IF EXISTS planner_count_distinct;");
    run_ddl_statement("DROP TABLE IF EXISTS planner_lineitem;");
    run_ddl_statement("DROP TABLE IF EXISTS planner_orders_not_null;");
    run_ddl_statement("DROP TABLE IF EXISTS planner_orders_fk;");
    run_ddl_statement("DROP TABLE IF EXISTS planner_orders;");
    run_ddl_statement("DROP TABLE IF EXISTS planner_customer_composite_key;");
    run_ddl_statement("DROP TABLE IF EXISTS planner_customer_no_key;");
    run_ddl_statement("DROP TABLE IF EXISTS planner_customer_pk;");
  }
};

TEST_F(PlannerRuleCoverage, LeftJoinCountDistributionRequiresUniqueLeftKeyAndForeignKey) {
  const auto catalog = QR::get()->getCatalog();
  ASSERT_NE(nullptr, catalog);
  const auto customer_pk_td = catalog->getMetadataForTable("planner_customer_pk", false);
  ASSERT_NE(nullptr, customer_pk_td);
  const auto customer_pk_constraints = catalog->getTableConstraints(customer_pk_td);
  ASSERT_EQ(size_t{1}, customer_pk_constraints.size()) << customer_pk_td->keyMetainfo;
  EXPECT_EQ(Catalog_Namespace::TableConstraintType::PrimaryKey,
            customer_pk_constraints.front().type);
  EXPECT_EQ(std::vector<std::string>{"c_custkey"},
            customer_pk_constraints.front().column_names);

  const std::string constrained_plan = rel_alg_plan(R"sql(
      SELECT c_count, COUNT(*) AS custdist
      FROM (
        SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count
        FROM planner_customer_pk c
        LEFT JOIN planner_orders_fk o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) c_orders
      GROUP BY c_count
  )sql");
  EXPECT_NE(std::string::npos, constrained_plan.find(R"("relOp": "LogicalUnion")"))
      << constrained_plan;
  EXPECT_EQ(std::string::npos, constrained_plan.find(R"("joinType": "left")"))
      << constrained_plan;
  // The rewritten positive distribution counts rows from the one-column
  // counted-key producer; it must not add an extra marker column that would
  // force the temporary producer to publish duplicate payload.
  EXPECT_EQ(std::string::npos, constrained_plan.find("count_marker")) << constrained_plan;

  const std::string constrained_explain = calcite_explain_plan(R"sql(
      SELECT c_count, COUNT(*) AS custdist
      FROM (
        SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count
        FROM planner_customer_pk c
        LEFT JOIN planner_orders_fk o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) c_orders
      GROUP BY c_count
  )sql");
  // Scalar aggregates return one row over an empty input. Suppress the synthetic
  // zero bucket unless at least one left-domain key actually belongs in it.
  EXPECT_NE(std::string::npos, constrained_explain.find(">($1, 0)"))
      << constrained_explain;

  const std::string no_foreign_key_plan = rel_alg_plan(R"sql(
      SELECT c_count, COUNT(*) AS custdist
      FROM (
        SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) c_orders
      GROUP BY c_count
  )sql");
  EXPECT_EQ(std::string::npos, no_foreign_key_plan.find(R"("relOp": "LogicalUnion")"))
      << no_foreign_key_plan;
  EXPECT_NE(std::string::npos, no_foreign_key_plan.find(R"("joinType": "left")"))
      << no_foreign_key_plan;

  const std::string unconstrained_plan = rel_alg_plan(R"sql(
      SELECT c_count, COUNT(*) AS custdist
      FROM (
        SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count
        FROM planner_customer_no_key c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) c_orders
      GROUP BY c_count
  )sql");
  EXPECT_EQ(std::string::npos, unconstrained_plan.find(R"("relOp": "LogicalUnion")"))
      << unconstrained_plan;
  EXPECT_NE(std::string::npos, unconstrained_plan.find(R"("joinType": "left")"))
      << unconstrained_plan;
}

TEST_F(PlannerRuleCoverage, ExperimentalQueryRewritesAreOptIn) {
  // Use a shape that is valid for the rewrite so this test isolates the opt-in gate;
  // nullable grouped arguments are covered separately below.
  const std::string query =
      "SELECT k, COUNT(DISTINCT v) AS distinct_v "
      "FROM planner_count_distinct_not_null GROUP BY k;";

  const std::string default_plan =
      rel_alg_plan(query, /*enable_experimental_query_rewrites=*/false);
  EXPECT_EQ(size_t{1}, count_occurrences(default_plan, R"("relOp": "LogicalAggregate")"))
      << default_plan;
  EXPECT_NE(std::string::npos, default_plan.find(R"("distinct": true)")) << default_plan;

  const std::string opt_in_plan = rel_alg_plan(query);
  EXPECT_EQ(size_t{2}, count_occurrences(opt_in_plan, R"("relOp": "LogicalAggregate")"))
      << opt_in_plan;
  EXPECT_EQ(std::string::npos, opt_in_plan.find(R"("distinct": true)")) << opt_in_plan;

  const std::string key_query =
      "SELECT c.c_custkey, c.c_name, SUM(o.o_orderkey) AS total_orderkey "
      "FROM planner_customer_pk c "
      "JOIN planner_orders o ON c.c_custkey = o.o_custkey "
      "GROUP BY c.c_custkey, c.c_name;";
  const std::string default_key_plan =
      calcite_explain_plan(key_query, /*enable_experimental_query_rewrites=*/false);
  expect_plan_order(default_key_plan, "LogicalAggregate(", "LogicalJoin(");

  const std::string untrusted_key_plan =
      calcite_explain_plan(key_query,
                           /*enable_experimental_query_rewrites=*/true,
                           /*trust_unenforced_table_constraints=*/false);
  expect_plan_order(untrusted_key_plan, "LogicalAggregate(", "LogicalJoin(");

  const std::string opt_in_key_plan = calcite_explain_plan(key_query);
  expect_plan_order(opt_in_key_plan, "LogicalJoin(", "LogicalAggregate(");
}

TEST_F(PlannerRuleCoverage, AverageOutputRemainsNullableForDownstreamCount) {
  const auto plan = rel_alg_plan(R"sql(
      SELECT COUNT(avg_v)
      FROM (
        SELECT k, AVG(v) AS avg_v
        FROM planner_count_distinct
        GROUP BY k
      ) grouped_values
  )sql");

  rapidjson::Document document;
  document.Parse(plan.c_str());
  ASSERT_FALSE(document.HasParseError()) << plan;
  EXPECT_EQ(size_t{1}, count_aggregate_calls_with_operands(document, "COUNT")) << plan;
}

TEST_F(PlannerRuleCoverage, LeftJoinCountDistributionRequiresCompleteLeftDomain) {
  const std::string projected_plan = rel_alg_plan(R"sql(
      SELECT c_count, COUNT(*) AS custdist
      FROM (
        SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count
        FROM (
          SELECT c_custkey, c_name
          FROM planner_customer_pk
        ) c
        LEFT JOIN planner_orders_fk o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) c_orders
      GROUP BY c_count
  )sql");
  EXPECT_NE(std::string::npos, projected_plan.find(R"("relOp": "LogicalUnion")"))
      << projected_plan;
  EXPECT_EQ(std::string::npos, projected_plan.find(R"("joinType": "left")"))
      << projected_plan;

  // The FK only proves inclusion in the base customer table. Filtering the referenced
  // side can remove a customer that still has orders, so the right-only distribution
  // would count a key absent from the left join domain.
  const std::string filtered_plan = rel_alg_plan(R"sql(
      SELECT c_count, COUNT(*) AS custdist
      FROM (
        SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count
        FROM (
          SELECT c_custkey, c_name
          FROM planner_customer_pk
          WHERE c_custkey > 0
        ) c
        LEFT JOIN planner_orders_fk o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) c_orders
      GROUP BY c_count
  )sql");
  EXPECT_EQ(std::string::npos, filtered_plan.find(R"("relOp": "LogicalUnion")"))
      << filtered_plan;
  EXPECT_NE(std::string::npos, filtered_plan.find(R"("joinType": "left")"))
      << filtered_plan;

  const std::string expression_plan = rel_alg_plan(R"sql(
      SELECT c_count, COUNT(*) AS custdist
      FROM (
        SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count
        FROM (
          SELECT c_custkey + 1 AS c_custkey
          FROM planner_customer_pk
        ) c
        LEFT JOIN planner_orders_fk o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) c_orders
      GROUP BY c_count
  )sql");
  EXPECT_EQ(std::string::npos, expression_plan.find(R"("relOp": "LogicalUnion")"))
      << expression_plan;
  EXPECT_NE(std::string::npos, expression_plan.find(R"("joinType": "left")"))
      << expression_plan;
}

TEST_F(PlannerRuleCoverage, LeftJoinCountDistributionRejectsMixedResidualJoinFilter) {
  const std::string plan = rel_alg_plan(R"sql(
      SELECT c_count, COUNT(*) AS custdist
      FROM (
        SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o
          ON c.c_custkey = o.o_custkey AND c.c_name = o.o_comment
        GROUP BY c.c_custkey
      ) c_orders
      GROUP BY c_count
  )sql");
  EXPECT_EQ(std::string::npos, plan.find(R"("relOp": "LogicalUnion")")) << plan;
  EXPECT_NE(std::string::npos, plan.find(R"("joinType": "left")")) << plan;
}

TEST_F(PlannerRuleCoverage, SingleCountDistinctUsesTwoPhaseGroupBy) {
  const std::string plan = rel_alg_plan(
      "SELECT k, COUNT(DISTINCT v) AS distinct_values "
      "FROM planner_count_distinct_not_null GROUP BY k;");

  EXPECT_EQ(size_t{2}, count_occurrences(plan, R"("relOp": "LogicalAggregate")")) << plan;
  EXPECT_EQ(std::string::npos, plan.find(R"("relOp": "LogicalFilter")")) << plan;
  EXPECT_EQ(std::string::npos, plan.find(R"("distinct": true)")) << plan;
}

TEST_F(PlannerRuleCoverage, SingleCountDistinctPreservesAllNullGroups) {
  // Filtering nullable values before the lower aggregate would erase a group whose
  // COUNT(DISTINCT value) must be reported as zero.
  const std::string grouped_plan = rel_alg_plan(
      "SELECT k, COUNT(DISTINCT v) AS distinct_values "
      "FROM planner_count_distinct GROUP BY k;");
  EXPECT_EQ(size_t{1}, count_occurrences(grouped_plan, R"("relOp": "LogicalAggregate")"))
      << grouped_plan;
  EXPECT_NE(std::string::npos, grouped_plan.find(R"("distinct": true)")) << grouped_plan;

  const std::string scalar_plan =
      rel_alg_plan("SELECT COUNT(DISTINCT v) FROM planner_count_distinct;");
  EXPECT_EQ(size_t{2}, count_occurrences(scalar_plan, R"("relOp": "LogicalAggregate")"))
      << scalar_plan;
  EXPECT_NE(std::string::npos, scalar_plan.find("IS NOT NULL")) << scalar_plan;
  EXPECT_EQ(std::string::npos, scalar_plan.find(R"("distinct": true)")) << scalar_plan;
}

TEST_F(PlannerRuleCoverage, SingleCountDistinctRejectsUnsupportedShapes) {
  const std::string multi_agg_plan = rel_alg_plan(
      "SELECT k, COUNT(DISTINCT v) AS distinct_values, SUM(v) AS total_v "
      "FROM planner_count_distinct GROUP BY k;");
  EXPECT_EQ(size_t{1},
            count_occurrences(multi_agg_plan, R"("relOp": "LogicalAggregate")"))
      << multi_agg_plan;
  EXPECT_NE(std::string::npos, multi_agg_plan.find(R"("distinct": true)"))
      << multi_agg_plan;

  const std::string grouped_arg_plan = rel_alg_plan(
      "SELECT k, COUNT(DISTINCT k) AS distinct_keys "
      "FROM planner_count_distinct GROUP BY k;");
  EXPECT_EQ(size_t{1},
            count_occurrences(grouped_arg_plan, R"("relOp": "LogicalAggregate")"))
      << grouped_arg_plan;
  EXPECT_NE(std::string::npos, grouped_arg_plan.find(R"("distinct": true)"))
      << grouped_arg_plan;

  const std::string array_arg_plan = rel_alg_plan(
      "SELECT k, COUNT(DISTINCT arr) AS distinct_arrays "
      "FROM planner_count_distinct_array GROUP BY k;");
  EXPECT_EQ(size_t{1},
            count_occurrences(array_arg_plan, R"("relOp": "LogicalAggregate")"))
      << array_arg_plan;
  EXPECT_NE(std::string::npos, array_arg_plan.find(R"("distinct": true)"))
      << array_arg_plan;

  const std::string string_arg_plan = rel_alg_plan(
      "SELECT c_custkey, COUNT(DISTINCT c_name) AS distinct_names "
      "FROM planner_customer_pk GROUP BY c_custkey;");
  EXPECT_EQ(size_t{1},
            count_occurrences(string_arg_plan, R"("relOp": "LogicalAggregate")"))
      << string_arg_plan;
  EXPECT_NE(std::string::npos, string_arg_plan.find(R"("distinct": true)"))
      << string_arg_plan;
}

TEST_F(PlannerRuleCoverage, DistinctAggregateJoinPruneRequiresNonEmptyUnusedSide) {
  const std::string guaranteed_non_empty_plan = calcite_explain_plan(R"sql(
      SELECT DISTINCT c.c_custkey
      FROM planner_customer_pk c
      JOIN (
        SELECT COUNT(*) AS row_count
        FROM planner_orders
      ) guaranteed_one ON TRUE
  )sql");
  EXPECT_EQ(std::string::npos, guaranteed_non_empty_plan.find("LogicalJoin("))
      << guaranteed_non_empty_plan;

  const std::string possibly_empty_plan = calcite_explain_plan(R"sql(
      SELECT DISTINCT c.c_custkey
      FROM planner_customer_pk c
      JOIN planner_orders o ON TRUE
  )sql");
  // Calcite may represent this Cartesian input as either LogicalJoin or MultiJoin.
  // The semantic invariant is that the possibly-empty orders input is not pruned.
  EXPECT_NE(std::string::npos,
            possibly_empty_plan.find("LogicalTableScan(table=[[test_db, "
                                     "planner_orders]])"))
      << possibly_empty_plan;
}

TEST_F(PlannerRuleCoverage, JoinFilterSplitSeparatesInnerJoinFilters) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, o.o_orderkey
      FROM planner_customer_pk c
      JOIN planner_orders o
        ON c.c_custkey = o.o_custkey
       AND c.c_custkey > 10
       AND o.o_orderkey < 100
       AND c.c_custkey < o.o_orderkey
  )sql");

  EXPECT_NE(std::string::npos, plan.find("LogicalJoin(condition=[AND(=($0, $")) << plan;
  EXPECT_NE(std::string::npos, plan.find(", <($0, $")) << plan;
  EXPECT_NE(std::string::npos, plan.find("LogicalFilter(condition=[>($0, 10)]")) << plan;
  EXPECT_NE(std::string::npos, plan.find("LogicalFilter(condition=[<($0, 100)]")) << plan;
}

TEST_F(PlannerRuleCoverage, JoinFilterSplitPushesOnlyRightSideLeftJoinFilters) {
  const std::string right_side_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, o.o_orderkey
      FROM planner_customer_pk c
      LEFT JOIN planner_orders o
        ON c.c_custkey = o.o_custkey
       AND o.o_orderkey > 10
  )sql");
  EXPECT_NE(std::string::npos, right_side_plan.find("LogicalJoin(condition=[=($0, $"))
      << right_side_plan;
  expect_join_type(right_side_plan, "left");
  EXPECT_NE(std::string::npos,
            right_side_plan.find("LogicalFilter(condition=[>($0, 10)]"))
      << right_side_plan;

  const std::string left_side_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, o.o_orderkey
      FROM planner_customer_pk c
      LEFT JOIN planner_orders o
        ON c.c_custkey = o.o_custkey
       AND c.c_custkey > 10
  )sql");
  expect_join_type(left_side_plan, "left");
  EXPECT_NE(std::string::npos, left_side_plan.find(">($0, 10)")) << left_side_plan;
  EXPECT_EQ(std::string::npos, left_side_plan.find("LogicalFilter(condition=[>($0, 10)]"))
      << left_side_plan;
}

TEST_F(PlannerRuleCoverage, JoinFilterSplitDerivesDomainFiltersFromResiduals) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, o.o_orderkey
      FROM planner_customer_pk c
      JOIN planner_orders o
        ON c.c_custkey = o.o_custkey
       AND (
             (c.c_custkey = 1 AND o.o_orderkey = 10)
          OR (c.c_custkey = 2 AND o.o_orderkey = 20)
       )
  )sql");

  EXPECT_NE(std::string::npos, plan.find("LogicalJoin(condition=[AND(=($0, $")) << plan;
  // Calcite 1.41 normalizes equality OR domains into SEARCH/Sarg filters.
  EXPECT_LE(size_t{1},
            count_occurrences(plan, "LogicalFilter(condition=[SEARCH($0, Sarg[1, 2])]"))
      << plan;
  EXPECT_LE(size_t{1},
            count_occurrences(plan, "LogicalFilter(condition=[SEARCH($0, Sarg[10, 20])]"))
      << plan;
  EXPECT_NE(std::string::npos, plan.find("OR(AND(=($0, 1), =($")) << plan;
}

TEST_F(PlannerRuleCoverage, KeyPreservingAggregateRequiresUniqueDimensionKey) {
  const std::string constrained_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, c.c_name, SUM(o.o_orderkey) AS total_orderkey
      FROM planner_customer_pk c
      JOIN planner_orders o ON c.c_custkey = o.o_custkey
      GROUP BY c.c_custkey, c.c_name
  )sql");
  expect_plan_order(constrained_plan, "LogicalJoin(", "LogicalAggregate(");
  EXPECT_NE(std::string::npos,
            constrained_plan.find("LogicalTableScan(table=[[test_db, "
                                  "planner_customer_pk]])"))
      << constrained_plan;
  EXPECT_NE(std::string::npos,
            constrained_plan.find("LogicalTableScan(table=[[test_db, planner_orders]])"))
      << constrained_plan;

  const std::string unconstrained_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, c.c_name, SUM(o.o_orderkey) AS total_orderkey
      FROM planner_customer_no_key c
      JOIN planner_orders o ON c.c_custkey = o.o_custkey
      GROUP BY c.c_custkey, c.c_name
  )sql");
  expect_plan_order(unconstrained_plan, "LogicalAggregate(", "LogicalJoin(");
}

TEST_F(PlannerRuleCoverage, KeyPreservingAggregateTracksCompositeKeys) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, c.c_name, SUM(o.o_orderkey) AS total_orderkey
      FROM planner_customer_composite_key c
      JOIN planner_orders o
        ON c.c_custkey = o.o_custkey AND c.c_name = o.o_comment
      GROUP BY c.c_custkey, c.c_name
  )sql");
  expect_plan_order(plan, "LogicalJoin(", "LogicalAggregate(");
}

TEST_F(PlannerRuleCoverage, KeyPreservingAggregateRejectsUnsafeShapes) {
  const std::string expression_key_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey + 1 AS shifted_key,
             c.c_name,
             SUM(o.o_orderkey) AS total_orderkey
      FROM planner_customer_pk c
      JOIN planner_orders o ON c.c_custkey = o.o_custkey
      GROUP BY c.c_custkey + 1, c.c_name
  )sql");
  expect_plan_order(expression_key_plan, "LogicalAggregate(", "LogicalJoin(");

  const std::string mixed_residual_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, c.c_name, SUM(o.o_orderkey) AS total_orderkey
      FROM planner_customer_pk c
      JOIN planner_orders o
        ON c.c_custkey = o.o_custkey AND c.c_custkey < o.o_orderkey
      GROUP BY c.c_custkey, c.c_name
  )sql");
  expect_plan_order(mixed_residual_plan, "LogicalAggregate(", "LogicalJoin(");

  const std::string filtered_precomputed_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, c.c_name, SUM(o.o_orderkey) AS total_orderkey
      FROM planner_customer_pk c
      JOIN planner_orders o ON c.c_custkey = o.o_custkey
      JOIN (
        SELECT o_custkey, SUM(o_orderkey) AS filtered_total
        FROM planner_orders
        WHERE o_comment = 'selected'
        GROUP BY o_custkey
      ) p ON c.c_custkey = p.o_custkey
      GROUP BY c.c_custkey, c.c_name
  )sql");
  EXPECT_EQ(size_t{2},
            count_occurrences(filtered_precomputed_plan,
                              "LogicalTableScan(table=[[test_db, planner_orders]])"))
      << filtered_precomputed_plan;
  expect_plan_order(filtered_precomputed_plan, "LogicalAggregate(", "LogicalJoin(");
}

TEST_F(PlannerRuleCoverage,
       AggregateOuterJoinStrengthReductionRecognizesNullRejectedAggregate) {
  const std::string comparison_plan = calcite_explain_plan(R"sql(
      SELECT *
      FROM (
        SELECT c.c_custkey, SUM(o.o_orderkey) AS total_orderkey
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) orders_by_customer
      WHERE total_orderkey > 0
  )sql");
  expect_join_type(comparison_plan, "inner");

  const std::string is_not_null_plan = calcite_explain_plan(R"sql(
      SELECT *
      FROM (
        SELECT c.c_custkey, SUM(o.o_orderkey) AS total_orderkey
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) orders_by_customer
      WHERE total_orderkey IS NOT NULL
  )sql");
  expect_join_type(is_not_null_plan, "inner");

  const std::string not_is_null_plan = calcite_explain_plan(R"sql(
      SELECT *
      FROM (
        SELECT c.c_custkey, SUM(o.o_orderkey) AS total_orderkey
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) orders_by_customer
      WHERE NOT (total_orderkey IS NULL)
  )sql");
  expect_join_type(not_is_null_plan, "inner");
}

TEST_F(PlannerRuleCoverage, OuterJoinNullRejectionReducesFullJoin) {
  const std::string left_rejected_plan = calcite_explain_plan(R"sql(
      SELECT l.l_orderkey, l.l_suppkey, l.l_receiptdate,
             o.o_orderkey, o.o_custkey, o.o_comment
      FROM planner_lineitem l
      FULL OUTER JOIN planner_orders o ON l.l_orderkey = o.o_orderkey
      WHERE l.l_orderkey IS NOT NULL AND l.l_receiptdate < 2
      ORDER BY l.l_orderkey, l.l_suppkey, l.l_receiptdate,
               o.o_orderkey, o.o_custkey, o.o_comment
  )sql",
                                                              false);
  expect_join_type(left_rejected_plan, "left");
  EXPECT_EQ(std::string::npos, left_rejected_plan.find("joinType=[full]"))
      << left_rejected_plan;

  const std::string both_rejected_plan = calcite_explain_plan(R"sql(
      SELECT l.l_orderkey, l.l_suppkey, o.o_orderkey
      FROM planner_lineitem l
      FULL OUTER JOIN planner_orders o ON l.l_orderkey = o.o_orderkey
      WHERE l.l_orderkey IS NOT NULL AND o.o_orderkey IS NOT NULL
  )sql",
                                                              false);
  expect_join_type(both_rejected_plan, "inner");
  EXPECT_EQ(std::string::npos, both_rejected_plan.find("joinType=[full]"))
      << both_rejected_plan;
}

TEST_F(PlannerRuleCoverage, OuterJoinNullRejectionRejectsNullSafeComparisons) {
  const std::string distinct_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, o.o_orderkey
      FROM planner_customer_pk c
      LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
      WHERE o.o_orderkey IS DISTINCT FROM 42
  )sql",
                                                         false);
  expect_join_type(distinct_plan, "left");

  const std::string not_distinct_null_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, o.o_orderkey
      FROM planner_customer_pk c
      LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
      WHERE o.o_orderkey IS NOT DISTINCT FROM NULL
  )sql",
                                                                  false);
  expect_join_type(not_distinct_null_plan, "left");
}

TEST_F(PlannerRuleCoverage, OuterJoinNullRejectionDoesNotConflateEqualSubtrees) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, o.o_orderkey
      FROM planner_customer_pk c
      LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
      WHERE o.o_orderkey IS NOT NULL
      UNION ALL
      SELECT c.c_custkey, o.o_orderkey
      FROM planner_customer_pk c
      LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
  )sql",
                                                false);
  EXPECT_EQ(size_t{1}, count_occurrences(plan, "joinType=[inner]")) << plan;
  EXPECT_EQ(size_t{1}, count_occurrences(plan, "joinType=[left]")) << plan;
}

TEST_F(PlannerRuleCoverage,
       AggregateOuterJoinStrengthReductionRejectsNullPreservingFilters) {
  const std::string left_sum_plan = calcite_explain_plan(R"sql(
      SELECT *
      FROM (
        SELECT c.c_custkey, SUM(c.c_custkey) AS total_key
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) orders_by_customer
      WHERE total_key > 0
  )sql");
  expect_join_type(left_sum_plan, "left");

  const std::string count_plan = calcite_explain_plan(R"sql(
      SELECT *
      FROM (
        SELECT c.c_custkey, COUNT(o.o_orderkey) AS order_count
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) orders_by_customer
      WHERE order_count > 0
  )sql");
  expect_join_type(count_plan, "left");

  const std::string or_plan = calcite_explain_plan(R"sql(
      SELECT *
      FROM (
        SELECT c.c_custkey, SUM(o.o_orderkey) AS total_orderkey
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) orders_by_customer
      WHERE total_orderkey > 0 OR c_custkey > 0
  )sql");
  expect_join_type(or_plan, "left");

  const std::string not_is_not_null_plan = calcite_explain_plan(R"sql(
      SELECT *
      FROM (
        SELECT c.c_custkey, SUM(o.o_orderkey) AS total_orderkey
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) orders_by_customer
      WHERE NOT (total_orderkey IS NOT NULL)
  )sql");
  expect_join_type(not_is_not_null_plan, "left");

  const std::string not_and_plan = calcite_explain_plan(R"sql(
      SELECT *
      FROM (
        SELECT c.c_custkey, SUM(o.o_orderkey) AS total_orderkey
        FROM planner_customer_pk c
        LEFT JOIN planner_orders o ON c.c_custkey = o.o_custkey
        GROUP BY c.c_custkey
      ) orders_by_customer
      WHERE NOT (total_orderkey > 0 AND c_custkey > 0)
  )sql");
  expect_join_type(not_and_plan, "left");
}

TEST_F(PlannerRuleCoverage, AggregateJoinReductionUsesFilteredKeyset) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, agg.order_sum
      FROM (
        SELECT c_custkey
        FROM planner_customer_pk
        WHERE c_name = 'A'
      ) c
      JOIN (
        SELECT o_custkey, SUM(o_orderkey) AS order_sum
        FROM planner_orders
        GROUP BY o_custkey
      ) agg ON c.c_custkey = agg.o_custkey
  )sql");

  EXPECT_EQ(size_t{2}, count_occurrences(plan, "LogicalJoin(")) << plan;
  EXPECT_EQ(size_t{1}, count_occurrences(plan, "LogicalAggregate(")) << plan;
  EXPECT_NE(std::string::npos, plan.find("LogicalFilter(condition=[=($1, 'A')]")) << plan;
}

TEST_F(PlannerRuleCoverage, AggregateJoinReductionDistinctsNonUniqueKeyset) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, agg.order_sum
      FROM (
        SELECT c_custkey
        FROM planner_customer_no_key
        WHERE c_name = 'A'
      ) c
      JOIN (
        SELECT o_custkey, SUM(o_orderkey) AS order_sum
        FROM planner_orders
        GROUP BY o_custkey
      ) agg ON c.c_custkey = agg.o_custkey
  )sql");

  EXPECT_EQ(size_t{2}, count_occurrences(plan, "LogicalJoin(")) << plan;
  EXPECT_EQ(size_t{2}, count_occurrences(plan, "LogicalAggregate(")) << plan;
}

TEST_F(PlannerRuleCoverage, AggregateJoinReductionTracksCompositeKeys) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, c.c_name, agg.order_sum
      FROM (
        SELECT c_custkey, c_name
        FROM planner_customer_composite_key
        WHERE c_name = 'A'
      ) c
      JOIN (
        SELECT o_custkey, o_comment, SUM(o_orderkey) AS order_sum
        FROM planner_orders
        GROUP BY o_custkey, o_comment
      ) agg
        ON c.c_custkey = agg.o_custkey AND c.c_name = agg.o_comment
  )sql");

  EXPECT_EQ(size_t{2}, count_occurrences(plan, "LogicalJoin(")) << plan;
  EXPECT_EQ(size_t{1}, count_occurrences(plan, "LogicalAggregate(")) << plan;
}

TEST_F(PlannerRuleCoverage, AggregateJoinReductionAllowsSameTableKeyset) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, agg.order_sum
      FROM (
        SELECT c_custkey
        FROM planner_customer_no_key
        WHERE c_name = 'A'
      ) c
      JOIN (
        SELECT c_custkey, SUM(c_custkey) AS order_sum
        FROM planner_customer_no_key
        GROUP BY c_custkey
      ) agg ON c.c_custkey = agg.c_custkey
  )sql");

  EXPECT_EQ(size_t{2}, count_occurrences(plan, "LogicalJoin(")) << plan;
  EXPECT_EQ(size_t{2}, count_occurrences(plan, "LogicalAggregate(")) << plan;
}

TEST_F(PlannerRuleCoverage, AggregateJoinReductionRejectsUnsafeShapes) {
  const std::string unfiltered_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, agg.order_sum
      FROM planner_customer_pk c
      JOIN (
        SELECT o_custkey, SUM(o_orderkey) AS order_sum
        FROM planner_orders
        GROUP BY o_custkey
      ) agg ON c.c_custkey = agg.o_custkey
  )sql");
  EXPECT_EQ(size_t{1}, count_occurrences(unfiltered_plan, "LogicalJoin("))
      << unfiltered_plan;

  const std::string left_join_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, agg.order_sum
      FROM (
        SELECT c_custkey
        FROM planner_customer_pk
        WHERE c_name = 'A'
      ) c
      LEFT JOIN (
        SELECT o_custkey, SUM(o_orderkey) AS order_sum
        FROM planner_orders
        GROUP BY o_custkey
      ) agg ON c.c_custkey = agg.o_custkey
  )sql");
  EXPECT_EQ(size_t{1}, count_occurrences(left_join_plan, "LogicalJoin("))
      << left_join_plan;
  expect_join_type(left_join_plan, "left");
}

TEST_F(PlannerRuleCoverage, ExistenceCountToGroupByRewritesCountStarExistenceSet) {
  const std::string distinct_plan = calcite_explain_plan(R"sql(
      SELECT c.c_name, COUNT(*) AS order_count
      FROM planner_customer_pk c
      JOIN (
        SELECT o_custkey
        FROM planner_orders
        GROUP BY o_custkey
      ) e ON c.c_custkey = e.o_custkey
      GROUP BY c.c_name
  )sql");
  expect_join_type(distinct_plan, "semi");
  EXPECT_EQ(size_t{2}, count_occurrences(distinct_plan, "LogicalJoin(")) << distinct_plan;
  EXPECT_EQ(size_t{3}, count_occurrences(distinct_plan, "LogicalAggregate("))
      << distinct_plan;

  const std::string marker_plan = calcite_explain_plan(R"sql(
      SELECT c.c_name, COUNT(*) AS order_count
      FROM planner_customer_pk c
      JOIN (
        SELECT o_custkey, MIN(TRUE) AS marker
        FROM planner_orders
        GROUP BY o_custkey
      ) e ON c.c_custkey = e.o_custkey
      GROUP BY c.c_name
  )sql");
  expect_join_type(marker_plan, "semi");
  EXPECT_EQ(size_t{2}, count_occurrences(marker_plan, "LogicalJoin(")) << marker_plan;
  EXPECT_EQ(size_t{3}, count_occurrences(marker_plan, "LogicalAggregate("))
      << marker_plan;
}

TEST_F(PlannerRuleCoverage, ExistenceCountToGroupByProvesDecorrelatedOuterKeyset) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT o.o_comment, COUNT(*) AS line_order_count
      FROM planner_orders o
      WHERE o.o_orderkey > 0
        AND EXISTS (
          SELECT 1
          FROM planner_lineitem l
          WHERE l.l_orderkey = o.o_orderkey
            AND l.l_commitdate < l.l_receiptdate
        )
      GROUP BY o.o_comment
  )sql");

  // Calcite decorrelation may first reduce the existence input by a projected
  // or distinct copy of the filtered outer key domain. That exact copy is safe
  // to replace; an unrelated self-join with the same base table is not.
  expect_join_type(plan, "semi");
  EXPECT_EQ(size_t{2}, count_occurrences(plan, "LogicalJoin(")) << plan;
}

TEST_F(PlannerRuleCoverage, ExistenceCountToGroupByRejectsResidualJoinPredicates) {
  const std::string residual_plan = calcite_explain_plan(R"sql(
      SELECT c.c_name, COUNT(*) AS order_count
      FROM planner_customer_pk c
      JOIN (
        SELECT o_custkey
        FROM planner_orders
        GROUP BY o_custkey
      ) e
        ON c.c_custkey = e.o_custkey AND c.c_custkey > e.o_custkey
      GROUP BY c.c_name
  )sql");
  expect_join_type(residual_plan, "inner");
  EXPECT_EQ(size_t{1}, count_occurrences(residual_plan, "LogicalJoin(")) << residual_plan;
  EXPECT_NE(std::string::npos, residual_plan.find(">($0, $")) << residual_plan;
}

TEST_F(PlannerRuleCoverage, ExistenceCountToGroupByDoesNotDropUnprovenSelfJoin) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_name, COUNT(*) AS order_count
      FROM planner_customer_pk c
      JOIN (
        SELECT o.o_custkey
        FROM planner_orders o
        JOIN planner_customer_pk c2 ON o.o_orderkey = c2.c_custkey
        GROUP BY o.o_custkey
      ) e ON c.c_custkey = e.o_custkey
      GROUP BY c.c_name
  )sql");

  // Seeing the same base table is not proof that the nested join is the outer
  // relation's decorrelation copy; its independent predicate must be preserved.
  EXPECT_EQ(std::string::npos, plan.find("joinType=[semi]")) << plan;
  EXPECT_EQ(size_t{2}, count_occurrences(plan, "LogicalJoin(")) << plan;
}

TEST_F(PlannerRuleCoverage, ExistenceCountToGroupByRejectsNonCountStarAggregates) {
  const std::string count_arg_plan = calcite_explain_plan(R"sql(
      SELECT c.c_name, COUNT(e.o_custkey) AS order_count
      FROM planner_customer_pk c
      JOIN (
        SELECT o_custkey
        FROM planner_orders
        GROUP BY o_custkey
      ) e ON c.c_custkey = e.o_custkey
      GROUP BY c.c_name
  )sql");
  expect_join_type(count_arg_plan, "inner");
  EXPECT_EQ(size_t{1}, count_occurrences(count_arg_plan, "LogicalJoin("))
      << count_arg_plan;

  const std::string multi_key_plan = calcite_explain_plan(R"sql(
      SELECT c.c_name, COUNT(*) AS order_count
      FROM planner_customer_pk c
      JOIN (
        SELECT o_custkey, o_comment
        FROM planner_orders
        GROUP BY o_custkey, o_comment
      ) e ON c.c_custkey = e.o_custkey
      GROUP BY c.c_name
  )sql");
  expect_join_type(multi_key_plan, "inner");
  EXPECT_EQ(size_t{1}, count_occurrences(multi_key_plan, "LogicalJoin("))
      << multi_key_plan;

  const std::string non_boolean_marker_plan = calcite_explain_plan(R"sql(
      SELECT c.c_name, COUNT(*) AS order_count
      FROM planner_customer_pk c
      JOIN (
        SELECT o_custkey, MIN(1) AS marker
        FROM planner_orders
        GROUP BY o_custkey
      ) e ON c.c_custkey = e.o_custkey
      GROUP BY c.c_name
  )sql");
  expect_join_type(non_boolean_marker_plan, "inner");
  EXPECT_NE(std::string::npos, non_boolean_marker_plan.find("LogicalAggregate("))
      << non_boolean_marker_plan;
}

TEST_F(PlannerRuleCoverage, DifferentValueAggregateJoinRewritesExists) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT l1.l_orderkey, l1.l_suppkey
      FROM planner_lineitem l1
      WHERE EXISTS (
        SELECT 1
        FROM planner_lineitem l2
        WHERE l2.l_orderkey = l1.l_orderkey
          AND l2.l_suppkey <> l1.l_suppkey
      )
  )sql");

  EXPECT_NE(std::string::npos, plan.find("min_diff_value=[MIN(")) << plan;
  EXPECT_NE(std::string::npos, plan.find("max_diff_value=[MAX(")) << plan;
  EXPECT_NE(std::string::npos, plan.find("OR(<>")) << plan;
}

TEST_F(PlannerRuleCoverage, DifferentValueAggregateJoinRejectsCrossResiduals) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT l1.l_orderkey, l1.l_suppkey
      FROM planner_lineitem l1
      WHERE EXISTS (
        SELECT 1
        FROM planner_lineitem l2
        WHERE l2.l_orderkey = l1.l_orderkey
          AND l2.l_suppkey <> l1.l_suppkey
          AND l2.l_receiptdate > l1.l_commitdate
      )
  )sql");

  EXPECT_EQ(std::string::npos, plan.find("min_diff_value=[MIN(")) << plan;
  EXPECT_EQ(std::string::npos, plan.find("max_diff_value=[MAX(")) << plan;
  EXPECT_NE(std::string::npos, plan.find(">($")) << plan;
}

TEST_F(PlannerRuleCoverage, DifferentValueAggregateJoinPreservesFilteredPairRelation) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT l1.l_orderkey, l1.l_suppkey
      FROM planner_lineitem l1
      JOIN (
        SELECT p.l_orderkey, p.l_suppkey, MIN(TRUE) AS marker
        FROM planner_lineitem p
        JOIN planner_lineitem d
          ON p.l_orderkey = d.l_orderkey
         AND p.l_suppkey <> d.l_suppkey
        WHERE p.l_receiptdate > p.l_commitdate
        GROUP BY p.l_orderkey, p.l_suppkey
      ) e
        ON l1.l_orderkey = e.l_orderkey
       AND l1.l_suppkey = e.l_suppkey
  )sql");

  // Replacing the filtered pair side with every outer pair would admit rows that
  // were absent from the original aggregate input.
  EXPECT_EQ(std::string::npos, plan.find("min_diff_value=[MIN(")) << plan;
  EXPECT_EQ(std::string::npos, plan.find("max_diff_value=[MAX(")) << plan;
  EXPECT_NE(std::string::npos, plan.find(">($")) << plan;
}

TEST_F(PlannerRuleCoverage, PairedDifferentValueStatsCombinesExistsAntiExists) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT l1.l_orderkey, COUNT(*) AS line_count
      FROM planner_lineitem l1
      WHERE l1.l_receiptdate > l1.l_commitdate
        AND EXISTS (
          SELECT 1
          FROM planner_lineitem l2
          WHERE l2.l_orderkey = l1.l_orderkey
            AND l2.l_suppkey <> l1.l_suppkey
        )
        AND NOT EXISTS (
          SELECT 1
          FROM planner_lineitem l3
          WHERE l3.l_orderkey = l1.l_orderkey
            AND l3.l_suppkey <> l1.l_suppkey
            AND l3.l_receiptdate > l3.l_commitdate
        )
      GROUP BY l1.l_orderkey
  )sql");

  EXPECT_NE(std::string::npos, plan.find("min_filtered_value=[MIN(")) << plan;
  EXPECT_NE(std::string::npos, plan.find("max_filtered_value=[MAX(")) << plan;
  EXPECT_NE(std::string::npos, plan.find("filtered_stats_value=[CASE(")) << plan;
  EXPECT_NE(std::string::npos, plan.find("OR(<>($")) << plan;
  EXPECT_EQ(std::string::npos, plan.find("joinType=[left]")) << plan;
  EXPECT_EQ(std::string::npos, plan.find("IS NULL(")) << plan;
}

TEST_F(PlannerRuleCoverage, PairedDifferentValueStatsHandlesAbsentFilteredRows) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT l1.l_orderkey, COUNT(*) AS line_count
      FROM planner_lineitem l1
      WHERE EXISTS (
          SELECT 1
          FROM planner_lineitem l2
          WHERE l2.l_orderkey = l1.l_orderkey
            AND l2.l_suppkey <> l1.l_suppkey
        )
        AND NOT EXISTS (
          SELECT 1
          FROM planner_lineitem l3
          WHERE l3.l_orderkey = l1.l_orderkey
            AND l3.l_suppkey <> l1.l_suppkey
            AND l3.l_receiptdate > l3.l_commitdate
        )
      GROUP BY l1.l_orderkey
  )sql");

  EXPECT_NE(std::string::npos, plan.find("min_filtered_value=[MIN(")) << plan;
  EXPECT_NE(std::string::npos, plan.find("max_filtered_value=[MAX(")) << plan;
  EXPECT_NE(std::string::npos, plan.find("IS NULL(")) << plan;
}

TEST_F(PlannerRuleCoverage, PairedDifferentValueStatsDoesNotCountNullValues) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_name, COUNT(*) AS line_count
      FROM planner_lineitem l1
      JOIN planner_customer_pk c ON l1.l_suppkey = c.c_custkey
      WHERE l1.l_receiptdate > l1.l_commitdate
        AND EXISTS (
          SELECT 1
          FROM planner_lineitem l2
          WHERE l2.l_orderkey = l1.l_orderkey
            AND l2.l_suppkey <> l1.l_suppkey
        )
        AND NOT EXISTS (
          SELECT 1
          FROM planner_lineitem l3
          WHERE l3.l_orderkey = l1.l_orderkey
            AND l3.l_suppkey <> l1.l_suppkey
            AND l3.l_receiptdate > l3.l_commitdate
        )
      GROUP BY c.c_name
  )sql");

  EXPECT_NE(std::string::npos, plan.find("filtered_count=[SUM(")) << plan;
  // MIN/MAX ignore NULL values. The fused count must do the same because
  // `nullable_value <> candidate_value` cannot satisfy the original EXISTS.
  EXPECT_NE(std::string::npos, plan.find("IS NOT NULL(")) << plan;
}

TEST_F(PlannerRuleCoverage, PairedDifferentValueStatsKeepsCandidateCountForExtraFilter) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_name, COUNT(*) AS line_count
      FROM planner_lineitem l1
      JOIN planner_customer_pk c ON l1.l_suppkey = c.c_custkey
      WHERE l1.l_receiptdate > l1.l_commitdate
        AND l1.l_commitdate = 42
        AND EXISTS (
          SELECT 1
          FROM planner_lineitem l2
          WHERE l2.l_orderkey = l1.l_orderkey
            AND l2.l_suppkey <> l1.l_suppkey
        )
        AND NOT EXISTS (
          SELECT 1
          FROM planner_lineitem l3
          WHERE l3.l_orderkey = l1.l_orderkey
            AND l3.l_suppkey <> l1.l_suppkey
            AND l3.l_receiptdate > l3.l_commitdate
        )
      GROUP BY c.c_name
  )sql");

  EXPECT_NE(std::string::npos, plan.find("min_filtered_value=[MIN(")) << plan;
  EXPECT_EQ(std::string::npos, plan.find("filtered_count=[SUM(")) << plan;
}

TEST_F(PlannerRuleCoverage, PairedDifferentValueStatsRejectsCrossResiduals) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT l1.l_orderkey, COUNT(*) AS line_count
      FROM planner_lineitem l1
      WHERE EXISTS (
          SELECT 1
          FROM planner_lineitem l2
          WHERE l2.l_orderkey = l1.l_orderkey
            AND l2.l_suppkey <> l1.l_suppkey
        )
        AND NOT EXISTS (
          SELECT 1
          FROM planner_lineitem l3
          WHERE l3.l_orderkey = l1.l_orderkey
            AND l3.l_suppkey <> l1.l_suppkey
            AND l3.l_receiptdate > l1.l_commitdate
        )
      GROUP BY l1.l_orderkey
  )sql");

  EXPECT_EQ(std::string::npos, plan.find("min_filtered_value=[MIN(")) << plan;
  EXPECT_EQ(std::string::npos, plan.find("max_filtered_value=[MAX(")) << plan;
  EXPECT_NE(std::string::npos, plan.find(">($")) << plan;
}

TEST_F(PlannerRuleCoverage, LeftJoinAntiSemiJoinRewritesExistenceMarkers) {
  const std::string not_exists_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      WHERE NOT EXISTS (
        SELECT 1
        FROM planner_orders o
        WHERE o.o_custkey = c.c_custkey
      )
  )sql");
  expect_join_type(not_exists_plan, "anti");
  EXPECT_EQ(std::string::npos, not_exists_plan.find("LogicalAggregate("))
      << not_exists_plan;

  const std::string marker_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      LEFT JOIN (
        SELECT o_custkey, MIN(TRUE) AS marker
        FROM planner_orders
        GROUP BY o_custkey
      ) m ON c.c_custkey = m.o_custkey
      WHERE m.marker IS NULL
  )sql");
  expect_join_type(marker_plan, "anti");
  EXPECT_EQ(std::string::npos, marker_plan.find("LogicalAggregate(")) << marker_plan;
}

TEST_F(PlannerRuleCoverage, LeftJoinAntiSemiJoinTracksCompositeMarkerKeys) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_composite_key c
      LEFT JOIN (
        SELECT o_custkey, o_comment, MIN(TRUE) AS marker
        FROM planner_orders
        GROUP BY o_custkey, o_comment
      ) m
        ON c.c_custkey = m.o_custkey AND c.c_name = m.o_comment
      WHERE m.marker IS NULL
  )sql");
  expect_join_type(plan, "anti");
  EXPECT_EQ(std::string::npos, plan.find("LogicalAggregate(")) << plan;
}

TEST_F(PlannerRuleCoverage, LeftJoinAntiSemiJoinCollapsesNestedMarkerDomain) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      LEFT JOIN (
        SELECT c2.c_custkey, m.marker
        FROM (
          SELECT c_custkey
          FROM planner_customer_pk
        ) c2
        LEFT JOIN (
          SELECT o_custkey, MIN(TRUE) AS marker
          FROM planner_orders
          GROUP BY o_custkey
        ) m ON c2.c_custkey = m.o_custkey
      ) r ON c.c_custkey = r.c_custkey
      WHERE r.marker IS NULL
  )sql");
  expect_join_type(plan, "anti");
  EXPECT_EQ(std::string::npos, plan.find("joinType=[left]")) << plan;
  EXPECT_EQ(std::string::npos, plan.find("LogicalAggregate(")) << plan;

  const std::string projected_domain_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey, r.c_custkey
      FROM planner_customer_pk c
      LEFT JOIN (
        SELECT c2.c_custkey, m.marker
        FROM (
          SELECT c_custkey
          FROM planner_customer_pk
        ) c2
        LEFT JOIN (
          SELECT o_custkey, MIN(TRUE) AS marker
          FROM planner_orders
          GROUP BY o_custkey
        ) m ON c2.c_custkey = m.o_custkey
      ) r ON c.c_custkey = r.c_custkey
      WHERE r.marker IS NULL
  )sql");
  // A matched domain row without a marker has a non-NULL r.c_custkey. The
  // existence-only anti-join form is valid only when that nested payload is unused.
  expect_join_type(projected_domain_plan, "left");
  EXPECT_NE(std::string::npos, projected_domain_plan.find("LogicalAggregate("))
      << projected_domain_plan;

  const std::string mismatched_domain_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      LEFT JOIN (
        SELECT c2.c_custkey, m.marker
        FROM (
          SELECT c_custkey
          FROM planner_customer_pk
          WHERE c_custkey > 10
        ) c2
        LEFT JOIN (
          SELECT o_custkey, MIN(TRUE) AS marker
          FROM planner_orders
          GROUP BY o_custkey
        ) m ON c2.c_custkey = m.o_custkey
      ) r ON c.c_custkey = r.c_custkey
      WHERE r.marker IS NULL
  )sql");
  expect_join_type(mismatched_domain_plan, "left");
  EXPECT_NE(std::string::npos, mismatched_domain_plan.find("LogicalAggregate("))
      << mismatched_domain_plan;

  const std::string non_unique_domain_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_no_key c
      LEFT JOIN (
        SELECT c2.c_custkey, m.marker
        FROM (
          SELECT c_custkey
          FROM planner_customer_no_key
        ) c2
        LEFT JOIN (
          SELECT o_custkey, MIN(TRUE) AS marker
          FROM planner_orders
          GROUP BY o_custkey
        ) m ON c2.c_custkey = m.o_custkey
      ) r ON c.c_custkey = r.c_custkey
      WHERE r.marker IS NULL
  )sql");
  expect_join_type(non_unique_domain_plan, "left");
  EXPECT_NE(std::string::npos, non_unique_domain_plan.find("LogicalAggregate("))
      << non_unique_domain_plan;

  const std::string incomplete_nested_key_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      LEFT JOIN (
        SELECT c2.c_custkey, m.marker
        FROM planner_customer_pk c2
        LEFT JOIN (
          SELECT o_custkey, o_comment, MIN(TRUE) AS marker
          FROM planner_orders
          GROUP BY o_custkey, o_comment
        ) m
          ON c2.c_custkey = m.o_custkey
         AND c2.c_name = m.o_comment
      ) r ON c.c_custkey = r.c_custkey
      WHERE r.marker IS NULL
  )sql");
  // The outer domain exposes only c_custkey, so collapsing directly to the marker
  // source would drop the nested c_name = o_comment predicate.
  expect_join_type(incomplete_nested_key_plan, "left");
  EXPECT_NE(std::string::npos, incomplete_nested_key_plan.find("LogicalAggregate("))
      << incomplete_nested_key_plan;
}

TEST_F(PlannerRuleCoverage, NotInToAntiJoinRejectsNullableKeys) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      WHERE c.c_name NOT IN (
        SELECT o.o_comment
        FROM planner_orders o
        GROUP BY o.o_comment
      )
  )sql");
  EXPECT_EQ(std::string::npos, plan.find("joinType=[anti]")) << plan;
  EXPECT_NE(std::string::npos, plan.find("LogicalFilter(condition=[NOT(IN(")) << plan;
  EXPECT_NE(std::string::npos, plan.find("LogicalAggregate(")) << plan;
}

TEST_F(PlannerRuleCoverage, NotInToAntiJoinRewritesNonNullKeys) {
  const std::string plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      WHERE c.c_custkey NOT IN (
        SELECT o.o_custkey
        FROM planner_orders_not_null o
      )
  )sql");

  expect_join_type(plan, "anti");
  EXPECT_EQ(std::string::npos, plan.find("LogicalFilter(condition=[NOT(IN(")) << plan;
}

TEST_F(PlannerRuleCoverage, LeftJoinAntiSemiJoinRejectsNonMarkerShapes) {
  const std::string count_marker_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      LEFT JOIN (
        SELECT o_custkey, COUNT(*) AS marker
        FROM planner_orders
        GROUP BY o_custkey
      ) m ON c.c_custkey = m.o_custkey
      WHERE m.marker IS NULL
  )sql");
  expect_join_type(count_marker_plan, "left");
  EXPECT_NE(std::string::npos, count_marker_plan.find("LogicalAggregate("))
      << count_marker_plan;

  const std::string non_boolean_marker_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      LEFT JOIN (
        SELECT o_custkey, MIN(1) AS marker
        FROM planner_orders
        GROUP BY o_custkey
      ) m ON c.c_custkey = m.o_custkey
      WHERE m.marker IS NULL
  )sql");
  expect_join_type(non_boolean_marker_plan, "left");
  EXPECT_NE(std::string::npos, non_boolean_marker_plan.find("LogicalAggregate("))
      << non_boolean_marker_plan;

  const std::string or_filter_plan = calcite_explain_plan(R"sql(
      SELECT c.c_custkey
      FROM planner_customer_pk c
      LEFT JOIN (
        SELECT o_custkey, MIN(TRUE) AS marker
        FROM planner_orders
        GROUP BY o_custkey
      ) m ON c.c_custkey = m.o_custkey
      WHERE m.marker IS NULL OR c.c_name = 'A'
  )sql");
  expect_join_type(or_filter_plan, "left");
  EXPECT_NE(std::string::npos, or_filter_plan.find("LogicalAggregate("))
      << or_filter_plan;
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  QR::init(BASE_PATH, TEST_USER, TEST_PASS, TEST_DB, "", true, 0, 256 << 20, true, true);
  g_calcite = QR::get()->getCatalog()->getCalciteMgr();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  Catalog_Namespace::DBMetadata db_metadata;
  if (Catalog_Namespace::SysCatalog::instance().getMetadataForDB(TEST_DB, db_metadata)) {
    Catalog_Namespace::SysCatalog::instance().dropDatabase(db_metadata);
  }
  Catalog_Namespace::SysCatalog::instance().dropUser(TEST_USER);

  QR::reset();
  return err;
}
