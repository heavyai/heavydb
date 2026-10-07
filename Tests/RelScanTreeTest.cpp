/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TestHelpers.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "Catalog/Catalog.h"
#include "ExecuteRenderInterface/RenderQueryUtils/RelScanTree.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/RelAlgDag.h"
#include "QueryRunner/QueryRunner.h"
#include "ThriftHandler/QueryParsing.h"
#include "ThriftHandler/QueryState.h"
#include "gen-cpp/CalciteServer.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_calcite_view_optimize;
extern bool g_enable_watchdog;

using QR = QueryRunner::QueryRunner;
using QueryRenderer::RelScanTree;

namespace {

std::shared_ptr<Calcite> g_calcite;

void run_ddl_statement(const std::string& stmt) {
  QR::get()->runDDLStatement(stmt);
}

// Mirrors RenderQueryRunner::buildRelAlgDagAndClassifyRender, which builds the DAG
// unoptimized, injects rowid, and only then optimizes. QueryRunner::getRelAlgDag is
// unusable here: it builds through RelAlgExecutor, whose constructor hardcodes
// buildDag(query_ra, true), and optimization fuses the project and join whose
// relationship these tests are about.
std::unique_ptr<RelAlgDag> build_unoptimized_dag(const std::string& sql) {
  auto query_state = QR::create_query_state(QR::get()->getSession(), sql);
  auto const parsing_option = g_calcite->getCalciteQueryParsingOption(true, false, false);
  auto const optimization_option = g_calcite->getCalciteOptimizationOption(
      g_enable_calcite_view_optimize, g_enable_watchdog, {});
  auto const plan = query_parsing::process_and_check_access_privileges(
      g_calcite.get(),
      query_state->createQueryStateProxy(),
      pg_shim(sql),
      parsing_option,
      optimization_option);
  return RelAlgDagBuilder::buildDag(plan.plan_result, /*optimize_dag=*/false);
}

std::optional<uint32_t> find_output_index(const RelProject& project,
                                          const std::string& field_name) {
  for (size_t i = 0; i < project.size(); ++i) {
    if (project.getFieldName(i) == field_name) {
      return static_cast<uint32_t>(i);
    }
  }
  return std::nullopt;
}

// inject_rowid_into_RA searches from the end because rowid is virtual and all but
// always the last scan field.
std::optional<int> find_rowid_column(const RelScan& scan) {
  for (int i = static_cast<int>(scan.size()) - 1; i >= 0; --i) {
    if (scan.getFieldName(i) == "rowid") {
      return i;
    }
  }
  return std::nullopt;
}

std::string node_str(const RelAlgNode* node) {
  return node ? node->toString(RelRexToStringConfig::defaults()) : std::string("null");
}

// Walks the single-input chain below `node` looking for a RelJoin. Does not descend
// into the join's own branches.
const RelJoin* find_join_below(const RelAlgNode* node) {
  for (auto const* curr = node; curr && curr->inputCount() == size_t(1);
       curr = curr->getInput(0)) {
    if (auto const* join = dynamic_cast<const RelJoin*>(curr->getInput(0))) {
      return join;
    }
  }
  return nullptr;
}

void expect_resolves_to_rowid(const RelScanTree& rel_scan_tree,
                              const uint32_t output_index,
                              const std::string& table_name) {
  const RelScan* scan{nullptr};
  uint32_t col_idx{0};
  ASSERT_NO_THROW(std::tie(scan, col_idx) =
                      rel_scan_tree.getScanNodeForOutputIndex(output_index));
  ASSERT_TRUE(scan);
  EXPECT_EQ(scan->getTableDescriptor()->tableName, table_name);
  EXPECT_EQ(scan->getFieldName(col_idx), "rowid");
}

// Scaled-down equivalent of the render query that exposed the bug: a correlated EXISTS
// over a GROUP BY MAX subquery, with rowid projected in both the inner and outer
// selects.
const std::string kCorrelatedExistsWithRowId{
    R"(SELECT lon AS x, lat AS y, grp AS color, rowid
       FROM (SELECT grp, ts, lon, lat, rowid
             FROM rst_events
             WHERE EXISTS (SELECT 1
                           FROM (SELECT grp, MAX(ts) AS max_ts
                                 FROM rst_events
                                 WHERE true
                                 GROUP BY grp) rhs
                           WHERE rst_events.grp = rhs.grp
                             AND rst_events.ts = rhs.max_ts))
       WHERE lon IS NOT NULL AND lat IS NOT NULL
         AND lon >= -71.2 AND lon <= -70.7
         AND lat >= 42.1 AND lat <= 42.5
       LIMIT 10000)"};

}  // namespace

class RelScanTreeTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    dropTables();
    run_ddl_statement(
        "CREATE TABLE rst_events (grp TEXT ENCODING DICT(8), ts TIMESTAMP(0), lon "
        "DOUBLE, lat DOUBLE);");
    run_ddl_statement("CREATE TABLE rst_lookup (grp TEXT ENCODING DICT(8), weight INT);");
  }

  static void TearDownTestSuite() { dropTables(); }

 private:
  static void dropTables() {
    run_ddl_statement("DROP TABLE IF EXISTS rst_events;");
    run_ddl_statement("DROP TABLE IF EXISTS rst_lookup;");
  }
};

// Regression tests for the RelScanTree rowid walk over a project-on-join.
//
// After bind_inputs(), a RelProject over a RelJoin sources its RexInputs at the join's
// children rather than at the join itself (see get_node_output(RelJoin)). The
// RelScanTree walk assumed the project's immediate input was the RexInput source, so
// resolving an explicit rowid threw
// "input_node != current_tree_node->lhs->rel_alg_node".
//
// Both sides of the join are covered: the left branch takes the index unchanged, the
// right branch offsets it by the left branch's width. Each of these fails against the
// pre-fix RelScanTree.cpp.
TEST_F(RelScanTreeTest, RowIdFromLeftSideOfJoin) {
  auto rel_alg_dag = build_unoptimized_dag(
      "SELECT a.lon, a.rowid, b.weight FROM rst_events a INNER JOIN rst_lookup b ON "
      "a.grp = b.grp;");
  auto rel_scan_tree = RelScanTree::create(*rel_alg_dag);
  ASSERT_TRUE(rel_scan_tree) << node_str(&rel_alg_dag->getRootNode());

  // Guard the plan shape before asserting on behaviour, so that a Calcite change which
  // stops producing project-over-join fails here rather than passing vacuously.
  auto const& root_project = rel_scan_tree->getRootProjectNode();
  ASSERT_EQ(root_project.inputCount(), size_t(1));
  auto const* join = dynamic_cast<const RelJoin*>(root_project.getInput(0));
  ASSERT_TRUE(join) << "expected RelProject directly over RelJoin, but the project's "
                       "input was: "
                    << node_str(root_project.getInput(0));

  auto const rowid_idx = find_output_index(root_project, "rowid");
  ASSERT_TRUE(rowid_idx.has_value()) << node_str(&root_project);

  auto const* rowid_input =
      dynamic_cast<const RexInput*>(root_project.getProjectAt(*rowid_idx));
  ASSERT_TRUE(rowid_input) << "rowid output is not a direct column reference";
  ASSERT_EQ(rowid_input->getSourceNode(), join->getInput(0))
      << "precondition for the fault: the rowid RexInput should source the join's left "
         "child, not the join itself. Source was: "
      << node_str(rowid_input->getSourceNode());

  expect_resolves_to_rowid(*rel_scan_tree, *rowid_idx, "rst_events");
}

TEST_F(RelScanTreeTest, RowIdFromRightSideOfJoin) {
  auto rel_alg_dag = build_unoptimized_dag(
      "SELECT a.lon, b.weight, b.rowid FROM rst_events a INNER JOIN rst_lookup b ON "
      "a.grp = b.grp;");
  auto rel_scan_tree = RelScanTree::create(*rel_alg_dag);
  ASSERT_TRUE(rel_scan_tree) << node_str(&rel_alg_dag->getRootNode());

  auto const& root_project = rel_scan_tree->getRootProjectNode();
  auto const* join = dynamic_cast<const RelJoin*>(root_project.getInput(0));
  ASSERT_TRUE(join) << "expected RelProject directly over RelJoin, but the project's "
                       "input was: "
                    << node_str(root_project.getInput(0));

  auto const rowid_idx = find_output_index(root_project, "rowid");
  ASSERT_TRUE(rowid_idx.has_value()) << node_str(&root_project);

  auto const* rowid_input =
      dynamic_cast<const RexInput*>(root_project.getProjectAt(*rowid_idx));
  ASSERT_TRUE(rowid_input);
  ASSERT_EQ(rowid_input->getSourceNode(), join->getInput(1))
      << "precondition for the fault: the rowid RexInput should source the join's right "
         "child, not the join itself. Source was: "
      << node_str(rowid_input->getSourceNode());

  expect_resolves_to_rowid(*rel_scan_tree, *rowid_idx, "rst_lookup");
}

// The query from the original bug report. Calcite 1.41 now leaves an extra RelProject
// between the root project and the join, carrying the decorrelation's grp0/max_ts/$f2
// columns, so the root project's RexInputs source that intermediate project rather than
// a join child. This query therefore does NOT reproduce the fault any more -- it passes
// against the pre-fix RelScanTree.cpp -- and the two tests above are what guard it.
//
// It is kept because rowid resolution through a decorrelated EXISTS is worth covering
// on its own, and because it will guard the fault again if Calcite returns to putting
// the root project directly on the join.
TEST_F(RelScanTreeTest, RowIdThroughDecorrelatedExists) {
  auto rel_alg_dag = build_unoptimized_dag(kCorrelatedExistsWithRowId);
  auto rel_scan_tree = RelScanTree::create(*rel_alg_dag);
  ASSERT_TRUE(rel_scan_tree) << node_str(&rel_alg_dag->getRootNode());

  auto const& root_project = rel_scan_tree->getRootProjectNode();
  ASSERT_TRUE(find_join_below(&root_project))
      << "expected the correlated EXISTS to decorrelate into a join below the top-level "
         "project, but found none under: "
      << node_str(&root_project);

  auto const rowid_idx = find_output_index(root_project, "rowid");
  ASSERT_TRUE(rowid_idx.has_value())
      << "no rowid output in the top-level project: " << node_str(&root_project);

  expect_resolves_to_rowid(*rel_scan_tree, *rowid_idx, "rst_events");
}

// Auto-injection appends a RexInput sourced at the join itself rather than at one of
// its children, so it kept working while the explicit-rowid walk was broken. Pin that
// down, since the fix added a branch for it.
TEST_F(RelScanTreeTest, InjectedRowIdResolvesBackToScan) {
  auto rel_alg_dag = build_unoptimized_dag(kCorrelatedExistsWithRowId);
  auto rel_scan_tree = RelScanTree::create(*rel_alg_dag);
  ASSERT_TRUE(rel_scan_tree) << node_str(&rel_alg_dag->getRootNode());

  const RelScan* events_scan_ptr{nullptr};
  for (size_t i = 0; i < rel_scan_tree->size(); ++i) {
    auto const& scan = (*rel_scan_tree)[i];
    if (scan.getTableDescriptor()->tableName == "rst_events") {
      events_scan_ptr = &scan;
      break;
    }
  }
  ASSERT_TRUE(events_scan_ptr) << "no rst_events scan leaf among "
                               << rel_scan_tree->size() << " leaves";

  auto const& events_scan = *events_scan_ptr;
  auto const rowid_column = find_rowid_column(events_scan);
  ASSERT_TRUE(rowid_column.has_value());

  auto const& root_project = rel_scan_tree->getRootProjectNode();
  auto const injected_idx = static_cast<uint32_t>(root_project.size());
  rel_scan_tree->injectInputColumn(events_scan, *rowid_column, "rowid0");
  ASSERT_EQ(root_project.size(), size_t(injected_idx) + 1);

  expect_resolves_to_rowid(*rel_scan_tree, injected_idx, "rst_events");
}

TEST_F(RelScanTreeTest, RowIdOverSimpleProject) {
  auto rel_alg_dag = build_unoptimized_dag("SELECT lon, rowid FROM rst_events;");
  auto rel_scan_tree = RelScanTree::create(*rel_alg_dag);
  ASSERT_TRUE(rel_scan_tree) << node_str(&rel_alg_dag->getRootNode());

  auto const& root_project = rel_scan_tree->getRootProjectNode();
  auto const rowid_idx = find_output_index(root_project, "rowid");
  ASSERT_TRUE(rowid_idx.has_value()) << node_str(&root_project);

  expect_resolves_to_rowid(*rel_scan_tree, *rowid_idx, "rst_events");
}

TEST_F(RelScanTreeTest, ExpressionOutputStillThrows) {
  auto rel_alg_dag =
      build_unoptimized_dag("SELECT lon * 2 AS scaled, rowid FROM rst_events;");
  auto rel_scan_tree = RelScanTree::create(*rel_alg_dag);
  ASSERT_TRUE(rel_scan_tree) << node_str(&rel_alg_dag->getRootNode());

  auto const& root_project = rel_scan_tree->getRootProjectNode();
  auto const scaled_idx = find_output_index(root_project, "scaled");
  ASSERT_TRUE(scaled_idx.has_value()) << node_str(&root_project);

  EXPECT_THROW(rel_scan_tree->getScanNodeForOutputIndex(*scaled_idx), std::runtime_error);
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
    err = -1;
  }

  g_calcite.reset();
  QR::reset();
  return err;
}
