#include <glog/logging.h>
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
#include "Shared/MapDParameters.h"
#include "gen-cpp/CalciteServer.h"

namespace {
std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;
std::shared_ptr<Calcite> g_calcite;

inline void run_ddl_statement(const std::string& query) {
  QueryRunner::run_ddl_statement(query, g_session);
}
}  // namespace

struct ViewObject : testing::Test {
  void setup_objects() {
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
  }

  void remove_objects() {
    run_ddl_statement("DROP VIEW view_view_table1;");
    run_ddl_statement("DROP VIEW view_table1;");
    run_ddl_statement("DROP TABLE table1");
    run_ddl_statement("DROP VIEW attribute_view");
    run_ddl_statement("DROP VIEW shape_view");
    run_ddl_statement("DROP TABLE shape_table");
    run_ddl_statement("DROP TABLE attribute_table");
  }

  explicit ViewObject() { setup_objects(); }
  ~ViewObject() override { remove_objects(); }
};

TEST_F(ViewObject, BasicTest) {
  TPlanResult tresult =
      g_calcite->process(*g_session, "select i1 from table1", {}, true, false, false);

  TPlanResult vresult = g_calcite->process(
      *g_session, "select i1 from view_view_table1", {}, true, false, false);

  EXPECT_NE(vresult.plan_result, tresult.plan_result);

  TPlanResult ovresult = g_calcite->process(
      *g_session, "select i1 from view_view_table1", {}, true, false, true);

  EXPECT_EQ(ovresult.plan_result, tresult.plan_result);

  TPlanResult tab_result = g_calcite->process(
      *g_session,
      "SELECT shape_table.rowid FROM shape_table, attribute_table WHERE "
      "shape_table.block_group_id = attribute_table.block_group_id",
      {},
      true,
      false,
      true);
  TPlanResult view_result =
      g_calcite->process(*g_session,
                         "SELECT shape_view.rowid FROM shape_view, attribute_view WHERE "
                         "shape_view.block_group_id = attribute_view.block_group_id",
                         {},
                         true,
                         false,
                         true);
  EXPECT_EQ(tab_result.plan_result, view_result.plan_result);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  g_session.reset(QueryRunner::get_session(BASE_PATH));
  g_calcite = g_session->getCatalog().getCalciteMgr();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  g_session.reset(nullptr);
  return err;
}
