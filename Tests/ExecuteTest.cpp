#include "../Analyzer/Analyzer.h"
#include "../Catalog/Catalog.h"
#include "../Parser/parser.h"
#include "../Parser/ParserNode.h"
#include "../Planner/Planner.h"
#include "../QueryEngine/Execute.h"
#include "../DataMgr/DataMgr.h"

#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

#include <string>
#include <cstdlib>
#include <memory>


namespace {

Catalog_Namespace::Catalog get_catalog() {
  string db_name { MAPD_SYSTEM_DB };
  string user_name { "mapd" };
  string passwd { "HyperInteractive" };
  boost::filesystem::path base_path { "/tmp" };
  CHECK(boost::filesystem::exists(base_path));
  auto system_db_file = base_path / "mapd_catalogs" / "mapd";
  CHECK(boost::filesystem::exists(system_db_file));
	auto data_dir = base_path / "mapd_data";
	Data_Namespace::DataMgr *dataMgr = new Data_Namespace::DataMgr(2, data_dir.string());
  Catalog_Namespace::SysCatalog sys_cat(base_path.string(), *dataMgr);
  Catalog_Namespace::UserMetadata user;
  CHECK(sys_cat.getMetadataForUser(user_name, user));
  CHECK_EQ(user.passwd, passwd);
  Catalog_Namespace::DBMetadata db;
  CHECK(sys_cat.getMetadataForDB(db_name, db));
  CHECK(user.isSuper || (user.userId == db.dbOwner));
  Catalog_Namespace::Catalog cat(base_path.string(), user, db, *dataMgr);
  return cat;
}

Catalog_Namespace::Catalog g_cat(get_catalog());

std::vector<Executor::AggResult> run_multiple_agg(const std::string& query_str) {
  SQLParser parser;
  list<Parser::Stmt*> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(query_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), 1);
  auto stmt = parse_trees.front();
  unique_ptr<Stmt> stmt_ptr(stmt); // make sure it's deleted
  Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt *>(stmt);
  CHECK(!ddl);
  Parser::DMLStmt *dml = dynamic_cast<Parser::DMLStmt*>(stmt);
  Analyzer::Query query;
  dml->analyze(g_cat, query);
  Planner::Optimizer optimizer(query, g_cat);
  Planner::RootPlan *plan = optimizer.optimize();
  unique_ptr<Planner::RootPlan> plan_ptr(plan); // make sure it's deleted
  Executor executor(plan);
  return executor.execute(ExecutorDeviceType::CPU, ExecutorOptLevel::LoopStrengthReduction);
}

Executor::AggResult run_simple_agg(const std::string& query_str) {
  return run_multiple_agg(query_str).front();
}

void run_ddl_statement(const std::string& create_table_stmt) {
  SQLParser parser;
  list<Parser::Stmt*> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(create_table_stmt, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), 1);
  auto stmt = parse_trees.front();
  unique_ptr<Stmt> stmt_ptr(stmt); // make sure it's deleted
  Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt *>(stmt);
  CHECK(ddl);
  if ( ddl != nullptr)
    ddl->execute(g_cat);
}

template<class T>
T v(const Executor::AggResult& r) {
  auto p = boost::get<T>(&r);
  CHECK(p);
  return *p;
}

const size_t g_num_rows { 10 };

}

TEST(Select, FilterAndSimpleAggregation) {
  for (size_t i = 0; i < g_num_rows; ++i) {
    run_multiple_agg("INSERT INTO test VALUES(7, 42);");
  }
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test;")), g_num_rows);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test;")), 7);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MAX(x) FROM test;")), 7);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y) FROM test;")), 49 * g_num_rows);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8;")), g_num_rows);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x <> 7;")), 0);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x + y = 49;")), g_num_rows);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y) FROM test WHERE x + y = 49;")), 49 * g_num_rows);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE y - x = 35;")), g_num_rows);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(2 * x) FROM test WHERE x = 7;")), 14 * g_num_rows);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y) FROM test WHERE y - x = 35;")), 49 * g_num_rows);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x * y + 15) FROM test WHERE x + y + 1 = 50;")), 309 * g_num_rows);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x * y + 15) FROM test WHERE x + y + 1 = 50;")), 309);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MAX(x * y + 15) FROM test WHERE x + y + 1 = 50;")), 309);
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE x <> 7;")), std::numeric_limits<int64_t>::max());
  ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE x = 7;")), 7);
  ASSERT_EQ(v<double>(run_simple_agg("SELECT AVG(x + y) FROM test;")), 49.);
  ASSERT_EQ(v<double>(run_simple_agg("SELECT AVG(y) FROM test WHERE x > 6 AND x < 8;")), 42.);
}

TEST(Select, FilterAndMultipleAggregation) {
  auto agg_results = run_multiple_agg("SELECT MIN(x), AVG(x * y), MAX(y + 7), COUNT(*) FROM test WHERE x + y > 47 AND x + y < 51;");
  CHECK_EQ(agg_results.size(), 4);
  ASSERT_EQ(v<int64_t>(agg_results[0]), 7);
  ASSERT_EQ(v<double>(agg_results[1]), 294.);
  ASSERT_EQ(v<int64_t>(agg_results[2]), 49);
  ASSERT_EQ(v<int64_t>(agg_results[3]), g_num_rows);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  try {
    run_ddl_statement("CREATE TABLE test(x int, y int);");
  } catch (...) {
    LOG(ERROR) << "Failed to create table 'test'";
    return -EEXIST;
  }
  int err { 0 };
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  run_ddl_statement("DROP TABLE test;");
  return err;
}
