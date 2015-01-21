#include "Analyzer/Analyzer.h"
#include "Catalog/Catalog.h"
#include "Parser/parser.h"
#include "Parser/ParserNode.h"
#include "Planner/Planner.h"
#include "QueryEngine/Execute.h"

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
  Catalog_Namespace::SysCatalog sys_cat(base_path.string());
  Catalog_Namespace::UserMetadata user;
	CHECK(sys_cat.getMetadataForUser(user_name, user));
	CHECK_EQ(user.passwd, passwd);
	Catalog_Namespace::DBMetadata db;
	CHECK(sys_cat.getMetadataForDB(db_name, db));
	CHECK(user.isSuper || (user.userId == db.dbOwner));
	Catalog_Namespace::Catalog cat(base_path.string(), user, db);
  return cat;
}

Catalog_Namespace::Catalog g_cat(get_catalog());

int64_t run_simple_agg(const std::string& query_str) {
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
  return executor.execute(ExecutorOptLevel::LoopStrengthReduction);
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

}

TEST(Select, FilterAndAggregation) {
  ASSERT_EQ(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 41 AND x < 43;"), 1000000);
  ASSERT_EQ(run_simple_agg("SELECT COUNT(*) FROM test WHERE x <> 42;"), 0);
  ASSERT_EQ(run_simple_agg("SELECT COUNT(*) FROM test WHERE x + y = 84;"), 1000000);
  ASSERT_EQ(run_simple_agg("SELECT SUM(x + y) FROM test WHERE x + y = 84;"), 84000000);
  ASSERT_EQ(run_simple_agg("SELECT COUNT(*) FROM test WHERE x - y = 0;"), 1000000);
  ASSERT_EQ(run_simple_agg("SELECT SUM(2 * x) FROM test WHERE x = 42;"), 84000000);
  ASSERT_EQ(run_simple_agg("SELECT SUM(x + y) FROM test WHERE x - y = 0;"), 84000000);
  ASSERT_EQ(run_simple_agg("SELECT SUM(x * y + 15) FROM test WHERE x + y + 6 = 90;"), 1779000000);
  ASSERT_EQ(run_simple_agg("SELECT MIN(x * y + 15) FROM test WHERE x + y + 6 = 90;"), 1779);
  ASSERT_EQ(run_simple_agg("SELECT MAX(x * y + 15) FROM test WHERE x + y + 6 = 90;"), 1779);
  ASSERT_EQ(run_simple_agg("SELECT MIN(x) FROM test WHERE x <> 42;"), std::numeric_limits<int64_t>::max());
  ASSERT_EQ(run_simple_agg("SELECT MIN(x) FROM test WHERE x = 42;"), 42);
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
