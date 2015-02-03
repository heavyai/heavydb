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


using namespace std;

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

vector<ResultRow> run_multiple_agg(
    const string& query_str,
    const ExecutorDeviceType device_type) {
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
  return executor.execute(device_type, ExecutorOptLevel::LoopStrengthReduction);
}

AggResult run_simple_agg(
    const string& query_str,
    const ExecutorDeviceType device_type = ExecutorDeviceType::CPU) {
  return run_multiple_agg(query_str, device_type).front().agg_results.front();
}

template<class T>
T v(const AggResult& r) {
  auto p = boost::get<T>(&r);
  CHECK(p);
  return *p;
}

void run_ddl_statement(const string& create_table_stmt) {
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

bool skip_tests(const ExecutorDeviceType device_type) {
  return device_type == ExecutorDeviceType::GPU && !g_cat.get_dataMgr().gpusPresent();
}

const ssize_t g_num_rows { 10 };

}

TEST(Select, FilterAndSimpleAggregation) {
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    run_multiple_agg("INSERT INTO test VALUES(7, 42, 101, 1001);", ExecutorDeviceType::CPU);
  }
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    run_multiple_agg("INSERT INTO test VALUES(8, 43, 102, 1002);", ExecutorDeviceType::CPU);
  }
  for (auto device_type : { ExecutorDeviceType::CPU, ExecutorDeviceType::GPU }) {
    if (skip_tests(device_type)) {
      CHECK(device_type == ExecutorDeviceType::GPU);
      LOG(WARNING) << "GPU not available, skipping GPU tests";
      continue;
    }
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test;", device_type)), 2 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test;", device_type)), 7);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MAX(x) FROM test;", device_type)), 8);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(z) FROM test;", device_type)), 101);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MAX(z) FROM test;", device_type)), 102);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(t) FROM test;", device_type)), 1001);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MAX(t) FROM test;", device_type)), 1002);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y) FROM test;", device_type)), 49 * g_num_rows + 51 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y + z) FROM test;", device_type)), 150 * g_num_rows + 153 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y + z + t) FROM test;", device_type)), 1151 * g_num_rows + 1155 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", device_type)), 2 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > 1000 AND t < 1002;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", device_type)), 2 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 102) OR (t > 1000 AND t < 1003);", device_type)), 2 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x <> 7;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE z <> 102;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE t <> 1002;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x + y = 49;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x + y + z = 150;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x + y + z + t = 1151;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y) FROM test WHERE x + y = 49;", device_type)), 49 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y + z) FROM test WHERE x + y = 49;", device_type)), 150 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y + z + t) FROM test WHERE x + y = 49;", device_type)), 1151 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x - y = -35;", device_type)), 2 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x - y + z = 66;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x - y + z + t = 1067;", device_type)), g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE y - x = 35;", device_type)), 2 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(2 * x) FROM test WHERE x = 7;", device_type)), 14 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(2 * x + z) FROM test WHERE x = 7;", device_type)), 115 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y) FROM test WHERE x - y = -35;", device_type)), 49 * g_num_rows + 51 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y) FROM test WHERE y - x = 35;", device_type)), 49 * g_num_rows + 51 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x + y - z) FROM test WHERE y - x = 35;", device_type)), -52 * g_num_rows + -51 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x * y + 15) FROM test WHERE x + y + 1 = 50;", device_type)), 309 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", device_type)), 309 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", device_type)), 309 * g_num_rows);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x * y + 15) FROM test WHERE x + y + 1 = 50;", device_type)), 309);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", device_type)), 309);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", device_type)), 309);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MAX(x * y + 15) FROM test WHERE x + y + 1 = 50;", device_type)), 309);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", device_type)), 309);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", device_type)), 309);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE x <> 7 AND x <> 8;", device_type)), numeric_limits<int64_t>::max());
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE z <> 101 AND z <> 102;", device_type)), numeric_limits<int64_t>::max());
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE t <> 1001 AND t <> 1002;", device_type)), numeric_limits<int64_t>::max());
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE x = 7;", device_type)), 7);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(z) FROM test WHERE z = 101;", device_type)), 101);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(t) FROM test WHERE t = 1001;", device_type)), 1001);
    ASSERT_EQ(v<double>(run_simple_agg("SELECT AVG(x + y) FROM test;", device_type)), 50.);
    ASSERT_EQ(v<double>(run_simple_agg("SELECT AVG(x + y + z) FROM test;", device_type)), 151.5);
    ASSERT_EQ(v<double>(run_simple_agg("SELECT AVG(x + y + z + t) FROM test;", device_type)), 1153.);
    ASSERT_EQ(v<double>(run_simple_agg("SELECT AVG(y) FROM test WHERE x > 6 AND x < 8;", device_type)), 42.);
    ASSERT_EQ(v<double>(run_simple_agg("SELECT AVG(y) FROM test WHERE z > 100 AND z < 102;", device_type)), 42.);
    ASSERT_EQ(v<double>(run_simple_agg("SELECT AVG(y) FROM test WHERE t > 1000 AND t < 1002;", device_type)), 42.);
  }
}

TEST(Select, FilterAndMultipleAggregation) {
  for (auto device_type : { ExecutorDeviceType::CPU, ExecutorDeviceType::GPU }) {
    if (skip_tests(device_type)) {
      CHECK(device_type == ExecutorDeviceType::GPU);
      LOG(WARNING) << "GPU not available, skipping GPU tests";
      continue;
    }
    auto agg_results = run_multiple_agg(
      "SELECT MIN(x), AVG(x * y), MAX(y + 7), COUNT(*) FROM test WHERE x + y > 47 AND x + y < 51;", device_type)
      .front()
      .agg_results;
    CHECK_EQ(agg_results.size(), 4);
    ASSERT_EQ(v<int64_t>(agg_results[0]), 7);
    ASSERT_EQ(v<double>(agg_results[1]), 294.);
    ASSERT_EQ(v<int64_t>(agg_results[2]), 49);
    ASSERT_EQ(v<int64_t>(agg_results[3]), g_num_rows);
  }
}

TEST(Select, FilterAndGroupBy) {
  const auto rows = run_multiple_agg(
    "SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", ExecutorDeviceType::CPU);
  CHECK_EQ(rows.size(), 2);
  {
  const auto row = rows[0];
  std::vector<int64_t> val_tuple { 7, 42 };
  ASSERT_EQ(row.value_tuple, val_tuple);
  ASSERT_EQ(v<int64_t>(row.agg_results[0]), 49);
  }
  {
  const auto row = rows[1];
  std::vector<int64_t> val_tuple { 8, 43 };
  ASSERT_EQ(row.value_tuple, val_tuple);
  ASSERT_EQ(v<int64_t>(row.agg_results[0]), 51);
  }
}

TEST(Select, FilterAndGroupByMultipleAgg) {
  const auto rows = run_multiple_agg(
    "SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", ExecutorDeviceType::CPU);
  CHECK_EQ(rows.size(), 2);
  {
  const auto row = rows[0];
  std::vector<int64_t> val_tuple { 7, 42 };
  ASSERT_EQ(row.value_tuple, val_tuple);
  ASSERT_EQ(v<int64_t>(row.agg_results[0]), 49);
  ASSERT_EQ(v<int64_t>(row.agg_results[1]), g_num_rows);
  ASSERT_EQ(v<double>(row.agg_results[2]), 8);
  }
  {
  const auto row = rows[1];
  std::vector<int64_t> val_tuple { 8, 43 };
  ASSERT_EQ(row.value_tuple, val_tuple);
  ASSERT_EQ(v<int64_t>(row.agg_results[0]), 51);
  ASSERT_EQ(v<int64_t>(row.agg_results[1]), g_num_rows);
  ASSERT_EQ(v<double>(row.agg_results[2]), 9);
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  try {
    run_ddl_statement("CREATE TABLE test(x int, y int, z smallint, t bigint);");
  } catch (...) {
    LOG(ERROR) << "Failed to create table 'test'";
    return -EEXIST;
  }
  int err { 0 };
  try {
    err = RUN_ALL_TESTS();
  } catch (const exception& e) {
    LOG(ERROR) << e.what();
  }
  run_ddl_statement("DROP TABLE test;");
  return err;
}
