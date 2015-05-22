#include "QueryRunner.h"

#include "../Parser/parser.h"

#include <boost/filesystem/operations.hpp>


Catalog_Namespace::SessionInfo* get_session(const char* db_path) {
  std::string db_name { MAPD_SYSTEM_DB };
  std::string user_name { "mapd" };
  std::string passwd { "HyperInteractive" };
  boost::filesystem::path base_path { db_path };
  CHECK(boost::filesystem::exists(base_path));
  auto system_db_file = base_path / "mapd_catalogs" / "mapd";
  CHECK(boost::filesystem::exists(system_db_file));
	auto data_dir = base_path / "mapd_data";
	Data_Namespace::DataMgr *dataMgr = new Data_Namespace::DataMgr(data_dir.string(), true); // true is for useGpus
  Catalog_Namespace::SysCatalog sys_cat(base_path.string(), *dataMgr);
  Catalog_Namespace::UserMetadata user;
  CHECK(sys_cat.getMetadataForUser(user_name, user));
  CHECK_EQ(user.passwd, passwd);
  Catalog_Namespace::DBMetadata db;
  CHECK(sys_cat.getMetadataForDB(db_name, db));
  CHECK(user.isSuper || (user.userId == db.dbOwner));
  Catalog_Namespace::Catalog *cat = new Catalog_Namespace::Catalog(base_path.string(), db, *dataMgr);
  return new Catalog_Namespace::SessionInfo(std::shared_ptr<Catalog_Namespace::Catalog>(cat), user, ExecutorDeviceType::GPU, 0);
}

ResultRows run_multiple_agg(
    const std::string& query_str,
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
    const ExecutorDeviceType device_type) {
  SQLParser parser;
  std::list<Parser::Stmt*> parse_trees;
  std::string last_parsed;
  CHECK_EQ(parser.parse(query_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), 1);
  auto stmt = parse_trees.front();
  std::unique_ptr<Stmt> stmt_ptr(stmt); // make sure it's deleted
  Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt *>(stmt);
  CHECK(!ddl);
  Parser::DMLStmt *dml = dynamic_cast<Parser::DMLStmt*>(stmt);
  Analyzer::Query query;
  auto &g_cat = session->get_catalog();
  dml->analyze(g_cat, query);
  Planner::Optimizer optimizer(query, g_cat);
  Planner::RootPlan *plan = optimizer.optimize();
  std::unique_ptr<Planner::RootPlan> plan_ptr(plan); // make sure it's deleted
  auto executor = Executor::getExecutor(g_cat.get_currentDB().dbId);
  return executor->execute(plan, true, device_type, ExecutorOptLevel::LoopStrengthReduction);
}

