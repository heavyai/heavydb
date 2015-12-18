#include "QueryRunner.h"

#include "../Parser/parser.h"
#ifdef HAVE_CALCITE
#include "../Calcite/Calcite.h"
#include "../QueryEngine/CalciteAdapter.h"
#endif  // HAVE_CALCITE

#include <boost/filesystem/operations.hpp>

Catalog_Namespace::SessionInfo* get_session(const char* db_path) {
  std::string db_name{MAPD_SYSTEM_DB};
  std::string user_name{"mapd"};
  std::string passwd{"HyperInteractive"};
  boost::filesystem::path base_path{db_path};
  CHECK(boost::filesystem::exists(base_path));
  auto system_db_file = base_path / "mapd_catalogs" / "mapd";
  CHECK(boost::filesystem::exists(system_db_file));
  auto data_dir = base_path / "mapd_data";
  Catalog_Namespace::UserMetadata user;
  Catalog_Namespace::DBMetadata db;
#ifdef HAVE_CUDA
  bool useGpus = true;
#else
  bool useGpus = false;
#endif
  {
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, useGpus, -1);
    Catalog_Namespace::SysCatalog sys_cat(base_path.string(), dataMgr);
    CHECK(sys_cat.getMetadataForUser(user_name, user));
    CHECK_EQ(user.passwd, passwd);
    CHECK(sys_cat.getMetadataForDB(db_name, db));
    CHECK(user.isSuper || (user.userId == db.dbOwner));
  }
  auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, useGpus, -1);
  return new Catalog_Namespace::SessionInfo(
      std::make_shared<Catalog_Namespace::Catalog>(base_path.string(), db, dataMgr), user, ExecutorDeviceType::GPU, 0);
}

namespace {

Planner::RootPlan* parse_plan_legacy(const std::string& query_str, const Catalog_Namespace::Catalog& cat) {
  SQLParser parser;
  std::list<Parser::Stmt*> parse_trees;
  std::string last_parsed;
  CHECK_EQ(parser.parse(query_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front();
  std::unique_ptr<Stmt> stmt_ptr(stmt);  // make sure it's deleted
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(!ddl);
  Parser::DMLStmt* dml = dynamic_cast<Parser::DMLStmt*>(stmt);
  Analyzer::Query query;
  dml->analyze(cat, query);
  Planner::Optimizer optimizer(query, cat);
  return optimizer.optimize();
}

#ifdef HAVE_CALCITE
Planner::RootPlan* parse_plan_calcite(const std::string& query_str,
                                      const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) noexcept {
  static Calcite calcite_cli(9092);
  const auto query_ra = calcite_cli.process(session->get_currentUser().userName,
                                            session->get_currentUser().passwd,
                                            session->get_catalog().get_currentDB().dbName,
                                            query_str);
  return translate_query(query_ra, session->get_catalog());
}
#endif  // HAVE_CALCITE

}  // namespace

ResultRows run_multiple_agg(const std::string& query_str,
                            const bool use_calcite,
                            const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
                            const ExecutorDeviceType device_type,
                            const NVVMBackend nvvm_backend) {
  const auto& cat = session->get_catalog();
#ifdef HAVE_CALCITE
  Planner::RootPlan* plan = use_calcite ? parse_plan_calcite(query_str, session) : parse_plan_legacy(query_str, cat);
#else
  Planner::RootPlan* plan = parse_plan_legacy(query_str, cat);
#endif                                                // HAVE_CALCITE
  std::unique_ptr<Planner::RootPlan> plan_ptr(plan);  // make sure it's deleted
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId);
#ifdef HAVE_CUDA
  return executor->execute(
      plan, *session, -1, true, device_type, nvvm_backend, ExecutorOptLevel::LoopStrengthReduction, true, true);
#else
  return executor->execute(
      plan, *session, -1, true, device_type, nvvm_backend, ExecutorOptLevel::LoopStrengthReduction, false, true);
#endif
}
