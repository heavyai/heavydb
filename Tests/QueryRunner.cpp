#include "QueryRunner.h"

#include "../Parser/parser.h"
#ifdef HAVE_CALCITE
#include "../QueryEngine/CalciteAdapter.h"
#include "../Parser/ParserWrapper.h"
#include "../Calcite/Calcite.h"
#endif  // HAVE_CALCITE

#ifdef HAVE_RAVM
#include "../QueryEngine/ExtensionFunctionsWhitelist.h"
#include "../QueryEngine/RelAlgExecutor.h"
#endif  // HAVE_RAVM

#include <boost/filesystem/operations.hpp>

#ifdef STANDALONE_CALCITE
#define CALCITEPORT 9093
#else
#define CALCITEPORT -1
#endif

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
#ifdef HAVE_CALCITE
  auto calcite = std::make_shared<Calcite>(CALCITEPORT, db_path, 1024);
#ifdef HAVE_RAVM
  ExtensionFunctionsWhitelist::add(calcite->getExtensionFunctionWhitelist());
#endif  // HAVE_RAVM
#endif  // HAVE_CALCITE
#ifdef HAVE_CUDA
  bool useGpus = true;
#else
  bool useGpus = false;
#endif
  {
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, useGpus, -1);
    Catalog_Namespace::SysCatalog sys_cat(base_path.string(),
                                          dataMgr
#ifdef HAVE_CALCITE
                                          ,
                                          calcite
#endif  // HAVE_CALCITE
                                          );
    CHECK(sys_cat.getMetadataForUser(user_name, user));
    CHECK_EQ(user.passwd, passwd);
    CHECK(sys_cat.getMetadataForDB(db_name, db));
    CHECK(user.isSuper || (user.userId == db.dbOwner));
  }
  auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, useGpus, -1);
  return new Catalog_Namespace::SessionInfo(std::make_shared<Catalog_Namespace::Catalog>(base_path.string(),
                                                                                         db,
                                                                                         dataMgr
#ifdef HAVE_CALCITE
                                                                                         ,
                                                                                         std::vector<LeafHostInfo>{},
                                                                                         calcite
#endif  // HAVE_CALCITE
                                                                                         ),
                                            user,
                                            ExecutorDeviceType::GPU,
                                            0);
}

namespace {

Planner::RootPlan* parse_plan_legacy(const std::string& query_str,
                                     const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  const auto& cat = session->get_catalog();
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  if (parser.parse(query_str, parse_trees, last_parsed)) {
    throw std::runtime_error("Failed to parse query");
  }
  CHECK_EQ(parse_trees.size(), size_t(1));
  const auto& stmt = parse_trees.front();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
  CHECK(!ddl);
  Parser::DMLStmt* dml = dynamic_cast<Parser::DMLStmt*>(stmt.get());
  Analyzer::Query query;
  dml->analyze(cat, query);
  Planner::Optimizer optimizer(query, cat);
  return optimizer.optimize();
}

#ifdef HAVE_CALCITE

Planner::RootPlan* parse_plan_calcite(const std::string& query_str,
                                      const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  ParserWrapper pw{query_str};
  if (pw.is_other_explain || pw.is_ddl || pw.is_update_dml) {
    return parse_plan_legacy(query_str, session);
  }

  const auto& cat = session->get_catalog();
  auto& calcite_mgr = cat.get_calciteMgr();
  const auto query_ra = calcite_mgr.process(session->get_currentUser().userName,
                                            session->get_currentUser().passwd,
                                            cat.get_currentDB().dbName,
                                            pg_shim(query_str),
                                            true,
                                            false);  //  if we want to be able to check plans we may want to calc this
  return translate_query(query_ra, cat);
}

#endif  // HAVE_CALCITE

Planner::RootPlan* parse_plan(const std::string& query_str,
                              const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
#ifdef HAVE_CALCITE
  Planner::RootPlan* plan = parse_plan_calcite(query_str, session);
#else
  Planner::RootPlan* plan = parse_plan_legacy(query_str, session);
#endif  // HAVE_CALCITE
  return plan;
}

}  // namespace

ResultRows run_multiple_agg(const std::string& query_str,
                            const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
                            const ExecutorDeviceType device_type,
                            const bool hoist_literals) {
  const auto& cat = session->get_catalog();
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId);

#ifdef HAVE_RAVM
  ParserWrapper pw{query_str};
  if (!(pw.is_other_explain || pw.is_ddl || pw.is_update_dml)) {
    CompilationOptions co = {device_type, true, ExecutorOptLevel::LoopStrengthReduction, false};
    ExecutionOptions eo = {false, true, false, true, false, false, false, false, 10000};
    auto& calcite_mgr = cat.get_calciteMgr();
    const auto query_ra = calcite_mgr.process(session->get_currentUser().userName,
                                              session->get_currentUser().passwd,
                                              cat.get_currentDB().dbName,
                                              pg_shim(query_str),
                                              true,
                                              false);
    RelAlgExecutor ra_executor(executor.get(), cat);
    return ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr).getRows();
  }
#endif  // HAVE_RAVM

  Planner::RootPlan* plan = parse_plan(query_str, session);

  std::unique_ptr<Planner::RootPlan> plan_ptr(plan);  // make sure it's deleted
#ifdef HAVE_CUDA
  return executor->execute(
      plan, *session, hoist_literals, device_type, ExecutorOptLevel::LoopStrengthReduction, true, true);
#else
  return executor->execute(
      plan, *session, hoist_literals, device_type, ExecutorOptLevel::LoopStrengthReduction, false, true);
#endif
}
