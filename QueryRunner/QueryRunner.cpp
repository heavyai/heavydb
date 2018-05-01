/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "QueryRunner.h"

#include "Parser/parser.h"
#include "QueryEngine/CalciteAdapter.h"
#include "Parser/ParserWrapper.h"
#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"

#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/RelAlgExecutor.h"

#include <boost/filesystem/operations.hpp>

#define CALCITEPORT 39093

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

Planner::RootPlan* parse_plan_calcite(const std::string& query_str,
                                      const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  ParserWrapper pw{query_str};
  if (pw.is_other_explain || pw.is_ddl || pw.is_update_dml) {
    return parse_plan_legacy(query_str, session);
  }

  const auto& cat = session->get_catalog();
  auto& calcite_mgr = cat.get_calciteMgr();
  const Catalog_Namespace::SessionInfo* sess = session.get();
  const auto query_ra = calcite_mgr.process(*sess,
                                            pg_shim(query_str),
                                            true,
                                            false).plan_result;  //  if we want to be able to check plans we may want to calc this
  return translate_query(query_ra, cat);
}

Planner::RootPlan* parse_plan(const std::string& query_str,
                              const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  Planner::RootPlan* plan = parse_plan_calcite(query_str, session);
  return plan;
}

}  // namespace

namespace QueryRunner {

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
  auto calcite = std::make_shared<Calcite>(-1, CALCITEPORT, db_path, 1024);
  ExtensionFunctionsWhitelist::add(calcite->getExtensionFunctionWhitelist());
#ifdef HAVE_CUDA
  bool useGpus = true;
#else
  bool useGpus = false;
#endif
  {
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, useGpus, -1);
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    sys_cat.init(base_path.string(), dataMgr, {}, calcite, false, false);
    CHECK(sys_cat.getMetadataForUser(user_name, user));
    CHECK_EQ(user.passwd, passwd);
    CHECK(sys_cat.getMetadataForDB(db_name, db));
    CHECK(user.isSuper || (user.userId == db.dbOwner));
  }
  auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, useGpus, -1);
  return new Catalog_Namespace::SessionInfo(std::make_shared<Catalog_Namespace::Catalog>(
                                                base_path.string(), db, dataMgr, std::vector<LeafHostInfo>{}, calcite),
                                            user,
                                            ExecutorDeviceType::GPU,
                                            "");
}

ExecutionResult run_select_query(const std::string& query_str,
                                 const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
                                 const ExecutorDeviceType device_type,
                                 const bool hoist_literals,
                                 const bool allow_loop_joins) {
  const auto& cat = session->get_catalog();
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId);
  CompilationOptions co = {device_type, true, ExecutorOptLevel::LoopStrengthReduction, false};
  ExecutionOptions eo = {false, true, false, allow_loop_joins, false, false, false, false, 10000};
  auto& calcite_mgr = cat.get_calciteMgr();
  const auto query_ra = calcite_mgr.process(*session, pg_shim(query_str), true, false).plan_result;
  RelAlgExecutor ra_executor(executor.get(), cat);
  return ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
                                            const ExecutorDeviceType device_type,
                                            const bool hoist_literals,
                                            const bool allow_loop_joins) {
  ParserWrapper pw{query_str};
  if (is_calcite_path_permissable(pw)) {
    const auto execution_result = run_select_query(query_str, session, device_type, hoist_literals, allow_loop_joins);
    return execution_result.getRows();
  }

  Planner::RootPlan* plan = parse_plan(query_str, session);
  std::unique_ptr<Planner::RootPlan> plan_ptr(plan);  // make sure it's deleted

  const auto& cat = session->get_catalog();
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId);

#ifdef HAVE_CUDA
  return executor->execute(
      plan, *session, hoist_literals, device_type, ExecutorOptLevel::LoopStrengthReduction, true, allow_loop_joins);
#else
  return executor->execute(
      plan, *session, hoist_literals, device_type, ExecutorOptLevel::LoopStrengthReduction, false, allow_loop_joins);
#endif
}

void run_ddl_statement(const std::string& create_table_stmt,
                       const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  CHECK_EQ(parser.parse(create_table_stmt, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front().get();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(ddl);
  if (ddl != nullptr)
    ddl->execute(*session);
}

}  // namespace QueryRUnner