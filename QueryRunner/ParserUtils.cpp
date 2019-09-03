/*
 * Copyright 2019 OmniSci, Inc.
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

#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"
#include "Parser/ParserWrapper.h"
#include "Parser/parser.h"
#include "Planner/Planner.h"
#include "QueryEngine/CalciteAdapter.h"
#include "gen-cpp/CalciteServer.h"

namespace QueryRunner {

Planner::RootPlan* QueryRunner::parsePlanLegacy(const std::string& query_str) {
  const auto& cat = session_info_->getCatalog();
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  if (parser.parse(query_str, parse_trees, last_parsed)) {
    throw std::runtime_error("Failed to parse query: " + query_str);
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

Planner::RootPlan* QueryRunner::parsePlanCalcite(QueryStateProxy query_state_proxy) {
  auto const& query_state = query_state_proxy.getQueryState();
  ParserWrapper pw{query_state.get_query_str()};
  if (pw.isOtherExplain() || pw.is_ddl || pw.is_update_dml) {
    return parsePlanLegacy(query_state.get_query_str());
  }

  const auto& cat = query_state.getConstSessionInfo()->getCatalog();
  auto calcite_mgr = cat.getCalciteMgr();
  const auto query_ra =
      calcite_mgr
          ->process(query_state_proxy,
                    pg_shim(query_state.get_query_str()),
                    {},
                    true,
                    false,
                    false)
          .plan_result;  //  if we want to be able to check plans we may want to calc this
  return translate_query(query_ra, cat);
}

Planner::RootPlan* QueryRunner::parsePlan(QueryStateProxy query_state_proxy) {
  return parsePlanCalcite(query_state_proxy);
}

}  // namespace QueryRunner
