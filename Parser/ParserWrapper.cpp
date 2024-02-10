/*
 * Copyright 2022 HEAVY.AI, Inc.
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

/**
 * @file:   ParserWrapper.cpp
 * @brief
 *
 */

#include "ParserWrapper.h"
#include "Shared/measure.h"

#include <boost/algorithm/string.hpp>

using namespace std;

namespace {  // anonymous namespace

const std::string explain_str = {"explain"};
const std::string calcite_explain_str = {"explain calcite"};
const std::string calcite_explain_detailed_str = {"explain calcite detailed"};
const std::string optimized_explain_str = {"explain optimized"};
const std::string plan_explain_str = {"explain plan"};
const std::string plan_explain_detailed_str = {"explain plan detailed"};

}  // namespace

ExplainInfo::ExplainInfo(std::string query_string) {
  // sets explain_type_ and caches a trimmed actual_query_
  explain_type_ = ExplainType::None;
  actual_query_ = "";

  if (boost::istarts_with(query_string, explain_str)) {
    if (boost::istarts_with(query_string, calcite_explain_str)) {
      if (boost::istarts_with(query_string, calcite_explain_detailed_str)) {
        explain_type_ = ExplainType::CalciteDetail;
        actual_query_ =
            boost::trim_copy(query_string.substr(calcite_explain_detailed_str.size()));
      } else {
        actual_query_ = boost::trim_copy(query_string.substr(calcite_explain_str.size()));
        explain_type_ = ExplainType::Calcite;
      }
    } else if (boost::istarts_with(query_string, optimized_explain_str)) {
      actual_query_ = boost::trim_copy(query_string.substr(optimized_explain_str.size()));
      explain_type_ = ExplainType::OptimizedIR;
    } else if (boost::istarts_with(query_string, plan_explain_str)) {
      if (boost::istarts_with(query_string, plan_explain_detailed_str)) {
        actual_query_ =
            boost::trim_copy(query_string.substr(plan_explain_detailed_str.size()));
        verbose_ = true;
      } else {
        actual_query_ = boost::trim_copy(query_string.substr(plan_explain_str.size()));
      }
      explain_type_ = ExplainType::ExecutionPlan;
    } else {
      actual_query_ = boost::trim_copy(query_string.substr(explain_str.size()));
      explain_type_ = ExplainType::IR;
    }
    // override condition: ddl, update_dml
    ParserWrapper inner{actual_query_};
    if (inner.is_ddl || inner.is_update_dml) {
      explain_type_ = ExplainType::Other;
    }
  }
}

const std::vector<std::string> ParserWrapper::ddl_cmd = {
    "ARCHIVE",  "ALTER",  "COPY",   "CREATE",   "DROP",     "DUMP",
    "EVALUATE", "GRANT",  "KILL",   "OPTIMIZE", "REFRESH",  "RENAME",
    "RESTORE",  "REVOKE", "SHOW",   "TRUNCATE", "REASSIGN", "VALIDATE",
    "CLEAR",    "PAUSE",  "RESUME", "COMMENT"};

const std::vector<std::string> ParserWrapper::update_dml_cmd = {"INSERT",
                                                                "DELETE",
                                                                "UPDATE"};

namespace {

void validate_no_leading_comments(const std::string& query_str) {
  if (boost::starts_with(query_str, "--") || boost::starts_with(query_str, "//") ||
      boost::starts_with(query_str, "/*")) {
    throw std::runtime_error(
        "SQL statements starting with comments are currently not allowed.");
  }
}

const std::string optimize_str = {"optimize"};
const std::string validate_str = {"validate"};

}  // namespace

ParserWrapper::ParserWrapper(std::string query_string) {
  validate_no_leading_comments(query_string);

  is_other_explain = ExplainInfo(query_string).isOtherExplain();

  query_type_ = QueryType::Read;
  for (std::string ddl : ddl_cmd) {
    if (boost::istarts_with(query_string, ddl)) {
      query_type_ = QueryType::SchemaWrite;
      is_ddl = true;

      if (ddl == "CREATE") {
        boost::regex ctas_regex{
            R"(CREATE\s+(TEMPORARY\s+|\s*)+TABLE.*(\"|\s)AS(\(|\s)+(SELECT|WITH).*)",
            boost::regex::extended | boost::regex::icase};
        if (boost::regex_match(query_string, ctas_regex)) {
          is_ctas = true;
        }
      } else if (ddl == "COPY") {
        is_copy = true;
        // now check if it is COPY TO
        boost::regex copy_to{R"(COPY\s*\(([^#])(.+)\)\s+TO\s+.*)",
                             boost::regex::extended | boost::regex::icase};
        if (boost::regex_match(query_string, copy_to)) {
          query_type_ = QueryType::Read;
          is_copy_to = true;
        } else {
          query_type_ = QueryType::Write;
        }
      } else if (ddl == "SHOW") {
        query_type_ = QueryType::SchemaRead;
      } else if (ddl == "KILL") {
        query_type_ = QueryType::Unknown;
      } else if (ddl == "VALIDATE") {
        query_type_ = QueryType::Unknown;
        // needs to execute in a different context from other DDL
        is_validate = true;
      } else if (ddl == "ALTER") {
        boost::regex alter_system_regex{R"(ALTER\s+(SYSTEM|SESSION).*)",
                                        boost::regex::extended | boost::regex::icase};
        if (boost::regex_match(query_string, alter_system_regex)) {
          query_type_ = QueryType::Unknown;
        }
      } else if (ddl == "ARCHIVE" || ddl == "DUMP") {
        query_type_ = QueryType::SchemaRead;
      } else if (ddl == "REFRESH") {
        is_refresh = true;
      }
      return;
    }
  }

  for (int i = 0; i < update_dml_cmd.size(); i++) {
    is_update_dml = boost::istarts_with(query_string, ParserWrapper::update_dml_cmd[i]);
    if (is_update_dml) {
      query_type_ = QueryType::Write;
      dml_type_ = (DMLType)(i);
      break;
    }
  }

  if (dml_type_ == DMLType::Insert) {
    boost::regex itas_regex{R"(INSERT\s+INTO\s+.*(\s+|\(|\")SELECT(\s|\(|\").*)",
                            boost::regex::extended | boost::regex::icase};
    if (boost::regex_match(query_string, itas_regex)) {
      is_itas = true;
      return;
    }
  }
}

ParserWrapper::~ParserWrapper() {}
