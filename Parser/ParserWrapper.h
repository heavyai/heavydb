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
 * @file    ParserWrapper.h
 * @brief   Classes used to wrap parser calls for calcite redirection
 *
 */

// TODO(sy): We already use Calcite, which is a powerful general-purpose parser tool, so
// ParserWrapper has no good reason to exist. Any work happening in here should be moved
// into our Java Calcite classes and ParserWrapper should be removed from HeavyDB.

#pragma once

#include <string>
#include <vector>

#include "Shared/clean_boost_regex.hpp"

class ExplainInfo {
 public:
  enum class ExplainType {
    None,
    IR,
    OptimizedIR,
    Calcite,
    CalciteDetail,
    ExecutionPlan,
    Other
  };

  ExplainInfo() : explain_type_(ExplainType::None), actual_query_("") {}
  ExplainInfo(ExplainType type) : explain_type_(type), actual_query_("") {}
  ExplainInfo(std::string query_string);

  bool isExplain() const { return explain_type_ != ExplainType::None; }

  bool isJustExplain() const {
    return explain_type_ == ExplainType::IR ||
           explain_type_ == ExplainType::OptimizedIR ||
           explain_type_ == ExplainType::ExecutionPlan;
  }

  bool isSelectExplain() const {
    return explain_type_ == ExplainType::IR ||
           explain_type_ == ExplainType::OptimizedIR ||
           explain_type_ == ExplainType::Calcite ||
           explain_type_ == ExplainType::CalciteDetail ||
           explain_type_ == ExplainType::ExecutionPlan;
  }

  bool isIRExplain() const {
    return explain_type_ == ExplainType::IR || explain_type_ == ExplainType::OptimizedIR;
  }

  bool isOptimizedExplain() const { return explain_type_ == ExplainType::OptimizedIR; }
  bool isCalciteExplain() const {
    return explain_type_ == ExplainType::Calcite ||
           explain_type_ == ExplainType::CalciteDetail;
  }
  bool isCalciteExplainDetail() const {
    return explain_type_ == ExplainType::CalciteDetail;
  }
  bool isPlanExplain() const { return explain_type_ == ExplainType::ExecutionPlan; }
  bool isOtherExplain() const { return explain_type_ == ExplainType::Other; }

  std::string ActualQuery() { return actual_query_; }

  bool isVerbose() const { return verbose_; }

 private:
  ExplainType explain_type_ = ExplainType::None;
  std::string actual_query_ = "";
  bool verbose_{false};
};

class ParserWrapper {
 public:
  // HACK:  This needs to go away as calcite takes over parsing
  enum class DMLType : int { Insert = 0, Delete, Update, NotDML };

  enum class QueryType { Unknown, Read, Write, SchemaRead, SchemaWrite };

  ParserWrapper(std::string query_string);
  std::string process(std::string user,
                      std::string passwd,
                      std::string catalog,
                      std::string sql_string,
                      const bool legacy_syntax);
  virtual ~ParserWrapper();

  bool is_ddl = false;
  bool is_update_dml = false;  // INSERT DELETE UPDATE
  bool is_ctas = false;
  bool is_itas = false;
  bool is_copy = false;
  bool is_copy_to = false;
  bool is_validate = false;
  bool is_other_explain = false;
  bool is_refresh = false;

  DMLType getDMLType() const { return dml_type_; }

  QueryType getQueryType() const { return query_type_; }

  bool isUpdateDelete() const {
    return dml_type_ == DMLType::Update || dml_type_ == DMLType::Delete;
  }

 private:
  DMLType dml_type_ = DMLType::NotDML;
  QueryType query_type_ = QueryType::Unknown;

  static const std::vector<std::string> ddl_cmd;
  static const std::vector<std::string> update_dml_cmd;
};
