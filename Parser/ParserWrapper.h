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

/**
 * @file    ParserWrapper.h
 * @author  michael
 * @brief   Classes used to wrap parser calls for calcite redirection
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#pragma once

#include <boost/regex.hpp>
#include <string>
#include <vector>

struct ExplainInfo {
  bool explain;
  bool explain_optimized;
  bool explain_plan;
  bool calcite_explain;

  static ExplainInfo defaults() { return ExplainInfo{false, false, false, false}; }

  bool justExplain() const { return explain || explain_plan || explain_optimized; }

  bool justCalciteExplain() const { return calcite_explain; }
};

class ParserWrapper {
 public:
  // HACK:  This needs to go away as calcite takes over parsing
  enum class DMLType : int { Insert = 0, Delete, Update, Upsert, NotDML };

  enum class ExplainType { None, IR, OptimizedIR, Calcite, ExecutionPlan, Other };

  enum class QueryType { Unknown, Read, Write, SchemaRead, SchemaWrite };

  ParserWrapper(std::string query_string);
  std::string process(std::string user,
                      std::string passwd,
                      std::string catalog,
                      std::string sql_string,
                      const bool legacy_syntax);
  virtual ~ParserWrapper();

  bool is_ddl = false;
  // is_update_dml does not imply UPDATE,
  // but rather any of the statement types: INSERT DELETE UPDATE UPSERT
  bool is_update_dml = false;
  bool is_ctas = false;
  bool is_itas = false;
  bool is_copy = false;
  bool is_copy_to = false;
  bool is_optimize = false;
  bool is_validate = false;
  std::string actual_query;

  DMLType getDMLType() const { return dml_type_; }

  ExplainInfo getExplainInfo() const;

  ExplainType getExplainType() const { return explain_type_; }

  QueryType getQueryType() const { return query_type_; }

  bool isUpdateDelete() const {
    return dml_type_ == DMLType::Update || dml_type_ == DMLType::Delete;
  }

  bool isCalciteExplain() const { return explain_type_ == ExplainType::Calcite; }

  bool isPlanExplain() const { return explain_type_ == ExplainType::ExecutionPlan; }

  bool isSelectExplain() const {
    return explain_type_ == ExplainType::Calcite || explain_type_ == ExplainType::IR ||
           explain_type_ == ExplainType::OptimizedIR ||
           explain_type_ == ExplainType::ExecutionPlan;
  }

  bool isIRExplain() const {
    return explain_type_ == ExplainType::IR || explain_type_ == ExplainType::OptimizedIR;
  }

  bool isCalcitePathPermissable(bool read_only_mode = false) {
    if (is_calcite_ddl_) {
      return isCalcitePermissableDdl(read_only_mode);
    }
    return (!is_legacy_ddl_ && !is_optimize && !is_validate &&
            isCalcitePermissableDml(read_only_mode) &&
            !(explain_type_ == ExplainType::Other));
  }

  bool isOtherExplain() const { return explain_type_ == ExplainType::Other; }

  bool isCalcitePermissableDml(bool read_only_mode = false) {
    if (is_itas) {
      return !read_only_mode;
    }
    if (read_only_mode) {
      return !is_update_dml;  // If we're read-only, no update/delete DML is permissable
    }
    return !is_update_dml || (getDMLType() == ParserWrapper::DMLType::Delete) ||
           (getDMLType() == ParserWrapper::DMLType::Update);
  }

  bool isCalcitePermissableDdl(bool read_only_mode) {
    if (read_only_mode) {
      // If we're read-only, no Write/SchemaWrite DDL is permissable
      return query_type_ != QueryType::Write && query_type_ != QueryType::SchemaWrite;
    }
    return true;
  }

  bool isCalciteDdl() const { return is_calcite_ddl_; }

 private:
  DMLType dml_type_ = DMLType::NotDML;
  ExplainType explain_type_ = ExplainType::None;
  QueryType query_type_ = QueryType::Unknown;

  static const std::vector<std::string> ddl_cmd;
  static const std::vector<std::string> update_dml_cmd;
  static const std::string explain_str;
  static const std::string calcite_explain_str;
  static const std::string optimized_explain_str;
  static const std::string plan_explain_str;
  static const std::string optimize_str;
  static const std::string validate_str;

  bool is_legacy_ddl_ = false;
  bool is_calcite_ddl_ = false;
};
