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

#include "Shared/ConfigResolve.h"

enum class CalciteDMLPathSelection : int {
  Unsupported = 0,
  OnlyUpdates = 1,
  OnlyDeletes = 2,
  UpdatesAndDeletes = 3,
};

constexpr inline CalciteDMLPathSelection yield_dml_path_selector() {
  int selector = 0;
  if (std::is_same<CalciteDeletePathSelector, PreprocessorTrue>::value) {
    selector |= 0x02;
  }
  if (std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    selector |= 0x01;
  }
  return static_cast<CalciteDMLPathSelection>(selector);
}

class ParserWrapper {
 public:
  // HACK:  This needs to go away as calcite takes over parsing
  enum class DMLType : int { Insert = 0, Delete, Update, Upsert, NotDML };

  enum class ExplainType { None, IR, OptimizedIR, Calcite, Other };

  ParserWrapper(std::string query_string);
  std::string process(std::string user,
                      std::string passwd,
                      std::string catalog,
                      std::string sql_string,
                      const bool legacy_syntax);
  virtual ~ParserWrapper();

  bool is_ddl = false;
  bool is_update_dml = false;
  bool is_ctas = false;
  bool is_itas = false;
  bool is_copy = false;
  bool is_copy_to = false;
  bool is_optimize = false;
  bool is_validate = false;
  std::string actual_query;

  DMLType getDMLType() const { return dml_type_; }

  ExplainType getExplainType() const { return explain_type_; }

  bool isCalciteExplain() const { return explain_type_ == ExplainType::Calcite; }

  bool isSelectExplain() const {
    return explain_type_ == ExplainType::Calcite || explain_type_ == ExplainType::IR ||
           explain_type_ == ExplainType::OptimizedIR;
  }

  bool isIRExplain() const {
    return explain_type_ == ExplainType::IR || explain_type_ == ExplainType::OptimizedIR;
  }

  bool isCalcitePathPermissable(bool read_only_mode = false) {
    return (!is_ddl && !is_optimize && !is_validate &&
            isCalcitePermissableDml(read_only_mode) &&
            !(explain_type_ == ExplainType::Other));
  }

  bool isOtherExplain() const { return explain_type_ == ExplainType::Other; }

  bool isCalcitePermissableDml(bool read_only_mode) {
    if (read_only_mode) {
      return !is_update_dml;  // If we're read-only, no DML is permissable
    }

    // TODO: A good place for if-constexpr
    switch (yield_dml_path_selector()) {
      case CalciteDMLPathSelection::OnlyUpdates:
        return !is_update_dml || (getDMLType() == ParserWrapper::DMLType::Update);
      case CalciteDMLPathSelection::OnlyDeletes:
        return !is_update_dml || (getDMLType() == ParserWrapper::DMLType::Delete);
      case CalciteDMLPathSelection::UpdatesAndDeletes:
        return !is_update_dml || (getDMLType() == ParserWrapper::DMLType::Delete) ||
               (getDMLType() == ParserWrapper::DMLType::Update);
      case CalciteDMLPathSelection::Unsupported:
      default:
        return false;
    }
  }

 private:
  DMLType dml_type_ = DMLType::NotDML;
  ExplainType explain_type_ = ExplainType::None;

  static const std::vector<std::string> ddl_cmd;
  static const std::vector<std::string> update_dml_cmd;
  static const std::string explain_str;
  static const std::string calcite_explain_str;
  static const std::string optimized_explain_str;
  static const std::string optimize_str;
  static const std::string validate_str;
};
