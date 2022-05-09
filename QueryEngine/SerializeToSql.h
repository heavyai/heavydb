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

#pragma once

#include "RelAlgExecutionUnit.h"
#include "ScalarExprVisitor.h"
#include "SchemaMgr/SchemaProvider.h"

class ScalarExprToSql : public ScalarExprVisitor<std::string> {
 public:
  ScalarExprToSql(const RelAlgExecutionUnit* ra_exe_unit,
                  SchemaProviderPtr schema_provider);

  std::string visitVar(const Analyzer::Var*) const override;

  std::string visitColumnVar(const Analyzer::ColumnVar* col_var) const override;

  std::string visitConstant(const Analyzer::Constant* constant) const override;

  std::string visitUOper(const Analyzer::UOper* uoper) const override;

  std::string visitBinOper(const Analyzer::BinOper* bin_oper) const override;

  std::string visitInValues(const Analyzer::InValues* in_values) const override;

  std::string visitLikeExpr(const Analyzer::LikeExpr* like) const override;

  std::string visitCaseExpr(const Analyzer::CaseExpr* case_) const override;

  std::string visitFunctionOper(const Analyzer::FunctionOper* func_oper) const override;

  std::string visitWindowFunction(
      const Analyzer::WindowFunction* window_func) const override;

  std::string visitAggExpr(const Analyzer::AggExpr* agg) const override;

  template <typename List>
  std::vector<std::string> visitList(const List& expressions) const;

 protected:
  std::string aggregateResult(const std::string& aggregate,
                              const std::string& next_result) const override {
    throw std::runtime_error("Expression not supported yet");
  }

  std::string defaultResult() const override {
    throw std::runtime_error("Expression not supported yet");
  }

 private:
  static std::string binOpTypeToString(const SQLOps op_type);

  const RelAlgExecutionUnit* ra_exe_unit_;
  SchemaProviderPtr schema_provider_;
};

std::string serialize_table_ref(int db_id,
                                const int table_id,
                                SchemaProviderPtr schema_provider);

std::string serialize_column_ref(int db_id,
                                 const int table_id,
                                 const int column_id,
                                 SchemaProviderPtr schema_provider);

struct ExecutionUnitSql {
  std::string query;
  std::string from_table;
};

ExecutionUnitSql serialize_to_sql(const RelAlgExecutionUnit* ra_exe_unit,
                                  SchemaProviderPtr schema_provider);
