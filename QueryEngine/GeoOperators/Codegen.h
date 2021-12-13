/*
 * Copyright 2021 OmniSci, Inc.
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

#include "Analyzer/Analyzer.h"
#include "QueryEngine/CodeGenerator.h"
#include "Shared/sqltypes.h"

#pragma once

namespace spatial_type {

class Codegen {
 public:
  Codegen(const Analyzer::GeoOperator* geo_operator) : operator_(geo_operator) {}

  static std::unique_ptr<Codegen> init(const Analyzer::GeoOperator* geo_operator);

  auto isNullable() const { return is_nullable_; }

  auto getTypeInfo() const { return operator_->get_type_info(); }

  std::string getName() const { return operator_->getName(); }

  virtual std::unique_ptr<CodeGenerator::NullCheckCodegen>
  getNullCheckCodegen(llvm::Value* null_lv, CgenState* cgen_state, Executor* executor);

  // number of loads/arguments for the operator
  virtual size_t size() const = 0;

  virtual SQLTypeInfo getNullType() const = 0;

  // by default index into the operator, but allow overloading for special cases. In those
  // special cases, we typically create a synthethic operator and manipulate state, so
  // this method cannot be const
  virtual const Analyzer::Expr* getOperand(const size_t index);

  // returns arguments lvs and null lv
  virtual std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) = 0;

  // codegen function operation post loads
  virtual std::vector<llvm::Value*> codegen(
      const std::vector<llvm::Value*>& args,
      CodeGenerator::NullCheckCodegen* nullcheck_codegen,
      CgenState* cgen_state,
      const CompilationOptions& co) = 0;

  virtual ~Codegen() {}

 protected:
  const Analyzer::GeoOperator* operator_;
  bool is_nullable_{true};
};

std::string suffix(SQLTypes type);

}  // namespace spatial_type
