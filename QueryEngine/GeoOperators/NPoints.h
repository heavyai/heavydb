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

#pragma once

#include "QueryEngine/GeoOperators/Codegen.h"
#include "Shared/sqltypes_geo.h"

namespace spatial_type {

class NPoints : public Codegen {
 public:
  NPoints(const Analyzer::GeoOperator* geo_operator)
      : Codegen(geo_operator) {}

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kINT); }

  const Analyzer::Expr* getOperand(const size_t index) final {
    CHECK_EQ(index, size_t(0));
    if (operand_owned_) {
      return operand_owned_.get();
    }

    const auto operand = operator_->getOperand(0);
    auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(operand);
    CHECK(col_var);

    geo_ti_ = col_var->get_type_info();
    CHECK(geo_ti_.is_geometry());
    is_nullable_ = !geo_ti_.get_notnull();

    // create a new operand which is just the coords and codegen it
    const auto coords_column_id = col_var->get_column_id() + 1;  // + 1 for coords
    auto ti = get_geo_physical_col_type(col_var->get_type_info(), 0);

    operand_owned_ = std::make_unique<Analyzer::ColumnVar>(
        ti, col_var->get_table_id(), coords_column_id, col_var->get_rte_idx());
    return operand_owned_.get();
  }

  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    CHECK_EQ(arg_lvs.size(), size_t(1));
    auto& argument_lv = arg_lvs.front();
    std::string fn_name("array_size");

    const auto& elem_ti = getOperand(0)->get_type_info().get_elem_type();
    std::vector<llvm::Value*> array_size_args{
        argument_lv,
        pos_lvs.front(),
        cgen_state->llInt(log2_bytes(elem_ti.get_logical_size()))};

    const bool is_nullable = isNullable();

    if (is_nullable) {
      fn_name += "_nullable";
      array_size_args.push_back(cgen_state->inlineIntNull(getTypeInfo()));
    }
    const auto coords_arr_sz_lv = cgen_state->emitExternalCall(
        fn_name, get_int_type(32, cgen_state->context_), array_size_args);
    return std::make_tuple(std::vector<llvm::Value*>{coords_arr_sz_lv}, coords_arr_sz_lv);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    CHECK_EQ(args.size(), size_t(1));

    // divide the coord size by the constant compression value and return it
    auto& builder = cgen_state->ir_builder_;
    llvm::Value* conversion_constant{nullptr};
    if (geo_ti_.get_compression() == kENCODING_GEOINT) {
      conversion_constant = cgen_state->llInt(4);
    } else {
      conversion_constant = cgen_state->llInt(8);
    }
    CHECK(conversion_constant);
    const auto total_num_pts = builder.CreateUDiv(args.front(), conversion_constant);
    const auto ret = builder.CreateUDiv(total_num_pts, cgen_state->llInt(2));
    if (isNullable()) {
      CHECK(nullcheck_codegen);
      return {nullcheck_codegen->finalize(cgen_state->inlineIntNull(getTypeInfo()), ret)};
    } else {
      return {ret};
    }
  }

 protected:
  SQLTypeInfo geo_ti_;
  std::unique_ptr<Analyzer::ColumnVar> operand_owned_;
};

}  // namespace spatial_type
