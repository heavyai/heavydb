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

namespace spatial_type {

class NRings : public Codegen {
 public:
  NRings(const Analyzer::GeoOperator* geo_operator,
         const Catalog_Namespace::Catalog* catalog)
      : Codegen(geo_operator, catalog) {}

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

    const auto& geo_ti = col_var->get_type_info();
    CHECK(geo_ti.is_geometry());
    is_nullable_ = !geo_ti.get_notnull();

    // create a new operand which is just the ring sizes and codegen it
    const auto ring_sizes_column_id = col_var->get_column_id() + 2;  // + 2 for ring sizes
    auto ring_sizes_cd =
        get_column_descriptor(ring_sizes_column_id, col_var->get_table_id(), *cat_);
    CHECK(ring_sizes_cd);

    operand_owned_ = std::make_unique<Analyzer::ColumnVar>(ring_sizes_cd->columnType,
                                                           col_var->get_table_id(),
                                                           ring_sizes_column_id,
                                                           col_var->get_rte_idx());
    return operand_owned_.get();
  }

  // returns arguments lvs and null lv
  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    CHECK_EQ(pos_lvs.size(), size());
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
    const auto total_num_rings_lv = cgen_state->emitExternalCall(
        fn_name, get_int_type(32, cgen_state->context_), array_size_args);
    return std::make_tuple(std::vector<llvm::Value*>{total_num_rings_lv},
                           total_num_rings_lv);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    CHECK_EQ(args.size(), size_t(1));
    if (isNullable()) {
      CHECK(nullcheck_codegen);
      return {nullcheck_codegen->finalize(cgen_state->inlineIntNull(getTypeInfo()),
                                          args.front())};
    }
    return {args.front()};
  }

 protected:
  std::unique_ptr<Analyzer::ColumnVar> operand_owned_;
};

}  // namespace spatial_type
