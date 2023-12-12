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

#pragma once

#include <regex>
#include "QueryEngine/GeoOperators/Codegen.h"

namespace spatial_type {

class NPoints : public Codegen {
 public:
  NPoints(const Analyzer::GeoOperator* geo_operator) : Codegen(geo_operator) {}

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kINT); }

  const Analyzer::Expr* getOperand(const size_t index) final {
    CHECK_EQ(index, size_t(0));
    if (col_var_owned_) {
      return col_var_owned_.get();
    }
    const auto operand = operator_->getOperand(0);
    geo_ti_ = operand->get_type_info();
    CHECK(geo_ti_.is_geometry());
    is_nullable_ = !geo_ti_.get_notnull();
    if (auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(operand)) {
      // create a new operand which is just the coords and codegen it
      auto column_key = col_var->getColumnKey();
      column_key.column_id = column_key.column_id + 1;  // + 1 for coords
      const auto coords_cd = get_column_descriptor(column_key);
      CHECK(coords_cd);
      col_var_owned_ = std::make_unique<Analyzer::ColumnVar>(
          coords_cd->columnType, column_key, col_var->get_rte_idx());
      return col_var_owned_.get();
    }
    return operand;
  }

  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    llvm::Value* coords_arr_sz_lv{nullptr};
    if (auto geo_constant = dynamic_cast<const Analyzer::GeoConstant*>(getOperand(0))) {
      // count points defined in the WKTString, i.e., POLYGON(1 1, 2 2, 3 3, 1 1)
      // the validation of the WKTString must be checked before entering this logic
      std::regex regex("-?[0-9]*\\.?[0-9]+\\s+-?[0-9]*\\.?[0-9]+");
      auto target = geo_constant->getWKTString();
      auto pt_begin = std::sregex_iterator(target.begin(), target.end(), regex);
      auto pt_end = std::sregex_iterator();
      auto num_pts = std::distance(pt_begin, pt_end);
      CHECK_GT(num_pts, 0);
      coords_arr_sz_lv = cgen_state->llInt(16 * num_pts);
    } else if (arg_lvs.size() == size_t(1)) {
      std::string fn_name("array_size");
      bool is_nullable = isNullable();
      if (auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(getOperand(0))) {
        auto column_key = col_var->getColumnKey();
        column_key.column_id = column_key.column_id - 1;
        const auto type_cd = get_column_descriptor(column_key);
        CHECK(type_cd);
        if (type_cd->columnType.get_type() == kPOINT) {
          fn_name = "point_coord_array_size";
        }
      }
      auto& argument_lv = arg_lvs.front();
      const auto& elem_ti = getOperand(0)->get_type_info().get_elem_type();
      std::vector<llvm::Value*> array_size_args{
          argument_lv,
          pos_lvs.front(),
          cgen_state->llInt(log2_bytes(elem_ti.get_logical_size()))};
      if (is_nullable) {
        fn_name += "_nullable";
        array_size_args.push_back(cgen_state->inlineIntNull(getTypeInfo()));
      }
      coords_arr_sz_lv = cgen_state->emitExternalCall(
          fn_name, get_int_type(32, cgen_state->context_), array_size_args);
    } else if (arg_lvs.size() == size_t(2)) {
      auto child_geo_oper =
          dynamic_cast<const Analyzer::GeoOperator*>(operator_->getOperand(0));
      CHECK(child_geo_oper);
      if (child_geo_oper->getName() == "ST_Point") {
        coords_arr_sz_lv = cgen_state->ir_builder_.CreateSelect(
            cgen_state->emitCall("point_double_is_null", {arg_lvs.front()}),
            cgen_state->inlineIntNull(getTypeInfo()),
            cgen_state->llInt(static_cast<int32_t>(16)));
      } else {
        CHECK(false) << "Not supported geo operator w/ ST_NPoints: "
                     << child_geo_oper->getName();
      }
    }
    CHECK(coords_arr_sz_lv);
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
  std::unique_ptr<Analyzer::ColumnVar> col_var_owned_;
};

}  // namespace spatial_type
