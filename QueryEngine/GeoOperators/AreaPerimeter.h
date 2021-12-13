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

#include "Geospatial/Compression.h"

namespace spatial_type {

class AreaPerimeter : public Codegen {
 public:
  AreaPerimeter(const Analyzer::GeoOperator* geo_operator) : Codegen(geo_operator) {
    CHECK_EQ(operator_->size(), size_t(1));
    const auto& ti = operator_->get_type_info();
    is_nullable_ = !ti.get_notnull();
  }

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kINT); }

  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    CHECK_EQ(pos_lvs.size(), size());
    const auto operand = getOperand(0);
    CHECK(operand);
    const auto& operand_ti = operand->get_type_info();

    std::string size_fn_name = "array_size";
    if (is_nullable_) {
      size_fn_name += "_nullable";
    }

    const uint32_t coords_elem_sz_bytes =
        operand_ti.get_compression() == kENCODING_GEOINT ? 1 : 8;

    std::vector<llvm::Value*> operand_lvs;
    // iterate over column inputs
    if (dynamic_cast<const Analyzer::ColumnVar*>(operand)) {
      for (size_t i = 0; i < arg_lvs.size(); i++) {
        auto lv = arg_lvs[i];
        operand_lvs.push_back(
            cgen_state->emitExternalCall("array_buff",
                                         llvm::Type::getInt8PtrTy(cgen_state->context_),
                                         {lv, pos_lvs.front()}));
        const auto ptr_type = llvm::dyn_cast_or_null<llvm::PointerType>(lv->getType());
        CHECK(ptr_type);
        const auto elem_type = ptr_type->getElementType();
        CHECK(elem_type);
        std::vector<llvm::Value*> array_sz_args{
            lv,
            pos_lvs.front(),
            cgen_state->llInt(log2_bytes(i == 0 ? coords_elem_sz_bytes : 4))};
        if (is_nullable_) {  // TODO: should we do this for all arguments, or just points?
          array_sz_args.push_back(
              cgen_state->llInt(static_cast<int32_t>(inline_int_null_value<int32_t>())));
        }
        operand_lvs.push_back(cgen_state->emitExternalCall(
            size_fn_name, get_int_type(32, cgen_state->context_), array_sz_args));
      }
    } else {
      operand_lvs = arg_lvs;
    }
    CHECK_EQ(operand_lvs.size(),
             size_t(2 * operand_ti.get_physical_coord_cols()));  // array ptr and size

    // use the points array size argument for nullability
    return std::make_tuple(operand_lvs, is_nullable_ ? operand_lvs[1] : nullptr);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    auto operand_lvs = args;

    const auto& operand_ti = getOperand(0)->get_type_info();
    CHECK(operand_ti.get_type() == kPOLYGON || operand_ti.get_type() == kMULTIPOLYGON);
    const bool is_geodesic =
        operand_ti.get_subtype() == kGEOGRAPHY && operand_ti.get_output_srid() == 4326;

    std::string func_name = getName() + suffix(operand_ti.get_type());
    if (is_geodesic && getName() == "ST_Perimeter") {
      func_name += "_Geodesic";
    }

    // push back ic, isr, osr for now
    operand_lvs.push_back(
        cgen_state->llInt(Geospatial::get_compression_scheme(operand_ti)));  // ic
    operand_lvs.push_back(cgen_state->llInt(operand_ti.get_input_srid()));   // in srid
    operand_lvs.push_back(cgen_state->llInt(operand_ti.get_output_srid()));  // out srid

    const auto& ret_ti = operator_->get_type_info();
    CHECK(ret_ti.get_type() == kDOUBLE || ret_ti.get_type() == kFLOAT);

    auto ret = cgen_state->emitExternalCall(
        func_name,
        ret_ti.get_type() == kDOUBLE ? llvm::Type::getDoubleTy(cgen_state->context_)
                                     : llvm::Type::getFloatTy(cgen_state->context_),
        operand_lvs);
    if (is_nullable_) {
      CHECK(nullcheck_codegen);
      ret = nullcheck_codegen->finalize(cgen_state->inlineFpNull(ret_ti), ret);
    }
    return {ret};
  }
};

}  // namespace spatial_type
