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

class Centroid : public Codegen {
 public:
  Centroid(const Analyzer::GeoOperator* geo_operator) : Codegen(geo_operator) {
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
    auto& builder = cgen_state->ir_builder_;

    std::vector<llvm::Value*> operand_lvs;
    // iterate over column inputs
    if (dynamic_cast<const Analyzer::ColumnVar*>(operand)) {
      for (size_t i = 0; i < arg_lvs.size(); i++) {
        auto lv = arg_lvs[i];
        auto array_buff_lv =
            cgen_state->emitExternalCall("array_buff",
                                         llvm::Type::getInt8PtrTy(cgen_state->context_),
                                         {lv, pos_lvs.front()});
        if (i > 0) {
          array_buff_lv = builder.CreateBitCast(
              array_buff_lv, llvm::Type::getInt32PtrTy(cgen_state->context_));
        }
        operand_lvs.push_back(array_buff_lv);
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
        operand_lvs.push_back(builder.CreateSExt(
            cgen_state->emitExternalCall(
                size_fn_name, get_int_type(32, cgen_state->context_), array_sz_args),
            llvm::Type::getInt64Ty(cgen_state->context_)));
      }
    } else {
      for (size_t i = 0; i < arg_lvs.size(); i++) {
        auto arg_lv = arg_lvs[i];
        if (i > 0 && arg_lv->getType()->isPointerTy()) {
          arg_lv = builder.CreateBitCast(arg_lv,
                                         llvm::Type::getInt32PtrTy(cgen_state->context_));
        }
        operand_lvs.push_back(arg_lv);
      }
    }
    CHECK_EQ(operand_lvs.size(),
             size_t(2 * operand_ti.get_physical_coord_cols()));  // array ptr and size

    // note that this block is the only one that differs from Area/Perimeter
    // use the points array size argument for nullability
    llvm::Value* null_check_operand_lv{nullptr};
    if (is_nullable_) {
      null_check_operand_lv = operand_lvs[1];
      if (null_check_operand_lv->getType() !=
          llvm::Type::getInt32Ty(cgen_state->context_)) {
        CHECK(null_check_operand_lv->getType() ==
              llvm::Type::getInt64Ty(cgen_state->context_));
        // Geos functions come out 64-bit, cast down to 32 for now

        null_check_operand_lv = builder.CreateTrunc(
            null_check_operand_lv, llvm::Type::getInt32Ty(cgen_state->context_));
      }
    }

    return std::make_tuple(operand_lvs, null_check_operand_lv);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    UNREACHABLE();
    return {nullptr, 0};
  }
};

}  // namespace spatial_type
