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

class StartEndPoint : public Codegen {
 public:
  StartEndPoint(const Analyzer::GeoOperator* geo_operator,
                const Catalog_Namespace::Catalog* catalog)
      : Codegen(geo_operator, catalog) {
    // nulls not supported yet
    this->is_nullable_ = false;
  }

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kNULLT); }

  const Analyzer::Expr* getPositionOperand() const final {
    return operator_->getOperand(0);
  }

  const Analyzer::Expr* getOperand(const size_t index) final {
    CHECK_EQ(operator_->size(), size_t(1));
    CHECK_EQ(index, size_t(0));

    return operator_->getOperand(0);
  }

  // returns arguments lvs and null lv
  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      llvm::Value* pos_lv,
      CgenState* cgen_state) final {
    // TODO: add null handling
    if (arg_lvs.size() == size_t(1)) {
      // col byte stream from column on disk
      auto operand = getOperand(0);
      CHECK(operand);
      const auto& geo_ti = operand->get_type_info();
      CHECK(geo_ti.get_type() == kLINESTRING);

      std::vector<llvm::Value*> array_operand_lvs;
      array_operand_lvs.push_back(
          cgen_state->emitExternalCall("array_buff",
                                       llvm::Type::getInt8PtrTy(cgen_state->context_),
                                       {arg_lvs.front(), pos_lv}));
      const bool is_nullable = !geo_ti.get_notnull();
      std::string size_fn_name = "array_size";
      if (is_nullable) {
        size_fn_name += "_nullable";
      }
      uint32_t elem_sz = 1;  // TINYINT coords array
      std::vector<llvm::Value*> array_sz_args{
          arg_lvs.front(), pos_lv, cgen_state->llInt(log2_bytes(elem_sz))};
      if (is_nullable) {
        array_sz_args.push_back(
            cgen_state->llInt(static_cast<int32_t>(inline_int_null_value<int32_t>())));
      }
      array_operand_lvs.push_back(cgen_state->emitExternalCall(
          size_fn_name, get_int_type(32, cgen_state->context_), array_sz_args));
      return std::make_tuple(array_operand_lvs, nullptr);
    }
    CHECK_EQ(arg_lvs.size(), size_t(2));  // ptr, size
    return std::make_tuple(arg_lvs, nullptr);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state) final {
    CHECK_EQ(args.size(), size_t(2));  // ptr, size
    const auto& geo_ti = getOperand(0)->get_type_info();
    CHECK(geo_ti.is_geometry());

    auto& builder = cgen_state->ir_builder_;
    llvm::Value* array_buff_cast{nullptr};
    int32_t elem_size_bytes = 0;
    if (geo_ti.get_compression() == kENCODING_GEOINT) {
      array_buff_cast = builder.CreateBitCast(
          args.front(), llvm::Type::getInt32PtrTy(cgen_state->context_));
      elem_size_bytes = 4;  // 4-byte ints
    } else {
      array_buff_cast = builder.CreateBitCast(
          args.front(), llvm::Type::getDoublePtrTy(cgen_state->context_));
      elem_size_bytes = 8;  // doubles
    }
    CHECK_GT(elem_size_bytes, 0);
    const bool is_end_point = getName() == "ST_EndPoint";
    const auto num_elements_lv =
        builder.CreateSDiv(args.back(), cgen_state->llInt(elem_size_bytes));
    const auto index_lv =
        is_end_point ? builder.CreateSub(num_elements_lv, cgen_state->llInt(int32_t(2)))
                     : cgen_state->llInt(int32_t(0));
    auto array_offset_lv =
        builder.CreateGEP(array_buff_cast, index_lv, operator_->getName() + "_Offset");
    return {array_offset_lv, args.back()};
  }
};

}  // namespace spatial_type
