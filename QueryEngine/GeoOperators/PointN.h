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

class PointN : public Codegen {
 public:
  PointN(const Analyzer::GeoOperator* geo_operator,
         const Catalog_Namespace::Catalog* catalog)
      : Codegen(geo_operator, catalog) {
    CHECK_EQ(operator_->size(), size_t(2));
    // set is nullable to true, because an index outside of the linestring will return
    // null
    // note we could probably just set this based on the operator type, as the operator
    // type needs to match
    this->is_nullable_ = true;
  }

  std::unique_ptr<CodeGenerator::NullCheckCodegen> getNullCheckCodegen(
      llvm::Value* null_lv,
      CgenState* cgen_state,
      Executor* executor) final {
    if (isNullable()) {
      CHECK(null_lv);
      return std::make_unique<CodeGenerator::NullCheckCodegen>(
          cgen_state, executor, null_lv, getNullType(), getName() + "_nullcheck");
    } else {
      return nullptr;
    }
  }

  size_t size() const final { return 2; }

  SQLTypeInfo getNullType() const final {
    // nullability is the expression `linestring is null OR size within bounds`
    return SQLTypeInfo(kBOOLEAN);
  }

  // returns arguments lvs and null lv
  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    CHECK_EQ(pos_lvs.size(), size());
    CHECK_EQ(pos_lvs.front(), pos_lvs.back());
    auto operand = getOperand(0);
    CHECK(operand);
    const auto& geo_ti = operand->get_type_info();
    CHECK(geo_ti.get_type() == kLINESTRING);

    auto& builder = cgen_state->ir_builder_;

    std::vector<llvm::Value*> array_operand_lvs;
    CHECK(!arg_lvs.empty());
    auto index_lv = builder.CreateMul(
        builder.CreateSub(arg_lvs.back(), cgen_state->llInt(static_cast<int32_t>(1))),
        cgen_state->llInt(static_cast<int32_t>(2)));
    llvm::Value* is_null_lv{nullptr};
    if (arg_lvs.size() == 2) {
      // col byte stream from column on disk
      array_operand_lvs.push_back(
          cgen_state->emitExternalCall("array_buff",
                                       llvm::Type::getInt8PtrTy(cgen_state->context_),
                                       {arg_lvs.front(), pos_lvs.front()}));
      const bool is_nullable = !geo_ti.get_notnull();
      std::string size_fn_name = "array_size";
      if (is_nullable) {
        size_fn_name += "_nullable";
      }

      uint32_t elem_sz = 1;  // TINYINT coords array
      std::vector<llvm::Value*> array_sz_args{
          arg_lvs.front(), pos_lvs.front(), cgen_state->llInt(log2_bytes(elem_sz))};
      if (is_nullable) {
        array_sz_args.push_back(
            cgen_state->llInt(static_cast<int32_t>(inline_int_null_value<int32_t>())));
      }
      array_operand_lvs.push_back(cgen_state->emitExternalCall(
          size_fn_name, get_int_type(32, cgen_state->context_), array_sz_args));

      auto geo_size_lv = array_operand_lvs.back();
      // convert the index to a byte index
      const auto outside_linestring_bounds_lv = builder.CreateNot(builder.CreateICmp(
          llvm::ICmpInst::ICMP_SLT,
          builder.CreateMul(index_lv, cgen_state->llInt(static_cast<int32_t>(8))),
          geo_size_lv));
      outside_linestring_bounds_lv->setName("outside_linestring_bounds");
      const auto input_is_null_lv = builder.CreateICmp(
          llvm::ICmpInst::ICMP_EQ,
          geo_size_lv,
          cgen_state->llInt(static_cast<int32_t>(inline_int_null_value<int32_t>())));
      input_is_null_lv->setName("input_is_null");
      is_null_lv = builder.CreateOr(outside_linestring_bounds_lv, input_is_null_lv);
    } else {
      CHECK_EQ(arg_lvs.size(), size_t(3));  // ptr, size, index
      array_operand_lvs.push_back(arg_lvs[0]);
      array_operand_lvs.push_back(arg_lvs[1]);

      const auto geo_size_lv = arg_lvs[1];
      // TODO: bounds indices are 64 bits but should be 32 bits, as array length is
      // limited to 32 bits
      is_null_lv = builder.CreateNot(
          builder.CreateICmp(llvm::ICmpInst::ICMP_SLT, index_lv, geo_size_lv));
    }
    array_operand_lvs.push_back(index_lv);
    return std::make_tuple(array_operand_lvs, is_null_lv);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    CHECK_EQ(args.size(), size_t(3));  // ptr, size, index
    const auto& geo_ti = getOperand(0)->get_type_info();
    CHECK(geo_ti.is_geometry());

    llvm::Value* array_buff_cast{nullptr};

    auto& builder = cgen_state->ir_builder_;
    if (geo_ti.get_compression() == kENCODING_GEOINT) {
      array_buff_cast = builder.CreateBitCast(
          args.front(), llvm::Type::getInt32PtrTy(cgen_state->context_));
    } else {
      array_buff_cast = builder.CreateBitCast(
          args.front(), llvm::Type::getDoublePtrTy(cgen_state->context_));
    }

    const auto index_lv = args.back();
    auto array_offset_lv =
        builder.CreateGEP(array_buff_cast, index_lv, operator_->getName() + "_Offset");
    CHECK(nullcheck_codegen);
    auto ret_lv = nullcheck_codegen->finalize(
        llvm::ConstantPointerNull::get(
            geo_ti.get_compression() == kENCODING_GEOINT
                ? llvm::PointerType::get(llvm::Type::getInt32Ty(cgen_state->context_), 0)
                : llvm::PointerType::get(llvm::Type::getDoubleTy(cgen_state->context_),
                                         0)),
        array_offset_lv);
    const auto geo_size_lv = args[1];
    return {ret_lv, geo_size_lv};
  }
};

}  // namespace spatial_type
