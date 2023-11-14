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

#include "QueryEngine/GeoOperators/Codegen.h"

namespace spatial_type {

class PointN : public Codegen {
 public:
  PointN(const Analyzer::GeoOperator* geo_operator) : Codegen(geo_operator) {
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

  llvm::Value* codegenGeoSize(CgenState* cgen_state,
                              SQLTypeInfo const& geo_ti,
                              const std::vector<llvm::Value*>& arg_lvs,
                              const std::vector<llvm::Value*>& pos_lvs) {
    llvm::Value* geo_size_lv{nullptr};
    if (arg_lvs.size() == 2) {
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
      geo_size_lv = cgen_state->emitExternalCall(
          size_fn_name, get_int_type(32, cgen_state->context_), array_sz_args);
    } else {
      geo_size_lv = arg_lvs[1];
    }
    CHECK(geo_size_lv);
    return geo_size_lv;
  }

  llvm::Value* codegenIndexOutOfBoundCheck(CgenState* cgen_state,
                                           llvm::Value* index_lv,
                                           llvm::Value* geosize_lv) {
    llvm::Value* is_null_lv = cgen_state->llBool(false);
    is_null_lv = cgen_state->ir_builder_.CreateOr(
        is_null_lv,
        cgen_state->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_SLT,
                                           index_lv,
                                           cgen_state->llInt(static_cast<int32_t>(0))));
    return cgen_state->ir_builder_.CreateOr(
        is_null_lv,
        cgen_state->ir_builder_.CreateICmp(
            llvm::ICmpInst::ICMP_SGE,
            cgen_state->ir_builder_.CreateMul(index_lv,
                                              cgen_state->llInt(static_cast<int32_t>(8))),
            geosize_lv));
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
    llvm::Value* raw_index_lv = arg_lvs.back();
    llvm::Value* geo_size_lv = codegenGeoSize(cgen_state, geo_ti, arg_lvs, pos_lvs);
    llvm::Value* pt_size_lv = cgen_state->llInt(16);
    llvm::Value* num_pts_lv = builder.CreateUDiv(geo_size_lv, pt_size_lv);
    llvm::Value* is_negative_lv =
        builder.CreateCmp(llvm::CmpInst::ICMP_SLT, raw_index_lv, cgen_state->llInt(0));
    llvm::Value* negative_raw_index_lv = builder.CreateAdd(raw_index_lv, num_pts_lv);
    llvm::Value* positive_raw_index_lv =
        builder.CreateSub(raw_index_lv, cgen_state->llInt(1));
    raw_index_lv = builder.CreateSelect(
        is_negative_lv, negative_raw_index_lv, positive_raw_index_lv);
    raw_index_lv =
        builder.CreateMul(raw_index_lv, cgen_state->llInt(static_cast<int32_t>(2)));
    llvm::Value* is_null_lv =
        codegenIndexOutOfBoundCheck(cgen_state, raw_index_lv, geo_size_lv);
    if (arg_lvs.size() == 2) {
      // col byte stream from column on disk
      array_operand_lvs.push_back(
          cgen_state->emitExternalCall("array_buff",
                                       llvm::Type::getInt8PtrTy(cgen_state->context_),
                                       {arg_lvs.front(), pos_lvs.front()}));
      array_operand_lvs.push_back(geo_size_lv);
      // convert the index to a byte index
      raw_index_lv =
          builder.CreateMul(raw_index_lv, cgen_state->llInt(static_cast<int32_t>(8)));
      const auto input_is_null_lv = builder.CreateICmp(
          llvm::ICmpInst::ICMP_EQ,
          geo_size_lv,
          cgen_state->llInt(static_cast<int32_t>(inline_int_null_value<int32_t>())));
      is_null_lv = builder.CreateOr(is_null_lv, input_is_null_lv);
    } else {
      CHECK_EQ(arg_lvs.size(), size_t(3));  // ptr, size, index
      array_operand_lvs.push_back(arg_lvs[0]);
      array_operand_lvs.push_back(arg_lvs[1]);
    }
    array_operand_lvs.push_back(raw_index_lv);
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
    auto array_offset_lv = builder.CreateGEP(
        array_buff_cast->getType()->getScalarType()->getPointerElementType(),
        array_buff_cast,
        index_lv,
        operator_->getName() + "_Offset");
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
