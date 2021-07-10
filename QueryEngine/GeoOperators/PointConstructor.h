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

#include "Shared/SqlTypesLayout.h"

namespace spatial_type {

// ST_Point
class PointConstructor : public Codegen {
 public:
  PointConstructor(const Analyzer::GeoOperator* geo_operator,
                   const Catalog_Namespace::Catalog* catalog)
      : Codegen(geo_operator, catalog) {
    const auto& ti = geo_operator->get_type_info();
    if (ti.get_notnull()) {
      is_nullable_ = false;
    } else {
      is_nullable_ = true;
    }
  }

  std::unique_ptr<CodeGenerator::NullCheckCodegen> getNullCheckCodegen(
      llvm::Value* null_lv,
      CgenState* cgen_state,
      Executor* executor) final {
    if (isNullable()) {
      // do the standard nullcheck codegen, but modify the null basic block to emplace the
      // null sentinel into the array
      auto nullcheck_codegen = std::make_unique<CodeGenerator::NullCheckCodegen>(
          cgen_state, executor, null_lv, getNullType(), getName() + "_nullcheck");

      auto& builder = cgen_state->ir_builder_;
      CHECK(pt_local_storage_lv_);

      auto prev_insert_block = builder.GetInsertBlock();
      auto crt_insert_block = nullcheck_codegen->null_check->cond_true_;
      CHECK(crt_insert_block);
      auto& instruction_list = crt_insert_block->getInstList();
      CHECK_EQ(instruction_list.size(), size_t(1));
      builder.SetInsertPoint(crt_insert_block, instruction_list.begin());

      auto x_coord_ptr = builder.CreateGEP(pt_local_storage_lv_,
                                           {cgen_state->llInt(0), cgen_state->llInt(0)},
                                           "x_coord_ptr");
      const auto& geo_ti = operator_->get_type_info();
      if (geo_ti.get_compression() == kENCODING_GEOINT) {
        // TODO: probably wrong
        builder.CreateStore(cgen_state->llInt(inline_int_null_val(SQLTypeInfo(kINT))),
                            x_coord_ptr);
      } else {
        builder.CreateStore(cgen_state->llFp(static_cast<double>(NULL_ARRAY_DOUBLE)),
                            x_coord_ptr);
      }
      builder.SetInsertPoint(prev_insert_block);

      return nullcheck_codegen;
    } else {
      return nullptr;
    }
  }

  size_t size() const final { return 2; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kBOOLEAN); }

  const Analyzer::Expr* getPositionOperand() const final {
    return operator_->getOperand(0);
  }

  const Analyzer::Expr* getOperand(const size_t index) final {
    CHECK_EQ(operator_->size(), size_t(2));
    return operator_->getOperand(index);
  }

  // returns arguments lvs and null lv
  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      llvm::Value* pos_lv,
      CgenState* cgen_state) final {
    CHECK_EQ(arg_lvs.size(), size_t(2));

    auto& builder = cgen_state->ir_builder_;

    llvm::Value* is_null{nullptr};
    auto x_operand = getOperand(0);
    const auto& x_ti = x_operand->get_type_info();
    if (!x_ti.get_notnull()) {
      CHECK(x_ti.is_integer() || x_ti.is_fp());
      // TODO: centralize nullcheck logic for all sqltypes
      is_null = x_ti.is_integer()
                    ? builder.CreateICmp(llvm::CmpInst::ICMP_EQ,
                                         arg_lvs.front(),
                                         cgen_state->llInt(inline_int_null_val(x_ti)))
                    : builder.CreateFCmp(llvm::FCmpInst::FCMP_OEQ,
                                         arg_lvs.front(),
                                         cgen_state->llFp(inline_fp_null_val(x_ti)));
    }

    auto y_operand = getOperand(0);
    const auto& y_ti = y_operand->get_type_info();
    if (!y_ti.get_notnull()) {
      auto y_is_null =
          y_ti.is_integer()
              ? builder.CreateICmp(llvm::CmpInst::ICMP_EQ,
                                   arg_lvs.front(),
                                   cgen_state->llInt(inline_int_null_val(y_ti)))
              : builder.CreateFCmp(llvm::FCmpInst::FCMP_OEQ,
                                   arg_lvs.front(),
                                   cgen_state->llFp(inline_fp_null_val(y_ti)));
      if (is_null) {
        is_null = builder.CreateAnd(is_null, y_is_null);
      } else {
        is_null = y_is_null;
      }
    }

    if (is_nullable_ && !is_null) {
      // if the inputs are not null, set the output to be not null
      // TODO: we should do this in the translator and just confirm it here
      is_nullable_ = false;
    }

    // do the alloca before nullcheck codegen, as either way we will return a point array
    const auto& geo_ti = operator_->get_type_info();
    CHECK(geo_ti.get_type() == kPOINT);

    llvm::ArrayType* arr_type{nullptr};
    if (geo_ti.get_compression() == kENCODING_GEOINT) {
      auto elem_ty = llvm::Type::getInt32Ty(cgen_state->context_);
      arr_type = llvm::ArrayType::get(elem_ty, 2);
    } else {
      CHECK(geo_ti.get_compression() == kENCODING_NONE);
      auto elem_ty = llvm::Type::getDoubleTy(cgen_state->context_);
      arr_type = llvm::ArrayType::get(elem_ty, 2);
    }
    pt_local_storage_lv_ =
        builder.CreateAlloca(arr_type, nullptr, operator_->getName() + "_Local_Storage");

    return std::make_tuple(arg_lvs, is_null);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state) final {
    CHECK_EQ(args.size(), size_t(2));

    const auto& geo_ti = operator_->get_type_info();
    CHECK(geo_ti.get_type() == kPOINT);

    auto& builder = cgen_state->ir_builder_;
    CHECK(pt_local_storage_lv_);

    const bool is_compressed = geo_ti.get_compression() == kENCODING_GEOINT;

    // store x coord
    auto x_coord_ptr = builder.CreateGEP(pt_local_storage_lv_,
                                         {cgen_state->llInt(0), cgen_state->llInt(0)},
                                         "x_coord_ptr");
    if (is_compressed) {
      auto compressed_lv =
          cgen_state->emitExternalCall("compress_x_coord_geoint",
                                       llvm::Type::getInt32Ty(cgen_state->context_),
                                       {args.front()});
      builder.CreateStore(compressed_lv, x_coord_ptr);
    } else {
      builder.CreateStore(args.front(), x_coord_ptr);
    }

    // store y coord
    auto y_coord_ptr = builder.CreateGEP(pt_local_storage_lv_,
                                         {cgen_state->llInt(0), cgen_state->llInt(1)},
                                         "y_coord_ptr");
    if (is_compressed) {
      auto compressed_lv =
          cgen_state->emitExternalCall("compress_y_coord_geoint",
                                       llvm::Type::getInt32Ty(cgen_state->context_),
                                       {args.back()});
      builder.CreateStore(compressed_lv, y_coord_ptr);
    } else {
      builder.CreateStore(args.back(), y_coord_ptr);
    }

    llvm::Value* ret = pt_local_storage_lv_;
    if (is_nullable_) {
      CHECK(nullcheck_codegen);
      ret = nullcheck_codegen->finalize(ret, ret);
    }
    return {
        builder.CreateBitCast(ret,
                              geo_ti.get_compression() == kENCODING_GEOINT
                                  ? llvm::Type::getInt32PtrTy(cgen_state->context_)
                                  : llvm::Type::getDoublePtrTy(cgen_state->context_))};
  }

 private:
  llvm::AllocaInst* pt_local_storage_lv_{nullptr};
};

}  // namespace spatial_type
