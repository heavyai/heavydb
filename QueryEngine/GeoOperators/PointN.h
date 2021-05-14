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
    // set is nullable to true, because an index outside of the linestring will return
    // null
    // note we could probably just set this based on the operator type, as the operator
    // type needs to match
    this->is_nullable_ = true;
  }

  size_t size() const final { return 2; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kBOOLEAN); }

  const Analyzer::Expr* getPositionOperand() const final { return nullptr; }

  const Analyzer::Expr* getOperand(const size_t index) final {
    CHECK_EQ(operator_->size(), size_t(2));
    CHECK_LT(index, operator_->size());

    if (index == 0) {
      const auto arg = operator_->getOperand(index);
      // first argument should be geo literal
      const auto geo_arg_expr = dynamic_cast<const Analyzer::GeoExpr*>(arg);
      CHECK(geo_arg_expr) << arg->toString();
      return arg;
    } else {
      return operator_->getOperand(index);
    }
  }

  // returns arguments lvs and null lv
  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      llvm::Value* pos_lv,
      CgenState* cgen_state) final {
    CHECK_EQ(arg_lvs.size(), size_t(3));  // ptr, size, index
    auto& builder = cgen_state->ir_builder_;
    const auto index_lv = arg_lvs[2];
    const auto geo_size_lv = arg_lvs[1];
    const auto is_null_lv = builder.CreateNot(
        builder.CreateICmp(llvm::ICmpInst::ICMP_SLT, index_lv, geo_size_lv));
    return std::make_tuple(arg_lvs, is_null_lv);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state) final {
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

    const auto index_lv = args[2];
    auto array_offset_lv =
        builder.CreateGEP(array_buff_cast, index_lv, operator_->getName() + "_Offset");
    CHECK(nullcheck_codegen);
    auto ret_lv = nullcheck_codegen->finalize(
        llvm::ConstantPointerNull::get(
            geo_ti.get_compression() == kENCODING_GEOINT
                ? llvm::Type::getInt32PtrTy(cgen_state->context_)
                : llvm::Type::getDoublePtrTy(cgen_state->context_)),
        array_offset_lv);
    const auto geo_size_lv = args[1];
    return {ret_lv, geo_size_lv};
  }
};

}  // namespace spatial_type
