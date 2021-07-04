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

// ST_X and ST_Y
class PointAccessors : public Codegen {
 public:
  PointAccessors(const Analyzer::GeoOperator* geo_operator,
                 const Catalog_Namespace::Catalog* catalog)
      : Codegen(geo_operator, catalog) {}

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kBOOLEAN); }

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
    auto operand = getOperand(0);
    CHECK(operand);
    const auto& geo_ti = operand->get_type_info();
    CHECK(geo_ti.is_geometry());
    auto& builder = cgen_state->ir_builder_;

    llvm::Value* array_buff_ptr{nullptr};
    llvm::Value* is_null{nullptr};
    if (arg_lvs.size() == 1) {
      if (dynamic_cast<const Analyzer::GeoExpr*>(operand)) {
        const auto ptr_type =
            llvm::dyn_cast<llvm::PointerType>(arg_lvs.front()->getType());
        CHECK(ptr_type);
        const auto is_null_lv =
            builder.CreateICmp(llvm::CmpInst::ICMP_EQ,
                               arg_lvs.front(),
                               llvm::ConstantPointerNull::get(ptr_type));
        return std::make_tuple(arg_lvs, is_null_lv);
      }
      // col byte stream, get the array buffer ptr and is null attributes and cache
      auto arr_load_lvs = CodeGenerator::codegenGeoArrayLoadAndNullcheck(
          arg_lvs.front(), pos_lv, geo_ti, cgen_state);
      array_buff_ptr = arr_load_lvs.buffer;
      is_null = arr_load_lvs.is_null;
    } else {
      // ptr and size
      CHECK_EQ(arg_lvs.size(), size_t(2));
      if (dynamic_cast<const Analyzer::GeoOperator*>(operand)) {
        // null check will be if the ptr is a nullptr
        is_null = builder.CreateICmp(
            llvm::CmpInst::ICMP_EQ,
            arg_lvs.front(),
            llvm::ConstantPointerNull::get(  // TODO: check ptr address space
                geo_ti.get_compression() == kENCODING_GEOINT
                    ? llvm::Type::getInt32PtrTy(cgen_state->context_)
                    : llvm::Type::getDoublePtrTy(cgen_state->context_)));
      }

      // TODO: nulls from other types not yet supported
      array_buff_ptr = arg_lvs.front();
    }
    CHECK(array_buff_ptr) << operator_->toString();
    if (!is_null) {
      is_nullable_ = false;
    }
    return std::make_tuple(std::vector<llvm::Value*>{array_buff_ptr}, is_null);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state) final {
    CHECK_EQ(args.size(), size_t(1));
    const auto array_buff_ptr = args.front();

    const auto& geo_ti = getOperand(0)->get_type_info();
    CHECK(geo_ti.is_geometry());
    auto& builder = cgen_state->ir_builder_;

    const bool is_x = operator_->getName() == "ST_X";
    const std::string expr_name = is_x ? "x" : "y";

    llvm::Value* coord_lv;
    if (geo_ti.get_compression() == kENCODING_GEOINT) {
      auto compressed_arr_ptr = builder.CreateBitCast(
          array_buff_ptr, llvm::Type::getInt32PtrTy(cgen_state->context_));
      auto coord_index = is_x ? cgen_state->llInt(0) : cgen_state->llInt(1);
      auto coord_lv_ptr =
          builder.CreateGEP(compressed_arr_ptr, coord_index, expr_name + "_coord_ptr");
      auto compressed_coord_lv =
          builder.CreateLoad(coord_lv_ptr, expr_name + "_coord_compressed");

      coord_lv =
          cgen_state->emitExternalCall("decompress_" + expr_name + "_coord_geoint",
                                       llvm::Type::getDoubleTy(cgen_state->context_),
                                       {compressed_coord_lv});
    } else {
      auto coord_arr_ptr = builder.CreateBitCast(
          array_buff_ptr, llvm::Type::getDoublePtrTy(cgen_state->context_));
      auto coord_index = is_x ? cgen_state->llInt(0) : cgen_state->llInt(1);
      auto coord_lv_ptr =
          builder.CreateGEP(coord_arr_ptr, coord_index, expr_name + "_coord_ptr");
      coord_lv = builder.CreateLoad(coord_lv_ptr, expr_name + "_coord");
    }

    // TODO: do this with transformation nodes explicitly
    if (geo_ti.get_input_srid() != geo_ti.get_output_srid()) {
      if (geo_ti.get_input_srid() == 4326) {
        if (geo_ti.get_output_srid() == 900913) {
          // convert WGS 84 -> Web mercator
          coord_lv =
              cgen_state->emitExternalCall("conv_4326_900913_" + expr_name,
                                           llvm::Type::getDoubleTy(cgen_state->context_),
                                           {coord_lv});
          coord_lv->setName(expr_name + "_coord_transformed");
        } else {
          throw std::runtime_error("Unsupported geo transformation: " +
                                   std::to_string(geo_ti.get_input_srid()) + " to " +
                                   std::to_string(geo_ti.get_output_srid()));
        }
      } else {
        throw std::runtime_error(
            "Unsupported geo transformation: " + std::to_string(geo_ti.get_input_srid()) +
            " to " + std::to_string(geo_ti.get_output_srid()));
      }
    }

    auto ret = coord_lv;
    if (is_nullable_) {
      CHECK(nullcheck_codegen);
      ret = nullcheck_codegen->finalize(cgen_state->inlineFpNull(SQLTypeInfo(kDOUBLE)),
                                        ret);
    }
    const auto key = operator_->toString();
    CHECK(cgen_state->geo_target_cache_.insert(std::make_pair(key, ret)).second);
    return {ret};
  }
};

}  // namespace spatial_type
