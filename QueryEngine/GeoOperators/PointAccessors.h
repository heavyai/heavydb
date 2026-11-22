/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryEngine/GeoOperators/Codegen.h"
#include "QueryEngine/GeoOperators/Transform.h"

namespace spatial_type {

// ST_X and ST_Y
class PointAccessors : public Codegen {
 public:
  PointAccessors(const Analyzer::GeoOperator* geo_operator) : Codegen(geo_operator) {
    CHECK_EQ(operator_->size(), size_t(1));
    initColumnTransformOp();
  }

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kBOOLEAN); }

  const Analyzer::Expr* getOperand(const size_t index) final {
    CHECK_EQ(index, size_t(0));
    initColumnTransformOp();
    if (column_transform_op_) {
      return column_transform_op_->getOperand(0);
    }
    return operator_->getOperand(0);
  }

  llvm::Value* codegenCmpEqNullptr(llvm::IRBuilder<>& builder, llvm::Value* arg_lv) {
    auto* const ptr_type = llvm::dyn_cast<llvm::PointerType>(arg_lv->getType());
    CHECK(ptr_type);
    return builder.CreateICmpEQ(arg_lv, llvm::ConstantPointerNull::get(ptr_type));
  }

  // returns arguments lvs and null lv
  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    CHECK_EQ(pos_lvs.size(), size());
    initColumnTransformOp();
    const auto analyzer_operand = operator_->getOperand(0);
    CHECK(analyzer_operand);
    const auto& geo_ti = column_transform_op_
                             ? column_transform_op_->getOperand(0)->get_type_info()
                             : analyzer_operand->get_type_info();
    CHECK(geo_ti.is_geometry());
    auto& builder = cgen_state->ir_builder_;

    llvm::Value* array_buff_ptr{nullptr};
    llvm::Value* is_null{nullptr};
    if (arg_lvs.size() == 1) {
      if (dynamic_cast<const Analyzer::GeoExpr*>(analyzer_operand) &&
          !column_transform_op_) {
        is_null = codegenCmpEqNullptr(builder, arg_lvs.front());
        return std::make_tuple(arg_lvs, is_null);
      }
      // col byte stream, get the array buffer ptr and is null attributes and cache
      auto arr_load_lvs = CodeGenerator::codegenGeoArrayLoadAndNullcheck(
          arg_lvs.front(), pos_lvs.front(), geo_ti, cgen_state);
      array_buff_ptr = arr_load_lvs.buffer;
      is_null = arr_load_lvs.is_null;
    } else {
      // ptr and size
      CHECK_EQ(arg_lvs.size(), size_t(2));
      if (dynamic_cast<const Analyzer::GeoOperator*>(analyzer_operand)) {
        if (geo_ti.get_type() == kPOINT && !geo_ti.is_variable_size()) {
          char const* const fname = pointIsNullFunctionName(geo_ti);
          is_null = cgen_state->emitCall(fname, {arg_lvs.front()});
        } else {
          // The above branch tests for both nullptr and null sentinel, whereas this
          // branch only tests for nullptr. If not for this branch, the GeospatialTest
          // LLVMOptimization test fails due to non-removal of the
          // decompress_{x,y}_coord_geoint function call in the generated IR. Required for
          // coord projection / LLVMOptimization (see GeospatialTest).
          is_null = codegenCmpEqNullptr(builder, arg_lvs.front());
        }
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
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    CHECK_EQ(args.size(), size_t(1));
    const auto array_buff_ptr = args.front();

    initColumnTransformOp();
    const auto analyzer_operand = operator_->getOperand(0);
    CHECK(analyzer_operand);
    const auto& geo_ti = column_transform_op_
                             ? column_transform_op_->getOperand(0)->get_type_info()
                             : analyzer_operand->get_type_info();
    CHECK(geo_ti.is_geometry());

    const bool is_x = operator_->getName() == "ST_X";
    llvm::Value* coord_lv;
    if (column_transform_op_) {
      const auto zero_lv =
          llvm::ConstantFP::get(llvm::Type::getDoubleTy(cgen_state->context_), 0.0);
      coord_lv = Transform::codegenPointCoord(
          array_buff_ptr,
          geo_ti,
          is_x ? Transform::PointCoordAxis::X : Transform::PointCoordAxis::Y,
          static_cast<unsigned>(column_transform_op_->getInputSRID()),
          static_cast<unsigned>(column_transform_op_->getOutputSRID()),
          cgen_state,
          co,
          zero_lv);
    } else {
      coord_lv = Transform::loadPointCoord(
          array_buff_ptr,
          geo_ti,
          is_x ? Transform::PointCoordAxis::X : Transform::PointCoordAxis::Y,
          cgen_state);
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

 private:
  void initColumnTransformOp() {
    if (column_transform_op_initialized_) {
      return;
    }
    column_transform_op_initialized_ = true;
    const auto* transform_op =
        dynamic_cast<const Analyzer::GeoTransformOperator*>(operator_->getOperand(0));
    if (transform_op &&
        dynamic_cast<const Analyzer::ColumnVar*>(transform_op->getOperand(0))) {
      column_transform_op_ = transform_op;
    }
  }

  const Analyzer::GeoTransformOperator* column_transform_op_{nullptr};
  bool column_transform_op_initialized_{false};
};

}  // namespace spatial_type
