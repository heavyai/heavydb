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

// ST_Transform
class Transform : public Codegen {
 public:
  Transform(const Analyzer::GeoOperator* geo_operator,
            const Catalog_Namespace::Catalog* catalog)
      : Codegen(geo_operator, catalog)
      , transform_operator_(
            dynamic_cast<const Analyzer::GeoTransformOperator*>(geo_operator)) {
    CHECK_EQ(operator_->size(), size_t(1));  // geo input expr
    CHECK(transform_operator_);
    const auto& ti = geo_operator->get_type_info();
    if (ti.get_notnull()) {
      is_nullable_ = false;
    } else {
      is_nullable_ = true;
    }
  }

  size_t size() const override { return 1; }

  SQLTypeInfo getNullType() const override { return SQLTypeInfo(kBOOLEAN); }

  inline static bool isUtm(unsigned const srid) {
    return (32601 <= srid && srid <= 32660) || (32701 <= srid && srid <= 32760);
  }

  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) override {
    CHECK_EQ(pos_lvs.size(), size());
    const auto geo_operand = getOperand(0);
    const auto& operand_ti = geo_operand->get_type_info();
    CHECK(operand_ti.is_geometry() && operand_ti.get_type() == kPOINT);

    if (dynamic_cast<const Analyzer::ColumnVar*>(geo_operand)) {
      CHECK_EQ(arg_lvs.size(), size_t(1));  // col_byte_stream
      auto arr_load_lvs = CodeGenerator::codegenGeoArrayLoadAndNullcheck(
          arg_lvs.front(), pos_lvs.front(), operand_ti, cgen_state);
      return std::make_tuple(std::vector<llvm::Value*>{arr_load_lvs.buffer},
                             arr_load_lvs.is_null);
    } else if (dynamic_cast<const Analyzer::GeoConstant*>(geo_operand)) {
      CHECK_EQ(arg_lvs.size(), size_t(2));  // ptr, size

      // nulls not supported, and likely compressed, so require a new buffer for the
      // transformation
      CHECK(!is_nullable_);
      return std::make_tuple(std::vector<llvm::Value*>{arg_lvs.front()}, nullptr);
    } else {
      CHECK(arg_lvs.size() == size_t(1) ||
            arg_lvs.size() == size_t(2));  // ptr or ptr, size
      // coming from a temporary, can modify the memory pointer directly
      can_transform_in_place_ = true;
      auto& builder = cgen_state->ir_builder_;

      const auto is_null = builder.CreateICmp(
          llvm::CmpInst::ICMP_EQ,
          arg_lvs.front(),
          llvm::ConstantPointerNull::get(  // TODO: check ptr address space
              operand_ti.get_compression() == kENCODING_GEOINT
                  ? llvm::Type::getInt32PtrTy(cgen_state->context_)
                  : llvm::Type::getDoublePtrTy(cgen_state->context_)));
      return std::make_tuple(std::vector<llvm::Value*>{arg_lvs.front()}, is_null);
    }
    UNREACHABLE();
    return std::make_tuple(std::vector<llvm::Value*>{}, nullptr);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) override {
    CHECK_EQ(args.size(), size_t(1));

    const auto geo_operand = getOperand(0);
    const auto& operand_ti = geo_operand->get_type_info();
    auto& builder = cgen_state->ir_builder_;

    llvm::Value* arr_buff_ptr = args.front();
    if (operand_ti.get_compression() == kENCODING_GEOINT) {
      // decompress
      auto new_arr_ptr =
          builder.CreateAlloca(llvm::Type::getDoubleTy(cgen_state->context_),
                               cgen_state->llInt(int32_t(2)),
                               getName() + "_Array");
      auto compressed_arr_ptr = builder.CreateBitCast(
          arr_buff_ptr, llvm::Type::getInt32PtrTy(cgen_state->context_));
      // x coord
      auto* gep = builder.CreateGEP(
          compressed_arr_ptr->getType()->getScalarType()->getPointerElementType(),
          compressed_arr_ptr,
          cgen_state->llInt(0));
      auto x_coord_lv = cgen_state->emitExternalCall(
          "decompress_x_coord_geoint",
          llvm::Type::getDoubleTy(cgen_state->context_),
          {builder.CreateLoad(
              gep->getType()->getPointerElementType(), gep, "compressed_x_coord")});
      builder.CreateStore(
          x_coord_lv,
          builder.CreateGEP(
              new_arr_ptr->getType()->getScalarType()->getPointerElementType(),
              new_arr_ptr,
              cgen_state->llInt(0)));
      gep = builder.CreateGEP(
          compressed_arr_ptr->getType()->getScalarType()->getPointerElementType(),
          compressed_arr_ptr,
          cgen_state->llInt(1));
      auto y_coord_lv = cgen_state->emitExternalCall(
          "decompress_y_coord_geoint",
          llvm::Type::getDoubleTy(cgen_state->context_),
          {builder.CreateLoad(
              gep->getType()->getPointerElementType(), gep, "compressed_y_coord")});
      builder.CreateStore(
          y_coord_lv,
          builder.CreateGEP(
              new_arr_ptr->getType()->getScalarType()->getPointerElementType(),
              new_arr_ptr,
              cgen_state->llInt(1)));
      arr_buff_ptr = new_arr_ptr;
    } else if (!can_transform_in_place_) {
      auto new_arr_ptr =
          builder.CreateAlloca(llvm::Type::getDoubleTy(cgen_state->context_),
                               cgen_state->llInt(int32_t(2)),
                               getName() + "_Array");
      const auto arr_buff_ptr_cast = builder.CreateBitCast(
          arr_buff_ptr, llvm::Type::getDoublePtrTy(cgen_state->context_));

      auto* gep = builder.CreateGEP(
          arr_buff_ptr_cast->getType()->getScalarType()->getPointerElementType(),
          arr_buff_ptr_cast,
          cgen_state->llInt(0));
      builder.CreateStore(
          builder.CreateLoad(gep->getType()->getPointerElementType(), gep),
          builder.CreateGEP(
              new_arr_ptr->getType()->getScalarType()->getPointerElementType(),
              new_arr_ptr,
              cgen_state->llInt(0)));
      gep = builder.CreateGEP(
          arr_buff_ptr_cast->getType()->getScalarType()->getPointerElementType(),
          arr_buff_ptr_cast,
          cgen_state->llInt(1));
      builder.CreateStore(
          builder.CreateLoad(gep->getType()->getPointerElementType(), gep),
          builder.CreateGEP(
              new_arr_ptr->getType()->getScalarType()->getPointerElementType(),
              new_arr_ptr,
              cgen_state->llInt(1)));
      arr_buff_ptr = new_arr_ptr;
    }
    CHECK(arr_buff_ptr->getType() == llvm::Type::getDoublePtrTy(cgen_state->context_));

    auto const srid_in = static_cast<unsigned>(transform_operator_->getInputSRID());
    auto const srid_out = static_cast<unsigned>(transform_operator_->getOutputSRID());
    if (srid_in == srid_out) {
      // noop
      return {args.front()};
    }

    // transform in place
    std::string transform_function_prefix{""};
    std::vector<llvm::Value*> transform_args;

    if (srid_out == 900913) {
      if (srid_in == 4326) {
        transform_function_prefix = "transform_4326_900913_";
      } else if (isUtm(srid_in)) {
        transform_function_prefix = "transform_utm_900913_";
        transform_args.push_back(cgen_state->llInt(srid_in));
      } else {
        throw std::runtime_error("Unsupported input SRID " + std::to_string(srid_in) +
                                 " for output SRID " + std::to_string(srid_out));
      }
    } else if (srid_out == 4326) {
      if (srid_in == 900913) {
        transform_function_prefix = "transform_900913_4326_";
      } else if (isUtm(srid_in)) {
        transform_function_prefix = "transform_utm_4326_";
        transform_args.push_back(cgen_state->llInt(srid_in));
      } else {
        throw std::runtime_error("Unsupported input SRID " + std::to_string(srid_in) +
                                 " for output SRID " + std::to_string(srid_out));
      }
    } else if (isUtm(srid_out)) {
      if (srid_in == 4326) {
        transform_function_prefix = "transform_4326_utm_";
      } else if (srid_in == 900913) {
        transform_function_prefix = "transform_900913_utm_";
      } else {
        throw std::runtime_error("Unsupported input SRID " + std::to_string(srid_in) +
                                 " for output SRID " + std::to_string(srid_out));
      }
      transform_args.push_back(cgen_state->llInt(srid_out));
    } else {
      throw std::runtime_error("Unsupported output SRID for ST_Transform: " +
                               std::to_string(srid_out));
    }
    CHECK(!transform_function_prefix.empty());

    auto x_coord_ptr_lv = builder.CreateGEP(
        arr_buff_ptr->getType()->getScalarType()->getPointerElementType(),
        arr_buff_ptr,
        cgen_state->llInt(0),
        "x_coord_ptr");
    transform_args.push_back(builder.CreateLoad(
        x_coord_ptr_lv->getType()->getPointerElementType(), x_coord_ptr_lv, "x_coord"));
    auto y_coord_ptr_lv = builder.CreateGEP(
        arr_buff_ptr->getType()->getScalarType()->getPointerElementType(),
        arr_buff_ptr,
        cgen_state->llInt(1),
        "y_coord_ptr");
    transform_args.push_back(builder.CreateLoad(
        y_coord_ptr_lv->getType()->getPointerElementType(), y_coord_ptr_lv, "y_coord"));
    if (co.device_type == ExecutorDeviceType::GPU) {
      auto fn_x = cgen_state->module_->getFunction(transform_function_prefix + 'x');
      CHECK(fn_x);
      cgen_state->maybeCloneFunctionRecursive(fn_x);
      CHECK(!fn_x->isDeclaration());

      auto gpu_functions_to_replace = cgen_state->gpuFunctionsToReplace(fn_x);
      for (const auto& fcn_name : gpu_functions_to_replace) {
        cgen_state->replaceFunctionForGpu(fcn_name, fn_x);
      }
      verify_function_ir(fn_x);
      auto transform_call = builder.CreateCall(fn_x, transform_args);
      builder.CreateStore(transform_call, x_coord_ptr_lv);

      auto fn_y = cgen_state->module_->getFunction(transform_function_prefix + 'y');
      CHECK(fn_y);
      cgen_state->maybeCloneFunctionRecursive(fn_y);
      CHECK(!fn_y->isDeclaration());

      gpu_functions_to_replace = cgen_state->gpuFunctionsToReplace(fn_y);
      for (const auto& fcn_name : gpu_functions_to_replace) {
        cgen_state->replaceFunctionForGpu(fcn_name, fn_y);
      }
      verify_function_ir(fn_y);
      transform_call = builder.CreateCall(fn_y, transform_args);
      builder.CreateStore(transform_call, y_coord_ptr_lv);
    } else {
      builder.CreateStore(
          cgen_state->emitCall(transform_function_prefix + 'x', transform_args),
          x_coord_ptr_lv);
      builder.CreateStore(
          cgen_state->emitCall(transform_function_prefix + 'y', transform_args),
          y_coord_ptr_lv);
    }
    auto ret = arr_buff_ptr;
    const auto& geo_ti = transform_operator_->get_type_info();

    if (is_nullable_) {
      CHECK(nullcheck_codegen);
      ret = nullcheck_codegen->finalize(
          llvm::ConstantPointerNull::get(
              geo_ti.get_compression() == kENCODING_GEOINT
                  ? llvm::PointerType::get(llvm::Type::getInt32Ty(cgen_state->context_),
                                           0)
                  : llvm::PointerType::get(llvm::Type::getDoubleTy(cgen_state->context_),
                                           0)),
          ret);
    }
    return {ret,
            cgen_state->llInt(static_cast<int32_t>(
                geo_ti.get_compression() == kENCODING_GEOINT ? 8 : 16))};
  }

 private:
  const Analyzer::GeoTransformOperator* transform_operator_;
  bool can_transform_in_place_{false};
};

}  // namespace spatial_type
