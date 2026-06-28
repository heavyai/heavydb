/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryEngine/GeoOperators/Codegen.h"
namespace spatial_type {

// ST_Transform
class Transform : public Codegen {
 public:
  Transform(const Analyzer::GeoOperator* geo_operator)
      : Codegen(geo_operator)
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

  static std::string transformFunctionPrefix(const unsigned srid_in,
                                             const unsigned srid_out,
                                             std::vector<llvm::Value*>& prefix_args,
                                             CgenState* cgen_state) {
    prefix_args.clear();
    if (srid_out == 900913) {
      if (srid_in == 4326) {
        return "transform_4326_900913_";
      }
      if (isUtm(srid_in)) {
        prefix_args.push_back(cgen_state->llInt(static_cast<int32_t>(srid_in)));
        return "transform_utm_900913_";
      }
    } else if (srid_out == 4326) {
      if (srid_in == 900913) {
        return "transform_900913_4326_";
      }
      if (isUtm(srid_in)) {
        prefix_args.push_back(cgen_state->llInt(static_cast<int32_t>(srid_in)));
        return "transform_utm_4326_";
      }
    } else if (isUtm(srid_out)) {
      if (srid_in == 4326) {
        prefix_args.push_back(cgen_state->llInt(static_cast<int32_t>(srid_out)));
        return "transform_4326_utm_";
      }
      if (srid_in == 900913) {
        prefix_args.push_back(cgen_state->llInt(static_cast<int32_t>(srid_out)));
        return "transform_900913_utm_";
      }
    }
    throw std::runtime_error("Unsupported geo transformation from " +
                             std::to_string(srid_in) + " to " + std::to_string(srid_out));
  }

  static llvm::Value* emitTransformAxisCall(
      const std::string& transform_function_prefix,
      const char axis,
      const std::vector<llvm::Value*>& transform_args,
      CgenState* cgen_state,
      const CompilationOptions& co) {
    auto& builder = cgen_state->ir_builder_;
    const std::string fn_name = transform_function_prefix + axis;
    if (co.device_type == ExecutorDeviceType::GPU) {
      auto fn = cgen_state->module_->getFunction(fn_name);
      CHECK(fn);
      cgen_state->maybeCloneFunctionRecursive(fn);
      CHECK(!fn->isDeclaration());
      auto gpu_functions_to_replace = cgen_state->gpuFunctionsToReplace(fn);
      for (const auto& fcn_name : gpu_functions_to_replace) {
        cgen_state->replaceFunctionForGpu(fcn_name, fn);
      }
      verify_function_ir(fn);
      return builder.CreateCall(fn, transform_args);
    }
    return cgen_state->emitCall(fn_name, transform_args);
  }

  enum class PointCoordAxis { X, Y };

  static llvm::Value* loadPointCoord(llvm::Value* array_buff_ptr,
                                     const SQLTypeInfo& geo_ti,
                                     const PointCoordAxis axis,
                                     CgenState* cgen_state,
                                     const bool decompressed_buffer = false) {
    auto& builder = cgen_state->ir_builder_;
    const auto is_x = axis == PointCoordAxis::X;
    const std::string expr_name = is_x ? "x" : "y";
    const auto coord_index = is_x ? cgen_state->llInt(0) : cgen_state->llInt(1);

    if (!decompressed_buffer && geo_ti.get_compression() == kENCODING_GEOINT) {
      const auto compressed_arr_ptr = builder.CreateBitCast(
          array_buff_ptr, get_int_ptr_type(32, cgen_state->context_));
      auto coord_lv_ptr = typed_gep(builder,
                                    get_int_type(32, cgen_state->context_),
                                    compressed_arr_ptr,
                                    coord_index);
      coord_lv_ptr->setName(expr_name + "_coord_ptr");
      const auto compressed_coord_lv = typed_load(builder,
                                                  get_int_type(32, cgen_state->context_),
                                                  coord_lv_ptr,
                                                  expr_name + "_coord_compressed");
      return cgen_state->emitExternalCall("decompress_" + expr_name + "_coord_geoint",
                                          llvm::Type::getDoubleTy(cgen_state->context_),
                                          std::vector<llvm::Value*>{compressed_coord_lv});
    }

    auto coord_arr_ptr =
        builder.CreateBitCast(array_buff_ptr, get_fp_ptr_type(64, cgen_state->context_));
    auto coord_lv_ptr = typed_gep(builder,
                                  llvm::Type::getDoubleTy(cgen_state->context_),
                                  coord_arr_ptr,
                                  coord_index);
    coord_lv_ptr->setName(expr_name + "_coord_ptr");
    return typed_load(builder,
                      llvm::Type::getDoubleTy(cgen_state->context_),
                      coord_lv_ptr,
                      expr_name + "_coord");
  }

  // Project a single transformed coordinate. companion_coord overrides the other axis
  // for the transform call (use zero for ST_X/ST_Y over ST_Transform(column)); when
  // null, load the companion from array_buff_ptr (full ST_Transform on decompressed
  // doubles).
  static llvm::Value* codegenPointCoord(llvm::Value* array_buff_ptr,
                                        const SQLTypeInfo& geo_ti,
                                        const PointCoordAxis axis,
                                        const unsigned srid_in,
                                        const unsigned srid_out,
                                        CgenState* cgen_state,
                                        const CompilationOptions& co,
                                        llvm::Value* companion_coord = nullptr,
                                        const bool decompressed_buffer = false) {
    const auto coord_lv =
        loadPointCoord(array_buff_ptr, geo_ti, axis, cgen_state, decompressed_buffer);
    if (srid_in == srid_out) {
      return coord_lv;
    }

    llvm::Value* other_coord_lv = companion_coord;
    if (!other_coord_lv) {
      const auto other_axis =
          axis == PointCoordAxis::X ? PointCoordAxis::Y : PointCoordAxis::X;
      other_coord_lv = loadPointCoord(
          array_buff_ptr, geo_ti, other_axis, cgen_state, decompressed_buffer);
    }

    std::vector<llvm::Value*> prefix_args;
    const auto transform_function_prefix =
        transformFunctionPrefix(srid_in, srid_out, prefix_args, cgen_state);
    std::vector<llvm::Value*> transform_args = prefix_args;
    if (axis == PointCoordAxis::X) {
      transform_args.push_back(coord_lv);
      transform_args.push_back(other_coord_lv);
      return emitTransformAxisCall(
          transform_function_prefix, 'x', transform_args, cgen_state, co);
    }
    transform_args.push_back(other_coord_lv);
    transform_args.push_back(coord_lv);
    return emitTransformAxisCall(
        transform_function_prefix, 'y', transform_args, cgen_state, co);
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
      char const* const fname = pointIsNullFunctionName(operand_ti);
      llvm::Value* const is_null = cgen_state->emitCall(fname, {arg_lvs.front()});
      return std::make_tuple(std::vector<llvm::Value*>{arg_lvs.front()}, is_null);
    }
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
      auto compressed_arr_ptr =
          builder.CreateBitCast(arr_buff_ptr, get_int_ptr_type(32, cgen_state->context_));
      // x coord
      auto* gep = typed_gep(builder,
                            get_int_type(32, cgen_state->context_),
                            compressed_arr_ptr,
                            cgen_state->llInt(0));
      auto x_coord_lv =
          cgen_state->emitExternalCall("decompress_x_coord_geoint",
                                       llvm::Type::getDoubleTy(cgen_state->context_),
                                       {typed_load(builder,
                                                   get_int_type(32, cgen_state->context_),
                                                   gep,
                                                   "compressed_x_coord")});
      builder.CreateStore(
          x_coord_lv,
          typed_alloca_element_ptr(builder,
                                   llvm::Type::getDoubleTy(cgen_state->context_),
                                   new_arr_ptr,
                                   cgen_state->llInt(0)));
      gep = typed_gep(builder,
                      get_int_type(32, cgen_state->context_),
                      compressed_arr_ptr,
                      cgen_state->llInt(1));
      auto y_coord_lv =
          cgen_state->emitExternalCall("decompress_y_coord_geoint",
                                       llvm::Type::getDoubleTy(cgen_state->context_),
                                       {typed_load(builder,
                                                   get_int_type(32, cgen_state->context_),
                                                   gep,
                                                   "compressed_y_coord")});
      builder.CreateStore(
          y_coord_lv,
          typed_alloca_element_ptr(builder,
                                   llvm::Type::getDoubleTy(cgen_state->context_),
                                   new_arr_ptr,
                                   cgen_state->llInt(1)));
      arr_buff_ptr = new_arr_ptr;
    } else if (!can_transform_in_place_) {
      auto new_arr_ptr =
          builder.CreateAlloca(llvm::Type::getDoubleTy(cgen_state->context_),
                               cgen_state->llInt(int32_t(2)),
                               getName() + "_Array");
      const auto arr_buff_ptr_cast =
          builder.CreateBitCast(arr_buff_ptr, get_fp_ptr_type(64, cgen_state->context_));

      auto* gep = typed_gep(builder,
                            llvm::Type::getDoubleTy(cgen_state->context_),
                            arr_buff_ptr_cast,
                            cgen_state->llInt(0));
      builder.CreateStore(
          typed_load(builder, llvm::Type::getDoubleTy(cgen_state->context_), gep),
          typed_alloca_element_ptr(builder,
                                   llvm::Type::getDoubleTy(cgen_state->context_),
                                   new_arr_ptr,
                                   cgen_state->llInt(0)));
      gep = typed_gep(builder,
                      llvm::Type::getDoubleTy(cgen_state->context_),
                      arr_buff_ptr_cast,
                      cgen_state->llInt(1));
      builder.CreateStore(
          typed_load(builder, llvm::Type::getDoubleTy(cgen_state->context_), gep),
          typed_alloca_element_ptr(builder,
                                   llvm::Type::getDoubleTy(cgen_state->context_),
                                   new_arr_ptr,
                                   cgen_state->llInt(1)));
      arr_buff_ptr = new_arr_ptr;
    }

    auto const srid_in = static_cast<unsigned>(transform_operator_->getInputSRID());
    auto const srid_out = static_cast<unsigned>(transform_operator_->getOutputSRID());
    if (srid_in == srid_out) {
      // noop
      return {args.front()};
    }

    // transform in place
    arr_buff_ptr =
        builder.CreateBitCast(arr_buff_ptr, get_fp_ptr_type(64, cgen_state->context_));
    auto x_coord_ptr_lv = typed_gep(builder,
                                    llvm::Type::getDoubleTy(cgen_state->context_),
                                    arr_buff_ptr,
                                    cgen_state->llInt(0));
    x_coord_ptr_lv->setName("x_coord_ptr");
    auto y_coord_ptr_lv = typed_gep(builder,
                                    llvm::Type::getDoubleTy(cgen_state->context_),
                                    arr_buff_ptr,
                                    cgen_state->llInt(1));
    y_coord_ptr_lv->setName("y_coord_ptr");

    auto decompressed_ti = operand_ti;
    decompressed_ti.set_compression(kENCODING_NONE);
    // Load source coords before any store so the Y transform still sees WGS x.
    const auto orig_x = loadPointCoord(
        arr_buff_ptr, decompressed_ti, PointCoordAxis::X, cgen_state, true);
    const auto orig_y = loadPointCoord(
        arr_buff_ptr, decompressed_ti, PointCoordAxis::Y, cgen_state, true);
    builder.CreateStore(codegenPointCoord(arr_buff_ptr,
                                          decompressed_ti,
                                          PointCoordAxis::X,
                                          srid_in,
                                          srid_out,
                                          cgen_state,
                                          co,
                                          orig_y,
                                          true),
                        x_coord_ptr_lv);
    builder.CreateStore(codegenPointCoord(arr_buff_ptr,
                                          decompressed_ti,
                                          PointCoordAxis::Y,
                                          srid_in,
                                          srid_out,
                                          cgen_state,
                                          co,
                                          orig_x,
                                          true),
                        y_coord_ptr_lv);
    auto ret = arr_buff_ptr;
    const auto& geo_ti = transform_operator_->get_type_info();

    if (is_nullable_) {
      CHECK(nullcheck_codegen);
      ret = nullcheck_codegen->finalize(
          llvm::ConstantPointerNull::get(geo_ti.get_compression() == kENCODING_GEOINT
                                             ? get_int_ptr_type(32, cgen_state->context_)
                                             : get_fp_ptr_type(64, cgen_state->context_)),
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
