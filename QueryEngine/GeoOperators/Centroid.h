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
  Centroid(const Analyzer::GeoOperator* geo_operator,
           const Catalog_Namespace::Catalog* catalog)
      : Codegen(geo_operator, catalog) {
    CHECK_EQ(operator_->size(), size_t(1));
    const auto& ti = operator_->get_type_info();
    is_nullable_ = !ti.get_notnull();
  }

  size_t size() const final { return 1; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kINT); }

  const Analyzer::Expr* getPositionOperand() const final {
    return operator_->getOperand(0);
  }

  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      llvm::Value* pos_lv,
      CgenState* cgen_state) final {
    const auto operand = getOperand(0);
    CHECK(operand);
    const auto& operand_ti = operand->get_type_info();

    std::string size_fn_name = "array_size";
    if (is_nullable_) {
      size_fn_name += "_nullable";
    }

    const uint32_t coords_elem_sz_bytes =
        operand_ti.get_compression() == kENCODING_GEOINT ? 1 : 8;

    std::vector<llvm::Value*> operand_lvs;
    // iterate over column inputs
    if (dynamic_cast<const Analyzer::ColumnVar*>(operand)) {
      for (size_t i = 0; i < arg_lvs.size(); i++) {
        auto lv = arg_lvs[i];
        operand_lvs.push_back(cgen_state->emitExternalCall(
            "array_buff", llvm::Type::getInt8PtrTy(cgen_state->context_), {lv, pos_lv}));
        const auto ptr_type = llvm::dyn_cast_or_null<llvm::PointerType>(lv->getType());
        CHECK(ptr_type);
        const auto elem_type = ptr_type->getElementType();
        CHECK(elem_type);
        std::vector<llvm::Value*> array_sz_args{
            lv, pos_lv, cgen_state->llInt(log2_bytes(i == 0 ? coords_elem_sz_bytes : 4))};
        if (is_nullable_) {  // TODO: should we do this for all arguments, or just points?
          array_sz_args.push_back(
              cgen_state->llInt(static_cast<int32_t>(inline_int_null_value<int32_t>())));
        }
        operand_lvs.push_back(cgen_state->emitExternalCall(
            size_fn_name, get_int_type(32, cgen_state->context_), array_sz_args));
      }
    } else {
      operand_lvs = arg_lvs;
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
        auto& builder = cgen_state->ir_builder_;

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
    std::string func_name = "ST_Centroid";
    const auto& ret_ti = operator_->get_type_info();
    CHECK(ret_ti.is_geometry() && ret_ti.get_type() == kPOINT);
    const auto& operand_ti = getOperand(0)->get_type_info();

    auto& builder = cgen_state->ir_builder_;

    // Allocate local storage for centroid point
    auto elem_ty = llvm::Type::getDoubleTy(cgen_state->context_);
    llvm::ArrayType* arr_type = llvm::ArrayType::get(elem_ty, 2);
    auto pt_local_storage_lv =
        builder.CreateAlloca(arr_type, nullptr, func_name + "_Local_Storage");

    llvm::Value* pt_compressed_local_storage_lv{NULL};
    // Allocate local storage for compressed centroid point
    if (ret_ti.get_compression() == kENCODING_GEOINT) {
      auto elem_ty = llvm::Type::getInt32Ty(cgen_state->context_);
      llvm::ArrayType* arr_type = llvm::ArrayType::get(elem_ty, 2);
      pt_compressed_local_storage_lv = builder.CreateAlloca(
          arr_type, nullptr, func_name + "_Compressed_Local_Storage");
    }

    func_name += spatial_type::suffix(operand_ti.get_type());

    auto operand_lvs = args;

    // push back ic, isr, osr for now
    operand_lvs.push_back(
        cgen_state->llInt(Geospatial::get_compression_scheme(operand_ti)));  // ic
    operand_lvs.push_back(cgen_state->llInt(operand_ti.get_input_srid()));   // in srid
    operand_lvs.push_back(cgen_state->llInt(operand_ti.get_output_srid()));  // out srid

    auto idx_lv = cgen_state->llInt(0);
    auto pt_local_storage_gep = llvm::GetElementPtrInst::CreateInBounds(
        pt_local_storage_lv, {idx_lv, idx_lv}, "", builder.GetInsertBlock());
    // Pass local storage to centroid function
    operand_lvs.push_back(pt_local_storage_gep);
    cgen_state->emitExternalCall(func_name,
                                 ret_ti.get_type() == kDOUBLE
                                     ? llvm::Type::getDoubleTy(cgen_state->context_)
                                     : llvm::Type::getFloatTy(cgen_state->context_),
                                 operand_lvs);

    llvm::Value* ret_coords = pt_local_storage_lv;
    if (ret_ti.get_compression() == kENCODING_GEOINT) {
      // Compress centroid point if requested
      // Take values out of local storage, compress, store in compressed local storage

      auto x_ptr = builder.CreateGEP(
          pt_local_storage_lv, {cgen_state->llInt(0), cgen_state->llInt(0)}, "x_ptr");
      auto x_lv = builder.CreateLoad(x_ptr);
      auto compressed_x_lv =
          cgen_state->emitExternalCall("compress_x_coord_geoint",
                                       llvm::Type::getInt32Ty(cgen_state->context_),
                                       {x_lv});
      auto compressed_x_ptr =
          builder.CreateGEP(pt_compressed_local_storage_lv,
                            {cgen_state->llInt(0), cgen_state->llInt(0)},
                            "compressed_x_ptr");
      builder.CreateStore(compressed_x_lv, compressed_x_ptr);

      auto y_ptr = builder.CreateGEP(
          pt_local_storage_lv, {cgen_state->llInt(0), cgen_state->llInt(1)}, "y_ptr");
      auto y_lv = builder.CreateLoad(y_ptr);
      auto compressed_y_lv =
          cgen_state->emitExternalCall("compress_y_coord_geoint",
                                       llvm::Type::getInt32Ty(cgen_state->context_),
                                       {y_lv});
      auto compressed_y_ptr =
          builder.CreateGEP(pt_compressed_local_storage_lv,
                            {cgen_state->llInt(0), cgen_state->llInt(1)},
                            "compressed_y_ptr");
      builder.CreateStore(compressed_y_lv, compressed_y_ptr);

      ret_coords = pt_compressed_local_storage_lv;
    } else {
      CHECK(ret_ti.get_compression() == kENCODING_NONE);
    }

    auto ret_ty = ret_ti.get_compression() == kENCODING_GEOINT
                      ? llvm::Type::getInt32PtrTy(cgen_state->context_)
                      : llvm::Type::getDoublePtrTy(cgen_state->context_);
    ret_coords = builder.CreateBitCast(ret_coords, ret_ty);

    if (is_nullable_) {
      CHECK(nullcheck_codegen);
      ret_coords = nullcheck_codegen->finalize(
          llvm::ConstantPointerNull::get(
              ret_ti.get_compression() == kENCODING_GEOINT
                  ? llvm::PointerType::get(llvm::Type::getInt32Ty(cgen_state->context_),
                                           0)
                  : llvm::PointerType::get(llvm::Type::getDoubleTy(cgen_state->context_),
                                           0)),
          ret_coords);
    }

    return {ret_coords,
            cgen_state->llInt(ret_ti.get_compression() == kENCODING_GEOINT ? 8 : 16)};
  }
};

}  // namespace spatial_type
