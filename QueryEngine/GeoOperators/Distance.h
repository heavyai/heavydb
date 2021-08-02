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

class Distance : public Codegen {
 public:
  Distance(const Analyzer::GeoOperator* geo_operator,
           const Catalog_Namespace::Catalog* catalog)
      : Codegen(geo_operator, catalog) {
    CHECK_EQ(operator_->size(), size_t(2));
    const auto& ti = operator_->get_type_info();
    is_nullable_ = !ti.get_notnull();
  }

  size_t size() const final { return 2; }

  SQLTypeInfo getNullType() const final { return SQLTypeInfo(kBOOLEAN); }

  std::tuple<std::vector<llvm::Value*>, llvm::Value*> codegenLoads(
      const std::vector<llvm::Value*>& arg_lvs,
      const std::vector<llvm::Value*>& pos_lvs,
      CgenState* cgen_state) final {
    CHECK_EQ(pos_lvs.size(), size());
    std::string size_fn_name = "array_size";
    if (is_nullable_) {
      size_fn_name += "_nullable";
    }

    auto& builder = cgen_state->ir_builder_;
    llvm::Value* is_null = cgen_state->llBool(false);

    std::vector<llvm::Value*> operand_lvs;
    size_t arg_lvs_index{0};
    for (size_t i = 0; i < size(); i++) {
      const auto operand = getOperand(i);
      CHECK(operand);
      const auto& operand_ti = operand->get_type_info();
      CHECK(IS_GEO(operand_ti.get_type()));
      const size_t num_physical_coord_lvs = operand_ti.get_physical_coord_cols();

      // iterate over column inputs
      bool is_coords_lv{true};
      if (dynamic_cast<const Analyzer::ColumnVar*>(operand)) {
        for (size_t j = 0; j < num_physical_coord_lvs; j++) {
          CHECK_LT(arg_lvs_index, arg_lvs.size());
          auto lv = arg_lvs[arg_lvs_index++];
          // TODO: fast fixlen array buff for coords
          auto array_buff_lv =
              cgen_state->emitExternalCall("array_buff",
                                           llvm::Type::getInt8PtrTy(cgen_state->context_),
                                           {lv, pos_lvs[i]});
          if (j > 0) {
            // cast additional columns to i32*
            array_buff_lv = builder.CreateBitCast(
                array_buff_lv, llvm::Type::getInt32PtrTy(cgen_state->context_));
          }
          operand_lvs.push_back(array_buff_lv);

          const auto ptr_type = llvm::dyn_cast_or_null<llvm::PointerType>(lv->getType());
          CHECK(ptr_type);
          const auto elem_type = ptr_type->getElementType();
          CHECK(elem_type);
          const uint32_t coords_elem_sz_bytes =
              operand_ti.get_compression() == kENCODING_NONE &&
                      operand_ti.get_type() == kPOINT
                  ? 8
                  : 1;
          std::vector<llvm::Value*> array_sz_args{
              lv,
              pos_lvs[i],
              cgen_state->llInt(log2_bytes(j == 0 ? coords_elem_sz_bytes : 4))};
          if (is_nullable_) {  // TODO: should we do this for all arguments, or just
                               // coords?
            array_sz_args.push_back(cgen_state->llInt(
                static_cast<int32_t>(inline_int_null_value<int32_t>())));
          }
          operand_lvs.push_back(cgen_state->emitExternalCall(
              size_fn_name, get_int_type(32, cgen_state->context_), array_sz_args));
          llvm::Value* operand_is_null_lv{nullptr};
          if (is_nullable_ && is_coords_lv) {
            if (operand_ti.get_type() == kPOINT) {
              operand_is_null_lv = cgen_state->emitExternalCall(
                  "point_coord_array_is_null",
                  llvm::Type::getInt1Ty(cgen_state->context_),
                  {lv, pos_lvs[i]});
            } else {
              operand_is_null_lv = builder.CreateICmpEQ(
                  operand_lvs.back(),
                  cgen_state->llInt(
                      static_cast<int32_t>(inline_int_null_value<int32_t>())));
            }
            is_null = builder.CreateOr(is_null, operand_is_null_lv);
          }
          is_coords_lv = false;
        }
      } else {
        bool is_coords_lv{true};
        for (size_t j = 0; j < num_physical_coord_lvs; j++) {
          // ptr
          CHECK_LT(arg_lvs_index, arg_lvs.size());
          auto array_buff_lv = arg_lvs[arg_lvs_index++];
          if (j == 0) {
            // cast alloca to i8*
            array_buff_lv = builder.CreateBitCast(
                array_buff_lv, llvm::Type::getInt8PtrTy(cgen_state->context_));
          } else {
            // cast additional columns to i32*
            array_buff_lv = builder.CreateBitCast(
                array_buff_lv, llvm::Type::getInt32PtrTy(cgen_state->context_));
          }
          operand_lvs.push_back(array_buff_lv);
          if (is_nullable_ && is_coords_lv) {
            auto coords_array_type =
                llvm::dyn_cast<llvm::PointerType>(operand_lvs.back()->getType());
            CHECK(coords_array_type);
            is_null = builder.CreateOr(
                is_null,
                builder.CreateICmpEQ(operand_lvs.back(),
                                     llvm::ConstantPointerNull::get(coords_array_type)));
          }
          is_coords_lv = false;
          CHECK_LT(arg_lvs_index, arg_lvs.size());
          operand_lvs.push_back(arg_lvs[arg_lvs_index++]);
        }
      }
    }
    CHECK_EQ(arg_lvs_index, arg_lvs.size());

    // use the points array size argument for nullability
    return std::make_tuple(operand_lvs, is_nullable_ ? is_null : nullptr);
  }

  std::vector<llvm::Value*> codegen(const std::vector<llvm::Value*>& args,
                                    CodeGenerator::NullCheckCodegen* nullcheck_codegen,
                                    CgenState* cgen_state,
                                    const CompilationOptions& co) final {
    const auto& first_operand_ti = getOperand(0)->get_type_info();
    const auto& second_operand_ti = getOperand(1)->get_type_info();

    const bool is_geodesic = first_operand_ti.get_subtype() == kGEOGRAPHY &&
                             first_operand_ti.get_output_srid() == 4326;

    if (is_geodesic && !((first_operand_ti.get_type() == kPOINT &&
                          second_operand_ti.get_type() == kPOINT) ||
                         (first_operand_ti.get_type() == kLINESTRING &&
                          second_operand_ti.get_type() == kPOINT) ||
                         (first_operand_ti.get_type() == kPOINT &&
                          second_operand_ti.get_type() == kLINESTRING))) {
      throw std::runtime_error(getName() +
                               " currently doesn't accept non-POINT geographies");
    }

    std::string func_name = getName() + suffix(first_operand_ti.get_type()) +
                            suffix(second_operand_ti.get_type());
    if (is_geodesic) {
      func_name += "_Geodesic";
    }
    auto& builder = cgen_state->ir_builder_;

    std::vector<llvm::Value*> operand_lvs;
    for (size_t i = 0; i < args.size(); i += 2) {
      operand_lvs.push_back(args[i]);
      operand_lvs.push_back(
          builder.CreateSExt(args[i + 1], llvm::Type::getInt64Ty(cgen_state->context_)));
    }

    const auto& ret_ti = operator_->get_type_info();
    // push back ic, isr, osr for now
    operand_lvs.push_back(
        cgen_state->llInt(Geospatial::get_compression_scheme(first_operand_ti)));  // ic 1
    operand_lvs.push_back(
        cgen_state->llInt(first_operand_ti.get_input_srid()));  // in srid 1
    operand_lvs.push_back(cgen_state->llInt(
        Geospatial::get_compression_scheme(second_operand_ti)));  // ic 2
    operand_lvs.push_back(
        cgen_state->llInt(second_operand_ti.get_input_srid()));          // in srid 2
    operand_lvs.push_back(cgen_state->llInt(ret_ti.get_output_srid()));  // out srid

    if (getName() == "ST_Distance" && first_operand_ti.get_subtype() != kGEOGRAPHY &&
        (first_operand_ti.get_type() != kPOINT ||
         second_operand_ti.get_type() != kPOINT)) {
      operand_lvs.push_back(cgen_state->llFp(double(0.0)));
    }

    CHECK(ret_ti.get_type() == kDOUBLE);
    auto ret = cgen_state->emitExternalCall(
        func_name, llvm::Type::getDoubleTy(cgen_state->context_), operand_lvs);
    if (is_nullable_) {
      CHECK(nullcheck_codegen);
      ret = nullcheck_codegen->finalize(cgen_state->inlineFpNull(ret_ti), ret);
    }
    return {ret};
  }
};

}  // namespace spatial_type
