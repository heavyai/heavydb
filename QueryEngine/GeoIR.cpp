/*
 * Copyright 2019 OmniSci, Inc.
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

#include "CodeGenerator.h"
#include "Execute.h"
#include "Shared/geo_compression.h"

std::vector<llvm::Value*> CodeGenerator::codegenGeoUOper(
    const Analyzer::GeoUOper* geo_expr,
    const CompilationOptions& co) {
  if (co.device_type == ExecutorDeviceType::GPU) {
    if (geo_expr->getOp() != Geo_namespace::GeoBase::GeoOp::kPROJECTION) {
      throw QueryMustRunOnCpu();
    }
  }

  auto argument_list = codegenGeoArgs(geo_expr->getArgs0(), co);

  if (geo_expr->getOp() == Geo_namespace::GeoBase::GeoOp::kPROJECTION) {
    return argument_list;
  }

#ifndef ENABLE_GEOS
  throw std::runtime_error("Geo operation requires GEOS support.");
#endif

  // Basic set of arguments is currently common to all Geos_* func invocations:
  // op kind, type of the first geo arg0, geo arg0 components
  std::string func = "Geos_Wkb"s;
  if (geo_expr->getTypeInfo0().get_output_srid() !=
      geo_expr->get_type_info().get_output_srid()) {
    throw std::runtime_error("GEOS runtime doesn't support geometry transforms.");
  }
  // Prepend arg0 geo SQLType
  argument_list.insert(
      argument_list.begin(),
      cgen_state_->llInt(static_cast<int>(geo_expr->getTypeInfo0().get_type())));
  // Prepend geo expr op
  argument_list.insert(argument_list.begin(),
                       cgen_state_->llInt(static_cast<int>(geo_expr->getOp())));
  for (auto i = 3; i > geo_expr->getTypeInfo0().get_physical_coord_cols(); i--) {
    argument_list.insert(argument_list.end(), cgen_state_->llInt(int64_t(0)));
    argument_list.insert(argument_list.end(),
                         llvm::ConstantPointerNull::get(
                             llvm::Type::getInt32PtrTy(cgen_state_->context_, 0)));
  }
  // Append geo expr compression
  argument_list.insert(
      argument_list.end(),
      cgen_state_->llInt(static_cast<int>(
          geospatial::get_compression_scheme(geo_expr->getTypeInfo0()))));

  // Deal with unary geo predicates
  if (geo_expr->getOp() == Geo_namespace::GeoBase::GeoOp::kISEMPTY ||
      geo_expr->getOp() == Geo_namespace::GeoBase::GeoOp::kISVALID) {
    return codegenGeosPredicateCall(func, argument_list, co);
  }

  throw std::runtime_error("Unsupported unary geo operation.");
  return {};
}

std::vector<llvm::Value*> CodeGenerator::codegenGeoBinOper(
    Analyzer::GeoBinOper const* geo_expr,
    CompilationOptions const& co) {
  if (co.device_type == ExecutorDeviceType::GPU) {
    throw QueryMustRunOnCpu();
  }
#ifndef ENABLE_GEOS
  throw std::runtime_error("Geo operation requires GEOS support.");
#endif

  auto argument_list = codegenGeoArgs(geo_expr->getArgs0(), co);

  // Basic set of arguments is currently common to all Geos_* func invocations:
  // op kind, type of the first geo arg0, geo arg0 components
  std::string func = "Geos_Wkb"s;
  if (geo_expr->getTypeInfo0().get_output_srid() !=
      geo_expr->get_type_info().get_output_srid()) {
    throw std::runtime_error("GEOS runtime doesn't support geometry transforms.");
  }
  // Prepend arg0 geo SQLType
  argument_list.insert(
      argument_list.begin(),
      cgen_state_->llInt(static_cast<int>(geo_expr->getTypeInfo0().get_type())));
  // Prepend geo expr op
  argument_list.insert(argument_list.begin(),
                       cgen_state_->llInt(static_cast<int>(geo_expr->getOp())));
  for (auto i = 3; i > geo_expr->getTypeInfo0().get_physical_coord_cols(); i--) {
    argument_list.insert(argument_list.end(), cgen_state_->llInt(int64_t(0)));
    argument_list.insert(argument_list.end(),
                         llvm::ConstantPointerNull::get(
                             llvm::Type::getInt32PtrTy(cgen_state_->context_, 0)));
  }
  // Append geo expr compression
  argument_list.insert(
      argument_list.end(),
      cgen_state_->llInt(static_cast<int>(
          geospatial::get_compression_scheme(geo_expr->getTypeInfo0()))));

  auto arg1_list = codegenGeoArgs(geo_expr->getArgs1(), co);

  if (geo_expr->getOp() == Geo_namespace::GeoBase::GeoOp::kDIFFERENCE ||
      geo_expr->getOp() == Geo_namespace::GeoBase::GeoOp::kINTERSECTION ||
      geo_expr->getOp() == Geo_namespace::GeoBase::GeoOp::kUNION) {
    func += "_Wkb"s;
    if (geo_expr->getTypeInfo1().get_output_srid() !=
        geo_expr->get_type_info().get_output_srid()) {
      throw std::runtime_error("GEOS runtime doesn't support geometry transforms.");
    }
    // Prepend arg1 geo SQLType
    arg1_list.insert(
        arg1_list.begin(),
        cgen_state_->llInt(static_cast<int>(geo_expr->getTypeInfo1().get_type())));
    for (auto i = 3; i > geo_expr->getTypeInfo1().get_physical_coord_cols(); i--) {
      arg1_list.insert(arg1_list.end(), cgen_state_->llInt(int64_t(0)));
      arg1_list.insert(arg1_list.end(),
                       llvm::ConstantPointerNull::get(
                           llvm::Type::getInt32PtrTy(cgen_state_->context_, 0)));
    }
    // Append geo expr compression
    arg1_list.insert(arg1_list.end(),
                     cgen_state_->llInt(static_cast<int>(
                         geospatial::get_compression_scheme(geo_expr->getTypeInfo1()))));
  } else if (geo_expr->getOp() == Geo_namespace::GeoBase::GeoOp::kBUFFER) {
    // Extra argument in this case is double
    func += "_double"s;
  } else {
    throw std::runtime_error("Unsupported binary geo operation.");
  }

  // Append arg1 to the list
  argument_list.insert(argument_list.end(), arg1_list.begin(), arg1_list.end());

  return codegenGeosConstructorCall(func, argument_list, co);
}

std::vector<llvm::Value*> CodeGenerator::codegenGeoArgs(
    const std::vector<std::shared_ptr<Analyzer::Expr>>& geo_args,
    const CompilationOptions& co) {
  std::vector<llvm::Value*> argument_list;
  bool coord_col = true;
  for (const auto& geo_arg : geo_args) {
    const auto arg = geo_arg.get();
    const auto& arg_ti = arg->get_type_info();
    const auto elem_ti = arg_ti.get_elem_type();
    const auto arg_lvs = codegen(arg, true, co);
    if (arg_ti.is_number()) {
      argument_list.emplace_back(arg_lvs.front());
      continue;
    }
    if (arg_ti.is_geometry()) {
      argument_list.insert(argument_list.end(), arg_lvs.begin(), arg_lvs.end());
      continue;
    }
    CHECK(arg_ti.is_array());
    if (arg_lvs.size() > 1) {
      CHECK_EQ(size_t(2), arg_lvs.size());
      auto ptr_lv = arg_lvs.front();
      if (coord_col) {
        coord_col = false;
      } else {
        ptr_lv = cgen_state_->ir_builder_.CreatePointerCast(
            ptr_lv, llvm::Type::getInt32PtrTy(cgen_state_->context_));
      }
      argument_list.emplace_back(ptr_lv);
      auto cast_len_lv = cgen_state_->ir_builder_.CreateZExt(
          arg_lvs.back(), get_int_type(64, cgen_state_->context_));
      argument_list.emplace_back(cast_len_lv);
    } else {
      CHECK_EQ(size_t(1), arg_lvs.size());
      if (arg_ti.get_size() > 0) {
        argument_list.emplace_back(arg_lvs.front());
        argument_list.emplace_back(cgen_state_->llInt<int64_t>(arg_ti.get_size()));
      } else {
        auto ptr_lv =
            cgen_state_->emitExternalCall("array_buff",
                                          llvm::Type::getInt8PtrTy(cgen_state_->context_),
                                          {arg_lvs.front(), posArg(arg)});
        if (coord_col) {
          coord_col = false;
        } else {
          ptr_lv = cgen_state_->ir_builder_.CreatePointerCast(
              ptr_lv, llvm::Type::getInt32PtrTy(cgen_state_->context_));
        }
        argument_list.emplace_back(ptr_lv);
        const auto len_lv = cgen_state_->emitExternalCall(
            "array_size",
            get_int_type(32, cgen_state_->context_),
            {arg_lvs.front(),
             posArg(arg),
             cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))});
        auto cast_len_lv = cgen_state_->ir_builder_.CreateZExt(
            len_lv, get_int_type(64, cgen_state_->context_));
        argument_list.emplace_back(cast_len_lv);
      }
    }
  }
  return argument_list;
}

std::vector<llvm::Value*> CodeGenerator::codegenGeosPredicateCall(
    const std::string& func,
    std::vector<llvm::Value*> argument_list,
    const CompilationOptions& co) {
  auto i8_type = get_int_type(8, cgen_state_->context_);
  auto result = cgen_state_->ir_builder_.CreateAlloca(i8_type, nullptr, "result");
  argument_list.emplace_back(result);

  // Generate call to GEOS wrapper
  cgen_state_->needs_geos_ = true;
  auto status_lv = cgen_state_->emitExternalCall(
      func, llvm::Type::getInt1Ty(cgen_state_->context_), argument_list);
  // Need to check the status and throw an error if this call has failed.
  llvm::BasicBlock* geos_pred_ok_bb{nullptr};
  llvm::BasicBlock* geos_pred_fail_bb{nullptr};
  geos_pred_ok_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "geos_pred_ok_bb", cgen_state_->row_func_);
  geos_pred_fail_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "geos_pred_fail_bb", cgen_state_->row_func_);
  if (!status_lv) {
    status_lv = cgen_state_->llBool(false);
  }
  cgen_state_->ir_builder_.CreateCondBr(status_lv, geos_pred_ok_bb, geos_pred_fail_bb);
  cgen_state_->ir_builder_.SetInsertPoint(geos_pred_fail_bb);
  cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt(Executor::ERR_GEOS));
  cgen_state_->needs_error_check_ = true;
  cgen_state_->ir_builder_.SetInsertPoint(geos_pred_ok_bb);
  auto res = cgen_state_->ir_builder_.CreateLoad(result);
  return {res};
}

std::vector<llvm::Value*> CodeGenerator::codegenGeosConstructorCall(
    const std::string& func,
    std::vector<llvm::Value*> argument_list,
    const CompilationOptions& co) {
  // Create output buffer pointers, append pointers to output args to
  auto i8_type = get_int_type(8, cgen_state_->context_);
  auto i32_type = get_int_type(32, cgen_state_->context_);
  auto i64_type = get_int_type(64, cgen_state_->context_);
  auto pi8_type = llvm::PointerType::get(i8_type, 0);
  auto pi32_type = llvm::PointerType::get(i32_type, 0);

  auto result_type =
      cgen_state_->ir_builder_.CreateAlloca(i32_type, nullptr, "result_type");
  auto result_coords =
      cgen_state_->ir_builder_.CreateAlloca(pi8_type, nullptr, "result_coords");
  auto result_coords_size =
      cgen_state_->ir_builder_.CreateAlloca(i64_type, nullptr, "result_coords_size");
  auto result_ring_sizes =
      cgen_state_->ir_builder_.CreateAlloca(pi32_type, nullptr, "result_ring_sizes");
  auto result_ring_sizes_size =
      cgen_state_->ir_builder_.CreateAlloca(i64_type, nullptr, "result_ring_sizes_size");
  auto result_poly_rings =
      cgen_state_->ir_builder_.CreateAlloca(pi32_type, nullptr, "result_poly_rings");
  auto result_poly_rings_size =
      cgen_state_->ir_builder_.CreateAlloca(i64_type, nullptr, "result_poly_rings_size");

  argument_list.emplace_back(result_type);
  argument_list.emplace_back(result_coords);
  argument_list.emplace_back(result_coords_size);
  argument_list.emplace_back(result_ring_sizes);
  argument_list.emplace_back(result_ring_sizes_size);
  argument_list.emplace_back(result_poly_rings);
  argument_list.emplace_back(result_poly_rings_size);

  // Generate call to GEOS wrapper
  cgen_state_->needs_geos_ = true;
  auto status_lv = cgen_state_->emitExternalCall(
      func, llvm::Type::getInt1Ty(cgen_state_->context_), argument_list);
  // Need to check the status and throw an error if this call has failed.
  llvm::BasicBlock* geos_ok_bb{nullptr};
  llvm::BasicBlock* geos_fail_bb{nullptr};
  geos_ok_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "geos_ok_bb", cgen_state_->row_func_);
  geos_fail_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "geos_fail_bb", cgen_state_->row_func_);
  if (!status_lv) {
    status_lv = cgen_state_->llBool(false);
  }
  cgen_state_->ir_builder_.CreateCondBr(status_lv, geos_ok_bb, geos_fail_bb);
  cgen_state_->ir_builder_.SetInsertPoint(geos_fail_bb);
  cgen_state_->ir_builder_.CreateRet(cgen_state_->llInt(Executor::ERR_GEOS));
  cgen_state_->needs_error_check_ = true;
  cgen_state_->ir_builder_.SetInsertPoint(geos_ok_bb);

  // TODO: Currently forcing the output to MULTIPOLYGON, but need to handle
  // other possible geometries that geos may return, e.g. a POINT, a LINESTRING
  // Need to handle empty result, e.g. empty intersection.
  // The type of result is returned in `result_type`

  // Load return values
  auto buf1 = cgen_state_->ir_builder_.CreateLoad(result_coords);
  auto buf1s = cgen_state_->ir_builder_.CreateLoad(result_coords_size);
  auto buf2 = cgen_state_->ir_builder_.CreateLoad(result_ring_sizes);
  auto buf2s = cgen_state_->ir_builder_.CreateLoad(result_ring_sizes_size);
  auto buf3 = cgen_state_->ir_builder_.CreateLoad(result_poly_rings);
  auto buf3s = cgen_state_->ir_builder_.CreateLoad(result_poly_rings_size);

  // generate register_buffer_with_executor_rsm() calls to register all output buffers
  cgen_state_->emitExternalCall(
      "register_buffer_with_executor_rsm",
      llvm::Type::getVoidTy(cgen_state_->context_),
      {cgen_state_->llInt(reinterpret_cast<int64_t>(executor())),
       cgen_state_->ir_builder_.CreatePointerCast(buf1, pi8_type)});
  cgen_state_->emitExternalCall(
      "register_buffer_with_executor_rsm",
      llvm::Type::getVoidTy(cgen_state_->context_),
      {cgen_state_->llInt(reinterpret_cast<int64_t>(executor())),
       cgen_state_->ir_builder_.CreatePointerCast(buf2, pi8_type)});
  cgen_state_->emitExternalCall(
      "register_buffer_with_executor_rsm",
      llvm::Type::getVoidTy(cgen_state_->context_),
      {cgen_state_->llInt(reinterpret_cast<int64_t>(executor())),
       cgen_state_->ir_builder_.CreatePointerCast(buf3, pi8_type)});

  return {cgen_state_->ir_builder_.CreatePointerCast(buf1, pi8_type),
          buf1s,
          cgen_state_->ir_builder_.CreatePointerCast(buf2, pi32_type),
          buf2s,
          cgen_state_->ir_builder_.CreatePointerCast(buf3, pi32_type),
          buf3s};
}
