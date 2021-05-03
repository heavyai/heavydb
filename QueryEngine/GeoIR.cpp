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

#include "Geospatial/Compression.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/Execute.h"

ArrayLoadCodegen CodeGenerator::codegenGeoArrayLoadAndNullcheck(llvm::Value* byte_stream,
                                                                llvm::Value* pos,
                                                                const SQLTypeInfo& ti) {
  CHECK(byte_stream);

  const auto key = std::make_pair(byte_stream, pos);
  auto cache_itr = cgen_state_->array_load_cache_.find(key);
  if (cache_itr != cgen_state_->array_load_cache_.end()) {
    return cache_itr->second;
  }
  const bool is_nullable = !ti.get_notnull();
  CHECK(ti.get_type() == kPOINT);  // TODO: lift this

  auto pt_arr_buf =
      cgen_state_->emitExternalCall("array_buff",
                                    llvm::Type::getInt8PtrTy(cgen_state_->context_),
                                    {key.first, key.second});
  llvm::Value* pt_is_null{nullptr};
  if (is_nullable) {
    pt_is_null =
        cgen_state_->emitExternalCall("point_coord_array_is_null",
                                      llvm::Type::getInt1Ty(cgen_state_->context_),
                                      {key.first, key.second});
  }
  ArrayLoadCodegen arr_load{pt_arr_buf, nullptr, pt_is_null};
  cgen_state_->array_load_cache_.insert(std::make_pair(key, arr_load));
  return arr_load;
}

std::vector<llvm::Value*> CodeGenerator::codegenGeoColumnVar(
    const Analyzer::GeoColumnVar* geo_col_var,
    const bool fetch_columns,
    const CompilationOptions& co) {
  const auto& ti = geo_col_var->get_type_info();
  if (ti.get_type() == kPOINT) {
    // create a new operand which is just the coords and codegen it
    const auto catalog = executor()->getCatalog();
    CHECK(catalog);
    const auto coords_column_id = geo_col_var->get_column_id() + 1;  // + 1 for coords
    auto coords_cd =
        get_column_descriptor(coords_column_id, geo_col_var->get_table_id(), *catalog);
    CHECK(coords_cd);

    const auto coords_col_var = Analyzer::ColumnVar(coords_cd->columnType,
                                                    geo_col_var->get_table_id(),
                                                    coords_column_id,
                                                    geo_col_var->get_rte_idx());
    const auto coords_lv = codegen(&coords_col_var, /*fetch_columns=*/true, co);
    CHECK_EQ(coords_lv.size(), size_t(1));  // ptr
    return coords_lv;
  } else {
    UNREACHABLE() << geo_col_var->toString();
  }
  return {};
}

std::vector<llvm::Value*> CodeGenerator::codegenGeoExpr(const Analyzer::GeoExpr* expr,
                                                        const CompilationOptions& co) {
  auto geo_constant = dynamic_cast<const Analyzer::GeoConstant*>(expr);
  if (geo_constant) {
    return codegenGeoConstant(geo_constant, co);
  }
  auto geo_operator = dynamic_cast<const Analyzer::GeoOperator*>(expr);
  if (geo_operator) {
    return codegenGeoOperator(geo_operator, co);
  }
  auto geo_function = dynamic_cast<const Analyzer::GeoFunctionOperator*>(expr);
  if (geo_function) {
    return codegenGeoFunctionOperator(geo_function, co);
  }
  UNREACHABLE() << expr->toString();
  return {};
}

std::vector<llvm::Value*> CodeGenerator::codegenGeoConstant(
    const Analyzer::GeoConstant* geo_constant,
    const CompilationOptions& co) {
  std::vector<llvm::Value*> ret;
  for (size_t i = 0; i < geo_constant->physicalCols(); i++) {
    auto physical_constant = geo_constant->makePhysicalConstant(i);
    auto operand_lvs = codegen(physical_constant.get(), /*fetch_columns=*/true, co);
    CHECK_EQ(operand_lvs.size(), size_t(2));
    ret.insert(ret.end(), operand_lvs.begin(), operand_lvs.end());
  }
  return ret;
}

std::vector<llvm::Value*> CodeGenerator::codegenGeoOperator(
    const Analyzer::GeoOperator* geo_operator,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);

  if (geo_operator->getName() == "ST_PointN") {
    CHECK_EQ(geo_operator->size(), size_t(2));
    const auto geo_arg = geo_operator->getOperand(0);
    const auto geo_arg_expr = dynamic_cast<const Analyzer::GeoExpr*>(geo_arg);
    CHECK(geo_arg_expr) << geo_arg->toString();
    auto geo_lvs = codegenGeoExpr(geo_arg_expr, co);  // array_buff, array_size
    CHECK_EQ(geo_lvs.size(), size_t(2));
    const auto index_expr = geo_operator->getOperand(1);
    const auto index_lv = codegen(index_expr, /*fetch_columns=*/true, co);

    auto& builder = cgen_state_->ir_builder_;

    // return a nullptr if index is out of bounds
    const auto is_null_lv = builder.CreateNot(
        builder.CreateICmp(llvm::ICmpInst::ICMP_SLT, index_lv.front(), geo_lvs.back()));
    CodeGenerator::NullCheckCodegen nullcheck_codegen(cgen_state_,
                                                      executor(),
                                                      is_null_lv,
                                                      SQLTypeInfo(kBOOLEAN),
                                                      "st_pointn_range_check");
    const auto& geo_ti = geo_arg_expr->get_type_info();
    llvm::Value* array_buff_cast{nullptr};
    if (geo_ti.get_compression() == kENCODING_GEOINT) {
      array_buff_cast = builder.CreateBitCast(
          geo_lvs.front(), llvm::Type::getInt32PtrTy(cgen_state_->context_));
    } else {
      array_buff_cast = builder.CreateBitCast(
          geo_lvs.front(), llvm::Type::getDoublePtrTy(cgen_state_->context_));
    }

    auto array_offset_lv =
        builder.CreateGEP(array_buff_cast, index_lv, "ST_PointN_Offset");
    auto ret_lv = nullcheck_codegen.finalize(
        llvm::ConstantPointerNull::get(
            geo_ti.get_compression() == kENCODING_GEOINT
                ? llvm::Type::getInt32PtrTy(cgen_state_->context_)
                : llvm::Type::getDoublePtrTy(cgen_state_->context_)),
        array_offset_lv);
    return {ret_lv, geo_lvs.back()};
  } else if (geo_operator->getName() == "ST_EndPoint" ||
             geo_operator->getName() == "ST_StartPoint") {
    CHECK_EQ(geo_operator->size(), size_t(1));
    const auto geo_arg = geo_operator->getOperand(0);
    const auto geo_arg_expr = dynamic_cast<const Analyzer::GeoExpr*>(geo_arg);
    CHECK(geo_arg_expr) << geo_arg->toString();
    auto geo_lvs = codegenGeoExpr(geo_arg_expr, co);  // array_buff, array_size
    CHECK_EQ(geo_lvs.size(), size_t(2));

    auto& builder = cgen_state_->ir_builder_;
    const auto& geo_ti = geo_arg_expr->get_type_info();
    llvm::Value* array_buff_cast{nullptr};
    size_t elem_size_bytes = 0;  // TODO: make int32_t
    if (geo_ti.get_compression() == kENCODING_GEOINT) {
      array_buff_cast = builder.CreateBitCast(
          geo_lvs.front(), llvm::Type::getInt32PtrTy(cgen_state_->context_));
      elem_size_bytes = 4;  // 4-byte ints
    } else {
      array_buff_cast = builder.CreateBitCast(
          geo_lvs.front(), llvm::Type::getDoublePtrTy(cgen_state_->context_));
      elem_size_bytes = 8;  // doubles
    }
    CHECK_GT(elem_size_bytes, 0);

    const auto num_elements_lv =
        builder.CreateSDiv(geo_lvs.back(), cgen_state_->llInt(int32_t(elem_size_bytes)));
    const auto end_index_lv =
        builder.CreateSub(num_elements_lv, cgen_state_->llInt(int32_t(2)));
    auto array_offset_lv = builder.CreateGEP(
        array_buff_cast, end_index_lv, geo_operator->getName() + "_Offset");
    return {array_offset_lv, geo_lvs.back()};
  } else if (geo_operator->getName() == "ST_NPoints") {
    const auto operand = geo_operator->getOperand(0);
    auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(operand);
    CHECK(col_var);

    const auto& geo_ti = col_var->get_type_info();
    CHECK(geo_ti.is_geometry());

    // create a new operand which is just the coords and codegen it
    const auto catalog = executor()->getCatalog();
    CHECK(catalog);
    const auto coords_column_id = col_var->get_column_id() + 1;  // + 1 for coords
    auto coords_cd =
        get_column_descriptor(coords_column_id, col_var->get_table_id(), *catalog);
    CHECK(coords_cd);

    const auto coords_col_var = Analyzer::ColumnVar(coords_cd->columnType,
                                                    col_var->get_table_id(),
                                                    coords_column_id,
                                                    col_var->get_rte_idx());
    const auto coords_lv = codegen(&coords_col_var, /*fetch_columns=*/true, co);
    CHECK_EQ(coords_lv.size(), size_t(1));  // ptr, size

    std::string fn_name("array_size");

    const auto& elem_ti = coords_cd->columnType.get_elem_type();
    std::vector<llvm::Value*> array_size_args{
        coords_lv.front(),
        posArg(&coords_col_var),
        cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))};

    const bool is_nullable = !geo_ti.get_notnull();

    if (is_nullable) {
      fn_name += "_nullable";
      array_size_args.push_back(
          cgen_state_->inlineIntNull(geo_operator->get_type_info()));
    }
    const auto coords_arr_sz = cgen_state_->emitExternalCall(
        fn_name, get_int_type(32, cgen_state_->context_), array_size_args);

    std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
    if (is_nullable) {
      nullcheck_codegen = std::make_unique<NullCheckCodegen>(cgen_state_,
                                                             executor(),
                                                             coords_arr_sz,
                                                             SQLTypeInfo(kINT),
                                                             "st_npoints_nullcheck");
    }

    // divide the coord size by the constant compression value and return it
    auto& builder = cgen_state_->ir_builder_;
    llvm::Value* conversion_constant{nullptr};
    if (geo_ti.get_compression() == kENCODING_GEOINT) {
      conversion_constant = cgen_state_->llInt(4);
    } else {
      conversion_constant = cgen_state_->llInt(8);
    }
    CHECK(conversion_constant);
    const auto total_num_pts = builder.CreateUDiv(coords_arr_sz, conversion_constant);
    auto ret = builder.CreateUDiv(total_num_pts, cgen_state_->llInt(2));
    if (is_nullable) {
      ret = nullcheck_codegen->finalize(
          cgen_state_->inlineIntNull(geo_operator->get_type_info()), ret);
    }
    return {ret};
  } else if (geo_operator->getName() == "ST_NRings") {
    const auto operand = geo_operator->getOperand(0);
    auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(operand);
    CHECK(col_var);

    const auto& geo_ti = col_var->get_type_info();
    CHECK(geo_ti.is_geometry());

    // create a new operand which is just the ring sizes and codegen it
    const auto catalog = executor()->getCatalog();
    CHECK(catalog);
    const auto ring_sizes_column_id = col_var->get_column_id() + 2;  // + 2 for ring sizes
    auto ring_sizes_cd =
        get_column_descriptor(ring_sizes_column_id, col_var->get_table_id(), *catalog);
    CHECK(ring_sizes_cd);

    const auto ring_sizes_col_var = Analyzer::ColumnVar(ring_sizes_cd->columnType,
                                                        col_var->get_table_id(),
                                                        ring_sizes_column_id,
                                                        col_var->get_rte_idx());
    const auto ring_sizes_lv = codegen(&ring_sizes_col_var, /*fetch_columns=*/true, co);
    CHECK_EQ(ring_sizes_lv.size(), size_t(1));  // ptr, size

    std::string fn_name("array_size");

    const auto& elem_ti = ring_sizes_cd->columnType.get_elem_type();
    std::vector<llvm::Value*> array_size_args{
        ring_sizes_lv.front(),
        posArg(&ring_sizes_col_var),
        cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))};

    const bool is_nullable = !geo_ti.get_notnull();

    if (is_nullable) {
      fn_name += "_nullable";
      array_size_args.push_back(
          cgen_state_->inlineIntNull(geo_operator->get_type_info()));
    }
    const auto total_num_rings_lv = cgen_state_->emitExternalCall(
        fn_name, get_int_type(32, cgen_state_->context_), array_size_args);

    std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
    if (is_nullable) {
      nullcheck_codegen = std::make_unique<NullCheckCodegen>(cgen_state_,
                                                             executor(),
                                                             total_num_rings_lv,
                                                             SQLTypeInfo(kINT),
                                                             "st_npoints_nullcheck");
    }

    auto ret = total_num_rings_lv;
    if (is_nullable) {
      ret = nullcheck_codegen->finalize(
          cgen_state_->inlineIntNull(geo_operator->get_type_info()), ret);
    }
    return {ret};
  } else if (geo_operator->getName() == "ST_X" || geo_operator->getName() == "ST_Y") {
    const auto key = geo_operator->toString();
    auto geo_target_cache_it = cgen_state_->geo_target_cache_.find(key);
    if (geo_target_cache_it != cgen_state_->geo_target_cache_.end()) {
      return {geo_target_cache_it->second};
    }

    auto& builder = cgen_state_->ir_builder_;

    const auto operand = geo_operator->getOperand(0);
    const auto& geo_ti = operand->get_type_info();
    const auto operand_lvs = codegen(operand, /*fetch_columns=*/true, co);
    llvm::Value* array_buff_ptr{nullptr};
    llvm::Value* is_null{nullptr};
    if (operand_lvs.size() == 1) {
      // col byte stream, get the array buffer ptr and is null attributes and cache
      auto arr_load_lvs =
          codegenGeoArrayLoadAndNullcheck(operand_lvs.front(), posArg(operand), geo_ti);
      array_buff_ptr = arr_load_lvs.buffer;
      is_null = arr_load_lvs.is_null;
    } else {
      // ptr and size
      CHECK_EQ(operand_lvs.size(), size_t(2));
      if (dynamic_cast<const Analyzer::GeoOperator*>(operand)) {
        // null check will be if the ptr is a nullptr
        is_null = builder.CreateICmp(
            llvm::CmpInst::ICMP_EQ,
            operand_lvs.front(),
            llvm::ConstantPointerNull::get(
                geo_ti.get_compression() == kENCODING_GEOINT
                    ? llvm::Type::getInt32PtrTy(cgen_state_->context_)
                    : llvm::Type::getDoublePtrTy(cgen_state_->context_)));
      }

      // TODO: nulls from other types not yet supported
      array_buff_ptr = operand_lvs.front();
    }
    CHECK(array_buff_ptr) << geo_operator->toString();

    const bool is_nullable = !geo_ti.get_notnull();
    std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
    if (is_nullable) {
      CHECK(is_null);
      nullcheck_codegen =
          std::make_unique<NullCheckCodegen>(cgen_state_,
                                             executor(),
                                             is_null,
                                             SQLTypeInfo(kBOOLEAN),
                                             geo_operator->getName() + "_nullcheck");
    }

    const bool is_x = geo_operator->getName() == "ST_X";
    const std::string expr_name = is_x ? "x" : "y";

    llvm::Value* coord_lv;
    if (geo_ti.get_compression() == kENCODING_GEOINT) {
      auto compressed_arr_ptr = builder.CreateBitCast(
          array_buff_ptr, llvm::Type::getInt32PtrTy(cgen_state_->context_));
      auto coord_index = is_x ? cgen_state_->llInt(0) : cgen_state_->llInt(1);
      auto coord_lv_ptr =
          builder.CreateGEP(compressed_arr_ptr, coord_index, expr_name + "_coord_ptr");
      auto compressed_coord_lv =
          builder.CreateLoad(coord_lv_ptr, expr_name + "_coord_compressed");

      coord_lv =
          cgen_state_->emitExternalCall("decompress_" + expr_name + "_coord_geoint",
                                        llvm::Type::getDoubleTy(cgen_state_->context_),
                                        {compressed_coord_lv});
    } else {
      auto coord_arr_ptr = builder.CreateBitCast(
          array_buff_ptr, llvm::Type::getDoublePtrTy(cgen_state_->context_));
      auto coord_index = is_x ? cgen_state_->llInt(0) : cgen_state_->llInt(1);
      auto coord_lv_ptr =
          builder.CreateGEP(coord_arr_ptr, coord_index, expr_name + "_coord_ptr");
      coord_lv = builder.CreateLoad(coord_lv_ptr, expr_name + "_coord");
    }

    // TODO: do this with transformation nodes explicitly
    if (geo_ti.get_input_srid() != geo_ti.get_output_srid()) {
      if (geo_ti.get_input_srid() == 4326) {
        if (geo_ti.get_output_srid() == 900913) {
          // convert WGS 84 -> Web mercator
          coord_lv = cgen_state_->emitExternalCall(
              "conv_4326_900913_" + expr_name,
              llvm::Type::getDoubleTy(cgen_state_->context_),
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
    if (is_nullable) {
      ret = nullcheck_codegen->finalize(cgen_state_->inlineFpNull(SQLTypeInfo(kDOUBLE)),
                                        ret);
    }
    CHECK(cgen_state_->geo_target_cache_.insert(std::make_pair(key, ret)).second);
    return {ret};
  }
  UNREACHABLE() << geo_operator->toString();
  return {};
}

namespace {

// TODO: de-dupe
std::string suffix(SQLTypes type) {
  if (type == kPOINT) {
    return std::string("_Point");
  }
  if (type == kLINESTRING) {
    return std::string("_LineString");
  }
  if (type == kPOLYGON) {
    return std::string("_Polygon");
  }
  if (type == kMULTIPOLYGON) {
    return std::string("_MultiPolygon");
  }
  throw std::runtime_error("Unsupported argument type");
}

}  // namespace

std::vector<llvm::Value*> CodeGenerator::codegenGeoFunctionOperator(
    const Analyzer::GeoFunctionOperator* geo_func_oper,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);

  if (geo_func_oper->getName() == "ST_Perimeter" ||
      geo_func_oper->getName() == "ST_Area") {
    const auto operand = geo_func_oper->getArg(0);
    const auto& arg_ti = operand->get_type_info();

    const bool is_nullable = !arg_ti.get_notnull();
    std::string size_fn_name = "array_size";
    if (is_nullable) {
      size_fn_name += "_nullable";
    }

    auto operand_lvs = codegen(operand, /*fetch_columns=*/true, co);
    if (dynamic_cast<const Analyzer::ColumnVar*>(operand)) {
      CHECK_EQ(operand_lvs.size(),
               size_t(arg_ti.get_physical_coord_cols()));  // chunk iter ptr
      // this will give us back the byte stream -- add codegen for the array loads
      std::vector<llvm::Value*> array_operand_lvs;
      for (auto lv : operand_lvs) {
        array_operand_lvs.push_back(
            cgen_state_->emitExternalCall("array_buff",
                                          llvm::Type::getInt8PtrTy(cgen_state_->context_),
                                          {lv, posArg(operand)}));
        const auto ptr_type = llvm::dyn_cast_or_null<llvm::PointerType>(lv->getType());
        CHECK(ptr_type);
        const auto elem_type = ptr_type->getElementType();
        CHECK(elem_type);
        std::vector<llvm::Value*> array_sz_args{
            lv,
            posArg(operand),
            cgen_state_->llInt(log2_bytes(elem_type->getPrimitiveSizeInBits() / 8))};
        if (is_nullable) {
          array_sz_args.push_back(cgen_state_->llInt(inline_int_null_value<int32_t>()));
        }
        array_operand_lvs.push_back(cgen_state_->emitExternalCall(
            size_fn_name, get_int_type(32, cgen_state_->context_), array_sz_args));
      }
      operand_lvs = array_operand_lvs;
    }
    CHECK_EQ(operand_lvs.size(),
             size_t(2 * arg_ti.get_physical_coord_cols()));  // array ptr and size

    const bool is_geodesic =
        arg_ti.get_subtype() == kGEOGRAPHY && arg_ti.get_output_srid() == 4326;
    std::string func_name = geo_func_oper->getName();

    std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
    if (is_nullable) {
      auto null_check_operand_lv = operand_lvs[1];
      if (null_check_operand_lv->getType() !=
          llvm::Type::getInt32Ty(cgen_state_->context_)) {
        CHECK(null_check_operand_lv->getType() ==
              llvm::Type::getInt64Ty(cgen_state_->context_));
        // Geos functions come out 64-bit, cast down to 32 for now
        auto& builder = cgen_state_->ir_builder_;
        null_check_operand_lv = builder.CreateTrunc(
            null_check_operand_lv, llvm::Type::getInt32Ty(cgen_state_->context_));
      }
      nullcheck_codegen =
          std::make_unique<NullCheckCodegen>(cgen_state_,
                                             executor(),
                                             null_check_operand_lv,  // coords size
                                             SQLTypeInfo(kINT),
                                             func_name + "_nullcheck");
    }

    CHECK(arg_ti.get_type() == kPOLYGON || arg_ti.get_type() == kMULTIPOLYGON);
    func_name += suffix(arg_ti.get_type());
    if (is_geodesic) {
      func_name += "_Geodesic";
    }
    // push back ic, isr, osr for now
    operand_lvs.push_back(
        cgen_state_->llInt(Geospatial::get_compression_scheme(arg_ti)));  // ic
    operand_lvs.push_back(cgen_state_->llInt(arg_ti.get_input_srid()));   // in srid
    operand_lvs.push_back(cgen_state_->llInt(arg_ti.get_output_srid()));  // out srid

    const auto& ret_ti = geo_func_oper->get_type_info();
    CHECK(ret_ti.get_type() == kDOUBLE || ret_ti.get_type() == kFLOAT);
    auto ret = cgen_state_->emitExternalCall(
        func_name,
        ret_ti.get_type() == kDOUBLE ? llvm::Type::getDoubleTy(cgen_state_->context_)
                                     : llvm::Type::getFloatTy(cgen_state_->context_),
        operand_lvs);
    if (is_nullable) {
      ret = nullcheck_codegen->finalize(
          cgen_state_->inlineFpNull(geo_func_oper->get_type_info()), ret);
    }
    return {ret};
  }
  UNREACHABLE() << geo_func_oper->toString();
  return {};
}

std::vector<llvm::Value*> CodeGenerator::codegenGeoUOper(
    const Analyzer::GeoUOper* geo_expr,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (co.device_type == ExecutorDeviceType::GPU) {
    if (geo_expr->getOp() != Geospatial::GeoBase::GeoOp::kPROJECTION) {
      throw QueryMustRunOnCpu();
    }
  }

  auto argument_list = codegenGeoArgs(geo_expr->getArgs0(), co);

  if (geo_expr->getOp() == Geospatial::GeoBase::GeoOp::kPROJECTION) {
    return argument_list;
  }

#ifndef ENABLE_GEOS
  throw std::runtime_error("Geo operation requires GEOS support.");
#endif

  // Basic set of arguments is currently common to all Geos_* func invocations:
  // op kind, type of the first geo arg0, geo arg0 components
  std::string func = "Geos_Wkb"s;
  if (geo_expr->getTypeInfo0().transforms()) {
    throw std::runtime_error(
        "GEOS runtime does not support transforms on geometry inputs.");
  }
  // Catch transforms applied to geometry construction only
  if (geo_expr->get_type_info().transforms()) {
    throw std::runtime_error(
        "GEOS runtime does not support transforms on geometry outputs.");
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
          Geospatial::get_compression_scheme(geo_expr->getTypeInfo0()))));
  // Append geo expr SRID
  argument_list.insert(
      argument_list.end(),
      cgen_state_->llInt(static_cast<int>(geo_expr->getTypeInfo0().get_output_srid())));

  // Deal with unary geo predicates
  if (geo_expr->getOp() == Geospatial::GeoBase::GeoOp::kISEMPTY ||
      geo_expr->getOp() == Geospatial::GeoBase::GeoOp::kISVALID) {
    return codegenGeosPredicateCall(func, argument_list, co);
  }

  throw std::runtime_error("Unsupported unary geo operation.");
  return {};
}

std::vector<llvm::Value*> CodeGenerator::codegenGeoBinOper(
    Analyzer::GeoBinOper const* geo_expr,
    CompilationOptions const& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
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
  if (geo_expr->getTypeInfo0().transforms() || geo_expr->getTypeInfo1().transforms()) {
    throw std::runtime_error(
        "GEOS runtime does not support transforms on geometry inputs.");
  }
  if (geo_expr->get_type_info().transforms()) {
    throw std::runtime_error(
        "GEOS runtime does not support transforms on geometry outputs.");
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
          Geospatial::get_compression_scheme(geo_expr->getTypeInfo0()))));
  // Append geo expr SRID
  argument_list.insert(
      argument_list.end(),
      cgen_state_->llInt(static_cast<int>(geo_expr->getTypeInfo0().get_output_srid())));

  auto arg1_list = codegenGeoArgs(geo_expr->getArgs1(), co);

  if (geo_expr->getOp() == Geospatial::GeoBase::GeoOp::kDIFFERENCE ||
      geo_expr->getOp() == Geospatial::GeoBase::GeoOp::kINTERSECTION ||
      geo_expr->getOp() == Geospatial::GeoBase::GeoOp::kUNION) {
    func += "_Wkb"s;
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
                         Geospatial::get_compression_scheme(geo_expr->getTypeInfo1()))));
    // Append geo expr compression
    arg1_list.insert(arg1_list.end(),
                     cgen_state_->llInt(geo_expr->getTypeInfo1().get_output_srid()));
  } else if (geo_expr->getOp() == Geospatial::GeoBase::GeoOp::kBUFFER) {
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
  AUTOMATIC_IR_METADATA(cgen_state_);
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
        // Set up the pointer lv for a dynamically generated point
        auto ptr_lv = arg_lvs.front();
        auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(arg);
        // Override for point coord column access
        if (col_var) {
          ptr_lv = cgen_state_->emitExternalCall(
              "fast_fixlen_array_buff",
              llvm::Type::getInt8PtrTy(cgen_state_->context_),
              {arg_lvs.front(), posArg(arg)});
        }
        if (coord_col) {
          coord_col = false;
        } else {
          ptr_lv = cgen_state_->ir_builder_.CreatePointerCast(
              ptr_lv, llvm::Type::getInt32PtrTy(cgen_state_->context_));
        }
        argument_list.emplace_back(ptr_lv);
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
  AUTOMATIC_IR_METADATA(cgen_state_);
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
      cgen_state_->context_, "geos_pred_ok_bb", cgen_state_->current_func_);
  geos_pred_fail_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "geos_pred_fail_bb", cgen_state_->current_func_);
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
  AUTOMATIC_IR_METADATA(cgen_state_);
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
      cgen_state_->context_, "geos_ok_bb", cgen_state_->current_func_);
  geos_fail_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "geos_fail_bb", cgen_state_->current_func_);
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
