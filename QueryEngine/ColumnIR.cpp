/*
 * Copyright 2017 MapD Technologies, Inc.
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
#include "Codec.h"
#include "Execute.h"
#include "WindowContext.h"

// Code generation routines and helpers for working with column expressions.

namespace {

// Return the right decoder for a given column expression. Doesn't handle
// variable length data. The decoder encapsulates the code generation logic.
std::shared_ptr<Decoder> get_col_decoder(const Analyzer::ColumnVar* col_var) {
  const auto enc_type = col_var->get_compression();
  const auto& ti = col_var->get_type_info();
  switch (enc_type) {
    case kENCODING_NONE: {
      const auto int_type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
      switch (int_type) {
        case kBOOLEAN:
          return std::make_shared<FixedWidthInt>(1);
        case kTINYINT:
          return std::make_shared<FixedWidthInt>(1);
        case kSMALLINT:
          return std::make_shared<FixedWidthInt>(2);
        case kINT:
          return std::make_shared<FixedWidthInt>(4);
        case kBIGINT:
          return std::make_shared<FixedWidthInt>(8);
        case kFLOAT:
          return std::make_shared<FixedWidthReal>(false);
        case kDOUBLE:
          return std::make_shared<FixedWidthReal>(true);
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          return std::make_shared<FixedWidthInt>(8);
        default:
          CHECK(false);
      }
    }
    case kENCODING_DICT:
      CHECK(ti.is_string());
      // For dictionary-encoded columns encoded on less than 4 bytes, we can use
      // unsigned representation for double the maximum cardinality. The inline
      // null value is going to be the maximum value of the underlying type.
      if (ti.get_size() < ti.get_logical_size()) {
        return std::make_shared<FixedWidthUnsigned>(ti.get_size());
      }
      return std::make_shared<FixedWidthInt>(ti.get_size());
    case kENCODING_FIXED: {
      const auto bit_width = col_var->get_comp_param();
      CHECK_EQ(0, bit_width % 8);
      return std::make_shared<FixedWidthInt>(bit_width / 8);
    }
    case kENCODING_DATE_IN_DAYS: {
      CHECK(ti.is_date_in_days());
      return col_var->get_comp_param() == 16 ? std::make_shared<FixedWidthSmallDate>(2)
                                             : std::make_shared<FixedWidthSmallDate>(4);
    }
    default:
      abort();
  }
}

size_t get_col_bit_width(const Analyzer::ColumnVar* col_var) {
  const auto& type_info = col_var->get_type_info();
  return get_bit_width(type_info);
}

int adjusted_range_table_index(const Analyzer::ColumnVar* col_var) {
  return col_var->get_rte_idx() == -1 ? 0 : col_var->get_rte_idx();
}

}  // namespace

std::vector<llvm::Value*> CodeGenerator::codegenColumn(const Analyzer::ColumnVar* col_var,
                                                       const bool fetch_column,
                                                       const CompilationOptions& co) {
  if (col_var->get_rte_idx() <= 0 ||
      cgen_state_->outer_join_match_found_per_level_.empty() ||
      !foundOuterJoinMatch(col_var->get_rte_idx())) {
    return codegenColVar(col_var, fetch_column, true, co);
  }
  return codegenOuterJoinNullPlaceholder(col_var, fetch_column, co);
}

std::vector<llvm::Value*> CodeGenerator::codegenColVar(const Analyzer::ColumnVar* col_var,
                                                       const bool fetch_column,
                                                       const bool update_query_plan,
                                                       const CompilationOptions& co) {
  const bool hoist_literals = co.hoist_literals_;
  auto col_id = col_var->get_column_id();
  const int rte_idx = adjusted_range_table_index(col_var);
  CHECK_LT(static_cast<size_t>(rte_idx), cgen_state_->frag_offsets_.size());
  const auto catalog = executor()->getCatalog();
  CHECK(catalog);
  if (col_var->get_table_id() > 0) {
    auto cd = get_column_descriptor(col_id, col_var->get_table_id(), *catalog);
    if (cd->isVirtualCol) {
      CHECK(cd->columnName == "rowid");
      return {codegenRowId(col_var, co)};
    }
    auto col_ti = cd->columnType;
    if (col_ti.get_physical_coord_cols() > 0) {
      std::vector<llvm::Value*> cols;
      for (auto i = 0; i < col_ti.get_physical_coord_cols(); i++) {
        auto cd0 =
            get_column_descriptor(col_id + i + 1, col_var->get_table_id(), *catalog);
        auto col0_ti = cd0->columnType;
        CHECK(!cd0->isVirtualCol);
        auto col0_var = makeExpr<Analyzer::ColumnVar>(
            col0_ti, col_var->get_table_id(), cd0->columnId, rte_idx);
        auto col = codegenColVar(col0_var.get(), fetch_column, false, co);
        cols.insert(cols.end(), col.begin(), col.end());
        if (!fetch_column && plan_state_->isLazyFetchColumn(col_var)) {
          plan_state_->columns_to_not_fetch_.insert(
              std::make_pair(col_var->get_table_id(), col0_var->get_column_id()));
        }
      }
      if (!fetch_column && plan_state_->isLazyFetchColumn(col_var)) {
        plan_state_->columns_to_not_fetch_.insert(
            std::make_pair(col_var->get_table_id(), col_var->get_column_id()));
      } else {
        plan_state_->columns_to_fetch_.insert(
            std::make_pair(col_var->get_table_id(), col_var->get_column_id()));
      }
      return cols;
    }
  } else {
    if (col_var->get_type_info().is_geometry()) {
      throw std::runtime_error(
          "Geospatial columns not supported in temporary tables yet");
    }
  }
  const auto grouped_col_lv = resolveGroupedColumnReference(col_var);
  if (grouped_col_lv) {
    return {grouped_col_lv};
  }
  const int local_col_id = plan_state_->getLocalColumnId(col_var, fetch_column);
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext();
  // only generate the decoding code once; if a column has been previously
  // fetched in the generated IR, we'll reuse it
  if (!window_func_context) {
    auto it = cgen_state_->fetch_cache_.find(local_col_id);
    if (it != cgen_state_->fetch_cache_.end()) {
      return {it->second};
    }
  }
  const auto hash_join_lhs = hashJoinLhs(col_var);
  // Use the already fetched left-hand side of an equi-join if the types are identical.
  // Currently, types can only be different because of different underlying dictionaries.
  if (hash_join_lhs && hash_join_lhs->get_type_info() == col_var->get_type_info()) {
    if (plan_state_->isLazyFetchColumn(col_var)) {
      plan_state_->columns_to_fetch_.insert(
          std::make_pair(col_var->get_table_id(), col_var->get_column_id()));
    }
    return codegen(hash_join_lhs.get(), fetch_column, co);
  }

  if (!plan_state_->isLazyFetchColumn(col_var)) {
    InputColDescriptor col{
        col_var->get_column_id(), col_var->get_table_id(), col_var->get_rte_idx()};
    if (cgen_state_->hashed_cols_.count(col)) {
      auto val = codegenFixedLengthColVar(col_var,
                                          cgen_state_->hashed_cols_.at(col),
                                          ll_int((int64_t)0, cgen_state_->context_));
      auto it_ok = cgen_state_->fetch_cache_.insert(
          std::make_pair(local_col_id, std::vector<llvm::Value*>{val}));
      CHECK(it_ok.second);
      return {it_ok.first->second};
    }
  }

  auto pos_arg = posArg(col_var);
  if (window_func_context) {
    pos_arg = codegenWindowPosition(window_func_context, pos_arg);
  }
  auto col_byte_stream = colByteStream(col_var, fetch_column, hoist_literals);
  if (plan_state_->isLazyFetchColumn(col_var)) {
    if (update_query_plan) {
      plan_state_->columns_to_not_fetch_.insert(
          std::make_pair(col_var->get_table_id(), col_var->get_column_id()));
    }
    if (rte_idx > 0) {
      const auto offset = cgen_state_->frag_offsets_[rte_idx];
      if (offset) {
        return {cgen_state_->ir_builder_.CreateAdd(pos_arg, offset)};
      } else {
        return {pos_arg};
      }
    }
    return {pos_arg};
  }
  const auto& col_ti = col_var->get_type_info();
  if (col_ti.is_string() && col_ti.get_compression() == kENCODING_NONE) {
    const auto varlen_str_column_lvs =
        codegenVariableLengthStringColVar(col_byte_stream, pos_arg);
    if (!window_func_context) {
      auto it_ok = cgen_state_->fetch_cache_.insert(
          std::make_pair(local_col_id, varlen_str_column_lvs));
      CHECK(it_ok.second);
    }
    return varlen_str_column_lvs;
  }
  if (col_ti.is_array()) {
    return {col_byte_stream};
  }
  if (window_func_context) {
    return {codegenFixedLengthColVarInWindow(col_var, col_byte_stream, pos_arg)};
  }
  const auto fixed_length_column_lv =
      codegenFixedLengthColVar(col_var, col_byte_stream, pos_arg);
  auto it_ok = cgen_state_->fetch_cache_.insert(
      std::make_pair(local_col_id, std::vector<llvm::Value*>{fixed_length_column_lv}));
  CHECK(it_ok.second);
  return {it_ok.first->second};
}

llvm::Value* CodeGenerator::codegenWindowPosition(
    WindowFunctionContext* window_func_context,
    llvm::Value* pos_arg) {
  const auto window_position = cgen_state_->emitCall(
      "row_number_window_func",
      {cgen_state_->llInt(reinterpret_cast<const int64_t>(window_func_context->output())),
       pos_arg});
  window_func_context->setRowNumber(window_position);
  return window_position;
}

// Generate code for fixed length column types (number, timestamp or date,
// dictionary-encoded string)
llvm::Value* CodeGenerator::codegenFixedLengthColVar(const Analyzer::ColumnVar* col_var,
                                                     llvm::Value* col_byte_stream,
                                                     llvm::Value* pos_arg) {
  const auto decoder = get_col_decoder(col_var);
  auto dec_val = decoder->codegenDecode(col_byte_stream, pos_arg, cgen_state_->module_);
  cgen_state_->ir_builder_.Insert(dec_val);
  auto dec_type = dec_val->getType();
  llvm::Value* dec_val_cast{nullptr};
  const auto& col_ti = col_var->get_type_info();
  if (dec_type->isIntegerTy()) {
    auto dec_width = static_cast<llvm::IntegerType*>(dec_type)->getBitWidth();
    auto col_width = get_col_bit_width(col_var);
    dec_val_cast = cgen_state_->ir_builder_.CreateCast(
        static_cast<size_t>(col_width) > dec_width ? llvm::Instruction::CastOps::SExt
                                                   : llvm::Instruction::CastOps::Trunc,
        dec_val,
        get_int_type(col_width, cgen_state_->context_));
    if ((col_ti.get_compression() == kENCODING_FIXED ||
         (col_ti.get_compression() == kENCODING_DICT && col_ti.get_size() < 4)) &&
        !col_ti.get_notnull()) {
      dec_val_cast = codgenAdjustFixedEncNull(dec_val_cast, col_ti);
    }
  } else {
    CHECK_EQ(kENCODING_NONE, col_ti.get_compression());
    CHECK(dec_type->isFloatTy() || dec_type->isDoubleTy());
    if (dec_type->isDoubleTy()) {
      CHECK(col_ti.get_type() == kDOUBLE);
    } else if (dec_type->isFloatTy()) {
      CHECK(col_ti.get_type() == kFLOAT);
    }
    dec_val_cast = dec_val;
  }
  CHECK(dec_val_cast);
  return dec_val_cast;
}

llvm::Value* CodeGenerator::codegenFixedLengthColVarInWindow(
    const Analyzer::ColumnVar* col_var,
    llvm::Value* col_byte_stream,
    llvm::Value* pos_arg) {
  const auto orig_bb = cgen_state_->ir_builder_.GetInsertBlock();
  const auto pos_is_valid =
      cgen_state_->ir_builder_.CreateICmpSGE(pos_arg, cgen_state_->llInt(int64_t(0)));
  const auto pos_valid_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "window.pos_valid", cgen_state_->row_func_);
  const auto pos_notvalid_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "window.pos_notvalid", cgen_state_->row_func_);
  cgen_state_->ir_builder_.CreateCondBr(pos_is_valid, pos_valid_bb, pos_notvalid_bb);
  cgen_state_->ir_builder_.SetInsertPoint(pos_valid_bb);
  const auto fixed_length_column_lv =
      codegenFixedLengthColVar(col_var, col_byte_stream, pos_arg);
  cgen_state_->ir_builder_.CreateBr(pos_notvalid_bb);
  cgen_state_->ir_builder_.SetInsertPoint(pos_notvalid_bb);
  const auto window_func_call_phi =
      cgen_state_->ir_builder_.CreatePHI(fixed_length_column_lv->getType(), 2);
  window_func_call_phi->addIncoming(fixed_length_column_lv, pos_valid_bb);
  const auto& col_ti = col_var->get_type_info();
  const auto null_lv =
      col_ti.is_fp() ? static_cast<llvm::Value*>(cgen_state_->inlineFpNull(col_ti))
                     : static_cast<llvm::Value*>(cgen_state_->inlineIntNull(col_ti));
  window_func_call_phi->addIncoming(null_lv, orig_bb);
  return window_func_call_phi;
}

std::vector<llvm::Value*> CodeGenerator::codegenVariableLengthStringColVar(
    llvm::Value* col_byte_stream,
    llvm::Value* pos_arg) {
  // real (not dictionary-encoded) strings; store the pointer to the payload
  auto ptr_and_len =
      cgen_state_->emitExternalCall("string_decode",
                                    get_int_type(64, cgen_state_->context_),
                                    {col_byte_stream, pos_arg});
  // Unpack the pointer + length, see string_decode function.
  auto str_lv = cgen_state_->emitCall("extract_str_ptr", {ptr_and_len});
  auto len_lv = cgen_state_->emitCall("extract_str_len", {ptr_and_len});
  return {ptr_and_len, str_lv, len_lv};
}

llvm::Value* CodeGenerator::codegenRowId(const Analyzer::ColumnVar* col_var,
                                         const CompilationOptions& co) {
  const auto offset_lv = cgen_state_->frag_offsets_[adjusted_range_table_index(col_var)];
  llvm::Value* start_rowid_lv{nullptr};
  const auto& table_generation = executor()->getTableGeneration(col_var->get_table_id());
  if (table_generation.start_rowid > 0) {
    // Handle the multi-node case: each leaf receives a start rowid used
    // to offset the local rowid and generate a cluster-wide unique rowid.
    Datum d;
    d.bigintval = table_generation.start_rowid;
    const auto start_rowid = makeExpr<Analyzer::Constant>(kBIGINT, false, d);
    const auto start_rowid_lvs = codegen(start_rowid.get(), kENCODING_NONE, -1, co);
    CHECK_EQ(size_t(1), start_rowid_lvs.size());
    start_rowid_lv = start_rowid_lvs.front();
  }
  auto rowid_lv = posArg(col_var);
  if (offset_lv) {
    rowid_lv = cgen_state_->ir_builder_.CreateAdd(rowid_lv, offset_lv);
  } else if (col_var->get_rte_idx() > 0) {
    auto frag_off_ptr = get_arg_by_name(cgen_state_->row_func_, "frag_row_off");
    auto input_off_ptr = cgen_state_->ir_builder_.CreateGEP(
        frag_off_ptr, cgen_state_->llInt(int32_t(col_var->get_rte_idx())));
    auto rowid_offset_lv = cgen_state_->ir_builder_.CreateLoad(input_off_ptr);
    rowid_lv = cgen_state_->ir_builder_.CreateAdd(rowid_lv, rowid_offset_lv);
  }
  if (table_generation.start_rowid > 0) {
    CHECK(start_rowid_lv);
    rowid_lv = cgen_state_->ir_builder_.CreateAdd(rowid_lv, start_rowid_lv);
  }
  return rowid_lv;
}

namespace {

SQLTypes get_phys_int_type(const size_t byte_sz) {
  switch (byte_sz) {
    case 1:
      return kBOOLEAN;
    // TODO: kTINYINT
    case 2:
      return kSMALLINT;
    case 4:
      return kINT;
    case 8:
      return kBIGINT;
    default:
      CHECK(false);
  }
  return kNULLT;
}

}  // namespace

llvm::Value* CodeGenerator::codgenAdjustFixedEncNull(llvm::Value* val,
                                                     const SQLTypeInfo& col_ti) {
  CHECK_LT(col_ti.get_size(), col_ti.get_logical_size());
  const auto col_phys_width = col_ti.get_size() * 8;
  auto from_typename = "int" + std::to_string(col_phys_width) + "_t";
  auto adjusted = cgen_state_->ir_builder_.CreateCast(
      llvm::Instruction::CastOps::Trunc,
      val,
      get_int_type(col_phys_width, cgen_state_->context_));
  if (col_ti.get_compression() == kENCODING_DICT) {
    from_typename = "u" + from_typename;
    llvm::Value* from_null{nullptr};
    switch (col_ti.get_size()) {
      case 1:
        from_null = cgen_state_->llInt(std::numeric_limits<uint8_t>::max());
        break;
      case 2:
        from_null = cgen_state_->llInt(std::numeric_limits<uint16_t>::max());
        break;
      default:
        CHECK(false);
    }
    return cgen_state_->emitCall(
        "cast_" + from_typename + "_to_" + numeric_type_name(col_ti) + "_nullable",
        {adjusted, from_null, cgen_state_->inlineIntNull(col_ti)});
  }
  SQLTypeInfo col_phys_ti(get_phys_int_type(col_ti.get_size()),
                          col_ti.get_dimension(),
                          col_ti.get_scale(),
                          false,
                          kENCODING_NONE,
                          0,
                          col_ti.get_subtype());
  return cgen_state_->emitCall(
      "cast_" + from_typename + "_to_" + numeric_type_name(col_ti) + "_nullable",
      {adjusted,
       cgen_state_->inlineIntNull(col_phys_ti),
       cgen_state_->inlineIntNull(col_ti)});
}

llvm::Value* CodeGenerator::foundOuterJoinMatch(const ssize_t nesting_level) const {
  CHECK_GE(nesting_level, ssize_t(0));
  CHECK_LE(nesting_level,
           static_cast<ssize_t>(cgen_state_->outer_join_match_found_per_level_.size()));
  return cgen_state_->outer_join_match_found_per_level_[nesting_level - 1];
}

std::vector<llvm::Value*> CodeGenerator::codegenOuterJoinNullPlaceholder(
    const Analyzer::ColumnVar* col_var,
    const bool fetch_column,
    const CompilationOptions& co) {
  const auto grouped_col_lv = resolveGroupedColumnReference(col_var);
  if (grouped_col_lv) {
    return {grouped_col_lv};
  }
  const auto bb = cgen_state_->ir_builder_.GetInsertBlock();
  const auto outer_join_args_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "outer_join_args", cgen_state_->row_func_);
  const auto outer_join_nulls_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "outer_join_nulls", cgen_state_->row_func_);
  const auto phi_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "outer_join_phi", cgen_state_->row_func_);
  cgen_state_->ir_builder_.SetInsertPoint(bb);
  const auto outer_join_match_lv = foundOuterJoinMatch(col_var->get_rte_idx());
  CHECK(outer_join_match_lv);
  cgen_state_->ir_builder_.CreateCondBr(
      outer_join_match_lv, outer_join_args_bb, outer_join_nulls_bb);
  const auto back_from_outer_join_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "back_from_outer_join", cgen_state_->row_func_);
  cgen_state_->ir_builder_.SetInsertPoint(outer_join_args_bb);
  Executor::FetchCacheAnchor anchor(cgen_state_);
  const auto orig_lvs = codegenColVar(col_var, fetch_column, true, co);
  cgen_state_->ir_builder_.CreateBr(phi_bb);
  cgen_state_->ir_builder_.SetInsertPoint(outer_join_nulls_bb);
  const auto& null_ti = col_var->get_type_info();
  if ((null_ti.is_string() && null_ti.get_compression() == kENCODING_NONE) ||
      null_ti.is_array() || null_ti.is_geometry()) {
    throw std::runtime_error("Projection type " + null_ti.get_type_name() +
                             " not supported for outer joins yet");
  }
  const auto null_constant = makeExpr<Analyzer::Constant>(null_ti, true, Datum{0});
  const auto null_target_lvs =
      codegen(null_constant.get(),
              false,
              CompilationOptions{
                  ExecutorDeviceType::CPU, false, ExecutorOptLevel::Default, false});
  cgen_state_->ir_builder_.CreateBr(phi_bb);
  CHECK_EQ(orig_lvs.size(), null_target_lvs.size());
  cgen_state_->ir_builder_.SetInsertPoint(phi_bb);
  std::vector<llvm::Value*> target_lvs;
  for (size_t i = 0; i < orig_lvs.size(); ++i) {
    const auto target_type = orig_lvs[i]->getType();
    CHECK_EQ(target_type, null_target_lvs[i]->getType());
    auto target_phi = cgen_state_->ir_builder_.CreatePHI(target_type, 2);
    target_phi->addIncoming(orig_lvs[i], outer_join_args_bb);
    target_phi->addIncoming(null_target_lvs[i], outer_join_nulls_bb);
    target_lvs.push_back(target_phi);
  }
  cgen_state_->ir_builder_.CreateBr(back_from_outer_join_bb);
  cgen_state_->ir_builder_.SetInsertPoint(back_from_outer_join_bb);
  return target_lvs;
}

llvm::Value* CodeGenerator::resolveGroupedColumnReference(
    const Analyzer::ColumnVar* col_var) {
  auto col_id = col_var->get_column_id();
  if (col_var->get_rte_idx() >= 0) {
    return nullptr;
  }
  CHECK((col_id == 0) || (col_var->get_rte_idx() >= 0 && col_var->get_table_id() > 0));
  const auto var = dynamic_cast<const Analyzer::Var*>(col_var);
  CHECK(var);
  col_id = var->get_varno();
  CHECK_GE(col_id, 1);
  if (var->get_which_row() == Analyzer::Var::kGROUPBY) {
    CHECK_LE(static_cast<size_t>(col_id), cgen_state_->group_by_expr_cache_.size());
    return cgen_state_->group_by_expr_cache_[col_id - 1];
  }
  return nullptr;
}

// returns the byte stream argument and the position for the given column
llvm::Value* CodeGenerator::colByteStream(const Analyzer::ColumnVar* col_var,
                                          const bool fetch_column,
                                          const bool hoist_literals) {
  CHECK_GE(cgen_state_->row_func_->arg_size(), size_t(3));
  const auto stream_arg_name =
      "col_buf" + std::to_string(plan_state_->getLocalColumnId(col_var, fetch_column));
  for (auto& arg : cgen_state_->row_func_->args()) {
    if (arg.getName() == stream_arg_name) {
      CHECK(arg.getType() == llvm::Type::getInt8PtrTy(cgen_state_->context_));
      return &arg;
    }
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* CodeGenerator::posArg(const Analyzer::Expr* expr) const {
  const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (col_var && col_var->get_rte_idx() > 0) {
    const auto hash_pos_it =
        cgen_state_->scan_idx_to_hash_pos_.find(col_var->get_rte_idx());
    CHECK(hash_pos_it != cgen_state_->scan_idx_to_hash_pos_.end());
    if (hash_pos_it->second->getType()->isPointerTy()) {
      CHECK(hash_pos_it->second->getType()->getPointerElementType()->isIntegerTy(32));
      llvm::Value* result = cgen_state_->ir_builder_.CreateLoad(hash_pos_it->second);
      result = cgen_state_->ir_builder_.CreateSExt(
          result, get_int_type(64, cgen_state_->context_));
      return result;
    }
    return hash_pos_it->second;
  }
  for (auto& arg : cgen_state_->row_func_->args()) {
    if (arg.getName() == "pos") {
      CHECK(arg.getType()->isIntegerTy(64));
      return &arg;
    }
  }
  abort();
}

const Analyzer::Expr* remove_cast_to_int(const Analyzer::Expr* expr) {
  const auto uoper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (!uoper || uoper->get_optype() != kCAST) {
    return nullptr;
  }
  const auto& target_ti = uoper->get_type_info();
  if (!target_ti.is_integer()) {
    return nullptr;
  }
  return uoper->get_operand();
}

std::shared_ptr<const Analyzer::Expr> CodeGenerator::hashJoinLhs(
    const Analyzer::ColumnVar* rhs) const {
  for (const auto tautological_eq : plan_state_->join_info_.equi_join_tautologies_) {
    CHECK(IS_EQUIVALENCE(tautological_eq->get_optype()));
    if (dynamic_cast<const Analyzer::ExpressionTuple*>(
            tautological_eq->get_left_operand())) {
      auto lhs_col = hashJoinLhsTuple(rhs, tautological_eq.get());
      if (lhs_col) {
        return lhs_col;
      }
    } else {
      auto eq_right_op = tautological_eq->get_right_operand();
      if (!rhs->get_type_info().is_string()) {
        eq_right_op = remove_cast_to_int(eq_right_op);
      }
      if (!eq_right_op) {
        eq_right_op = tautological_eq->get_right_operand();
      }
      if (*eq_right_op == *rhs) {
        auto eq_left_op = tautological_eq->get_left_operand();
        if (!eq_left_op->get_type_info().is_string()) {
          eq_left_op = remove_cast_to_int(eq_left_op);
        }
        if (!eq_left_op) {
          eq_left_op = tautological_eq->get_left_operand();
        }
        if (eq_left_op->get_type_info().is_geometry()) {
          // skip cast for a geospatial lhs, since the rhs is likely to be a geospatial
          // physical col without geospatial type info
          return nullptr;
        }
        const auto eq_left_op_col = dynamic_cast<const Analyzer::ColumnVar*>(eq_left_op);
        CHECK(eq_left_op_col);
        if (eq_left_op_col->get_rte_idx() != 0) {
          return nullptr;
        }
        if (rhs->get_type_info().is_string()) {
          return eq_left_op->deep_copy();
        }
        return makeExpr<Analyzer::UOper>(
            rhs->get_type_info(), false, kCAST, eq_left_op->deep_copy());
      }
    }
  }
  return nullptr;
}

std::shared_ptr<const Analyzer::ColumnVar> CodeGenerator::hashJoinLhsTuple(
    const Analyzer::ColumnVar* rhs,
    const Analyzer::BinOper* tautological_eq) const {
  const auto lhs_tuple_expr =
      dynamic_cast<const Analyzer::ExpressionTuple*>(tautological_eq->get_left_operand());
  const auto rhs_tuple_expr = dynamic_cast<const Analyzer::ExpressionTuple*>(
      tautological_eq->get_right_operand());
  CHECK(lhs_tuple_expr && rhs_tuple_expr);
  const auto& lhs_tuple = lhs_tuple_expr->getTuple();
  const auto& rhs_tuple = rhs_tuple_expr->getTuple();
  CHECK_EQ(lhs_tuple.size(), rhs_tuple.size());
  for (size_t i = 0; i < lhs_tuple.size(); ++i) {
    if (*rhs_tuple[i] == *rhs) {
      const auto lhs_col =
          std::static_pointer_cast<const Analyzer::ColumnVar>(lhs_tuple[i]);
      return lhs_col->get_rte_idx() == 0 ? lhs_col : nullptr;
    }
  }
  return nullptr;
}
