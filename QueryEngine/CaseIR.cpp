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

#include "CodeGenerator.h"
#include "Execute.h"

std::vector<llvm::Value*> CodeGenerator::codegen(const Analyzer::CaseExpr* case_expr,
                                                 const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto case_ti = case_expr->get_type_info();
  llvm::Type* case_llvm_type = nullptr;
  bool is_real_str = false;
  if (case_ti.is_integer() || case_ti.is_time() || case_ti.is_decimal()) {
    case_llvm_type = get_int_type(get_bit_width(case_ti), cgen_state_->context_);
  } else if (case_ti.is_fp()) {
    case_llvm_type = case_ti.get_type() == kFLOAT
                         ? llvm::Type::getFloatTy(cgen_state_->context_)
                         : llvm::Type::getDoubleTy(cgen_state_->context_);
  } else if (case_ti.is_string()) {
    if (case_ti.get_compression() == kENCODING_DICT) {
      case_llvm_type =
          get_int_type(8 * case_ti.get_logical_size(), cgen_state_->context_);
    } else {
      is_real_str = true;
      case_llvm_type = createStringViewStructType();
    }
  } else if (case_ti.is_boolean()) {
    case_llvm_type = get_int_type(8 * case_ti.get_logical_size(), cgen_state_->context_);
  } else if (case_ti.is_geometry()) {
    throw std::runtime_error(
        "Geospatial column projections are currently not supported in conditional "
        "expressions.");
  } else if (case_ti.is_array()) {
    throw std::runtime_error(
        "Array column projections are currently not supported in conditional "
        "expressions.");
  }
  CHECK(case_llvm_type);
  const auto& else_ti = case_expr->get_else_expr()->get_type_info();
  CHECK_EQ(else_ti.get_type(), case_ti.get_type());
  llvm::Value* case_val = codegenCase(case_expr, case_llvm_type, is_real_str, co);
  std::vector<llvm::Value*> ret_vals{case_val};
  if (is_real_str) {
    ret_vals.push_back(cgen_state_->ir_builder_.CreateExtractValue(case_val, 0));
    ret_vals.push_back(cgen_state_->ir_builder_.CreateExtractValue(case_val, 1));
    ret_vals.back() = cgen_state_->ir_builder_.CreateTrunc(
        ret_vals.back(), llvm::Type::getInt32Ty(cgen_state_->context_));
  }
  return ret_vals;
}

llvm::Value* CodeGenerator::codegenCase(const Analyzer::CaseExpr* case_expr,
                                        llvm::Type* case_llvm_type,
                                        const bool is_real_str,
                                        const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  // Here the linear control flow will diverge and expressions cached during the
  // code branch code generation (currently just column decoding) are not going
  // to be available once we're done generating the case. Take a snapshot of
  // the cache with FetchCacheAnchor and restore it once we're done with CASE.
  Executor::FetchCacheAnchor anchor(cgen_state_);
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  std::vector<llvm::Value*> then_lvs;
  std::vector<llvm::BasicBlock*> then_bbs;
  const auto end_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "end_case", cgen_state_->current_func_);
  for (const auto& expr_pair : expr_pair_list) {
    Executor::FetchCacheAnchor branch_anchor(cgen_state_);
    const auto when_lv = toBool(codegen(expr_pair.first.get(), true, co).front());
    const auto cmp_bb = cgen_state_->ir_builder_.GetInsertBlock();
    const auto then_bb = llvm::BasicBlock::Create(cgen_state_->context_,
                                                  "then_case",
                                                  cgen_state_->current_func_,
                                                  /*insert_before=*/end_bb);
    cgen_state_->ir_builder_.SetInsertPoint(then_bb);
    auto then_bb_lvs = codegen(expr_pair.second.get(), true, co);
    if (is_real_str) {
      if (then_bb_lvs.size() == 3) {
        then_lvs.push_back(
            cgen_state_->emitCall("string_pack", {then_bb_lvs[1], then_bb_lvs[2]}));
      } else {
        then_lvs.push_back(then_bb_lvs.front());
      }
    } else {
      CHECK_EQ(size_t(1), then_bb_lvs.size());
      then_lvs.push_back(then_bb_lvs.front());
    }
    then_bbs.push_back(cgen_state_->ir_builder_.GetInsertBlock());
    cgen_state_->ir_builder_.CreateBr(end_bb);
    const auto when_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "when_case", cgen_state_->current_func_);
    cgen_state_->ir_builder_.SetInsertPoint(cmp_bb);
    cgen_state_->ir_builder_.CreateCondBr(when_lv, then_bb, when_bb);
    cgen_state_->ir_builder_.SetInsertPoint(when_bb);
  }
  const auto else_expr = case_expr->get_else_expr();
  CHECK(else_expr);
  auto else_lvs = codegen(else_expr, true, co);
  llvm::Value* else_lv{nullptr};
  if (else_lvs.size() == 3) {
    else_lv = cgen_state_->emitCall("string_pack", {else_lvs[1], else_lvs[2]});
  } else {
    else_lv = else_lvs.front();
  }
  CHECK(else_lv);
  auto else_bb = cgen_state_->ir_builder_.GetInsertBlock();
  cgen_state_->ir_builder_.CreateBr(end_bb);
  cgen_state_->ir_builder_.SetInsertPoint(end_bb);
  auto then_phi =
      cgen_state_->ir_builder_.CreatePHI(case_llvm_type, expr_pair_list.size() + 1);
  CHECK_EQ(then_bbs.size(), then_lvs.size());
  for (size_t i = 0; i < then_bbs.size(); ++i) {
    then_phi->addIncoming(then_lvs[i], then_bbs[i]);
  }
  then_phi->addIncoming(else_lv, else_bb);
  return then_phi;
}
