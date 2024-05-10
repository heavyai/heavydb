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
#include "GeoOperators/Codegen.h"
#include "NullableValue.h"

#include <llvm/IR/MDBuilder.h>

namespace {

bool contains_unsafe_division(const Analyzer::Expr* expr) {
  auto is_div = [](const Analyzer::Expr* e) -> bool {
    auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(e);
    if (bin_oper && bin_oper->get_optype() == kDIVIDE) {
      auto rhs = bin_oper->get_right_operand();
      auto rhs_constant = dynamic_cast<const Analyzer::Constant*>(rhs);
      if (!rhs_constant || rhs_constant->get_is_null()) {
        return true;
      }
      const auto& datum = rhs_constant->get_constval();
      const auto& ti = rhs_constant->get_type_info();
      const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
      if ((type == kBOOLEAN && datum.boolval == 0) ||
          (type == kTINYINT && datum.tinyintval == 0) ||
          (type == kSMALLINT && datum.smallintval == 0) ||
          (type == kINT && datum.intval == 0) ||
          (type == kBIGINT && datum.bigintval == 0LL) ||
          (type == kFLOAT && datum.floatval == 0.0) ||
          (type == kDOUBLE && datum.doubleval == 0.0)) {
        return true;
      }
    }
    return false;
  };
  std::list<const Analyzer::Expr*> binoper_list;
  expr->find_expr(is_div, binoper_list);
  return !binoper_list.empty();
}

bool should_defer_eval(const std::shared_ptr<Analyzer::Expr> expr) {
  if (std::dynamic_pointer_cast<Analyzer::LikeExpr>(expr)) {
    return true;
  }
  if (std::dynamic_pointer_cast<Analyzer::RegexpExpr>(expr)) {
    return true;
  }
  if (std::dynamic_pointer_cast<Analyzer::FunctionOper>(expr)) {
    return true;
  }
  if (!std::dynamic_pointer_cast<Analyzer::BinOper>(expr)) {
    return false;
  }
  const auto bin_expr = std::static_pointer_cast<Analyzer::BinOper>(expr);
  if (contains_unsafe_division(bin_expr.get())) {
    return true;
  }
  if (bin_expr->is_bbox_intersect_oper()) {
    return false;
  }
  const auto rhs = bin_expr->get_right_operand();
  return rhs->get_type_info().is_array();
}

Likelihood get_likelihood(const Analyzer::Expr* expr) {
  Likelihood truth{1.0};
  auto likelihood_expr = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
  if (likelihood_expr) {
    return Likelihood(likelihood_expr->get_likelihood());
  }
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_oper) {
    Likelihood oper_likelihood = get_likelihood(u_oper->get_operand());
    if (oper_likelihood.isInvalid()) {
      return Likelihood();
    }
    if (u_oper->get_optype() == kNOT) {
      return truth - oper_likelihood;
    }
    return oper_likelihood;
  }
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    auto lhs = bin_oper->get_left_operand();
    auto rhs = bin_oper->get_right_operand();
    Likelihood lhs_likelihood = get_likelihood(lhs);
    Likelihood rhs_likelihood = get_likelihood(rhs);
    if (lhs_likelihood.isInvalid() && rhs_likelihood.isInvalid()) {
      return Likelihood();
    }
    const auto optype = bin_oper->get_optype();
    if (optype == kOR) {
      auto both_false = (truth - lhs_likelihood) * (truth - rhs_likelihood);
      return truth - both_false;
    }
    if (optype == kAND) {
      return lhs_likelihood * rhs_likelihood;
    }
    return (lhs_likelihood + rhs_likelihood) / 2.0;
  }

  return Likelihood();
}

Weight get_weight(const Analyzer::Expr* expr, int depth = 0) {
  auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
  if (like_expr) {
    // heavy weight expr, start valid weight propagation
    return Weight((like_expr->get_is_simple()) ? 200 : 1000);
  }
  auto regexp_expr = dynamic_cast<const Analyzer::RegexpExpr*>(expr);
  if (regexp_expr) {
    // heavy weight expr, start valid weight propagation
    return Weight(2000);
  }
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_oper) {
    auto weight = get_weight(u_oper->get_operand(), depth + 1);
    return weight + 1;
  }
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    auto lhs = bin_oper->get_left_operand();
    auto rhs = bin_oper->get_right_operand();
    auto lhs_weight = get_weight(lhs, depth + 1);
    auto rhs_weight = get_weight(rhs, depth + 1);
    if (rhs->get_type_info().is_array()) {
      // heavy weight expr, start valid weight propagation
      rhs_weight = rhs_weight + Weight(100);
    }
    auto weight = lhs_weight + rhs_weight;
    return weight + 1;
  }

  if (depth > 4) {
    return Weight(1);
  }

  return Weight();
}

}  // namespace

bool CodeGenerator::prioritizeQuals(const RelAlgExecutionUnit& ra_exe_unit,
                                    std::vector<Analyzer::Expr*>& primary_quals,
                                    std::vector<Analyzer::Expr*>& deferred_quals,
                                    const PlanState::HoistedFiltersSet& hoisted_quals) {
  for (auto expr : ra_exe_unit.simple_quals) {
    if (hoisted_quals.find(expr) != hoisted_quals.end()) {
      continue;
    }
    if (should_defer_eval(expr)) {
      deferred_quals.push_back(expr.get());
      continue;
    }
    primary_quals.push_back(expr.get());
  }

  bool short_circuit = false;

  for (auto expr : ra_exe_unit.quals) {
    if (hoisted_quals.find(expr) != hoisted_quals.end()) {
      continue;
    }

    if (get_likelihood(expr.get()) < 0.10 && !contains_unsafe_division(expr.get())) {
      if (!short_circuit) {
        primary_quals.push_back(expr.get());
        short_circuit = true;
        continue;
      }
    }
    if (short_circuit || should_defer_eval(expr)) {
      deferred_quals.push_back(expr.get());
      continue;
    }
    primary_quals.push_back(expr.get());
  }

  return short_circuit;
}

llvm::Value* CodeGenerator::codegenLogicalShortCircuit(const Analyzer::BinOper* bin_oper,
                                                       const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto optype = bin_oper->get_optype();
  auto lhs = bin_oper->get_left_operand();
  auto rhs = bin_oper->get_right_operand();

  if (contains_unsafe_division(rhs)) {
    // rhs contains a possible div-by-0: short-circuit
  } else if (contains_unsafe_division(lhs)) {
    // lhs contains a possible div-by-0: swap and short-circuit
    std::swap(rhs, lhs);
  } else if (((optype == kOR && get_likelihood(lhs) > 0.90) ||
              (optype == kAND && get_likelihood(lhs) < 0.10)) &&
             get_weight(rhs) > 10) {
    // short circuit if we're likely to see either (trueA || heavyB) or (falseA && heavyB)
  } else if (((optype == kOR && get_likelihood(rhs) > 0.90) ||
              (optype == kAND && get_likelihood(rhs) < 0.10)) &&
             get_weight(lhs) > 10) {
    // swap and short circuit if we're likely to see either (heavyA || trueB) or (heavyA
    // && falseB)
    std::swap(rhs, lhs);
  } else {
    // no motivation to short circuit
    return nullptr;
  }

  const auto& ti = bin_oper->get_type_info();
  auto lhs_lv = codegen(lhs, true, co).front();

  // Here the linear control flow will diverge and expressions cached during the
  // code branch code generation (currently just column decoding) are not going
  // to be available once we're done generating the short-circuited logic.
  // Take a snapshot of the cache with FetchCacheAnchor and restore it once
  // the control flow converges.
  Executor::FetchCacheAnchor anchor(cgen_state_);

  auto rhs_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "rhs_bb", cgen_state_->current_func_);
  auto ret_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "ret_bb", cgen_state_->current_func_);
  llvm::BasicBlock* nullcheck_ok_bb{nullptr};
  llvm::BasicBlock* nullcheck_fail_bb{nullptr};

  if (!ti.get_notnull()) {
    // need lhs nullcheck before short circuiting
    nullcheck_ok_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "nullcheck_ok_bb", cgen_state_->current_func_);
    nullcheck_fail_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "nullcheck_fail_bb", cgen_state_->current_func_);
    if (lhs_lv->getType()->isIntegerTy(1)) {
      lhs_lv = cgen_state_->castToTypeIn(lhs_lv, 8);
    }
    auto lhs_nullcheck =
        cgen_state_->ir_builder_.CreateICmpEQ(lhs_lv, cgen_state_->inlineIntNull(ti));
    cgen_state_->ir_builder_.CreateCondBr(
        lhs_nullcheck, nullcheck_fail_bb, nullcheck_ok_bb);
    cgen_state_->ir_builder_.SetInsertPoint(nullcheck_ok_bb);
  }

  auto sc_check_bb = cgen_state_->ir_builder_.GetInsertBlock();
  auto cnst_lv = llvm::ConstantInt::get(lhs_lv->getType(), (optype == kOR));
  // Branch to codegen rhs if NOT getting (true || rhs) or (false && rhs), likelihood of
  // the branch is < 0.10
  cgen_state_->ir_builder_.CreateCondBr(
      cgen_state_->ir_builder_.CreateICmpNE(lhs_lv, cnst_lv),
      rhs_bb,
      ret_bb,
      llvm::MDBuilder(cgen_state_->context_).createBranchWeights(10, 90));

  // Codegen rhs when unable to short circuit.
  cgen_state_->ir_builder_.SetInsertPoint(rhs_bb);
  auto rhs_lv = codegen(rhs, true, co).front();
  if (!ti.get_notnull()) {
    // need rhs nullcheck as well
    if (rhs_lv->getType()->isIntegerTy(1)) {
      rhs_lv = cgen_state_->castToTypeIn(rhs_lv, 8);
    }
    auto rhs_nullcheck =
        cgen_state_->ir_builder_.CreateICmpEQ(rhs_lv, cgen_state_->inlineIntNull(ti));
    cgen_state_->ir_builder_.CreateCondBr(rhs_nullcheck, nullcheck_fail_bb, ret_bb);
  } else {
    cgen_state_->ir_builder_.CreateBr(ret_bb);
  }
  auto rhs_codegen_bb = cgen_state_->ir_builder_.GetInsertBlock();

  if (!ti.get_notnull()) {
    cgen_state_->ir_builder_.SetInsertPoint(nullcheck_fail_bb);
    cgen_state_->ir_builder_.CreateBr(ret_bb);
  }

  cgen_state_->ir_builder_.SetInsertPoint(ret_bb);
  auto result_phi =
      cgen_state_->ir_builder_.CreatePHI(lhs_lv->getType(), (!ti.get_notnull()) ? 3 : 2);
  if (!ti.get_notnull()) {
    result_phi->addIncoming(cgen_state_->inlineIntNull(ti), nullcheck_fail_bb);
  }
  result_phi->addIncoming(cnst_lv, sc_check_bb);
  result_phi->addIncoming(rhs_lv, rhs_codegen_bb);
  return result_phi;
}

llvm::Value* CodeGenerator::codegenLogical(const Analyzer::BinOper* bin_oper,
                                           const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto optype = bin_oper->get_optype();
  CHECK(IS_LOGIC(optype));

  if (llvm::Value* short_circuit = codegenLogicalShortCircuit(bin_oper, co)) {
    return short_circuit;
  }

  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  auto lhs_lv = codegen(lhs, true, co).front();
  auto rhs_lv = codegen(rhs, true, co).front();
  const auto& ti = bin_oper->get_type_info();
  if (ti.get_notnull()) {
    switch (optype) {
      case kAND:
        return cgen_state_->ir_builder_.CreateAnd(toBool(lhs_lv), toBool(rhs_lv));
      case kOR:
        return cgen_state_->ir_builder_.CreateOr(toBool(lhs_lv), toBool(rhs_lv));
      default:
        CHECK(false);
    }
  }
  CHECK(lhs_lv->getType()->isIntegerTy(1) || lhs_lv->getType()->isIntegerTy(8));
  CHECK(rhs_lv->getType()->isIntegerTy(1) || rhs_lv->getType()->isIntegerTy(8));
  if (lhs_lv->getType()->isIntegerTy(1)) {
    lhs_lv = cgen_state_->castToTypeIn(lhs_lv, 8);
  }
  if (rhs_lv->getType()->isIntegerTy(1)) {
    rhs_lv = cgen_state_->castToTypeIn(rhs_lv, 8);
  }
  switch (optype) {
    case kAND:
      return cgen_state_->emitCall("logical_and",
                                   {lhs_lv, rhs_lv, cgen_state_->inlineIntNull(ti)});
    case kOR:
      return cgen_state_->emitCall("logical_or",
                                   {lhs_lv, rhs_lv, cgen_state_->inlineIntNull(ti)});
    default:
      abort();
  }
}

llvm::Value* CodeGenerator::toBool(llvm::Value* lv) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(lv->getType()->isIntegerTy());
  if (static_cast<llvm::IntegerType*>(lv->getType())->getBitWidth() > 1) {
    return cgen_state_->ir_builder_.CreateICmp(
        llvm::ICmpInst::ICMP_SGT, lv, llvm::ConstantInt::get(lv->getType(), 0));
  }
  return lv;
}

namespace {

bool is_qualified_bin_oper(const Analyzer::Expr* expr) {
  const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  return bin_oper && bin_oper->get_qualifier() != kONE;
}

}  // namespace

llvm::Value* CodeGenerator::codegenLogical(const Analyzer::UOper* uoper,
                                           const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto optype = uoper->get_optype();
  CHECK_EQ(kNOT, optype);
  const auto operand = uoper->get_operand();
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_boolean());
  const auto operand_lv = codegen(operand, true, co).front();
  CHECK(operand_lv->getType()->isIntegerTy());
  const bool not_null = (operand_ti.get_notnull() || is_qualified_bin_oper(operand));
  CHECK(not_null || operand_lv->getType()->isIntegerTy(8));
  return not_null
             ? cgen_state_->ir_builder_.CreateNot(toBool(operand_lv))
             : cgen_state_->emitCall(
                   "logical_not", {operand_lv, cgen_state_->inlineIntNull(operand_ti)});
}

llvm::Value* CodeGenerator::codegenIsNull(const Analyzer::UOper* uoper,
                                          const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto operand = uoper->get_operand();
  if (dynamic_cast<const Analyzer::Constant*>(operand) &&
      dynamic_cast<const Analyzer::Constant*>(operand)->get_is_null()) {
    // for null constants, short-circuit to true
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 1);
  }
  const auto& ti = operand->get_type_info();
  CHECK(ti.is_integer() || ti.is_boolean() || ti.is_decimal() || ti.is_time() ||
        ti.is_string() || ti.is_fp() || ti.is_array() || ti.is_geometry());
  // if the type is inferred as non null, short-circuit to false
  if (ti.get_notnull()) {
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 0);
  }
  llvm::Value* operand_lv = codegen(operand, true, co).front();
  // NULL-check array or geo's coords array
  if (ti.get_type() == kPOINT && dynamic_cast<Analyzer::GeoOperator const*>(operand)) {
    char const* const fname = spatial_type::Codegen::pointIsNullFunctionName(ti);
    return cgen_state_->emitCall(fname, {operand_lv});
  } else if (ti.is_array() || ti.is_geometry()) {
    // POINT [un]compressed coord check requires custom checker and chunk iterator
    // Non-POINT NULL geographies will have a normally encoded null coord array
    auto fname =
        (ti.get_type() == kPOINT) ? "point_coord_array_is_null" : "array_is_null";
    return cgen_state_->emitExternalCall(
        fname, get_int_type(1, cgen_state_->context_), {operand_lv, posArg(operand)});
  } else if (ti.is_none_encoded_string()) {
    operand_lv = cgen_state_->ir_builder_.CreateExtractValue(operand_lv, 0);
    operand_lv = cgen_state_->castToTypeIn(operand_lv, sizeof(int64_t) * 8);
  }
  return codegenIsNullNumber(operand_lv, ti);
}

llvm::Value* CodeGenerator::codegenIsNullNumber(llvm::Value* operand_lv,
                                                const SQLTypeInfo& ti) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (ti.is_fp()) {
    return cgen_state_->ir_builder_.CreateFCmp(llvm::FCmpInst::FCMP_OEQ,
                                               operand_lv,
                                               ti.get_type() == kFLOAT
                                                   ? cgen_state_->llFp(NULL_FLOAT)
                                                   : cgen_state_->llFp(NULL_DOUBLE));
  }
  return cgen_state_->ir_builder_.CreateICmp(
      llvm::ICmpInst::ICMP_EQ, operand_lv, cgen_state_->inlineIntNull(ti));
}
