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

#include "Execute.h"
#include "NullableValue.h"

#include <llvm/IR/MDBuilder.h>

namespace {

bool contains_unsafe_division(const Analyzer::Expr* expr) {
  auto is_div = [](const Analyzer::Expr* e) -> bool {
    auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(e);
    if (bin_oper && bin_oper->get_optype() == kDIVIDE) {
      auto rhs = bin_oper->get_right_operand();
      auto rhs_constant = dynamic_cast<const Analyzer::Constant*>(rhs);
      if (!rhs_constant || rhs_constant->get_is_null())
        return true;
      const auto& datum = rhs_constant->get_constval();
      const auto& ti = rhs_constant->get_type_info();
      const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
      if ((type == kBOOLEAN && datum.boolval == 0) || (type == kSMALLINT && datum.smallintval == 0) ||
          (type == kINT && datum.intval == 0) || (type == kBIGINT && datum.bigintval == 0LL) ||
          (type == kFLOAT && datum.floatval == 0.0) || (type == kDOUBLE && datum.doubleval == 0.0)) {
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
  if (!std::dynamic_pointer_cast<Analyzer::BinOper>(expr)) {
    return false;
  }
  const auto bin_expr = std::static_pointer_cast<Analyzer::BinOper>(expr);
  if (contains_unsafe_division(bin_expr.get())) {
    return true;
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
    if (oper_likelihood.isInvalid())
      return Likelihood();
    if (u_oper->get_optype() == kNOT)
      return truth - oper_likelihood;
    return oper_likelihood;
  }
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    auto lhs = bin_oper->get_left_operand();
    auto rhs = bin_oper->get_right_operand();
    Likelihood lhs_likelihood = get_likelihood(lhs);
    Likelihood rhs_likelihood = get_likelihood(rhs);
    if (lhs_likelihood.isInvalid() && rhs_likelihood.isInvalid())
      return Likelihood();
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

  if (depth > 4)
    return Weight(1);

  return Weight();
}

void sort_eq_joins_by_rte_indices(std::vector<Analyzer::Expr*>& join_conditions) {
  auto cmp = [](const Analyzer::Expr* lhs, const Analyzer::Expr* rhs) -> bool {
    auto lhs_eq = dynamic_cast<const Analyzer::BinOper*>(lhs);
    auto rhs_eq = dynamic_cast<const Analyzer::BinOper*>(rhs);
    CHECK(lhs_eq && IS_EQUIVALENCE(lhs_eq->get_optype()));
    CHECK(rhs_eq && IS_EQUIVALENCE(rhs_eq->get_optype()));
    std::set<int> rte_idx_set;
    lhs_eq->collect_rte_idx(rte_idx_set);
    CHECK_EQ(rte_idx_set.size(), size_t(2));
    auto lhs_outer_rte = *rte_idx_set.begin();
    auto lhs_inner_rtx = *std::next(rte_idx_set.begin());
    CHECK_LT(lhs_outer_rte, lhs_inner_rtx);

    rte_idx_set.clear();
    rhs_eq->collect_rte_idx(rte_idx_set);
    CHECK_EQ(rte_idx_set.size(), size_t(2));
    auto rhs_outer_rte = *rte_idx_set.begin();
    auto rhs_inner_rtx = *std::next(rte_idx_set.begin());
    if (lhs_outer_rte == rhs_outer_rte) {
      return lhs_inner_rtx < rhs_inner_rtx;
    }
    return lhs_outer_rte < rhs_outer_rte;
  };

  std::sort(join_conditions.begin(), join_conditions.end(), cmp);
}

}  // namespace

bool Executor::prioritizeQuals(const RelAlgExecutionUnit& ra_exe_unit,
                               std::vector<Analyzer::Expr*>& primary_quals,
                               std::vector<Analyzer::Expr*>& deferred_quals) {
  std::vector<std::shared_ptr<Analyzer::Expr>> remaining_inner_join_quals;
  std::unordered_set<Analyzer::Expr*> equi_join_conds;
  for (auto cond : plan_state_->join_info_.equi_join_tautologies_) {
    equi_join_conds.insert(cond.get());
  }
  for (auto expr : ra_exe_unit.inner_join_quals) {
    if (auto bin_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(expr)) {
      if (equi_join_conds.count(bin_oper.get())) {
        primary_quals.push_back(expr.get());
        continue;
      }
    }
    remaining_inner_join_quals.push_back(expr);
  }
  sort_eq_joins_by_rte_indices(primary_quals);

  for (auto expr : remaining_inner_join_quals) {
    primary_quals.push_back(expr.get());
  }

  for (auto expr : ra_exe_unit.simple_quals) {
    if (should_defer_eval(expr)) {
      deferred_quals.push_back(expr.get());
      continue;
    }
    primary_quals.push_back(expr.get());
  }

  bool short_circuit = false;

  for (auto expr : ra_exe_unit.quals) {
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

llvm::Value* Executor::codegenLogicalShortCircuit(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  auto lhs = bin_oper->get_left_operand();
  auto rhs = bin_oper->get_right_operand();

  if (contains_unsafe_division(rhs)) {
    // rhs contains a possible div-by-0: short-circuit
  } else if (contains_unsafe_division(lhs)) {
    // lhs contains a possible div-by-0: swap and short-circuit
    std::swap(rhs, lhs);
  } else if (((optype == kOR && get_likelihood(lhs) > 0.90) || (optype == kAND && get_likelihood(lhs) < 0.10)) &&
             get_weight(rhs) > 10) {
    // short circuit if we're likely to see either (trueA || heavyB) or (falseA && heavyB)
  } else if (((optype == kOR && get_likelihood(rhs) > 0.90) || (optype == kAND && get_likelihood(rhs) < 0.10)) &&
             get_weight(lhs) > 10) {
    // swap and short circuit if we're likely to see either (heavyA || trueB) or (heavyA && falseB)
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
  FetchCacheAnchor anchor(cgen_state_.get());

  auto rhs_bb = llvm::BasicBlock::Create(cgen_state_->context_, "rhs_bb", cgen_state_->row_func_);
  auto ret_bb = llvm::BasicBlock::Create(cgen_state_->context_, "ret_bb", cgen_state_->row_func_);
  llvm::BasicBlock* nullcheck_ok_bb{nullptr};
  llvm::BasicBlock* nullcheck_fail_bb{nullptr};

  if (!ti.get_notnull()) {
    // need lhs nullcheck before short circuiting
    nullcheck_ok_bb = llvm::BasicBlock::Create(cgen_state_->context_, "nullcheck_ok_bb", cgen_state_->row_func_);
    nullcheck_fail_bb = llvm::BasicBlock::Create(cgen_state_->context_, "nullcheck_fail_bb", cgen_state_->row_func_);
    if (lhs_lv->getType()->isIntegerTy(1)) {
      lhs_lv = castToTypeIn(lhs_lv, 8);
    }
    auto lhs_nullcheck = cgen_state_->ir_builder_.CreateICmpEQ(lhs_lv, inlineIntNull(ti));
    cgen_state_->ir_builder_.CreateCondBr(lhs_nullcheck, nullcheck_fail_bb, nullcheck_ok_bb);
    cgen_state_->ir_builder_.SetInsertPoint(nullcheck_ok_bb);
  }

  auto sc_check_bb = cgen_state_->ir_builder_.GetInsertBlock();
  auto cnst_lv = llvm::ConstantInt::get(lhs_lv->getType(), (optype == kOR));
  // Branch to codegen rhs if NOT getting (true || rhs) or (false && rhs), likelihood of the branch is < 0.10
  cgen_state_->ir_builder_.CreateCondBr(cgen_state_->ir_builder_.CreateICmpNE(lhs_lv, cnst_lv),
                                        rhs_bb,
                                        ret_bb,
                                        llvm::MDBuilder(cgen_state_->context_).createBranchWeights(10, 90));

  // Codegen rhs when unable to short circuit.
  cgen_state_->ir_builder_.SetInsertPoint(rhs_bb);
  auto rhs_lv = codegen(rhs, true, co).front();
  if (!ti.get_notnull()) {
    // need rhs nullcheck as well
    if (rhs_lv->getType()->isIntegerTy(1)) {
      rhs_lv = castToTypeIn(rhs_lv, 8);
    }
    auto rhs_nullcheck = cgen_state_->ir_builder_.CreateICmpEQ(rhs_lv, inlineIntNull(ti));
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
  auto result_phi = cgen_state_->ir_builder_.CreatePHI(lhs_lv->getType(), (!ti.get_notnull()) ? 3 : 2);
  if (!ti.get_notnull())
    result_phi->addIncoming(inlineIntNull(ti), nullcheck_fail_bb);
  result_phi->addIncoming(cnst_lv, sc_check_bb);
  result_phi->addIncoming(rhs_lv, rhs_codegen_bb);
  return result_phi;
}

llvm::Value* Executor::codegenLogical(const Analyzer::BinOper* bin_oper, const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  CHECK(IS_LOGIC(optype));

  if (llvm::Value* short_circuit = codegenLogicalShortCircuit(bin_oper, co))
    return short_circuit;

  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  auto lhs_lv = codegen(lhs, true, co).front();
  auto rhs_lv = codegen(rhs, true, co).front();
  const auto& ti = bin_oper->get_type_info();
  if (ti.get_notnull()) {
    switch (optype) {
      case kAND:
        return cgen_state_->ir_builder_.CreateAnd(lhs_lv, rhs_lv);
      case kOR:
        return cgen_state_->ir_builder_.CreateOr(lhs_lv, rhs_lv);
      default:
        CHECK(false);
    }
  }
  CHECK(lhs_lv->getType()->isIntegerTy(1) || lhs_lv->getType()->isIntegerTy(8));
  CHECK(rhs_lv->getType()->isIntegerTy(1) || rhs_lv->getType()->isIntegerTy(8));
  if (lhs_lv->getType()->isIntegerTy(1)) {
    lhs_lv = castToTypeIn(lhs_lv, 8);
  }
  if (rhs_lv->getType()->isIntegerTy(1)) {
    rhs_lv = castToTypeIn(rhs_lv, 8);
  }
  switch (optype) {
    case kAND:
      return cgen_state_->emitCall("logical_and", {lhs_lv, rhs_lv, inlineIntNull(ti)});
    case kOR:
      return cgen_state_->emitCall("logical_or", {lhs_lv, rhs_lv, inlineIntNull(ti)});
    default:
      abort();
  }
}

llvm::Value* Executor::toBool(llvm::Value* lv) {
  CHECK(lv->getType()->isIntegerTy());
  if (static_cast<llvm::IntegerType*>(lv->getType())->getBitWidth() > 1) {
    return cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_SGT, lv, llvm::ConstantInt::get(lv->getType(), 0));
  }
  return lv;
}

namespace {

bool is_qualified_bin_oper(const Analyzer::Expr* expr) {
  const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  return bin_oper && bin_oper->get_qualifier() != kONE;
}

}  // namespace

llvm::Value* Executor::codegenLogical(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  const auto optype = uoper->get_optype();
  CHECK_EQ(kNOT, optype);
  const auto operand = uoper->get_operand();
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_boolean());
  const auto operand_lv = codegen(operand, true, co).front();
  CHECK(operand_lv->getType()->isIntegerTy());
  const bool not_null = (operand_ti.get_notnull() || is_qualified_bin_oper(operand));
  CHECK(not_null || operand_lv->getType()->isIntegerTy(8));
  return not_null ? cgen_state_->ir_builder_.CreateNot(toBool(operand_lv))
                  : cgen_state_->emitCall("logical_not", {operand_lv, inlineIntNull(operand_ti)});
}

llvm::Value* Executor::codegenIsNull(const Analyzer::UOper* uoper, const CompilationOptions& co) {
  const auto operand = uoper->get_operand();
  if (dynamic_cast<const Analyzer::Constant*>(operand) &&
      dynamic_cast<const Analyzer::Constant*>(operand)->get_is_null()) {
    // for null constants, short-circuit to true
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 1);
  }
  const auto& ti = operand->get_type_info();
  CHECK(ti.is_integer() || ti.is_boolean() || ti.is_decimal() || ti.is_time() || ti.is_string() || ti.is_fp() ||
        ti.is_array());
  // if the type is inferred as non null, short-circuit to false
  if (ti.get_notnull() && !ti.is_array()) {
    return llvm::ConstantInt::get(get_int_type(1, cgen_state_->context_), 0);
  }
  const auto operand_lv = codegen(operand, true, co).front();
  if (ti.is_array()) {
    return cgen_state_->emitExternalCall(
        "array_is_null", get_int_type(1, cgen_state_->context_), {operand_lv, posArg(operand)});
  }
  return codegenIsNullNumber(operand_lv, ti);
}

llvm::Value* Executor::codegenIsNullNumber(llvm::Value* operand_lv, const SQLTypeInfo& ti) {
  if (ti.is_fp()) {
    return cgen_state_->ir_builder_.CreateFCmp(
        llvm::FCmpInst::FCMP_OEQ, operand_lv, ti.get_type() == kFLOAT ? ll_fp(NULL_FLOAT) : ll_fp(NULL_DOUBLE));
  }
  return cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_EQ, operand_lv, inlineIntNull(ti));
}
