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

#include "ExpressionRewrite.h"
#include "../Analyzer/Analyzer.h"
#include "../Shared/sqldefs.h"
#include "DeepCopyVisitor.h"
#include "Execute.h"
#include "ScalarExprVisitor.h"

#include <glog/logging.h>

namespace {

class OrToInVisitor : public ScalarExprVisitor<std::shared_ptr<Analyzer::InValues>> {
 protected:
  std::shared_ptr<Analyzer::InValues> visitBinOper(
      const Analyzer::BinOper* bin_oper) const override {
    switch (bin_oper->get_optype()) {
      case kEQ: {
        const auto rhs_owned = bin_oper->get_own_right_operand();
        auto rhs_no_cast = extract_cast_arg(rhs_owned.get());
        if (!dynamic_cast<const Analyzer::Constant*>(rhs_no_cast)) {
          return nullptr;
        }
        const auto arg = bin_oper->get_own_left_operand();
        const auto& arg_ti = arg->get_type_info();
        auto rhs = rhs_no_cast->deep_copy()->add_cast(arg_ti);
        return makeExpr<Analyzer::InValues>(
            arg, std::list<std::shared_ptr<Analyzer::Expr>>{rhs});
      }
      case kOR: {
        return aggregateResult(visit(bin_oper->get_left_operand()),
                               visit(bin_oper->get_right_operand()));
      }
      default:
        break;
    }
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitUOper(
      const Analyzer::UOper* uoper) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitInValues(
      const Analyzer::InValues*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitInIntegerSet(
      const Analyzer::InIntegerSet*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitCharLength(
      const Analyzer::CharLengthExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitLikeExpr(
      const Analyzer::LikeExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitRegexpExpr(
      const Analyzer::RegexpExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitCaseExpr(
      const Analyzer::CaseExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitDatetruncExpr(
      const Analyzer::DatetruncExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitDatediffExpr(
      const Analyzer::DatediffExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitDateaddExpr(
      const Analyzer::DateaddExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitExtractExpr(
      const Analyzer::ExtractExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitLikelihood(
      const Analyzer::LikelihoodExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitAggExpr(
      const Analyzer::AggExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> aggregateResult(
      const std::shared_ptr<Analyzer::InValues>& lhs,
      const std::shared_ptr<Analyzer::InValues>& rhs) const override {
    if (!lhs || !rhs) {
      return nullptr;
    }
    if (!(*lhs->get_arg() == *rhs->get_arg())) {
      return nullptr;
    }
    auto union_values = lhs->get_value_list();
    const auto& rhs_values = rhs->get_value_list();
    union_values.insert(union_values.end(), rhs_values.begin(), rhs_values.end());
    return makeExpr<Analyzer::InValues>(lhs->get_own_arg(), union_values);
  }
};

class RecursiveOrToInVisitor : public DeepCopyVisitor {
 protected:
  std::shared_ptr<Analyzer::Expr> visitBinOper(
      const Analyzer::BinOper* bin_oper) const override {
    OrToInVisitor simple_visitor;
    if (bin_oper->get_optype() == kOR) {
      auto rewritten = simple_visitor.visit(bin_oper);
      if (rewritten) {
        return rewritten;
      }
    }
    auto lhs = bin_oper->get_own_left_operand();
    auto rhs = bin_oper->get_own_right_operand();
    auto rewritten_lhs = visit(lhs.get());
    auto rewritten_rhs = visit(rhs.get());
    return makeExpr<Analyzer::BinOper>(bin_oper->get_type_info(),
                                       bin_oper->get_contains_agg(),
                                       bin_oper->get_optype(),
                                       bin_oper->get_qualifier(),
                                       rewritten_lhs ? rewritten_lhs : lhs,
                                       rewritten_rhs ? rewritten_rhs : rhs);
  }
};

class IndirectToDirectColVisitor : public DeepCopyVisitor {
 public:
  IndirectToDirectColVisitor(
      const std::list<std::shared_ptr<const InputColDescriptor>>& col_descs) {
    for (auto& desc : col_descs) {
      if (!std::dynamic_pointer_cast<const IndirectInputColDescriptor>(desc)) {
        continue;
      }
      ind_col_id_to_desc_.insert(std::make_pair(desc->getColId(), desc.get()));
    }
  }

 protected:
  RetType visitColumnVar(const Analyzer::ColumnVar* col_var) const override {
    if (!ind_col_id_to_desc_.count(col_var->get_column_id())) {
      return col_var->deep_copy();
    }
    auto desc_it = ind_col_id_to_desc_.find(col_var->get_column_id());
    CHECK(desc_it != ind_col_id_to_desc_.end());
    CHECK(desc_it->second);
    if (desc_it->second->getScanDesc().getTableId() != col_var->get_table_id()) {
      return col_var->deep_copy();
    }
    auto ind_col_desc = dynamic_cast<const IndirectInputColDescriptor*>(desc_it->second);
    CHECK(ind_col_desc);
    return makeExpr<Analyzer::ColumnVar>(col_var->get_type_info(),
                                         ind_col_desc->getIndirectDesc().getTableId(),
                                         ind_col_desc->getRefColIndex(),
                                         col_var->get_rte_idx());
  }

  RetType visitColumnVarTuple(
      const Analyzer::ExpressionTuple* col_var_tuple) const override {
    std::vector<std::shared_ptr<Analyzer::Expr>> redirected_tuple;
    for (const auto& tuple_component : col_var_tuple->getTuple()) {
      const auto redirected_component = visit(tuple_component.get());
      redirected_tuple.push_back(redirected_component);
    }
    return std::make_shared<Analyzer::ExpressionTuple>(redirected_tuple);
  }

 private:
  std::unordered_map<int, const InputColDescriptor*> ind_col_id_to_desc_;
};

class ArrayElementStringLiteralEncodingVisitor : public DeepCopyVisitor {
 protected:
  using RetType = DeepCopyVisitor::RetType;

  RetType visitArrayOper(const Analyzer::ArrayExpr* array_expr) const override {
    std::vector<std::shared_ptr<Analyzer::Expr>> args_copy;
    for (size_t i = 0; i < array_expr->getElementCount(); ++i) {
      auto const element_expr_ptr = visit(array_expr->getElement(i));
      auto const& element_expr_type_info = element_expr_ptr->get_type_info();

      if (!element_expr_type_info.is_string() ||
          element_expr_type_info.get_compression() != kENCODING_NONE) {
        args_copy.push_back(element_expr_ptr);
      } else {
        auto transient_dict_type_info = element_expr_type_info;

        transient_dict_type_info.set_compression(kENCODING_DICT);
        transient_dict_type_info.set_comp_param(TRANSIENT_DICT_ID);
        transient_dict_type_info.set_fixed_size();
        args_copy.push_back(element_expr_ptr->add_cast(transient_dict_type_info));
      }
    }

    const auto& type_info = array_expr->get_type_info();
    return makeExpr<Analyzer::ArrayExpr>(
        type_info, args_copy, array_expr->getExprIndex());
  }
};

class ConstantFoldingVisitor : public DeepCopyVisitor {
 protected:
  template <typename T1, typename T2>
  auto foldComparison(SQLOps optype, T1&& t1, T2&& t2) const
      -> decltype(std::forward<T1>(t1) == std::forward<T2>(t2)) {
    switch (optype) {
      case kEQ:
        return std::forward<T1>(t1) == std::forward<T2>(t2);
      case kNE:
        return std::forward<T1>(t1) != std::forward<T2>(t2);
      case kLT:
        return std::forward<T1>(t1) < std::forward<T2>(t2);
      case kLE:
        return std::forward<T1>(t1) <= std::forward<T2>(t2);
      case kGT:
        return std::forward<T1>(t1) > std::forward<T2>(t2);
      case kGE:
        return std::forward<T1>(t1) >= std::forward<T2>(t2);
      default:
        break;
    }
    throw std::runtime_error("Unable to fold");
    return false;
  }

  template <typename T1, typename T2>
  auto foldLogic(SQLOps optype, T1&& t1, T2&& t2) const
      -> decltype(std::forward<T1>(t1) && std::forward<T2>(t2)) {
    switch (optype) {
      case kAND:
        return std::forward<T1>(t1) && std::forward<T2>(t2);
      case kOR:
        return std::forward<T1>(t1) || std::forward<T2>(t2);
      case kNOT:
        return !std::forward<T1>(t1);
      default:
        break;
    }
    throw std::runtime_error("Unable to fold");
    return false;
  }

  template <typename T1, typename T2>
  auto foldArithmetic(SQLOps optype, T1&& t1, T2&& t2) const
      -> decltype(std::forward<T1>(t1) + std::forward<T2>(t2)) {
    switch (optype) {
      case kPLUS:
        if (((std::forward<T1>(t1) + std::forward<T2>(t2)) - std::forward<T2>(t2)) !=
            std::forward<T1>(t1)) {
          throw std::runtime_error("Plus overflow");
        }
        return std::forward<T1>(t1) + std::forward<T2>(t2);
      case kMINUS:
        if (((std::forward<T1>(t1) - std::forward<T2>(t2)) + std::forward<T2>(t2)) !=
            std::forward<T1>(t1)) {
          throw std::runtime_error("Minus overflow");
        }
        return std::forward<T1>(t1) - std::forward<T2>(t2);
      case kMULTIPLY:
        if ((std::forward<T2>(t1) * std::forward<T2>(t2)) ==
            (std::forward<T1>(t1) - std::forward<T1>(t1))) {
          return (std::forward<T2>(t1) * std::forward<T2>(t2));
        }
        if (((std::forward<T1>(t1) * std::forward<T2>(t2)) / std::forward<T2>(t2)) !=
            std::forward<T1>(t1)) {
          throw std::runtime_error("Mul overflow");
        }
        return std::forward<T1>(t1) * std::forward<T2>(t2);
      case kDIVIDE:
        if ((std::forward<T2>(t2) + std::forward<T2>(t2)) == std::forward<T2>(t2)) {
          throw std::runtime_error("Will not fold division by zero");
        }
        return std::forward<T1>(t1) / std::forward<T2>(t2);
      default:
        break;
    }
    throw std::runtime_error("Unable to fold");
  }

  bool foldOper(SQLOps optype,
                SQLTypes type,
                Datum lhs,
                Datum rhs,
                Datum& result,
                SQLTypes& result_type) const {
    result_type = type;

    try {
      switch (type) {
        case kBOOLEAN:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison(optype, lhs.boolval, rhs.boolval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_LOGIC(optype)) {
            result.boolval = foldLogic(optype, lhs.boolval, rhs.boolval);
            result_type = kBOOLEAN;
            return true;
          }
          CHECK(!IS_ARITHMETIC(optype));
          break;
        case kTINYINT:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison(optype, lhs.tinyintval, rhs.tinyintval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.tinyintval = foldArithmetic(optype, lhs.tinyintval, rhs.tinyintval);
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kSMALLINT:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison(optype, lhs.smallintval, rhs.smallintval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.smallintval = foldArithmetic(optype, lhs.smallintval, rhs.smallintval);
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kINT:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison(optype, lhs.intval, rhs.intval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.intval = foldArithmetic(optype, lhs.intval, rhs.intval);
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kBIGINT:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison(optype, lhs.bigintval, rhs.bigintval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.bigintval = foldArithmetic(optype, lhs.bigintval, rhs.bigintval);
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kFLOAT:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison(optype, lhs.floatval, rhs.floatval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.floatval = foldArithmetic(optype, lhs.floatval, rhs.floatval);
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kDOUBLE:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison(optype, lhs.doubleval, rhs.doubleval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.doubleval = foldArithmetic(optype, lhs.doubleval, rhs.doubleval);
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        default:
          break;
      }
    } catch (...) {
      return false;
    }
    return false;
  }

  std::shared_ptr<Analyzer::Expr> visitUOper(
      const Analyzer::UOper* uoper) const override {
    const auto unvisited_operand = uoper->get_operand();
    const auto optype = uoper->get_optype();
    const Analyzer::BinOper* unvisited_binoper = nullptr;
    if (optype == kCAST) {
      unvisited_binoper = dynamic_cast<const Analyzer::BinOper*>(unvisited_operand);
    }
    const auto operand = (unvisited_binoper) ? visitBinOper(unvisited_binoper, uoper)
                                             : visit(unvisited_operand);
    const auto& operand_ti = operand->get_type_info();
    const auto operand_type =
        operand_ti.is_decimal() ? decimal_to_int_type(operand_ti) : operand_ti.get_type();
    const auto const_operand =
        std::dynamic_pointer_cast<const Analyzer::Constant>(operand);

    if (const_operand) {
      const auto operand_datum = const_operand->get_constval();
      Datum result_datum;
      SQLTypes result_type;
      switch (optype) {
        case kNOT:
          if (foldOper(optype,
                       operand_type,
                       operand_datum,
                       operand_datum,
                       result_datum,
                       result_type)) {
            return makeExpr<Analyzer::Constant>(result_type, false, result_datum);
          }
          break;
        case kMINUS: {
          Datum zero_datum;
          if (!foldOper(optype,
                        operand_type,
                        operand_datum,
                        operand_datum,
                        zero_datum,
                        result_type)) {
            break;
          }
          if (foldOper(optype,
                       operand_type,
                       zero_datum,
                       operand_datum,
                       result_datum,
                       result_type)) {
            return makeExpr<Analyzer::Constant>(result_type, false, result_datum);
          }
          break;
        }
        default:
          break;
      }
    }

    return makeExpr<Analyzer::UOper>(
        uoper->get_type_info(), uoper->get_contains_agg(), optype, operand);
  }

  std::shared_ptr<Analyzer::Expr> visitBinOper(
      const Analyzer::BinOper* bin_oper) const override {
    return visitBinOper(bin_oper, nullptr);
  }

  std::shared_ptr<Analyzer::Expr> visitBinOper(const Analyzer::BinOper* bin_oper,
                                               const Analyzer::UOper* cast) const {
    const auto lhs = visit(bin_oper->get_left_operand());
    const auto rhs = visit(bin_oper->get_right_operand());
    const auto const_lhs = std::dynamic_pointer_cast<const Analyzer::Constant>(lhs);
    const auto const_rhs = std::dynamic_pointer_cast<const Analyzer::Constant>(rhs);
    const auto optype = bin_oper->get_optype();
    const auto& lhs_ti = lhs->get_type_info();
    const auto& rhs_ti = rhs->get_type_info();
    auto lhs_type = lhs_ti.is_decimal() ? decimal_to_int_type(lhs_ti) : lhs_ti.get_type();
    auto rhs_type = rhs_ti.is_decimal() ? decimal_to_int_type(rhs_ti) : rhs_ti.get_type();

    if (const_lhs && const_rhs && lhs_type == rhs_type) {
      auto lhs_datum = const_lhs->get_constval();
      auto rhs_datum = const_rhs->get_constval();
      if (cast) {
        if (optype == kMINUS || optype == kPLUS || optype == kMULTIPLY) {
          const auto cast_ti = cast->get_type_info();
          auto lhs_copy = const_lhs->deep_copy();
          auto rhs_copy = const_rhs->deep_copy();
          auto cast_lhs = lhs_copy->add_cast(cast_ti);
          auto cast_rhs = rhs_copy->add_cast(cast_ti);
          auto const_cast_lhs =
              std::dynamic_pointer_cast<const Analyzer::Constant>(cast_lhs);
          auto const_cast_rhs =
              std::dynamic_pointer_cast<const Analyzer::Constant>(cast_rhs);
          if (const_cast_lhs && const_cast_rhs) {
            lhs_datum = const_cast_lhs->get_constval();
            rhs_datum = const_cast_rhs->get_constval();
            lhs_type = cast_ti.get_type();
          }
        }
      }
      Datum result_datum;
      SQLTypes result_type;
      if (foldOper(optype, lhs_type, lhs_datum, rhs_datum, result_datum, result_type)) {
        return makeExpr<Analyzer::Constant>(result_type, false, result_datum);
      }
    }

    if (optype == kAND && lhs_type == rhs_type && lhs_type == kBOOLEAN) {
      if (const_rhs) {
        auto rhs_datum = const_rhs->get_constval();
        if (rhs_datum.boolval == false) {
          Datum d;
          d.boolval = false;
          // lhs && false --> false
          return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
        }
        // lhs && true --> lhs
        return lhs;
      }
      if (const_lhs) {
        auto lhs_datum = const_lhs->get_constval();
        if (lhs_datum.boolval == false) {
          Datum d;
          d.boolval = false;
          // false && rhs --> false
          return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
        }
        // true && rhs --> rhs
        return rhs;
      }
    }
    if (optype == kOR && lhs_type == rhs_type && lhs_type == kBOOLEAN) {
      if (const_rhs) {
        auto rhs_datum = const_rhs->get_constval();
        if (rhs_datum.boolval == true) {
          Datum d;
          d.boolval = true;
          // lhs || true --> true
          return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
        }
        // lhs || false --> lhs
        return lhs;
      }
      if (const_lhs) {
        auto lhs_datum = const_lhs->get_constval();
        if (lhs_datum.boolval == true) {
          Datum d;
          d.boolval = true;
          // true || rhs --> true
          return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
        }
        // false || rhs --> rhs
        return rhs;
      }
    }

    return makeExpr<Analyzer::BinOper>(bin_oper->get_type_info(),
                                       bin_oper->get_contains_agg(),
                                       bin_oper->get_optype(),
                                       bin_oper->get_qualifier(),
                                       lhs,
                                       rhs);
  }
};

const Analyzer::Expr* strip_likelihood(const Analyzer::Expr* expr) {
  const auto with_likelihood = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
  if (!with_likelihood) {
    return expr;
  }
  return with_likelihood->get_arg();
}

}  // namespace

Analyzer::ExpressionPtr rewrite_array_elements(Analyzer::Expr const* expr) {
  return ArrayElementStringLiteralEncodingVisitor().visit(expr);
}

Analyzer::ExpressionPtr rewrite_expr(const Analyzer::Expr* expr) {
  const auto expr_no_likelihood = strip_likelihood(expr);
  // The following check is not strictly needed, but seems silly to transform a
  // simple string comparison to an IN just to codegen the same thing anyway.

  RecursiveOrToInVisitor visitor;
  auto rewritten_expr = visitor.visit(expr_no_likelihood);
  const auto expr_with_likelihood =
      std::dynamic_pointer_cast<const Analyzer::LikelihoodExpr>(rewritten_expr);
  if (expr_with_likelihood) {
    // Add back likelihood
    return std::make_shared<Analyzer::LikelihoodExpr>(
        rewritten_expr, expr_with_likelihood->get_likelihood());
  }
  return rewritten_expr;
}

std::list<std::shared_ptr<Analyzer::Expr>> redirect_exprs(
    const std::list<std::shared_ptr<Analyzer::Expr>>& exprs,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_descs) {
  bool has_indirect_col = false;
  for (const auto& desc : col_descs) {
    if (std::dynamic_pointer_cast<const IndirectInputColDescriptor>(desc)) {
      has_indirect_col = true;
      break;
    }
  }

  if (!has_indirect_col) {
    return exprs;
  }

  IndirectToDirectColVisitor visitor(col_descs);
  std::list<std::shared_ptr<Analyzer::Expr>> new_exprs;
  for (auto& e : exprs) {
    new_exprs.push_back(e ? visitor.visit(e.get()) : nullptr);
  }
  return new_exprs;
}

std::vector<std::shared_ptr<Analyzer::Expr>> redirect_exprs(
    const std::vector<Analyzer::Expr*>& exprs,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_descs) {
  bool has_indirect_col = false;
  for (const auto& desc : col_descs) {
    if (std::dynamic_pointer_cast<const IndirectInputColDescriptor>(desc)) {
      has_indirect_col = true;
      break;
    }
  }

  std::vector<std::shared_ptr<Analyzer::Expr>> new_exprs;
  if (!has_indirect_col) {
    for (auto& e : exprs) {
      new_exprs.push_back(e ? e->deep_copy() : nullptr);
    }
    return new_exprs;
  }

  IndirectToDirectColVisitor visitor(col_descs);
  for (auto& e : exprs) {
    new_exprs.push_back(e ? visitor.visit(e) : nullptr);
  }
  return new_exprs;
}

std::shared_ptr<Analyzer::Expr> redirect_expr(
    const Analyzer::Expr* expr,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_descs) {
  if (!expr) {
    return nullptr;
  }
  IndirectToDirectColVisitor visitor(col_descs);
  return visitor.visit(expr);
}

std::shared_ptr<Analyzer::Expr> fold_expr(const Analyzer::Expr* expr) {
  if (!expr) {
    return nullptr;
  }
  const auto expr_no_likelihood = strip_likelihood(expr);
  ConstantFoldingVisitor visitor;
  auto rewritten_expr = visitor.visit(expr_no_likelihood);
  const auto expr_with_likelihood = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
  if (expr_with_likelihood) {
    // Add back likelihood
    return std::make_shared<Analyzer::LikelihoodExpr>(
        rewritten_expr, expr_with_likelihood->get_likelihood());
  }
  return rewritten_expr;
}
