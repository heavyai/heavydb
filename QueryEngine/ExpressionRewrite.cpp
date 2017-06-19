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
#include "Execute.h"
#include "ScalarExprVisitor.h"

#include <glog/logging.h>

namespace {

class OrToInVisitor : public ScalarExprVisitor<std::shared_ptr<Analyzer::InValues>> {
 protected:
  std::shared_ptr<Analyzer::InValues> visitBinOper(const Analyzer::BinOper* bin_oper) const override {
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
        return makeExpr<Analyzer::InValues>(arg, std::list<std::shared_ptr<Analyzer::Expr>>{rhs});
      }
      case kOR: {
        return aggregateResult(visit(bin_oper->get_left_operand()), visit(bin_oper->get_right_operand()));
      }
      default:
        break;
    }
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitUOper(const Analyzer::UOper* uoper) const override { return nullptr; }

  std::shared_ptr<Analyzer::InValues> visitInValues(const Analyzer::InValues*) const override { return nullptr; }

  std::shared_ptr<Analyzer::InValues> visitInIntegerSet(const Analyzer::InIntegerSet*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitCharLength(const Analyzer::CharLengthExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitLikeExpr(const Analyzer::LikeExpr*) const override { return nullptr; }

  std::shared_ptr<Analyzer::InValues> visitRegexpExpr(const Analyzer::RegexpExpr*) const override { return nullptr; }

  std::shared_ptr<Analyzer::InValues> visitCaseExpr(const Analyzer::CaseExpr*) const override { return nullptr; }

  std::shared_ptr<Analyzer::InValues> visitDatetruncExpr(const Analyzer::DatetruncExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitDatediffExpr(const Analyzer::DatediffExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitExtractExpr(const Analyzer::ExtractExpr*) const override { return nullptr; }

  std::shared_ptr<Analyzer::InValues> visitLikelihood(const Analyzer::LikelihoodExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitAggExpr(const Analyzer::AggExpr*) const override { return nullptr; }

  std::shared_ptr<Analyzer::InValues> aggregateResult(const std::shared_ptr<Analyzer::InValues>& lhs,
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

class DeepCopyVisitor : public ScalarExprVisitor<std::shared_ptr<Analyzer::Expr>> {
 protected:
  typedef std::shared_ptr<Analyzer::Expr> RetType;
  RetType visitColumnVar(const Analyzer::ColumnVar* col_var) const override { return col_var->deep_copy(); }

  RetType visitVar(const Analyzer::Var* var) const override { return var->deep_copy(); }

  RetType visitConstant(const Analyzer::Constant* constant) const override { return constant->deep_copy(); }

  RetType visitIterator(const Analyzer::IterExpr* iter) const override { return iter->deep_copy(); }

  RetType visitUOper(const Analyzer::UOper* uoper) const override {
    return makeExpr<Analyzer::UOper>(
        uoper->get_type_info(), uoper->get_contains_agg(), uoper->get_optype(), visit(uoper->get_operand()));
  }

  RetType visitBinOper(const Analyzer::BinOper* bin_oper) const override {
    return makeExpr<Analyzer::BinOper>(bin_oper->get_type_info(),
                                       bin_oper->get_contains_agg(),
                                       bin_oper->get_optype(),
                                       bin_oper->get_qualifier(),
                                       visit(bin_oper->get_left_operand()),
                                       visit(bin_oper->get_right_operand()));
  }

  RetType visitInValues(const Analyzer::InValues* in_values) const override {
    const auto& value_list = in_values->get_value_list();
    std::list<RetType> new_list;
    for (const auto in_value : value_list) {
      new_list.push_back(visit(in_value.get()));
    }
    return makeExpr<Analyzer::InValues>(visit(in_values->get_arg()), new_list);
  }

  RetType visitInIntegerSet(const Analyzer::InIntegerSet* in_integer_set) const override {
    return makeExpr<Analyzer::InIntegerSet>(visit(in_integer_set->get_arg()),
                                            in_integer_set->get_value_list(),
                                            in_integer_set->get_type_info().get_notnull());
  }

  RetType visitCharLength(const Analyzer::CharLengthExpr* char_length) const override {
    return makeExpr<Analyzer::CharLengthExpr>(visit(char_length->get_arg()), char_length->get_calc_encoded_length());
  }

  RetType visitLikeExpr(const Analyzer::LikeExpr* like) const override {
    auto escape_expr = like->get_escape_expr();
    return makeExpr<Analyzer::LikeExpr>(visit(like->get_arg()),
                                        visit(like->get_like_expr()),
                                        escape_expr ? visit(escape_expr) : nullptr,
                                        like->get_is_ilike(),
                                        like->get_is_simple());
  }

  RetType visitRegexpExpr(const Analyzer::RegexpExpr* regexp) const override {
    auto escape_expr = regexp->get_escape_expr();
    return makeExpr<Analyzer::RegexpExpr>(
        visit(regexp->get_arg()), visit(regexp->get_pattern_expr()), escape_expr ? visit(escape_expr) : nullptr);
  }

  RetType visitCaseExpr(const Analyzer::CaseExpr* case_expr) const override {
    std::list<std::pair<RetType, RetType>> new_list;
    for (auto p : case_expr->get_expr_pair_list()) {
      new_list.push_back(std::make_pair(visit(p.first.get()), visit(p.second.get())));
    }
    auto else_expr = case_expr->get_else_expr();
    return makeExpr<Analyzer::CaseExpr>(case_expr->get_type_info(),
                                        case_expr->get_contains_agg(),
                                        new_list,
                                        else_expr == nullptr ? nullptr : visit(else_expr));
  }

  RetType visitDatetruncExpr(const Analyzer::DatetruncExpr* datetrunc) const override {
    return makeExpr<Analyzer::DatetruncExpr>(datetrunc->get_type_info(),
                                             datetrunc->get_contains_agg(),
                                             datetrunc->get_field(),
                                             visit(datetrunc->get_from_expr()));
  }

  RetType visitExtractExpr(const Analyzer::ExtractExpr* extract) const override {
    return makeExpr<Analyzer::ExtractExpr>(
        extract->get_type_info(), extract->get_contains_agg(), extract->get_field(), visit(extract->get_from_expr()));
  }

  RetType visitFunctionOper(const Analyzer::FunctionOper* func_oper) const override {
    std::vector<std::shared_ptr<Analyzer::Expr>> args_copy;
    for (size_t i = 0; i < func_oper->getArity(); ++i) {
      args_copy.push_back(visit(func_oper->getArg(i)));
    }
    const auto& type_info = func_oper->get_type_info();
    return makeExpr<Analyzer::FunctionOper>(type_info, func_oper->getName(), args_copy);
  }

  RetType visitDatediffExpr(const Analyzer::DatediffExpr* datediff) const override {
    return makeExpr<Analyzer::DatediffExpr>(datediff->get_type_info(),
                                            datediff->get_field(),
                                            visit(datediff->get_start_expr()),
                                            visit(datediff->get_end_expr()));
  }

  RetType visitFunctionOperWithCustomTypeHandling(
      const Analyzer::FunctionOperWithCustomTypeHandling* func_oper) const override {
    std::vector<std::shared_ptr<Analyzer::Expr>> args_copy;
    for (size_t i = 0; i < func_oper->getArity(); ++i) {
      args_copy.push_back(visit(func_oper->getArg(i)));
    }
    const auto& type_info = func_oper->get_type_info();
    return makeExpr<Analyzer::FunctionOperWithCustomTypeHandling>(type_info, func_oper->getName(), args_copy);
  }

  RetType visitLikelihood(const Analyzer::LikelihoodExpr* likelihood) const override {
    return makeExpr<Analyzer::LikelihoodExpr>(visit(likelihood->get_arg()), likelihood->get_likelihood());
  }

  RetType visitAggExpr(const Analyzer::AggExpr* agg) const override {
    RetType arg = agg->get_arg() ? visit(agg->get_arg()) : nullptr;
    return makeExpr<Analyzer::AggExpr>(agg->get_type_info(), agg->get_aggtype(), arg, agg->get_is_distinct());
  }
};

class IndirectToDirectColVisitor : public DeepCopyVisitor {
 public:
  IndirectToDirectColVisitor(const std::list<std::shared_ptr<const InputColDescriptor>>& col_descs) {
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

 private:
  std::unordered_map<int, const InputColDescriptor*> ind_col_id_to_desc_;
};

class ConstantFoldingVisitor : public DeepCopyVisitor {
 protected:
  std::shared_ptr<Analyzer::Expr> visitUOper(const Analyzer::UOper* uoper) const override {
    const auto unvisited_operand = uoper->get_operand();
    const auto optype = uoper->get_optype();
    const Analyzer::BinOper* unvisited_binoper = nullptr;
    if (optype == kCAST) {
      unvisited_binoper = dynamic_cast<const Analyzer::BinOper*>(unvisited_operand);
    }
    const auto operand = (unvisited_binoper) ? visitBinOper(unvisited_binoper, uoper) : visit(unvisited_operand);
    const auto& operand_ti = operand->get_type_info();
    const auto operand_type = operand_ti.is_decimal() ? decimal_to_int_type(operand_ti) : operand_ti.get_type();
    const auto const_operand = std::dynamic_pointer_cast<const Analyzer::Constant>(operand);
    bool notnull = true;
    if (const_operand && !operand_ti.get_notnull() && isNull(const_operand.get()))
      notnull = false;

    if (const_operand && notnull) {
      switch (optype) {
        case kNOT: {
          if (operand_ti.is_boolean()) {
            Datum d;
            d.boolval = !const_operand->get_constval().boolval;
            return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
          }
          break;
        }
        case kMINUS: {
          int64_t max_int{0}, min_int{0};
          std::tie(max_int, min_int) = inline_int_max_min(operand_ti.get_size());
          bool fold = true;
          Datum d;
          switch (operand_type) {
            case kSMALLINT: {
              auto c = const_operand->get_constval().smallintval;
              if (c == min_int)
                fold = false;
              else
                d.smallintval = -c;
              break;
            }
            case kINT: {
              auto c = const_operand->get_constval().intval;
              if (c == min_int)
                fold = false;
              else
                d.intval = -c;
              break;
            }
            case kBIGINT: {
              auto c = const_operand->get_constval().bigintval;
              if (c == min_int)
                fold = false;
              else
                d.bigintval = -c;
              break;
            }
            case kFLOAT:
              d.floatval = -const_operand->get_constval().floatval;
              break;
            case kDOUBLE:
              d.doubleval = -const_operand->get_constval().doubleval;
              break;
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(operand_type, false, d);
          }
          break;
        }
        case kCAST:
        case kISNULL:
        case kUNNEST:
          break;
        default:
          break;
      }
    }

    return makeExpr<Analyzer::UOper>(uoper->get_type_info(), uoper->get_contains_agg(), optype, operand);
  }

  std::shared_ptr<Analyzer::Expr> visitBinOper(const Analyzer::BinOper* bin_oper) const override {
    return visitBinOper(bin_oper, nullptr);
  }

  std::shared_ptr<Analyzer::Expr> visitBinOper(const Analyzer::BinOper* bin_oper, const Analyzer::UOper* cast) const {
    const auto lhs = visit(bin_oper->get_left_operand());
    const auto rhs = visit(bin_oper->get_right_operand());
    const auto const_lhs = std::dynamic_pointer_cast<const Analyzer::Constant>(lhs);
    const auto const_rhs = std::dynamic_pointer_cast<const Analyzer::Constant>(rhs);
    const auto optype = bin_oper->get_optype();
    const auto& lhs_ti = lhs->get_type_info();
    const auto& rhs_ti = rhs->get_type_info();
    auto lhs_type = lhs_ti.is_decimal() ? decimal_to_int_type(lhs_ti) : lhs_ti.get_type();
    auto rhs_type = rhs_ti.is_decimal() ? decimal_to_int_type(rhs_ti) : rhs_ti.get_type();

    bool notnull = true;
    if (const_lhs && !lhs_ti.get_notnull() && isNull(const_lhs.get()))
      notnull = false;
    else if (const_rhs && !rhs_ti.get_notnull() && isNull(const_rhs.get()))
      notnull = false;

    if (const_lhs && const_rhs && lhs_type == rhs_type && notnull) {
      auto lhs_datum = const_lhs->get_constval();
      auto rhs_datum = const_rhs->get_constval();
      switch (optype) {
        case kEQ: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kSMALLINT:
              d.boolval = (lhs_datum.smallintval == rhs_datum.smallintval);
              break;
            case kINT:
              d.boolval = (lhs_datum.intval == rhs_datum.intval);
              break;
            case kBIGINT:
              d.boolval = (lhs_datum.bigintval == rhs_datum.bigintval);
              break;
            case kFLOAT:
              d.boolval = (lhs_datum.floatval == rhs_datum.floatval);
              break;
            case kDOUBLE:
              d.boolval = (lhs_datum.doubleval == rhs_datum.doubleval);
              break;
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
          }
          break;
        }
        case kNE: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kSMALLINT:
              d.boolval = (lhs_datum.smallintval != rhs_datum.smallintval);
              break;
            case kINT:
              d.boolval = (lhs_datum.intval != rhs_datum.intval);
              break;
            case kBIGINT:
              d.boolval = (lhs_datum.bigintval != rhs_datum.bigintval);
              break;
            case kFLOAT:
              d.boolval = (lhs_datum.floatval != rhs_datum.floatval);
              break;
            case kDOUBLE:
              d.boolval = (lhs_datum.doubleval != rhs_datum.doubleval);
              break;
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
          }
          break;
        }
        case kLT: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kSMALLINT:
              d.boolval = (lhs_datum.smallintval < rhs_datum.smallintval);
              break;
            case kINT:
              d.boolval = (lhs_datum.intval < rhs_datum.intval);
              break;
            case kBIGINT:
              d.boolval = (lhs_datum.bigintval < rhs_datum.bigintval);
              break;
            case kFLOAT:
              d.boolval = (lhs_datum.floatval < rhs_datum.floatval);
              break;
            case kDOUBLE:
              d.boolval = (lhs_datum.doubleval < rhs_datum.doubleval);
              break;
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
          }
          break;
        }
        case kLE: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kSMALLINT:
              d.boolval = (lhs_datum.smallintval <= rhs_datum.smallintval);
              break;
            case kINT:
              d.boolval = (lhs_datum.intval <= rhs_datum.intval);
              break;
            case kBIGINT:
              d.boolval = (lhs_datum.bigintval <= rhs_datum.bigintval);
              break;
            case kFLOAT:
              d.boolval = (lhs_datum.floatval <= rhs_datum.floatval);
              break;
            case kDOUBLE:
              d.boolval = (lhs_datum.doubleval <= rhs_datum.doubleval);
              break;
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
          }
          break;
        }
        case kGT: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kSMALLINT:
              d.boolval = (lhs_datum.smallintval > rhs_datum.smallintval);
              break;
            case kINT:
              d.boolval = (lhs_datum.intval > rhs_datum.intval);
              break;
            case kBIGINT:
              d.boolval = (lhs_datum.bigintval > rhs_datum.bigintval);
              break;
            case kFLOAT:
              d.boolval = (lhs_datum.floatval > rhs_datum.floatval);
              break;
            case kDOUBLE:
              d.boolval = (lhs_datum.doubleval > rhs_datum.doubleval);
              break;
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
          }
          break;
        }
        case kGE: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kSMALLINT:
              d.boolval = (lhs_datum.smallintval >= rhs_datum.smallintval);
              break;
            case kINT:
              d.boolval = (lhs_datum.intval >= rhs_datum.intval);
              break;
            case kBIGINT:
              d.boolval = (lhs_datum.bigintval >= rhs_datum.bigintval);
              break;
            case kFLOAT:
              d.boolval = (lhs_datum.floatval >= rhs_datum.floatval);
              break;
            case kDOUBLE:
              d.boolval = (lhs_datum.doubleval >= rhs_datum.doubleval);
              break;
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
          }
          break;
        }
        case kAND: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kBOOLEAN:
              d.boolval = (lhs_datum.boolval && rhs_datum.boolval);
              break;
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
          }
          break;
        }
        case kOR: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kBOOLEAN:
              d.boolval = (lhs_datum.boolval || rhs_datum.boolval);
              break;
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
          }
          break;
        }
        case kMINUS:
        case kPLUS: {
          auto result_type = lhs_type;
          if (cast) {
            const auto cast_ti = cast->get_type_info();
            result_type = cast_ti.get_type();
          }
          bool fold = true;
          int64_t bigintval = 0;
          int64_t rhs_bigintval = 0;
          double doubleval = 0.0;
          double rhs_doubleval = 0.0;
          switch (lhs_type) {
            case kSMALLINT:
              rhs_bigintval = (optype == kMINUS) ? -((int64_t)rhs_datum.smallintval) : (int64_t)rhs_datum.smallintval;
              bigintval = (int64_t)lhs_datum.smallintval + rhs_bigintval;
              doubleval = (double)bigintval;
              break;
            case kINT:
              rhs_bigintval = (optype == kMINUS) ? -((int64_t)rhs_datum.intval) : (int64_t)rhs_datum.intval;
              bigintval = (int64_t)lhs_datum.intval + rhs_bigintval;
              doubleval = (double)bigintval;
              break;
            case kBIGINT:
              rhs_bigintval = (optype == kMINUS) ? -rhs_datum.bigintval : rhs_datum.bigintval;
              bigintval = lhs_datum.bigintval + rhs_bigintval;
              doubleval = (double)bigintval;
              break;
            case kFLOAT:
              rhs_doubleval = (optype == kMINUS) ? -((double)rhs_datum.doubleval) : (double)rhs_datum.doubleval;
              doubleval = (double)lhs_datum.floatval + rhs_doubleval;
              bigintval = (int64_t)doubleval;
              break;
            case kDOUBLE:
              rhs_doubleval = (optype == kMINUS) ? -rhs_datum.doubleval : rhs_datum.doubleval;
              doubleval = lhs_datum.doubleval + rhs_doubleval;
              bigintval = (int64_t)doubleval;
              break;
            default:
              fold = false;
              break;
          }
          Datum d;
          if (fold) {
            switch (result_type) {
              case kSMALLINT:
                d.smallintval = (int16_t)bigintval;
                if ((int64_t)d.smallintval != bigintval)
                  fold = false;
                break;
              case kINT:
                d.intval = (int32_t)bigintval;
                if ((int64_t)d.intval != bigintval)
                  fold = false;
                break;
              case kBIGINT:
                d.bigintval = bigintval;
                break;
              case kFLOAT:
                d.floatval = (float)doubleval;
                if ((double)d.floatval != doubleval)
                  fold = false;
                break;
              case kDOUBLE:
                d.doubleval = doubleval;
                break;
              default:
                fold = false;
                break;
            }
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(result_type, false, d);
          }
          break;
        }
        case kMULTIPLY: {
          auto result_type = lhs_type;
          if (cast) {
            const auto cast_ti = cast->get_type_info();
            result_type = cast_ti.get_type();
          }
          bool fold = true;
          int64_t bigintval = 0;
          double doubleval = 0.0;
          switch (lhs_type) {
            case kSMALLINT:
              bigintval = (int64_t)lhs_datum.smallintval * (int64_t)rhs_datum.smallintval;
              doubleval = (double)bigintval;
              break;
            case kINT:
              bigintval = (int64_t)lhs_datum.intval * (int64_t)rhs_datum.intval;
              doubleval = (double)bigintval;
              break;
            case kBIGINT:
              bigintval = lhs_datum.bigintval * rhs_datum.bigintval;
              doubleval = (double)bigintval;
              break;
            case kFLOAT:
              doubleval = (double)lhs_datum.floatval * (double)rhs_datum.floatval;
              bigintval = (int64_t)doubleval;
              break;
            case kDOUBLE:
              doubleval = lhs_datum.doubleval * rhs_datum.doubleval;
              bigintval = (int64_t)doubleval;
              break;
            default:
              fold = false;
              break;
          }
          Datum d;
          if (fold) {
            switch (result_type) {
              case kSMALLINT:
                d.smallintval = (int16_t)bigintval;
                if ((int64_t)d.smallintval != bigintval)
                  fold = false;
                break;
              case kINT:
                d.intval = (int32_t)bigintval;
                if ((int64_t)d.intval != bigintval)
                  fold = false;
                break;
              case kBIGINT:
                d.bigintval = bigintval;
                break;
              case kFLOAT:
                d.floatval = (float)doubleval;
                if ((double)d.floatval != doubleval)
                  fold = false;
                break;
              case kDOUBLE:
                d.doubleval = doubleval;
                break;
              default:
                fold = false;
                break;
            }
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(result_type, false, d);
          }
          break;
        }
        case kDIVIDE: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kSMALLINT: {
              auto c2 = rhs_datum.smallintval;
              if (c2 == 0)
                fold = false;
              else
                d.smallintval = (lhs_datum.smallintval / c2);
              break;
            }
            case kINT: {
              auto c2 = rhs_datum.intval;
              if (c2 == 0)
                fold = false;
              else
                d.intval = (lhs_datum.intval / c2);
              break;
            }
            case kBIGINT: {
              auto c2 = rhs_datum.bigintval;
              if (c2 == 0)
                fold = false;
              else
                d.bigintval = (lhs_datum.bigintval / c2);
              break;
            }
            case kFLOAT: {
              auto c2 = rhs_datum.floatval;
              if (c2 == 0.0)
                fold = false;
              else
                d.floatval = (lhs_datum.floatval / c2);
              break;
            }
            case kDOUBLE: {
              auto c2 = rhs_datum.doubleval;
              if (c2 == 0.0)
                fold = false;
              else
                d.doubleval = (lhs_datum.doubleval / c2);
              break;
            }
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(lhs_type, false, d);
          }
          break;
        }
        case kMODULO: {
          bool fold = true;
          Datum d;
          switch (lhs_type) {
            case kSMALLINT: {
              auto c2 = rhs_datum.smallintval;
              if (c2 == 0)
                fold = false;
              else
                d.smallintval = (lhs_datum.smallintval % c2);
              break;
            }
            case kINT: {
              auto c2 = rhs_datum.intval;
              if (c2 == 0)
                fold = false;
              else
                d.intval = (lhs_datum.intval % c2);
              break;
            }
            case kBIGINT: {
              auto c2 = rhs_datum.bigintval;
              if (c2 == 0)
                fold = false;
              else
                d.bigintval = (lhs_datum.bigintval % c2);
              break;
            }
            default:
              fold = false;
              break;
          }
          if (fold) {
            return makeExpr<Analyzer::Constant>(lhs_type, false, d);
          }
          break;
        }
        case kARRAY_AT:
        default:
          break;
      }
    }

    if (optype == kAND && lhs_type == rhs_type && lhs_type == kBOOLEAN && notnull) {
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
    if (optype == kOR && lhs_type == rhs_type && lhs_type == kBOOLEAN && notnull) {
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

private:
  bool isNull(const Analyzer::Constant* c) const {
    const auto& ti = c->get_type_info();
    const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
    auto datum = c->get_constval();
    switch (type) {
      case kSMALLINT: return (datum.smallintval == NULL_SMALLINT);
      case kINT:      return (datum.intval == NULL_INT);
      case kBIGINT:   return (datum.bigintval == NULL_BIGINT);
      case kFLOAT:    return (datum.floatval == NULL_FLOAT);
      case kDOUBLE:   return (datum.doubleval == NULL_DOUBLE);
      default: break;
    }
    return false;
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

std::shared_ptr<Analyzer::Expr> rewrite_expr(const Analyzer::Expr* expr) {
  const auto expr_no_likelihood = strip_likelihood(expr);
  // The following check is not strictly needed, but seems silly to transform a
  // simple string comparison to an IN just to codegen the same thing anyway.
  const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr_no_likelihood);
  if (!bin_oper || bin_oper->get_optype() != kOR) {
    return nullptr;
  }
  OrToInVisitor visitor;
  auto rewritten_expr = visitor.visit(expr_no_likelihood);
  const auto expr_with_likelihood = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
  if (expr_with_likelihood) {
    // Add back likelihood
    return std::make_shared<Analyzer::LikelihoodExpr>(rewritten_expr, expr_with_likelihood->get_likelihood());
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

std::shared_ptr<Analyzer::Expr> redirect_expr(const Analyzer::Expr* expr,
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
    return std::make_shared<Analyzer::LikelihoodExpr>(rewritten_expr, expr_with_likelihood->get_likelihood());
  }
  return rewritten_expr;
}
