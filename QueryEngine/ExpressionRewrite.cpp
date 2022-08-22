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

#include "QueryEngine/ExpressionRewrite.h"

#include <algorithm>
#include <boost/locale/conversion.hpp>
#include <unordered_set>

#include "Analyzer/Analyzer.h"
#include "Logger/Logger.h"
#include "Parser/ParserNode.h"
#include "QueryEngine/DeepCopyVisitor.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "QueryEngine/ScalarExprVisitor.h"
#include "QueryEngine/WindowExpressionRewrite.h"
#include "Shared/sqldefs.h"
#include "StringOps/StringOps.h"

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

  std::shared_ptr<Analyzer::InValues> visitKeyForString(
      const Analyzer::KeyForStringExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitSampleRatio(
      const Analyzer::SampleRatioExpr*) const override {
    return nullptr;
  }

  std::shared_ptr<Analyzer::InValues> visitCardinality(
      const Analyzer::CardinalityExpr*) const override {
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

    if (lhs->get_arg()->get_type_info() == rhs->get_arg()->get_type_info() &&
        (*lhs->get_arg() == *rhs->get_arg())) {
      auto union_values = lhs->get_value_list();
      const auto& rhs_values = rhs->get_value_list();
      union_values.insert(union_values.end(), rhs_values.begin(), rhs_values.end());
      return makeExpr<Analyzer::InValues>(lhs->get_own_arg(), union_values);
    }
    return nullptr;
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
        type_info, args_copy, array_expr->isNull(), array_expr->isLocalAlloc());
  }
};

class ConstantFoldingVisitor : public DeepCopyVisitor {
  template <typename T>
  bool foldComparison(SQLOps optype, T t1, T t2) const {
    switch (optype) {
      case kEQ:
        return t1 == t2;
      case kNE:
        return t1 != t2;
      case kLT:
        return t1 < t2;
      case kLE:
        return t1 <= t2;
      case kGT:
        return t1 > t2;
      case kGE:
        return t1 >= t2;
      default:
        break;
    }
    throw std::runtime_error("Unable to fold");
    return false;
  }

  template <typename T>
  bool foldLogic(SQLOps optype, T t1, T t2) const {
    switch (optype) {
      case kAND:
        return t1 && t2;
      case kOR:
        return t1 || t2;
      case kNOT:
        return !t1;
      default:
        break;
    }
    throw std::runtime_error("Unable to fold");
    return false;
  }

  template <typename T>
  T foldArithmetic(SQLOps optype, T t1, T t2) const {
    bool t2_is_zero = (t2 == (t2 - t2));
    bool t2_is_negative = (t2 < (t2 - t2));
    switch (optype) {
      case kPLUS:
        // The MIN limit for float and double is the smallest representable value,
        // not the lowest negative value! Switching to C++11 lowest.
        if ((t2_is_negative && t1 < std::numeric_limits<T>::lowest() - t2) ||
            (!t2_is_negative && t1 > std::numeric_limits<T>::max() - t2)) {
          num_overflows_++;
          throw std::runtime_error("Plus overflow");
        }
        return t1 + t2;
      case kMINUS:
        if ((t2_is_negative && t1 > std::numeric_limits<T>::max() + t2) ||
            (!t2_is_negative && t1 < std::numeric_limits<T>::lowest() + t2)) {
          num_overflows_++;
          throw std::runtime_error("Minus overflow");
        }
        return t1 - t2;
      case kMULTIPLY: {
        if (t2_is_zero) {
          return t2;
        }
        auto ct1 = t1;
        auto ct2 = t2;
        // Need to keep t2's sign on the left
        if (t2_is_negative) {
          if (t1 == std::numeric_limits<T>::lowest() ||
              t2 == std::numeric_limits<T>::lowest()) {
            // negation could overflow - bail
            num_overflows_++;
            throw std::runtime_error("Mul neg overflow");
          }
          ct1 = -t1;  // ct1 gets t2's negativity
          ct2 = -t2;  // ct2 is now positive
        }
        // Don't check overlow if we are folding FP mul by a fraction
        bool ct2_is_fraction = (ct2 < (ct2 / ct2));
        if (!ct2_is_fraction) {
          if (ct1 > std::numeric_limits<T>::max() / ct2 ||
              ct1 < std::numeric_limits<T>::lowest() / ct2) {
            num_overflows_++;
            throw std::runtime_error("Mul overflow");
          }
        }
        return t1 * t2;
      }
      case kDIVIDE:
        if (t2_is_zero) {
          throw std::runtime_error("Will not fold division by zero");
        }
        return t1 / t2;
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
            result.boolval = foldComparison<bool>(optype, lhs.boolval, rhs.boolval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_LOGIC(optype)) {
            result.boolval = foldLogic<bool>(optype, lhs.boolval, rhs.boolval);
            result_type = kBOOLEAN;
            return true;
          }
          CHECK(!IS_ARITHMETIC(optype));
          break;
        case kTINYINT:
          if (IS_COMPARISON(optype)) {
            result.boolval =
                foldComparison<int8_t>(optype, lhs.tinyintval, rhs.tinyintval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.tinyintval =
                foldArithmetic<int8_t>(optype, lhs.tinyintval, rhs.tinyintval);
            result_type = kTINYINT;
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kSMALLINT:
          if (IS_COMPARISON(optype)) {
            result.boolval =
                foldComparison<int16_t>(optype, lhs.smallintval, rhs.smallintval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.smallintval =
                foldArithmetic<int16_t>(optype, lhs.smallintval, rhs.smallintval);
            result_type = kSMALLINT;
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kINT:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison<int32_t>(optype, lhs.intval, rhs.intval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.intval = foldArithmetic<int32_t>(optype, lhs.intval, rhs.intval);
            result_type = kINT;
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kBIGINT:
          if (IS_COMPARISON(optype)) {
            result.boolval =
                foldComparison<int64_t>(optype, lhs.bigintval, rhs.bigintval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.bigintval =
                foldArithmetic<int64_t>(optype, lhs.bigintval, rhs.bigintval);
            result_type = kBIGINT;
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kFLOAT:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison<float>(optype, lhs.floatval, rhs.floatval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.floatval = foldArithmetic<float>(optype, lhs.floatval, rhs.floatval);
            result_type = kFLOAT;
            return true;
          }
          CHECK(!IS_LOGIC(optype));
          break;
        case kDOUBLE:
          if (IS_COMPARISON(optype)) {
            result.boolval = foldComparison<double>(optype, lhs.doubleval, rhs.doubleval);
            result_type = kBOOLEAN;
            return true;
          }
          if (IS_ARITHMETIC(optype)) {
            result.doubleval =
                foldArithmetic<double>(optype, lhs.doubleval, rhs.doubleval);
            result_type = kDOUBLE;
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
    const auto& ti = uoper->get_type_info();
    if (optype == kCAST) {
      // Cache the cast type so it could be used in operand rewriting/folding
      casts_.insert({unvisited_operand, ti});
    }
    const auto operand = visit(unvisited_operand);

    const auto& operand_ti = operand->get_type_info();
    const auto operand_type =
        operand_ti.is_decimal() ? decimal_to_int_type(operand_ti) : operand_ti.get_type();
    const auto const_operand =
        std::dynamic_pointer_cast<const Analyzer::Constant>(operand);

    if (const_operand) {
      const auto operand_datum = const_operand->get_constval();
      Datum zero_datum = {};
      Datum result_datum = {};
      SQLTypes result_type;
      switch (optype) {
        case kNOT: {
          if (foldOper(kEQ,
                       operand_type,
                       zero_datum,
                       operand_datum,
                       result_datum,
                       result_type)) {
            CHECK_EQ(result_type, kBOOLEAN);
            return makeExpr<Analyzer::Constant>(result_type, false, result_datum);
          }
          break;
        }
        case kUMINUS: {
          if (foldOper(kMINUS,
                       operand_type,
                       zero_datum,
                       operand_datum,
                       result_datum,
                       result_type)) {
            if (!operand_ti.is_decimal()) {
              return makeExpr<Analyzer::Constant>(result_type, false, result_datum);
            }
            return makeExpr<Analyzer::Constant>(ti, false, result_datum);
          }
          break;
        }
        case kCAST: {
          // Trying to fold number to number casts only
          if (!ti.is_number() || !operand_ti.is_number()) {
            break;
          }
          // Disallowing folding of FP to DECIMAL casts for now:
          // allowing them would make this test pass:
          //    update dectest set d=cast( 1234.0 as float );
          // which is expected to throw in Update.ImplicitCastToNumericTypes
          // due to cast codegen currently not supporting these casts
          if (ti.is_decimal() && operand_ti.is_fp()) {
            break;
          }
          auto operand_copy = const_operand->deep_copy();
          auto cast_operand = operand_copy->add_cast(ti);
          auto const_cast_operand =
              std::dynamic_pointer_cast<const Analyzer::Constant>(cast_operand);
          if (const_cast_operand) {
            auto const_cast_datum = const_cast_operand->get_constval();
            return makeExpr<Analyzer::Constant>(ti, false, const_cast_datum);
          }
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
    const auto optype = bin_oper->get_optype();
    auto ti = bin_oper->get_type_info();
    auto left_operand = bin_oper->get_own_left_operand();
    auto right_operand = bin_oper->get_own_right_operand();

    // Check if bin_oper result is cast to a larger int or fp type
    if (casts_.find(bin_oper) != casts_.end()) {
      const auto cast_ti = casts_[bin_oper];
      const auto& lhs_ti = bin_oper->get_left_operand()->get_type_info();
      // Propagate cast down to the operands for folding
      if ((cast_ti.is_integer() || cast_ti.is_fp()) && lhs_ti.is_integer() &&
          cast_ti.get_size() > lhs_ti.get_size() &&
          (optype == kMINUS || optype == kPLUS || optype == kMULTIPLY)) {
        // Before folding, cast the operands to the bigger type to avoid overflows.
        // Currently upcasting smaller integer types to larger integers or double.
        left_operand = left_operand->deep_copy()->add_cast(cast_ti);
        right_operand = right_operand->deep_copy()->add_cast(cast_ti);
        ti = cast_ti;
      }
    }

    const auto lhs = visit(left_operand.get());
    const auto rhs = visit(right_operand.get());

    auto const_lhs = std::dynamic_pointer_cast<Analyzer::Constant>(lhs);
    auto const_rhs = std::dynamic_pointer_cast<Analyzer::Constant>(rhs);
    const auto& lhs_ti = lhs->get_type_info();
    const auto& rhs_ti = rhs->get_type_info();
    auto lhs_type = lhs_ti.is_decimal() ? decimal_to_int_type(lhs_ti) : lhs_ti.get_type();
    auto rhs_type = rhs_ti.is_decimal() ? decimal_to_int_type(rhs_ti) : rhs_ti.get_type();

    if (const_lhs && const_rhs && lhs_type == rhs_type) {
      auto lhs_datum = const_lhs->get_constval();
      auto rhs_datum = const_rhs->get_constval();
      Datum result_datum = {};
      SQLTypes result_type;
      if (foldOper(optype, lhs_type, lhs_datum, rhs_datum, result_datum, result_type)) {
        // Fold all ops that don't take in decimal operands, and also decimal comparisons
        if (!lhs_ti.is_decimal() || IS_COMPARISON(optype)) {
          return makeExpr<Analyzer::Constant>(result_type, false, result_datum);
        }
        // Decimal arithmetic has been done as kBIGINT. Selectively fold some decimal ops,
        // using result_datum and BinOper expr typeinfo which was adjusted for these ops.
        if (optype == kMINUS || optype == kPLUS || optype == kMULTIPLY) {
          return makeExpr<Analyzer::Constant>(ti, false, result_datum);
        }
      }
    }

    if (optype == kAND && lhs_type == rhs_type && lhs_type == kBOOLEAN) {
      if (const_rhs && !const_rhs->get_is_null()) {
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
      if (const_lhs && !const_lhs->get_is_null()) {
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
      if (const_rhs && !const_rhs->get_is_null()) {
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
      if (const_lhs && !const_lhs->get_is_null()) {
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
    if (*lhs == *rhs) {
      if (!lhs_ti.get_notnull()) {
        CHECK(!rhs_ti.get_notnull());
        // We can't fold the ostensible tautaulogy
        // for nullable lhs and rhs types, as
        // lhs <> rhs when they are null

        // We likely could turn this into a lhs is not null
        // operatation, but is it worth it?
        return makeExpr<Analyzer::BinOper>(ti,
                                           bin_oper->get_contains_agg(),
                                           bin_oper->get_optype(),
                                           bin_oper->get_qualifier(),
                                           lhs,
                                           rhs);
      }
      CHECK(rhs_ti.get_notnull());
      // Tautologies: v=v; v<=v; v>=v
      if (optype == kEQ || optype == kLE || optype == kGE) {
        Datum d;
        d.boolval = true;
        return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
      }
      // Contradictions: v!=v; v<v; v>v
      if (optype == kNE || optype == kLT || optype == kGT) {
        Datum d;
        d.boolval = false;
        return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
      }
      // v-v
      if (optype == kMINUS) {
        Datum d = {};
        return makeExpr<Analyzer::Constant>(lhs_type, false, d);
      }
    }
    // Convert fp division by a constant to multiplication by 1/constant
    if (optype == kDIVIDE && const_rhs && rhs_ti.is_fp()) {
      auto rhs_datum = const_rhs->get_constval();
      std::shared_ptr<Analyzer::Expr> recip_rhs = nullptr;
      if (rhs_ti.get_type() == kFLOAT) {
        if (rhs_datum.floatval == 1.0) {
          return lhs;
        }
        auto f = std::fabs(rhs_datum.floatval);
        if (f > 1.0 || (f != 0.0 && 1.0 < f * std::numeric_limits<float>::max())) {
          rhs_datum.floatval = 1.0 / rhs_datum.floatval;
          recip_rhs = makeExpr<Analyzer::Constant>(rhs_type, false, rhs_datum);
        }
      } else if (rhs_ti.get_type() == kDOUBLE) {
        if (rhs_datum.doubleval == 1.0) {
          return lhs;
        }
        auto d = std::fabs(rhs_datum.doubleval);
        if (d > 1.0 || (d != 0.0 && 1.0 < d * std::numeric_limits<double>::max())) {
          rhs_datum.doubleval = 1.0 / rhs_datum.doubleval;
          recip_rhs = makeExpr<Analyzer::Constant>(rhs_type, false, rhs_datum);
        }
      }
      if (recip_rhs) {
        return makeExpr<Analyzer::BinOper>(ti,
                                           bin_oper->get_contains_agg(),
                                           kMULTIPLY,
                                           bin_oper->get_qualifier(),
                                           lhs,
                                           recip_rhs);
      }
    }

    return makeExpr<Analyzer::BinOper>(ti,
                                       bin_oper->get_contains_agg(),
                                       bin_oper->get_optype(),
                                       bin_oper->get_qualifier(),
                                       lhs,
                                       rhs);
  }

  std::shared_ptr<Analyzer::Expr> visitStringOper(
      const Analyzer::StringOper* string_oper) const override {
    // Todo(todd): For clarity and modularity we should move string
    // operator rewrites into their own visitor class.
    // String operation rewrites were originally put here as they only
    // handled string operators on rewrite, but now handle variable
    // inputs as well.
    const auto original_args = string_oper->getOwnArgs();
    std::vector<std::shared_ptr<Analyzer::Expr>> rewritten_args;
    const auto non_literal_arity = string_oper->getNonLiteralsArity();
    const auto parent_in_string_op_chain = in_string_op_chain_;
    const auto in_string_op_chain = non_literal_arity <= 1UL;
    in_string_op_chain_ = in_string_op_chain;

    size_t rewritten_arg_literal_arity = 0;
    for (auto original_arg : original_args) {
      rewritten_args.emplace_back(visit(original_arg.get()));
      if (dynamic_cast<const Analyzer::Constant*>(rewritten_args.back().get())) {
        rewritten_arg_literal_arity++;
      }
    }
    in_string_op_chain_ = parent_in_string_op_chain;
    const auto kind = string_oper->get_kind();
    const auto& return_ti = string_oper->get_type_info();

    if (string_oper->getArity() == rewritten_arg_literal_arity) {
      Analyzer::StringOper literal_string_oper(
          kind, string_oper->get_type_info(), rewritten_args);
      const auto literal_args = literal_string_oper.getLiteralArgs();
      const auto string_op_info =
          StringOps_Namespace::StringOpInfo(kind, return_ti, literal_args);
      if (return_ti.is_string()) {
        const auto literal_result =
            StringOps_Namespace::apply_string_op_to_literals(string_op_info);
        return Parser::StringLiteral::analyzeValue(literal_result.first,
                                                   literal_result.second);
      }
      const auto literal_datum =
          StringOps_Namespace::apply_numeric_op_to_literals(string_op_info);
      auto nullable_return_ti = return_ti;
      nullable_return_ti.set_notnull(false);
      return makeExpr<Analyzer::Constant>(nullable_return_ti,
                                          IsNullDatum(literal_datum, nullable_return_ti),
                                          literal_datum);
    }
    chained_string_op_exprs_.emplace_back(
        makeExpr<Analyzer::StringOper>(kind, return_ti, rewritten_args));
    if (parent_in_string_op_chain && in_string_op_chain) {
      CHECK(rewritten_args[0]->get_type_info().is_string());
      return rewritten_args[0]->deep_copy();
    } else {
      auto new_string_oper = makeExpr<Analyzer::StringOper>(
          kind, return_ti, rewritten_args, chained_string_op_exprs_);
      chained_string_op_exprs_.clear();
      return new_string_oper;
    }
  }

 protected:
  mutable bool in_string_op_chain_{false};
  mutable std::vector<std::shared_ptr<Analyzer::Expr>> chained_string_op_exprs_;
  mutable std::unordered_map<const Analyzer::Expr*, const SQLTypeInfo> casts_;
  mutable int32_t num_overflows_;

 public:
  ConstantFoldingVisitor() : num_overflows_(0) {}
  int32_t get_num_overflows() { return num_overflows_; }
  void reset_num_overflows() { num_overflows_ = 0; }
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
  const auto sum_window = rewrite_sum_window(expr);
  if (sum_window) {
    return sum_window;
  }
  const auto avg_window = rewrite_avg_window(expr);
  if (avg_window) {
    return avg_window;
  }
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

boost::optional<OverlapsJoinConjunction> rewrite_overlaps_conjunction(
    const std::shared_ptr<Analyzer::Expr> expr,
    const std::vector<InputDescriptor>& input_table_info,
    const OverlapsJoinRewriteType rewrite_type,
    const Executor* executor) {
  auto collect_table_cardinality = [](const Analyzer::Expr* lhs,
                                      const Analyzer::Expr* rhs,
                                      const Executor* executor) {
    const auto lhs_cv = dynamic_cast<const Analyzer::ColumnVar*>(lhs);
    const auto rhs_cv = dynamic_cast<const Analyzer::ColumnVar*>(rhs);
    if (lhs_cv && rhs_cv) {
      const auto cat = executor->getCatalog();
      const auto inner_table_metadata = cat->getMetadataForTable(lhs_cv->get_table_id());
      const auto outer_table_metadata = cat->getMetadataForTable(rhs_cv->get_table_id());
      if (inner_table_metadata->fragmenter && outer_table_metadata->fragmenter) {
        return std::make_pair<int64_t, int64_t>(
            inner_table_metadata->fragmenter->getNumRows(),
            outer_table_metadata->fragmenter->getNumRows());
      }
    }
    // otherwise, return an invalid table cardinality
    return std::make_pair<int64_t, int64_t>(-1, -1);
  };

  auto has_invalid_join_col_order = [](const Analyzer::Expr* lhs,
                                       const Analyzer::Expr* rhs) {
    // Check for compatible join ordering. If the join ordering does not match expected
    // ordering for overlaps, the join builder will fail.
    std::set<int> lhs_rte_idx;
    lhs->collect_rte_idx(lhs_rte_idx);
    CHECK(!lhs_rte_idx.empty());
    std::set<int> rhs_rte_idx;
    rhs->collect_rte_idx(rhs_rte_idx);
    CHECK(!rhs_rte_idx.empty());
    auto has_invalid_num_join_cols = lhs_rte_idx.size() > 1 || rhs_rte_idx.size() > 1;
    auto has_invalid_rte_idx = lhs_rte_idx > rhs_rte_idx;
    return std::make_pair(has_invalid_num_join_cols || has_invalid_rte_idx,
                          has_invalid_rte_idx);
  };

  auto convert_to_range_join_oper =
      [&](std::string_view func_name,
          const std::shared_ptr<Analyzer::Expr> expr,
          const Analyzer::BinOper* range_join_expr,
          const Analyzer::GeoOperator* lhs,
          const Analyzer::Constant* rhs,
          const Executor* executor) -> std::shared_ptr<Analyzer::BinOper> {
    if (OverlapsJoinSupportedFunction::is_range_join_rewrite_target_func(func_name)) {
      CHECK_EQ(lhs->size(), size_t(2));
      auto l_arg = lhs->getOperand(0);
      auto r_arg = lhs->getOperand(1);
      const bool is_geography = l_arg->get_type_info().get_subtype() == kGEOGRAPHY ||
                                r_arg->get_type_info().get_subtype() == kGEOGRAPHY;
      if (is_geography) {
        VLOG(1) << "Range join not yet supported for geodesic distance "
                << expr->toString();
        return nullptr;
      }

      // Check for compatible join ordering. If the join ordering does not match expected
      // ordering for overlaps, the join builder will fail.
      Analyzer::Expr* range_join_arg = r_arg;
      Analyzer::Expr* bin_oper_arg = l_arg;
      auto invalid_range_join_qual =
          has_invalid_join_col_order(bin_oper_arg, range_join_arg);
      if (invalid_range_join_qual.second &&
          collect_table_cardinality(range_join_arg, bin_oper_arg, executor).first > 0 &&
          lhs->getOperand(0)->get_type_info().get_type() == kPOINT) {
        r_arg = lhs->getOperand(0);
        l_arg = lhs->getOperand(1);
        VLOG(1) << "Swap range join qual's input arguments to exploit overlaps "
                   "hash join framework";
        invalid_range_join_qual.first = false;
      }

      if (invalid_range_join_qual.first) {
        LOG(INFO) << "Unable to rewrite " << func_name
                  << " to overlaps conjunction. Cannot build hash table over LHS type. "
                     "Check join order.\n"
                  << range_join_expr->toString();
        return nullptr;
      }

      const bool inclusive = range_join_expr->get_optype() == kLE;
      auto range_expr = makeExpr<Analyzer::RangeOper>(
          inclusive, inclusive, r_arg->deep_copy(), rhs->deep_copy());
      VLOG(1) << "Successfully converted to overlaps join";
      return makeExpr<Analyzer::BinOper>(
          kBOOLEAN, kOVERLAPS, kONE, l_arg->deep_copy(), range_expr);
    }
    return nullptr;
  };

  /*
   * Currently, our overlaps hash join framework supports limited join quals especially
   * when 1) the FunctionOperator is listed in the function list, i.e.,
   * is_overlaps_supported_func, 2) the argument order of the join qual must match the
   * input argument order of the corresponding native function, and 3) input tables match
   * rte index requirement (the column used to build a hash table has larger rte compared
   * with that of probing column) And depending on the type of the function, we try to
   * convert it to corresponding overlaps hash join qual if possible After rewriting, we
   * create an overlaps join operator which is converted from the original expression and
   * return OverlapsJoinConjunction object which is a pair of 1) the original expr and 2)
   * converted overlaps join expr Here, returning the original expr means we additionally
   * call its corresponding native function to compute the result accurately (i.e.,
   * overlaps hash join operates a kind of filter expression which may include
   * false-positive of the true resultset) Note that ST_Overlaps is the only function that
   * does not return the original expr
   * */
  std::shared_ptr<Analyzer::BinOper> overlaps_oper{nullptr};
  bool needs_to_return_original_expr = false;
  std::string func_name{""};
  if (rewrite_type == OverlapsJoinRewriteType::OVERLAPS_JOIN) {
    auto func_oper = dynamic_cast<Analyzer::FunctionOper*>(expr.get());
    CHECK(func_oper);
    func_name = func_oper->getName();
    if (!g_enable_hashjoin_many_to_many &&
        OverlapsJoinSupportedFunction::is_many_to_many_func(func_name)) {
      LOG(WARNING) << "Many-to-many hashjoin support is disabled, unable to rewrite "
                   << func_oper->toString() << " to use accelerated geo join.";
      return boost::none;
    }
    DeepCopyVisitor deep_copy_visitor;
    if (func_name == OverlapsJoinSupportedFunction::ST_OVERLAPS_sv) {
      CHECK_GE(func_oper->getArity(), size_t(2));
      // this case returns {empty quals, overlaps join quals} b/c our join key matching
      // logic for this case is the same as the implementation of ST_Overlaps function
      // Note that we can build an overlaps join hash table regardless of table ordering
      // and the argument order in this case b/c selecting lhs and rhs by arguments 0
      // and 1 always match the rte index requirement (rte_lhs < rte_rhs)
      // so what table ordering we take, the rte index requirement satisfies
      // TODO(adb): we will likely want to actually check for true overlaps, but this
      // works for now
      auto lhs = func_oper->getOwnArg(0);
      auto rewritten_lhs = deep_copy_visitor.visit(lhs.get());
      CHECK(rewritten_lhs);

      auto rhs = func_oper->getOwnArg(1);
      auto rewritten_rhs = deep_copy_visitor.visit(rhs.get());
      CHECK(rewritten_rhs);
      overlaps_oper = makeExpr<Analyzer::BinOper>(
          kBOOLEAN, kOVERLAPS, kONE, rewritten_lhs, rewritten_rhs);
    } else if (func_name == OverlapsJoinSupportedFunction::ST_DWITHIN_POINT_POINT_sv) {
      CHECK_EQ(func_oper->getArity(), size_t(8));
      const auto lhs = func_oper->getOwnArg(0);
      const auto rhs = func_oper->getOwnArg(1);
      // the correctness of geo args used in the ST_DWithin function is checked by
      // geo translation logic, i.e., RelAlgTranslator::translateTernaryGeoFunction
      const auto distance_const_val =
          dynamic_cast<const Analyzer::Constant*>(func_oper->getArg(7));
      if (lhs && rhs && distance_const_val) {
        std::vector<std::shared_ptr<Analyzer::Expr>> args{lhs, rhs};
        auto range_oper = makeExpr<Analyzer::GeoOperator>(
            SQLTypeInfo(kDOUBLE, 0, 8, true),
            OverlapsJoinSupportedFunction::ST_DISTANCE_sv.data(),
            args,
            std::nullopt);
        auto distance_oper = makeExpr<Analyzer::BinOper>(
            kBOOLEAN, kLE, kONE, range_oper, distance_const_val->deep_copy());
        VLOG(1) << "Rewrite " << func_oper->getName() << " to ST_Distance_Point_Point";
        overlaps_oper =
            convert_to_range_join_oper(OverlapsJoinSupportedFunction::ST_DISTANCE_sv,
                                       distance_oper,
                                       distance_oper.get(),
                                       range_oper.get(),
                                       distance_const_val,
                                       executor);
        needs_to_return_original_expr = true;
      }
    } else if (OverlapsJoinSupportedFunction::is_poly_mpoly_rewrite_target_func(
                   func_name)) {
      // in the five functions fall into this case,
      // ST_Contains is for a pair of polygons, and for ST_Intersect cases they are
      // combo of polygon and multipolygon so what table orders we choose, rte index
      // requirement for overlaps join can be satisfied if we choose lhs and rhs
      // from left-to-right order (i.e., get lhs from the arg-1 instead of arg-3)
      // Note that we choose them from right-to-left argument order in the past
      CHECK_GE(func_oper->getArity(), size_t(4));
      auto lhs = func_oper->getOwnArg(1);
      auto rewritten_lhs = deep_copy_visitor.visit(lhs.get());
      CHECK(rewritten_lhs);
      auto rhs = func_oper->getOwnArg(3);
      auto rewritten_rhs = deep_copy_visitor.visit(rhs.get());
      CHECK(rewritten_rhs);

      overlaps_oper = makeExpr<Analyzer::BinOper>(
          kBOOLEAN, kOVERLAPS, kONE, rewritten_lhs, rewritten_rhs);
      needs_to_return_original_expr = true;
    } else if (OverlapsJoinSupportedFunction::is_point_poly_rewrite_target_func(
                   func_name)) {
      // now, we try to look at one more chance to exploit overlaps hash join by
      // rewriting the qual as: ST_INTERSECT(POLY, POINT) -> ST_INTERSECT(POINT, POLY)
      // to support efficient evaluation of 1) ST_Intersects_Point_Polygon and
      // 2) ST_Intersects_Point_MultiPolygon based on our overlaps hash join framework
      // here, we have implementation of native functions for both 1) Point-Polygon pair
      // and 2) Polygon-Point pair, but we currently do not support hash table
      // generation on top of point column thus, the goal of this rewriting is to place
      // a non-point geometry to the right-side of the overlaps join operator (to build
      // hash table based on it) iff the inner table is larger than that of non-point
      // geometry (to reduce expensive hash join performance)
      size_t point_arg_idx = 0;
      size_t poly_arg_idx = 2;
      if (func_oper->getOwnArg(point_arg_idx)->get_type_info().get_type() != kPOINT) {
        point_arg_idx = 2;
        poly_arg_idx = 1;
      }
      auto point_cv = func_oper->getOwnArg(point_arg_idx);
      auto poly_cv = func_oper->getOwnArg(poly_arg_idx);
      CHECK_EQ(point_cv->get_type_info().get_type(), kPOINT);
      CHECK_EQ(poly_cv->get_type_info().get_type(), kARRAY);
      auto rewritten_lhs = deep_copy_visitor.visit(point_cv.get());
      CHECK(rewritten_lhs);
      auto rewritten_rhs = deep_copy_visitor.visit(poly_cv.get());
      CHECK(rewritten_rhs);
      VLOG(1) << "Rewriting the " << func_name << " to use overlaps join with lhs as "
              << rewritten_lhs->toString() << " and rhs as " << rewritten_rhs->toString();
      overlaps_oper = makeExpr<Analyzer::BinOper>(
          kBOOLEAN, kOVERLAPS, kONE, rewritten_lhs, rewritten_rhs);
      needs_to_return_original_expr = true;
    } else if (OverlapsJoinSupportedFunction::is_poly_point_rewrite_target_func(
                   func_name)) {
      // rest of functions reaching here is poly and point geo join query
      // to use overlaps hash join in this case, poly column must have its rte == 1
      // lhs is the point col_var
      auto lhs = func_oper->getOwnArg(2);
      auto rewritten_lhs = deep_copy_visitor.visit(lhs.get());
      CHECK(rewritten_lhs);
      const auto& lhs_ti = rewritten_lhs->get_type_info();

      if (!lhs_ti.is_geometry() && !is_constructed_point(rewritten_lhs.get())) {
        // TODO(adb): If ST_Contains is passed geospatial literals instead of columns,
        // the function will be expanded during translation rather than during code
        // generation. While this scenario does not make sense for the overlaps join, we
        // need to detect and abort the overlaps rewrite. Adding a GeospatialConstant
        // dervied class to the Analyzer may prove to be a better way to handle geo
        // literals, but for now we ensure the LHS type is a geospatial type, which
        // would mean the function has not been expanded to the physical types, yet.
        LOG(INFO) << "Unable to rewrite " << func_name
                  << " to overlaps conjunction. LHS input type is neither a geospatial "
                     "column nor a constructed point"
                  << func_oper->toString();
        return boost::none;
      }

      // rhs is coordinates of the poly col
      auto rhs = func_oper->getOwnArg(1);
      auto rewritten_rhs = deep_copy_visitor.visit(rhs.get());
      CHECK(rewritten_rhs);

      if (has_invalid_join_col_order(lhs.get(), rhs.get()).first) {
        LOG(INFO) << "Unable to rewrite " << func_name
                  << " to overlaps conjunction. Cannot build hash table over LHS type. "
                     "Check join order."
                  << func_oper->toString();
        return boost::none;
      }

      VLOG(1) << "Rewriting " << func_name << " to use overlaps join with lhs as "
              << rewritten_lhs->toString() << " and rhs as " << rewritten_rhs->toString();

      overlaps_oper = makeExpr<Analyzer::BinOper>(
          kBOOLEAN, kOVERLAPS, kONE, rewritten_lhs, rewritten_rhs);
      if (func_name !=
          OverlapsJoinSupportedFunction::ST_APPROX_OVERLAPS_MULTIPOLYGON_POINT_sv) {
        needs_to_return_original_expr = true;
      }
    }
  } else if (rewrite_type == OverlapsJoinRewriteType::RANGE_JOIN) {
    auto bin_oper = dynamic_cast<Analyzer::BinOper*>(expr.get());
    CHECK(bin_oper);
    auto lhs = dynamic_cast<const Analyzer::GeoOperator*>(bin_oper->get_left_operand());
    CHECK(lhs);
    auto rhs = dynamic_cast<const Analyzer::Constant*>(bin_oper->get_right_operand());
    CHECK(rhs);
    func_name = lhs->getName();
    overlaps_oper =
        convert_to_range_join_oper(func_name, expr, bin_oper, lhs, rhs, executor);
    needs_to_return_original_expr = true;
  }
  const auto expr_str = !func_name.empty() ? func_name : expr->toString();
  if (overlaps_oper) {
    VLOG(1) << "Successfully converted " << expr_str << " to overlaps join";
    if (needs_to_return_original_expr) {
      return OverlapsJoinConjunction{{expr}, {overlaps_oper}};
    } else {
      return OverlapsJoinConjunction{{}, {overlaps_oper}};
    }
  }
  VLOG(1) << "Overlaps join not enabled for " << expr_str;
  return boost::none;
}

/**
 * JoinCoveredQualVisitor returns true if the visited qual is true if and only if a
 * corresponding equijoin qual is true. During the pre-filtered count we can elide the
 * visited qual decreasing query run time while upper bounding the number of rows passing
 * the filter. Currently only used for expressions of the form `a OVERLAPS b AND Expr<a,
 * b>`. Strips `Expr<a,b>` if the expression has been pre-determined to be expensive to
 * compute twice.
 */
class JoinCoveredQualVisitor : public ScalarExprVisitor<bool> {
 public:
  JoinCoveredQualVisitor(const JoinQualsPerNestingLevel& join_quals) {
    for (const auto& join_condition : join_quals) {
      for (const auto& qual : join_condition.quals) {
        auto qual_bin_oper = dynamic_cast<Analyzer::BinOper*>(qual.get());
        if (qual_bin_oper) {
          join_qual_pairs.emplace_back(qual_bin_oper->get_left_operand(),
                                       qual_bin_oper->get_right_operand());
        }
      }
    }
  }

  bool visitFunctionOper(const Analyzer::FunctionOper* func_oper) const override {
    if (OverlapsJoinSupportedFunction::is_overlaps_supported_func(func_oper->getName())) {
      const auto lhs = func_oper->getArg(2);
      const auto rhs = func_oper->getArg(1);
      for (const auto& qual_pair : join_qual_pairs) {
        if (*lhs == *qual_pair.first && *rhs == *qual_pair.second) {
          return true;
        }
      }
    }
    return false;
  }

  bool defaultResult() const override { return false; }

 private:
  std::vector<std::pair<const Analyzer::Expr*, const Analyzer::Expr*>> join_qual_pairs;
};

std::list<std::shared_ptr<Analyzer::Expr>> strip_join_covered_filter_quals(
    const std::list<std::shared_ptr<Analyzer::Expr>>& quals,
    const JoinQualsPerNestingLevel& join_quals) {
  if (!g_strip_join_covered_quals) {
    return quals;
  }

  if (join_quals.empty()) {
    return quals;
  }

  std::list<std::shared_ptr<Analyzer::Expr>> quals_to_return;

  JoinCoveredQualVisitor visitor(join_quals);
  for (const auto& qual : quals) {
    if (!visitor.visit(qual.get())) {
      // Not a covered qual, don't elide it from the filtered count
      quals_to_return.push_back(qual);
    }
  }

  return quals_to_return;
}

std::shared_ptr<Analyzer::Expr> fold_expr(const Analyzer::Expr* expr) {
  if (!expr) {
    return nullptr;
  }
  const auto expr_no_likelihood = strip_likelihood(expr);
  ConstantFoldingVisitor visitor;
  auto rewritten_expr = visitor.visit(expr_no_likelihood);
  if (visitor.get_num_overflows() > 0 && rewritten_expr->get_type_info().is_integer() &&
      rewritten_expr->get_type_info().get_type() != kBIGINT) {
    auto rewritten_expr_const =
        std::dynamic_pointer_cast<const Analyzer::Constant>(rewritten_expr);
    if (!rewritten_expr_const) {
      // Integer expression didn't fold completely the first time due to
      // overflows in smaller type subexpressions, trying again with a cast
      const auto& ti = SQLTypeInfo(kBIGINT, false);
      auto bigint_expr_no_likelihood = expr_no_likelihood->deep_copy()->add_cast(ti);
      auto rewritten_expr_take2 = visitor.visit(bigint_expr_no_likelihood.get());
      auto rewritten_expr_take2_const =
          std::dynamic_pointer_cast<Analyzer::Constant>(rewritten_expr_take2);
      if (rewritten_expr_take2_const) {
        // Managed to fold, switch to the new constant
        rewritten_expr = rewritten_expr_take2_const;
      }
    }
  }
  const auto expr_with_likelihood = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
  if (expr_with_likelihood) {
    // Add back likelihood
    return std::make_shared<Analyzer::LikelihoodExpr>(
        rewritten_expr, expr_with_likelihood->get_likelihood());
  }
  return rewritten_expr;
}

bool self_join_not_covered_by_left_deep_tree(const Analyzer::ColumnVar* key_side,
                                             const Analyzer::ColumnVar* val_side,
                                             const int max_rte_covered) {
  if (key_side->get_table_id() == val_side->get_table_id() &&
      key_side->get_rte_idx() == val_side->get_rte_idx() &&
      key_side->get_rte_idx() > max_rte_covered) {
    return true;
  }
  return false;
}

const int get_max_rte_scan_table(
    std::unordered_map<int, llvm::Value*>& scan_idx_to_hash_pos) {
  int ret = INT32_MIN;
  for (auto& kv : scan_idx_to_hash_pos) {
    if (kv.first > ret) {
      ret = kv.first;
    }
  }
  return ret;
}
