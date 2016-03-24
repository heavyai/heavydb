#ifndef QUERYENGINE_SCALAREXPRVISITOR_H
#define QUERYENGINE_SCALAREXPRVISITOR_H

#include "../Analyzer/Analyzer.h"

template <class T>
class ScalarExprVisitor {
 public:
  T visit(const Analyzer::Expr* expr) const {
    CHECK(expr);
    const auto var = dynamic_cast<const Analyzer::Var*>(expr);
    if (var) {
      return visitVar(var);
    }
    const auto column_var = dynamic_cast<const Analyzer::ColumnVar*>(expr);
    if (column_var) {
      return visitColumnVar(column_var);
    }
    const auto constant = dynamic_cast<const Analyzer::Constant*>(expr);
    if (constant) {
      return visitConstant(constant);
    }
    const auto uoper = dynamic_cast<const Analyzer::UOper*>(expr);
    if (uoper) {
      return visitUOper(uoper);
    }
    const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
    if (bin_oper) {
      return visitBinOper(bin_oper);
    }
    const auto in_values = dynamic_cast<const Analyzer::InValues*>(expr);
    if (in_values) {
      return visitInValues(in_values);
    }
    const auto char_length = dynamic_cast<const Analyzer::CharLengthExpr*>(expr);
    if (char_length) {
      return visitCharLength(char_length);
    }
    const auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
    if (like_expr) {
      return visitLikeExpr(like_expr);
    }
    const auto case_ = dynamic_cast<const Analyzer::CaseExpr*>(expr);
    if (case_) {
      return visitCaseExpr(case_);
    }
    const auto datetrunc = dynamic_cast<const Analyzer::DatetruncExpr*>(expr);
    if (datetrunc) {
      return visitDatetruncExpr(datetrunc);
    }
    const auto extract = dynamic_cast<const Analyzer::ExtractExpr*>(expr);
    if (extract) {
      return visitExtractExpr(extract);
    }
    return defaultResult();
  }

 protected:
  virtual T visitVar(const Analyzer::Var*) const { return defaultResult(); }

  virtual T visitColumnVar(const Analyzer::ColumnVar*) const { return defaultResult(); }

  virtual T visitConstant(const Analyzer::Constant*) const { return defaultResult(); }

  virtual T visitUOper(const Analyzer::UOper* uoper) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(uoper->get_operand()));
    return result;
  }

  virtual T visitBinOper(const Analyzer::BinOper* bin_oper) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(bin_oper->get_left_operand()));
    result = aggregateResult(result, visit(bin_oper->get_right_operand()));
    return result;
  }

  virtual T visitInValues(const Analyzer::InValues* in_values) const {
    T result = defaultResult();
    const auto& value_list = in_values->get_value_list();
    for (const auto in_value : value_list) {
      result = aggregateResult(result, visit(in_value.get()));
    }
    return result;
  }

  virtual T visitCharLength(const Analyzer::CharLengthExpr* char_length) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(char_length->get_arg()));
    return result;
  }

  virtual T visitLikeExpr(const Analyzer::LikeExpr* like) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(like->get_arg()));
    result = aggregateResult(result, visit(like->get_like_expr()));
    if (like->get_escape_expr()) {
      result = aggregateResult(result, visit(like->get_escape_expr()));
    }
    return result;
  }

  virtual T visitCaseExpr(const Analyzer::CaseExpr* case_) const {
    T result = defaultResult();
    const auto& expr_pair_list = case_->get_expr_pair_list();
    for (const auto& expr_pair : expr_pair_list) {
      result = aggregateResult(result, visit(expr_pair.first.get()));
      result = aggregateResult(result, visit(expr_pair.second.get()));
    }
    return result;
  }

  virtual T visitDatetruncExpr(const Analyzer::DatetruncExpr* datetrunc) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(datetrunc->get_from_expr()));
    return result;
  }

  virtual T visitExtractExpr(const Analyzer::ExtractExpr* extract) const {
    T result = defaultResult();
    result = aggregateResult(result, visit(extract->get_from_expr()));
    return result;
  }

 protected:
  virtual T aggregateResult(const T& aggregate, const T& next_result) const { return next_result; }

  virtual T defaultResult() const { return T{}; }
};

#endif  // QUERYENGINE_SCALAREXPRVISITOR_H
