#include "ExpressionRewrite.h"

#include "../Analyzer/Analyzer.h"

#include <glog/logging.h>

namespace {

template <class T>
class ScalarExprVisitor {
 public:
  T visit(const Analyzer::Expr* expr) const {
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
    CHECK(false);
    return defaultResult();
  }

  virtual T visitVar(const Analyzer::Var*) { return defaultResult(); }

  virtual T visitColumnVar(const Analyzer::ColumnVar*) { return defaultResult(); }

  virtual T visitConstant(const Analyzer::Constant*) { return defaultResult(); }

  virtual T visitUOper(const Analyzer::UOper*) { return defaultResult(); }

  virtual T visitBinOper(const Analyzer::BinOper*) { return defaultResult(); }

  virtual T visitInValues(const Analyzer::InValues*) { return defaultResult(); }

  virtual T visitCharLength(const Analyzer::CharLengthExpr*) { return defaultResult(); }

  virtual T visitLikeExpr(const Analyzer::LikeExpr*) { return defaultResult(); }

  virtual T visitCaseExpr(const Analyzer::CaseExpr*) { return defaultResult(); }

  virtual T visitDatetruncExpr(const Analyzer::DatetruncExpr*) { return defaultResult(); }

  virtual T visitExtractExpr(const Analyzer::ExtractExpr*) { return defaultResult(); }

 protected:
  virtual T aggregateResult(const T& aggregate, const T& next_result) const { return next_result; }

  virtual T defaultResult() const { return T{}; }
};

}  // namespace

std::shared_ptr<Analyzer::InValues> or_to_in(const Analyzer::Expr*) {
  CHECK(false);
  return nullptr;
}
