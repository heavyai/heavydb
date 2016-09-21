#ifndef QUERYENGINE_EXPRESSIONREWRITE_H
#define QUERYENGINE_EXPRESSIONREWRITE_H

#include <memory>
#include <list>
#include <vector>

namespace Analyzer {

class Expr;

class InValues;

}  // namespace Analyzer

class InputColDescriptor;

// Rewrites an OR tree where leaves are equality compare against literals.
std::shared_ptr<Analyzer::Expr> rewrite_expr(const Analyzer::Expr*);

std::list<std::shared_ptr<Analyzer::Expr>> redirect_exprs(
    const std::list<std::shared_ptr<Analyzer::Expr>>& exprs,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_descs);

std::vector<std::shared_ptr<Analyzer::Expr>> redirect_exprs(
    const std::vector<Analyzer::Expr*>& exprs,
    const std::list<std::shared_ptr<const InputColDescriptor>>& col_descs);

std::shared_ptr<Analyzer::Expr> redirect_expr(const Analyzer::Expr* expr,
                                              const std::list<std::shared_ptr<const InputColDescriptor>>& col_descs);

#endif  // QUERYENGINE_EXPRESSIONREWRITE_H
