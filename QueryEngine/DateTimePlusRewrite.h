#ifndef QUERYENGINE_DATETIMEPLUSREWRITE_H
#define QUERYENGINE_DATETIMEPLUSREWRITE_H

#include <memory>

namespace Analyzer {

class Expr;

class FunctionOper;

}  // namespace Analyzer

std::shared_ptr<Analyzer::Expr> rewrite_to_date_trunc(const Analyzer::FunctionOper*);

#endif  // QUERYENGINE_DATETIMEPLUSREWRITE_H
