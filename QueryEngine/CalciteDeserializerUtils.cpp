#include "CalciteDeserializerUtils.h"

#include "../Analyzer/Analyzer.h"

SQLTypeInfo get_agg_type(const SQLAgg agg_kind, const Analyzer::Expr* arg_expr) {
  switch (agg_kind) {
    case kCOUNT:
      return SQLTypeInfo(kINT, false);
    case kMIN:
    case kMAX:
      return arg_expr->get_type_info();
    case kSUM:
      return arg_expr->get_type_info().is_integer() ? SQLTypeInfo(kBIGINT, false) : arg_expr->get_type_info();
    case kAVG:
      return SQLTypeInfo(kDOUBLE, false);
    default:
      CHECK(false);
  }
  CHECK(false);
  return SQLTypeInfo();
}
