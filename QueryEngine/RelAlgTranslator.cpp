#ifdef HAVE_CALCITE
#include "RelAlgTranslator.h"

#include "CalciteDeserializerUtils.h"
#include "RelAlgAbstractInterpreter.h"

#include "../Analyzer/Analyzer.h"
#include "../Parser/ParserNode.h"

namespace {

SQLTypeInfo build_adjusted_type_info(const SQLTypes sql_type, const int type_scale, const int type_precision) {
  SQLTypeInfo type_ti(sql_type, 0, 0, false);
  type_ti.set_scale(type_scale);
  type_ti.set_precision(type_precision);
  if (type_ti.is_number() && !type_scale) {
    switch (type_precision) {
      case 5:
        return SQLTypeInfo(kSMALLINT, false);
      case 10:
        return SQLTypeInfo(kINT, false);
      case 19:
        return SQLTypeInfo(kBIGINT, false);
      default:
        CHECK(false);
    }
  }
  return type_ti;
}

SQLTypeInfo build_type_info(const SQLTypes sql_type, const int scale, const int precision) {
  SQLTypeInfo ti(sql_type, 0, 0, false);
  ti.set_scale(scale);
  ti.set_precision(precision);
  return ti;
}

std::shared_ptr<Analyzer::Expr> translate_literal(const RexLiteral* rex_literal) {
  const auto lit_ti = build_type_info(rex_literal->getType(), rex_literal->getScale(), rex_literal->getPrecision());
  const auto target_ti =
      build_adjusted_type_info(rex_literal->getType(), rex_literal->getTypeScale(), rex_literal->getTypePrecision());
  switch (rex_literal->getType()) {
    case kDECIMAL: {
      const auto val = rex_literal->getVal<int64_t>();
      const int precision = rex_literal->getPrecision();
      const int scale = rex_literal->getScale();
      auto lit_expr =
          scale ? Parser::FixedPtLiteral::analyzeValue(val, scale, precision) : Parser::IntLiteral::analyzeValue(val);
      return scale && lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kTEXT: {
      return Parser::StringLiteral::analyzeValue(rex_literal->getVal<std::string>());
    }
    case kBOOLEAN: {
      Datum d;
      d.boolval = rex_literal->getVal<bool>();
      return makeExpr<Analyzer::Constant>(kBOOLEAN, false, d);
    }
    case kDOUBLE: {
      Datum d;
      d.doubleval = rex_literal->getVal<double>();
      auto lit_expr = makeExpr<Analyzer::Constant>(kDOUBLE, false, d);
      return lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kNULLT: {
      return makeExpr<Analyzer::Constant>(kNULLT, true);
    }
    default: { LOG(FATAL) << "Unexpected literal type " << lit_ti.get_type_name(); }
  }
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> translate_input(const RexInput* rex_input,
                                                const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                                const Catalog_Namespace::Catalog& cat) {
  const auto source = rex_input->getSourceNode();
  const auto it_rte_idx = input_to_nest_level.find(source);
  CHECK(it_rte_idx != input_to_nest_level.end());
  const int rte_idx = it_rte_idx->second;
  const auto scan_source = dynamic_cast<const RelScan*>(source);
  const auto& in_metainfo = source->getOutputMetainfo();
  if (scan_source) {
    // We're at leaf (scan) level and not supposed to have input metadata,
    // the name and type information come directly from the catalog.
    CHECK(in_metainfo.empty());
    const auto& field_names = scan_source->getFieldNames();
    CHECK_LT(static_cast<size_t>(rex_input->getIndex()), field_names.size());
    const auto& col_name = field_names[rex_input->getIndex()];
    const auto table_desc = scan_source->getTableDescriptor();
    const auto cd = cat.getMetadataForColumn(table_desc->tableId, col_name);
    CHECK(cd);
    return std::make_shared<Analyzer::ColumnVar>(cd->columnType, table_desc->tableId, cd->columnId, rte_idx);
  }
  CHECK(!in_metainfo.empty());
  CHECK_GE(rte_idx, 0);
  const size_t col_id = rex_input->getIndex();
  CHECK_LT(col_id, in_metainfo.size());
  return std::make_shared<Analyzer::ColumnVar>(in_metainfo[col_id].get_type_info(), -source->getId(), col_id, rte_idx);
}

std::shared_ptr<Analyzer::Expr> remove_cast(const std::shared_ptr<Analyzer::Expr> expr) {
  const auto cast_expr = std::dynamic_pointer_cast<const Analyzer::UOper>(expr);
  return cast_expr && cast_expr->get_optype() == kCAST ? cast_expr->get_own_operand() : expr;
}

std::shared_ptr<Analyzer::Expr> translate_uoper(const RexOperator* rex_operator,
                                                const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                                const Catalog_Namespace::Catalog& cat) {
  CHECK_EQ(size_t(1), rex_operator->size());
  const auto operand_expr = translate_scalar_rex(rex_operator->getOperand(0), input_to_nest_level, cat);
  const auto sql_op = rex_operator->getOperator();
  switch (sql_op) {
    case kCAST: {
      const auto& target_ti = rex_operator->getType();
      CHECK_NE(kNULLT, target_ti.get_type());
      if (target_ti.is_time()) {  // TODO(alex): check and unify with the rest of the cases
        return operand_expr->add_cast(target_ti);
      }
      return std::make_shared<Analyzer::UOper>(target_ti, false, sql_op, operand_expr);
    }
    case kNOT:
    case kISNULL: {
      return std::make_shared<Analyzer::UOper>(kBOOLEAN, sql_op, operand_expr);
    }
    case kISNOTNULL: {
      auto is_null = std::make_shared<Analyzer::UOper>(kBOOLEAN, kISNULL, operand_expr);
      return std::make_shared<Analyzer::UOper>(kBOOLEAN, kNOT, is_null);
    }
    case kMINUS: {
      const auto& ti = operand_expr->get_type_info();
      return std::make_shared<Analyzer::UOper>(ti, false, kUMINUS, operand_expr);
    }
    case kUNNEST: {
      const auto& ti = operand_expr->get_type_info();
      CHECK(ti.is_array());
      return makeExpr<Analyzer::UOper>(ti.get_elem_type(), false, kUNNEST, operand_expr);
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

std::pair<std::shared_ptr<Analyzer::Expr>, SQLQualifier> get_quantified_rhs(
    const RexScalar* rex_scalar,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const Catalog_Namespace::Catalog& cat) {
  std::shared_ptr<Analyzer::Expr> rhs;
  SQLQualifier sql_qual{kONE};
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex_scalar);
  if (!rex_operator) {
    return std::make_pair(rhs, sql_qual);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex_operator);
  const auto qual_str = rex_function ? rex_function->getName() : "";
  if (qual_str == std::string("PG_ANY") || qual_str == std::string("PG_ALL")) {
    CHECK_EQ(size_t(1), rex_function->size());
    rhs = translate_scalar_rex(rex_function->getOperand(0), input_to_nest_level, cat);
    sql_qual = qual_str == std::string("PG_ANY") ? kANY : kALL;
  }
  if (!rhs && rex_operator->getOperator() == kCAST) {
    CHECK_EQ(size_t(1), rex_operator->size());
    std::tie(rhs, sql_qual) = get_quantified_rhs(rex_operator->getOperand(0), input_to_nest_level, cat);
  }
  return std::make_pair(rhs, sql_qual);
}

std::shared_ptr<Analyzer::Expr> translate_oper(const RexOperator* rex_operator,
                                               const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                               const Catalog_Namespace::Catalog& cat) {
  CHECK_GT(rex_operator->size(), size_t(0));
  if (rex_operator->size() == 1) {
    return translate_uoper(rex_operator, input_to_nest_level, cat);
  }
  const auto sql_op = rex_operator->getOperator();
  auto lhs = translate_scalar_rex(rex_operator->getOperand(0), input_to_nest_level, cat);
  for (size_t i = 1; i < rex_operator->size(); ++i) {
    std::shared_ptr<Analyzer::Expr> rhs;
    SQLQualifier sql_qual{kONE};
    const auto rhs_op = rex_operator->getOperand(i);
    std::tie(rhs, sql_qual) = get_quantified_rhs(rhs_op, input_to_nest_level, cat);
    if (!rhs) {
      rhs = translate_scalar_rex(rhs_op, input_to_nest_level, cat);
    }
    CHECK(rhs);
    if (sql_op == kEQ || sql_op == kNE) {
      lhs = remove_cast(lhs);
      rhs = remove_cast(rhs);
    }
    lhs = Parser::OperExpr::normalize(sql_op, sql_qual, lhs, rhs);
  }
  return lhs;
}

std::shared_ptr<Analyzer::Expr> translate_case(const RexCase* rex_case,
                                               const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                               const Catalog_Namespace::Catalog& cat) {
  std::shared_ptr<Analyzer::Expr> else_expr;
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>> expr_list;
  for (size_t i = 0; i < rex_case->branchCount(); ++i) {
    const auto when_expr = translate_scalar_rex(rex_case->getWhen(i), input_to_nest_level, cat);
    const auto then_expr = translate_scalar_rex(rex_case->getThen(i), input_to_nest_level, cat);
    expr_list.emplace_back(when_expr, then_expr);
  }
  if (rex_case->getElse()) {
    else_expr = translate_scalar_rex(rex_case->getElse(), input_to_nest_level, cat);
  }
  return Parser::CaseExpr::normalize(expr_list, else_expr);
}

std::shared_ptr<Analyzer::Expr> translate_like(const RexFunctionOperator* rex_function,
                                               const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                               const Catalog_Namespace::Catalog& cat) {
  CHECK(rex_function->size() == 2 || rex_function->size() == 3);
  const auto arg = translate_scalar_rex(rex_function->getOperand(0), input_to_nest_level, cat);
  const auto like = translate_scalar_rex(rex_function->getOperand(1), input_to_nest_level, cat);
  const auto escape = (rex_function->size() == 3)
                          ? translate_scalar_rex(rex_function->getOperand(2), input_to_nest_level, cat)
                          : nullptr;
  const bool is_ilike = rex_function->getName() == std::string("PG_ILIKE");
  return Parser::LikeExpr::get(arg, like, escape, is_ilike, false);
}

std::shared_ptr<Analyzer::Expr> translate_extract(const RexFunctionOperator* rex_function,
                                                  const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                                  const Catalog_Namespace::Catalog& cat) {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto timeunit = translate_scalar_rex(rex_function->getOperand(0), input_to_nest_level, cat);
  const auto timeunit_lit = std::dynamic_pointer_cast<Analyzer::Constant>(timeunit);
  if (!timeunit_lit) {
    throw std::runtime_error("The time unit parameter must be a literal.");
  }
  const auto from_expr = translate_scalar_rex(rex_function->getOperand(1), input_to_nest_level, cat);
  const bool is_date_trunc = rex_function->getName() == std::string("PG_DATE_TRUNC");
  return is_date_trunc ? Parser::DatetruncExpr::get(from_expr, *timeunit_lit->get_constval().stringval)
                       : Parser::ExtractExpr::get(from_expr, *timeunit_lit->get_constval().stringval);
}

std::shared_ptr<Analyzer::Expr> translate_length(const RexFunctionOperator* rex_function,
                                                 const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                                 const Catalog_Namespace::Catalog& cat) {
  CHECK_EQ(size_t(1), rex_function->size());
  const auto str_arg = translate_scalar_rex(rex_function->getOperand(0), input_to_nest_level, cat);
  return makeExpr<Analyzer::CharLengthExpr>(str_arg->decompress(),
                                            rex_function->getName() == std::string("CHAR_LENGTH"));
}

std::shared_ptr<Analyzer::Expr> translate_item(const RexFunctionOperator* rex_function,
                                               const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                                               const Catalog_Namespace::Catalog& cat) {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto base = translate_scalar_rex(rex_function->getOperand(0), input_to_nest_level, cat);
  const auto index = translate_scalar_rex(rex_function->getOperand(1), input_to_nest_level, cat);
  return makeExpr<Analyzer::BinOper>(base->get_type_info().get_elem_type(), false, kARRAY_AT, kONE, base, index);
}

std::shared_ptr<Analyzer::Expr> translate_function(
    const RexFunctionOperator* rex_function,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const Catalog_Namespace::Catalog& cat) {
  if (rex_function->getName() == std::string("LIKE") || rex_function->getName() == std::string("PG_ILIKE")) {
    return translate_like(rex_function, input_to_nest_level, cat);
  }
  if (rex_function->getName() == std::string("PG_EXTRACT") || rex_function->getName() == std::string("PG_DATE_TRUNC")) {
    return translate_extract(rex_function, input_to_nest_level, cat);
  }
  if (rex_function->getName() == std::string("LENGTH") || rex_function->getName() == std::string("CHAR_LENGTH")) {
    return translate_length(rex_function, input_to_nest_level, cat);
  }
  if (rex_function->getName() == std::string("ITEM")) {
    return translate_item(rex_function, input_to_nest_level, cat);
  }
  CHECK(false);
  return nullptr;
}

}  // namespace

std::shared_ptr<Analyzer::Expr> translate_scalar_rex(
    const RexScalar* rex,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const Catalog_Namespace::Catalog& cat) {
  const auto rex_input = dynamic_cast<const RexInput*>(rex);
  if (rex_input) {
    return translate_input(rex_input, input_to_nest_level, cat);
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex);
  if (rex_literal) {
    return translate_literal(rex_literal);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex);
  if (rex_function) {
    return translate_function(rex_function, input_to_nest_level, cat);
  }
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex);
  if (rex_operator) {
    return translate_oper(rex_operator, input_to_nest_level, cat);
  }
  const auto rex_case = dynamic_cast<const RexCase*>(rex);
  if (rex_case) {
    return translate_case(rex_case, input_to_nest_level, cat);
  }
  CHECK(false);
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> translate_aggregate_rex(
    const RexAgg* rex,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources) {
  const auto agg_kind = rex->getKind();
  const bool is_distinct = rex->isDistinct();
  const auto operand = rex->getOperand();
  const bool takes_arg{operand >= 0};
  if (takes_arg) {
    CHECK_LT(operand, static_cast<ssize_t>(scalar_sources.size()));
  }
  const auto arg_expr = takes_arg ? scalar_sources[operand] : nullptr;
  const auto agg_ti = get_agg_type(agg_kind, arg_expr.get());
  return makeExpr<Analyzer::AggExpr>(agg_ti, agg_kind, arg_expr, is_distinct);
}
#endif  // HAVE_CALCITE
