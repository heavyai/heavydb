#ifdef HAVE_CALCITE
#include "RelAlgTranslator.h"
#include "SqlTypesLayout.h"

#include "CalciteDeserializerUtils.h"
#include "ExtensionFunctionsWhitelist.h"
#include "RelAlgAbstractInterpreter.h"

#include "../Analyzer/Analyzer.h"
#include "../Parser/ParserNode.h"

namespace {

SQLTypeInfo build_type_info(const SQLTypes sql_type, const int scale, const int precision) {
  SQLTypeInfo ti(sql_type, 0, 0, false);
  ti.set_scale(scale);
  ti.set_precision(precision);
  return ti;
}

std::shared_ptr<Analyzer::Expr> remove_cast(const std::shared_ptr<Analyzer::Expr> expr) {
  const auto cast_expr = std::dynamic_pointer_cast<const Analyzer::UOper>(expr);
  return cast_expr && cast_expr->get_optype() == kCAST ? cast_expr->get_own_operand() : expr;
}

std::pair<std::shared_ptr<Analyzer::Expr>, SQLQualifier> get_quantified_rhs(const RexScalar* rex_scalar,
                                                                            const RelAlgTranslator& translator) {
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
    rhs = translator.translateScalarRex(rex_function->getOperand(0));
    sql_qual = qual_str == std::string("PG_ANY") ? kANY : kALL;
  }
  if (!rhs && rex_operator->getOperator() == kCAST) {
    CHECK_EQ(size_t(1), rex_operator->size());
    std::tie(rhs, sql_qual) = get_quantified_rhs(rex_operator->getOperand(0), translator);
  }
  return std::make_pair(rhs, sql_qual);
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateScalarRex(const RexScalar* rex) const {
  const auto rex_input = dynamic_cast<const RexInput*>(rex);
  if (rex_input) {
    return translateInput(rex_input);
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex);
  if (rex_literal) {
    return translateLiteral(rex_literal);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex);
  if (rex_function) {
    return translateFunction(rex_function);
  }
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex);
  if (rex_operator) {
    return translateOper(rex_operator);
  }
  const auto rex_case = dynamic_cast<const RexCase*>(rex);
  if (rex_case) {
    return translateCase(rex_case);
  }
  CHECK(false);
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateAggregateRex(
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

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateLiteral(const RexLiteral* rex_literal) {
  const auto lit_ti = build_type_info(rex_literal->getType(), rex_literal->getScale(), rex_literal->getPrecision());
  const auto target_ti =
      build_type_info(rex_literal->getTargetType(), rex_literal->getTypeScale(), rex_literal->getTypePrecision());
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
      return makeExpr<Analyzer::Constant>(rex_literal->getTargetType(), true, Datum{0});
    }
    default: { LOG(FATAL) << "Unexpected literal type " << lit_ti.get_type_name(); }
  }
  return nullptr;
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateInput(const RexInput* rex_input) const {
  const auto source = rex_input->getSourceNode();
  const auto it_rte_idx = input_to_nest_level_.find(source);
  CHECK(it_rte_idx != input_to_nest_level_.end());
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
    const auto cd = cat_.getMetadataForColumn(table_desc->tableId, col_name);
    CHECK(cd);
    auto col_ti = cd->columnType;
    if (rte_idx > 0 && join_type_ == JoinType::LEFT) {
      col_ti.set_notnull(false);
    }
    return std::make_shared<Analyzer::ColumnVar>(col_ti, table_desc->tableId, cd->columnId, rte_idx);
  }
  CHECK(!in_metainfo.empty());
  CHECK_GE(rte_idx, 0);
  const size_t col_id = rex_input->getIndex();
  CHECK_LT(col_id, in_metainfo.size());
  auto col_ti = in_metainfo[col_id].get_type_info();
  if (rte_idx > 0 && join_type_ == JoinType::LEFT) {
    col_ti.set_notnull(false);
  }
  return std::make_shared<Analyzer::ColumnVar>(col_ti, -source->getId(), col_id, rte_idx);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateUoper(const RexOperator* rex_operator) const {
  CHECK_EQ(size_t(1), rex_operator->size());
  const auto operand_expr = translateScalarRex(rex_operator->getOperand(0));
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

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateOper(const RexOperator* rex_operator) const {
  CHECK_GT(rex_operator->size(), size_t(0));
  if (rex_operator->size() == 1) {
    return translateUoper(rex_operator);
  }
  const auto sql_op = rex_operator->getOperator();
  auto lhs = translateScalarRex(rex_operator->getOperand(0));
  for (size_t i = 1; i < rex_operator->size(); ++i) {
    std::shared_ptr<Analyzer::Expr> rhs;
    SQLQualifier sql_qual{kONE};
    const auto rhs_op = rex_operator->getOperand(i);
    std::tie(rhs, sql_qual) = get_quantified_rhs(rhs_op, *this);
    if (!rhs) {
      rhs = translateScalarRex(rhs_op);
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

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateCase(const RexCase* rex_case) const {
  std::shared_ptr<Analyzer::Expr> else_expr;
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>> expr_list;
  for (size_t i = 0; i < rex_case->branchCount(); ++i) {
    const auto when_expr = translateScalarRex(rex_case->getWhen(i));
    const auto then_expr = translateScalarRex(rex_case->getThen(i));
    expr_list.emplace_back(when_expr, then_expr);
  }
  if (rex_case->getElse()) {
    else_expr = translateScalarRex(rex_case->getElse());
  }
  return Parser::CaseExpr::normalize(expr_list, else_expr);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateLike(const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 2 || rex_function->size() == 3);
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto like = translateScalarRex(rex_function->getOperand(1));
  const auto escape = (rex_function->size() == 3) ? translateScalarRex(rex_function->getOperand(2)) : nullptr;
  const bool is_ilike = rex_function->getName() == std::string("PG_ILIKE");
  return Parser::LikeExpr::get(arg, like, escape, is_ilike, false);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateExtract(const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<Analyzer::Constant>(timeunit);
  if (!timeunit_lit) {
    throw std::runtime_error("The time unit parameter must be a literal.");
  }
  const auto from_expr = translateScalarRex(rex_function->getOperand(1));
  const bool is_date_trunc = rex_function->getName() == std::string("PG_DATE_TRUNC");
  return is_date_trunc ? Parser::DatetruncExpr::get(from_expr, *timeunit_lit->get_constval().stringval)
                       : Parser::ExtractExpr::get(from_expr, *timeunit_lit->get_constval().stringval);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateDatediff(const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(3), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<Analyzer::Constant>(timeunit);
  if (!timeunit_lit) {
    throw std::runtime_error("The time unit parameter must be a literal.");
  }
  const auto start = translateScalarRex(rex_function->getOperand(1));
  const auto end = translateScalarRex(rex_function->getOperand(2));
  return makeExpr<Analyzer::DatediffExpr>(
      SQLTypeInfo(kBIGINT, false), to_datediff_field(*timeunit_lit->get_constval().stringval), start, end);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateDatepart(const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<Analyzer::Constant>(timeunit);
  if (!timeunit_lit) {
    throw std::runtime_error("The time unit parameter must be a literal.");
  }
  const auto from_expr = translateScalarRex(rex_function->getOperand(1));
  return Parser::ExtractExpr::get(from_expr, to_datepart_field(*timeunit_lit->get_constval().stringval));
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateLength(const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());
  const auto str_arg = translateScalarRex(rex_function->getOperand(0));
  return makeExpr<Analyzer::CharLengthExpr>(str_arg->decompress(),
                                            rex_function->getName() == std::string("CHAR_LENGTH"));
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateItem(const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto base = translateScalarRex(rex_function->getOperand(0));
  const auto index = translateScalarRex(rex_function->getOperand(1));
  return makeExpr<Analyzer::BinOper>(base->get_type_info().get_elem_type(), false, kARRAY_AT, kONE, base, index);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateNow() const {
  return Parser::TimestampLiteral::get(now_);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateDatetime(const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto arg_lit = std::dynamic_pointer_cast<Analyzer::Constant>(arg);
  const std::string datetime_err{R"(Only DATETIME('NOW') supported for now.)"};
  if (!arg_lit) {
    throw std::runtime_error(datetime_err);
  }
  CHECK(arg_lit->get_type_info().is_string());
  if (*arg_lit->get_constval().stringval != std::string("NOW")) {
    throw std::runtime_error(datetime_err);
  }
  return translateNow();
}

namespace {

std::shared_ptr<Analyzer::Constant> makeNumericConstant(const SQLTypeInfo& ti, const int val) {
  CHECK(ti.is_number());
  Datum datum{0};
  switch (ti.get_type()) {
    case kSMALLINT: {
      datum.smallintval = val;
      break;
    }
    case kINT: {
      datum.intval = val;
      break;
    }
    case kBIGINT: {
      datum.bigintval = val;
      break;
    }
    case kDECIMAL:
    case kNUMERIC: {
      datum.bigintval = val * exp_to_scale(ti.get_scale());
      break;
    }
    case kFLOAT: {
      datum.floatval = val;
      break;
    }
    case kDOUBLE: {
      datum.doubleval = val;
      break;
    }
    default:
      CHECK(false);
  }
  return makeExpr<Analyzer::Constant>(ti, false, datum);
}

}  // namespace

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateAbs(const RexFunctionOperator* rex_function) const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>> expr_list;
  CHECK_EQ(size_t(1), rex_function->size());
  const auto operand = translateScalarRex(rex_function->getOperand(0));
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_number());
  const auto zero = makeNumericConstant(operand_ti, 0);
  const auto lt_zero = makeExpr<Analyzer::BinOper>(kBOOLEAN, kLT, kONE, operand, zero);
  const auto uminus_operand = makeExpr<Analyzer::UOper>(operand_ti.get_type(), kUMINUS, operand);
  expr_list.emplace_back(lt_zero, uminus_operand);
  return makeExpr<Analyzer::CaseExpr>(operand_ti, false, expr_list, operand);
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateSign(const RexFunctionOperator* rex_function) const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>> expr_list;
  CHECK_EQ(size_t(1), rex_function->size());
  const auto operand = translateScalarRex(rex_function->getOperand(0));
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_number());
  const auto zero = makeNumericConstant(operand_ti, 0);
  const auto lt_zero = makeExpr<Analyzer::BinOper>(kBOOLEAN, kLT, kONE, operand, zero);
  expr_list.emplace_back(lt_zero, makeNumericConstant(operand_ti, -1));
  const auto eq_zero = makeExpr<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, operand, zero);
  expr_list.emplace_back(eq_zero, makeNumericConstant(operand_ti, 0));
  const auto gt_zero = makeExpr<Analyzer::BinOper>(kBOOLEAN, kGT, kONE, operand, zero);
  expr_list.emplace_back(gt_zero, makeNumericConstant(operand_ti, 1));
  return makeExpr<Analyzer::CaseExpr>(
      operand_ti, false, expr_list, makeExpr<Analyzer::Constant>(operand_ti, true, Datum{0}));
}

std::shared_ptr<Analyzer::Expr> RelAlgTranslator::translateFunction(const RexFunctionOperator* rex_function) const {
  if (rex_function->getName() == std::string("LIKE") || rex_function->getName() == std::string("PG_ILIKE")) {
    return translateLike(rex_function);
  }
  if (rex_function->getName() == std::string("PG_EXTRACT") || rex_function->getName() == std::string("PG_DATE_TRUNC")) {
    return translateExtract(rex_function);
  }
  if (rex_function->getName() == std::string("DATEDIFF")) {
    return translateDatediff(rex_function);
  }
  if (rex_function->getName() == std::string("DATEPART")) {
    return translateDatepart(rex_function);
  }
  if (rex_function->getName() == std::string("LENGTH") || rex_function->getName() == std::string("CHAR_LENGTH")) {
    return translateLength(rex_function);
  }
  if (rex_function->getName() == std::string("ITEM")) {
    return translateItem(rex_function);
  }
  if (rex_function->getName() == std::string("NOW")) {
    return translateNow();
  }
  if (rex_function->getName() == std::string("DATETIME")) {
    return translateDatetime(rex_function);
  }
  if (rex_function->getName() == std::string("ABS")) {
    return translateAbs(rex_function);
  }
  if (rex_function->getName() == std::string("SIGN")) {
    return translateSign(rex_function);
  }
  if (rex_function->getName() == std::string("CEIL") || rex_function->getName() == std::string("FLOOR") ||
      rex_function->getName() == std::string("TRUNCATE")) {
    return makeExpr<Analyzer::FunctionOperWithCustomTypeHandling>(
        rex_function->getType(), rex_function->getName(), translateFunctionArgs(rex_function));
  }
  if (!ExtensionFunctionsWhitelist::get(rex_function->getName())) {
    throw QueryNotSupported("Function " + rex_function->getName() + " not supported");
  }
  return makeExpr<Analyzer::FunctionOper>(
      rex_function->getType(), rex_function->getName(), translateFunctionArgs(rex_function));
}

std::vector<std::shared_ptr<Analyzer::Expr>> RelAlgTranslator::translateFunctionArgs(
    const RexFunctionOperator* rex_function) const {
  std::vector<std::shared_ptr<Analyzer::Expr>> args;
  for (size_t i = 0; i < rex_function->size(); ++i) {
    args.push_back(translateScalarRex(rex_function->getOperand(i)));
  }
  return args;
}
#endif  // HAVE_CALCITE
