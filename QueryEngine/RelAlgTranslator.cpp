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

#include "RelAlgTranslator.h"
#include "CalciteDeserializerUtils.h"
#include "DateTimePlusRewrite.h"
#include "DateTimeTranslator.h"
#include "DeepCopyVisitor.h"
#include "ExpressionRewrite.h"
#include "ExtensionFunctionsBinding.h"
#include "ExtensionFunctionsWhitelist.h"
#include "RelAlgDagBuilder.h"
#include "WindowContext.h"

#include "Analyzer/Analyzer.h"
#include "Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/SessionInfo.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/likely.h"
#include "Shared/thread_count.h"

#include <future>

namespace {

SQLTypeInfo build_type_info(const SQLTypes sql_type,
                            const int scale,
                            const int precision) {
  SQLTypeInfo ti(sql_type, 0, 0, true);
  if (ti.is_decimal()) {
    ti.set_scale(scale);
    ti.set_precision(precision);
  }
  return ti;
}

std::pair<hdk::ir::ExprPtr, SQLQualifier> get_quantified_rhs(
    const RexScalar* rex_scalar,
    const RelAlgTranslator& translator) {
  hdk::ir::ExprPtr rhs;
  SQLQualifier sql_qual{kONE};
  const auto rex_operator = dynamic_cast<const RexOperator*>(rex_scalar);
  if (!rex_operator) {
    return std::make_pair(rhs, sql_qual);
  }
  const auto rex_function = dynamic_cast<const RexFunctionOperator*>(rex_operator);
  const auto qual_str = rex_function ? rex_function->getName() : "";
  if (qual_str == "PG_ANY"sv || qual_str == "PG_ALL"sv) {
    CHECK_EQ(size_t(1), rex_function->size());
    rhs = translator.translateScalarRex(rex_function->getOperand(0));
    sql_qual = (qual_str == "PG_ANY"sv) ? kANY : kALL;
  }
  if (!rhs && rex_operator->getOperator() == kCAST) {
    CHECK_EQ(size_t(1), rex_operator->size());
    std::tie(rhs, sql_qual) = get_quantified_rhs(rex_operator->getOperand(0), translator);
  }
  return std::make_pair(rhs, sql_qual);
}

std::pair<Datum, bool> datum_from_scalar_tv(const ScalarTargetValue* scalar_tv,
                                            const SQLTypeInfo& ti) noexcept {
  Datum d{0};
  bool is_null_const{false};
  switch (ti.get_type()) {
    case kBOOLEAN: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.boolval = *ival;
      }
      break;
    }
    case kTINYINT: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.tinyintval = *ival;
      }
      break;
    }
    case kSMALLINT: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.smallintval = *ival;
      }
      break;
    }
    case kINT: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.intval = *ival;
      }
      break;
    }
    case kDECIMAL:
    case kNUMERIC:
    case kBIGINT:
    case kDATE:
    case kTIME:
    case kTIMESTAMP: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_val(ti)) {
        is_null_const = true;
      } else {
        d.bigintval = *ival;
      }
      break;
    }
    case kDOUBLE: {
      const auto dval = boost::get<double>(scalar_tv);
      CHECK(dval);
      if (*dval == inline_fp_null_val(ti)) {
        is_null_const = true;
      } else {
        d.doubleval = *dval;
      }
      break;
    }
    case kFLOAT: {
      const auto fval = boost::get<float>(scalar_tv);
      CHECK(fval);
      if (*fval == inline_fp_null_val(ti)) {
        is_null_const = true;
      } else {
        d.floatval = *fval;
      }
      break;
    }
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      auto nullable_sptr = boost::get<NullableString>(scalar_tv);
      CHECK(nullable_sptr);
      if (boost::get<void*>(nullable_sptr)) {
        is_null_const = true;
      } else {
        auto sptr = boost::get<std::string>(nullable_sptr);
        d.stringval = new std::string(*sptr);
      }
      break;
    }
    default:
      CHECK(false) << "Unhandled type: " << ti.get_type_name();
  }
  return {d, is_null_const};
}

hdk::ir::ExprPtr translateScalarSubqueryResult(
    std::shared_ptr<const ExecutionResult> result) {
  auto row_set = result->getRows();
  const size_t row_count = row_set->rowCount();
  CHECK_EQ(size_t(1), row_set->colCount());
  auto ti = row_set->getColType(0);
  if (row_count > size_t(1)) {
    throw std::runtime_error("Scalar sub-query returned multiple rows");
  }
  if (row_count == size_t(0)) {
    if (row_set->isValidationOnlyRes()) {
      Datum d{0};
      return hdk::ir::makeExpr<hdk::ir::Constant>(ti, false, d);
    }
    throw std::runtime_error("Scalar sub-query returned no results");
  }
  CHECK_EQ(row_count, size_t(1));
  row_set->moveToBegin();
  auto first_row = row_set->getNextRow(false, false);
  CHECK_EQ(first_row.size(), size_t(1));
  auto scalar_tv = boost::get<ScalarTargetValue>(&first_row[0]);
  if (ti.is_string()) {
    throw std::runtime_error("Scalar sub-queries which return strings not supported");
  }
  Datum d{0};
  bool is_null_const{false};
  std::tie(d, is_null_const) = datum_from_scalar_tv(scalar_tv, ti);
  return hdk::ir::makeExpr<hdk::ir::Constant>(ti, is_null_const, d);
}

class NormalizerVisitor : public DeepCopyVisitor {
 public:
  NormalizerVisitor(const RelAlgTranslator& translator,
                    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                    const std::vector<JoinType>& join_types,
                    const Executor* executor)
      : translator_(translator)
      , input_to_nest_level_(input_to_nest_level)
      , join_types_(join_types)
      , executor_(executor) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    auto source = col_ref->getNode();
    auto col_idx = col_ref->getIndex();
    const auto it_rte_idx = input_to_nest_level_.find(source);
    const int rte_idx = it_rte_idx == input_to_nest_level_.end() ? 0 : it_rte_idx->second;
    const auto scan_source = dynamic_cast<const RelScan*>(source);
    const auto& in_metainfo = source->getOutputMetainfo();
    if (scan_source) {
      // We're at leaf (scan) level and not supposed to have input metadata,
      // the name and type information come directly from the catalog.
      CHECK(in_metainfo.empty());
      auto col_info = scan_source->getColumnInfoBySpi(col_idx + 1);
      auto col_ti = col_info->type;
      if (col_ti.is_string()) {
        col_ti.set_type(kTEXT);
      }
      CHECK_LE(static_cast<size_t>(rte_idx), join_types_.size());
      if (rte_idx > 0 && join_types_[rte_idx - 1] == JoinType::LEFT) {
        col_ti.set_notnull(false);
      }
      return std::make_shared<hdk::ir::ColumnVar>(col_info, rte_idx);
    }
    CHECK_GE(rte_idx, 0);
    SQLTypeInfo col_ti;
    CHECK(!in_metainfo.empty()) << "for " << source->toString();
    CHECK_LT(col_idx, in_metainfo.size());
    col_ti = in_metainfo[col_idx].get_type_info();

    {
      // TODO: remove check when rex are removed.
      // We expect logical types of column ref type to match output meta.
      auto meta_type = col_ti;  // in_metainfo[col_idx].get_physical_type_info();
      auto col_type = get_logical_type_info(col_ref->get_type_info());
      // Allow diff in nullable flag.
      col_type.set_notnull(meta_type.get_notnull());
      // During DAG build we don't have enough information to properly choose between
      // dictionaries. It's OK because normalization will add all required casts.
      if (col_type.is_dict_encoded_string()) {
        CHECK(meta_type.is_dict_encoded_string());
        col_type.set_comp_param(meta_type.get_comp_param());
      }
      // Cast to BIGINT for COUNT aggregates.
      bool distinct_count = false;
      if (auto agg_node = dynamic_cast<const RelAggregate*>(source)) {
        if (col_ref->getIndex() >= agg_node->getGroupByCount()) {
          auto expr =
              agg_node
                  ->getAggregateExprs()[col_ref->getIndex() - agg_node->getGroupByCount()]
                  .get();
          if (auto agg = dynamic_cast<const hdk::ir::AggExpr*>(expr)) {
            if (agg->get_is_distinct()) {
              distinct_count = true;
            }
          }
        }
      }
      if (auto proj_node = dynamic_cast<const RelProject*>(source)) {
        auto expr = proj_node->getExprs()[col_ref->getIndex()].get();
        if (auto agg = dynamic_cast<const hdk::ir::AggExpr*>(expr)) {
          if (agg->get_is_distinct()) {
            distinct_count = true;
          }
        }
      }
      if (auto compound_node = dynamic_cast<const RelCompound*>(source)) {
        auto expr = compound_node->getExprs()[col_ref->getIndex()].get();
        if (auto agg = dynamic_cast<const hdk::ir::AggExpr*>(expr)) {
          if (agg->get_is_distinct()) {
            distinct_count = true;
          }
        }
      }
      if (distinct_count) {
        CHECK(col_type.is_integer()) << col_type.toString();
      } else if (meta_type.is_string()) {
        CHECK(col_type.is_string()) << "meta_type=" << meta_type.toString() << std::endl
                                    << "col_type=" << col_type.toString() << std::endl
                                    << "col_ref=" << col_ref->toString() << std::endl
                                    << "source=" << source->toString() << std::endl;
      } else {
        CHECK(meta_type == col_type)
            << "Type mismatch:" << std::endl
            << "meta_type=" << meta_type.toString() << std::endl
            << "col_type=" << col_type.toString() << std::endl
            << "col_ref=" << col_ref->toString() << std::endl
            << "source=" << source->toString() << std::endl
            << "getColumnType=" << getColumnType(source, col_ref->getIndex()).toString();
      }
    }

    if (join_types_.size() > 0) {
      CHECK_LE(static_cast<size_t>(rte_idx), join_types_.size());
      if (rte_idx > 0 && join_types_[rte_idx - 1] == JoinType::LEFT) {
        col_ti.set_notnull(false);
      }
    }

    return std::make_shared<hdk::ir::ColumnVar>(
        col_ti, -source->getId(), col_idx, rte_idx);
  }

  hdk::ir::ExprPtr visitBinOper(const hdk::ir::BinOper* bin_oper) const override {
    auto lhs = visit(bin_oper->get_left_operand());
    auto rhs = visit(bin_oper->get_right_operand());
    // Some binary expressions are not results of operation translation. They are
    // not covered in Analyzer normalization.
    if (bin_oper->get_optype() == kARRAY_AT) {
      return hdk::ir::makeExpr<hdk::ir::BinOper>(
          bin_oper->get_type_info(),
          lhs->get_contains_agg() || rhs->get_contains_agg(),
          bin_oper->get_optype(),
          bin_oper->get_qualifier(),
          std::move(lhs),
          std::move(rhs));
    }
    return Analyzer::normalizeOperExpr(bin_oper->get_optype(),
                                       bin_oper->get_qualifier(),
                                       std::move(lhs),
                                       std::move(rhs),
                                       executor_);
  }

  hdk::ir::ExprPtr visitUOper(const hdk::ir::UOper* uoper) const override {
    // Casts introduced on DAG build stage might become NOPs.
    if (uoper->get_optype() == kCAST) {
      auto op = visit(uoper->get_operand());
      if (uoper->get_type_info() == op->get_type_info()) {
        return op;
      }
      return hdk::ir::makeExpr<hdk::ir::UOper>(
          uoper->get_type_info(), op->get_contains_agg(), kCAST, op);
    }
    return DeepCopyVisitor::visitUOper(uoper);
  }

  hdk::ir::ExprPtr visitCaseExpr(const hdk::ir::CaseExpr* case_expr) const override {
    std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
    for (auto& pr : case_expr->get_expr_pair_list()) {
      expr_list.emplace_back(visit(pr.first.get()), visit(pr.second.get()));
    }
    auto else_expr = visit(case_expr->get_else_expr());
    return Analyzer::normalizeCaseExpr(expr_list, else_expr, executor_);
  }

  hdk::ir::ExprPtr visitScalarSubquery(
      const hdk::ir::ScalarSubquery* subquery) const override {
    return translateScalarSubqueryResult(subquery->getNode()->getResult());
  }

  hdk::ir::ExprPtr visitInSubquery(const hdk::ir::InSubquery* subquery) const override {
    return translator_.translateInSubquery(visit(subquery->getArg().get()),
                                           subquery->getNode()->getResult());
  }

 private:
  const RelAlgTranslator& translator_;
  const std::unordered_map<const RelAlgNode*, int> input_to_nest_level_;
  const std::vector<JoinType> join_types_;
  const Executor* executor_;
};

}  // namespace

RelAlgTranslator::RelAlgTranslator(
    const Executor* executor,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const std::vector<JoinType>& join_types,
    const time_t now,
    const bool just_explain)
    : executor_(executor)
    , config_(executor->getConfig())
    , input_to_nest_level_(input_to_nest_level)
    , join_types_(join_types)
    , now_(now)
    , just_explain_(just_explain)
    , for_dag_builder_(false) {}

RelAlgTranslator::RelAlgTranslator(const Config& config,
                                   const time_t now,
                                   const bool just_explain)
    : executor_(nullptr)
    , config_(config)
    , now_(now)
    , just_explain_(just_explain)
    , for_dag_builder_(true) {}

hdk::ir::ExprPtr RelAlgTranslator::translateScalarRex(const RexScalar* rex) const {
  const auto rex_input = dynamic_cast<const RexInput*>(rex);
  if (rex_input) {
    return translateInput(rex_input);
  }
  const auto rex_literal = dynamic_cast<const RexLiteral*>(rex);
  if (rex_literal) {
    return translateLiteral(rex_literal);
  }
  const auto rex_window_function = dynamic_cast<const RexWindowFunctionOperator*>(rex);
  if (rex_window_function) {
    return translateWindowFunction(rex_window_function);
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
  const auto rex_subquery = dynamic_cast<const RexSubQuery*>(rex);
  if (rex_subquery) {
    return translateScalarSubquery(rex_subquery);
  }
  CHECK(false);
  return nullptr;
}

namespace {

bool is_agg_supported_for_type(const SQLAgg& agg_kind, const SQLTypeInfo& arg_ti) {
  if ((agg_kind == kMIN || agg_kind == kMAX || agg_kind == kSUM || agg_kind == kAVG) &&
      !(arg_ti.is_number() || arg_ti.is_boolean() || arg_ti.is_time())) {
    return false;
  }

  return true;
}

}  // namespace

hdk::ir::ExprPtr RelAlgTranslator::translateAggregateRex(
    const RexAgg* rex,
    const std::vector<hdk::ir::ExprPtr>& scalar_sources,
    bool bigint_count) {
  SQLAgg agg_kind = rex->getKind();
  const bool is_distinct = rex->isDistinct();
  const bool takes_arg{rex->size() > 0};
  hdk::ir::ExprPtr arg_expr;
  std::shared_ptr<hdk::ir::Constant> arg1;  // 2nd aggregate parameter
  if (takes_arg) {
    const auto operand = rex->getOperand(0);
    CHECK_LT(operand, scalar_sources.size());
    CHECK_LE(rex->size(), 2u);
    arg_expr = scalar_sources[operand];
    if (agg_kind == kAPPROX_COUNT_DISTINCT && rex->size() == 2) {
      arg1 = std::dynamic_pointer_cast<hdk::ir::Constant>(
          scalar_sources[rex->getOperand(1)]);
      if (!arg1 || arg1->get_type_info().get_type() != kINT ||
          arg1->get_constval().intval < 1 || arg1->get_constval().intval > 100) {
        throw std::runtime_error(
            "APPROX_COUNT_DISTINCT's second parameter should be SMALLINT literal between "
            "1 and 100");
      }
    } else if (agg_kind == kAPPROX_QUANTILE) {
      // If second parameter is not given then APPROX_MEDIAN is assumed.
      if (rex->size() == 2) {
        arg1 = std::dynamic_pointer_cast<hdk::ir::Constant>(
            std::dynamic_pointer_cast<hdk::ir::Constant>(
                scalar_sources[rex->getOperand(1)])
                ->add_cast(SQLTypeInfo(kDOUBLE)));
      } else {
#ifdef _WIN32
        Datum median;
        median.doubleval = 0.5;
#else
        constexpr Datum median{.doubleval = 0.5};
#endif
        arg1 = std::make_shared<hdk::ir::Constant>(kDOUBLE, false, median);
      }
    }
    const auto& arg_ti = arg_expr->get_type_info();
    if (!is_agg_supported_for_type(agg_kind, arg_ti)) {
      throw std::runtime_error("Aggregate on " + arg_ti.get_type_name() +
                               " is not supported yet.");
    }
  }
  const auto agg_ti = get_agg_type(agg_kind, arg_expr.get(), bigint_count);
  return hdk::ir::makeExpr<hdk::ir::AggExpr>(
      agg_ti, agg_kind, arg_expr, is_distinct, arg1);
}

hdk::ir::ExprPtr RelAlgTranslator::normalize(const hdk::ir::Expr* expr) const {
  NormalizerVisitor visitor(*this, input_to_nest_level_, join_types_, executor_);
  return visitor.visit(expr);
}

hdk::ir::ExprPtr RelAlgTranslator::translateLiteral(const RexLiteral* rex_literal) {
  auto lit_ti = build_type_info(
      rex_literal->getType(), rex_literal->getScale(), rex_literal->getPrecision());
  auto target_ti = build_type_info(rex_literal->getTargetType(),
                                   rex_literal->getTargetScale(),
                                   rex_literal->getTargetPrecision());
  switch (rex_literal->getType()) {
    case kINT:
    case kBIGINT: {
      Datum d;
      d.bigintval = rex_literal->getVal<int64_t>();
      return hdk::ir::makeExpr<hdk::ir::Constant>(rex_literal->getType(), false, d);
    }
    case kDECIMAL: {
      const auto val = rex_literal->getVal<int64_t>();
      const int precision = rex_literal->getPrecision();
      const int scale = rex_literal->getScale();
      if (target_ti.is_fp() && !scale) {
        return make_fp_constant(val, target_ti);
      }
      auto lit_expr = scale ? Analyzer::analyzeFixedPtValue(val, scale, precision)
                            : Analyzer::analyzeIntValue(val);
      return lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kTEXT: {
      return Analyzer::analyzeStringValue(rex_literal->getVal<std::string>());
    }
    case kBOOLEAN: {
      Datum d;
      d.boolval = rex_literal->getVal<bool>();
      return hdk::ir::makeExpr<hdk::ir::Constant>(kBOOLEAN, false, d);
    }
    case kDOUBLE: {
      Datum d;
      d.doubleval = rex_literal->getVal<double>();
      auto lit_expr = hdk::ir::makeExpr<hdk::ir::Constant>(kDOUBLE, false, d);
      return lit_ti != target_ti ? lit_expr->add_cast(target_ti) : lit_expr;
    }
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH: {
      Datum d;
      d.bigintval = rex_literal->getVal<int64_t>();
      return hdk::ir::makeExpr<hdk::ir::Constant>(rex_literal->getType(), false, d);
    }
    case kTIME:
    case kTIMESTAMP: {
      Datum d;
      d.bigintval =
          rex_literal->getType() == kTIMESTAMP && rex_literal->getPrecision() > 0
              ? rex_literal->getVal<int64_t>()
              : rex_literal->getVal<int64_t>() / 1000;
      return hdk::ir::makeExpr<hdk::ir::Constant>(
          SQLTypeInfo(rex_literal->getType(), rex_literal->getPrecision(), 0, false),
          false,
          d);
    }
    case kDATE: {
      Datum d;
      d.bigintval = rex_literal->getVal<int64_t>() * 24 * 3600;
      return hdk::ir::makeExpr<hdk::ir::Constant>(rex_literal->getType(), false, d);
    }
    case kNULLT: {
      if (target_ti.is_array()) {
        hdk::ir::ExprPtrVector args;
        // defaulting to valid sub-type for convenience
        target_ti.set_subtype(kBOOLEAN);
        return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(target_ti, args, true);
      }
      return hdk::ir::makeExpr<hdk::ir::Constant>(
          rex_literal->getTargetType(), true, Datum{0});
    }
    default: {
      LOG(FATAL) << "Unexpected literal type " << lit_ti.get_type_name();
    }
  }
  return nullptr;
}

hdk::ir::ExprPtr RelAlgTranslator::translateScalarSubquery(
    const RexSubQuery* rex_subquery) const {
  if (just_explain_) {
    throw std::runtime_error("EXPLAIN is not supported with sub-queries");
  }
  if (for_dag_builder_) {
    auto node = rex_subquery->getRelAlgShared();
    auto ti = getColumnType(node.get(), 0);
    return hdk::ir::makeExpr<hdk::ir::ScalarSubquery>(ti, node);
  }
  CHECK(rex_subquery);
  auto result = rex_subquery->getExecutionResult();
  CHECK(rex_subquery->getRelAlg()->getResult());
  return translateScalarSubqueryResult(result);
}

hdk::ir::ExprPtr RelAlgTranslator::translateInput(const RexInput* rex_input) const {
  const auto source = rex_input->getSourceNode();
  const auto it_rte_idx = input_to_nest_level_.find(source);
  CHECK(for_dag_builder_ || it_rte_idx != input_to_nest_level_.end())
      << "Not found in input_to_nest_level_, source=" << source->toString();
  const int rte_idx = it_rte_idx == input_to_nest_level_.end() ? 0 : it_rte_idx->second;
  const auto scan_source = dynamic_cast<const RelScan*>(source);
  const auto& in_metainfo = source->getOutputMetainfo();
  if (scan_source) {
    // We're at leaf (scan) level and not supposed to have input metadata,
    // the name and type information come directly from the catalog.
    CHECK(in_metainfo.empty());
    auto col_info = scan_source->getColumnInfoBySpi(rex_input->getIndex() + 1);
    auto& col_ti = col_info->type;
    if (col_ti.is_string()) {
      col_ti.set_type(kTEXT);
    }
    if (col_info->is_rowid) {
      // TODO(alex): remove at some point, we only need this fixup for backwards
      // compatibility with old imported data
      col_ti.set_size(8);
    }
    CHECK_LE(static_cast<size_t>(rte_idx), join_types_.size());
    if (rte_idx > 0 && join_types_[rte_idx - 1] == JoinType::LEFT) {
      col_ti.set_notnull(false);
    }
    if (for_dag_builder_) {
      return hdk::ir::makeExpr<hdk::ir::ColumnRef>(col_ti, source, rex_input->getIndex());
    }
    return std::make_shared<hdk::ir::ColumnVar>(col_info, rte_idx);
  }
  CHECK_GE(rte_idx, 0);
  const size_t col_id = rex_input->getIndex();
  SQLTypeInfo col_ti;
  if (for_dag_builder_) {
    col_ti = getColumnType(source, col_id);
  } else {
    CHECK(!in_metainfo.empty()) << "for " << source->toString();
    CHECK_LT(col_id, in_metainfo.size());
    col_ti = in_metainfo[col_id].get_type_info();
  }

  if (join_types_.size() > 0) {
    CHECK_LE(static_cast<size_t>(rte_idx), join_types_.size());
    if (rte_idx > 0 && join_types_[rte_idx - 1] == JoinType::LEFT) {
      col_ti.set_notnull(false);
    }
  }

  if (for_dag_builder_) {
    return hdk::ir::makeExpr<hdk::ir::ColumnRef>(col_ti, source, rex_input->getIndex());
  }
  return std::make_shared<hdk::ir::ColumnVar>(col_ti, -source->getId(), col_id, rte_idx);
}

hdk::ir::ExprPtr RelAlgTranslator::translateUoper(const RexOperator* rex_operator) const {
  CHECK_EQ(size_t(1), rex_operator->size());
  const auto operand_expr = translateScalarRex(rex_operator->getOperand(0));
  const auto sql_op = rex_operator->getOperator();
  switch (sql_op) {
    case kCAST: {
      const auto& target_ti = rex_operator->getType();
      CHECK_NE(kNULLT, target_ti.get_type());
      const auto& operand_ti = operand_expr->get_type_info();
      if (operand_ti.is_string() && target_ti.is_string()) {
        return operand_expr;
      }
      if (target_ti.is_time() ||
          operand_ti
              .is_string()) {  // TODO(alex): check and unify with the rest of the cases
        // Do not propogate encoding on small dates
        return target_ti.is_date_in_days()
                   ? operand_expr->add_cast(SQLTypeInfo(kDATE, false))
                   : operand_expr->add_cast(target_ti);
      }
      if (!operand_ti.is_string() && target_ti.is_string()) {
        return operand_expr->add_cast(target_ti);
      }

      return std::make_shared<hdk::ir::UOper>(target_ti, false, sql_op, operand_expr);
    }
    case kNOT:
    case kISNULL: {
      return std::make_shared<hdk::ir::UOper>(kBOOLEAN, sql_op, operand_expr);
    }
    case kISNOTNULL: {
      auto is_null = std::make_shared<hdk::ir::UOper>(kBOOLEAN, kISNULL, operand_expr);
      return std::make_shared<hdk::ir::UOper>(kBOOLEAN, kNOT, is_null);
    }
    case kMINUS: {
      const auto& ti = operand_expr->get_type_info();
      return std::make_shared<hdk::ir::UOper>(ti, false, kUMINUS, operand_expr);
    }
    case kUNNEST: {
      const auto& ti = operand_expr->get_type_info();
      CHECK(ti.is_array());
      return hdk::ir::makeExpr<hdk::ir::UOper>(
          ti.get_elem_type(), false, kUNNEST, operand_expr);
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

namespace {

hdk::ir::ExprPtr get_in_values_expr(hdk::ir::ExprPtr arg,
                                    const ResultSet& val_set,
                                    bool enable_watchdog) {
  if (!result_set::can_use_parallel_algorithms(val_set)) {
    return nullptr;
  }
  if (val_set.rowCount() > 5000000 && enable_watchdog) {
    throw std::runtime_error(
        "Unable to handle 'expr IN (subquery)', subquery returned 5M+ rows.");
  }
  std::list<hdk::ir::ExprPtr> value_exprs;
  const size_t fetcher_count = cpu_threads();
  std::vector<std::list<hdk::ir::ExprPtr>> expr_set(fetcher_count,
                                                    std::list<hdk::ir::ExprPtr>());
  std::vector<std::future<void>> fetcher_threads;
  const auto& ti = arg->get_type_info();
  const auto entry_count = val_set.entryCount();
  for (size_t i = 0,
              start_entry = 0,
              stride = (entry_count + fetcher_count - 1) / fetcher_count;
       i < fetcher_count && start_entry < entry_count;
       ++i, start_entry += stride) {
    const auto end_entry = std::min(start_entry + stride, entry_count);
    fetcher_threads.push_back(std::async(
        std::launch::async,
        [&](std::list<hdk::ir::ExprPtr>& in_vals, const size_t start, const size_t end) {
          for (auto index = start; index < end; ++index) {
            auto row = val_set.getRowAt(index);
            if (row.empty()) {
              continue;
            }
            auto scalar_tv = boost::get<ScalarTargetValue>(&row[0]);
            Datum d{0};
            bool is_null_const{false};
            std::tie(d, is_null_const) = datum_from_scalar_tv(scalar_tv, ti);
            if (ti.is_string() && ti.get_compression() != kENCODING_NONE) {
              auto ti_none_encoded = ti;
              ti_none_encoded.set_compression(kENCODING_NONE);
              auto none_encoded_string =
                  hdk::ir::makeExpr<hdk::ir::Constant>(ti, is_null_const, d);
              auto dict_encoded_string =
                  std::make_shared<hdk::ir::UOper>(ti, false, kCAST, none_encoded_string);
              in_vals.push_back(dict_encoded_string);
            } else {
              in_vals.push_back(
                  hdk::ir::makeExpr<hdk::ir::Constant>(ti, is_null_const, d));
            }
          }
        },
        std::ref(expr_set[i]),
        start_entry,
        end_entry));
  }
  for (auto& child : fetcher_threads) {
    child.get();
  }

  val_set.moveToBegin();
  for (auto& exprs : expr_set) {
    value_exprs.splice(value_exprs.end(), exprs);
  }
  return hdk::ir::makeExpr<hdk::ir::InValues>(arg, value_exprs);
}

}  // namespace

// Creates an Analyzer expression for an IN subquery which subsequently goes through the
// regular Executor::codegen() mechanism. The creation of the expression out of
// subquery's result set is parallelized whenever possible. In addition, take advantage
// of additional information that elements in the right hand side are constants; see
// getInIntegerSetExpr().
hdk::ir::ExprPtr RelAlgTranslator::translateInOper(
    const RexOperator* rex_operator) const {
  if (just_explain_) {
    throw std::runtime_error("EXPLAIN is not supported with sub-queries");
  }
  CHECK(rex_operator->size() == 2);
  const auto lhs = translateScalarRex(rex_operator->getOperand(0));
  const auto rhs = rex_operator->getOperand(1);
  const auto rex_subquery = dynamic_cast<const RexSubQuery*>(rhs);
  CHECK(rex_subquery);
  auto ti = lhs->get_type_info();
  if (for_dag_builder_) {
    SQLTypeInfo ti(kBOOLEAN);
    return hdk::ir::makeExpr<hdk::ir::InSubquery>(
        ti, lhs, rex_subquery->getRelAlgShared());
  }
  auto result = rex_subquery->getExecutionResult();
  return translateInSubquery(lhs, result);
}

hdk::ir::ExprPtr RelAlgTranslator::translateInSubquery(
    hdk::ir::ExprPtr lhs,
    std::shared_ptr<const ExecutionResult> result) const {
  CHECK(result);
  auto ti = lhs->get_type_info();
  auto& row_set = result->getRows();
  CHECK_EQ(size_t(1), row_set->colCount());
  const auto& rhs_ti = row_set->getColType(0);
  if (rhs_ti.get_type() != ti.get_type()) {
    throw std::runtime_error(
        "The two sides of the IN operator must have the same type; found " +
        ti.get_type_name() + " and " + rhs_ti.get_type_name());
  }
  row_set->moveToBegin();
  if (row_set->entryCount() > 10000) {
    hdk::ir::ExprPtr expr;
    if ((ti.is_integer() || (ti.is_string() && ti.get_compression() == kENCODING_DICT)) &&
        !row_set->getQueryMemDesc().didOutputColumnar()) {
      expr = getInIntegerSetExpr(lhs, *row_set);
      // Handle the highly unlikely case when the InIntegerSet ended up being tiny.
      // Just let it fall through the usual InValues path at the end of this method,
      // its codegen knows to use inline comparisons for few values.
      if (expr && std::static_pointer_cast<hdk::ir::InIntegerSet>(expr)
                          ->get_value_list()
                          .size() <= 100) {
        expr = nullptr;
      }
    } else {
      expr = get_in_values_expr(lhs, *row_set, config_.exec.watchdog.enable);
    }
    if (expr) {
      return expr;
    }
  }
  std::list<hdk::ir::ExprPtr> value_exprs;
  while (true) {
    auto row = row_set->getNextRow(true, false);
    if (row.empty()) {
      break;
    }
    if (config_.exec.watchdog.enable && value_exprs.size() >= 10000) {
      throw std::runtime_error(
          "Unable to handle 'expr IN (subquery)', subquery returned 10000+ rows.");
    }
    auto scalar_tv = boost::get<ScalarTargetValue>(&row[0]);
    Datum d{0};
    bool is_null_const{false};
    std::tie(d, is_null_const) = datum_from_scalar_tv(scalar_tv, ti);
    if (ti.is_string() && ti.get_compression() != kENCODING_NONE) {
      auto ti_none_encoded = ti;
      ti_none_encoded.set_compression(kENCODING_NONE);
      auto none_encoded_string =
          hdk::ir::makeExpr<hdk::ir::Constant>(ti, is_null_const, d);
      auto dict_encoded_string =
          std::make_shared<hdk::ir::UOper>(ti, false, kCAST, none_encoded_string);
      value_exprs.push_back(dict_encoded_string);
    } else {
      value_exprs.push_back(hdk::ir::makeExpr<hdk::ir::Constant>(ti, is_null_const, d));
    }
  }
  return hdk::ir::makeExpr<hdk::ir::InValues>(lhs, value_exprs);
}

namespace {

const size_t g_max_integer_set_size{1 << 25};

void fill_dictionary_encoded_in_vals(
    std::vector<int64_t>& in_vals,
    std::atomic<size_t>& total_in_vals_count,
    const ResultSet* values_rowset,
    const std::pair<int64_t, int64_t> values_rowset_slice,
    const StringDictionaryProxy* source_dict,
    const StringDictionaryProxy* dest_dict,
    const int64_t needle_null_val,
    bool enable_watchdog) {
  CHECK(in_vals.empty());
  bool dicts_are_equal = source_dict == dest_dict;
  for (auto index = values_rowset_slice.first; index < values_rowset_slice.second;
       ++index) {
    const auto row = values_rowset->getOneColRow(index);
    if (UNLIKELY(!row.valid)) {
      continue;
    }
    if (dicts_are_equal) {
      in_vals.push_back(row.value);
    } else {
      const int string_id =
          row.value == needle_null_val
              ? needle_null_val
              : dest_dict->getIdOfString(source_dict->getString(row.value));
      if (string_id != StringDictionary::INVALID_STR_ID) {
        in_vals.push_back(string_id);
      }
    }
    if (UNLIKELY(enable_watchdog && (in_vals.size() & 1023) == 0 &&
                 total_in_vals_count.fetch_add(1024) >= g_max_integer_set_size)) {
      throw std::runtime_error(
          "Unable to handle 'expr IN (subquery)', subquery returned 30M+ rows.");
    }
  }
}

void fill_integer_in_vals(std::vector<int64_t>& in_vals,
                          std::atomic<size_t>& total_in_vals_count,
                          const ResultSet* values_rowset,
                          const std::pair<int64_t, int64_t> values_rowset_slice,
                          bool enable_watchdog) {
  CHECK(in_vals.empty());
  for (auto index = values_rowset_slice.first; index < values_rowset_slice.second;
       ++index) {
    const auto row = values_rowset->getOneColRow(index);
    if (row.valid) {
      in_vals.push_back(row.value);
      if (UNLIKELY(enable_watchdog && (in_vals.size() & 1023) == 0 &&
                   total_in_vals_count.fetch_add(1024) >= g_max_integer_set_size)) {
        throw std::runtime_error(
            "Unable to handle 'expr IN (subquery)', subquery returned 30M+ rows.");
      }
    }
  }
}

}  // namespace

// The typical IN subquery involves either dictionary-encoded strings or integers.
// hdk::ir::InValues is a very heavy representation of the right hand side of such
// a query since we already know the right hand would be a list of hdk::ir::Constant
// shared pointers. We can avoid the big overhead of each hdk::ir::Constant and the
// refcounting associated with shared pointers by creating an abbreviated InIntegerSet
// representation of the IN expression which takes advantage of the this information.
hdk::ir::ExprPtr RelAlgTranslator::getInIntegerSetExpr(hdk::ir::ExprPtr arg,
                                                       const ResultSet& val_set) const {
  if (!result_set::can_use_parallel_algorithms(val_set)) {
    return nullptr;
  }
  std::vector<int64_t> value_exprs;
  const size_t fetcher_count = cpu_threads();
  std::vector<std::vector<int64_t>> expr_set(fetcher_count);
  std::vector<std::future<void>> fetcher_threads;
  const auto& arg_type = arg->get_type_info();
  const auto entry_count = val_set.entryCount();
  CHECK_EQ(size_t(1), val_set.colCount());
  const auto& col_type = val_set.getColType(0);
  std::atomic<size_t> total_in_vals_count{0};
  for (size_t i = 0,
              start_entry = 0,
              stride = (entry_count + fetcher_count - 1) / fetcher_count;
       i < fetcher_count && start_entry < entry_count;
       ++i, start_entry += stride) {
    expr_set[i].reserve(entry_count / fetcher_count);
    const auto end_entry = std::min(start_entry + stride, entry_count);
    if (arg_type.is_string()) {
      CHECK_EQ(kENCODING_DICT, arg_type.get_compression());
      // const int32_t dest_dict_id = arg_type.get_comp_param();
      // const int32_t source_dict_id = col_type.get_comp_param();
      const auto dd = executor_->getStringDictionaryProxy(
          arg_type.get_comp_param(), val_set.getRowSetMemOwner(), true);
      const auto sd = executor_->getStringDictionaryProxy(
          col_type.get_comp_param(), val_set.getRowSetMemOwner(), true);
      CHECK(sd);
      const auto needle_null_val = inline_int_null_val(arg_type);
      fetcher_threads.push_back(std::async(
          std::launch::async,
          [this, &val_set, &total_in_vals_count, sd, dd, needle_null_val](
              std::vector<int64_t>& in_vals, const size_t start, const size_t end) {
            fill_dictionary_encoded_in_vals(in_vals,
                                            total_in_vals_count,
                                            &val_set,
                                            {start, end},
                                            sd,
                                            dd,
                                            needle_null_val,
                                            config_.exec.watchdog.enable);
          },
          std::ref(expr_set[i]),
          start_entry,
          end_entry));
    } else {
      CHECK(arg_type.is_integer());
      fetcher_threads.push_back(std::async(
          std::launch::async,
          [this, &val_set, &total_in_vals_count](
              std::vector<int64_t>& in_vals, const size_t start, const size_t end) {
            fill_integer_in_vals(in_vals,
                                 total_in_vals_count,
                                 &val_set,
                                 {start, end},
                                 config_.exec.watchdog.enable);
          },
          std::ref(expr_set[i]),
          start_entry,
          end_entry));
    }
  }
  for (auto& child : fetcher_threads) {
    child.get();
  }

  val_set.moveToBegin();
  value_exprs.reserve(entry_count);
  for (auto& exprs : expr_set) {
    value_exprs.insert(value_exprs.end(), exprs.begin(), exprs.end());
  }
  return hdk::ir::makeExpr<hdk::ir::InIntegerSet>(
      arg, value_exprs, arg_type.get_notnull() && col_type.get_notnull());
}

hdk::ir::ExprPtr RelAlgTranslator::translateOper(const RexOperator* rex_operator) const {
  CHECK_GT(rex_operator->size(), size_t(0));
  if (rex_operator->size() == 1) {
    return translateUoper(rex_operator);
  }
  const auto sql_op = rex_operator->getOperator();
  if (sql_op == kIN) {
    return translateInOper(rex_operator);
  }
  if (sql_op == kMINUS || sql_op == kPLUS) {
    auto date_plus_minus = translateDatePlusMinus(rex_operator);
    if (date_plus_minus) {
      return date_plus_minus;
    }
  }
  auto lhs = translateScalarRex(rex_operator->getOperand(0));
  for (size_t i = 1; i < rex_operator->size(); ++i) {
    hdk::ir::ExprPtr rhs;
    SQLQualifier sql_qual{kONE};
    const auto rhs_op = rex_operator->getOperand(i);
    std::tie(rhs, sql_qual) = get_quantified_rhs(rhs_op, *this);
    if (!rhs) {
      rhs = translateScalarRex(rhs_op);
    }
    CHECK(rhs);
    // Pass in executor to get string proxy info if cast needed between
    // string columns
    lhs = Analyzer::normalizeOperExpr(sql_op, sql_qual, lhs, rhs, executor_);
  }
  return lhs;
}

hdk::ir::ExprPtr RelAlgTranslator::translateCase(const RexCase* rex_case) const {
  hdk::ir::ExprPtr else_expr;
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
  for (size_t i = 0; i < rex_case->branchCount(); ++i) {
    const auto when_expr = translateScalarRex(rex_case->getWhen(i));
    const auto then_expr = translateScalarRex(rex_case->getThen(i));
    expr_list.emplace_back(when_expr, then_expr);
  }
  if (rex_case->getElse()) {
    else_expr = translateScalarRex(rex_case->getElse());
  }
  return Analyzer::normalizeCaseExpr(expr_list, else_expr, executor_);
}

hdk::ir::ExprPtr RelAlgTranslator::translateWidthBucket(
    const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 4);
  auto target_value = translateScalarRex(rex_function->getOperand(0));
  auto lower_bound = translateScalarRex(rex_function->getOperand(1));
  auto upper_bound = translateScalarRex(rex_function->getOperand(2));
  auto partition_count = translateScalarRex(rex_function->getOperand(3));
  if (!partition_count->get_type_info().is_integer()) {
    throw std::runtime_error(
        "PARTITION_COUNT expression of width_bucket function expects an integer type.");
  }
  auto check_numeric_type =
      [](const std::string& col_name, const hdk::ir::Expr* expr, bool allow_null_type) {
        if (expr->get_type_info().get_type() == kNULLT) {
          if (!allow_null_type) {
            throw std::runtime_error(
                col_name + " expression of width_bucket function expects non-null type.");
          }
          return;
        }
        if (!expr->get_type_info().is_number()) {
          throw std::runtime_error(
              col_name + " expression of width_bucket function expects a numeric type.");
        }
      };
  // target value may have null value
  check_numeric_type("TARGET_VALUE", target_value.get(), true);
  check_numeric_type("LOWER_BOUND", lower_bound.get(), false);
  check_numeric_type("UPPER_BOUND", upper_bound.get(), false);

  auto cast_to_double_if_necessary = [](hdk::ir::ExprPtr arg) {
    const auto& arg_ti = arg->get_type_info();
    if (arg_ti.get_type() != kDOUBLE) {
      const auto& double_ti = SQLTypeInfo(kDOUBLE, arg_ti.get_notnull());
      return arg->add_cast(double_ti);
    }
    return arg;
  };
  target_value = cast_to_double_if_necessary(target_value);
  lower_bound = cast_to_double_if_necessary(lower_bound);
  upper_bound = cast_to_double_if_necessary(upper_bound);
  return hdk::ir::makeExpr<hdk::ir::WidthBucketExpr>(
      target_value, lower_bound, upper_bound, partition_count);
}

hdk::ir::ExprPtr RelAlgTranslator::translateLike(
    const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 2 || rex_function->size() == 3);
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto like = translateScalarRex(rex_function->getOperand(1));
  if (!std::dynamic_pointer_cast<const hdk::ir::Constant>(like)) {
    throw std::runtime_error("The matching pattern must be a literal.");
  }
  const auto escape = (rex_function->size() == 3)
                          ? translateScalarRex(rex_function->getOperand(2))
                          : nullptr;
  const bool is_ilike = rex_function->getName() == "PG_ILIKE"sv;
  return Analyzer::getLikeExpr(arg, like, escape, is_ilike, false);
}

hdk::ir::ExprPtr RelAlgTranslator::translateRegexp(
    const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 2 || rex_function->size() == 3);
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto pattern = translateScalarRex(rex_function->getOperand(1));
  if (!std::dynamic_pointer_cast<const hdk::ir::Constant>(pattern)) {
    throw std::runtime_error("The matching pattern must be a literal.");
  }
  const auto escape = (rex_function->size() == 3)
                          ? translateScalarRex(rex_function->getOperand(2))
                          : nullptr;
  return Analyzer::getRegexpExpr(arg, pattern, escape, false);
}

hdk::ir::ExprPtr RelAlgTranslator::translateLikely(
    const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 1);
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  return hdk::ir::makeExpr<hdk::ir::LikelihoodExpr>(arg, 0.9375);
}

hdk::ir::ExprPtr RelAlgTranslator::translateUnlikely(
    const RexFunctionOperator* rex_function) const {
  CHECK(rex_function->size() == 1);
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  return hdk::ir::makeExpr<hdk::ir::LikelihoodExpr>(arg, 0.0625);
}

namespace {

inline void validate_datetime_datepart_argument(
    const std::shared_ptr<hdk::ir::Constant> literal_expr) {
  if (!literal_expr || literal_expr->get_is_null()) {
    throw std::runtime_error("The 'DatePart' argument must be a not 'null' literal.");
  }
}

}  // namespace

hdk::ir::ExprPtr RelAlgTranslator::translateExtract(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<hdk::ir::Constant>(timeunit);
  validate_datetime_datepart_argument(timeunit_lit);
  const auto from_expr = translateScalarRex(rex_function->getOperand(1));
  const bool is_date_trunc = rex_function->getName() == "PG_DATE_TRUNC"sv;
  if (is_date_trunc) {
    return DateTruncExpr::generate(from_expr, *timeunit_lit->get_constval().stringval);
  } else {
    return ExtractExpr::generate(from_expr, *timeunit_lit->get_constval().stringval);
  }
}

hdk::ir::ExprPtr RelAlgTranslator::translateDateadd(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(3), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<hdk::ir::Constant>(timeunit);
  validate_datetime_datepart_argument(timeunit_lit);
  const auto number_units = translateScalarRex(rex_function->getOperand(1));
  const auto number_units_const =
      std::dynamic_pointer_cast<hdk::ir::Constant>(number_units);
  if (number_units_const && number_units_const->get_is_null()) {
    throw std::runtime_error("The 'Interval' argument literal must not be 'null'.");
  }
  const auto cast_number_units = number_units->add_cast(SQLTypeInfo(kBIGINT, false));
  const auto datetime = translateScalarRex(rex_function->getOperand(2));
  const auto& datetime_ti = datetime->get_type_info();
  if (datetime_ti.get_type() == kTIME) {
    throw std::runtime_error("DateAdd operation not supported for TIME.");
  }
  const auto& field = to_dateadd_field(*timeunit_lit->get_constval().stringval);
  const int dim = datetime_ti.get_dimension();
  return hdk::ir::makeExpr<hdk::ir::DateaddExpr>(
      SQLTypeInfo(kTIMESTAMP, dim, 0, false), field, cast_number_units, datetime);
}

namespace {

std::string get_datetimeplus_rewrite_funcname(const SQLOps& op) {
  CHECK(op == kPLUS);
  return "DATETIME_PLUS"s;
}

}  // namespace

hdk::ir::ExprPtr RelAlgTranslator::translateDatePlusMinus(
    const RexOperator* rex_operator) const {
  if (rex_operator->size() != 2) {
    return nullptr;
  }
  const auto datetime = translateScalarRex(rex_operator->getOperand(0));
  const auto datetime_ti = datetime->get_type_info();
  if (!datetime_ti.is_timestamp() && !datetime_ti.is_date()) {
    if (datetime_ti.get_type() == kTIME) {
      throw std::runtime_error("DateTime addition/subtraction not supported for TIME.");
    }
    return nullptr;
  }
  const auto rhs = translateScalarRex(rex_operator->getOperand(1));
  const auto rhs_ti = rhs->get_type_info();
  if (rhs_ti.get_type() == kTIMESTAMP || rhs_ti.get_type() == kDATE) {
    if (datetime_ti.is_high_precision_timestamp() ||
        rhs_ti.is_high_precision_timestamp()) {
      throw std::runtime_error(
          "High Precision timestamps are not supported for TIMESTAMPDIFF operation. "
          "Use "
          "DATEDIFF.");
    }
    auto bigint_ti = SQLTypeInfo(kBIGINT, false);
    const auto& rex_operator_ti = rex_operator->getType();
    const auto datediff_field =
        (rex_operator_ti.get_type() == kINTERVAL_DAY_TIME) ? dtSECOND : dtMONTH;
    auto result = hdk::ir::makeExpr<hdk::ir::DatediffExpr>(
        bigint_ti, datediff_field, rhs, datetime);
    // multiply 1000 to result since expected result should be in millisecond precision.
    if (rex_operator_ti.get_type() == kINTERVAL_DAY_TIME) {
      return hdk::ir::makeExpr<hdk::ir::BinOper>(
          bigint_ti.get_type(),
          kMULTIPLY,
          kONE,
          result,
          hdk::ir::Constant::make(bigint_ti, 1000));
    } else {
      return result;
    }
  }
  const auto op = rex_operator->getOperator();
  if (op == kPLUS) {
    std::vector<hdk::ir::ExprPtr> args = {datetime, rhs};
    auto dt_plus = hdk::ir::makeExpr<hdk::ir::FunctionOper>(
        datetime_ti, get_datetimeplus_rewrite_funcname(op), args);
    const auto date_trunc = rewrite_to_date_trunc(dt_plus.get());
    if (date_trunc) {
      return date_trunc;
    }
  }
  const auto interval = fold_expr(rhs.get());
  auto interval_ti = interval->get_type_info();
  auto bigint_ti = SQLTypeInfo(kBIGINT, false);
  const auto interval_lit = std::dynamic_pointer_cast<hdk::ir::Constant>(interval);
  if (interval_ti.get_type() == kINTERVAL_DAY_TIME) {
    hdk::ir::ExprPtr interval_sec;
    if (interval_lit) {
      interval_sec = hdk::ir::Constant::make(
          bigint_ti,
          (op == kMINUS ? -interval_lit->get_constval().bigintval
                        : interval_lit->get_constval().bigintval) /
              1000);
    } else {
      interval_sec =
          hdk::ir::makeExpr<hdk::ir::BinOper>(bigint_ti.get_type(),
                                              kDIVIDE,
                                              kONE,
                                              interval,
                                              hdk::ir::Constant::make(bigint_ti, 1000));
      if (op == kMINUS) {
        interval_sec =
            std::make_shared<hdk::ir::UOper>(bigint_ti, false, kUMINUS, interval_sec);
      }
    }
    return hdk::ir::makeExpr<hdk::ir::DateaddExpr>(
        datetime_ti, daSECOND, interval_sec, datetime);
  }
  CHECK(interval_ti.get_type() == kINTERVAL_YEAR_MONTH);
  const auto interval_months =
      op == kMINUS ? std::make_shared<hdk::ir::UOper>(bigint_ti, false, kUMINUS, interval)
                   : interval;
  return hdk::ir::makeExpr<hdk::ir::DateaddExpr>(
      datetime_ti, daMONTH, interval_months, datetime);
}

hdk::ir::ExprPtr RelAlgTranslator::translateDatediff(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(3), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<hdk::ir::Constant>(timeunit);
  validate_datetime_datepart_argument(timeunit_lit);
  const auto start = translateScalarRex(rex_function->getOperand(1));
  const auto end = translateScalarRex(rex_function->getOperand(2));
  const auto field = to_datediff_field(*timeunit_lit->get_constval().stringval);
  return hdk::ir::makeExpr<hdk::ir::DatediffExpr>(
      SQLTypeInfo(kBIGINT, false), field, start, end);
}

hdk::ir::ExprPtr RelAlgTranslator::translateDatepart(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto timeunit = translateScalarRex(rex_function->getOperand(0));
  const auto timeunit_lit = std::dynamic_pointer_cast<hdk::ir::Constant>(timeunit);
  validate_datetime_datepart_argument(timeunit_lit);
  const auto from_expr = translateScalarRex(rex_function->getOperand(1));
  return ExtractExpr::generate(
      from_expr, to_datepart_field(*timeunit_lit->get_constval().stringval));
}

hdk::ir::ExprPtr RelAlgTranslator::translateLength(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());
  const auto str_arg = translateScalarRex(rex_function->getOperand(0));
  return hdk::ir::makeExpr<hdk::ir::CharLengthExpr>(
      str_arg->decompress(), rex_function->getName() == "CHAR_LENGTH"sv);
}

hdk::ir::ExprPtr RelAlgTranslator::translateKeyForString(
    const RexFunctionOperator* rex_function) const {
  const auto& args = translateFunctionArgs(rex_function);
  CHECK_EQ(size_t(1), args.size());
  const auto expr = dynamic_cast<hdk::ir::Expr*>(args[0].get());
  if (nullptr == expr || !expr->get_type_info().is_string() ||
      expr->get_type_info().is_varlen()) {
    throw std::runtime_error(rex_function->getName() +
                             " expects a dictionary encoded text column.");
  }
  return hdk::ir::makeExpr<hdk::ir::KeyForStringExpr>(args[0]);
}

hdk::ir::ExprPtr RelAlgTranslator::translateSampleRatio(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());
  auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto& arg_ti = arg->get_type_info();
  if (arg_ti.get_type() != kDOUBLE) {
    const auto& double_ti = SQLTypeInfo(kDOUBLE, arg_ti.get_notnull());
    arg = arg->add_cast(double_ti);
  }
  return hdk::ir::makeExpr<hdk::ir::SampleRatioExpr>(arg);
}

hdk::ir::ExprPtr RelAlgTranslator::translateCurrentUser(
    const RexFunctionOperator* rex_function) const {
  std::string user{"SESSIONLESS_USER"};
  return Analyzer::getUserLiteral(user);
}

hdk::ir::ExprPtr RelAlgTranslator::translateLower(
    const RexFunctionOperator* rex_function) const {
  const auto& args = translateFunctionArgs(rex_function);
  CHECK_EQ(size_t(1), args.size());
  CHECK(args[0]);

  if (args[0]->get_type_info().is_dict_encoded_string() ||
      dynamic_cast<hdk::ir::Constant*>(args[0].get())) {
    return hdk::ir::makeExpr<hdk::ir::LowerExpr>(args[0]);
  }

  throw std::runtime_error(rex_function->getName() +
                           " expects a dictionary encoded text column or a literal.");
}

hdk::ir::ExprPtr RelAlgTranslator::translateCardinality(
    const RexFunctionOperator* rex_function) const {
  const auto ret_ti = rex_function->getType();
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto arg_ti = arg->get_type_info();
  if (!arg_ti.is_array()) {
    throw std::runtime_error(rex_function->getName() + " expects an array expression.");
  }
  if (arg_ti.get_subtype() == kARRAY) {
    throw std::runtime_error(rex_function->getName() +
                             " expects one-dimension array expression.");
  }
  const auto array_size = arg_ti.get_size();
  const auto array_elem_size = arg_ti.get_elem_type().get_array_context_logical_size();

  if (array_size > 0) {
    if (array_elem_size <= 0) {
      throw std::runtime_error(rex_function->getName() +
                               ": unexpected array element type.");
    }
    // Return cardinality of a fixed length array
    return hdk::ir::Constant::make(ret_ti, array_size / array_elem_size);
  }
  // Variable length array cardinality will be calculated at runtime
  return hdk::ir::makeExpr<hdk::ir::CardinalityExpr>(arg);
}

hdk::ir::ExprPtr RelAlgTranslator::translateItem(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(2), rex_function->size());
  const auto base = translateScalarRex(rex_function->getOperand(0));
  const auto index = translateScalarRex(rex_function->getOperand(1));
  return hdk::ir::makeExpr<hdk::ir::BinOper>(
      base->get_type_info().get_elem_type(), false, kARRAY_AT, kONE, base, index);
}

hdk::ir::ExprPtr RelAlgTranslator::translateCurrentDate() const {
  constexpr bool is_null = false;
  Datum datum;
  datum.bigintval = now_ - now_ % (24 * 60 * 60);  // Assumes 0 < now_.
  return hdk::ir::makeExpr<hdk::ir::Constant>(kDATE, is_null, datum);
}

hdk::ir::ExprPtr RelAlgTranslator::translateCurrentTime() const {
  constexpr bool is_null = false;
  Datum datum;
  datum.bigintval = now_ % (24 * 60 * 60);  // Assumes 0 < now_.
  return hdk::ir::makeExpr<hdk::ir::Constant>(kTIME, is_null, datum);
}

hdk::ir::ExprPtr RelAlgTranslator::translateCurrentTimestamp() const {
  return Analyzer::getTimestampLiteral(now_);
}

hdk::ir::ExprPtr RelAlgTranslator::translateDatetime(
    const RexFunctionOperator* rex_function) const {
  CHECK_EQ(size_t(1), rex_function->size());
  const auto arg = translateScalarRex(rex_function->getOperand(0));
  const auto arg_lit = std::dynamic_pointer_cast<hdk::ir::Constant>(arg);
  const std::string datetime_err{R"(Only DATETIME('NOW') supported for now.)"};
  if (!arg_lit || arg_lit->get_is_null()) {
    throw std::runtime_error(datetime_err);
  }
  CHECK(arg_lit->get_type_info().is_string());
  if (*arg_lit->get_constval().stringval != "NOW"sv) {
    throw std::runtime_error(datetime_err);
  }
  return translateCurrentTimestamp();
}

hdk::ir::ExprPtr RelAlgTranslator::translateAbs(
    const RexFunctionOperator* rex_function) const {
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
  CHECK_EQ(size_t(1), rex_function->size());
  const auto operand = translateScalarRex(rex_function->getOperand(0));
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_number());
  const auto zero = hdk::ir::Constant::make(operand_ti, 0);
  const auto lt_zero = Analyzer::normalizeOperExpr(kLT, kONE, operand, zero);
  const auto uminus_operand =
      hdk::ir::makeExpr<hdk::ir::UOper>(operand_ti, false, kUMINUS, operand);
  expr_list.emplace_back(lt_zero, uminus_operand);
  return Analyzer::normalizeCaseExpr(expr_list, operand, executor_);
}

hdk::ir::ExprPtr RelAlgTranslator::translateSign(
    const RexFunctionOperator* rex_function) const {
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
  CHECK_EQ(size_t(1), rex_function->size());
  const auto operand = translateScalarRex(rex_function->getOperand(0));
  const auto& operand_ti = operand->get_type_info();
  CHECK(operand_ti.is_number());
  const auto zero = hdk::ir::Constant::make(operand_ti, 0);
  const auto lt_zero =
      hdk::ir::makeExpr<hdk::ir::BinOper>(kBOOLEAN, kLT, kONE, operand, zero);
  expr_list.emplace_back(lt_zero, hdk::ir::Constant::make(operand_ti, -1));
  const auto eq_zero =
      hdk::ir::makeExpr<hdk::ir::BinOper>(kBOOLEAN, kEQ, kONE, operand, zero);
  expr_list.emplace_back(eq_zero, hdk::ir::Constant::make(operand_ti, 0));
  const auto gt_zero =
      hdk::ir::makeExpr<hdk::ir::BinOper>(kBOOLEAN, kGT, kONE, operand, zero);
  expr_list.emplace_back(gt_zero, hdk::ir::Constant::make(operand_ti, 1));
  return Analyzer::normalizeCaseExpr(expr_list, nullptr, executor_);
}

hdk::ir::ExprPtr RelAlgTranslator::translateOffsetInFragment() const {
  return hdk::ir::makeExpr<hdk::ir::OffsetInFragment>();
}

hdk::ir::ExprPtr RelAlgTranslator::translateArrayFunction(
    const RexFunctionOperator* rex_function) const {
  if (rex_function->getType().get_subtype() == kNULLT) {
    auto sql_type = rex_function->getType();
    CHECK(sql_type.get_type() == kARRAY);

    // FIX-ME:  Deal with NULL arrays
    auto translated_function_args(translateFunctionArgs(rex_function));
    if (translated_function_args.size() > 0) {
      const auto first_element_logical_type =
          get_nullable_logical_type_info(translated_function_args[0]->get_type_info());

      auto diff_elem_itr =
          std::find_if(translated_function_args.begin(),
                       translated_function_args.end(),
                       [first_element_logical_type](const auto expr) {
                         return first_element_logical_type !=
                                get_nullable_logical_type_info(expr->get_type_info());
                       });
      if (diff_elem_itr != translated_function_args.end()) {
        throw std::runtime_error(
            "Element " +
            std::to_string(diff_elem_itr - translated_function_args.begin()) +
            " is not of the same type as other elements of the array. Consider casting "
            "to force this condition.\nElement Type: " +
            get_nullable_logical_type_info((*diff_elem_itr)->get_type_info())
                .to_string() +
            "\nArray type: " + first_element_logical_type.to_string());
      }

      if (first_element_logical_type.is_string() &&
          !first_element_logical_type.is_dict_encoded_string()) {
        sql_type.set_subtype(first_element_logical_type.get_type());
        sql_type.set_compression(kENCODING_FIXED);
      } else if (first_element_logical_type.is_dict_encoded_string()) {
        sql_type.set_subtype(first_element_logical_type.get_type());
        sql_type.set_comp_param(TRANSIENT_DICT_ID);
      } else {
        sql_type.set_subtype(first_element_logical_type.get_type());
        sql_type.set_scale(first_element_logical_type.get_scale());
        sql_type.set_precision(first_element_logical_type.get_precision());
      }

      return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(sql_type, translated_function_args);
    } else {
      // defaulting to valid sub-type for convenience
      sql_type.set_subtype(kBOOLEAN);
      return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(sql_type, translated_function_args);
    }
  } else {
    return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(rex_function->getType(),
                                                 translateFunctionArgs(rex_function));
  }
}

hdk::ir::ExprPtr RelAlgTranslator::translateFunction(
    const RexFunctionOperator* rex_function) const {
  if (func_resolve(rex_function->getName(), "LIKE"sv, "PG_ILIKE"sv)) {
    return translateLike(rex_function);
  }
  if (rex_function->getName() == "REGEXP_LIKE"sv) {
    return translateRegexp(rex_function);
  }
  if (rex_function->getName() == "LIKELY"sv) {
    return translateLikely(rex_function);
  }
  if (rex_function->getName() == "UNLIKELY"sv) {
    return translateUnlikely(rex_function);
  }
  if (func_resolve(rex_function->getName(), "PG_EXTRACT"sv, "PG_DATE_TRUNC"sv)) {
    return translateExtract(rex_function);
  }
  if (rex_function->getName() == "DATEADD"sv) {
    return translateDateadd(rex_function);
  }
  if (rex_function->getName() == "DATEDIFF"sv) {
    return translateDatediff(rex_function);
  }
  if (rex_function->getName() == "DATEPART"sv) {
    return translateDatepart(rex_function);
  }
  if (func_resolve(rex_function->getName(), "LENGTH"sv, "CHAR_LENGTH"sv)) {
    return translateLength(rex_function);
  }
  if (rex_function->getName() == "KEY_FOR_STRING"sv) {
    return translateKeyForString(rex_function);
  }
  if (rex_function->getName() == "WIDTH_BUCKET"sv) {
    return translateWidthBucket(rex_function);
  }
  if (rex_function->getName() == "SAMPLE_RATIO"sv) {
    return translateSampleRatio(rex_function);
  }
  if (rex_function->getName() == "CURRENT_USER"sv) {
    return translateCurrentUser(rex_function);
  }
  if (config_.exec.enable_experimental_string_functions &&
      rex_function->getName() == "LOWER"sv) {
    return translateLower(rex_function);
  }
  if (func_resolve(rex_function->getName(), "CARDINALITY"sv, "ARRAY_LENGTH"sv)) {
    return translateCardinality(rex_function);
  }
  if (rex_function->getName() == "ITEM"sv) {
    return translateItem(rex_function);
  }
  if (rex_function->getName() == "CURRENT_DATE"sv) {
    return translateCurrentDate();
  }
  if (rex_function->getName() == "CURRENT_TIME"sv) {
    return translateCurrentTime();
  }
  if (rex_function->getName() == "CURRENT_TIMESTAMP"sv) {
    return translateCurrentTimestamp();
  }
  if (rex_function->getName() == "NOW"sv) {
    return translateCurrentTimestamp();
  }
  if (rex_function->getName() == "DATETIME"sv) {
    return translateDatetime(rex_function);
  }
  if (func_resolve(rex_function->getName(), "usTIMESTAMP"sv, "nsTIMESTAMP"sv)) {
    return translateHPTLiteral(rex_function);
  }
  if (rex_function->getName() == "ABS"sv) {
    return translateAbs(rex_function);
  }
  if (rex_function->getName() == "SIGN"sv) {
    return translateSign(rex_function);
  }
  if (func_resolve(rex_function->getName(), "CEIL"sv, "FLOOR"sv)) {
    return hdk::ir::makeExpr<hdk::ir::FunctionOperWithCustomTypeHandling>(
        rex_function->getType(),
        rex_function->getName(),
        translateFunctionArgs(rex_function));
  } else if (rex_function->getName() == "ROUND"sv) {
    std::vector<hdk::ir::ExprPtr> args = translateFunctionArgs(rex_function);

    if (rex_function->size() == 1) {
      // push a 0 constant if 2nd operand is missing.
      // this needs to be done as calcite returns
      // only the 1st operand without defaulting the 2nd one
      // when the user did not specify the 2nd operand.
      SQLTypes t = kSMALLINT;
      Datum d;
      d.smallintval = 0;
      args.push_back(hdk::ir::makeExpr<hdk::ir::Constant>(t, false, d));
    }

    // make sure we have only 2 operands
    CHECK(args.size() == 2);

    if (!args[0]->get_type_info().is_number()) {
      throw std::runtime_error("Only numeric 1st operands are supported");
    }

    // the 2nd operand does not need to be a constant
    // it can happily reference another integer column
    if (!args[1]->get_type_info().is_integer()) {
      throw std::runtime_error("Only integer 2nd operands are supported");
    }

    // Calcite may upcast decimals in a way that is
    // incompatible with the extension function input. Play it safe and stick with the
    // argument type instead.
    const SQLTypeInfo ret_ti = args[0]->get_type_info().is_decimal()
                                   ? args[0]->get_type_info()
                                   : rex_function->getType();

    return hdk::ir::makeExpr<hdk::ir::FunctionOperWithCustomTypeHandling>(
        ret_ti, rex_function->getName(), args);
  }
  if (rex_function->getName() == "DATETIME_PLUS"sv) {
    auto dt_plus =
        hdk::ir::makeExpr<hdk::ir::FunctionOper>(rex_function->getType(),
                                                 rex_function->getName(),
                                                 translateFunctionArgs(rex_function));
    const auto date_trunc = rewrite_to_date_trunc(dt_plus.get());
    if (date_trunc) {
      return date_trunc;
    }
    return translateDateadd(rex_function);
  }
  if (rex_function->getName() == "/INT"sv) {
    CHECK_EQ(size_t(2), rex_function->size());
    hdk::ir::ExprPtr lhs = translateScalarRex(rex_function->getOperand(0));
    hdk::ir::ExprPtr rhs = translateScalarRex(rex_function->getOperand(1));
    const auto rhs_lit = std::dynamic_pointer_cast<hdk::ir::Constant>(rhs);
    return Analyzer::normalizeOperExpr(kDIVIDE, kONE, lhs, rhs);
  }
  if (rex_function->getName() == "Reinterpret"sv) {
    CHECK_EQ(size_t(1), rex_function->size());
    return translateScalarRex(rex_function->getOperand(0));
  }
  if (rex_function->getName() == "OFFSET_IN_FRAGMENT"sv) {
    CHECK_EQ(size_t(0), rex_function->size());
    return translateOffsetInFragment();
  }
  if (rex_function->getName() == "ARRAY"sv) {
    // Var args; currently no check.  Possible fix-me -- can array have 0 elements?
    return translateArrayFunction(rex_function);
  }

  auto arg_expr_list = translateFunctionArgs(rex_function);
  if (rex_function->getName() == std::string("||") ||
      rex_function->getName() == std::string("SUBSTRING")) {
    SQLTypeInfo ret_ti(kTEXT, false);
    return hdk::ir::makeExpr<hdk::ir::FunctionOper>(
        ret_ti, rex_function->getName(), arg_expr_list);
  }
  // Reset possibly wrong return type of rex_function to the return
  // type of the optimal valid implementation. The return type can be
  // wrong in the case of multiple implementations of UDF functions
  // that have different return types but Calcite specifies the return
  // type according to the first implementation.
  SQLTypeInfo ret_ti;
  try {
    auto ext_func_sig = bind_function(rex_function->getName(), arg_expr_list);

    auto ext_func_args = ext_func_sig.getArgs();
    CHECK_EQ(arg_expr_list.size(), ext_func_args.size());
    for (size_t i = 0; i < arg_expr_list.size(); i++) {
      // fold casts on constants
      if (auto constant =
              std::dynamic_pointer_cast<hdk::ir::Constant>(arg_expr_list[i])) {
        auto ext_func_arg_ti = ext_arg_type_to_type_info(ext_func_args[i]);
        if (ext_func_arg_ti != arg_expr_list[i]->get_type_info()) {
          arg_expr_list[i] = constant->add_cast(ext_func_arg_ti);
        }
      }
    }

    ret_ti = ext_arg_type_to_type_info(ext_func_sig.getRet());
  } catch (ExtensionFunctionBindingError& e) {
    LOG(WARNING) << "RelAlgTranslator::translateFunction: " << e.what();
    throw;
  }

  // By default, the extension function type will not allow nulls. If one of the arguments
  // is nullable, the extension function must also explicitly allow nulls.
  bool arguments_not_null = true;
  for (const auto& arg_expr : arg_expr_list) {
    if (!arg_expr->get_type_info().get_notnull()) {
      arguments_not_null = false;
      break;
    }
  }
  ret_ti.set_notnull(arguments_not_null);

  return hdk::ir::makeExpr<hdk::ir::FunctionOper>(
      ret_ti, rex_function->getName(), arg_expr_list);
}

namespace {

std::vector<hdk::ir::OrderEntry> translate_collation(
    const std::vector<SortField>& sort_fields) {
  std::vector<hdk::ir::OrderEntry> collation;
  for (size_t i = 0; i < sort_fields.size(); ++i) {
    const auto& sort_field = sort_fields[i];
    collation.emplace_back(i,
                           sort_field.getSortDir() == SortDirection::Descending,
                           sort_field.getNullsPosition() == NullSortedPosition::First);
  }
  return collation;
}

bool supported_lower_bound(
    const RexWindowFunctionOperator::RexWindowBound& window_bound) {
  return window_bound.unbounded && window_bound.preceding && !window_bound.following &&
         !window_bound.is_current_row && !window_bound.offset &&
         window_bound.order_key == 0;
}

bool supported_upper_bound(const RexWindowFunctionOperator* rex_window_function) {
  const auto& window_bound = rex_window_function->getUpperBound();
  const bool to_current_row = !window_bound.unbounded && !window_bound.preceding &&
                              !window_bound.following && window_bound.is_current_row &&
                              !window_bound.offset && window_bound.order_key == 1;
  switch (rex_window_function->getKind()) {
    case SqlWindowFunctionKind::ROW_NUMBER:
    case SqlWindowFunctionKind::RANK:
    case SqlWindowFunctionKind::DENSE_RANK:
    case SqlWindowFunctionKind::CUME_DIST: {
      return to_current_row;
    }
    default: {
      return rex_window_function->getOrderKeys().empty()
                 ? (window_bound.unbounded && !window_bound.preceding &&
                    window_bound.following && !window_bound.is_current_row &&
                    !window_bound.offset && window_bound.order_key == 2)
                 : to_current_row;
    }
  }
}

}  // namespace

hdk::ir::ExprPtr RelAlgTranslator::translateWindowFunction(
    const RexWindowFunctionOperator* rex_window_function) const {
  if (!supported_lower_bound(rex_window_function->getLowerBound()) ||
      !supported_upper_bound(rex_window_function) ||
      ((rex_window_function->getKind() == SqlWindowFunctionKind::ROW_NUMBER) !=
       rex_window_function->isRows())) {
    throw std::runtime_error("Frame specification not supported");
  }
  std::vector<hdk::ir::ExprPtr> args;
  for (size_t i = 0; i < rex_window_function->size(); ++i) {
    args.push_back(translateScalarRex(rex_window_function->getOperand(i)));
  }
  std::vector<hdk::ir::ExprPtr> partition_keys;
  for (const auto& partition_key : rex_window_function->getPartitionKeys()) {
    partition_keys.push_back(translateScalarRex(partition_key.get()));
  }
  std::vector<hdk::ir::ExprPtr> order_keys;
  for (const auto& order_key : rex_window_function->getOrderKeys()) {
    order_keys.push_back(translateScalarRex(order_key.get()));
  }
  auto ti = rex_window_function->getType();
  if (window_function_is_value(rex_window_function->getKind())) {
    CHECK_GE(args.size(), 1u);
    ti = args.front()->get_type_info();
  }
  return hdk::ir::makeExpr<hdk::ir::WindowFunction>(
      ti,
      rex_window_function->getKind(),
      args,
      partition_keys,
      order_keys,
      translate_collation(rex_window_function->getCollation()));
}

hdk::ir::ExprPtrVector RelAlgTranslator::translateFunctionArgs(
    const RexFunctionOperator* rex_function) const {
  std::vector<hdk::ir::ExprPtr> args;
  for (size_t i = 0; i < rex_function->size(); ++i) {
    args.push_back(translateScalarRex(rex_function->getOperand(i)));
  }
  return args;
}

QualsConjunctiveForm qual_to_conjunctive_form(const hdk::ir::ExprPtr qual_expr) {
  CHECK(qual_expr);
  auto bin_oper = std::dynamic_pointer_cast<const hdk::ir::BinOper>(qual_expr);
  if (!bin_oper) {
    const auto rewritten_qual_expr = rewrite_expr(qual_expr.get());
    return {{}, {rewritten_qual_expr ? rewritten_qual_expr : qual_expr}};
  }

  if (bin_oper->get_optype() == kAND) {
    const auto lhs_cf = qual_to_conjunctive_form(bin_oper->get_own_left_operand());
    const auto rhs_cf = qual_to_conjunctive_form(bin_oper->get_own_right_operand());
    auto simple_quals = lhs_cf.simple_quals;
    simple_quals.insert(
        simple_quals.end(), rhs_cf.simple_quals.begin(), rhs_cf.simple_quals.end());
    auto quals = lhs_cf.quals;
    quals.insert(quals.end(), rhs_cf.quals.begin(), rhs_cf.quals.end());
    return {simple_quals, quals};
  }
  int rte_idx{0};
  const auto simple_qual = bin_oper->normalize_simple_predicate(rte_idx);
  return simple_qual ? QualsConjunctiveForm{{simple_qual}, {}}
                     : QualsConjunctiveForm{{}, {qual_expr}};
}

std::vector<hdk::ir::ExprPtr> qual_to_disjunctive_form(
    const hdk::ir::ExprPtr& qual_expr) {
  CHECK(qual_expr);
  const auto bin_oper = std::dynamic_pointer_cast<const hdk::ir::BinOper>(qual_expr);
  if (!bin_oper) {
    const auto rewritten_qual_expr = rewrite_expr(qual_expr.get());
    return {rewritten_qual_expr ? rewritten_qual_expr : qual_expr};
  }
  if (bin_oper->get_optype() == kOR) {
    const auto lhs_df = qual_to_disjunctive_form(bin_oper->get_own_left_operand());
    const auto rhs_df = qual_to_disjunctive_form(bin_oper->get_own_right_operand());
    auto quals = lhs_df;
    quals.insert(quals.end(), rhs_df.begin(), rhs_df.end());
    return quals;
  }
  return {qual_expr};
}

hdk::ir::ExprPtr RelAlgTranslator::translateHPTLiteral(
    const RexFunctionOperator* rex_function) const {
  /* since calcite uses Avatica package called DateTimeUtils to parse timestamp strings.
     Therefore any string having fractional seconds more 3 places after the decimal
     (milliseconds) will get truncated to 3 decimal places, therefore we lose precision
     (us|ns). Issue: [BE-2461] Here we are hijacking literal cast to Timestamp(6|9) from
     calcite and translating them to generate our own casts.
  */
  CHECK_EQ(size_t(1), rex_function->size());
  const auto operand = translateScalarRex(rex_function->getOperand(0));
  const auto& operand_ti = operand->get_type_info();
  const auto& target_ti = rex_function->getType();
  if (!operand_ti.is_string()) {
    throw std::runtime_error(
        "High precision timestamp cast argument must be a string. Input type is: " +
        operand_ti.get_type_name());
  } else if (!target_ti.is_high_precision_timestamp()) {
    throw std::runtime_error(
        "Cast target type should be high precision timestamp. Input type is: " +
        target_ti.get_type_name());
  } else if (target_ti.get_dimension() != 6 && target_ti.get_dimension() != 9) {
    throw std::runtime_error(
        "Cast target type should be TIMESTAMP(6|9). Input type is: TIMESTAMP(" +
        std::to_string(target_ti.get_dimension()) + ")");
  } else {
    return operand->add_cast(target_ti);
  }
}
