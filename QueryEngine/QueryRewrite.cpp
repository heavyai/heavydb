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

#include "QueryRewrite.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "ExpressionRange.h"
#include "ExpressionRewrite.h"
#include "Logger/Logger.h"
#include "Parser/ParserNode.h"
#include "Shared/sqltypes.h"

RelAlgExecutionUnit QueryRewriter::rewrite(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  auto rewritten_exe_unit = rewriteConstrainedByIn(ra_exe_unit_in);
  return rewriteAggregateOnGroupByColumn(rewritten_exe_unit);
}

RelAlgExecutionUnit QueryRewriter::rewriteConstrainedByIn(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  if (ra_exe_unit_in.groupby_exprs.empty()) {
    return ra_exe_unit_in;
  }
  if (ra_exe_unit_in.groupby_exprs.size() == 1 && !ra_exe_unit_in.groupby_exprs.front()) {
    return ra_exe_unit_in;
  }
  if (!ra_exe_unit_in.simple_quals.empty()) {
    return ra_exe_unit_in;
  }
  if (ra_exe_unit_in.quals.size() != 1) {
    return ra_exe_unit_in;
  }
  auto in_vals =
      std::dynamic_pointer_cast<Analyzer::InValues>(ra_exe_unit_in.quals.front());
  if (!in_vals) {
    in_vals = std::dynamic_pointer_cast<Analyzer::InValues>(
        rewrite_expr(ra_exe_unit_in.quals.front().get()));
  }
  if (!in_vals || in_vals->get_value_list().empty()) {
    return ra_exe_unit_in;
  }
  for (const auto& in_val : in_vals->get_value_list()) {
    if (!std::dynamic_pointer_cast<Analyzer::Constant>(in_val)) {
      break;
    }
  }
  if (dynamic_cast<const Analyzer::CaseExpr*>(in_vals->get_arg())) {
    return ra_exe_unit_in;
  }
  auto in_val_cv = dynamic_cast<const Analyzer::ColumnVar*>(in_vals->get_arg());
  if (in_val_cv) {
    auto it = std::find_if(
        ra_exe_unit_in.groupby_exprs.begin(),
        ra_exe_unit_in.groupby_exprs.end(),
        [&in_val_cv](std::shared_ptr<Analyzer::Expr> groupby_expr) {
          if (auto groupby_cv =
                  std::dynamic_pointer_cast<Analyzer::ColumnVar>(groupby_expr)) {
            return *in_val_cv == *groupby_cv.get();
          }
          return false;
        });
    if (it != ra_exe_unit_in.groupby_exprs.end()) {
      // we do not need to deploy case-when rewriting when in_val cv is listed as groupby
      // col i.e., ... WHERE v IN (SELECT DISTINCT v FROM ...)
      return ra_exe_unit_in;
    }
  }
  auto case_expr = generateCaseForDomainValues(in_vals.get());
  return rewriteConstrainedByInImpl(ra_exe_unit_in, case_expr, in_vals.get());
}

RelAlgExecutionUnit QueryRewriter::rewriteConstrainedByInImpl(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const std::shared_ptr<Analyzer::CaseExpr> case_expr,
    const Analyzer::InValues* in_vals) const {
  std::list<std::shared_ptr<Analyzer::Expr>> new_groupby_list;
  std::vector<Analyzer::Expr*> new_target_exprs;
  bool rewrite{false};
  size_t groupby_idx{0};
  auto it = ra_exe_unit_in.groupby_exprs.begin();
  for (const auto& group_expr : ra_exe_unit_in.groupby_exprs) {
    CHECK(group_expr);
    ++groupby_idx;
    if (*group_expr == *in_vals->get_arg()) {
      const auto expr_range = getExpressionRange(it->get(), query_infos_, executor_);
      if (expr_range.getType() != ExpressionRangeType::Integer) {
        ++it;
        continue;
      }
      const size_t range_sz = expr_range.getIntMax() - expr_range.getIntMin() + 1;
      if (range_sz <= in_vals->get_value_list().size() * g_constrained_by_in_threshold) {
        ++it;
        continue;
      }
      new_groupby_list.push_back(case_expr);
      for (size_t i = 0; i < ra_exe_unit_in.target_exprs.size(); ++i) {
        const auto target = ra_exe_unit_in.target_exprs[i];
        if (*target == *in_vals->get_arg()) {
          auto var_case_expr = makeExpr<Analyzer::Var>(
              case_expr->get_type_info(), Analyzer::Var::kGROUPBY, groupby_idx);
          target_exprs_owned_.push_back(var_case_expr);
          new_target_exprs.push_back(var_case_expr.get());
        } else {
          new_target_exprs.push_back(target);
        }
      }
      rewrite = true;
    } else {
      new_groupby_list.push_back(group_expr);
    }
    ++it;
  }
  if (!rewrite) {
    return ra_exe_unit_in;
  }
  return {ra_exe_unit_in.input_descs,
          ra_exe_unit_in.input_col_descs,
          ra_exe_unit_in.simple_quals,
          ra_exe_unit_in.quals,
          ra_exe_unit_in.join_quals,
          new_groupby_list,
          new_target_exprs,
          ra_exe_unit_in.target_exprs_original_type_infos,
          nullptr,
          ra_exe_unit_in.sort_info,
          ra_exe_unit_in.scan_limit,
          ra_exe_unit_in.query_hint,
          ra_exe_unit_in.query_plan_dag_hash,
          ra_exe_unit_in.hash_table_build_plan_dag,
          ra_exe_unit_in.table_id_to_node_map};
}

std::shared_ptr<Analyzer::CaseExpr> QueryRewriter::generateCaseForDomainValues(
    const Analyzer::InValues* in_vals) {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      case_expr_list;
  auto in_val_arg = in_vals->get_arg()->deep_copy();
  for (const auto& in_val : in_vals->get_value_list()) {
    auto case_cond = makeExpr<Analyzer::BinOper>(
        SQLTypeInfo(kBOOLEAN, true), false, kEQ, kONE, in_val_arg, in_val);
    auto in_val_copy = in_val->deep_copy();
    auto ti = in_val_copy->get_type_info();
    if (ti.is_string() && ti.get_compression() == kENCODING_DICT) {
      ti.set_comp_param(TRANSIENT_DICT_ID);
      ti.setStringDictKey(shared::StringDictKey::kTransientDictKey);
    }
    in_val_copy->set_type_info(ti);
    case_expr_list.emplace_back(case_cond, in_val_copy);
  }
  // TODO(alex): refine the expression range for case with empty else expression;
  //             for now, add a dummy else which should never be taken
  auto else_expr = case_expr_list.front().second;
  return makeExpr<Analyzer::CaseExpr>(
      case_expr_list.front().second->get_type_info(), false, case_expr_list, else_expr);
}

std::shared_ptr<Analyzer::CaseExpr>
QueryRewriter::generateCaseExprForCountDistinctOnGroupByCol(
    std::shared_ptr<Analyzer::Expr> expr) const {
  std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
      case_expr_list;
  auto is_null = std::make_shared<Analyzer::UOper>(kBOOLEAN, kISNULL, expr);
  auto is_not_null = std::make_shared<Analyzer::UOper>(kBOOLEAN, kNOT, is_null);
  Datum then_d;
  then_d.bigintval = 1;
  const auto then_constant = makeExpr<Analyzer::Constant>(kBIGINT, false, then_d);
  case_expr_list.emplace_back(is_not_null, then_constant);
  Datum else_d;
  else_d.bigintval = 0;
  const auto else_constant = makeExpr<Analyzer::Constant>(kBIGINT, false, else_d);
  auto case_expr = makeExpr<Analyzer::CaseExpr>(
      then_constant->get_type_info(), false, case_expr_list, else_constant);
  return case_expr;
}

namespace {

// TODO(adb): centralize and share (e..g with insert_one_dict_str)
bool check_string_id_overflow(const int32_t string_id, const SQLTypeInfo& ti) {
  switch (ti.get_size()) {
    case 1:
      return string_id > max_valid_int_value<int8_t>();
    case 2:
      return string_id > max_valid_int_value<int16_t>();
    case 4:
      return string_id > max_valid_int_value<int32_t>();
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return false;
}

}  // namespace

/* Rewrites an update query of the form `SELECT new_value, OFFSET_IN_FRAGMENT() FROM t
 * WHERE <update_filter_condition>` to `SELECT CASE WHEN <update_filer_condition> THEN
 * new_value ELSE existing value END FROM t`
 */
RelAlgExecutionUnit QueryRewriter::rewriteColumnarUpdate(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    std::shared_ptr<Analyzer::ColumnVar> column_to_update) const {
  CHECK_EQ(ra_exe_unit_in.target_exprs.size(), size_t(2));
  CHECK(ra_exe_unit_in.groupby_exprs.size() == 1 &&
        !ra_exe_unit_in.groupby_exprs.front());

  if (ra_exe_unit_in.join_quals.size() > 0) {
    throw std::runtime_error("Update via join not yet supported for temporary tables.");
  }

  auto new_column_value = ra_exe_unit_in.target_exprs.front()->deep_copy();
  const auto& new_column_ti = new_column_value->get_type_info();
  if (column_to_update->get_type_info().is_dict_encoded_string()) {
    CHECK(new_column_ti.is_dict_encoded_string());
    if (new_column_ti.getStringDictKey().dict_id > 0 &&
        new_column_ti.getStringDictKey() !=
            column_to_update->get_type_info().getStringDictKey()) {
      throw std::runtime_error(
          "Updating a dictionary encoded string using another dictionary encoded string "
          "column is not yet supported, unless both columns share dictionaries.");
    }
    if (auto uoper = dynamic_cast<Analyzer::UOper*>(new_column_value.get())) {
      if (uoper->get_optype() == kCAST &&
          dynamic_cast<const Analyzer::Constant*>(uoper->get_operand())) {
        const auto original_constant_expr =
            dynamic_cast<const Analyzer::Constant*>(uoper->get_operand());
        CHECK(original_constant_expr);
        CHECK(original_constant_expr->get_type_info().is_string());
        // extract the string, insert it into the dict for the table we are updating,
        // and place the dictionary ID in the oper

        CHECK(column_to_update->get_type_info().is_dict_encoded_string());
        const auto& dict_key = column_to_update->get_type_info().getStringDictKey();
        std::map<int, StringDictionary*> string_dicts;
        const auto catalog =
            Catalog_Namespace::SysCatalog::instance().getCatalog(dict_key.db_id);
        CHECK(catalog);
        const auto dd = catalog->getMetadataForDict(dict_key.dict_id, /*load_dict=*/true);
        CHECK(dd);
        auto string_dict = dd->stringDict;
        CHECK(string_dict);

        auto string_id =
            string_dict->getOrAdd(*original_constant_expr->get_constval().stringval);
        if (check_string_id_overflow(string_id, column_to_update->get_type_info())) {
          throw std::runtime_error(
              "Ran out of space in dictionary, cannot update column with dictionary "
              "encoded string value. Dictionary ID: " +
              std::to_string(dict_key.dict_id));
        }
        if (string_id == inline_int_null_value<int32_t>()) {
          string_id = inline_fixed_encoding_null_val(column_to_update->get_type_info());
        }

        // Codegen expects a string value. The string will be
        // resolved to its ID during Constant codegen. Copy the string from the
        // original expr
        Datum datum;
        datum.stringval =
            new std::string(*original_constant_expr->get_constval().stringval);
        Datum new_string_datum{datum};

        new_column_value =
            makeExpr<Analyzer::Constant>(column_to_update->get_type_info(),
                                         original_constant_expr->get_is_null(),
                                         new_string_datum);

        // Roll the string dict generation forward, as we have added a string
        auto row_set_mem_owner = executor_->getRowSetMemoryOwner();
        CHECK(row_set_mem_owner);
        auto& str_dict_generations = row_set_mem_owner->getStringDictionaryGenerations();
        if (str_dict_generations.getGeneration(dict_key) > -1) {
          str_dict_generations.updateGeneration(dict_key,
                                                string_dict->storageEntryCount());
        } else {
          // Simple update with no filters does not use a CASE, and therefore does not add
          // a valid generation
          str_dict_generations.setGeneration(dict_key, string_dict->storageEntryCount());
        }
      }
    }
  }

  auto input_col_descs = ra_exe_unit_in.input_col_descs;

  std::shared_ptr<Analyzer::Expr> filter;
  std::vector<std::shared_ptr<Analyzer::Expr>> filter_exprs;
  filter_exprs.insert(filter_exprs.end(),
                      ra_exe_unit_in.simple_quals.begin(),
                      ra_exe_unit_in.simple_quals.end());
  filter_exprs.insert(
      filter_exprs.end(), ra_exe_unit_in.quals.begin(), ra_exe_unit_in.quals.end());

  if (filter_exprs.size() > 0) {
    std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
        case_expr_list;
    if (filter_exprs.size() == 1) {
      filter = filter_exprs.front();
    } else {
      filter = std::accumulate(
          std::next(filter_exprs.begin()),
          filter_exprs.end(),
          filter_exprs.front(),
          [](const std::shared_ptr<Analyzer::Expr> a,
             const std::shared_ptr<Analyzer::Expr> b) {
            CHECK_EQ(a->get_type_info().get_type(), b->get_type_info().get_type());
            return makeExpr<Analyzer::BinOper>(a->get_type_info().get_type(),
                                               SQLOps::kAND,
                                               SQLQualifier::kONE,
                                               a->deep_copy(),
                                               b->deep_copy());
          });
    }
    auto when_expr = filter;  // only one filter, will be a BinOper if multiple filters
    case_expr_list.emplace_back(std::make_pair(when_expr, new_column_value));
    auto case_expr = Parser::CaseExpr::normalize(case_expr_list, column_to_update);

    auto col_to_update_var =
        std::dynamic_pointer_cast<Analyzer::ColumnVar>(column_to_update);
    CHECK(col_to_update_var);
    const auto& column_key = col_to_update_var->getColumnKey();
    auto col_to_update_desc =
        std::make_shared<const InputColDescriptor>(column_key.column_id,
                                                   column_key.table_id,
                                                   column_key.db_id,
                                                   col_to_update_var->get_rte_idx());
    auto existing_col_desc_it = std::find_if(
        input_col_descs.begin(),
        input_col_descs.end(),
        [&col_to_update_desc](const std::shared_ptr<const InputColDescriptor>& in) {
          return *in == *col_to_update_desc;
        });
    if (existing_col_desc_it == input_col_descs.end()) {
      input_col_descs.push_back(col_to_update_desc);
    }
    target_exprs_owned_.emplace_back(case_expr);
  } else {
    // no filters, simply project the update value
    target_exprs_owned_.emplace_back(new_column_value);
  }

  std::vector<Analyzer::Expr*> target_exprs;
  CHECK_EQ(target_exprs_owned_.size(), size_t(1));
  target_exprs.emplace_back(target_exprs_owned_.front().get());

  RelAlgExecutionUnit rewritten_exe_unit{ra_exe_unit_in.input_descs,
                                         input_col_descs,
                                         {},
                                         {},
                                         ra_exe_unit_in.join_quals,
                                         ra_exe_unit_in.groupby_exprs,
                                         target_exprs,
                                         ra_exe_unit_in.target_exprs_original_type_infos,
                                         ra_exe_unit_in.estimator,
                                         ra_exe_unit_in.sort_info,
                                         ra_exe_unit_in.scan_limit,
                                         ra_exe_unit_in.query_hint,
                                         ra_exe_unit_in.query_plan_dag_hash,
                                         ra_exe_unit_in.hash_table_build_plan_dag,
                                         ra_exe_unit_in.table_id_to_node_map,
                                         ra_exe_unit_in.use_bump_allocator,
                                         ra_exe_unit_in.union_all,
                                         ra_exe_unit_in.query_state};
  return rewritten_exe_unit;
}

/* Rewrites a delete query of the form `SELECT OFFSET_IN_FRAGMENT() FROM t
 * WHERE <delete_filter_condition>` to `SELECT CASE WHEN <delete_filter_condition> THEN
 * true ELSE existing value END FROM t`
 */
RelAlgExecutionUnit QueryRewriter::rewriteColumnarDelete(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    std::shared_ptr<Analyzer::ColumnVar> delete_column) const {
  CHECK_EQ(ra_exe_unit_in.target_exprs.size(), size_t(1));
  CHECK(ra_exe_unit_in.groupby_exprs.size() == 1 &&
        !ra_exe_unit_in.groupby_exprs.front());

  // TODO(adb): is this possible?
  if (ra_exe_unit_in.join_quals.size() > 0) {
    throw std::runtime_error("Delete via join not yet supported for temporary tables.");
  }

  Datum true_datum;
  true_datum.boolval = true;
  const auto deleted_constant =
      makeExpr<Analyzer::Constant>(delete_column->get_type_info(), false, true_datum);

  auto input_col_descs = ra_exe_unit_in.input_col_descs;

  std::shared_ptr<Analyzer::Expr> filter;
  std::vector<std::shared_ptr<Analyzer::Expr>> filter_exprs;
  filter_exprs.insert(filter_exprs.end(),
                      ra_exe_unit_in.simple_quals.begin(),
                      ra_exe_unit_in.simple_quals.end());
  filter_exprs.insert(
      filter_exprs.end(), ra_exe_unit_in.quals.begin(), ra_exe_unit_in.quals.end());

  if (filter_exprs.size() > 0) {
    std::list<std::pair<std::shared_ptr<Analyzer::Expr>, std::shared_ptr<Analyzer::Expr>>>
        case_expr_list;
    if (filter_exprs.size() == 1) {
      filter = filter_exprs.front();
    } else {
      filter = std::accumulate(
          std::next(filter_exprs.begin()),
          filter_exprs.end(),
          filter_exprs.front(),
          [](const std::shared_ptr<Analyzer::Expr> a,
             const std::shared_ptr<Analyzer::Expr> b) {
            CHECK_EQ(a->get_type_info().get_type(), b->get_type_info().get_type());
            return makeExpr<Analyzer::BinOper>(a->get_type_info().get_type(),
                                               SQLOps::kAND,
                                               SQLQualifier::kONE,
                                               a->deep_copy(),
                                               b->deep_copy());
          });
    }
    std::shared_ptr<Analyzer::Expr> column_to_update{nullptr};
    auto when_expr = filter;  // only one filter, will be a BinOper if multiple filters
    case_expr_list.emplace_back(std::make_pair(when_expr, deleted_constant));
    auto case_expr = Parser::CaseExpr::normalize(case_expr_list, delete_column);

    // the delete column should not be projected, but check anyway
    auto delete_col_desc_it = std::find_if(
        input_col_descs.begin(),
        input_col_descs.end(),
        [&delete_column](const std::shared_ptr<const InputColDescriptor>& in) {
          return in->getColId() == delete_column->getColumnKey().column_id;
        });
    CHECK(delete_col_desc_it == input_col_descs.end());
    const auto& column_key = delete_column->getColumnKey();
    auto delete_col_desc =
        std::make_shared<const InputColDescriptor>(column_key.column_id,
                                                   column_key.table_id,
                                                   column_key.db_id,
                                                   delete_column->get_rte_idx());
    input_col_descs.push_back(delete_col_desc);
    target_exprs_owned_.emplace_back(case_expr);
  } else {
    // no filters, simply project the deleted=true column value for all rows
    const auto& column_key = delete_column->getColumnKey();
    auto delete_col_desc =
        std::make_shared<const InputColDescriptor>(column_key.column_id,
                                                   column_key.table_id,
                                                   column_key.db_id,
                                                   delete_column->get_rte_idx());
    input_col_descs.push_back(delete_col_desc);
    target_exprs_owned_.emplace_back(deleted_constant);
  }

  std::vector<Analyzer::Expr*> target_exprs;
  CHECK_EQ(target_exprs_owned_.size(), size_t(1));
  target_exprs.emplace_back(target_exprs_owned_.front().get());

  RelAlgExecutionUnit rewritten_exe_unit{ra_exe_unit_in.input_descs,
                                         input_col_descs,
                                         {},
                                         {},
                                         ra_exe_unit_in.join_quals,
                                         ra_exe_unit_in.groupby_exprs,
                                         target_exprs,
                                         ra_exe_unit_in.target_exprs_original_type_infos,
                                         ra_exe_unit_in.estimator,
                                         ra_exe_unit_in.sort_info,
                                         ra_exe_unit_in.scan_limit,
                                         ra_exe_unit_in.query_hint,
                                         ra_exe_unit_in.query_plan_dag_hash,
                                         ra_exe_unit_in.hash_table_build_plan_dag,
                                         ra_exe_unit_in.table_id_to_node_map,
                                         ra_exe_unit_in.use_bump_allocator,
                                         ra_exe_unit_in.union_all,
                                         ra_exe_unit_in.query_state};
  return rewritten_exe_unit;
}

std::pair<bool, std::set<size_t>> QueryRewriter::is_all_groupby_exprs_are_col_var(
    const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs) const {
  std::set<size_t> gby_col_exprs_hash;
  for (auto gby_expr : groupby_exprs) {
    if (auto gby_col_var = std::dynamic_pointer_cast<Analyzer::ColumnVar>(gby_expr)) {
      gby_col_exprs_hash.insert(boost::hash_value(gby_col_var->toString()));
    } else {
      return {false, {}};
    }
  }
  return {true, gby_col_exprs_hash};
}

RelAlgExecutionUnit QueryRewriter::rewriteAggregateOnGroupByColumn(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  auto check_precond = is_all_groupby_exprs_are_col_var(ra_exe_unit_in.groupby_exprs);
  auto is_expr_on_gby_col = [&check_precond](const Analyzer::AggExpr* agg_expr) {
    CHECK(agg_expr);
    if (agg_expr->get_arg()) {
      // some expr does not have its own arg, i.e., count(*)
      auto agg_expr_hash = boost::hash_value(agg_expr->get_arg()->toString());
      // a valid expr should have hashed value > 0
      CHECK_GT(agg_expr_hash, 0u);
      if (check_precond.second.count(agg_expr_hash)) {
        return true;
      }
    }
    return false;
  };
  if (!check_precond.first) {
    // return the input ra_exe_unit if we have gby expr which is not col_var
    // i.e., group by x+1, y instead of group by x, y
    // todo (yoonmin) : can we relax this with a simple analysis of groupby / agg exprs?
    return ra_exe_unit_in;
  }

  std::vector<Analyzer::Expr*> new_target_exprs;
  for (auto expr : ra_exe_unit_in.target_exprs) {
    bool rewritten = false;
    if (auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(expr)) {
      if (is_expr_on_gby_col(agg_expr)) {
        auto target_expr = agg_expr->get_arg();
        // we have some issues when this rewriting is applied to float_type groupby column
        // in subquery, i.e., SELECT MIN(v1) FROM (SELECT v1, AGG(v1) FROM T GROUP BY v1);
        if (target_expr && target_expr->get_type_info().get_type() != SQLTypes::kFLOAT) {
          switch (agg_expr->get_aggtype()) {
            case SQLAgg::kCOUNT:
            case SQLAgg::kCOUNT_IF:
            case SQLAgg::kAPPROX_COUNT_DISTINCT: {
              if (agg_expr->get_aggtype() == SQLAgg::kCOUNT &&
                  !agg_expr->get_is_distinct()) {
                break;
              }
              auto case_expr =
                  generateCaseExprForCountDistinctOnGroupByCol(agg_expr->get_own_arg());
              new_target_exprs.push_back(case_expr.get());
              target_exprs_owned_.emplace_back(case_expr);
              rewritten = true;
              break;
            }
            case SQLAgg::kAPPROX_QUANTILE:
            case SQLAgg::kAVG:
            case SQLAgg::kSAMPLE:
            case SQLAgg::kMAX:
            case SQLAgg::kMIN: {
              // we just replace the agg_expr into a plain expr
              // i.e, avg(x1) --> x1
              auto agg_expr_ti = agg_expr->get_type_info();
              auto target_expr = agg_expr->get_own_arg();
              if (agg_expr_ti != target_expr->get_type_info()) {
                target_expr = target_expr->add_cast(agg_expr_ti);
              }
              new_target_exprs.push_back(target_expr.get());
              target_exprs_owned_.emplace_back(target_expr);
              rewritten = true;
              break;
            }
            default:
              break;
          }
        }
      }
    }
    if (!rewritten) {
      new_target_exprs.push_back(expr);
    }
  }

  RelAlgExecutionUnit rewritten_exe_unit{ra_exe_unit_in.input_descs,
                                         ra_exe_unit_in.input_col_descs,
                                         ra_exe_unit_in.simple_quals,
                                         ra_exe_unit_in.quals,
                                         ra_exe_unit_in.join_quals,
                                         ra_exe_unit_in.groupby_exprs,
                                         new_target_exprs,
                                         ra_exe_unit_in.target_exprs_original_type_infos,
                                         ra_exe_unit_in.estimator,
                                         ra_exe_unit_in.sort_info,
                                         ra_exe_unit_in.scan_limit,
                                         ra_exe_unit_in.query_hint,
                                         ra_exe_unit_in.query_plan_dag_hash,
                                         ra_exe_unit_in.hash_table_build_plan_dag,
                                         ra_exe_unit_in.table_id_to_node_map,
                                         ra_exe_unit_in.use_bump_allocator,
                                         ra_exe_unit_in.union_all,
                                         ra_exe_unit_in.query_state};
  return rewritten_exe_unit;
}
