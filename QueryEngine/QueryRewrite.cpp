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

#include "QueryRewrite.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "ExpressionRange.h"
#include "ExpressionRewrite.h"
#include "Parser/ParserNode.h"
#include "Shared/Logger.h"

RelAlgExecutionUnit QueryRewriter::rewrite(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  auto rewritten_exe_unit = rewriteConstrainedByIn(ra_exe_unit_in);
  return rewriteOverlapsJoin(rewritten_exe_unit);
}

RelAlgExecutionUnit QueryRewriter::rewriteOverlapsJoin(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  if (!g_enable_overlaps_hashjoin) {
    return ra_exe_unit_in;
  }
  if (ra_exe_unit_in.join_quals.empty()) {
    return ra_exe_unit_in;
  }

  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  quals.insert(quals.end(), ra_exe_unit_in.quals.begin(), ra_exe_unit_in.quals.end());

  JoinQualsPerNestingLevel join_condition_per_nesting_level;
  for (const auto& join_condition_in : ra_exe_unit_in.join_quals) {
    JoinCondition join_condition{{}, join_condition_in.type};

    for (const auto& join_qual_expr_in : join_condition_in.quals) {
      auto new_overlaps_quals = rewrite_overlaps_conjunction(join_qual_expr_in);
      if (new_overlaps_quals) {
        const auto& overlaps_quals = *new_overlaps_quals;

        // Add overlaps qual
        join_condition.quals.insert(join_condition.quals.end(),
                                    overlaps_quals.join_quals.begin(),
                                    overlaps_quals.join_quals.end());

        // Add original quals
        join_condition.quals.insert(join_condition.quals.end(),
                                    overlaps_quals.quals.begin(),
                                    overlaps_quals.quals.end());
      } else {
        join_condition.quals.push_back(join_qual_expr_in);
      }
    }
    join_condition_per_nesting_level.push_back(join_condition);
  }
  return {ra_exe_unit_in.input_descs,
          ra_exe_unit_in.input_col_descs,
          ra_exe_unit_in.simple_quals,
          quals,
          join_condition_per_nesting_level,
          ra_exe_unit_in.groupby_exprs,
          ra_exe_unit_in.target_exprs,
          ra_exe_unit_in.estimator,
          ra_exe_unit_in.sort_info,
          ra_exe_unit_in.scan_limit,
          ra_exe_unit_in.use_bump_allocator};
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
          nullptr,
          ra_exe_unit_in.sort_info,
          ra_exe_unit_in.scan_limit};
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
      ti.set_comp_param(0);
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
    std::shared_ptr<Analyzer::Expr> column_to_update) const {
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
    if (new_column_ti.get_comp_param() > 0 &&
        new_column_ti.get_comp_param() !=
            column_to_update->get_type_info().get_comp_param()) {
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
        auto cat = executor_->getCatalog();
        CHECK(cat);

        CHECK(column_to_update->get_type_info().is_dict_encoded_string());
        const auto dict_id = column_to_update->get_type_info().get_comp_param();
        std::map<int, StringDictionary*> string_dicts;
        const auto dd = cat->getMetadataForDict(dict_id, /*load_dict=*/true);
        CHECK(dd);
        auto string_dict = dd->stringDict;
        CHECK(string_dict);

        auto string_id =
            string_dict->getOrAdd(*original_constant_expr->get_constval().stringval);
        if (check_string_id_overflow(string_id, column_to_update->get_type_info())) {
          throw std::runtime_error(
              "Ran out of space in dictionary, cannot update column with dictionary "
              "encoded string value. Dictionary ID: " +
              std::to_string(dict_id));
        }
        if (string_id == inline_int_null_value<int32_t>()) {
          string_id = inline_fixed_encoding_null_val(column_to_update->get_type_info());
        }

        // Codegen expects a string value. The string will be
        // resolved to its ID during Constant codegen. Copy the string from the
        // original expr
        Datum new_string_datum{.stringval = new std::string(
                                   *original_constant_expr->get_constval().stringval)};

        new_column_value =
            makeExpr<Analyzer::Constant>(column_to_update->get_type_info(),
                                         original_constant_expr->get_is_null(),
                                         new_string_datum);

        // Roll the string dict generation forward, as we have added a string
        if (executor_->string_dictionary_generations_.getGeneration(dict_id) > -1) {
          executor_->string_dictionary_generations_.updateGeneration(
              dict_id, string_dict->storageEntryCount());
        } else {
          // Simple update with no filters does not use a CASE, and therefore does not add
          // a valid generation
          executor_->string_dictionary_generations_.setGeneration(
              dict_id, string_dict->storageEntryCount());
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
    auto col_to_update_desc =
        std::make_shared<const InputColDescriptor>(col_to_update_var->get_column_id(),
                                                   col_to_update_var->get_table_id(),
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
                                         ra_exe_unit_in.estimator,
                                         ra_exe_unit_in.sort_info,
                                         ra_exe_unit_in.scan_limit,
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

  const auto true_datum = Datum{.boolval = true};
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
          return in->getColId() == delete_column->get_column_id();
        });
    CHECK(delete_col_desc_it == input_col_descs.end());
    auto delete_col_desc =
        std::make_shared<const InputColDescriptor>(delete_column->get_column_id(),
                                                   delete_column->get_table_id(),
                                                   delete_column->get_rte_idx());
    input_col_descs.push_back(delete_col_desc);
    target_exprs_owned_.emplace_back(case_expr);
  } else {
    // no filters, simply project the deleted=true column value for all rows
    auto delete_col_desc =
        std::make_shared<const InputColDescriptor>(delete_column->get_column_id(),
                                                   delete_column->get_table_id(),
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
                                         ra_exe_unit_in.estimator,
                                         ra_exe_unit_in.sort_info,
                                         ra_exe_unit_in.scan_limit,
                                         ra_exe_unit_in.use_bump_allocator,
                                         ra_exe_unit_in.union_all,
                                         ra_exe_unit_in.query_state};
  return rewritten_exe_unit;
}
