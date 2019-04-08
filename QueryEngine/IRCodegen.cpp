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

#include "../Parser/ParserNode.h"
#include "Execute.h"
#include "MaxwellCodegenPatch.h"
#include "RelAlgTranslator.h"

// Driver methods for the IR generation.

std::vector<llvm::Value*> Executor::codegen(const Analyzer::Expr* expr,
                                            const bool fetch_columns,
                                            const CompilationOptions& co) {
  if (!expr) {
    return {posArg(expr)};
  }
  auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper) {
    return {codegen(bin_oper, co)};
  }
  auto u_oper = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_oper) {
    return {codegen(u_oper, co)};
  }
  auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (col_var) {
    return codegen(col_var, fetch_columns, co);
  }
  auto constant = dynamic_cast<const Analyzer::Constant*>(expr);
  if (constant) {
    if (constant->get_is_null()) {
      const auto& ti = constant->get_type_info();
      return {ti.is_fp() ? static_cast<llvm::Value*>(inlineFpNull(ti))
                         : static_cast<llvm::Value*>(inlineIntNull(ti))};
    }
    // The dictionary encoding case should be handled by the parent expression
    // (cast, for now), here is too late to know the dictionary id
    CHECK_NE(kENCODING_DICT, constant->get_type_info().get_compression());
    return {codegen(constant, constant->get_type_info().get_compression(), 0, co)};
  }
  auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (case_expr) {
    return {codegen(case_expr, co)};
  }
  auto extract_expr = dynamic_cast<const Analyzer::ExtractExpr*>(expr);
  if (extract_expr) {
    return {codegen(extract_expr, co)};
  }
  auto dateadd_expr = dynamic_cast<const Analyzer::DateaddExpr*>(expr);
  if (dateadd_expr) {
    return {codegen(dateadd_expr, co)};
  }
  auto datediff_expr = dynamic_cast<const Analyzer::DatediffExpr*>(expr);
  if (datediff_expr) {
    return {codegen(datediff_expr, co)};
  }
  auto datetrunc_expr = dynamic_cast<const Analyzer::DatetruncExpr*>(expr);
  if (datetrunc_expr) {
    return {codegen(datetrunc_expr, co)};
  }
  auto charlength_expr = dynamic_cast<const Analyzer::CharLengthExpr*>(expr);
  if (charlength_expr) {
    return {codegen(charlength_expr, co)};
  }
  auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
  if (like_expr) {
    return {codegen(like_expr, co)};
  }
  auto regexp_expr = dynamic_cast<const Analyzer::RegexpExpr*>(expr);
  if (regexp_expr) {
    return {codegen(regexp_expr, co)};
  }
  auto likelihood_expr = dynamic_cast<const Analyzer::LikelihoodExpr*>(expr);
  if (likelihood_expr) {
    return {codegen(likelihood_expr->get_arg(), fetch_columns, co)};
  }
  auto in_expr = dynamic_cast<const Analyzer::InValues*>(expr);
  if (in_expr) {
    return {codegen(in_expr, co)};
  }
  auto in_integer_set_expr = dynamic_cast<const Analyzer::InIntegerSet*>(expr);
  if (in_integer_set_expr) {
    return {codegen(in_integer_set_expr, co)};
  }
  auto function_oper_with_custom_type_handling_expr =
      dynamic_cast<const Analyzer::FunctionOperWithCustomTypeHandling*>(expr);
  if (function_oper_with_custom_type_handling_expr) {
    return {codegenFunctionOperWithCustomTypeHandling(
        function_oper_with_custom_type_handling_expr, co)};
  }
  auto array_oper_expr = dynamic_cast<const Analyzer::ArrayExpr*>(expr);
  if (array_oper_expr) {
    return {codegenArrayExpr(array_oper_expr, co)};
  }
  auto function_oper_expr = dynamic_cast<const Analyzer::FunctionOper*>(expr);
  if (function_oper_expr) {
    return {codegenFunctionOper(function_oper_expr, co)};
  }
  if (dynamic_cast<const Analyzer::OffsetInFragment*>(expr)) {
    return {posArg(nullptr)};
  }
  abort();
}

llvm::Value* Executor::codegen(const Analyzer::BinOper* bin_oper,
                               const CompilationOptions& co) {
  const auto optype = bin_oper->get_optype();
  if (IS_ARITHMETIC(optype)) {
    return codegenArith(bin_oper, co);
  }
  if (IS_COMPARISON(optype)) {
    return codegenCmp(bin_oper, co);
  }
  if (IS_LOGIC(optype)) {
    return codegenLogical(bin_oper, co);
  }
  if (optype == kARRAY_AT) {
    return codegenArrayAt(bin_oper, co);
  }
  abort();
}

llvm::Value* Executor::codegen(const Analyzer::UOper* u_oper,
                               const CompilationOptions& co) {
  const auto optype = u_oper->get_optype();
  switch (optype) {
    case kNOT:
      return codegenLogical(u_oper, co);
    case kCAST:
      return codegenCast(u_oper, co);
    case kUMINUS:
      return codegenUMinus(u_oper, co);
    case kISNULL:
      return codegenIsNull(u_oper, co);
    case kUNNEST:
      return codegenUnnest(u_oper, co);
    default:
      abort();
  }
}

namespace {

void add_qualifier_to_execution_unit(RelAlgExecutionUnit& ra_exe_unit,
                                     const std::shared_ptr<Analyzer::Expr>& qual) {
  const auto qual_cf = qual_to_conjunctive_form(qual);
  ra_exe_unit.simple_quals.insert(ra_exe_unit.simple_quals.end(),
                                  qual_cf.simple_quals.begin(),
                                  qual_cf.simple_quals.end());
  ra_exe_unit.quals.insert(
      ra_exe_unit.quals.end(), qual_cf.quals.begin(), qual_cf.quals.end());
}

void check_if_loop_join_is_allowed(RelAlgExecutionUnit& ra_exe_unit,
                                   const ExecutionOptions& eo,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const size_t level_idx,
                                   const std::string& fail_reason) {
  if (eo.allow_loop_joins) {
    return;
  }
  if (level_idx + 1 != ra_exe_unit.join_quals.size()) {
    throw std::runtime_error(
        "Hash join failed, reason(s): " + fail_reason +
        " | Cannot fall back to loop join for intermediate join quals");
  }
  if (!is_trivial_loop_join(query_infos, ra_exe_unit)) {
    throw std::runtime_error(
        "Hash join failed, reason(s): " + fail_reason +
        " | Cannot fall back to loop join for non-trivial inner table size");
  }
}

}  // namespace

std::vector<JoinLoop> Executor::buildJoinLoops(
    RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const std::vector<InputTableInfo>& query_infos,
    ColumnCacheMap& column_cache) {
  INJECT_TIMER(buildJoinLoops);
  std::vector<JoinLoop> join_loops;
  for (size_t level_idx = 0, current_hash_table_idx = 0;
       level_idx < ra_exe_unit.join_quals.size();
       ++level_idx) {
    const auto& current_level_join_conditions = ra_exe_unit.join_quals[level_idx];
    std::vector<std::string> fail_reasons;
    const auto current_level_hash_table =
        buildCurrentLevelHashTable(current_level_join_conditions,
                                   ra_exe_unit,
                                   co,
                                   query_infos,
                                   column_cache,
                                   fail_reasons);
    const auto found_outer_join_matches_cb =
        [this, level_idx](llvm::Value* found_outer_join_matches) {
          CHECK_LT(level_idx, cgen_state_->outer_join_match_found_per_level_.size());
          CHECK(!cgen_state_->outer_join_match_found_per_level_[level_idx]);
          cgen_state_->outer_join_match_found_per_level_[level_idx] =
              found_outer_join_matches;
        };
    const auto is_deleted_cb = buildIsDeletedCb(ra_exe_unit, level_idx, co);
    if (current_level_hash_table) {
      if (current_level_hash_table->getHashType() == JoinHashTable::HashType::OneToOne) {
        join_loops.emplace_back(
            JoinLoopKind::Singleton,
            current_level_join_conditions.type,
            [this, current_hash_table_idx, level_idx, current_level_hash_table, &co](
                const std::vector<llvm::Value*>& prev_iters) {
              addJoinLoopIterator(prev_iters, level_idx);
              JoinLoopDomain domain{{0}};
              domain.slot_lookup_result =
                  current_level_hash_table->codegenSlot(co, current_hash_table_idx);
              return domain;
            },
            nullptr,
            current_level_join_conditions.type == JoinType::LEFT
                ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
                : nullptr,
            is_deleted_cb);
      } else {
        join_loops.emplace_back(
            JoinLoopKind::Set,
            current_level_join_conditions.type,
            [this, current_hash_table_idx, level_idx, current_level_hash_table, &co](
                const std::vector<llvm::Value*>& prev_iters) {
              addJoinLoopIterator(prev_iters, level_idx);
              JoinLoopDomain domain{{0}};
              const auto matching_set = current_level_hash_table->codegenMatchingSet(
                  co, current_hash_table_idx);
              domain.values_buffer = matching_set.elements;
              domain.element_count = matching_set.count;
              return domain;
            },
            nullptr,
            current_level_join_conditions.type == JoinType::LEFT
                ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
                : nullptr,
            is_deleted_cb);
      }
      ++current_hash_table_idx;
    } else {
      const auto fail_reasons_str = current_level_join_conditions.quals.empty()
                                        ? "No equijoin expression found"
                                        : boost::algorithm::join(fail_reasons, " | ");
      check_if_loop_join_is_allowed(
          ra_exe_unit, eo, query_infos, level_idx, fail_reasons_str);
      // Callback provided to the `JoinLoop` framework to evaluate the (outer) join
      // condition.
      VLOG(1) << "Unable to build hash table, falling back to loop join: "
              << fail_reasons_str;
      const auto outer_join_condition_cb =
          [this, level_idx, &co, &current_level_join_conditions](
              const std::vector<llvm::Value*>& prev_iters) {
            // The values generated for the match path don't dominate all uses
            // since on the non-match path nulls are generated. Reset the cache
            // once the condition is generated to avoid incorrect reuse.
            FetchCacheAnchor anchor(cgen_state_.get());
            addJoinLoopIterator(prev_iters, level_idx + 1);
            llvm::Value* left_join_cond = ll_bool(true);
            for (auto expr : current_level_join_conditions.quals) {
              left_join_cond = cgen_state_->ir_builder_.CreateAnd(
                  left_join_cond, toBool(codegen(expr.get(), true, co).front()));
            }
            return left_join_cond;
          };
      join_loops.emplace_back(
          JoinLoopKind::UpperBound,
          current_level_join_conditions.type,
          [this, level_idx](const std::vector<llvm::Value*>& prev_iters) {
            addJoinLoopIterator(prev_iters, level_idx);
            JoinLoopDomain domain{{0}};
            const auto rows_per_scan_ptr = cgen_state_->ir_builder_.CreateGEP(
                get_arg_by_name(cgen_state_->row_func_, "num_rows_per_scan"),
                ll_int(int32_t(level_idx + 1)));
            domain.upper_bound = cgen_state_->ir_builder_.CreateLoad(rows_per_scan_ptr,
                                                                     "num_rows_per_scan");
            return domain;
          },
          current_level_join_conditions.type == JoinType::LEFT
              ? std::function<llvm::Value*(const std::vector<llvm::Value*>&)>(
                    outer_join_condition_cb)
              : nullptr,
          current_level_join_conditions.type == JoinType::LEFT
              ? std::function<void(llvm::Value*)>(found_outer_join_matches_cb)
              : nullptr,
          is_deleted_cb);
    }
  }
  return join_loops;
}

std::function<llvm::Value*(const std::vector<llvm::Value*>&, llvm::Value*)>
Executor::buildIsDeletedCb(const RelAlgExecutionUnit& ra_exe_unit,
                           const size_t level_idx,
                           const CompilationOptions& co) {
  CHECK_LT(level_idx + 1, ra_exe_unit.input_descs.size());
  const auto input_desc = ra_exe_unit.input_descs[level_idx + 1];
  if (input_desc.getSourceType() != InputSourceType::TABLE) {
    return nullptr;
  }
  const auto td = catalog_->getMetadataForTable(input_desc.getTableId());
  CHECK(td);
  const auto deleted_cd = catalog_->getDeletedColumnIfRowsDeleted(td);
  if (!deleted_cd) {
    return nullptr;
  }
  CHECK(deleted_cd->columnType.is_boolean());
  const auto deleted_expr = makeExpr<Analyzer::ColumnVar>(deleted_cd->columnType,
                                                          input_desc.getTableId(),
                                                          deleted_cd->columnId,
                                                          input_desc.getNestLevel());
  return [this, deleted_expr, level_idx, &co](const std::vector<llvm::Value*>& prev_iters,
                                              llvm::Value* have_more_inner_rows) {
    const auto matching_row_index = addJoinLoopIterator(prev_iters, level_idx + 1);
    // Avoid fetching the deleted column from a position which is not valid.
    // An invalid position can be returned by a one to one hash lookup (negative)
    // or at the end of iteration over a set of matching values.
    llvm::Value* is_valid_it{nullptr};
    if (have_more_inner_rows) {
      is_valid_it = have_more_inner_rows;
    } else {
      is_valid_it = cgen_state_->ir_builder_.CreateICmp(
          llvm::ICmpInst::ICMP_SGE, matching_row_index, ll_int<int64_t>(0));
    }
    const auto it_valid_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "it_valid", cgen_state_->row_func_);
    const auto it_not_valid_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "it_not_valid", cgen_state_->row_func_);
    cgen_state_->ir_builder_.CreateCondBr(is_valid_it, it_valid_bb, it_not_valid_bb);
    const auto row_is_deleted_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "row_is_deleted", cgen_state_->row_func_);
    cgen_state_->ir_builder_.SetInsertPoint(it_valid_bb);
    const auto row_is_deleted = toBool(codegen(deleted_expr.get(), true, co).front());
    cgen_state_->ir_builder_.CreateBr(row_is_deleted_bb);
    cgen_state_->ir_builder_.SetInsertPoint(it_not_valid_bb);
    const auto row_is_deleted_default = ll_bool(false);
    cgen_state_->ir_builder_.CreateBr(row_is_deleted_bb);
    cgen_state_->ir_builder_.SetInsertPoint(row_is_deleted_bb);
    auto row_is_deleted_or_default =
        cgen_state_->ir_builder_.CreatePHI(row_is_deleted->getType(), 2);
    row_is_deleted_or_default->addIncoming(row_is_deleted, it_valid_bb);
    row_is_deleted_or_default->addIncoming(row_is_deleted_default, it_not_valid_bb);
    return row_is_deleted_or_default;
  };
}

std::shared_ptr<JoinHashTableInterface> Executor::buildCurrentLevelHashTable(
    const JoinCondition& current_level_join_conditions,
    RelAlgExecutionUnit& ra_exe_unit,
    const CompilationOptions& co,
    const std::vector<InputTableInfo>& query_infos,
    ColumnCacheMap& column_cache,
    std::vector<std::string>& fail_reasons) {
  if (current_level_join_conditions.type != JoinType::INNER &&
      current_level_join_conditions.quals.size() > 1) {
    fail_reasons.emplace_back("No equijoin expression found for outer join");
    return nullptr;
  }
  std::shared_ptr<JoinHashTableInterface> current_level_hash_table;
  for (const auto& join_qual : current_level_join_conditions.quals) {
    auto qual_bin_oper = std::dynamic_pointer_cast<Analyzer::BinOper>(join_qual);
    if (!qual_bin_oper || !IS_EQUIVALENCE(qual_bin_oper->get_optype())) {
      fail_reasons.emplace_back("No equijoin expression found");
      if (current_level_join_conditions.type == JoinType::INNER) {
        add_qualifier_to_execution_unit(ra_exe_unit, join_qual);
      }
      continue;
    }
    JoinHashTableOrError hash_table_or_error;
    if (!current_level_hash_table) {
      hash_table_or_error = buildHashTableForQualifier(
          qual_bin_oper,
          query_infos,
          ra_exe_unit,
          co.device_type_ == ExecutorDeviceType::GPU ? MemoryLevel::GPU_LEVEL
                                                     : MemoryLevel::CPU_LEVEL,
          column_cache);
      current_level_hash_table = hash_table_or_error.hash_table;
    }
    if (hash_table_or_error.hash_table) {
      plan_state_->join_info_.join_hash_tables_.push_back(hash_table_or_error.hash_table);
      plan_state_->join_info_.equi_join_tautologies_.push_back(qual_bin_oper);
    } else {
      fail_reasons.push_back(hash_table_or_error.fail_reason);
      if (current_level_join_conditions.type == JoinType::INNER) {
        add_qualifier_to_execution_unit(ra_exe_unit, qual_bin_oper);
      }
    }
  }
  return current_level_hash_table;
}

llvm::Value* Executor::addJoinLoopIterator(const std::vector<llvm::Value*>& prev_iters,
                                           const size_t level_idx) {
  // Iterators are added for loop-outer joins when the head of the loop is generated,
  // then once again when the body if generated. Allow this instead of special handling
  // of call sites.
  const auto it = cgen_state_->scan_idx_to_hash_pos_.find(level_idx);
  if (it != cgen_state_->scan_idx_to_hash_pos_.end()) {
    return it->second;
  }
  CHECK(!prev_iters.empty());
  llvm::Value* matching_row_index = prev_iters.back();
  const auto it_ok =
      cgen_state_->scan_idx_to_hash_pos_.emplace(level_idx, matching_row_index);
  CHECK(it_ok.second);
  return matching_row_index;
}

void Executor::codegenJoinLoops(const std::vector<JoinLoop>& join_loops,
                                const RelAlgExecutionUnit& ra_exe_unit,
                                GroupByAndAggregate& group_by_and_aggregate,
                                llvm::Function* query_func,
                                llvm::BasicBlock* entry_bb,
                                const CompilationOptions& co,
                                const ExecutionOptions& eo) {
  const auto exit_bb =
      llvm::BasicBlock::Create(cgen_state_->context_, "exit", cgen_state_->row_func_);
  cgen_state_->ir_builder_.SetInsertPoint(exit_bb);
  cgen_state_->ir_builder_.CreateRet(ll_int<int32_t>(0));
  cgen_state_->ir_builder_.SetInsertPoint(entry_bb);
  const auto loops_entry_bb = JoinLoop::codegen(
      join_loops,
      [this, query_func, &co, &eo, &group_by_and_aggregate, &join_loops, &ra_exe_unit](
          const std::vector<llvm::Value*>& prev_iters) {
        addJoinLoopIterator(prev_iters, join_loops.size());
        auto& builder = cgen_state_->ir_builder_;
        const auto loop_body_bb = llvm::BasicBlock::Create(
            builder.getContext(), "loop_body", builder.GetInsertBlock()->getParent());
        builder.SetInsertPoint(loop_body_bb);
        const bool can_return_error =
            compileBody(ra_exe_unit, group_by_and_aggregate, co);
        if (can_return_error || cgen_state_->needs_error_check_ ||
            eo.with_dynamic_watchdog) {
          createErrorCheckControlFlow(
              query_func, eo.with_dynamic_watchdog, co.device_type_);
        }
        return loop_body_bb;
      },
      posArg(nullptr),
      exit_bb,
      cgen_state_->ir_builder_);
  cgen_state_->ir_builder_.SetInsertPoint(entry_bb);
  cgen_state_->ir_builder_.CreateBr(loops_entry_bb);
}

Executor::GroupColLLVMValue Executor::groupByColumnCodegen(
    Analyzer::Expr* group_by_col,
    const size_t col_width,
    const CompilationOptions& co,
    const bool translate_null_val,
    const int64_t translated_null_val,
    GroupByAndAggregate::DiamondCodegen& diamond_codegen,
    std::stack<llvm::BasicBlock*>& array_loops,
    const bool thread_mem_shared) {
  CHECK_GE(col_width, sizeof(int32_t));
  auto group_key = codegen(group_by_col, true, co).front();
  auto key_to_cache = group_key;
  if (dynamic_cast<Analyzer::UOper*>(group_by_col) &&
      static_cast<Analyzer::UOper*>(group_by_col)->get_optype() == kUNNEST) {
    auto preheader = cgen_state_->ir_builder_.GetInsertBlock();
    auto array_loop_head = llvm::BasicBlock::Create(cgen_state_->context_,
                                                    "array_loop_head",
                                                    cgen_state_->row_func_,
                                                    preheader->getNextNode());
    diamond_codegen.setFalseTarget(array_loop_head);
    const auto ret_ty = get_int_type(32, cgen_state_->context_);
    auto array_idx_ptr = cgen_state_->ir_builder_.CreateAlloca(ret_ty);
    CHECK(array_idx_ptr);
    cgen_state_->ir_builder_.CreateStore(ll_int(int32_t(0)), array_idx_ptr);
    const auto arr_expr = static_cast<Analyzer::UOper*>(group_by_col)->get_operand();
    const auto& array_ti = arr_expr->get_type_info();
    CHECK(array_ti.is_array());
    const auto& elem_ti = array_ti.get_elem_type();
    auto array_len = (array_ti.get_size() > 0)
                         ? ll_int(array_ti.get_size() / elem_ti.get_size())
                         : cgen_state_->emitExternalCall(
                               "array_size",
                               ret_ty,
                               {group_key,
                                posArg(arr_expr),
                                ll_int(log2_bytes(elem_ti.get_logical_size()))});
    cgen_state_->ir_builder_.CreateBr(array_loop_head);
    cgen_state_->ir_builder_.SetInsertPoint(array_loop_head);
    CHECK(array_len);
    auto array_idx = cgen_state_->ir_builder_.CreateLoad(array_idx_ptr);
    auto bound_check = cgen_state_->ir_builder_.CreateICmp(
        llvm::ICmpInst::ICMP_SLT, array_idx, array_len);
    auto array_loop_body = llvm::BasicBlock::Create(
        cgen_state_->context_, "array_loop_body", cgen_state_->row_func_);
    cgen_state_->ir_builder_.CreateCondBr(
        bound_check,
        array_loop_body,
        array_loops.empty() ? diamond_codegen.orig_cond_false_ : array_loops.top());
    cgen_state_->ir_builder_.SetInsertPoint(array_loop_body);
    cgen_state_->ir_builder_.CreateStore(
        cgen_state_->ir_builder_.CreateAdd(array_idx, ll_int(int32_t(1))), array_idx_ptr);
    const auto array_at_fname = "array_at_" + numeric_type_name(elem_ti);
    const auto ar_ret_ty =
        elem_ti.is_fp()
            ? (elem_ti.get_type() == kDOUBLE
                   ? llvm::Type::getDoubleTy(cgen_state_->context_)
                   : llvm::Type::getFloatTy(cgen_state_->context_))
            : get_int_type(elem_ti.get_logical_size() * 8, cgen_state_->context_);
    group_key = cgen_state_->emitExternalCall(
        array_at_fname, ar_ret_ty, {group_key, posArg(arr_expr), array_idx});
    if (need_patch_unnest_double(
            elem_ti, isArchMaxwell(co.device_type_), thread_mem_shared)) {
      key_to_cache = spillDoubleElement(group_key, ar_ret_ty);
    } else {
      key_to_cache = group_key;
    }
    CHECK(array_loop_head);
    array_loops.push(array_loop_head);
  }
  cgen_state_->group_by_expr_cache_.push_back(key_to_cache);
  llvm::Value* orig_group_key{nullptr};
  if (translate_null_val) {
    const std::string translator_func_name(
        col_width == sizeof(int32_t) ? "translate_null_key_i32_" : "translate_null_key_");
    const auto& ti = group_by_col->get_type_info();
    const auto key_type = get_int_type(ti.get_logical_size() * 8, cgen_state_->context_);
    orig_group_key = group_key;
    group_key = cgen_state_->emitCall(translator_func_name + numeric_type_name(ti),
                                      {group_key,
                                       static_cast<llvm::Value*>(llvm::ConstantInt::get(
                                           key_type, inline_int_null_val(ti))),
                                       static_cast<llvm::Value*>(llvm::ConstantInt::get(
                                           key_type, translated_null_val))});
  }
  group_key = cgen_state_->ir_builder_.CreateBitCast(
      castToTypeIn(group_key, col_width * 8),
      get_int_type(col_width * 8, cgen_state_->context_));
  if (orig_group_key) {
    orig_group_key = cgen_state_->ir_builder_.CreateBitCast(
        castToTypeIn(orig_group_key, col_width * 8),
        get_int_type(col_width * 8, cgen_state_->context_));
  }
  return {group_key, orig_group_key};
}
