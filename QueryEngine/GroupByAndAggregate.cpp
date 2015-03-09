#include "GroupByAndAggregate.h"

#include <glog/logging.h>


namespace {

int64_t extract_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  switch (ti.get_type()) {
  case kSMALLINT:
    return datum.smallintval;
  case kINT:
  case kCHAR:
  case kVARCHAR:
  case kTEXT:
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    return datum.intval;
  case kBIGINT:
    return datum.bigintval;
  case kTIME:
  case kTIMESTAMP:
  case kDATE:
    return datum.timeval;
  default:
    CHECK(false);
  }
}

int64_t extract_min_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.min, ti);
}

int64_t extract_max_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.max, ti);
}

std::vector<Analyzer::Expr*> get_agg_target_exprs(const Planner::Plan* plan) {
  const auto& target_list = plan->get_targetlist();
  std::vector<Analyzer::Expr*> result;
  for (auto target : target_list) {
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    result.push_back(target_expr);
  }
  return result;
}

int64_t get_agg_count(const Planner::Plan* plan) {
  int64_t agg_count { 0 };
  const auto target_exprs = get_agg_target_exprs(plan);
  for (auto target_expr : target_exprs) {
    CHECK(target_expr);
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr) {
      ++agg_count;
      continue;
    }
    if (agg_expr && agg_expr->get_aggtype() == kAVG) {
      agg_count += 2;
    } else {
      ++agg_count;
    }
  }
  return agg_count;
}

}  // namespace

#define FIND_STAT_FRAG(stat_name)                                                             \
  const auto stat_name##_frag = std::stat_name##_element(fragments.begin(), fragments.end(),  \
    [group_col_id, group_by_ti](const Fragmenter_Namespace::FragmentInfo& lhs,                \
                                 const Fragmenter_Namespace::FragmentInfo& rhs) {             \
      auto lhs_meta_it = lhs.chunkMetadataMap.find(group_col_id);                             \
      CHECK(lhs_meta_it != lhs.chunkMetadataMap.end());                                       \
      auto rhs_meta_it = rhs.chunkMetadataMap.find(group_col_id);                             \
      CHECK(rhs_meta_it != rhs.chunkMetadataMap.end());                                       \
      return extract_##stat_name##_stat(lhs_meta_it->second.chunkStats, group_by_ti) <        \
             extract_##stat_name##_stat(rhs_meta_it->second.chunkStats, group_by_ti);         \
  });                                                                                         \
  if (stat_name##_frag == fragments.end()) {                                                  \
    return { GroupByAndAggregate::ColRangeType::OneColGuessedRange, 0, guessed_range_max };                                                                 \
  }

GroupByAndAggregate::ColRangeInfo GroupByAndAggregate::getColRangeInfo(
    const Planner::AggPlan* agg_plan,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  const int64_t guessed_range_max { 255 };  // TODO(alex): replace with educated guess
  if (!agg_plan) {
    return { GroupByAndAggregate::ColRangeType::Scan, 0, 0 };
  }
  const auto& groupby_exprs = agg_plan->get_groupby_list();
  if (groupby_exprs.size() != 1) {
    return { GroupByAndAggregate::ColRangeType::MultiCol, 0, 0 };
  }
  const auto group_col_expr = dynamic_cast<Analyzer::ColumnVar*>(groupby_exprs.front());
  if (!group_col_expr) {
    return { GroupByAndAggregate::ColRangeType::OneColGuessedRange, 0, guessed_range_max };
  }
  const int group_col_id = group_col_expr->get_column_id();
  const auto group_by_ti = group_col_expr->get_type_info();
  switch (group_by_ti.get_type()) {
  case kTEXT:
  case kCHAR:
  case kVARCHAR:
    CHECK(group_by_ti.get_compression() != kENCODING_DICT);
  case kSMALLINT:
  case kINT:
  case kBIGINT: {
    FIND_STAT_FRAG(min);
    FIND_STAT_FRAG(max);
    const auto min_it = min_frag->chunkMetadataMap.find(group_col_id);
    CHECK(min_it != min_frag->chunkMetadataMap.end());
    const auto max_it = max_frag->chunkMetadataMap.find(group_col_id);
    CHECK(max_it != max_frag->chunkMetadataMap.end());
    const auto min_val = extract_min_stat(min_it->second.chunkStats, group_by_ti);
    const auto max_val = extract_max_stat(max_it->second.chunkStats, group_by_ti);
    CHECK_GE(max_val, min_val);
    return {
      group_by_ti.is_string()
        ? GroupByAndAggregate::ColRangeType::OneColConsecutiveKeys
        : GroupByAndAggregate::ColRangeType::OneColKnownRange,
      min_val,
      max_val
    };
  }
  default:
    return { GroupByAndAggregate::ColRangeType::Unknown, 0, 0 };
  }
}

#undef FIND_STAT_FRAG

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_INT(v) executor_->ll_int(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

GroupByAndAggregate::GroupByAndAggregate(
    Executor* executor,
    llvm::Value* filter_result,
    const Planner::AggPlan* agg_plan,
    const Fragmenter_Namespace::QueryInfo& query_info)
  : executor_(executor)
  , filter_result_(filter_result)
  , agg_plan_(agg_plan) {
  CHECK(filter_result_);
  CHECK(agg_plan_);
  CHECK(!agg_plan_->get_groupby_list().empty());
  col_range_info_ = getColRangeInfo(agg_plan_, query_info.fragments);
}

void GroupByAndAggregate::codegen(const ExecutorDeviceType device_type,
                              const bool hoist_literals) {
  auto filter_true = llvm::BasicBlock::Create(
    LL_CONTEXT, "filter_true", ROW_FUNC);
  auto filter_false = llvm::BasicBlock::Create(
    LL_CONTEXT, "filter_false", ROW_FUNC);

  LL_BUILDER.CreateCondBr(filter_result_, filter_true, filter_false);
  LL_BUILDER.SetInsertPoint(filter_true);

  switch (col_range_info_.hash_type_) {
  case ColRangeType::OneColKnownRange:
  case ColRangeType::OneColConsecutiveKeys: {
    CHECK_EQ(1, agg_plan_->get_groupby_list().size());
    const auto group_min_val = col_range_info_.min;
    const auto group_expr = agg_plan_->get_groupby_list().front();
    const auto group_expr_lv = executor_->groupByColumnCodegen(group_expr, hoist_literals);
    auto& groups_buffer = ROW_FUNC->getArgumentList().front();
    const auto agg_out_start_ptr = emitCall(
      "get_group_value_fast",
      {
        &groups_buffer,
        group_expr_lv,
        LL_INT(group_min_val),
        LL_INT(get_agg_count(agg_plan_))
      });
    codegenAggCalls(agg_out_start_ptr, device_type, hoist_literals);
    break;
  }
  default:
    CHECK(false);
    break;
  }
  CHECK(false);
}

namespace {

std::string agg_fn_name(const Analyzer::Expr* target_expr) {
  std::string agg_fp_suffix { "_double" };
  const auto target_expr_ti = target_expr->get_type_info();
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
  if (!agg_expr) {
    return "agg_id" + (target_expr_ti.is_fp() ? agg_fp_suffix : "");
  }
  const auto agg_type = agg_expr->get_aggtype();
  const auto agg_arg_ti = agg_expr->get_arg()->get_type_info();
  agg_fp_suffix = (agg_arg_ti.is_fp() ? agg_fp_suffix : "");
  switch (agg_type) {
  case kCOUNT: {
    return "agg_count" + agg_fp_suffix;
  }
  case kMIN: {
    return "agg_min" + agg_fp_suffix;
  }
  case kMAX: {
    return "agg_max" + agg_fp_suffix;
  }
  case kSUM: {
    return "agg_sum" + agg_fp_suffix;
  }
  default:
    CHECK(false);
  }
}

size_t next_agg_out_off(const size_t crt_off, const Analyzer::Expr* target_expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
  if (!agg_expr) {
    return crt_off + 1;
  }
  const auto agg_type = agg_expr->get_aggtype();
  return crt_off + (agg_type == kAVG ? 2 : 1);
}

}  // namespace

void GroupByAndAggregate::codegenAggCalls(llvm::Value* agg_out_start_ptr,
                                      const ExecutorDeviceType device_type,
                                      const bool hoist_literals) {
  const auto& target_list = agg_plan_->get_targetlist();
  size_t agg_out_off { 0 };
  for (auto target : target_list) {
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    emitCall(
      agg_fn_name(target_expr),
      {
        LL_BUILDER.CreateGEP(agg_out_start_ptr, LL_INT(agg_out_off)),
        toDoublePrecision(codegenAggArg(target_expr, hoist_literals))
      });
    agg_out_off = next_agg_out_off(agg_out_off, target_expr);
  }
  CHECK(false);
}

llvm::Value* GroupByAndAggregate::toDoublePrecision(llvm::Value* val) {
  if (val->getType()->isIntegerTy()) {
    auto val_width = static_cast<llvm::IntegerType*>(val->getType())->getBitWidth();
    CHECK_LE(val_width, 64);
    return val_width < 64
      ? LL_BUILDER.CreateCast(llvm::Instruction::CastOps::SExt, val, llvm::Type::getInt64Ty(LL_CONTEXT))
      : val;
  }
  CHECK(val->getType()->isFloatTy() || val->getType()->isDoubleTy());
  return val->getType()->isFloatTy()
    ? LL_BUILDER.CreateFPExt(val, llvm::Type::getDoubleTy(LL_CONTEXT))
    : val;
}

llvm::Value* GroupByAndAggregate::codegenAggArg(const Analyzer::Expr* target_expr,
                                            const bool hoist_literals) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
  return agg_expr
    ? executor_->codegen(agg_expr->get_arg(), hoist_literals)
    : executor_->codegen(target_expr, hoist_literals);
}

llvm::Function* GroupByAndAggregate::getFunction(const std::string& name) const {
  auto f = executor_->cgen_state_->module_->getFunction(name);
  CHECK(f);
  return f;
}

#undef ROW_FUNC
#undef LL_INT
#undef LL_BUILDER
#undef LL_CONTEXT

void GroupByAndAggregate::allocateBuffers(const ExecutorDeviceType) {
  CHECK(false);
}
