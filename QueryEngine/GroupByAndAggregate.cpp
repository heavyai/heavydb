#include "GroupByAndAggregate.h"

#include <glog/logging.h>


namespace {

int64_t extract_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  switch (ti.get_type()) {
  case kSMALLINT:
    return datum.smallintval;
  case kCHAR:
  case kVARCHAR:
  case kTEXT:
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
  case kINT:
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

int32_t get_agg_count(const Planner::Plan* plan) {
  int32_t agg_count { 0 };
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
    return { GroupByColRangeType::OneColGuessedRange, 0, guessed_range_max };                                                                 \
  }

GroupByAndAggregate::ColRangeInfo GroupByAndAggregate::getColRangeInfo(
    const Planner::Plan* plan,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan_);
  const int64_t guessed_range_max { 255 };  // TODO(alex): replace with educated guess
  if (!agg_plan) {
    return { GroupByColRangeType::Scan, 0, guessed_range_max };
  }
  const auto& groupby_exprs = agg_plan->get_groupby_list();
  if (groupby_exprs.size() != 1) {
    return { GroupByColRangeType::MultiCol, 0, 0 };
  }
  const auto group_col_expr = dynamic_cast<Analyzer::ColumnVar*>(groupby_exprs.front());
  if (!group_col_expr) {
    return { GroupByColRangeType::OneColGuessedRange, 0, guessed_range_max };
  }
  const int group_col_id = group_col_expr->get_column_id();
  const auto& group_by_ti = group_col_expr->get_type_info();
  switch (group_by_ti.get_type()) {
  case kTEXT:
  case kCHAR:
  case kVARCHAR:
    CHECK_EQ(kENCODING_DICT, group_by_ti.get_compression());
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
        ? GroupByColRangeType::OneColConsecutiveKeys
        : GroupByColRangeType::OneColKnownRange,
      min_val,
      max_val
    };
  }
  case kFLOAT:
  case kDOUBLE:
    return { GroupByColRangeType::OneColGuessedRange, 0, guessed_range_max };
  default:
    return { GroupByColRangeType::MultiCol, 0, 0 };
  }
}

#undef FIND_STAT_FRAG

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_INT(v) executor_->ll_int(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

GroupByAndAggregate::GroupByAndAggregate(
    Executor* executor,
    const Planner::Plan* plan,
    const Fragmenter_Namespace::QueryInfo& query_info)
  : executor_(executor)
  , plan_(plan)
  , query_info_(query_info) {
  CHECK(plan_);
}

namespace {

std::list<Analyzer::Expr*> group_by_exprs(const Planner::Plan* plan) {
  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan);
  // For non-aggregate (scan only) plans, execute them like a group by
  // row index -- the null pointer means row index to Executor::codegen().
  return agg_plan
    ? agg_plan->get_groupby_list()
    : std::list<Analyzer::Expr*> { nullptr };
}

struct TargetInfo {
  bool is_agg;
  SQLAgg agg_kind;
  SQLTypes sql_type;
  bool skip_null_val;
  bool is_distinct;
};

TargetInfo target_info(const Analyzer::Expr* target_expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
  if (!agg_expr) {
    return { false, kCOUNT, target_expr->get_type_info().get_type(), false, false };
  }
  const auto agg_type = agg_expr->get_aggtype();
  const auto agg_arg = agg_expr->get_arg();
  if (!agg_arg) {
    CHECK_EQ(kCOUNT, agg_type);
    CHECK(!agg_expr->get_is_distinct());
    return { true, kCOUNT, kBIGINT, false, false };
  }
  const auto& agg_arg_ti = agg_arg->get_type_info();
  bool is_distinct { false };
  if (agg_expr->get_aggtype() == kCOUNT) {
    CHECK(agg_expr->get_is_distinct());
    CHECK(!agg_arg_ti.is_fp());
    is_distinct = true;
  }
  // TODO(alex): null support for all types
  bool skip_null = !agg_arg_ti.get_notnull() && (agg_arg_ti.is_integer() || agg_arg_ti.is_time());
  return { true, agg_expr->get_aggtype(), agg_arg_ti.get_type(),
           skip_null, is_distinct };
}

template<class T>
std::vector<int8_t> get_col_byte_widths(const T& col_expr_list) {
  std::vector<int8_t> col_widths;
  for (const auto col_expr : col_expr_list) {
    if (!col_expr) {
      // row index
      col_widths.push_back(sizeof(int64_t));
    } else {
      const auto agg_info = target_info(col_expr);
      const auto col_expr_bitwidth = get_bit_width(agg_info.sql_type);
      CHECK_EQ(0, col_expr_bitwidth % 8);
      col_widths.push_back(col_expr_bitwidth / 8);
      // for average, we'll need to keep the count as well
      if (agg_info.agg_kind == kAVG) {
        CHECK(agg_info.is_agg);
        col_widths.push_back(sizeof(int64_t));
      }
    }
  }
  return col_widths;
}

}  // namespace

GroupByBufferDescriptor GroupByAndAggregate::getGroupByBufferDescriptor() {
  auto group_col_widths = get_col_byte_widths(group_by_exprs(plan_));
  const auto& target_list = plan_->get_targetlist();
  std::vector<Analyzer::Expr*> target_expr_list;
  for (const auto target : target_list) {
    target_expr_list.push_back(target->get_expr());
  }
  auto agg_col_widths = get_col_byte_widths(target_expr_list);

  if (group_col_widths.empty()) {
    return {
      GroupByColRangeType::Scan,
      group_col_widths, agg_col_widths,
      0, false,
      0, false,
      0, GroupByMemSharing::Shared };
  }

  const auto col_range_info = getColRangeInfo(plan_, query_info_.fragments);

  switch (col_range_info.hash_type_) {
  case GroupByColRangeType::OneColKnownRange:
  case GroupByColRangeType::OneColConsecutiveKeys:
  case GroupByColRangeType::OneColGuessedRange:
  case GroupByColRangeType::Scan: {
    if (col_range_info.hash_type_ == GroupByColRangeType::OneColGuessedRange ||
        col_range_info.hash_type_ == GroupByColRangeType::Scan ||
        col_range_info.max - col_range_info.min >=
        static_cast<int64_t>(executor_->max_groups_buffer_entry_count_)) {
      return {
        col_range_info.hash_type_,
        group_col_widths, agg_col_widths,
        executor_->max_groups_buffer_entry_count_, false,
        executor_->small_groups_buffer_entry_count_, false,
        col_range_info.min, GroupByMemSharing::Shared
      };
    } else {
      return {
        col_range_info.hash_type_,
        group_col_widths, agg_col_widths,
        static_cast<size_t>(col_range_info.max - col_range_info.min + 1), false,
        0, false,
        col_range_info.min, GroupByMemSharing::Shared
      };
    }
  }
  case GroupByColRangeType::MultiCol: {
    return {
      col_range_info.hash_type_,
      group_col_widths, agg_col_widths,
      executor_->max_groups_buffer_entry_count_, false,
      0, false,
      0, GroupByMemSharing::Shared
    };
  }
  default:
    CHECK(false);
  }
}

void GroupByAndAggregate::codegen(
    llvm::Value* filter_result,
    const ExecutorDeviceType device_type,
    const bool hoist_literals) {
  CHECK(filter_result);

  auto filter_true = llvm::BasicBlock::Create(
    LL_CONTEXT, "filter_true", ROW_FUNC);
  auto filter_false = llvm::BasicBlock::Create(
    LL_CONTEXT, "filter_false", ROW_FUNC);

  LL_BUILDER.CreateCondBr(filter_result, filter_true, filter_false);
  LL_BUILDER.SetInsertPoint(filter_true);

  const auto groupby_list = group_by_exprs(plan_);

  GroupByBufferDescriptor group_buff_desc;

  if (groupby_list.empty()) {
    auto arg_it = ROW_FUNC->arg_begin();
    std::vector<llvm::Value*> agg_out_vec;
    for (int32_t i = 0; i < get_agg_count(plan_); ++i) {
      agg_out_vec.push_back(arg_it++);
    }
    codegenAggCalls(nullptr, agg_out_vec, device_type, hoist_literals);
  } else {
    auto agg_out_start_ptr = codegenGroupBy(hoist_literals);
    codegenAggCalls(agg_out_start_ptr, {}, device_type, hoist_literals);
  }

  LL_BUILDER.CreateBr(filter_false);
  LL_BUILDER.SetInsertPoint(filter_false);
  LL_BUILDER.CreateRetVoid();
}

llvm::Value* GroupByAndAggregate::codegenGroupBy(const bool hoist_literals) {
  auto arg_it = ROW_FUNC->arg_begin();
  auto groups_buffer = arg_it++;

  llvm::Value* agg_out_start_ptr { nullptr };

  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan_);

  // For non-aggregate (scan only) plans, execute them like a group by
  // row index -- the null pointer means row index to Executor::codegen().
  const auto groupby_list = agg_plan
    ? agg_plan->get_groupby_list()
    : std::list<Analyzer::Expr*> { nullptr };

  auto group_buff_desc = getGroupByBufferDescriptor();

  switch (group_buff_desc.hash_type) {
  case GroupByColRangeType::OneColKnownRange:
  case GroupByColRangeType::OneColConsecutiveKeys:
  case GroupByColRangeType::OneColGuessedRange:
  case GroupByColRangeType::Scan: {
    CHECK_EQ(1, groupby_list.size());
    const auto group_expr = groupby_list.front();
    const auto group_expr_lv = executor_->groupByColumnCodegen(group_expr, hoist_literals);
    auto small_groups_buffer = arg_it;
    if (group_buff_desc.entry_count_small) {
      agg_out_start_ptr = emitCall(
        "get_group_value_one_key",
        {
          groups_buffer,
          LL_INT(static_cast<int32_t>(group_buff_desc.entry_count)),
          small_groups_buffer,
          LL_INT(static_cast<int32_t>(group_buff_desc.entry_count_small)),
          toDoublePrecision(group_expr_lv),
          LL_INT(group_buff_desc.min_val),
          LL_INT(static_cast<int32_t>(group_buff_desc.agg_col_widths.size()))
        });
    } else {
      agg_out_start_ptr = emitCall(
        "get_group_value_fast",
        {
          groups_buffer,
          toDoublePrecision(group_expr_lv),
          LL_INT(group_buff_desc.min_val),
          LL_INT(static_cast<int32_t>(group_buff_desc.agg_col_widths.size()))
        });
    }
    break;
  }
  case GroupByColRangeType::MultiCol: {
    auto key_size_lv = LL_INT(static_cast<int32_t>(group_buff_desc.group_col_widths.size()));
    // create the key buffer
    auto group_key = LL_BUILDER.CreateAlloca(
      llvm::Type::getInt64Ty(LL_CONTEXT),
      key_size_lv);
    int32_t subkey_idx = 0;
    for (const auto group_expr : groupby_list) {
      const auto group_expr_lv = executor_->groupByColumnCodegen(group_expr, hoist_literals);
      // store the sub-key to the buffer
      LL_BUILDER.CreateStore(group_expr_lv, LL_BUILDER.CreateGEP(group_key, LL_INT(subkey_idx++)));
    }
    agg_out_start_ptr = emitCall(
      "get_group_value",
      {
        groups_buffer,
        LL_INT(static_cast<int32_t>(group_buff_desc.entry_count)),
        group_key,
        key_size_lv,
        LL_INT(static_cast<int32_t>(group_buff_desc.agg_col_widths.size()))
      });
    break;
  }
  default:
    CHECK(false);
    break;
  }

  CHECK(agg_out_start_ptr);

  return agg_out_start_ptr;
}

llvm::Value* GroupByAndAggregate::emitCall(const std::string& fname,
                                           const std::vector<llvm::Value*>& args) {
  return LL_BUILDER.CreateCall(getFunction(fname), args);
}

namespace {

std::vector<std::string> agg_fn_base_names(const TargetInfo& target_info) {
  if (!target_info.is_agg) {
    return { "agg_id" };
  }
  switch (target_info.agg_kind) {
  case kAVG:
    return { "agg_sum", "agg_count" };
  case kCOUNT:
    return { target_info.is_distinct ? "agg_count_distinct" : "agg_count" };
  case kMAX:
    return { "agg_max" };
  case kMIN:
    return { "agg_min" };
  case kSUM:
    return { "agg_sum" };
  default:
    CHECK(false);
  }
}

}  // namespace

extern std::set<std::tuple<int64_t, int64_t, int64_t>>* count_distinct_set;

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

void GroupByAndAggregate::codegenAggCalls(
    llvm::Value* agg_out_start_ptr,
    const std::vector<llvm::Value*>& agg_out_vec,
    const ExecutorDeviceType device_type,
    const bool hoist_literals) {
  // TODO(alex): unify the two cases, the output for non-group by queries
  //             should be a contiguous buffer
  const bool is_group_by { agg_out_start_ptr };
  if (is_group_by) {
    CHECK(agg_out_vec.empty());
  } else {
    CHECK(!agg_out_vec.empty());
  }

  const auto& target_list = plan_->get_targetlist();
  int32_t agg_out_off { 0 };
  for (auto target : target_list) {
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    const auto agg_info = target_info(target_expr);
    for (const auto& agg_base_name : agg_fn_base_names(agg_info)) {
      auto target_lv = toDoublePrecision(codegenAggArg(target_expr, hoist_literals));
      std::vector<llvm::Value*> agg_args {
        is_group_by
          ? LL_BUILDER.CreateGEP(agg_out_start_ptr, LL_INT(agg_out_off))
          : agg_out_vec[agg_out_off],
        // TODO(alex): simply use target_lv once we're done with refactoring,
        //             for now just generate the same IR for easy debugging
        (agg_info.is_agg && agg_info.agg_kind == kCOUNT && !agg_info.is_distinct)
          ? LL_INT(0L)
          : target_lv
      };
      if (agg_info.is_distinct) {
        agg_args.push_back(LL_INT(static_cast<int64_t>(agg_out_off)));
        if (is_group_by) {
          auto& groups_buffer = ROW_FUNC->getArgumentList().front();
          agg_args.push_back(&groups_buffer);
        } else {
          agg_args.push_back(llvm::ConstantPointerNull::get(
            llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0)));
        }
        agg_args.push_back(LL_INT(reinterpret_cast<int64_t>(count_distinct_set)));
      }
      std::string agg_fname { agg_base_name };
      if (agg_info.sql_type == kFLOAT || agg_info.sql_type == kDOUBLE) {
        agg_fname += "_double";
      }
      if (agg_info.skip_null_val) {
        agg_fname += "_skip_val";
        auto null_lv = toDoublePrecision(executor_->inlineIntNull(agg_info.sql_type));
        agg_args.push_back(null_lv);
      }
      emitCall(
        device_type == ExecutorDeviceType::GPU && is_group_by ? agg_fname + "_shared" : agg_fname,
        agg_args);
      ++agg_out_off;
    }
  }
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
