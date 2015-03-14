#include "GroupByAndAggregate.h"

#include "Execute.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"
#include "../CudaMgr/CudaMgr.h"


QueryExecutionContext::QueryExecutionContext(
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<int64_t>& init_agg_vals,
    const Executor* executor,
    const ExecutorDeviceType device_type)
  : query_mem_desc_(query_mem_desc)
  , executor_(executor)
  , device_type_(device_type)
  , num_buffers_ { device_type == ExecutorDeviceType::CPU
      ? 1
      : executor->block_size_x_ * executor->grid_size_x_ } {
  if (query_mem_desc_.group_col_widths.empty()) {
    return;
  }
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto group_by_buffer = static_cast<int64_t*>(malloc(query_mem_desc_.getBufferSize()));
    init_groups(group_by_buffer, query_mem_desc_.entry_count, query_mem_desc_.group_col_widths.size(),
      &init_agg_vals[0], query_mem_desc_.agg_col_widths.size());
    group_by_buffers_.push_back(group_by_buffer);
    if (query_mem_desc_.getSmallBufferSize()) {
      auto group_by_small_buffer = static_cast<int64_t*>(malloc(query_mem_desc_.getSmallBufferSize()));
      init_groups(group_by_small_buffer, query_mem_desc_.entry_count_small, query_mem_desc_.group_col_widths.size(),
        &init_agg_vals[0], query_mem_desc_.agg_col_widths.size());
      small_group_by_buffers_.push_back(group_by_small_buffer);
    }
  }
}

QueryExecutionContext::~QueryExecutionContext() {
  for (auto group_by_buffer : group_by_buffers_) {
    free(group_by_buffer);
  }
  for (auto small_group_by_buffer : small_group_by_buffers_) {
    free(small_group_by_buffer);
  }
}

Executor::ResultRows QueryExecutionContext::getRowSet(const std::vector<Analyzer::Expr*>& targets) const {
  std::vector<Executor::ResultRows> results_per_sm;
  CHECK_EQ(num_buffers_, group_by_buffers_.size());
  for (size_t i = 0; i < group_by_buffers_.size(); ++i) {
    auto results = groupBufferToResults(i, targets);
    if (device_type_ == ExecutorDeviceType::GPU) {
      results_per_sm.push_back(results);
    } else {
      return results;
    }
  }
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  return executor_->reduceMultiDeviceResults(results_per_sm);
}

Executor::ResultRows QueryExecutionContext::groupBufferToResults(
    const size_t i,
    const std::vector<Analyzer::Expr*>& targets) const {
  const size_t group_by_col_count { query_mem_desc_.group_col_widths.size() };
  const size_t agg_col_count { query_mem_desc_.agg_col_widths.size() };
  auto impl = [group_by_col_count, agg_col_count, this, &targets](
      const size_t groups_buffer_entry_count,
      const int64_t* group_by_buffer) {
    std::vector<ResultRow> results;
    for (size_t i = 0; i < groups_buffer_entry_count; ++i) {
      const size_t key_off = (group_by_col_count + agg_col_count) * i;
      if (group_by_buffer[key_off] != EMPTY_KEY) {
        size_t out_vec_idx = 0;
        ResultRow result_row(executor_);
        for (size_t val_tuple_idx = 0; val_tuple_idx < group_by_col_count; ++val_tuple_idx) {
          const int64_t key_comp = group_by_buffer[key_off + val_tuple_idx];
          CHECK_NE(key_comp, EMPTY_KEY);
          result_row.value_tuple_.push_back(key_comp);
        }
        for (const auto target_expr : targets) {
          const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
          // If the target is not an aggregate, use kMIN since
          // additive would be incorrect for the reduce phase.
          const auto agg_type = agg_expr ? agg_expr->get_aggtype() : kMIN;
          result_row.agg_results_idx_.push_back(result_row.agg_results_.size());
          result_row.agg_kinds_.push_back(agg_type);
          if (agg_type == kAVG) {
            CHECK(agg_expr->get_arg());
            result_row.agg_types_.push_back(agg_expr->get_arg()->get_type_info());
            CHECK(!target_expr->get_type_info().is_string());
            result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count]);
            result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count + 1]);
            out_vec_idx += 2;
          } else {
            result_row.agg_types_.push_back(target_expr->get_type_info());
            result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count]);
            ++out_vec_idx;
          }
        }
        results.push_back(result_row);
      }
    }
    return results;
  };
  std::vector<ResultRow> results;
  if (query_mem_desc_.getSmallBufferSize()) {
    results = impl(query_mem_desc_.entry_count_small, small_group_by_buffers_[i]);
  }
  CHECK_LT(i, group_by_buffers_.size());
  auto more_results = impl(query_mem_desc_.entry_count, group_by_buffers_[i]);
  results.insert(results.end(), more_results.begin(), more_results.end());
  return results;
}

std::vector<int64_t*> QueryExecutionContext::launchGpuCode(
    const std::vector<void*>& cu_functions,
    const bool hoist_literals,
    const std::vector<int8_t>& literal_buff,
    std::vector<const int8_t*> col_buffers,
    const int64_t num_rows,
    const std::vector<int64_t>& init_agg_vals,
    Data_Namespace::DataMgr* data_mgr,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id) const {
  data_mgr->cudaMgr_->setContext(device_id);
  auto cu_func = static_cast<CUfunction>(cu_functions[device_id]);
  std::vector<int64_t*> out_vec;
  CUdeviceptr col_buffers_dev_ptr;
  {
    const size_t col_count { col_buffers.size() };
    std::vector<CUdeviceptr> col_dev_buffers;
    for (auto col_buffer : col_buffers) {
      col_dev_buffers.push_back(reinterpret_cast<CUdeviceptr>(col_buffer));
    }
    if (!col_dev_buffers.empty()) {
      col_buffers_dev_ptr = alloc_gpu_mem(
        data_mgr, col_count * sizeof(CUdeviceptr), device_id);
      copy_to_gpu(data_mgr, col_buffers_dev_ptr, &col_dev_buffers[0],
        col_count * sizeof(CUdeviceptr), device_id);
    }
  }
  CUdeviceptr literals_dev_ptr { 0 };
  if (!literal_buff.empty()) {
    CHECK(hoist_literals);
    literals_dev_ptr = alloc_gpu_mem(data_mgr, literal_buff.size(), device_id);
    copy_to_gpu(data_mgr, literals_dev_ptr, &literal_buff[0], literal_buff.size(), device_id);
  }
  CUdeviceptr num_rows_dev_ptr;
  {
    num_rows_dev_ptr = alloc_gpu_mem(data_mgr, sizeof(int64_t), device_id);
    copy_to_gpu(data_mgr, num_rows_dev_ptr, &num_rows,
      sizeof(int64_t), device_id);
  }
  CUdeviceptr init_agg_vals_dev_ptr;
  {
    init_agg_vals_dev_ptr = alloc_gpu_mem(
      data_mgr, init_agg_vals.size() * sizeof(int64_t), device_id);
    copy_to_gpu(data_mgr, init_agg_vals_dev_ptr, &init_agg_vals[0],
      init_agg_vals.size() * sizeof(int64_t), device_id);
  }
  {
    const unsigned block_size_y = 1;
    const unsigned block_size_z = 1;
    const unsigned grid_size_y  = 1;
    const unsigned grid_size_z  = 1;
    if (query_mem_desc_.getBufferSize() > 0) {  // group by path
      CHECK(!group_by_buffers_.empty());
      CUdeviceptr group_by_dev_ptr;
      std::vector<CUdeviceptr> group_by_dev_buffers;
      auto gpu_query_mem = create_dev_group_by_buffers(
        data_mgr, group_by_buffers_, query_mem_desc_,
        block_size_x, grid_size_x, device_id);
      std::tie(group_by_dev_ptr, group_by_dev_buffers) = gpu_query_mem.group_by_buffers;
      CUdeviceptr small_group_by_dev_ptr;
      std::vector<CUdeviceptr> small_group_by_dev_buffers;
      std::tie(small_group_by_dev_ptr, small_group_by_dev_buffers) = gpu_query_mem.small_group_by_buffers;
      if (hoist_literals) {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &literals_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &group_by_dev_ptr,
          &small_group_by_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       query_mem_desc_.sharedMemBytes(), nullptr, kernel_params, nullptr));
      } else {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &group_by_dev_ptr,
          &small_group_by_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       query_mem_desc_.sharedMemBytes(), nullptr, kernel_params, nullptr));
      }
      copy_group_by_buffers_from_gpu(data_mgr, this, gpu_query_mem, block_size_x, grid_size_x, device_id);
    } else {
      std::vector<CUdeviceptr> out_vec_dev_buffers;
      const size_t agg_col_count { init_agg_vals.size() };
      for (size_t i = 0; i < agg_col_count; ++i) {
        auto out_vec_dev_buffer = alloc_gpu_mem(
          data_mgr, block_size_x * grid_size_x * sizeof(int64_t), device_id);
        out_vec_dev_buffers.push_back(out_vec_dev_buffer);
      }
      auto out_vec_dev_ptr = alloc_gpu_mem(data_mgr, agg_col_count * sizeof(CUdeviceptr), device_id);
      copy_to_gpu(data_mgr, out_vec_dev_ptr, &out_vec_dev_buffers[0],
        agg_col_count * sizeof(CUdeviceptr), device_id);
      CUdeviceptr unused_dev_ptr { 0 };
      if (hoist_literals) {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &literals_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &out_vec_dev_ptr,
          &unused_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       0, nullptr, kernel_params, nullptr));
      } else {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &out_vec_dev_ptr,
          &unused_dev_ptr
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       0, nullptr, kernel_params, nullptr));
      }
      for (size_t i = 0; i < agg_col_count; ++i) {
        int64_t* host_out_vec = new int64_t[block_size_x * grid_size_x * sizeof(int64_t)];
        copy_from_gpu(data_mgr, host_out_vec, out_vec_dev_buffers[i],
          block_size_x * grid_size_x * sizeof(int64_t),
          device_id);
        out_vec.push_back(host_out_vec);
      }
    }
  }
  return out_vec;
}

std::unique_ptr<QueryExecutionContext> QueryMemoryDescriptor::getQueryExecutionContext(
    const std::vector<int64_t>& init_agg_vals,
    const Executor* executor,
    const ExecutorDeviceType device_type) const {
  return std::unique_ptr<QueryExecutionContext>(
    new QueryExecutionContext(*this, init_agg_vals, executor, device_type));
}

size_t QueryMemoryDescriptor::getBufferSize() const {
  return (group_col_widths.size() + agg_col_widths.size()) * entry_count * sizeof(int64_t);
}

size_t QueryMemoryDescriptor::getSmallBufferSize() const {
  return (group_col_widths.size() + agg_col_widths.size()) * entry_count_small * sizeof(int64_t);
}

namespace {

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

}  // namespace

QueryMemoryDescriptor GroupByAndAggregate::getQueryMemoryDescriptor() {
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
      0, 0,
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
        executor_->max_groups_buffer_entry_count_,
        executor_->small_groups_buffer_entry_count_,
        col_range_info.min, GroupByMemSharing::Shared
      };
    } else {
      return {
        col_range_info.hash_type_,
        group_col_widths, agg_col_widths,
        static_cast<size_t>(col_range_info.max - col_range_info.min + 1), 0,
        col_range_info.min, GroupByMemSharing::Shared
      };
    }
  }
  case GroupByColRangeType::MultiCol: {
    return {
      col_range_info.hash_type_,
      group_col_widths, agg_col_widths,
      executor_->max_groups_buffer_entry_count_, 0,
      0, GroupByMemSharing::Shared
    };
  }
  default:
    CHECK(false);
  }
}

bool QueryMemoryDescriptor::usesGetGroupValueFast() const {
  return ((hash_type == GroupByColRangeType::OneColKnownRange ||
           hash_type == GroupByColRangeType::OneColConsecutiveKeys) &&
          !getSmallBufferSize());
}

bool QueryMemoryDescriptor::threadsShareMemory() const {
  return usesGetGroupValueFast();
}

size_t QueryMemoryDescriptor::sharedMemBytes() const {
  const size_t shared_mem_threshold { 0 };
  const size_t shared_mem_bytes { getBufferSize() };
  if (!usesGetGroupValueFast() || shared_mem_bytes > shared_mem_threshold) {
    return 0;
  }
  return shared_mem_bytes;
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

  auto query_mem_desc = getQueryMemoryDescriptor();

  if (group_by_exprs(plan_).empty()) {
    auto arg_it = ROW_FUNC->arg_begin();
    std::vector<llvm::Value*> agg_out_vec;
    for (int32_t i = 0; i < get_agg_count(plan_); ++i) {
      agg_out_vec.push_back(arg_it++);
    }
    codegenAggCalls(nullptr, agg_out_vec, query_mem_desc, device_type, hoist_literals);
  } else {
    auto agg_out_start_ptr = codegenGroupBy(query_mem_desc, hoist_literals);
    codegenAggCalls(agg_out_start_ptr, {}, query_mem_desc, device_type, hoist_literals);
  }

  LL_BUILDER.CreateBr(filter_false);
  LL_BUILDER.SetInsertPoint(filter_false);
  LL_BUILDER.CreateRetVoid();
}

llvm::Value* GroupByAndAggregate::codegenGroupBy(
    const QueryMemoryDescriptor& query_mem_desc,
    const bool hoist_literals) {
  auto arg_it = ROW_FUNC->arg_begin();
  auto groups_buffer = arg_it++;

  llvm::Value* agg_out_start_ptr { nullptr };

  const auto agg_plan = dynamic_cast<const Planner::AggPlan*>(plan_);

  // For non-aggregate (scan only) plans, execute them like a group by
  // row index -- the null pointer means row index to Executor::codegen().
  const auto groupby_list = agg_plan
    ? agg_plan->get_groupby_list()
    : std::list<Analyzer::Expr*> { nullptr };

  switch (query_mem_desc.hash_type) {
  case GroupByColRangeType::OneColKnownRange:
  case GroupByColRangeType::OneColConsecutiveKeys:
  case GroupByColRangeType::OneColGuessedRange:
  case GroupByColRangeType::Scan: {
    CHECK_EQ(1, groupby_list.size());
    const auto group_expr = groupby_list.front();
    const auto group_expr_lv = executor_->groupByColumnCodegen(group_expr, hoist_literals);
    auto small_groups_buffer = arg_it;
    if (query_mem_desc.usesGetGroupValueFast()) {
      agg_out_start_ptr = emitCall(
        "get_group_value_fast",
        {
          groups_buffer,
          group_expr_lv,
          LL_INT(query_mem_desc.min_val),
          LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size()))
        });
    } else {
      agg_out_start_ptr = emitCall(
        "get_group_value_one_key",
        {
          groups_buffer,
          LL_INT(static_cast<int32_t>(query_mem_desc.entry_count)),
          small_groups_buffer,
          LL_INT(static_cast<int32_t>(query_mem_desc.entry_count_small)),
          group_expr_lv,
          LL_INT(query_mem_desc.min_val),
          LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size()))
        });
    }
    break;
  }
  case GroupByColRangeType::MultiCol: {
    auto key_size_lv = LL_INT(static_cast<int32_t>(query_mem_desc.group_col_widths.size()));
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
        LL_INT(static_cast<int32_t>(query_mem_desc.entry_count)),
        group_key,
        key_size_lv,
        LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size()))
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

void GroupByAndAggregate::codegenAggCalls(
    llvm::Value* agg_out_start_ptr,
    const std::vector<llvm::Value*>& agg_out_vec,
    const QueryMemoryDescriptor& query_mem_desc,
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
      auto target_lv = executor_->toDoublePrecision(codegenAggArg(target_expr, hoist_literals));
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
        auto null_lv = executor_->toDoublePrecision(executor_->inlineIntNull(agg_info.sql_type));
        agg_args.push_back(null_lv);
      }
      emitCall(
        (device_type == ExecutorDeviceType::GPU && query_mem_desc.threadsShareMemory())
          ? agg_fname + "_shared"
          : agg_fname,
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
