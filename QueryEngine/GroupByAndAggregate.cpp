#include "GroupByAndAggregate.h"

#include "Execute.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"
#include "../CudaMgr/CudaMgr.h"

#include <numeric>
#include <thread>


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

  std::vector<int64_t> group_by_buffer_template(query_mem_desc_.getBufferSizeQuad(device_type));
  init_groups(
    &group_by_buffer_template[0],
    query_mem_desc_.entry_count,
    query_mem_desc_.group_col_widths.size(),
    &init_agg_vals[0],
    query_mem_desc_.agg_col_widths.size(),
    query_mem_desc_.keyless_hash,
    query_mem_desc_.interleavedBins(device_type_) ? executor_->warpSize() : 1);

  if (query_mem_desc_.interleavedBins(device_type_)) {
    CHECK(query_mem_desc_.keyless_hash);
  }

  if (query_mem_desc_.keyless_hash) {
    CHECK_EQ(0, query_mem_desc_.getSmallBufferSizeQuad());
  }

  std::vector<int64_t> group_by_small_buffer_template;
  if (query_mem_desc_.getSmallBufferSizeBytes()) {
    group_by_small_buffer_template.resize(query_mem_desc_.getSmallBufferSizeQuad());
    init_groups(
      &group_by_small_buffer_template[0],
      query_mem_desc_.entry_count_small,
      query_mem_desc_.group_col_widths.size(),
      &init_agg_vals[0],
      query_mem_desc_.agg_col_widths.size(),
      false,
      1);
  }

  size_t step { device_type_ == ExecutorDeviceType::GPU && query_mem_desc_.threadsShareMemory()
    ? executor_->block_size_x_ : 1 };

  for (size_t i = 0; i < num_buffers_; i += step) {
    auto group_by_buffer = static_cast<int64_t*>(malloc(query_mem_desc_.getBufferSizeBytes(device_type_)));
    memcpy(group_by_buffer, &group_by_buffer_template[0], query_mem_desc_.getBufferSizeBytes(device_type_));
    group_by_buffers_.push_back(group_by_buffer);
    for (size_t j = 1; j < step; ++j) {
      group_by_buffers_.push_back(nullptr);
    }
    if (query_mem_desc_.getSmallBufferSizeBytes()) {
      auto group_by_small_buffer = static_cast<int64_t*>(malloc(query_mem_desc_.getSmallBufferSizeBytes()));
      memcpy(group_by_small_buffer, &group_by_buffer_template[0], query_mem_desc_.getSmallBufferSizeBytes());
      small_group_by_buffers_.push_back(group_by_small_buffer);
      for (size_t j = 1; j < step; ++j) {
        small_group_by_buffers_.push_back(nullptr);
      }
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
  if (device_type_ == ExecutorDeviceType::CPU) {
    CHECK_EQ(1, num_buffers_);
    return groupBufferToResults(0, targets);
  }
  size_t step { query_mem_desc_.threadsShareMemory() ? executor_->block_size_x_ : 1 };
  for (size_t i = 0; i < group_by_buffers_.size(); i += step) {
    results_per_sm.emplace_back(groupBufferToResults(i, targets));
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
    if (query_mem_desc_.keyless_hash) {
      CHECK_EQ(1, group_by_col_count);
      CHECK_EQ(targets.size(), agg_col_count);
      for (size_t bin = 0; bin < groups_buffer_entry_count; ++bin) {
        const int8_t warp_count = query_mem_desc_.interleavedBins(device_type_) ? executor_->warpSize() : 1;
        std::vector<int64_t> partial_agg_vals(agg_col_count, 0);
        std::vector<int64_t> agg_vals(agg_col_count, 0);
        ResultRow result_row(executor_);
        result_row.value_tuple_.push_back(bin + query_mem_desc_.min_val);
        bool discard_row = true;
        size_t group_by_buffer_base_idx { warp_count * bin * agg_col_count };
        for (int8_t warp_idx = 0; warp_idx < warp_count; ++warp_idx) {
          bool discard_partial_result = true;
          for (size_t target_idx = 0; target_idx < agg_col_count; ++target_idx) {
            const auto agg_info = target_info(targets[target_idx]);
            CHECK(!agg_info.is_agg || (agg_info.is_agg && agg_info.agg_kind == kCOUNT));
            partial_agg_vals[target_idx] = group_by_buffer[group_by_buffer_base_idx + target_idx];
            if (agg_info.is_agg && partial_agg_vals[target_idx]) {
              discard_partial_result = false;
              break;
            }
          }
          group_by_buffer_base_idx += agg_col_count;
          if (discard_partial_result) {
            continue;
          }
          discard_row = false;
          for (size_t target_idx = 0; target_idx < agg_col_count; ++target_idx) {
            const auto target_expr = targets[target_idx];
            const auto agg_info = target_info(target_expr);
            auto partial_bin_val = partial_agg_vals[target_idx];
            if (agg_info.is_agg) {
              CHECK_EQ(kCOUNT, agg_info.agg_kind);
              agg_vals[target_idx] += partial_bin_val;
            } else {
              if (agg_vals[target_idx]) {
                CHECK_EQ(agg_vals[target_idx], partial_bin_val);
              } else {
                agg_vals[target_idx] = partial_bin_val;
              }
            }
          }
        }
        if (discard_row) {
          continue;
        }
        for (size_t target_idx = 0; target_idx < agg_col_count; ++target_idx) {
          result_row.agg_results_idx_.push_back(target_idx);
          CHECK_EQ(target_idx, result_row.agg_results_.size());
          result_row.agg_results_.push_back(agg_vals[target_idx]);
          const auto target_expr = targets[target_idx];
          const auto agg_info = target_info(target_expr);
          result_row.agg_kinds_.push_back(agg_info.agg_kind);
          result_row.agg_types_.push_back(target_expr->get_type_info());
        }
        results.push_back(result_row);
      }
      return results;
    }
    for (size_t bin = 0; bin < groups_buffer_entry_count; ++bin) {
      const size_t key_off = (group_by_col_count + agg_col_count) * bin;
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
          bool is_real_string = (target_expr && target_expr->get_type_info().is_string() &&
            target_expr->get_type_info().get_compression() == kENCODING_NONE);
          if (agg_type == kAVG) {
            CHECK(agg_expr->get_arg());
            result_row.agg_types_.push_back(agg_expr->get_arg()->get_type_info());
            CHECK(!target_expr->get_type_info().is_string());
            result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count]);
            result_row.agg_results_.push_back(group_by_buffer[key_off + out_vec_idx + group_by_col_count + 1]);
            out_vec_idx += 2;
          } else if (is_real_string) {
            result_row.agg_types_.push_back(target_expr->get_type_info());
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
  if (query_mem_desc_.getSmallBufferSizeBytes()) {
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
    if (query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU) > 0) {  // group by path
      CHECK(!group_by_buffers_.empty());
      auto gpu_query_mem = create_dev_group_by_buffers(
        data_mgr, group_by_buffers_, query_mem_desc_,
        block_size_x, grid_size_x, device_id);
      if (hoist_literals) {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &literals_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &gpu_query_mem.group_by_buffers.first,
          &gpu_query_mem.small_group_by_buffers.first
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                                       nullptr, kernel_params, nullptr));
      } else {
        void* kernel_params[] = {
          &col_buffers_dev_ptr,
          &num_rows_dev_ptr,
          &init_agg_vals_dev_ptr,
          &gpu_query_mem.group_by_buffers.first,
          &gpu_query_mem.small_group_by_buffers.first
        };
        checkCudaErrors(cuLaunchKernel(cu_func, grid_size_x, grid_size_y, grid_size_z,
                                       block_size_x, block_size_y, block_size_z,
                                       query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                                       nullptr, kernel_params, nullptr));
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

size_t QueryMemoryDescriptor::getBufferSizeQuad(const ExecutorDeviceType device_type) const {
  if (keyless_hash) {
    CHECK_EQ(1, group_col_widths.size());
    return (interleavedBins(device_type)
      ? executor_->warpSize() * agg_col_widths.size()
      : agg_col_widths.size()) * entry_count;
  }
  return (group_col_widths.size() + agg_col_widths.size()) * entry_count;
}

size_t QueryMemoryDescriptor::getSmallBufferSizeQuad() const {
  CHECK(!keyless_hash || entry_count_small == 0);
  return (group_col_widths.size() + agg_col_widths.size()) * entry_count_small;
}

size_t QueryMemoryDescriptor::getBufferSizeBytes(const ExecutorDeviceType device_type) const {
  return getBufferSizeQuad(device_type) * sizeof(int64_t);
}

size_t QueryMemoryDescriptor::getSmallBufferSizeBytes() const {
  return getSmallBufferSizeQuad() * sizeof(int64_t);
}

namespace {

int32_t get_agg_count(const Planner::Plan* plan) {
  int32_t agg_count { 0 };
  const auto target_exprs = get_agg_target_exprs(plan);
  for (auto target_expr : target_exprs) {
    CHECK(target_expr);
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr) {
      const auto& ti = target_expr->get_type_info();
      if (ti.is_string() && ti.get_compression() != kENCODING_DICT) {
        agg_count += 2;
      } else {
        ++agg_count;
      }
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
      GroupByColRangeType::OneColKnownRange,
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
      executor_,
      GroupByColRangeType::Scan, false, false,
      group_col_widths, agg_col_widths,
      0, 0,
      0, GroupByMemSharing::Private };
  }

  const auto col_range_info = getColRangeInfo(query_info_.fragments);

  switch (col_range_info.hash_type_) {
  case GroupByColRangeType::OneColKnownRange:
  case GroupByColRangeType::OneColGuessedRange:
  case GroupByColRangeType::Scan: {
    if (col_range_info.hash_type_ == GroupByColRangeType::OneColGuessedRange ||
        col_range_info.hash_type_ == GroupByColRangeType::Scan ||
        col_range_info.max - col_range_info.min >=
        static_cast<int64_t>(executor_->max_groups_buffer_entry_count_)) {
      return {
        executor_,
        col_range_info.hash_type_, false, false,
        group_col_widths, agg_col_widths,
        executor_->max_groups_buffer_entry_count_,
        executor_->small_groups_buffer_entry_count_,
        col_range_info.min, GroupByMemSharing::Shared
      };
    } else {
      bool keyless = true;
      if (keyless) {
        for (const auto target_expr : target_expr_list) {
          auto agg_info = target_info(target_expr);
          if (agg_info.is_agg && agg_info.agg_kind != kCOUNT) {
            keyless = false;
            break;
          }
          if (!agg_info.is_agg && !dynamic_cast<Analyzer::ColumnVar*>(target_expr)) {
            keyless = false;
            break;
          }
        }
      }
      const size_t bin_count = col_range_info.max - col_range_info.min + 1;
      const size_t interleaved_max_threshold { 20 };
      bool interleaved_bins = keyless &&
        bin_count <= interleaved_max_threshold;
      return {
        executor_,
        col_range_info.hash_type_, keyless, interleaved_bins,
        group_col_widths, agg_col_widths,
        bin_count, 0,
        col_range_info.min, GroupByMemSharing::Shared
      };
    }
  }
  case GroupByColRangeType::MultiCol: {
    return {
      executor_,
      col_range_info.hash_type_, false, false,
      group_col_widths, agg_col_widths,
      executor_->max_groups_buffer_entry_count_, 0,
      0, GroupByMemSharing::Private
    };
  }
  default:
    CHECK(false);
  }
}

bool QueryMemoryDescriptor::usesGetGroupValueFast() const {
  return (hash_type == GroupByColRangeType::OneColKnownRange && !getSmallBufferSizeBytes());
}

bool QueryMemoryDescriptor::threadsShareMemory() const {
  return sharing == GroupByMemSharing::Shared;
}

bool QueryMemoryDescriptor::interleavedBins(const ExecutorDeviceType device_type) const {
  return interleaved_bins_on_gpu && device_type == ExecutorDeviceType::GPU;
}

size_t QueryMemoryDescriptor::sharedMemBytes(const ExecutorDeviceType device_type) const {
  CHECK(device_type == ExecutorDeviceType::CPU || device_type == ExecutorDeviceType::GPU);
  if (device_type == ExecutorDeviceType::CPU) {
    return 0;
  }
  const size_t shared_mem_threshold { 0 };
  const size_t shared_mem_bytes { getBufferSizeBytes(ExecutorDeviceType::GPU) };
  if (!usesGetGroupValueFast() || shared_mem_bytes > shared_mem_threshold) {
    return 0;
  }
  return shared_mem_bytes;
}

GroupByAndAggregate::DiamondCodegen::DiamondCodegen(llvm::Value* cond, const Executor* executor) : executor_(executor) {
  cond_true_ = llvm::BasicBlock::Create(
    LL_CONTEXT, "cond_true", ROW_FUNC);
  cond_false_ = llvm::BasicBlock::Create(
    LL_CONTEXT, "cond_false", ROW_FUNC);

  LL_BUILDER.CreateCondBr(cond, cond_true_, cond_false_);
  LL_BUILDER.SetInsertPoint(cond_true_);
}

GroupByAndAggregate::DiamondCodegen::~DiamondCodegen() {
  LL_BUILDER.CreateBr(cond_false_);
  LL_BUILDER.SetInsertPoint(cond_false_);
}

void GroupByAndAggregate::codegen(
    llvm::Value* filter_result,
    const ExecutorDeviceType device_type,
    const bool hoist_literals) {
  CHECK(filter_result);

  {
    DiamondCodegen diamond_codegen(filter_result, executor_);

    auto query_mem_desc = getQueryMemoryDescriptor();

    if (group_by_exprs(plan_).empty()) {
      auto arg_it = ROW_FUNC->arg_begin();
      std::vector<llvm::Value*> agg_out_vec;
      for (int32_t i = 0; i < get_agg_count(plan_); ++i) {
        agg_out_vec.push_back(arg_it++);
      }
      codegenAggCalls(nullptr, agg_out_vec, query_mem_desc, device_type, hoist_literals);
    } else {
      auto agg_out_start_ptr = codegenGroupBy(query_mem_desc, device_type, hoist_literals);
      codegenAggCalls(agg_out_start_ptr, {}, query_mem_desc, device_type, hoist_literals);
    }
  }

  LL_BUILDER.CreateRetVoid();
}

llvm::Value* GroupByAndAggregate::codegenGroupBy(
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
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
  case GroupByColRangeType::OneColGuessedRange:
  case GroupByColRangeType::Scan: {
    CHECK_EQ(1, groupby_list.size());
    const auto group_expr = groupby_list.front();
    const auto group_expr_lv = executor_->groupByColumnCodegen(group_expr, hoist_literals);
    auto small_groups_buffer = arg_it;
    if (query_mem_desc.usesGetGroupValueFast()) {
      std::string get_group_fn_name { "get_group_value_fast" };
      if (query_mem_desc.keyless_hash) {
        get_group_fn_name += "_keyless";
      }
      if (query_mem_desc.interleavedBins(device_type)) {
        CHECK(query_mem_desc.keyless_hash);
        get_group_fn_name += "_semiprivate";
      }
      std::vector<llvm::Value*> get_group_fn_args {
        groups_buffer,
        group_expr_lv,
        LL_INT(query_mem_desc.min_val)
      };
      if (!query_mem_desc.keyless_hash) {
        get_group_fn_args.push_back(LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size())));
      } else {
        get_group_fn_args.push_back(LL_INT(static_cast<int32_t>(query_mem_desc.agg_col_widths.size())));
        if (query_mem_desc.interleavedBins(device_type)) {
          auto warp_idx = emitCall("thread_warp_idx", { LL_INT(executor_->warpSize()) });
          get_group_fn_args.push_back(warp_idx);
          get_group_fn_args.push_back(LL_INT(executor_->warpSize()));
        }
      }
      agg_out_start_ptr = emitCall(get_group_fn_name, get_group_fn_args);
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
    if (target_info.sql_type.is_string() && target_info.sql_type.get_compression() == kENCODING_NONE) {
      return { "agg_id", "agg_id" };
    }
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
    const auto agg_fn_names = agg_fn_base_names(agg_info);
    auto target_lvs = codegenAggArg(target_expr, hoist_literals);
    if (target_lvs.size() < agg_fn_names.size()) {
      CHECK_EQ(1, target_lvs.size());
      CHECK_EQ(2, agg_fn_names.size());
      for (size_t i = 1; i < agg_fn_names.size(); ++i) {
        target_lvs.push_back(target_lvs.front());
      }
    } else {
      CHECK_EQ(agg_fn_names.size(), target_lvs.size());
      CHECK(target_lvs.size() == 1 || target_lvs.size() == 2);
    }
    auto agg_out_lv = is_group_by
      ? LL_BUILDER.CreateGEP(agg_out_start_ptr, LL_INT(agg_out_off))
      : agg_out_vec[agg_out_off];
    const bool is_simple_count = agg_info.is_agg && agg_info.agg_kind == kCOUNT && !agg_info.is_distinct;
    if (device_type == ExecutorDeviceType::GPU && query_mem_desc.threadsShareMemory() && is_simple_count) {
      CHECK_EQ(1, agg_fn_names.size());
      // TODO(alex): use 32-bit wherever possible, avoid casts
      auto acc_i32 = LL_BUILDER.CreateCast(
        llvm::Instruction::CastOps::BitCast,
        agg_out_lv,
        llvm::PointerType::get(get_int_type(32, LL_CONTEXT), 0));
      LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add, acc_i32, LL_INT(1),
        llvm::AtomicOrdering::Monotonic);
      continue;
    }
    size_t target_lv_idx = 0;
    for (const auto& agg_base_name : agg_fn_names) {
      auto target_lv = executor_->toDoublePrecision(target_lvs[target_lv_idx]);
      std::vector<llvm::Value*> agg_args {
        is_group_by
          ? LL_BUILDER.CreateGEP(agg_out_start_ptr, LL_INT(agg_out_off))
          : agg_out_vec[agg_out_off],
        // TODO(alex): simply use target_lv once we're done with refactoring,
        //             for now just generate the same IR for easy debugging
        is_simple_count ? LL_INT(0L) : target_lv
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
      if (agg_info.sql_type.is_fp()) {
        agg_fname += "_double";
      }
      if (agg_info.skip_null_val) {
        agg_fname += "_skip_val";
        auto null_lv = executor_->toDoublePrecision(executor_->inlineIntNull(
          agg_info.sql_type.get_type()));
        agg_args.push_back(null_lv);
      }
      emitCall(
        (device_type == ExecutorDeviceType::GPU && query_mem_desc.threadsShareMemory())
          ? agg_fname + "_shared"
          : agg_fname,
        agg_args);
      ++agg_out_off;
      ++target_lv_idx;
    }
  }
}

std::vector<llvm::Value*> GroupByAndAggregate::codegenAggArg(
    const Analyzer::Expr* target_expr,
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
