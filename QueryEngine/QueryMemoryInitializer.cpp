/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryMemoryInitializer.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Execute.h"
#include "GpuInitGroups.h"
#include "Logger/Logger.h"
#include "OutputBufferInitialization.h"
#include "QueryEngine/QueryEngine.h"
#include "ResultSetBufferAccessors.h"
#include "Shared/checked_alloc.h"
#include "StreamingTopN.h"
#include "Utils/FlatBuffer.h"

// 8 GB, the limit of perfect hash group by under normal conditions
int64_t g_bitmap_memory_limit{8LL * 1000 * 1000 * 1000};

namespace {

struct AddNbytes {
  size_t const entry_count;
  size_t operator()(size_t const sum, ApproxQuantileDescriptor const aqd) const {
    return sum +
           entry_count * quantile::TDigest::nbytes(aqd.buffer_size, aqd.centroids_size);
  }
};

size_t available_buffer_memory_bytes(const Buffer_Namespace::MemoryInfo& memory_info) {
  return Buffer_Namespace::get_reclaimable_size_bytes(memory_info);
}

size_t min_available_buffer_memory_bytes(const Executor* executor,
                                         const ExecutorDeviceType device_type) {
  CHECK(executor);
  const auto memory_level = device_type == ExecutorDeviceType::GPU
                                ? Data_Namespace::MemoryLevel::GPU_LEVEL
                                : Data_Namespace::MemoryLevel::CPU_LEVEL;
  const auto memory_info = executor->getDataMgr()->getMemoryInfo(memory_level);
  if (memory_info.empty()) {
    return std::numeric_limits<size_t>::max();
  }

  size_t min_available_bytes = std::numeric_limits<size_t>::max();
  for (const auto& info : memory_info) {
    min_available_bytes =
        std::min(min_available_bytes, available_buffer_memory_bytes(info));
  }
  return min_available_bytes;
}

size_t count_distinct_bitmap_memory_budget_bytes(const Executor* executor,
                                                 const ExecutorDeviceType device_type) {
  if (!g_enable_result_reduction_pipeline || device_type != ExecutorDeviceType::GPU) {
    return static_cast<size_t>(g_bitmap_memory_limit);
  }
  constexpr size_t kCountDistinctMemoryHeadroomDivisor{5};
  const auto available_bytes = min_available_buffer_memory_bytes(executor, device_type);
  return available_bytes - (available_bytes / kCountDistinctMemoryHeadroomDivisor);
}

size_t count_distinct_bitmap_memory_bytes(const QueryMemoryDescriptor& query_mem_desc) {
  size_t bytes_per_entry{0};
  const auto descriptor_count = query_mem_desc.getCountDistinctDescriptorsSize();
  for (size_t descriptor_idx = 0; descriptor_idx < descriptor_count; ++descriptor_idx) {
    const auto descriptor = query_mem_desc.getCountDistinctDescriptor(descriptor_idx);
    if (descriptor.impl_type_ != CountDistinctImplType::Bitmap) {
      continue;
    }
    const auto bitmap_bytes = descriptor.bitmapPaddedSizeBytes();
    if (__builtin_add_overflow(bytes_per_entry, bitmap_bytes, &bytes_per_entry)) {
      throw OutOfHostMemory(std::numeric_limits<size_t>::max());
    }
  }

  size_t total_bytes{0};
  if (__builtin_mul_overflow(
          bytes_per_entry, query_mem_desc.getEntryCount(), &total_bytes)) {
    throw OutOfHostMemory(std::numeric_limits<size_t>::max());
  }
  return total_bytes;
}

inline void check_total_bitmap_memory(const QueryMemoryDescriptor& query_mem_desc,
                                      const ExecutorDeviceType device_type,
                                      const Executor* executor) {
  const auto total_bytes = count_distinct_bitmap_memory_bytes(query_mem_desc);
  const auto bitmap_memory_budget =
      count_distinct_bitmap_memory_budget_bytes(executor, device_type);
  if (total_bytes >= bitmap_memory_budget) {
    throw OutOfHostMemory(total_bytes);
  }
}

std::pair<int64_t*, bool> alloc_group_by_buffer(
    const size_t numBytes,
    RenderAllocatorMap* render_allocator_map,
    const size_t thread_idx,
    RowSetMemoryOwner* mem_owner,
    const bool reuse_existing_buffer_for_thread) {
  if (render_allocator_map) {
    // NOTE(adb): If we got here, we are performing an in-situ rendering query and are not
    // using CUDA buffers. Therefore we need to allocate result set storage using CPU
    // memory.
    const auto gpu_idx = 0;  // Only 1 GPU supported in CUDA-disabled rendering mode
    auto render_allocator_ptr = render_allocator_map->getRenderAllocator(gpu_idx);
    return std::make_pair(
        reinterpret_cast<int64_t*>(render_allocator_ptr->alloc(numBytes)), false);
  } else if (reuse_existing_buffer_for_thread) {
    return mem_owner->allocateCachedGroupByBuffer(numBytes, thread_idx);
  }
  return std::make_pair(
      reinterpret_cast<int64_t*>(mem_owner->allocate(numBytes, thread_idx)), false);
}

bool baseline_gpu_reduction_target_supported(const TargetInfo& target_info) {
  if (target_info.is_distinct || target_info.sql_type.is_array() ||
      target_info.sql_type.is_geometry() || target_info.sql_type.is_varlen()) {
    return false;
  }
  if (!target_info.is_agg) {
    return false;
  }
  switch (target_info.agg_kind) {
    case kCOUNT:
    case kCOUNT_IF:
    case kSUM:
    case kSUM_IF:
    case kAVG:
      return true;
    case kMIN:
    case kMAX:
      return !target_info.sql_type.is_fp() && !takes_float_argument(target_info);
    default:
      return false;
  }
}

std::optional<size_t> target_init_val_index_for_slot(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t slot_idx) {
  if (slot_idx >= query_mem_desc.getSlotCount() ||
      query_mem_desc.getPaddedSlotWidthBytes(slot_idx) <= 0) {
    return std::nullopt;
  }
  size_t init_val_idx = 0;
  for (size_t previous_slot_idx = 0; previous_slot_idx < slot_idx; ++previous_slot_idx) {
    if (query_mem_desc.getPaddedSlotWidthBytes(previous_slot_idx) > 0) {
      ++init_val_idx;
    }
  }
  return init_val_idx;
}

bool baseline_gpu_reduction_targets_supported(const RelAlgExecutionUnit& ra_exe_unit,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              const std::vector<int64_t>& init_vals) {
  const auto targets = target_exprs_to_infos(ra_exe_unit.target_exprs, query_mem_desc);
  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
      if (target_idx >= query_mem_desc.targetGroupbyIndicesSize()) {
        return false;
      }
      if (query_mem_desc.getTargetGroupbyIndex(target_idx) >= 0) {
        continue;
      }
    }
    const auto& target_info = targets[target_idx];
    if (!baseline_gpu_reduction_target_supported(target_info)) {
      return false;
    }
    const auto& col_slots = query_mem_desc.getColSlotContext().getSlotsForCol(target_idx);
    const auto expected_slot_count = target_info.agg_kind == kAVG ? size_t(2) : size_t(1);
    if (col_slots.size() != expected_slot_count) {
      return false;
    }
    for (size_t target_slot_idx = 0; target_slot_idx < col_slots.size();
         ++target_slot_idx) {
      const auto slot_idx = col_slots[target_slot_idx];
      if (query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
        return false;
      }
      const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
      if (slot_width != sizeof(int32_t) && slot_width != sizeof(int64_t)) {
        return false;
      }
      const auto payload_width = get_rowwise_agg_payload_width(
          target_info, static_cast<size_t>(slot_width), target_slot_idx);
      if (payload_width != sizeof(int32_t) && payload_width != sizeof(int64_t)) {
        return false;
      }
      const auto init_val_idx = target_init_val_index_for_slot(query_mem_desc, slot_idx);
      if (!init_val_idx || *init_val_idx >= init_vals.size()) {
        return false;
      }
    }
  }
  return true;
}

bool all_targets_are_group_keys(const QueryMemoryDescriptor& query_mem_desc) {
  const auto target_count = query_mem_desc.targetGroupbyIndicesSize();
  if (target_count == 0) {
    return false;
  }
  for (size_t target_idx = 0; target_idx < target_count; ++target_idx) {
    if (query_mem_desc.getTargetGroupbyIndex(target_idx) < 0) {
      return false;
    }
  }
  return true;
}

bool can_retain_perfect_hash_rowwise_for_gpu_reduction(
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<TargetInfo>& targets,
    const std::vector<int64_t>& init_vals,
    const size_t entry_count) {
  if (query_mem_desc.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByPerfectHash ||
      query_mem_desc.didOutputColumnar() || query_mem_desc.hasVarlenOutput() ||
      query_mem_desc.getNumModeTargets() > 0 ||
      (!query_mem_desc.countDistinctDescriptorsLogicallyEmpty() &&
       !all_targets_are_group_keys(query_mem_desc)) ||
      (query_mem_desc.getEffectiveKeyWidth() != size_t(4) &&
       query_mem_desc.getEffectiveKeyWidth() != size_t(8)) ||
      query_mem_desc.getRowSize() == 0 ||
      query_mem_desc.getRowSize() % sizeof(int64_t) != 0 ||
      entry_count > std::numeric_limits<size_t>::max() / query_mem_desc.getRowSize() ||
      entry_count * query_mem_desc.getRowSize() < kMinGpuPerfectHashReductionInputBytes) {
    return false;
  }

  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
      if (target_idx >= query_mem_desc.targetGroupbyIndicesSize()) {
        return false;
      }
      if (query_mem_desc.getTargetGroupbyIndex(target_idx) >= 0) {
        continue;
      }
    }
    const auto& target_info = targets[target_idx];
    // Non-aggregate targets of a perfect-hash group-by are group-key projections.
    // Equal bins have equal values, so merging the physical keys needs no payload op.
    if (!target_info.is_agg) {
      continue;
    }
    if (!baseline_gpu_reduction_target_supported(target_info)) {
      return false;
    }
    const auto& col_slots = query_mem_desc.getColSlotContext().getSlotsForCol(target_idx);
    const auto expected_slot_count = target_info.agg_kind == kAVG ? size_t(2) : size_t(1);
    if (col_slots.size() != expected_slot_count) {
      return false;
    }
    for (size_t target_slot_idx = 0; target_slot_idx < col_slots.size();
         ++target_slot_idx) {
      const auto slot_idx = col_slots[target_slot_idx];
      if (query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
        return false;
      }
      const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
      if (slot_width != sizeof(int32_t) && slot_width != sizeof(int64_t)) {
        return false;
      }
      const auto payload_width = get_rowwise_agg_payload_width(
          target_info, static_cast<size_t>(slot_width), target_slot_idx);
      if (payload_width != sizeof(int32_t) && payload_width != sizeof(int64_t)) {
        return false;
      }
      const auto init_val_idx = target_init_val_index_for_slot(query_mem_desc, slot_idx);
      if (!init_val_idx || *init_val_idx >= init_vals.size() ||
          query_mem_desc.getColOffInBytes(slot_idx) >
              std::numeric_limits<uint32_t>::max()) {
        return false;
      }
    }
  }
  return true;
}

bool can_defer_gpu_group_by_host_storage(const RelAlgExecutionUnit& ra_exe_unit,
                                         const QueryMemoryDescriptor& query_mem_desc,
                                         const ExecutorDeviceType device_type,
                                         const bool output_columnar,
                                         const std::vector<int64_t>& init_vals) {
  if (!g_enable_result_reduction_pipeline) {
    return false;
  }
  const bool has_device_resident_consumer =
      !ra_exe_unit.deferred_sparse_baseline_preserved_keys.empty() ||
      ra_exe_unit.defer_gpu_baseline_hash_host_storage_before_copy ||
      ra_exe_unit.sort_info.limit.has_value();
  const bool is_gpu = device_type == ExecutorDeviceType::GPU;
  const bool is_group_by_baseline_hash = query_mem_desc.getQueryDescriptionType() ==
                                         QueryDescriptionType::GroupByBaselineHash;
  const bool is_group_by_perfect_hash = query_mem_desc.getQueryDescriptionType() ==
                                        QueryDescriptionType::GroupByPerfectHash;
  const bool targets_supported =
      baseline_gpu_reduction_targets_supported(ra_exe_unit, query_mem_desc, init_vals);
  const bool count_distinct_descriptors_safe =
      query_mem_desc.countDistinctDescriptorsLogicallyEmpty() ||
      all_targets_are_group_keys(query_mem_desc);
  const bool can_defer =
      is_gpu && !output_columnar && has_device_resident_consumer &&
      is_group_by_baseline_hash && !query_mem_desc.didOutputColumnar() &&
      !query_mem_desc.hasKeylessHash() && !query_mem_desc.hasVarlenOutput() &&
      count_distinct_descriptors_safe && query_mem_desc.getNumModeTargets() == 0 &&
      targets_supported;
  const bool can_defer_keyless_perfect_for_consumer =
      is_gpu && !output_columnar && has_device_resident_consumer &&
      is_group_by_perfect_hash && query_mem_desc.hasKeylessHash() &&
      !query_mem_desc.didOutputColumnar() && !query_mem_desc.hasVarlenOutput() &&
      count_distinct_descriptors_safe && query_mem_desc.getNumModeTargets() == 0 &&
      targets_supported;
  // Multi-device reduction is itself a device-resident consumer. A CPU reduction
  // fallback or final host boundary materializes these retained rows through ResultSet.
  const bool can_defer_keyless_perfect_for_reduction =
      is_gpu && !output_columnar && is_group_by_perfect_hash &&
      query_mem_desc.hasKeylessHash() && query_mem_desc.blocksShareMemory() &&
      can_retain_perfect_hash_rowwise_for_gpu_reduction(
          query_mem_desc,
          target_exprs_to_infos(ra_exe_unit.target_exprs, query_mem_desc),
          init_vals,
          query_mem_desc.getEntryCount());
  const bool can_defer_keyed_perfect =
      is_gpu && !output_columnar && has_device_resident_consumer &&
      is_group_by_perfect_hash && !query_mem_desc.hasKeylessHash() &&
      query_mem_desc.blocksShareMemory() &&
      can_retain_perfect_hash_rowwise_for_gpu_reduction(
          query_mem_desc,
          target_exprs_to_infos(ra_exe_unit.target_exprs, query_mem_desc),
          init_vals,
          query_mem_desc.getEntryCount());
  return can_defer || can_defer_keyless_perfect_for_consumer ||
         can_defer_keyless_perfect_for_reduction || can_defer_keyed_perfect;
}

inline int64_t get_consistent_frag_size(const std::vector<uint64_t>& frag_offsets) {
  if (frag_offsets.size() < 2) {
    return int64_t(-1);
  }
  const auto frag_size = frag_offsets[1] - frag_offsets[0];
  for (size_t i = 2; i < frag_offsets.size(); ++i) {
    const auto curr_size = frag_offsets[i] - frag_offsets[i - 1];
    if (curr_size != frag_size) {
      return int64_t(-1);
    }
  }
  return !frag_size ? std::numeric_limits<int64_t>::max()
                    : static_cast<int64_t>(frag_size);
}

inline std::vector<int64_t> get_consistent_frags_sizes(
    const std::vector<std::vector<uint64_t>>& frag_offsets) {
  if (frag_offsets.empty()) {
    return {};
  }
  std::vector<int64_t> frag_sizes;
  for (size_t tab_idx = 0; tab_idx < frag_offsets[0].size(); ++tab_idx) {
    std::vector<uint64_t> tab_offs;
    for (auto& offsets : frag_offsets) {
      tab_offs.push_back(offsets[tab_idx]);
    }
    frag_sizes.push_back(get_consistent_frag_size(tab_offs));
  }
  return frag_sizes;
}

inline std::vector<int64_t> get_consistent_frags_sizes(
    const std::vector<Analyzer::Expr*>& target_exprs,
    const std::vector<int64_t>& table_frag_sizes) {
  std::vector<int64_t> col_frag_sizes;
  for (auto expr : target_exprs) {
    if (const auto col_var = dynamic_cast<Analyzer::ColumnVar*>(expr)) {
      if (col_var->get_rte_idx() < 0) {
        CHECK_EQ(-1, col_var->get_rte_idx());
        col_frag_sizes.push_back(int64_t(-1));
      } else {
        col_frag_sizes.push_back(table_frag_sizes[col_var->get_rte_idx()]);
      }
    } else {
      col_frag_sizes.push_back(int64_t(-1));
    }
  }
  return col_frag_sizes;
}

inline std::vector<std::vector<int64_t>> get_col_frag_offsets(
    const std::vector<Analyzer::Expr*>& target_exprs,
    const std::vector<std::vector<uint64_t>>& table_frag_offsets) {
  std::vector<std::vector<int64_t>> col_frag_offsets;
  for (auto& table_offsets : table_frag_offsets) {
    std::vector<int64_t> col_offsets;
    for (auto expr : target_exprs) {
      if (const auto col_var = dynamic_cast<Analyzer::ColumnVar*>(expr)) {
        if (col_var->get_rte_idx() < 0) {
          CHECK_EQ(-1, col_var->get_rte_idx());
          col_offsets.push_back(int64_t(-1));
        } else {
          CHECK_LT(static_cast<size_t>(col_var->get_rte_idx()), table_offsets.size());
          col_offsets.push_back(
              static_cast<int64_t>(table_offsets[col_var->get_rte_idx()]));
        }
      } else {
        col_offsets.push_back(int64_t(-1));
      }
    }
    col_frag_offsets.push_back(col_offsets);
  }
  return col_frag_offsets;
}

// Return the RelAlg input index of outer_table_id based on ra_exe_unit.input_descs.
// Used by UNION queries to get the target_exprs corresponding to the current subquery.
int get_input_idx(RelAlgExecutionUnit const& ra_exe_unit,
                  const shared::TableKey& outer_table_key) {
  auto match_table_key = [=](auto& desc) {
    return outer_table_key == desc.getTableKey();
  };
  auto& input_descs = ra_exe_unit.input_descs;
  auto itr = std::find_if(input_descs.begin(), input_descs.end(), match_table_key);
  return itr == input_descs.end() ? 0 : itr->getNestLevel();
}

void check_count_distinct_expr_metadata(const QueryMemoryDescriptor& query_mem_desc,
                                        const RelAlgExecutionUnit& ra_exe_unit) {
  const size_t agg_col_count{query_mem_desc.getSlotCount()};
  CHECK_GE(agg_col_count, ra_exe_unit.target_exprs.size());
  for (size_t target_idx = 0; target_idx < ra_exe_unit.target_exprs.size();
       ++target_idx) {
    const auto target_expr = ra_exe_unit.target_exprs[target_idx];
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    if (is_distinct_target(agg_info)) {
      CHECK(agg_info.is_agg &&
            (agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kCOUNT_IF ||
             agg_info.agg_kind == kAPPROX_COUNT_DISTINCT));
      CHECK(!agg_info.sql_type.is_varlen());
      const size_t agg_col_idx = query_mem_desc.getSlotIndexForSingleSlotCol(target_idx);
      CHECK_LT(static_cast<size_t>(agg_col_idx), agg_col_count);
      CHECK_EQ(static_cast<size_t>(query_mem_desc.getLogicalSlotWidthBytes(agg_col_idx)),
               sizeof(int64_t));
      const auto& count_distinct_desc =
          query_mem_desc.getCountDistinctDescriptor(target_idx);
      CHECK(count_distinct_desc.impl_type_ != CountDistinctImplType::Invalid);
    }
  }
}

QueryMemoryInitializer::TargetAggOpsMetadata collect_target_expr_metadata(
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit) {
  QueryMemoryInitializer::TargetAggOpsMetadata agg_op_metadata;
  if (!query_mem_desc.countDistinctDescriptorsLogicallyEmpty()) {
    agg_op_metadata.has_count_distinct = true;
  }
  std::for_each(
      ra_exe_unit.target_exprs.begin(),
      ra_exe_unit.target_exprs.end(),
      [&agg_op_metadata](const Analyzer::Expr* expr) {
        if (auto const* agg_expr = dynamic_cast<Analyzer::AggExpr const*>(expr)) {
          if (agg_expr->get_aggtype() == kMODE) {
            agg_op_metadata.has_mode = true;
          } else if (agg_expr->get_aggtype() == kAPPROX_QUANTILE) {
            agg_op_metadata.has_tdigest = true;
          }
        }
      });
  return agg_op_metadata;
}

}  // namespace

// Row-based execution constructor
QueryMemoryInitializer::QueryMemoryInitializer(
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const int device_id,
    const ExecutorDeviceType device_type,
    const ExecutorDispatchMode dispatch_mode,
    const bool output_columnar,
    const bool sort_on_gpu,
    const shared::TableKey& outer_table_key,
    const int64_t num_rows,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const ColumnBufferLayouts& col_buffer_layouts,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    RenderAllocatorMap* render_allocator_map,
    RenderInfo* render_info,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    const Executor* executor)
    : num_rows_(num_rows)
    , row_set_mem_owner_(row_set_mem_owner)
    , init_agg_vals_(executor->plan_state_->init_agg_vals_)
    , num_buffers_(computeNumberOfBuffers(query_mem_desc, device_type, executor))
    , varlen_output_buffer_(0)
    , varlen_output_buffer_host_ptr_(nullptr)
    , count_distinct_bitmap_device_mem_ptr_(0)
    , count_distinct_bitmap_mem_size_(0)
    , count_distinct_bitmap_host_crt_ptr_(nullptr)
    , count_distinct_bitmap_host_mem_ptr_(nullptr)
    , device_allocator_(device_allocator)
    , thread_idx_(thread_idx) {
  CHECK(!sort_on_gpu || output_columnar);
  executor->logSystemCPUMemoryStatus("Before Query Memory Initialization", thread_idx);

  const auto& consistent_frag_sizes = get_consistent_frags_sizes(frag_offsets);
  if (consistent_frag_sizes.empty()) {
    // No fragments in the input, no underlying buffers will be needed.
    return;
  }

  TargetAggOpsMetadata agg_op_metadata =
      collect_target_expr_metadata(query_mem_desc, ra_exe_unit);
  if (agg_op_metadata.has_count_distinct) {
    check_count_distinct_expr_metadata(query_mem_desc, ra_exe_unit);
    if (!ra_exe_unit.use_bump_allocator) {
      check_total_bitmap_memory(query_mem_desc, device_type, executor);
    }
    agg_op_metadata.count_distinct_buf_size =
        calculateCountDistinctBufferSize(query_mem_desc, ra_exe_unit);
    const auto total_buffer_size = count_distinct_bitmap_memory_bytes(query_mem_desc);
    if (device_type == ExecutorDeviceType::GPU) {
      allocateCountDistinctGpuMem(query_mem_desc, total_buffer_size);
      row_set_mem_owner_->initCountDistinctBufferForFastAllocation(
          count_distinct_bitmap_host_crt_ptr_, total_buffer_size, thread_idx_);
    } else {
      row_set_mem_owner_->allocAndInitCountDistinctBufferForFastAllocator(
          total_buffer_size, thread_idx_);
    }
  }

  if (agg_op_metadata.has_tdigest) {
    auto const& descs = query_mem_desc.getApproxQuantileDescriptors();
    // Pre-allocate all TDigest memory for this thread.
    AddNbytes const add_nbytes{query_mem_desc.getEntryCount()};
    size_t const capacity =
        std::accumulate(descs.begin(), descs.end(), size_t(0), add_nbytes);
    VLOG(2) << "row_set_mem_owner_->reserveTDigestMemory(" << thread_idx_ << ','
            << capacity << ") query_mem_desc.getEntryCount()("
            << query_mem_desc.getEntryCount() << ')';
    row_set_mem_owner_->reserveTDigestMemory(thread_idx_, capacity);
  }
  allocateModeMem(device_type, query_mem_desc, executor, device_id);

  if (render_allocator_map || !query_mem_desc.isGroupBy()) {
    if (agg_op_metadata.has_count_distinct) {
      fastAllocateCountDistinctBuffers(query_mem_desc, ra_exe_unit);
    }
    if (agg_op_metadata.has_mode) {
      allocateModeBuffer(query_mem_desc, ra_exe_unit);
    }
    if (agg_op_metadata.has_tdigest) {
      allocateTDigestsBuffer(query_mem_desc, ra_exe_unit);
    }
    if (render_info && render_info->useCudaBuffers()) {
      return;
    }
  }

  if (query_mem_desc.isGroupBy()) {
    if (agg_op_metadata.has_mode) {
      agg_op_metadata.mode_index_set =
          initializeModeIndexSet(query_mem_desc, ra_exe_unit);
    }
    if (agg_op_metadata.has_tdigest) {
      agg_op_metadata.quantile_params =
          initializeQuantileParams(query_mem_desc, ra_exe_unit);
    }
  }

  if (ra_exe_unit.estimator) {
    return;
  }

  const auto thread_count = device_type == ExecutorDeviceType::GPU
                                ? executor->blockSize() * executor->gridSize()
                                : 1;

  size_t group_buffer_size{0};
  if (ra_exe_unit.use_bump_allocator) {
    // For kernel per fragment execution, just allocate a buffer equivalent to the size of
    // the fragment
    if (dispatch_mode == ExecutorDispatchMode::KernelPerFragment) {
      group_buffer_size = query_mem_desc.didOutputColumnar()
                              ? query_mem_desc.getBufferSizeBytes(device_type, num_rows)
                              : num_rows * query_mem_desc.getRowSize();
    } else {
      // otherwise, allocate a GPU buffer equivalent to the maximum GPU allocation size
      group_buffer_size = g_max_memory_allocation_size / query_mem_desc.getRowSize();
    }
  } else {
    group_buffer_size =
        query_mem_desc.getBufferSizeBytes(ra_exe_unit, thread_count, device_type);
  }
  CHECK_GE(group_buffer_size, size_t(0));

  // In-situ rendering owns the host output allocation and expects it to be populated by
  // the execution path. Device-only deferred storage is valid only for ordinary query
  // ResultSets, which can materialize through ResultSet on demand.
  defer_gpu_baseline_hash_host_storage_ =
      !render_allocator_map && !render_info &&
      can_defer_gpu_group_by_host_storage(
          ra_exe_unit, query_mem_desc, device_type, output_columnar, init_agg_vals_);

  const auto group_buffers_count = !query_mem_desc.isGroupBy() ? 1 : num_buffers_;
  int64_t* group_by_buffer_template{nullptr};

  if (!defer_gpu_baseline_hash_host_storage_ &&
      !query_mem_desc.lazyInitGroups(device_type) && group_buffers_count > 1) {
    group_by_buffer_template = reinterpret_cast<int64_t*>(
        row_set_mem_owner_->allocate(group_buffer_size, thread_idx_));
    initGroupByBuffer(group_by_buffer_template,
                      ra_exe_unit,
                      query_mem_desc,
                      agg_op_metadata,
                      device_type,
                      output_columnar,
                      executor);
  }

  if (query_mem_desc.interleavedBins(device_type)) {
    CHECK(query_mem_desc.hasKeylessHash());
  }

  const auto step = device_type == ExecutorDeviceType::GPU &&
                            query_mem_desc.threadsShareMemory() &&
                            query_mem_desc.isGroupBy()
                        ? executor->blockSize()
                        : size_t(1);
  const auto index_buffer_qw = device_type == ExecutorDeviceType::GPU && sort_on_gpu &&
                                       query_mem_desc.hasKeylessHash()
                                   ? query_mem_desc.getEntryCount()
                                   : size_t(0);
  const auto actual_group_buffer_size =
      group_buffer_size + index_buffer_qw * sizeof(int64_t);
  CHECK_GE(actual_group_buffer_size, group_buffer_size);
  const auto host_group_buffer_size =
      defer_gpu_baseline_hash_host_storage_ ? sizeof(int64_t) : actual_group_buffer_size;

  if (query_mem_desc.hasVarlenOutput()) {
    const auto varlen_buffer_elem_size_opt = query_mem_desc.varlenOutputBufferElemSize();
    CHECK(varlen_buffer_elem_size_opt);  // TODO(adb): relax
    auto const varlen_buffer_sz =
        query_mem_desc.getEntryCount() * varlen_buffer_elem_size_opt.value();
    auto varlen_output_buffer =
        reinterpret_cast<int64_t*>(row_set_mem_owner_->allocate(varlen_buffer_sz));
    varlen_output_buffer_host_ptr_ = reinterpret_cast<int8_t*>(varlen_output_buffer);
    varlen_output_buffer_ =
        static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(varlen_output_buffer));
    auto varlen_output_info = getVarlenOutputInfo();
    varlen_output_info->gpu_start_address = static_cast<int64_t>(varlen_output_buffer_);
    varlen_output_info->cpu_buffer_ptr = varlen_output_buffer_host_ptr_;
    varlen_output_info->buffer_size_bytes = varlen_buffer_sz;
    num_buffers_ += 1;
    group_by_buffers_.push_back(varlen_output_buffer);
  }

  if (query_mem_desc.threadsCanReuseGroupByBuffers()) {
    // Sanity checks, intra-thread buffer reuse should only
    // occur on CPU for group-by queries, which also means
    // that only one group-by buffer should be allocated
    // (multiple-buffer allocation only occurs for GPU)
    CHECK(device_type == ExecutorDeviceType::CPU);
    CHECK(query_mem_desc.isGroupBy());
    CHECK_EQ(group_buffers_count, size_t(1));
  }

  // Group-by buffer reuse assumes 1 group-by-buffer per query step
  // Multiple group-by-buffers should only be used on GPU,
  // whereas buffer reuse only is done on CPU
  CHECK(group_buffers_count <= 1 || !query_mem_desc.threadsCanReuseGroupByBuffers());
  for (size_t i = 0; i < group_buffers_count; i += step) {
    auto group_by_info =
        alloc_group_by_buffer(host_group_buffer_size,
                              render_allocator_map,
                              thread_idx_,
                              row_set_mem_owner_.get(),
                              query_mem_desc.threadsCanReuseGroupByBuffers());

    auto group_by_buffer = group_by_info.first;
    const bool was_cached = group_by_info.second;
    if (!was_cached) {
      if (!defer_gpu_baseline_hash_host_storage_ &&
          !query_mem_desc.lazyInitGroups(device_type)) {
        if (group_by_buffer_template) {
          memcpy(group_by_buffer + index_buffer_qw,
                 group_by_buffer_template,
                 group_buffer_size);
        } else {
          initGroupByBuffer(group_by_buffer + index_buffer_qw,
                            ra_exe_unit,
                            query_mem_desc,
                            agg_op_metadata,
                            device_type,
                            output_columnar,
                            executor);
        }
      }
    }

    size_t old_size = group_by_buffers_.size();
    group_by_buffers_.resize(old_size + std::max(size_t(1), step), nullptr);
    group_by_buffers_[old_size] = group_by_buffer;

    const bool use_target_exprs_union =
        ra_exe_unit.union_all && get_input_idx(ra_exe_unit, outer_table_key);
    const auto& target_exprs = use_target_exprs_union ? ra_exe_unit.target_exprs_union
                                                      : ra_exe_unit.target_exprs;
    const auto column_frag_offsets = get_col_frag_offsets(target_exprs, frag_offsets);
    const auto column_frag_sizes =
        get_consistent_frags_sizes(target_exprs, consistent_frag_sizes);

    old_size = result_sets_.size();
    result_sets_.resize(old_size + std::max(size_t(1), step));
    result_sets_[old_size] = std::make_unique<ResultSet>(
        target_exprs_to_infos(target_exprs, query_mem_desc),
        executor->getColLazyFetchInfo(
            target_exprs, may_use_storage_local_lazy_fetch_rowid(ra_exe_unit)),
        col_buffers,
        col_buffer_layouts,
        column_frag_offsets,
        column_frag_sizes,
        device_type,
        device_id,
        thread_idx,
        ResultSet::fixupQueryMemoryDescriptor(query_mem_desc),
        row_set_mem_owner_,
        executor->blockSize(),
        executor->gridSize());
    if (device_type == ExecutorDeviceType::GPU) {
      result_sets_[old_size]->setCudaAllocator(executor, device_id);
      result_sets_[old_size]->setCudaStream(executor, device_id);
    }
    result_sets_[old_size]->allocateStorage(reinterpret_cast<int8_t*>(group_by_buffer),
                                            executor->plan_state_->init_agg_vals_,
                                            getVarlenOutputInfo(),
                                            host_group_buffer_size);
  }
}

void QueryMemoryInitializer::setDeferredLazyFetchChunks(
    const DeferredLazyFetchChunks& deferred_lazy_fetch_chunks) {
  if (deferred_lazy_fetch_chunks.empty()) {
    return;
  }
  for (auto& result_set : result_sets_) {
    if (result_set) {
      result_set->setDeferredLazyFetchChunks(deferred_lazy_fetch_chunks);
    }
  }
}

void QueryMemoryInitializer::setLazyFetchSourceMetadata(
    const LazyFetchSourceMetadata& lazy_fetch_source_metadata) {
  if (lazy_fetch_source_metadata.empty()) {
    return;
  }
  for (auto& result_set : result_sets_) {
    if (result_set) {
      result_set->setLazyFetchSourceMetadata(lazy_fetch_source_metadata);
    }
  }
}

// Table functions execution constructor
QueryMemoryInitializer::QueryMemoryInitializer(
    const TableFunctionExecutionUnit& exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const int device_id,
    const ExecutorDeviceType device_type,
    const int64_t num_rows,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const ColumnBufferLayouts& col_buffer_layouts,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    DeviceAllocator* device_allocator,
    const Executor* executor)
    : num_rows_(num_rows)
    , row_set_mem_owner_(row_set_mem_owner)
    , init_agg_vals_(init_agg_val_vec(exe_unit.target_exprs, {}, query_mem_desc))
    , num_buffers_(1)
    , varlen_output_buffer_(0)
    , varlen_output_buffer_host_ptr_(nullptr)
    , count_distinct_bitmap_device_mem_ptr_(0)
    , count_distinct_bitmap_mem_size_(0)
    , count_distinct_bitmap_host_crt_ptr_(nullptr)
    , count_distinct_bitmap_host_mem_ptr_(nullptr)
    , device_allocator_(device_allocator)
    , thread_idx_(0) {
  // Table functions output columnar, basically treat this as a projection
  const auto& consistent_frag_sizes = get_consistent_frags_sizes(frag_offsets);
  if (consistent_frag_sizes.empty()) {
    // No fragments in the input, no underlying buffers will be needed.
    return;
  }

  const size_t num_columns =
      query_mem_desc.getBufferColSlotCount();  // shouldn't we use getColCount() ???
  size_t total_group_by_buffer_size{0};
  for (size_t i = 0; i < num_columns; ++i) {
    auto ti = exe_unit.target_exprs[i]->get_type_info();
    if (ti.usesFlatBuffer()) {
      // See TableFunctionManager.h for info regarding flatbuffer
      // memory managment.
      auto slot_idx = query_mem_desc.getSlotIndexForSingleSlotCol(i);
      CHECK(query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx));
      checked_int64_t flatbuffer_size = query_mem_desc.getFlatBufferSize(slot_idx);
      try {
        total_group_by_buffer_size = align_to_int64(
            static_cast<int64_t>(total_group_by_buffer_size + flatbuffer_size));
      } catch (...) {
        throw OutOfHostMemory(std::numeric_limits<int64_t>::max() / 8);
      }
    } else {
      const checked_int64_t col_width = ti.get_size();
      try {
        const checked_int64_t group_buffer_size = col_width * num_rows_;
        total_group_by_buffer_size = align_to_int64(
            static_cast<int64_t>(group_buffer_size + total_group_by_buffer_size));
      } catch (...) {
        throw OutOfHostMemory(std::numeric_limits<int64_t>::max() / 8);
      }
    }
  }

#ifdef __SANITIZE_ADDRESS__
  // AddressSanitizer will reject allocation sizes above 1 TiB
#define MAX_BUFFER_SIZE 0x10000000000ll
#else
  // otherwise, we'll set the limit to 16 TiB, feel free to increase
  // the limit if needed
#define MAX_BUFFER_SIZE 0x100000000000ll
#endif

  if (total_group_by_buffer_size >= MAX_BUFFER_SIZE) {
    throw OutOfHostMemory(total_group_by_buffer_size);
  }

  CHECK_EQ(num_buffers_, size_t(1));
  auto group_by_buffer = alloc_group_by_buffer(total_group_by_buffer_size,
                                               nullptr,
                                               thread_idx_,
                                               row_set_mem_owner.get(),
                                               false)
                             .first;
  group_by_buffers_.push_back(group_by_buffer);

  const auto column_frag_offsets =
      get_col_frag_offsets(exe_unit.target_exprs, frag_offsets);
  const auto column_frag_sizes =
      get_consistent_frags_sizes(exe_unit.target_exprs, consistent_frag_sizes);
  result_sets_.emplace_back(
      new ResultSet(target_exprs_to_infos(exe_unit.target_exprs, query_mem_desc),
                    /*col_lazy_fetch_info=*/{},
                    col_buffers,
                    col_buffer_layouts,
                    column_frag_offsets,
                    column_frag_sizes,
                    device_type,
                    device_id,
                    -1, /*thread_idx*/
                    ResultSet::fixupQueryMemoryDescriptor(query_mem_desc),
                    row_set_mem_owner_,
                    executor->blockSize(),
                    executor->gridSize()));
  result_sets_.back()->allocateStorage(reinterpret_cast<int8_t*>(group_by_buffer),
                                       init_agg_vals_);
}

void QueryMemoryInitializer::initGroupByBuffer(
    int64_t* buffer,
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    TargetAggOpsMetadata& agg_op_metadata,
    const ExecutorDeviceType device_type,
    const bool output_columnar,
    const Executor* executor) {
  if (output_columnar) {
    initColumnarGroups(query_mem_desc, buffer, init_agg_vals_, executor, ra_exe_unit);
  } else {
    auto rows_ptr = buffer;
    auto actual_entry_count = query_mem_desc.getEntryCount();
    const auto thread_count = device_type == ExecutorDeviceType::GPU
                                  ? executor->blockSize() * executor->gridSize()
                                  : 1;
    auto warp_size =
        query_mem_desc.interleavedBins(device_type) ? executor->warpSize() : 1;
    if (query_mem_desc.useStreamingTopN()) {
      const auto node_count_size = thread_count * sizeof(int64_t);
      memset(rows_ptr, 0, node_count_size);
      const auto n =
          ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit.value_or(0);
      const auto rows_offset = streaming_top_n::get_rows_offset_of_heaps(n, thread_count);
      memset(rows_ptr + thread_count, -1, rows_offset - node_count_size);
      rows_ptr += rows_offset / sizeof(int64_t);
      actual_entry_count = n * thread_count;
      warp_size = 1;
    }
    initRowGroups(query_mem_desc,
                  rows_ptr,
                  init_agg_vals_,
                  agg_op_metadata,
                  actual_entry_count,
                  warp_size,
                  executor,
                  ra_exe_unit);
  }
}

void QueryMemoryInitializer::initRowGroups(const QueryMemoryDescriptor& query_mem_desc,
                                           int64_t* groups_buffer,
                                           const std::vector<int64_t>& init_vals,
                                           TargetAggOpsMetadata& agg_op_metadata,
                                           const int32_t groups_buffer_entry_count,
                                           const size_t warp_size,
                                           const Executor* executor,
                                           const RelAlgExecutionUnit& ra_exe_unit) {
  const size_t key_count{query_mem_desc.getGroupbyColCount()};
  const size_t row_size{query_mem_desc.getRowSize()};
  const size_t col_base_off{query_mem_desc.getColOffInBytes(0)};

  auto buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);
  const auto query_mem_desc_fixedup =
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc);
  auto const key_sz = query_mem_desc.getEffectiveKeyWidth();
  // not COUNT DISTINCT / APPROX_COUNT_DISTINCT / APPROX_QUANTILE
  // we use the default implementation in those agg ops
  if (!(agg_op_metadata.has_count_distinct || agg_op_metadata.has_mode ||
        agg_op_metadata.has_tdigest) &&
      g_optimize_row_initialization) {
    std::vector<int8_t> sample_row(row_size - col_base_off);
    auto const num_available_cpu_threads =
        std::min(query_mem_desc.getAvailableCpuThreads(),
                 static_cast<size_t>(std::max(cpu_threads(), 1)));
    tbb::task_arena initialization_arena(num_available_cpu_threads);

    initColumnsPerRow(
        query_mem_desc_fixedup, sample_row.data(), init_vals, agg_op_metadata);

    if (query_mem_desc.hasKeylessHash()) {
      CHECK(warp_size >= 1);
      CHECK(key_count == 1 || warp_size == 1);
      initialization_arena.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, groups_buffer_entry_count * warp_size),
            [&](const tbb::blocked_range<size_t>& r) {
              auto cur_row_buf = buffer_ptr + (row_size * r.begin());
              for (size_t i = r.begin(); i != r.end(); ++i, cur_row_buf += row_size) {
                memcpy(cur_row_buf + col_base_off, sample_row.data(), sample_row.size());
              }
            });
      });
      return;
    }
    initialization_arena.execute([&] {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, groups_buffer_entry_count),
          [&](const tbb::blocked_range<size_t>& r) {
            auto cur_row_buf = buffer_ptr + (row_size * r.begin());
            for (size_t i = r.begin(); i != r.end(); ++i, cur_row_buf += row_size) {
              memcpy(cur_row_buf + col_base_off, sample_row.data(), sample_row.size());
              result_set::fill_empty_key(cur_row_buf, key_count, key_sz);
            }
          });
    });
  } else {
    // todo(yoonmin): allow parallelization of `initColumnsPerRow`
    if (query_mem_desc.hasKeylessHash()) {
      CHECK(warp_size >= 1);
      CHECK(key_count == 1 || warp_size == 1);
      for (size_t warp_idx = 0; warp_idx < warp_size; ++warp_idx) {
        for (size_t bin = 0; bin < static_cast<size_t>(groups_buffer_entry_count);
             ++bin, buffer_ptr += row_size) {
          initColumnsPerRow(query_mem_desc_fixedup,
                            &buffer_ptr[col_base_off],
                            init_vals,
                            agg_op_metadata);
        }
      }
      return;
    }

    for (size_t bin = 0; bin < static_cast<size_t>(groups_buffer_entry_count);
         ++bin, buffer_ptr += row_size) {
      result_set::fill_empty_key(
          buffer_ptr, key_count, query_mem_desc.getEffectiveKeyWidth());
      initColumnsPerRow(
          query_mem_desc_fixedup, &buffer_ptr[col_base_off], init_vals, agg_op_metadata);
    }
  }
}

namespace {

template <typename T>
int8_t* initColumnarBuffer(T* buffer_ptr, const T init_val, const uint32_t entry_count) {
  static_assert(sizeof(T) <= sizeof(int64_t), "Unsupported template type");
  for (uint32_t i = 0; i < entry_count; ++i) {
    buffer_ptr[i] = init_val;
  }
  return reinterpret_cast<int8_t*>(buffer_ptr + entry_count);
}

}  // namespace

void QueryMemoryInitializer::initColumnarGroups(
    const QueryMemoryDescriptor& query_mem_desc,
    int64_t* groups_buffer,
    const std::vector<int64_t>& init_vals,
    const Executor* executor,
    const RelAlgExecutionUnit& ra_exe_unit) {
  CHECK(groups_buffer);

  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    CHECK(!is_distinct_target(agg_info));
  }
  const int32_t agg_col_count = query_mem_desc.getSlotCount();
  auto buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);

  const auto groups_buffer_entry_count = query_mem_desc.getEntryCount();
  if (!query_mem_desc.hasKeylessHash()) {
    const size_t key_count{query_mem_desc.getGroupbyColCount()};
    for (size_t i = 0; i < key_count; ++i) {
      buffer_ptr = initColumnarBuffer<int64_t>(reinterpret_cast<int64_t*>(buffer_ptr),
                                               EMPTY_KEY_64,
                                               groups_buffer_entry_count);
    }
  }

  if (query_mem_desc.getQueryDescriptionType() != QueryDescriptionType::Projection) {
    // initializing all aggregate columns:
    int32_t init_val_idx = 0;
    for (int32_t i = 0; i < agg_col_count; ++i) {
      if (query_mem_desc.getPaddedSlotWidthBytes(i) > 0) {
        CHECK_LT(static_cast<size_t>(init_val_idx), init_vals.size());
        switch (query_mem_desc.getPaddedSlotWidthBytes(i)) {
          case 1:
            buffer_ptr = initColumnarBuffer<int8_t>(
                buffer_ptr, init_vals[init_val_idx++], groups_buffer_entry_count);
            break;
          case 2:
            buffer_ptr =
                initColumnarBuffer<int16_t>(reinterpret_cast<int16_t*>(buffer_ptr),
                                            init_vals[init_val_idx++],
                                            groups_buffer_entry_count);
            break;
          case 4:
            buffer_ptr =
                initColumnarBuffer<int32_t>(reinterpret_cast<int32_t*>(buffer_ptr),
                                            init_vals[init_val_idx++],
                                            groups_buffer_entry_count);
            break;
          case 8:
            buffer_ptr =
                initColumnarBuffer<int64_t>(reinterpret_cast<int64_t*>(buffer_ptr),
                                            init_vals[init_val_idx++],
                                            groups_buffer_entry_count);
            break;
          case 0:
            break;
          default:
            CHECK(false);
        }

        buffer_ptr = align_to_int64(buffer_ptr);
      }
    }
  }
}

void QueryMemoryInitializer::initColumnsPerRow(
    const QueryMemoryDescriptor& query_mem_desc,
    int8_t* row_ptr,
    const std::vector<int64_t>& init_vals,
    const TargetAggOpsMetadata& agg_op_metadata) {
  int8_t* col_ptr = row_ptr;
  size_t init_vec_idx = 0;
  size_t approx_quantile_descriptors_idx = 0;
  for (size_t col_idx = 0; col_idx < query_mem_desc.getSlotCount();
       col_ptr += query_mem_desc.getNextColOffInBytesRowOnly(col_ptr, col_idx++)) {
    int64_t init_val{0};
    if (query_mem_desc.isGroupBy()) {
      if (agg_op_metadata.has_count_distinct &&
          agg_op_metadata.count_distinct_buf_size[col_idx]) {
        // COUNT DISTINCT / APPROX_COUNT_DISTINCT
        // create a data structure for count_distinct operator per entries
        const int64_t bm_sz{agg_op_metadata.count_distinct_buf_size[col_idx]};
        CHECK_EQ(static_cast<size_t>(query_mem_desc.getPaddedSlotWidthBytes(col_idx)),
                 sizeof(int64_t));
        init_val =
            bm_sz > 0 ? allocateCountDistinctBitmap(bm_sz) : allocateCountDistinctSet();
        CHECK_NE(init_val, 0);
        ++init_vec_idx;
      } else if (agg_op_metadata.has_tdigest &&
                 agg_op_metadata.quantile_params[col_idx]) {
        auto const q = *agg_op_metadata.quantile_params[col_idx];
        auto const& descs = query_mem_desc.getApproxQuantileDescriptors();
        auto const& desc = descs.at(approx_quantile_descriptors_idx++);
        init_val = reinterpret_cast<int64_t>(
            row_set_mem_owner_->initTDigest(thread_idx_, desc, q));
        CHECK_NE(init_val, 0);
        ++init_vec_idx;
      } else if (agg_op_metadata.has_mode &&
                 agg_op_metadata.mode_index_set.count(col_idx)) {
        init_val = allocateAggMode();
        CHECK_NE(init_val, 0);
        ++init_vec_idx;
      }
    }
    auto const col_slot_width = query_mem_desc.getPaddedSlotWidthBytes(col_idx);
    if (init_val == 0 && col_slot_width > 0) {
      CHECK_LT(init_vec_idx, init_vals.size());
      init_val = init_vals[init_vec_idx++];
    }
    switch (col_slot_width) {
      case 1:
        *col_ptr = static_cast<int8_t>(init_val);
        break;
      case 2:
        *reinterpret_cast<int16_t*>(col_ptr) = (int16_t)init_val;
        break;
      case 4:
        *reinterpret_cast<int32_t*>(col_ptr) = (int32_t)init_val;
        break;
      case 8:
        *reinterpret_cast<int64_t*>(col_ptr) = init_val;
        break;
      case 0:
        continue;
      default:
        CHECK(false);
    }
  }
}

void QueryMemoryInitializer::allocateModeMem(ExecutorDeviceType const device_type,
                                             const QueryMemoryDescriptor& query_mem_desc,
                                             const Executor* executor,
                                             int device_id) {
  if (size_t const ncolumns = query_mem_desc.getNumModeTargets()) {
    size_t const nmodes = ncolumns * query_mem_desc.getEntryCount();
    VLOG(2) << "device_type(" << device_type << ") nmodes = " << nmodes << " = "
            << ncolumns << " * " << query_mem_desc.getEntryCount();
    agg_mode_hash_tables_cpu_.reserve(nmodes);
    if (device_type == ExecutorDeviceType::GPU) {
#ifdef HAVE_CUDA
      // agg_mode_hash_tables_gpu_ also uses the cudaStream from device_allocator_.
      auto* const allocator = dynamic_cast<CudaAllocator*>(device_allocator_);
      CHECK(allocator) << "CudaAllocator expected for ExecutorDeviceType::GPU.";
      agg_mode_hash_tables_gpu_.init(
          allocator, executor->getCudaStream(device_id), nmodes);
#else
      UNREACHABLE();
#endif
    }
  }
}

void QueryMemoryInitializer::allocateCountDistinctGpuMem(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t total_bytes) {
  const auto descriptor_count = query_mem_desc.getCountDistinctDescriptorsSize();
  for (size_t descriptor_idx = 0; descriptor_idx < descriptor_count; ++descriptor_idx) {
    const auto descriptor = query_mem_desc.getCountDistinctDescriptor(descriptor_idx);
    CHECK(descriptor.impl_type_ == CountDistinctImplType::Invalid ||
          descriptor.impl_type_ == CountDistinctImplType::Bitmap);
  }
  if (total_bytes == 0) {
    return;
  }
  CHECK(device_allocator_);
  count_distinct_bitmap_mem_size_ = total_bytes;
  auto cuda_allocator = dynamic_cast<CudaAllocator*>(device_allocator_);
  CHECK(cuda_allocator);
  VLOG(1) << "Allocate count distinct buffer on GPU " << cuda_allocator->getDeviceId()
          << " (size: " << count_distinct_bitmap_mem_size_ << " bytes)";
  count_distinct_bitmap_device_mem_ptr_ = reinterpret_cast<CUdeviceptr>(
      cuda_allocator->alloc(count_distinct_bitmap_mem_size_));
  cuda_allocator->zeroDeviceMem(
      reinterpret_cast<int8_t*>(count_distinct_bitmap_device_mem_ptr_),
      count_distinct_bitmap_mem_size_);
  VLOG(1) << "Allocate count distinct buffer on CPU (size: "
          << count_distinct_bitmap_mem_size_ << " bytes)";
  count_distinct_bitmap_host_crt_ptr_ = count_distinct_bitmap_host_mem_ptr_ =
      row_set_mem_owner_->allocate(count_distinct_bitmap_mem_size_, thread_idx_);
}

std::vector<int64_t> QueryMemoryInitializer::calculateCountDistinctBufferSize(
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit) const {
  const size_t agg_col_count{query_mem_desc.getSlotCount()};
  std::vector<int64_t> agg_bitmap_size(agg_col_count);
  for (size_t target_idx = 0; target_idx < ra_exe_unit.target_exprs.size();
       ++target_idx) {
    const auto target_expr = ra_exe_unit.target_exprs[target_idx];
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    if (is_distinct_target(agg_info)) {
      const size_t agg_col_idx = query_mem_desc.getSlotIndexForSingleSlotCol(target_idx);
      const auto& count_distinct_desc =
          query_mem_desc.getCountDistinctDescriptor(target_idx);
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
        const auto bitmap_byte_sz = count_distinct_desc.bitmapPaddedSizeBytes();
        agg_bitmap_size[agg_col_idx] = bitmap_byte_sz;
      } else {
        CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::UnorderedSet);
        agg_bitmap_size[agg_col_idx] = -1;
      }
    }
  }
  return agg_bitmap_size;
}

void QueryMemoryInitializer::fastAllocateCountDistinctBuffers(
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit) {
  for (size_t target_idx = 0; target_idx < ra_exe_unit.target_exprs.size();
       ++target_idx) {
    const auto target_expr = ra_exe_unit.target_exprs[target_idx];
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
    if (is_distinct_target(agg_info)) {
      const size_t agg_col_idx = query_mem_desc.getSlotIndexForSingleSlotCol(target_idx);
      const auto& count_distinct_desc =
          query_mem_desc.getCountDistinctDescriptor(target_idx);
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
        const auto bitmap_byte_sz = count_distinct_desc.bitmapPaddedSizeBytes();
        init_agg_vals_[agg_col_idx] = allocateCountDistinctBitmap(bitmap_byte_sz);
      } else {
        CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::UnorderedSet);
        init_agg_vals_[agg_col_idx] = allocateCountDistinctSet();
      }
    }
  }
}

// Increments count_distinct_bitmap_host_crt_ptr_ to point to the next cell in
// count_distinct_bitmap_host_mem_ptr_.
int64_t QueryMemoryInitializer::allocateCountDistinctBitmap(const size_t bitmap_byte_sz) {
  if (count_distinct_bitmap_host_mem_ptr_) {
    CHECK(count_distinct_bitmap_host_crt_ptr_);
    auto ptr = count_distinct_bitmap_host_crt_ptr_;
    count_distinct_bitmap_host_crt_ptr_ += bitmap_byte_sz;
    row_set_mem_owner_->addCountDistinctBuffer(
        ptr, bitmap_byte_sz, /*physial_buffer=*/false);
    return reinterpret_cast<int64_t>(ptr);
  }
  return reinterpret_cast<int64_t>(
      row_set_mem_owner_->fastAllocateCountDistinctBuffer(bitmap_byte_sz, thread_idx_));
}

int64_t QueryMemoryInitializer::allocateCountDistinctSet() {
  auto count_distinct_set = new CountDistinctSet();
  row_set_mem_owner_->addCountDistinctSet(count_distinct_set);
  return reinterpret_cast<int64_t>(count_distinct_set);
}

// Return CPU: AggMode* or GPU: (1<<63 | i<<32 | j+1)
// When run on GPU the bit fields are:
// 1<<63 : used to distinguish the packed 32-bit indices vs pointer,
//         since a pointer uses only the lower 48 bits.
// i : used in agg_mode_func_gpu() as index into gpu hash tables array.
// j : used in ResultSet::makeTargetValue() as index into
//     std::deque<AggMode> RowSetMemoryOwner::agg_modes_.
int64_t QueryMemoryInitializer::allocateAggMode() {
  constexpr size_t high_bit = size_t(1) << 31;
  bool const is_gpu = static_cast<bool>(getNumAggModeHashTablesGpu());
  if (is_gpu && high_bit <= agg_mode_hash_tables_cpu_.size()) {
    throw QueryMustRunOnCpu();
  }
  auto const idx_ptr = row_set_mem_owner_->allocateMode();  // (index, AggMode*)
  size_t const cpu_idx = agg_mode_hash_tables_cpu_.size();
  agg_mode_hash_tables_cpu_.push_back(idx_ptr.second);
  return is_gpu ? static_cast<int64_t>((high_bit | cpu_idx) << 32) |
                      static_cast<int64_t>(idx_ptr.first)
                : reinterpret_cast<int64_t>(idx_ptr.second);
}

namespace {

void eachAggregateTargetIdxOfType(
    std::vector<Analyzer::Expr*> const& target_exprs,
    SQLAgg const agg_type,
    std::function<void(Analyzer::AggExpr const*, size_t)> lambda) {
  for (size_t target_idx = 0; target_idx < target_exprs.size(); ++target_idx) {
    auto const target_expr = target_exprs[target_idx];
    if (auto const* agg_expr = dynamic_cast<Analyzer::AggExpr const*>(target_expr)) {
      if (agg_expr->get_aggtype() == agg_type) {
        lambda(agg_expr, target_idx);
      }
    }
  }
}

}  // namespace

QueryMemoryInitializer::ModeIndexSet QueryMemoryInitializer::initializeModeIndexSet(
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit) {
  size_t const slot_count = query_mem_desc.getSlotCount();
  CHECK_LE(ra_exe_unit.target_exprs.size(), slot_count);
  ModeIndexSet mode_index_set;
  eachAggregateTargetIdxOfType(
      ra_exe_unit.target_exprs,
      kMODE,
      [&](Analyzer::AggExpr const*, size_t const target_idx) {
        size_t const agg_col_idx =
            query_mem_desc.getSlotIndexForSingleSlotCol(target_idx);
        CHECK_LT(agg_col_idx, slot_count);
        mode_index_set.emplace(agg_col_idx);
      });
  return mode_index_set;
}

void QueryMemoryInitializer::allocateModeBuffer(
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit) {
  size_t const slot_count = query_mem_desc.getSlotCount();
  CHECK_LE(ra_exe_unit.target_exprs.size(), slot_count);
  ra_exe_unit.eachAggTarget<kMODE>([&](Analyzer::AggExpr const*,
                                       size_t const target_idx) {
    size_t const agg_col_idx = query_mem_desc.getSlotIndexForSingleSlotCol(target_idx);
    CHECK_LT(agg_col_idx, slot_count);
    init_agg_vals_[agg_col_idx] = allocateAggMode();
  });
}

std::vector<QueryMemoryInitializer::QuantileParam>
QueryMemoryInitializer::initializeQuantileParams(
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit) {
  size_t const slot_count = query_mem_desc.getSlotCount();
  CHECK_LE(ra_exe_unit.target_exprs.size(), slot_count);
  std::vector<QuantileParam> quantile_params(slot_count);
  ra_exe_unit.eachAggTarget<kAPPROX_QUANTILE>([&](Analyzer::AggExpr const* const agg_expr,
                                                  size_t const target_idx) {
    size_t const agg_col_idx = query_mem_desc.getSlotIndexForSingleSlotCol(target_idx);
    CHECK_LT(agg_col_idx, slot_count);
    CHECK_EQ(static_cast<int8_t>(sizeof(int64_t)),
             query_mem_desc.getLogicalSlotWidthBytes(agg_col_idx));
    auto const q_expr =
        dynamic_cast<Analyzer::Constant const*>(agg_expr->get_arg1().get());
    CHECK(q_expr);
    quantile_params[agg_col_idx] = q_expr->get_constval().doubleval;
  });
  return quantile_params;
}

void QueryMemoryInitializer::allocateTDigestsBuffer(
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit) {
  size_t const slot_count = query_mem_desc.getSlotCount();
  CHECK_LE(ra_exe_unit.target_exprs.size(), slot_count);

  auto const& descs = query_mem_desc.getApproxQuantileDescriptors();
  size_t approx_quantile_descriptors_idx = 0u;
  ra_exe_unit.eachAggTarget<kAPPROX_QUANTILE>([&](Analyzer::AggExpr const* const agg_expr,
                                                  size_t const target_idx) {
    size_t const agg_col_idx = query_mem_desc.getSlotIndexForSingleSlotCol(target_idx);
    CHECK_LT(agg_col_idx, slot_count);
    CHECK_EQ(static_cast<int8_t>(sizeof(int64_t)),
             query_mem_desc.getLogicalSlotWidthBytes(agg_col_idx));
    auto const q_expr =
        dynamic_cast<Analyzer::Constant const*>(agg_expr->get_arg1().get());
    CHECK(q_expr);
    auto const q = q_expr->get_constval().doubleval;
    auto const& desc = descs.at(approx_quantile_descriptors_idx++);
    init_agg_vals_[agg_col_idx] =
        reinterpret_cast<int64_t>(row_set_mem_owner_->initTDigest(thread_idx_, desc, q));
  });
}

GpuGroupByBuffers QueryMemoryInitializer::prepareTopNHeapsDevBuffer(
    const QueryMemoryDescriptor& query_mem_desc,
    const int8_t* init_agg_vals_dev_ptr,
    const size_t n,
    const int device_id,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    CUstream cuda_stream) {
#ifdef HAVE_CUDA
  CHECK(device_allocator_);
  const auto thread_count = block_size_x * grid_size_x;
  const auto total_buff_size =
      streaming_top_n::get_heap_size(query_mem_desc.getRowSize(), n, thread_count);
  int8_t* dev_buffer = device_allocator_->alloc(total_buff_size);

  std::vector<int8_t*> dev_buffers(thread_count);

  for (size_t i = 0; i < thread_count; ++i) {
    dev_buffers[i] = dev_buffer;
  }

  auto dev_ptr = device_allocator_->alloc(thread_count * sizeof(int8_t*));
  device_allocator_->copyToDevice(dev_ptr,
                                  dev_buffers.data(),
                                  thread_count * sizeof(int8_t*),
                                  "Streaming Top-N buffer ptrs");

  CHECK(query_mem_desc.lazyInitGroups(ExecutorDeviceType::GPU));

  device_allocator_->zeroDeviceMem(reinterpret_cast<int8_t*>(dev_buffer),
                                   thread_count * sizeof(int64_t));

  device_allocator_->setDeviceMem(
      reinterpret_cast<int8_t*>(dev_buffer + thread_count * sizeof(int64_t)),
      (unsigned char)-1,
      thread_count * n * sizeof(int64_t));

  init_group_by_buffer_on_device(
      reinterpret_cast<int64_t*>(
          dev_buffer + streaming_top_n::get_rows_offset_of_heaps(n, thread_count)),
      reinterpret_cast<const int64_t*>(init_agg_vals_dev_ptr),
      n * thread_count,
      query_mem_desc.getGroupbyColCount(),
      query_mem_desc.getEffectiveKeyWidth(),
      query_mem_desc.getRowSize() / sizeof(int64_t),
      query_mem_desc.hasKeylessHash(),
      1,
      block_size_x,
      grid_size_x,
      cuda_stream);

  return {dev_ptr, dev_buffer};
#else
  UNREACHABLE();
  return {};
#endif
}

GpuGroupByBuffers QueryMemoryInitializer::createAndInitializeGroupByBufferGpu(
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const int8_t* init_agg_vals_dev_ptr,
    const int device_id,
    CUstream cuda_stream,
    const ExecutorDispatchMode dispatch_mode,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int8_t warp_size,
    const bool can_sort_on_gpu,
    const bool output_columnar,
    RenderAllocator* render_allocator) {
#ifdef HAVE_CUDA
  if (query_mem_desc.useStreamingTopN()) {
    if (render_allocator) {
      throw StreamingTopNNotSupportedInRenderQuery();
    }
    const auto n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit.value_or(0);
    CHECK(!output_columnar);

    return prepareTopNHeapsDevBuffer(query_mem_desc,
                                     init_agg_vals_dev_ptr,
                                     n,
                                     device_id,
                                     block_size_x,
                                     grid_size_x,
                                     cuda_stream);
  }

  auto dev_group_by_buffers =
      create_dev_group_by_buffers(device_allocator_,
                                  group_by_buffers_,
                                  query_mem_desc,
                                  block_size_x,
                                  grid_size_x,
                                  device_id,
                                  dispatch_mode,
                                  num_rows_,
                                  can_sort_on_gpu,
                                  false,
                                  ra_exe_unit.use_bump_allocator,
                                  query_mem_desc.hasVarlenOutput(),
                                  render_allocator);
  if (query_mem_desc.hasVarlenOutput()) {
    CHECK(dev_group_by_buffers.varlen_output_buffer);
    varlen_output_buffer_ =
        reinterpret_cast<CUdeviceptr>(dev_group_by_buffers.varlen_output_buffer);
    CHECK(query_mem_desc.varlenOutputBufferElemSize());
    const size_t varlen_output_buf_bytes =
        query_mem_desc.getEntryCount() *
        query_mem_desc.varlenOutputBufferElemSize().value();
    varlen_output_buffer_host_ptr_ =
        row_set_mem_owner_->allocate(varlen_output_buf_bytes, thread_idx_);
    CHECK(varlen_output_info_);
    varlen_output_info_->gpu_start_address = static_cast<int64_t>(varlen_output_buffer_);
    varlen_output_info_->cpu_buffer_ptr = varlen_output_buffer_host_ptr_;
    varlen_output_info_->buffer_size_bytes = varlen_output_buf_bytes;
  }
  if (render_allocator) {
    CHECK_EQ(size_t(0), render_allocator->getAllocatedSize() % 8);
  }
  if (query_mem_desc.lazyInitGroups(ExecutorDeviceType::GPU)) {
    CHECK(!render_allocator);

    const size_t step{query_mem_desc.threadsShareMemory() ? block_size_x : 1};
    size_t groups_buffer_size{query_mem_desc.getBufferSizeBytes(
        ExecutorDeviceType::GPU, dev_group_by_buffers.entry_count)};
    auto group_by_dev_buffer = dev_group_by_buffers.data;
    const size_t col_count = query_mem_desc.getSlotCount();
    int8_t* col_widths_dev_ptr{nullptr};
    if (output_columnar) {
      std::vector<int8_t> compact_col_widths(col_count);
      for (size_t idx = 0; idx < col_count; ++idx) {
        compact_col_widths[idx] = query_mem_desc.getPaddedSlotWidthBytes(idx);
      }
      col_widths_dev_ptr = device_allocator_->alloc(col_count * sizeof(int8_t));
      device_allocator_->copyToDevice(col_widths_dev_ptr,
                                      compact_col_widths.data(),
                                      col_count * sizeof(int8_t),
                                      "Compact column widths");
    }
    const int8_t warp_count =
        query_mem_desc.interleavedBins(ExecutorDeviceType::GPU) ? warp_size : 1;
    const auto num_group_by_buffers =
        getGroupByBuffersSize() - (query_mem_desc.hasVarlenOutput() ? 1 : 0);
    for (size_t i = 0; i < num_group_by_buffers; i += step) {
      if (output_columnar) {
        init_columnar_group_by_buffer_on_device(
            reinterpret_cast<int64_t*>(group_by_dev_buffer),
            reinterpret_cast<const int64_t*>(init_agg_vals_dev_ptr),
            dev_group_by_buffers.entry_count,
            query_mem_desc.getGroupbyColCount(),
            col_count,
            col_widths_dev_ptr,
            /*need_padding = */ true,
            query_mem_desc.hasKeylessHash(),
            sizeof(int64_t),
            block_size_x,
            grid_size_x,
            cuda_stream);
      } else {
        init_group_by_buffer_on_device(
            reinterpret_cast<int64_t*>(group_by_dev_buffer),
            reinterpret_cast<const int64_t*>(init_agg_vals_dev_ptr),
            dev_group_by_buffers.entry_count,
            query_mem_desc.getGroupbyColCount(),
            query_mem_desc.getEffectiveKeyWidth(),
            query_mem_desc.getRowSize() / sizeof(int64_t),
            query_mem_desc.hasKeylessHash(),
            warp_count,
            block_size_x,
            grid_size_x,
            cuda_stream);
      }
      group_by_dev_buffer += groups_buffer_size;
    }
  }
  return dev_group_by_buffers;
#else
  UNREACHABLE();
  return {};
#endif
}

GpuGroupByBuffers QueryMemoryInitializer::setupTableFunctionGpuBuffers(
    const QueryMemoryDescriptor& query_mem_desc,
    const int device_id,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const bool zero_initialize_buffers) {
  const size_t num_columns = query_mem_desc.getBufferColSlotCount();
  CHECK_GT(num_columns, size_t(0));
  size_t total_group_by_buffer_size{0};
  const auto col_slot_context = query_mem_desc.getColSlotContext();

  std::vector<size_t> col_byte_offsets;
  col_byte_offsets.reserve(num_columns);

  for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
    const size_t col_width = col_slot_context.getSlotInfo(col_idx).logical_size;
    size_t group_buffer_size = num_rows_ * col_width;
    col_byte_offsets.emplace_back(total_group_by_buffer_size);
    total_group_by_buffer_size =
        align_to_int64(total_group_by_buffer_size + group_buffer_size);
  }

  int8_t* dev_buffers_allocation{nullptr};
  dev_buffers_allocation = device_allocator_->alloc(total_group_by_buffer_size);
  CHECK(dev_buffers_allocation);
  if (zero_initialize_buffers) {
    device_allocator_->zeroDeviceMem(dev_buffers_allocation, total_group_by_buffer_size);
  }

  auto dev_buffers_mem = dev_buffers_allocation;
  std::vector<int8_t*> dev_buffers(num_columns);
  for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
    dev_buffers[col_idx] = dev_buffers_allocation + col_byte_offsets[col_idx];
  }
  auto dev_ptrs = device_allocator_->alloc(num_columns * sizeof(CUdeviceptr));
  device_allocator_->copyToDevice(dev_ptrs,
                                  dev_buffers.data(),
                                  num_columns * sizeof(CUdeviceptr),
                                  "Table function input column ptrs");

  return {dev_ptrs, dev_buffers_mem, (size_t)num_rows_};
}

void QueryMemoryInitializer::copyFromTableFunctionGpuBuffers(
    DeviceAllocator* device_allocator,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t entry_count,
    const GpuGroupByBuffers& gpu_group_by_buffers,
    const int device_id,
    const unsigned block_size_x,
    const unsigned grid_size_x) {
  const size_t num_columns = query_mem_desc.getBufferColSlotCount();

  int8_t* dev_buffer = gpu_group_by_buffers.data;
  int8_t* host_buffer = reinterpret_cast<int8_t*>(group_by_buffers_[0]);

  const size_t original_entry_count = gpu_group_by_buffers.entry_count;
  CHECK_LE(entry_count, original_entry_count);
  size_t output_device_col_offset{0};
  size_t output_host_col_offset{0};

  const auto col_slot_context = query_mem_desc.getColSlotContext();

  for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
    const size_t col_width = col_slot_context.getSlotInfo(col_idx).logical_size;
    const size_t output_device_col_size = original_entry_count * col_width;
    const size_t output_host_col_size = entry_count * col_width;
    device_allocator->copyFromDevice(host_buffer + output_host_col_offset,
                                     dev_buffer + output_device_col_offset,
                                     output_host_col_size,
                                     "Table function output column buffer");
    output_device_col_offset =
        align_to_int64(output_device_col_offset + output_device_col_size);
    output_host_col_offset =
        align_to_int64(output_host_col_offset + output_host_col_size);
  }
}

size_t QueryMemoryInitializer::computeNumberOfBuffers(
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    const Executor* executor) const {
  return device_type == ExecutorDeviceType::CPU
             ? 1
             : executor->blockSize() *
                   (query_mem_desc.blocksShareMemory() ? 1 : executor->gridSize());
}

namespace {

// in-place compaction of output buffer
void compact_projection_buffer_for_cpu_columnar(
    const QueryMemoryDescriptor& query_mem_desc,
    int8_t* projection_buffer,
    const size_t projection_count) {
  // the first column (row indices) remains unchanged.
  CHECK(projection_count <= query_mem_desc.getEntryCount());
  constexpr size_t row_index_width = sizeof(int64_t);
  size_t buffer_offset1{projection_count * row_index_width};
  // other columns are actual non-lazy columns for the projection:
  for (size_t i = 0; i < query_mem_desc.getSlotCount(); i++) {
    if (query_mem_desc.getPaddedSlotWidthBytes(i) > 0) {
      auto column_proj_size =
          projection_count * query_mem_desc.getPaddedSlotWidthBytes(i);
      auto buffer_offset2 = query_mem_desc.getColOffInBytes(i);
      if (buffer_offset1 + column_proj_size >= buffer_offset2) {
        // overlapping
        std::memmove(projection_buffer + buffer_offset1,
                     projection_buffer + buffer_offset2,
                     column_proj_size);
      } else {
        std::memcpy(projection_buffer + buffer_offset1,
                    projection_buffer + buffer_offset2,
                    column_proj_size);
      }
      buffer_offset1 += align_to_int64(column_proj_size);
    }
  }
}

#ifdef HAVE_CUDA
struct RowwiseColumnPublishSpec {
  size_t column_idx{0};
  size_t source_offset{0};
  size_t source_width{0};
  size_t output_width{0};
  std::optional<int64_t> dict_entry_count;
  int64_t source_null_val{QueryMemoryDescriptor::noTranslatedGroupbyNull()};
  int64_t normalized_null_val{0};
};

bool is_supported_int_publish_width(const size_t width) {
  return width == sizeof(int8_t) || width == sizeof(int16_t) ||
         width == sizeof(int32_t) || width == sizeof(int64_t);
}

bool can_publish_width_conversion(const SQLTypeInfo& logical_ti,
                                  const size_t source_width,
                                  const size_t output_width,
                                  const bool allow_width_conversion) {
  if (source_width == output_width) {
    return true;
  }
  if (!allow_width_conversion) {
    return false;
  }
  if (!is_supported_int_publish_width(source_width) ||
      !is_supported_int_publish_width(output_width)) {
    return false;
  }
  return logical_ti.is_integer() || logical_ti.is_boolean() || logical_ti.is_time() ||
         logical_ti.is_timeinterval() || logical_ti.is_dict_encoded_string();
}

bool all_output_columns_are_group_keys(const QueryMemoryDescriptor& query_mem_desc,
                                       const size_t column_count) {
  if (query_mem_desc.targetGroupbyIndicesSize() != column_count) {
    return false;
  }
  for (size_t column_idx = 0; column_idx < column_count; ++column_idx) {
    if (query_mem_desc.getTargetGroupbyIndex(column_idx) < 0) {
      return false;
    }
  }
  return true;
}

bool publish_group_by_device_columns_from_rowwise(
    ResultSet& result_set,
    DeviceAllocator& device_allocator,
    const QueryMemoryDescriptor& query_mem_desc,
    const int8_t* rowwise_device_buffer,
    const size_t row_count,
    const std::vector<int64_t>* excluded_keys,
    const std::vector<size_t>* selected_column_indices,
    const int device_id,
    CUstream cuda_stream) {
  const auto& layout_query_mem_desc = result_set.getQueryMemDesc();
  CHECK_EQ(query_mem_desc.getQueryDescriptionType(),
           layout_query_mem_desc.getQueryDescriptionType());
  const auto query_type = layout_query_mem_desc.getQueryDescriptionType();
  const bool publish_selected_columns =
      selected_column_indices && !selected_column_indices->empty();
  const bool publish_selected_projection_columns =
      publish_selected_columns && query_type == QueryDescriptionType::Projection;
  const bool supported_query_type =
      query_type == QueryDescriptionType::GroupByBaselineHash ||
      query_type == QueryDescriptionType::GroupByPerfectHash ||
      publish_selected_projection_columns;
  const bool count_distinct_descriptors_safe =
      query_type == QueryDescriptionType::Projection ||
      layout_query_mem_desc.countDistinctDescriptorsLogicallyEmpty() ||
      all_output_columns_are_group_keys(layout_query_mem_desc, result_set.colCount());
  if (!rowwise_device_buffer || row_count == 0 || !supported_query_type ||
      layout_query_mem_desc.didOutputColumnar() ||
      (layout_query_mem_desc.hasKeylessHash() &&
       query_type != QueryDescriptionType::GroupByPerfectHash) ||
      (layout_query_mem_desc.hasVarlenOutput() && !publish_selected_projection_columns) ||
      !count_distinct_descriptors_safe || layout_query_mem_desc.getNumModeTargets() > 0) {
    return false;
  }

  const auto row_size = layout_query_mem_desc.getRowSize();
  const auto key_width = layout_query_mem_desc.getEffectiveKeyWidth();
  const auto& col_slot_context = layout_query_mem_desc.getColSlotContext();
  std::vector<bool> selected_columns(result_set.colCount(), false);
  size_t selected_column_count{0};
  if (publish_selected_columns) {
    for (const auto column_idx : *selected_column_indices) {
      if (column_idx >= result_set.colCount()) {
        return false;
      }
      if (!selected_columns[column_idx]) {
        selected_columns[column_idx] = true;
        ++selected_column_count;
      }
    }
  }
  const bool allow_width_conversion =
      all_output_columns_are_group_keys(layout_query_mem_desc, result_set.colCount());
  const bool require_all_columns = !publish_selected_columns && allow_width_conversion;
  const auto& lazy_fetch_info = result_set.getLazyFetchInfo();
  std::vector<RowwiseColumnPublishSpec> specs;
  specs.reserve(publish_selected_columns ? selected_column_indices->size()
                                         : result_set.colCount());
  for (size_t column_idx = 0; column_idx < result_set.colCount(); ++column_idx) {
    if (publish_selected_columns && !selected_columns[column_idx]) {
      continue;
    }
    if (!lazy_fetch_info.empty()) {
      CHECK_LT(column_idx, lazy_fetch_info.size());
      if (lazy_fetch_info[column_idx].is_lazily_fetched) {
        if (require_all_columns) {
          return false;
        }
        continue;
      }
    }

    const auto logical_ti = get_logical_type_info(result_set.getColType(column_idx));
    if (logical_ti.is_varlen() ||
        (logical_ti.is_string() && !logical_ti.is_dict_encoded_string())) {
      if (require_all_columns) {
        return false;
      }
      continue;
    }
    const auto elem_size = logical_ti.get_size();
    if (elem_size <= 0) {
      if (require_all_columns) {
        return false;
      }
      continue;
    }
    const auto& slots = col_slot_context.getSlotsForCol(column_idx);
    if (slots.size() != 1) {
      if (require_all_columns) {
        return false;
      }
      continue;
    }

    const auto slot_idx = slots.front();
    size_t source_offset{0};
    size_t source_width{0};
    int64_t target_groupby_idx{-1};
    const auto& target_info = result_set.getTargetInfos()[column_idx];
    if (!target_info.is_agg && layout_query_mem_desc.targetGroupbyIndicesSize() > 0) {
      CHECK_LT(column_idx, layout_query_mem_desc.targetGroupbyIndicesSize());
      target_groupby_idx = layout_query_mem_desc.getTargetGroupbyIndex(column_idx);
    }
    if (target_groupby_idx >= 0) {
      if (layout_query_mem_desc.usesGetGroupValueFast() &&
          !layout_query_mem_desc.mustUseBaselineSort()) {
        if (require_all_columns) {
          return false;
        }
        continue;
      }
      if (layout_query_mem_desc.hasKeylessHash()) {
        if (require_all_columns) {
          return false;
        }
        continue;
      }
      if (layout_query_mem_desc.getPaddedSlotWidthBytes(slot_idx) != 0) {
        if (require_all_columns) {
          return false;
        }
        continue;
      }
      if (key_width == 0 || static_cast<size_t>(target_groupby_idx) >
                                std::numeric_limits<size_t>::max() / key_width) {
        return false;
      }
      source_offset = static_cast<size_t>(target_groupby_idx) * key_width;
      source_width = key_width;
    } else {
      if (layout_query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
        if (require_all_columns) {
          return false;
        }
        continue;
      }
      const auto padded_slot_width =
          layout_query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
      if (padded_slot_width <= 0) {
        if (require_all_columns) {
          return false;
        }
        continue;
      }
      source_offset = layout_query_mem_desc.getColOffInBytes(slot_idx);
      source_width = get_rowwise_agg_payload_width(
          target_info, static_cast<size_t>(padded_slot_width));
    }
    const auto output_width = static_cast<size_t>(elem_size);
    const auto allow_column_width_conversion =
        allow_width_conversion || target_groupby_idx >= 0 || target_info.is_agg;
    const auto publish_width_supported = can_publish_width_conversion(
        logical_ti, source_width, output_width, allow_column_width_conversion);
    if (source_offset > row_size || source_width > row_size - source_offset ||
        !publish_width_supported) {
      if (require_all_columns) {
        return false;
      }
      continue;
    }
    std::optional<int64_t> dict_entry_count;
    int64_t source_null_val{QueryMemoryDescriptor::noTranslatedGroupbyNull()};
    int64_t normalized_null_val{0};
    if (!target_info.is_agg) {
      const auto translated_null_key =
          layout_query_mem_desc.getTranslatedGroupbyNullForTarget(column_idx);
      if (translated_null_key) {
        source_null_val = *translated_null_key;
        normalized_null_val = inline_fixed_encoding_null_val(logical_ti);
      }
    }
    if (logical_ti.is_dict_encoded_string()) {
      auto* const string_dict_proxy =
          result_set.getStringDictionaryProxy(logical_ti.getStringDictKey());
      CHECK(string_dict_proxy);
      dict_entry_count = static_cast<int64_t>(string_dict_proxy->storageEntryCount());
      normalized_null_val = inline_fixed_encoding_null_val(logical_ti);
    }
    specs.push_back(RowwiseColumnPublishSpec{column_idx,
                                             source_offset,
                                             source_width,
                                             output_width,
                                             dict_entry_count,
                                             source_null_val,
                                             normalized_null_val});
  }

  std::vector<int64_t> sorted_excluded_keys;
  const int64_t* device_excluded_keys{nullptr};
  size_t publish_row_count = row_count;
  const int8_t* publish_device_buffer = rowwise_device_buffer;
  if (excluded_keys && !excluded_keys->empty()) {
    if (layout_query_mem_desc.hasKeylessHash()) {
      return false;
    }
    sorted_excluded_keys = *excluded_keys;
    std::sort(sorted_excluded_keys.begin(), sorted_excluded_keys.end());
    sorted_excluded_keys.erase(
        std::unique(sorted_excluded_keys.begin(), sorted_excluded_keys.end()),
        sorted_excluded_keys.end());
    if (sorted_excluded_keys.size() >
        std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
      return false;
    }
    const auto excluded_keys_bytes = sorted_excluded_keys.size() * sizeof(int64_t);
    auto* mutable_device_excluded_keys =
        reinterpret_cast<int64_t*>(device_allocator.alloc(excluded_keys_bytes));
    device_allocator.copyToDevice(
        reinterpret_cast<int8_t*>(mutable_device_excluded_keys),
        reinterpret_cast<const int8_t*>(sorted_excluded_keys.data()),
        excluded_keys_bytes,
        "Compacted baseline hash excluded keys");
    device_excluded_keys = mutable_device_excluded_keys;
    auto* row_count_device =
        reinterpret_cast<uint64_t*>(device_allocator.alloc(sizeof(uint64_t)));
    publish_row_count =
        count_baseline_hash_rows_excluding_keys_on_device(rowwise_device_buffer,
                                                          row_count,
                                                          row_size,
                                                          key_width,
                                                          device_excluded_keys,
                                                          sorted_excluded_keys.size(),
                                                          row_count_device,
                                                          device_id,
                                                          cuda_stream);
    if (publish_row_count == 0 || publish_row_count > row_count ||
        publish_row_count > std::numeric_limits<size_t>::max() / row_size) {
      return false;
    }
    auto* filtered_device_buffer = device_allocator.alloc(publish_row_count * row_size);
    auto* filtered_row_count =
        reinterpret_cast<uint64_t*>(device_allocator.alloc(sizeof(uint64_t)));
    compact_baseline_hash_rows_excluding_keys_on_device(rowwise_device_buffer,
                                                        filtered_device_buffer,
                                                        filtered_row_count,
                                                        row_count,
                                                        row_size,
                                                        key_width,
                                                        device_excluded_keys,
                                                        sorted_excluded_keys.size(),
                                                        device_id,
                                                        cuda_stream);
    uint64_t verified_row_count{0};
    device_allocator.copyFromDevice(&verified_row_count,
                                    filtered_row_count,
                                    sizeof(verified_row_count),
                                    "Compacted baseline hash excluded row count");
    CHECK_EQ(publish_row_count, static_cast<size_t>(verified_row_count));
    publish_device_buffer = filtered_device_buffer;
  }

  size_t published_columns{0};

  for (const auto& spec : specs) {
    if (publish_row_count > std::numeric_limits<size_t>::max() / spec.output_width) {
      return false;
    }
    const auto column_bytes = publish_row_count * spec.output_width;
    auto* column_buffer = device_allocator.alloc(column_bytes);
    extract_fixed_width_column_from_rows_on_device(publish_device_buffer,
                                                   column_buffer,
                                                   publish_row_count,
                                                   row_size,
                                                   spec.source_offset,
                                                   spec.source_width,
                                                   spec.output_width,
                                                   spec.dict_entry_count.value_or(-1),
                                                   spec.source_null_val,
                                                   spec.normalized_null_val,
                                                   device_id,
                                                   cuda_stream);
    result_set.addDeviceColumnarBufferFragment(
        spec.column_idx, device_id, column_buffer, publish_row_count);
    ++published_columns;
  }

  return published_columns > 0 &&
         (!publish_selected_columns || published_columns == selected_column_count);
}
#endif

}  // namespace

void QueryMemoryInitializer::compactProjectionBuffersCpu(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t projection_count) {
  const auto num_allocated_rows =
      std::min(projection_count, query_mem_desc.getEntryCount());
  const size_t buffer_start_idx = query_mem_desc.hasVarlenOutput() ? 1 : 0;

  // copy the results from the main buffer into projection_buffer
  compact_projection_buffer_for_cpu_columnar(
      query_mem_desc,
      reinterpret_cast<int8_t*>(group_by_buffers_[buffer_start_idx]),
      num_allocated_rows);

  // update the entry count for the result set, and its underlying storage
  CHECK(!result_sets_.empty());
  result_sets_.front()->updateStorageEntryCount(num_allocated_rows);
}

void QueryMemoryInitializer::compactProjectionBuffersGpu(
    const QueryMemoryDescriptor& query_mem_desc,
    DeviceAllocator* device_allocator,
    const GpuGroupByBuffers& gpu_group_by_buffers,
    const size_t projection_count,
    const int device_id,
    const bool defer_cpu_materialization) {
  // store total number of allocated rows:
  const auto num_allocated_rows =
      std::min(projection_count, query_mem_desc.getEntryCount());

  const size_t buffer_start_idx = query_mem_desc.hasVarlenOutput() ? 1 : 0;

  CHECK(!result_sets_.empty());
  auto* const result_set = result_sets_.front().get();
  result_set->updateStorageEntryCount(num_allocated_rows);
  result_set->setCachedRowCount(num_allocated_rows);
  if (defer_cpu_materialization && num_allocated_rows > 0) {
    const auto col_slot_context = query_mem_desc.getColSlotContext();
    for (size_t column_idx = 0; column_idx < result_set->colCount(); ++column_idx) {
      const auto logical_ti = get_logical_type_info(result_set->getColType(column_idx));
      if (logical_ti.is_varlen()) {
        continue;
      }
      const auto elem_size = logical_ti.get_size();
      if (elem_size <= 0) {
        continue;
      }
      const auto& slots = col_slot_context.getSlotsForCol(column_idx);
      if (slots.size() != 1) {
        continue;
      }
      const auto slot_idx = slots.front();
      if (query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
        continue;
      }
      if (query_mem_desc.getPaddedSlotWidthBytes(slot_idx) != elem_size) {
        continue;
      }
      result_set->addDeviceColumnarBufferFragment(
          column_idx,
          device_id,
          gpu_group_by_buffers.data + query_mem_desc.getColOffInBytes(slot_idx),
          num_allocated_rows);
    }
  }

  const bool can_defer_cpu_materialization =
      defer_cpu_materialization && result_set->canDeferDeviceColumnarCpuMaterialization();
  if (can_defer_cpu_materialization) {
    result_set->markDeviceColumnarCpuStorageInvalid();
    return;
  }
  result_set->clearDeviceColumnarBufferFragments();

  // copy the results from the main buffer into projection_buffer
  copy_projection_buffer_from_gpu_columnar(
      device_allocator,
      gpu_group_by_buffers,
      query_mem_desc,
      reinterpret_cast<int8_t*>(group_by_buffers_[buffer_start_idx]),
      num_allocated_rows,
      device_id);
}

void QueryMemoryInitializer::copyGroupByBuffersFromGpu(
    DeviceAllocator& device_allocator,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t entry_count,
    const GpuGroupByBuffers& gpu_group_by_buffers,
    const RelAlgExecutionUnit* ra_exe_unit,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    CUstream cuda_stream,
    const bool prepend_index_buffer,
    const bool defer_cpu_materialization) {
#ifdef HAVE_CUDA
  const auto thread_count = block_size_x * grid_size_x;

  const bool has_exact_projection_row_count =
      ra_exe_unit &&
      (ra_exe_unit->use_bump_allocator ||
       (ra_exe_unit->per_device_cardinality.size() == size_t(1) &&
        ra_exe_unit->per_device_cardinality.front().second == entry_count));
  const bool can_defer_dense_rowwise_projection =
      defer_cpu_materialization && g_enable_result_reduction_pipeline && ra_exe_unit &&
      has_exact_projection_row_count && entry_count > 0 &&
      entry_count <= query_mem_desc.getEntryCount() &&
      query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection &&
      !query_mem_desc.didOutputColumnar() && !query_mem_desc.hasVarlenOutput() &&
      !prepend_index_buffer && (query_mem_desc.blocksShareMemory() || grid_size_x == 1) &&
      !ra_exe_unit->device_resident_output_column_indices.empty() &&
      !result_sets_.empty();
  if (can_defer_dense_rowwise_projection) {
    auto& result_set = *result_sets_.front();
    result_set.updateStorageEntryCount(entry_count);
    result_set.setCachedRowCount(entry_count);
    result_set.clearDeviceColumnarBufferFragments();
    result_set.clearDeviceRowwiseBufferFragments();
    const auto published = publish_group_by_device_columns_from_rowwise(
        result_set,
        device_allocator,
        query_mem_desc,
        gpu_group_by_buffers.data,
        entry_count,
        nullptr,
        &ra_exe_unit->device_resident_output_column_indices,
        device_id,
        cuda_stream);
    if (published) {
      result_set.markDeviceColumnarFragmentsCoverLogicalRows();
      result_set.addDeviceRowwiseBufferFragment(
          device_id, gpu_group_by_buffers.data, entry_count);
      if (result_set.canDeferDeviceColumnarCpuMaterialization()) {
        result_set.markDeviceColumnarCpuStorageInvalid();
        return;
      }
      result_set.clearDeviceRowwiseBufferFragments();
      result_set.markDeviceColumnarCpuStorageValid();
    }
    result_set.clearDeviceColumnarBufferFragments();
  }

  size_t total_buff_size{0};
  if (ra_exe_unit && query_mem_desc.useStreamingTopN()) {
    const size_t n =
        ra_exe_unit->sort_info.offset + ra_exe_unit->sort_info.limit.value_or(0);
    total_buff_size =
        streaming_top_n::get_heap_size(query_mem_desc.getRowSize(), n, thread_count);
  } else {
    total_buff_size =
        query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU, entry_count);
  }
  const ResultSetEntryFilter* entry_filter{nullptr};
  const std::vector<int64_t>* preserved_keys{nullptr};
  const std::vector<TargetInfo>* target_infos{nullptr};
  if (!result_sets_.empty()) {
    target_infos = &result_sets_.front()->getTargetInfos();
  }
  if (ra_exe_unit && !ra_exe_unit->deferred_sparse_baseline_preserved_keys.empty()) {
    preserved_keys = &ra_exe_unit->deferred_sparse_baseline_preserved_keys;
  }
  if (ra_exe_unit && ra_exe_unit->apply_deferred_sparse_baseline_filter_before_copy &&
      ra_exe_unit->deferred_sparse_baseline_filter) {
    entry_filter = &*ra_exe_unit->deferred_sparse_baseline_filter;
  }
  // A completed host copy and its device rows are equivalent immutable views. Retain
  // the device view for a downstream GPU reducer; CPU reducers clear it before
  // mutating the host storage.
  const bool reducer_compatible_perfect_hash =
      g_enable_result_reduction_pipeline && ra_exe_unit && target_infos &&
      can_retain_perfect_hash_rowwise_for_gpu_reduction(
          query_mem_desc, *target_infos, init_agg_vals_, entry_count);
  const bool can_retain_perfect_hash_rowwise =
      query_mem_desc.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash &&
      !query_mem_desc.didOutputColumnar() && !query_mem_desc.hasVarlenOutput() &&
      (query_mem_desc.blocksShareMemory() || grid_size_x == 1) && !prepend_index_buffer &&
      !result_sets_.empty() &&
      (can_defer_keyless_perfect_hash_rowwise_result(query_mem_desc, entry_count) ||
       reducer_compatible_perfect_hash);
  const bool skip_perfect_hash_host_copy =
      defer_gpu_baseline_hash_host_storage_ && can_retain_perfect_hash_rowwise;
  int8_t* compacted_device_buffer{nullptr};
  bool host_copy_performed{false};
  bool entry_filter_applied{false};
  const auto compacted_row_count =
      copy_group_by_buffers_from_gpu(device_allocator,
                                     group_by_buffers_,
                                     total_buff_size,
                                     gpu_group_by_buffers.data,
                                     query_mem_desc,
                                     block_size_x,
                                     grid_size_x,
                                     device_id,
                                     cuda_stream,
                                     prepend_index_buffer,
                                     query_mem_desc.hasVarlenOutput(),
                                     target_infos,
                                     &init_agg_vals_,
                                     &compacted_device_buffer,
                                     defer_gpu_baseline_hash_host_storage_,
                                     skip_perfect_hash_host_copy,
                                     &host_copy_performed,
                                     entry_filter,
                                     preserved_keys,
                                     &entry_filter_applied);
  if (entry_filter_applied) {
    CHECK(!result_sets_.empty());
    result_sets_.front()->markSparseBaselineEntryFilterAppliedBeforeCopy();
  }
  if (host_copy_performed && entry_count > 0 && ra_exe_unit &&
      query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection &&
      !ra_exe_unit->device_resident_output_column_indices.empty()) {
    CHECK(!result_sets_.empty());
    auto& result_set = *result_sets_.front();
    // Rowwise projection storage is a dense prefix. Publish only that exact prefix;
    // per-device allocation tails must never become visible to downstream consumers.
    const auto logical_row_count = result_set.rowCount();
    result_set.clearDeviceColumnarBufferFragments();
    const auto published =
        logical_row_count > 0 && publish_group_by_device_columns_from_rowwise(
                                     result_set,
                                     device_allocator,
                                     query_mem_desc,
                                     gpu_group_by_buffers.data,
                                     logical_row_count,
                                     nullptr,
                                     &ra_exe_unit->device_resident_output_column_indices,
                                     device_id,
                                     cuda_stream);
    if (published) {
      result_set.markDeviceColumnarFragmentsCoverLogicalRows();
    }
  }
  if (!compacted_row_count && defer_gpu_baseline_hash_host_storage_ &&
      !can_retain_perfect_hash_rowwise && !host_copy_performed) {
    throw OutOfHostMemory(total_buff_size);
  }
  if (compacted_row_count) {
    const auto row_count = *compacted_row_count;
    CHECK(!result_sets_.empty());
    result_sets_.front()->updateStorageEntryCount(row_count);
    result_sets_.front()->setCachedRowCount(row_count);
    result_sets_.front()->markBaselineHashDenseForReduction(row_count);
    result_sets_.front()->clearDeviceColumnarBufferFragments();
    result_sets_.front()->clearDeviceRowwiseBufferFragments();
    bool published_column_fragments = false;
    if (row_count > 0 && defer_gpu_baseline_hash_host_storage_) {
      CHECK(compacted_device_buffer);
      result_sets_.front()->addDeviceRowwiseBufferFragment(
          device_id, compacted_device_buffer, row_count);
      published_column_fragments =
          publish_group_by_device_columns_from_rowwise(*result_sets_.front(),
                                                       device_allocator,
                                                       query_mem_desc,
                                                       compacted_device_buffer,
                                                       row_count,
                                                       preserved_keys,
                                                       nullptr,
                                                       device_id,
                                                       cuda_stream);
      if (published_column_fragments && preserved_keys && !preserved_keys->empty()) {
        result_sets_.front()->markDeviceColumnarFragmentsExcludeBaselineBoundaryKeys();
      }
      if (published_column_fragments) {
        result_sets_.front()->markDeviceColumnarFragmentsCoverLogicalRows();
      }
    }
    if (row_count > 0 && defer_gpu_baseline_hash_host_storage_) {
      if (!result_sets_.front()->canDeferDeviceColumnarCpuMaterialization()) {
        throw OutOfHostMemory(total_buff_size);
      }
      result_sets_.front()->markDeviceColumnarCpuStorageInvalid();
    }
  } else if (can_retain_perfect_hash_rowwise) {
    result_sets_.front()->addDeviceRowwiseBufferFragment(
        device_id, gpu_group_by_buffers.data, entry_count);
    if (host_copy_performed && !defer_gpu_baseline_hash_host_storage_) {
      result_sets_.front()->markDeviceColumnarCpuStorageValid();
    }
    result_sets_.front()->setCachedRowCount(entry_count);
    bool published_column_fragments = false;
    if (entry_count > 0 && defer_gpu_baseline_hash_host_storage_) {
      published_column_fragments =
          publish_group_by_device_columns_from_rowwise(*result_sets_.front(),
                                                       device_allocator,
                                                       query_mem_desc,
                                                       gpu_group_by_buffers.data,
                                                       entry_count,
                                                       nullptr,
                                                       nullptr,
                                                       device_id,
                                                       cuda_stream);
    }
    if (published_column_fragments && defer_gpu_baseline_hash_host_storage_) {
      if (!result_sets_.front()->canDeferDeviceColumnarCpuMaterialization()) {
        throw OutOfHostMemory(total_buff_size);
      }
      result_sets_.front()->markDeviceColumnarCpuStorageInvalid();
    }
  }
#else
  static_cast<void>(device_allocator);
  static_cast<void>(query_mem_desc);
  static_cast<void>(entry_count);
  static_cast<void>(gpu_group_by_buffers);
  static_cast<void>(ra_exe_unit);
  static_cast<void>(block_size_x);
  static_cast<void>(grid_size_x);
  static_cast<void>(device_id);
  static_cast<void>(cuda_stream);
  static_cast<void>(prepend_index_buffer);
  static_cast<void>(defer_cpu_materialization);
  UNREACHABLE();
#endif
}

void QueryMemoryInitializer::copyFromDeviceForAggMode() {
#ifdef HAVE_CUDA
  CHECK_EQ(agg_mode_hash_tables_cpu_.size(), agg_mode_hash_tables_gpu_.size());
  try {
    for (size_t i = 0; i < agg_mode_hash_tables_gpu_.size(); ++i) {
      *agg_mode_hash_tables_cpu_[i] = agg_mode_hash_tables_gpu_.moveToHost(i);
    }
  } catch (std::runtime_error const& ex) {
    LOG(INFO) << ex.what();  // E.g. too many keys for fixed gpu hash table
    throw QueryMustRunOnCpu();
  }
#else
  UNREACHABLE();
#endif
}

void QueryMemoryInitializer::applyStreamingTopNOffsetCpu(
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit) {
  const size_t buffer_start_idx = query_mem_desc.hasVarlenOutput() ? 1 : 0;
  CHECK_EQ(group_by_buffers_.size(), buffer_start_idx + 1);

  const auto rows_copy = streaming_top_n::get_rows_copy_from_heaps(
      group_by_buffers_[buffer_start_idx],
      query_mem_desc.getBufferSizeBytes(ra_exe_unit, 1, ExecutorDeviceType::CPU),
      ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit.value_or(0),
      1);
  CHECK_EQ(rows_copy.size(),
           query_mem_desc.getEntryCount() * query_mem_desc.getRowSize());
  memcpy(group_by_buffers_[buffer_start_idx], &rows_copy[0], rows_copy.size());
}

void QueryMemoryInitializer::applyStreamingTopNOffsetGpu(
    Data_Namespace::DataMgr* data_mgr,
    CudaAllocator* cuda_allocator,
    const QueryMemoryDescriptor& query_mem_desc,
    const GpuGroupByBuffers& gpu_group_by_buffers,
    const RelAlgExecutionUnit& ra_exe_unit,
    const unsigned total_thread_count,
    const int device_id,
    CUstream cuda_stream) {
#ifdef HAVE_CUDA
  CHECK(cuda_allocator);
  CHECK_EQ(group_by_buffers_.size(), num_buffers_);
  const size_t buffer_start_idx = query_mem_desc.hasVarlenOutput() ? 1 : 0;

  const auto rows_copy = pick_top_n_rows_from_dev_heaps(
      data_mgr,
      cuda_allocator,
      reinterpret_cast<int64_t*>(gpu_group_by_buffers.data),
      ra_exe_unit,
      query_mem_desc,
      total_thread_count,
      device_id,
      cuda_stream);
  CHECK_EQ(
      rows_copy.size(),
      static_cast<size_t>(query_mem_desc.getEntryCount() * query_mem_desc.getRowSize()));
  memcpy(group_by_buffers_[buffer_start_idx], &rows_copy[0], rows_copy.size());
#else
  UNREACHABLE();
#endif
}

std::vector<int8_t> QueryMemoryInitializer::getAggModeHashTablesGpu() const {
#ifdef HAVE_CUDA
  return agg_mode_hash_tables_gpu_.serialize();
#else
  UNREACHABLE();
  return {};
#endif
}

std::shared_ptr<VarlenOutputInfo> QueryMemoryInitializer::getVarlenOutputInfo() {
  if (varlen_output_info_) {
    return varlen_output_info_;
  }

  // shared_ptr so that both the ResultSet and QMI can hold on to the varlen info object
  // and update it as needed
  varlen_output_info_ = std::make_shared<VarlenOutputInfo>(VarlenOutputInfo{
      static_cast<int64_t>(varlen_output_buffer_), varlen_output_buffer_host_ptr_, 0});
  return varlen_output_info_;
}
