/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA

#include "DataMgr/Allocators/DeviceAllocator.h"
#include "GpuInitGroups.h"
#include "GpuMemUtils.h"
#include "Logger/Logger.h"
#include "ResultSetBufferAccessors.h"
#include "ResultSetStorage.h"
#include "StreamingTopN.h"

#include "../CudaMgr/CudaMgr.h"
#include "GroupByAndAggregate.h"
#include "Shared/SqlTypesLayout.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

extern size_t g_max_memory_allocation_size;
extern size_t g_min_memory_allocation_size;
extern double g_bump_allocator_step_reduction;

void copy_to_nvidia_gpu(Data_Namespace::DataMgr* data_mgr,
                        CUstream cuda_stream,
                        CUdeviceptr dst,
                        const void* src,
                        const size_t num_bytes,
                        const int device_id,
                        std::string_view tag) {
#ifdef HAVE_CUDA
  if (!data_mgr) {  // only for unit tests
    checkCudaErrors(cuMemcpyHtoDAsync(dst, src, num_bytes, cuda_stream));
    checkCudaErrors(cuStreamSynchronize(cuda_stream));
    return;
  }
  const auto cuda_mgr = data_mgr->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->copyHostToDevice(reinterpret_cast<int8_t*>(dst),
                             static_cast<const int8_t*>(src),
                             num_bytes,
                             device_id,
                             tag,
                             cuda_stream);
#else
  CHECK(false);
#endif  // HAVE_CUDA
}

namespace {

size_t checked_size_add(const size_t lhs,
                        const size_t rhs,
                        const char* const description) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    throw std::overflow_error(description);
  }
  return lhs + rhs;
}

size_t checked_size_multiply(const size_t lhs,
                             const size_t rhs,
                             const char* const description) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(description);
  }
  return lhs * rhs;
}

size_t checked_align_to_int64(const size_t value, const char* const description) {
  const auto remainder = value % sizeof(int64_t);
  return remainder == 0
             ? value
             : checked_size_add(value, sizeof(int64_t) - remainder, description);
}

inline size_t coalesced_size(const QueryMemoryDescriptor& query_mem_desc,
                             const size_t group_by_one_buffer_size,
                             const unsigned grid_size_x) {
  CHECK(query_mem_desc.threadsShareMemory());
  return checked_size_multiply(static_cast<size_t>(grid_size_x),
                               group_by_one_buffer_size,
                               "Coalesced group-by buffer size overflow");
}

#ifdef HAVE_CUDA
size_t rowwise_agg_payload_width(const TargetInfo& target_info,
                                 const size_t padded_slot_width) {
  CHECK_GT(padded_slot_width, size_t(0));
  if (takes_float_argument(target_info) && target_info.agg_kind != kAVG) {
    CHECK_GE(padded_slot_width, sizeof(float));
    return sizeof(float);
  }
  return padded_slot_width;
}

std::optional<int64_t> checked_scale_decimal_value(const int64_t value,
                                                   const unsigned scale) {
  const auto unsigned_factor = exp_to_scale(scale);
  if (unsigned_factor > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  int64_t scaled_value{0};
  if (__builtin_mul_overflow(
          value, static_cast<int64_t>(unsigned_factor), &scaled_value)) {
    return std::nullopt;
  }
  return scaled_value;
}

std::optional<int64_t> entry_filter_literal_as_integral_value(
    const ResultSetEntryLiteral& literal,
    const SQLTypeInfo& target_type) {
  if (literal.is_null || literal.type_info.is_fp() || target_type.is_fp()) {
    return std::nullopt;
  }
  if (literal.type_info.is_decimal()) {
    if (target_type.is_decimal()) {
      return convert_decimal_value_to_scale(
          literal.int_val, literal.type_info, target_type);
    }
    return literal.type_info.get_scale() == 0 ? std::optional<int64_t>(literal.int_val)
                                              : std::nullopt;
  }
  if (target_type.is_decimal()) {
    return checked_scale_decimal_value(literal.int_val, target_type.get_scale());
  }
  if (literal.type_info.get_type() == kBOOLEAN) {
    return literal.bool_val ? 1 : 0;
  }
  return literal.int_val;
}

std::optional<double> entry_filter_literal_as_fp_value(
    const ResultSetEntryLiteral& literal,
    const SQLTypeInfo& target_type) {
  if (literal.is_null) {
    return std::nullopt;
  }
  if (literal.type_info.is_fp()) {
    return literal.double_val;
  }
  if (literal.type_info.is_decimal()) {
    return literal.int_val /
           static_cast<double>(exp_to_scale(literal.type_info.get_scale()));
  }
  if (literal.type_info.get_type() == kBOOLEAN) {
    return literal.bool_val ? 1.0 : 0.0;
  }
  if (target_type.is_decimal()) {
    return literal.int_val * static_cast<double>(exp_to_scale(target_type.get_scale()));
  }
  return static_cast<double>(literal.int_val);
}

std::optional<DeviceResultSetEntryComparison> make_device_entry_comparison(
    const ResultSetEntryComparison& comparison,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<TargetInfo>& targets) {
  if (comparison.target_idx >= targets.size()) {
    return std::nullopt;
  }
  const auto& target_info = targets[comparison.target_idx];
  if (target_info.agg_kind == kAVG || target_info.sql_type.is_varlen() ||
      target_info.sql_type.is_array() || target_info.sql_type.is_geometry()) {
    return std::nullopt;
  }

  DeviceResultSetEntryComparison device_comparison;
  if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
    const auto groupby_idx = query_mem_desc.getTargetGroupbyIndex(comparison.target_idx);
    if (groupby_idx >= 0) {
      if ((query_mem_desc.usesGetGroupValueFast() &&
           !query_mem_desc.mustUseBaselineSort()) ||
          query_mem_desc.hasKeylessHash()) {
        return std::nullopt;
      }
      device_comparison.target_width =
          static_cast<uint8_t>(query_mem_desc.getEffectiveKeyWidth());
      device_comparison.target_offset =
          static_cast<uint64_t>(groupby_idx) * device_comparison.target_width;
    }
  }
  if (!device_comparison.target_width) {
    const auto& slots =
        query_mem_desc.getColSlotContext().getSlotsForCol(comparison.target_idx);
    if (slots.size() != size_t(1)) {
      return std::nullopt;
    }
    const auto slot_idx = slots.front();
    if (query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
      return std::nullopt;
    }
    const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
    if (slot_width <= 0) {
      return std::nullopt;
    }
    const auto payload_width =
        rowwise_agg_payload_width(target_info, static_cast<size_t>(slot_width));
    if (payload_width != sizeof(int8_t) && payload_width != sizeof(int16_t) &&
        payload_width != sizeof(int32_t) && payload_width != sizeof(int64_t)) {
      return std::nullopt;
    }
    const auto target_offset = query_mem_desc.getColOffInBytes(slot_idx);
    if (target_offset > query_mem_desc.getRowSize() ||
        payload_width > query_mem_desc.getRowSize() - target_offset) {
      return std::nullopt;
    }
    device_comparison.target_width = static_cast<uint8_t>(payload_width);
    device_comparison.target_offset = target_offset;
  }

  const auto& target_type = target_info.sql_type;
  device_comparison.op = comparison.op;
  device_comparison.nullable = !target_type.get_notnull();
  device_comparison.null_bits =
      null_val_bit_pattern(target_type, takes_float_argument(target_info));
  device_comparison.is_fp = target_type.is_fp();
  device_comparison.is_float =
      target_type.is_fp() && (target_type.get_type() == kFLOAT ||
                              device_comparison.target_width == sizeof(int32_t));
  if (device_comparison.is_fp) {
    const auto literal =
        entry_filter_literal_as_fp_value(comparison.literal, target_type);
    if (!literal) {
      return std::nullopt;
    }
    device_comparison.fp_literal = *literal;
  } else {
    const auto literal =
        entry_filter_literal_as_integral_value(comparison.literal, target_type);
    if (!literal) {
      return std::nullopt;
    }
    device_comparison.int_literal = *literal;
  }
  return device_comparison;
}

std::optional<std::vector<DeviceResultSetEntryComparison>> make_device_entry_filter(
    const ResultSetEntryFilter& entry_filter,
    const QueryMemoryDescriptor& query_mem_desc,
    const std::vector<TargetInfo>& targets) {
  if (entry_filter.empty()) {
    return std::nullopt;
  }
  std::vector<DeviceResultSetEntryComparison> comparisons;
  comparisons.reserve(entry_filter.size());
  for (const auto& comparison : entry_filter) {
    auto device_comparison =
        make_device_entry_comparison(comparison, query_mem_desc, targets);
    if (!device_comparison) {
      return std::nullopt;
    }
    comparisons.push_back(*device_comparison);
  }
  return comparisons;
}

constexpr size_t kBaselineGpuReductionEntryCountMultiplier = 2;

std::optional<size_t> baseline_reduction_entry_count_for_gpu(
    const size_t source_entry_count) {
  if (source_entry_count == 0 ||
      source_entry_count > std::numeric_limits<size_t>::max() /
                               kBaselineGpuReductionEntryCountMultiplier) {
    return std::nullopt;
  }
  return source_entry_count * kBaselineGpuReductionEntryCountMultiplier;
}

std::optional<DeviceBaselineHashReductionSlot::Op> baseline_gpu_reduction_op(
    const TargetInfo& target_info) {
  if (target_info.is_distinct || target_info.sql_type.is_array() ||
      target_info.sql_type.is_geometry() || target_info.sql_type.is_varlen()) {
    return std::nullopt;
  }
  if (!target_info.is_agg) {
    return std::nullopt;
  }
  switch (target_info.agg_kind) {
    case kCOUNT:
    case kCOUNT_IF:
    case kSUM:
    case kSUM_IF:
    case kAVG:
      return DeviceBaselineHashReductionSlot::Sum;
    case kMIN:
      if (target_info.sql_type.is_fp() || takes_float_argument(target_info)) {
        return std::nullopt;
      }
      return DeviceBaselineHashReductionSlot::Min;
    case kMAX:
      if (target_info.sql_type.is_fp() || takes_float_argument(target_info)) {
        return std::nullopt;
      }
      return DeviceBaselineHashReductionSlot::Max;
    default:
      return std::nullopt;
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

bool count_distinct_descriptors_safe_for_group_key_output(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t target_count) {
  if (query_mem_desc.countDistinctDescriptorsLogicallyEmpty()) {
    return true;
  }
  if (target_count == 0 || query_mem_desc.targetGroupbyIndicesSize() != target_count) {
    return false;
  }
  for (size_t target_idx = 0; target_idx < target_count; ++target_idx) {
    if (query_mem_desc.getTargetGroupbyIndex(target_idx) < 0) {
      return false;
    }
  }
  return true;
}

std::optional<std::vector<DeviceBaselineHashReductionSlot>>
make_baseline_gpu_reduction_slots(const QueryMemoryDescriptor& query_mem_desc,
                                  const std::vector<TargetInfo>& targets,
                                  const std::vector<int64_t>& init_vals) {
  if (query_mem_desc.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByBaselineHash ||
      query_mem_desc.didOutputColumnar() || query_mem_desc.hasKeylessHash() ||
      query_mem_desc.hasVarlenOutput() ||
      !count_distinct_descriptors_safe_for_group_key_output(query_mem_desc,
                                                            targets.size()) ||
      query_mem_desc.getNumModeTargets() > 0) {
    return std::nullopt;
  }
  if (query_mem_desc.getEffectiveKeyWidth() != size_t(4) &&
      query_mem_desc.getEffectiveKeyWidth() != size_t(8)) {
    return std::nullopt;
  }
  if (query_mem_desc.getRowSize() == 0 ||
      query_mem_desc.getRowSize() % sizeof(int64_t) != 0) {
    return std::nullopt;
  }

  std::vector<DeviceBaselineHashReductionSlot> slots;
  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
      CHECK_LT(target_idx, query_mem_desc.targetGroupbyIndicesSize());
      if (query_mem_desc.getTargetGroupbyIndex(target_idx) >= 0) {
        continue;
      }
    }
    const auto& target_info = targets[target_idx];
    const auto op = baseline_gpu_reduction_op(target_info);
    if (!op) {
      return std::nullopt;
    }
    const auto& col_slots = query_mem_desc.getColSlotContext().getSlotsForCol(target_idx);
    const auto expected_slot_count = target_info.agg_kind == kAVG ? size_t(2) : size_t(1);
    if (col_slots.size() != expected_slot_count) {
      return std::nullopt;
    }
    for (size_t target_slot_idx = 0; target_slot_idx < col_slots.size();
         ++target_slot_idx) {
      const auto slot_idx = col_slots[target_slot_idx];
      if (query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
        return std::nullopt;
      }
      const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
      if (slot_width != sizeof(int32_t) && slot_width != sizeof(int64_t)) {
        return std::nullopt;
      }
      const auto payload_width =
          rowwise_agg_payload_width(target_info, static_cast<size_t>(slot_width));
      if (payload_width != sizeof(int32_t) && payload_width != sizeof(int64_t)) {
        return std::nullopt;
      }
      const auto init_val_idx = target_init_val_index_for_slot(query_mem_desc, slot_idx);
      if (!init_val_idx || *init_val_idx >= init_vals.size()) {
        return std::nullopt;
      }
      const auto slot_offset = query_mem_desc.getColOffInBytes(slot_idx);
      if (slot_offset > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
      }
      const bool avg_count_slot =
          target_info.agg_kind == kAVG && target_slot_idx == size_t(1);
      slots.push_back(DeviceBaselineHashReductionSlot{
          static_cast<uint32_t>(slot_offset),
          init_vals[*init_val_idx],
          static_cast<uint8_t>(payload_width),
          static_cast<uint8_t>(*op),
          avg_count_slot ? false : target_info.skip_null_val,
          avg_count_slot
              ? false
              : target_info.sql_type.is_fp() || takes_float_argument(target_info)});
    }
  }
  return slots;
}
#endif

std::optional<size_t> reduce_and_compact_baseline_group_by_buffers_from_gpu(
    DeviceAllocator& device_allocator,
    const std::vector<int64_t*>& group_by_buffers,
    const size_t groups_buffer_size,
    const int8_t* group_by_dev_buffers_mem,
    const QueryMemoryDescriptor& query_mem_desc,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    CUstream cuda_stream,
    const unsigned block_buffer_count,
    const size_t first_group_buffer_idx,
    const std::vector<TargetInfo>* target_infos,
    const std::vector<int64_t>* init_vals,
    int8_t** compacted_device_buffer,
    const bool skip_host_copy_for_compacted) {
#ifdef HAVE_CUDA
  if (!target_infos || !init_vals || block_buffer_count <= 1) {
    return std::nullopt;
  }
  const auto slots =
      make_baseline_gpu_reduction_slots(query_mem_desc, *target_infos, *init_vals);
  if (!slots) {
    return std::nullopt;
  }

  const auto source_entry_count = query_mem_desc.getEntryCount();
  if (source_entry_count == 0) {
    return std::nullopt;
  }
  const auto row_size = query_mem_desc.getRowSize();
  const auto key_width = query_mem_desc.getEffectiveKeyWidth();
  if (block_buffer_count > std::numeric_limits<size_t>::max() / source_entry_count) {
    return std::nullopt;
  }
  const auto total_sparse_entries = source_entry_count * block_buffer_count;

  const auto destination_entry_count_opt =
      baseline_reduction_entry_count_for_gpu(total_sparse_entries);
  if (!destination_entry_count_opt) {
    return std::nullopt;
  }
  const auto destination_entry_count = *destination_entry_count_opt;
  if (destination_entry_count > std::numeric_limits<size_t>::max() / row_size) {
    return std::nullopt;
  }
  const auto destination_bytes = destination_entry_count * row_size;
  if (destination_bytes > g_max_memory_allocation_size) {
    return std::nullopt;
  }
  auto* destination_buffer = device_allocator.alloc(destination_bytes);

  int64_t* init_vals_device{nullptr};
  if (!init_vals->empty()) {
    if (init_vals->size() > std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
      return std::nullopt;
    }
    const auto init_vals_bytes = init_vals->size() * sizeof(int64_t);
    init_vals_device =
        reinterpret_cast<int64_t*>(device_allocator.alloc(init_vals_bytes));
    device_allocator.copyToDevice(init_vals_device,
                                  init_vals->data(),
                                  init_vals_bytes,
                                  "GPU baseline hash multi-buffer init values");
  }
  init_group_by_buffer_on_device(reinterpret_cast<int64_t*>(destination_buffer),
                                 init_vals_device,
                                 destination_entry_count,
                                 query_mem_desc.getGroupbyColCount(),
                                 query_mem_desc.getEffectiveKeyWidth(),
                                 query_mem_desc.getRowSize() / sizeof(int64_t),
                                 query_mem_desc.hasKeylessHash(),
                                 1,
                                 block_size_x,
                                 grid_size_x,
                                 cuda_stream);

  DeviceBaselineHashReductionSlot* slots_device{nullptr};
  if (!slots->empty()) {
    if (slots->size() >
        std::numeric_limits<size_t>::max() / sizeof(DeviceBaselineHashReductionSlot)) {
      return std::nullopt;
    }
    const auto slots_bytes = slots->size() * sizeof(DeviceBaselineHashReductionSlot);
    slots_device = reinterpret_cast<DeviceBaselineHashReductionSlot*>(
        device_allocator.alloc(slots_bytes));
    device_allocator.copyToDevice(slots_device,
                                  slots->data(),
                                  slots_bytes,
                                  "GPU baseline hash multi-buffer reduction slots");
  }
  auto* reduction_scratch =
      reinterpret_cast<uint64_t*>(device_allocator.alloc(sizeof(uint64_t)));

  const bool reduction_success =
      reduce_baseline_hash_buffers_on_device(destination_buffer,
                                             destination_entry_count,
                                             group_by_dev_buffers_mem,
                                             source_entry_count,
                                             groups_buffer_size,
                                             block_buffer_count,
                                             row_size,
                                             key_width,
                                             query_mem_desc.getGroupbyColCount(),
                                             slots_device,
                                             slots->size(),
                                             reinterpret_cast<int*>(reduction_scratch),
                                             device_id,
                                             cuda_stream);
  if (!reduction_success) {
    return std::nullopt;
  }

  const auto compacted_row_count =
      count_non_empty_baseline_hash_rows_on_device(destination_buffer,
                                                   destination_entry_count,
                                                   row_size,
                                                   key_width,
                                                   reduction_scratch,
                                                   device_id,
                                                   cuda_stream);
  if (compacted_row_count == 0) {
    return compacted_row_count;
  }
  if (compacted_row_count > destination_entry_count) {
    LOG(ERROR) << "GPU baseline compaction row count exceeds its source allocation: "
               << compacted_row_count << " > " << destination_entry_count;
    return std::nullopt;
  }
  const auto compacted_bytes = checked_size_multiply(
      compacted_row_count, row_size, "Compacted baseline buffer size overflow");
  auto* compacted_dev_buffer = device_allocator.alloc(compacted_bytes);
  if (compacted_device_buffer) {
    *compacted_device_buffer = compacted_dev_buffer;
  }
  auto* compacted_row_count_dev =
      reinterpret_cast<uint64_t*>(device_allocator.alloc(sizeof(uint64_t)));
  compact_baseline_hash_rows_on_device(destination_buffer,
                                       compacted_dev_buffer,
                                       compacted_row_count_dev,
                                       destination_entry_count,
                                       row_size,
                                       key_width,
                                       device_id,
                                       cuda_stream);
  uint64_t verified_row_count{0};
  device_allocator.copyFromDevice(&verified_row_count,
                                  compacted_row_count_dev,
                                  sizeof(verified_row_count),
                                  "Compacted multi-buffer group-by row count");
  CHECK_EQ(compacted_row_count, static_cast<size_t>(verified_row_count));
  if (!skip_host_copy_for_compacted) {
    device_allocator.copyFromDevice(group_by_buffers[first_group_buffer_idx],
                                    compacted_dev_buffer,
                                    compacted_bytes,
                                    "Compacted multi-buffer group-by buffer");
  }
  return compacted_row_count;
#else
  return std::nullopt;
#endif
}

}  // namespace

GpuGroupByBuffers create_dev_group_by_buffers(
    DeviceAllocator* device_allocator,
    const std::vector<int64_t*>& group_by_buffers,
    const QueryMemoryDescriptor& query_mem_desc,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    const ExecutorDispatchMode dispatch_mode,
    const int64_t num_input_rows,
    const bool prepend_index_buffer,
    const bool always_init_group_by_on_host,
    const bool use_bump_allocator,
    const bool has_varlen_output,
    Allocator* insitu_allocator) {
  if (group_by_buffers.empty() && !insitu_allocator) {
    return {0, 0, 0, 0};
  }
  CHECK(device_allocator);

  size_t groups_buffer_size{0};
  int8_t* group_by_dev_buffers_mem{nullptr};
  size_t mem_size{0};
  size_t entry_count{0};

  if (use_bump_allocator) {
    CHECK(!prepend_index_buffer);
    CHECK(!insitu_allocator);

    if (dispatch_mode == ExecutorDispatchMode::KernelPerFragment) {
      // Allocate an output buffer equal to the size of the number of rows in the
      // fragment. The kernel per fragment path is only used for projections with lazy
      // fetched outputs. Therefore, the resulting output buffer should be relatively
      // narrow compared to the width of an input row, offsetting the larger allocation.

      CHECK_GT(num_input_rows, int64_t(0));
      entry_count = num_input_rows;
      groups_buffer_size =
          query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU, entry_count);
      mem_size = coalesced_size(query_mem_desc,
                                groups_buffer_size,
                                query_mem_desc.blocksShareMemory() ? 1 : grid_size_x);
      // TODO(adb): render allocator support
      VLOG(1) << "Prepare query output buffer on GPU, bump_allocator: on, dispatch_mode: "
                 "KernelPerFragment, entry_count: "
              << entry_count << ", buffer_size: " << mem_size << " bytes";
      group_by_dev_buffers_mem = device_allocator->alloc(mem_size);
    } else {
      // Attempt to allocate increasingly small buffers until we have less than 256B of
      // memory remaining on the device. This may have the side effect of evicting
      // memory allocated for previous queries. However, at current maximum slab sizes
      // (2GB) we expect these effects to be minimal.
      size_t max_memory_size{g_max_memory_allocation_size};
      while (true) {
        entry_count = max_memory_size / query_mem_desc.getRowSize();
        groups_buffer_size =
            query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU, entry_count);

        try {
          mem_size = coalesced_size(query_mem_desc,
                                    groups_buffer_size,
                                    query_mem_desc.blocksShareMemory() ? 1 : grid_size_x);
          CHECK_LE(entry_count, std::numeric_limits<uint32_t>::max());

          // TODO(adb): render allocator support
          VLOG(1) << "Allocating query output buffer on GPU, bump_allocator: on, "
                     "entry_count: "
                  << entry_count << ", buffer_size: " << mem_size << " bytes";
          group_by_dev_buffers_mem = device_allocator->alloc(mem_size);
        } catch (const OutOfMemory& e) {
          LOG(WARNING) << e.what();
          max_memory_size = max_memory_size * g_bump_allocator_step_reduction;
          if (max_memory_size < g_min_memory_allocation_size) {
            throw;
          }

          LOG(WARNING) << "Ran out of memory for projection query output. Retrying with "
                       << std::to_string(max_memory_size) << " bytes";

          continue;
        }
        break;
      }
    }
  } else {
    entry_count = query_mem_desc.getEntryCount();
    CHECK_GT(entry_count, size_t(0));
    groups_buffer_size =
        query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU, entry_count);
    mem_size = coalesced_size(query_mem_desc,
                              groups_buffer_size,
                              query_mem_desc.blocksShareMemory() ? 1 : grid_size_x);
    const size_t prepended_buff_size =
        prepend_index_buffer
            ? checked_align_to_int64(
                  checked_size_multiply(entry_count,
                                        sizeof(int32_t),
                                        "Prepended group-by index size overflow"),
                  "Prepended group-by index alignment overflow")
            : 0;

    int8_t* group_by_dev_buffers_allocation{nullptr};
    const auto group_by_dev_buffer_size = checked_size_add(
        mem_size, prepended_buff_size, "Group-by device buffer size overflow");
    VLOG(1) << "Allocating query output buffer on GPU, entry_count: " << entry_count
            << ", buffer_size: " << group_by_dev_buffer_size
            << " bytes (prepend_index_buffer_size: " << prepended_buff_size << " bytes)";
    if (insitu_allocator) {
      group_by_dev_buffers_allocation = insitu_allocator->alloc(group_by_dev_buffer_size);
    } else {
      group_by_dev_buffers_allocation = device_allocator->alloc(group_by_dev_buffer_size);
    }
    CHECK(group_by_dev_buffers_allocation);

    group_by_dev_buffers_mem = group_by_dev_buffers_allocation + prepended_buff_size;
  }
  CHECK_GT(groups_buffer_size, size_t(0));
  CHECK(group_by_dev_buffers_mem);

  CHECK(query_mem_desc.threadsShareMemory());
  const size_t step{block_size_x};

  if (!insitu_allocator && (always_init_group_by_on_host ||
                            !query_mem_desc.lazyInitGroups(ExecutorDeviceType::GPU))) {
    std::vector<int8_t> buff_to_gpu(mem_size);
    auto buff_to_gpu_ptr = buff_to_gpu.data();

    const size_t start = has_varlen_output ? 1 : 0;
    for (size_t i = start; i < group_by_buffers.size(); i += step) {
      memcpy(buff_to_gpu_ptr, group_by_buffers[i], groups_buffer_size);
      buff_to_gpu_ptr += groups_buffer_size;
    }
    device_allocator->copyToDevice(reinterpret_cast<int8_t*>(group_by_dev_buffers_mem),
                                   buff_to_gpu.data(),
                                   buff_to_gpu.size(),
                                   "Group-by buffer");
  }

  auto group_by_dev_buffer = group_by_dev_buffers_mem;

  const size_t num_ptrs =
      checked_size_add(checked_size_multiply(static_cast<size_t>(block_size_x),
                                             static_cast<size_t>(grid_size_x),
                                             "Group-by device pointer count overflow"),
                       has_varlen_output ? size_t(1) : size_t(0),
                       "Group-by device pointer count overflow");

  std::vector<int8_t*> group_by_dev_buffers(num_ptrs);

  const size_t start_index = has_varlen_output ? 1 : 0;
  for (size_t i = start_index; i < num_ptrs; i += step) {
    for (size_t j = 0; j < step; ++j) {
      group_by_dev_buffers[i + j] = group_by_dev_buffer;
    }
    if (!query_mem_desc.blocksShareMemory()) {
      group_by_dev_buffer += groups_buffer_size;
    }
  }

  int8_t* varlen_output_buffer{nullptr};
  if (has_varlen_output) {
    const auto varlen_buffer_elem_size_opt = query_mem_desc.varlenOutputBufferElemSize();
    CHECK(varlen_buffer_elem_size_opt);  // TODO(adb): relax
    const auto buf_size = checked_size_multiply(query_mem_desc.getEntryCount(),
                                                varlen_buffer_elem_size_opt.value(),
                                                "Varlen GPU output buffer size overflow");
    VLOG(1) << "Allocating varlen output buffer on GPU, entry_count: "
            << query_mem_desc.getEntryCount() << ", buffer_size: " << buf_size
            << " bytes";
    group_by_dev_buffers[0] = device_allocator->alloc(buf_size);
    varlen_output_buffer = group_by_dev_buffers[0];
  }

  const auto dev_ptr_buf_size = checked_size_multiply(
      num_ptrs, sizeof(CUdeviceptr), "Group-by device pointer buffer size overflow");
  VLOG(1) << "Allocating group-by buffer buffer on device-" << device_id
          << ", size: " << dev_ptr_buf_size << " bytes";
  auto group_by_dev_ptr = device_allocator->alloc(dev_ptr_buf_size);
  device_allocator->copyToDevice(group_by_dev_ptr,
                                 reinterpret_cast<int8_t*>(group_by_dev_buffers.data()),
                                 dev_ptr_buf_size,
                                 "Group-by buffer");

  return {group_by_dev_ptr, group_by_dev_buffers_mem, entry_count, varlen_output_buffer};
}

namespace {

std::optional<size_t> compact_and_copy_baseline_group_by_buffer_from_gpu(
    DeviceAllocator& device_allocator,
    const std::vector<int64_t*>& group_by_buffers,
    const int8_t* group_by_dev_buffers_mem,
    const QueryMemoryDescriptor& query_mem_desc,
    const int device_id,
    CUstream cuda_stream,
    const size_t first_group_buffer_idx,
    const std::vector<TargetInfo>* target_infos,
    int8_t** compacted_device_buffer,
    const bool skip_host_copy_for_compacted,
    const ResultSetEntryFilter* entry_filter,
    const std::vector<int64_t>* preserved_keys,
    bool* entry_filter_applied) {
#ifdef HAVE_CUDA
  if (query_mem_desc.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByBaselineHash ||
      query_mem_desc.didOutputColumnar() || query_mem_desc.hasKeylessHash() ||
      query_mem_desc.hasVarlenOutput() ||
      !count_distinct_descriptors_safe_for_group_key_output(
          query_mem_desc, target_infos ? target_infos->size() : size_t(0)) ||
      query_mem_desc.getNumModeTargets() > 0) {
    return std::nullopt;
  }

  const auto has_entry_filter = entry_filter && !entry_filter->empty();
  constexpr size_t min_compaction_entry_count = 1000000;
  const auto entry_count = query_mem_desc.getEntryCount();
  if (entry_count < min_compaction_entry_count && !skip_host_copy_for_compacted) {
    return std::nullopt;
  }

  const auto row_size = query_mem_desc.getRowSize();
  if (row_size == 0 || row_size % sizeof(int64_t) != 0) {
    return std::nullopt;
  }
  const auto key_width = query_mem_desc.getEffectiveKeyWidth();
  if (key_width != size_t(4) && key_width != size_t(8)) {
    return std::nullopt;
  }

  std::vector<DeviceResultSetEntryComparison> device_entry_filter;
  if (has_entry_filter) {
    if (!target_infos) {
      return std::nullopt;
    }
    auto maybe_device_entry_filter =
        make_device_entry_filter(*entry_filter, query_mem_desc, *target_infos);
    if (!maybe_device_entry_filter) {
      return std::nullopt;
    }
    device_entry_filter = std::move(*maybe_device_entry_filter);
  }

  std::vector<int64_t> sorted_preserved_keys;
  if (preserved_keys && !preserved_keys->empty()) {
    sorted_preserved_keys = *preserved_keys;
    std::sort(sorted_preserved_keys.begin(), sorted_preserved_keys.end());
    sorted_preserved_keys.erase(
        std::unique(sorted_preserved_keys.begin(), sorted_preserved_keys.end()),
        sorted_preserved_keys.end());
  }

  DeviceResultSetEntryComparison* device_entry_filter_ptr{nullptr};
  if (!device_entry_filter.empty()) {
    const auto filter_bytes =
        checked_size_multiply(device_entry_filter.size(),
                              sizeof(DeviceResultSetEntryComparison),
                              "ResultSet entry filter size overflow");
    device_entry_filter_ptr = reinterpret_cast<DeviceResultSetEntryComparison*>(
        device_allocator.alloc(filter_bytes));
    device_allocator.copyToDevice(
        reinterpret_cast<int8_t*>(device_entry_filter_ptr),
        reinterpret_cast<const int8_t*>(device_entry_filter.data()),
        filter_bytes,
        "ResultSet entry filter");
  }

  int64_t* preserved_keys_ptr{nullptr};
  if (!sorted_preserved_keys.empty()) {
    const auto preserved_keys_bytes =
        checked_size_multiply(sorted_preserved_keys.size(),
                              sizeof(int64_t),
                              "ResultSet preserved key buffer size overflow");
    preserved_keys_ptr =
        reinterpret_cast<int64_t*>(device_allocator.alloc(preserved_keys_bytes));
    device_allocator.copyToDevice(
        reinterpret_cast<int8_t*>(preserved_keys_ptr),
        reinterpret_cast<const int8_t*>(sorted_preserved_keys.data()),
        preserved_keys_bytes,
        "ResultSet preserved keys");
  }

  auto* row_count_device =
      reinterpret_cast<uint64_t*>(device_allocator.alloc(sizeof(uint64_t)));

  const auto compacted_row_count =
      has_entry_filter
          ? count_matching_baseline_hash_rows_on_device(group_by_dev_buffers_mem,
                                                        entry_count,
                                                        row_size,
                                                        key_width,
                                                        device_entry_filter_ptr,
                                                        device_entry_filter.size(),
                                                        preserved_keys_ptr,
                                                        sorted_preserved_keys.size(),
                                                        row_count_device,
                                                        device_id,
                                                        cuda_stream)
          : count_non_empty_baseline_hash_rows_on_device(group_by_dev_buffers_mem,
                                                         entry_count,
                                                         row_size,
                                                         key_width,
                                                         row_count_device,
                                                         device_id,
                                                         cuda_stream);
  if (compacted_row_count == 0) {
    if (has_entry_filter && entry_filter_applied) {
      *entry_filter_applied = true;
    }
    return compacted_row_count;
  }
  if (compacted_row_count > entry_count) {
    LOG(ERROR) << "GPU baseline compaction row count exceeds its source allocation: "
               << compacted_row_count << " > " << entry_count;
    return std::nullopt;
  }

  constexpr size_t max_dense_numerator = 3;
  constexpr size_t max_dense_denominator = 4;
  if (static_cast<unsigned __int128>(compacted_row_count) * max_dense_denominator >=
      static_cast<unsigned __int128>(entry_count) * max_dense_numerator) {
    if (!skip_host_copy_for_compacted) {
      return std::nullopt;
    }
  }

  const auto compacted_bytes = checked_size_multiply(
      compacted_row_count, row_size, "Compacted group-by buffer size overflow");
  auto* compacted_dev_buffer = device_allocator.alloc(compacted_bytes);
  if (compacted_device_buffer) {
    *compacted_device_buffer = compacted_dev_buffer;
  }
  auto* compacted_row_count_dev =
      reinterpret_cast<uint64_t*>(device_allocator.alloc(sizeof(uint64_t)));
  if (has_entry_filter) {
    compact_matching_baseline_hash_rows_on_device(group_by_dev_buffers_mem,
                                                  compacted_dev_buffer,
                                                  compacted_row_count_dev,
                                                  entry_count,
                                                  row_size,
                                                  key_width,
                                                  device_entry_filter_ptr,
                                                  device_entry_filter.size(),
                                                  preserved_keys_ptr,
                                                  sorted_preserved_keys.size(),
                                                  device_id,
                                                  cuda_stream);
  } else {
    compact_baseline_hash_rows_on_device(group_by_dev_buffers_mem,
                                         compacted_dev_buffer,
                                         compacted_row_count_dev,
                                         entry_count,
                                         row_size,
                                         key_width,
                                         device_id,
                                         cuda_stream);
  }
  uint64_t verified_row_count{0};
  device_allocator.copyFromDevice(&verified_row_count,
                                  compacted_row_count_dev,
                                  sizeof(verified_row_count),
                                  "Compacted group-by row count");
  CHECK_EQ(compacted_row_count, static_cast<size_t>(verified_row_count));
  if (has_entry_filter && entry_filter_applied) {
    *entry_filter_applied = true;
  }
  if (!skip_host_copy_for_compacted) {
    device_allocator.copyFromDevice(group_by_buffers[first_group_buffer_idx],
                                    compacted_dev_buffer,
                                    compacted_bytes,
                                    "Compacted group-by buffer");
  }
  return compacted_row_count;
#else
  return std::nullopt;
#endif
}

}  // namespace

bool can_defer_keyless_perfect_hash_rowwise_result(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t entry_count) {
  constexpr size_t kMaxCpuMaterializedKeylessPerfectHashEntries = 4096;
  if (entry_count <= kMaxCpuMaterializedKeylessPerfectHashEntries) {
    return false;
  }
  if (query_mem_desc.getQueryDescriptionType() !=
          QueryDescriptionType::GroupByPerfectHash ||
      query_mem_desc.didOutputColumnar() || query_mem_desc.hasVarlenOutput() ||
      !query_mem_desc.hasKeylessHash() || !query_mem_desc.usesGetGroupValueFast() ||
      query_mem_desc.mustUseBaselineSort()) {
    return false;
  }
  for (size_t target_idx = 0; target_idx < query_mem_desc.targetGroupbyIndicesSize();
       ++target_idx) {
    if (query_mem_desc.getTargetGroupbyIndex(target_idx) >= 0) {
      return false;
    }
  }
  const auto& col_slot_context = query_mem_desc.getColSlotContext();
  for (size_t column_idx = 0; column_idx < query_mem_desc.getColCount(); ++column_idx) {
    const auto& slots = col_slot_context.getSlotsForCol(column_idx);
    if (slots.size() != size_t(1)) {
      return false;
    }
    const auto slot_idx = slots.front();
    if (query_mem_desc.checkSlotUsesFlatBufferFormat(slot_idx)) {
      return false;
    }
    const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
    if (slot_width != 1 && slot_width != 2 && slot_width != 4 && slot_width != 8) {
      return false;
    }
  }
  return query_mem_desc.getColCount() > size_t(0);
}

std::optional<size_t> copy_group_by_buffers_from_gpu(
    DeviceAllocator& device_allocator,
    const std::vector<int64_t*>& group_by_buffers,
    const size_t groups_buffer_size,
    const int8_t* group_by_dev_buffers_mem,
    const QueryMemoryDescriptor& query_mem_desc,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    CUstream cuda_stream,
    const bool prepend_index_buffer,
    const bool has_varlen_output,
    const std::vector<TargetInfo>* target_infos,
    const std::vector<int64_t>* init_vals,
    int8_t** compacted_device_buffer,
    const bool skip_host_copy_for_compacted,
    const bool skip_host_copy_for_perfect_hash,
    bool* host_copy_performed,
    const ResultSetEntryFilter* entry_filter,
    const std::vector<int64_t>* preserved_keys,
    bool* entry_filter_applied) {
  if (host_copy_performed) {
    *host_copy_performed = false;
  }
  if (compacted_device_buffer) {
    *compacted_device_buffer = nullptr;
  }
  if (entry_filter_applied) {
    *entry_filter_applied = false;
  }
  if (group_by_buffers.empty()) {
    return std::nullopt;
  }
  const size_t first_group_buffer_idx = has_varlen_output ? 1 : 0;

  const unsigned block_buffer_count{query_mem_desc.blocksShareMemory() ? 1 : grid_size_x};
  if (block_buffer_count == 1 && !prepend_index_buffer) {
    CHECK_EQ(coalesced_size(query_mem_desc, groups_buffer_size, block_buffer_count),
             groups_buffer_size);
    if (auto compacted_row_count = compact_and_copy_baseline_group_by_buffer_from_gpu(
            device_allocator,
            group_by_buffers,
            group_by_dev_buffers_mem,
            query_mem_desc,
            device_id,
            cuda_stream,
            first_group_buffer_idx,
            target_infos,
            compacted_device_buffer,
            skip_host_copy_for_compacted,
            entry_filter,
            preserved_keys,
            entry_filter_applied)) {
      return compacted_row_count;
    }
    if ((skip_host_copy_for_compacted || skip_host_copy_for_perfect_hash) &&
        query_mem_desc.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByPerfectHash &&
        !query_mem_desc.didOutputColumnar() && !has_varlen_output &&
        (skip_host_copy_for_perfect_hash ||
         can_defer_keyless_perfect_hash_rowwise_result(query_mem_desc,
                                                       query_mem_desc.getEntryCount()))) {
      return std::nullopt;
    }
    device_allocator.copyFromDevice(group_by_buffers[first_group_buffer_idx],
                                    group_by_dev_buffers_mem,
                                    groups_buffer_size,
                                    "Group-by buffer");
    if (host_copy_performed) {
      *host_copy_performed = true;
    }
    return std::nullopt;
  }
  if (!prepend_index_buffer && !has_varlen_output) {
    if (auto compacted_row_count = reduce_and_compact_baseline_group_by_buffers_from_gpu(
            device_allocator,
            group_by_buffers,
            groups_buffer_size,
            group_by_dev_buffers_mem,
            query_mem_desc,
            block_size_x,
            grid_size_x,
            device_id,
            cuda_stream,
            block_buffer_count,
            first_group_buffer_idx,
            target_infos,
            init_vals,
            compacted_device_buffer,
            skip_host_copy_for_compacted)) {
      return compacted_row_count;
    }
  }
  const size_t index_buffer_sz =
      prepend_index_buffer ? checked_size_multiply(query_mem_desc.getEntryCount(),
                                                   sizeof(int64_t),
                                                   "Group-by index buffer size overflow")
                           : 0;
  std::vector<int8_t> buff_from_gpu(checked_size_add(
      coalesced_size(query_mem_desc, groups_buffer_size, block_buffer_count),
      index_buffer_sz,
      "Copied group-by buffer size overflow"));
  device_allocator.copyFromDevice(&buff_from_gpu[0],
                                  group_by_dev_buffers_mem - index_buffer_sz,
                                  buff_from_gpu.size(),
                                  "Group-by buffer");
  if (host_copy_performed) {
    *host_copy_performed = true;
  }
  auto buff_from_gpu_ptr = &buff_from_gpu[0];
  for (size_t i = 0; i < block_buffer_count; ++i) {
    const size_t buffer_idx = (i * block_size_x) + first_group_buffer_idx;
    CHECK_LT(buffer_idx, group_by_buffers.size());
    memcpy(
        group_by_buffers[buffer_idx],
        buff_from_gpu_ptr,
        checked_size_add(
            groups_buffer_size, index_buffer_sz, "Host group-by buffer size overflow"));
    buff_from_gpu_ptr += groups_buffer_size;
  }
  return std::nullopt;
}

/**
 * Returns back total number of allocated rows per device (i.e., number of matched
 * elements in projections).
 *
 * TODO(Saman): revisit this for bump allocators
 */
size_t get_num_allocated_rows_from_gpu(DeviceAllocator& device_allocator,
                                       int8_t* projection_size_gpu,
                                       const int device_id) {
  int32_t num_rows{0};
  device_allocator.copyFromDevice(
      &num_rows, projection_size_gpu, sizeof(num_rows), "# allocated rows");
  CHECK(num_rows >= 0);
  return static_cast<size_t>(num_rows);
}

/**
 * For projection queries we only copy back as many elements as necessary, not the whole
 * output buffer. The goal is to be able to build a compact ResultSet, particularly useful
 * for columnar outputs.
 *
 * NOTE: Saman: we should revisit this function when we have a bump allocator
 */
void copy_projection_buffer_from_gpu_columnar(
    DeviceAllocator* device_allocator,
    const GpuGroupByBuffers& gpu_group_by_buffers,
    const QueryMemoryDescriptor& query_mem_desc,
    int8_t* projection_buffer,
    const size_t projection_count,
    const int device_id) {
#ifdef HAVE_CUDA
  CHECK(query_mem_desc.didOutputColumnar());
  CHECK(query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection);
  CHECK(device_allocator);
  constexpr size_t row_index_width = sizeof(int64_t);
  const auto row_index_bytes = checked_size_multiply(
      projection_count, row_index_width, "Projection row index size overflow");

  // copy all the row indices back to the host
  device_allocator->copyFromDevice(projection_buffer,
                                   gpu_group_by_buffers.data,
                                   row_index_bytes,
                                   "Output column indices");
  size_t buffer_offset_cpu{row_index_bytes};
  // other columns are actual non-lazy columns for the projection:
  for (size_t i = 0; i < query_mem_desc.getSlotCount(); i++) {
    if (query_mem_desc.getPaddedSlotWidthBytes(i) > 0) {
      const auto column_proj_size = checked_size_multiply(
          projection_count,
          static_cast<size_t>(query_mem_desc.getPaddedSlotWidthBytes(i)),
          "Projection column size overflow");
      device_allocator->copyFromDevice(
          projection_buffer + buffer_offset_cpu,
          gpu_group_by_buffers.data + query_mem_desc.getColOffInBytes(i),
          column_proj_size,
          "Output column buffer");
      buffer_offset_cpu =
          checked_size_add(buffer_offset_cpu,
                           checked_align_to_int64(column_proj_size,
                                                  "Projection column alignment overflow"),
                           "Projection output buffer offset overflow");
    }
  }
#else
  CHECK(false);
#endif  // HAVE_CUDA
}
