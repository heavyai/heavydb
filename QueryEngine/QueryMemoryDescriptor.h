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

/**
 * @file    QueryMemoryDescriptor.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Descriptor for the result set buffer layout.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_QUERYMEMORYDESCRIPTOR_H
#define QUERYENGINE_QUERYMEMORYDESCRIPTOR_H

#include "CompilationOptions.h"
#include "CountDistinct.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

extern bool g_cluster;
extern bool g_use_result_set;

class Executor;
class QueryExecutionContext;
class RenderInfo;
class RowSetMemoryOwner;
struct InputTableInfo;

enum class GroupByColRangeType {
  OneColKnownRange,    // statically known range, only possible for column expressions
  OneColGuessedRange,  // best guess: small hash for the guess plus overflow for outliers
  MultiCol,
  MultiColPerfectHash,
  Projection,
  Scan,  // the plan is not a group by plan
  Estimator
};

struct ColWidths {
  int8_t actual;   // the real width of the type
  int8_t compact;  // with padding
};

inline bool operator==(const ColWidths& lhs, const ColWidths& rhs) {
  return lhs.actual == rhs.actual && lhs.compact == rhs.compact;
}

// Private: each thread has its own memory, no atomic operations required
// Shared: threads in the same block share memory, atomic operations required
// SharedForKeylessOneColumnKnownRange: special case of "Shared", but for keyless
// aggregates with single column group by
enum class GroupByMemSharing { Private, Shared, SharedForKeylessOneColumnKnownRange };

struct RelAlgExecutionUnit;

class QueryMemoryDescriptor {
 public:
  QueryMemoryDescriptor();

  QueryMemoryDescriptor(const Executor* executor, const size_t entry_count, const GroupByColRangeType hash_type);

  QueryMemoryDescriptor(const Executor* executor,
                        const bool allow_multifrag,
                        const GroupByColRangeType hash_type,
                        const bool keyless_hash,
                        const bool interleaved_bins_on_gpu,
                        const int32_t idx_target_as_key,
                        const int64_t init_val,
                        const std::vector<int8_t>& group_col_widths,
#ifdef ENABLE_KEY_COMPACTION
                        const int8_t group_col_compact_width,
#endif
                        const std::vector<ColWidths>& agg_col_widths,
                        const std::vector<ssize_t>& target_groupby_indices,
                        const size_t entry_count,
                        const size_t entry_count_small,
                        const int64_t min_val,
                        const int64_t max_val,
                        const int64_t bucket,
                        const bool hash_nulls,
                        const GroupByMemSharing sharing,
                        const CountDistinctDescriptors count_distinct_descriptors,
                        const bool sort_on_gpu,
                        const bool is_sort_plan,
                        const bool output_columnar,
                        const bool reder_output,
                        const std::vector<int8_t>& key_column_pad_bytes,
                        const std::vector<int8_t>& target_column_pad_bytes,
                        const bool must_use_baseline_sort);

  std::unique_ptr<QueryExecutionContext> getQueryExecutionContext(
      const RelAlgExecutionUnit&,
      const std::vector<int64_t>& init_agg_vals,
      const Executor* executor,
      const ExecutorDeviceType device_type,
      const int device_id,
      const std::vector<std::vector<const int8_t*>>& col_buffers,
      const std::vector<std::vector<const int8_t*>>& iter_buffers,
      const std::vector<std::vector<uint64_t>>& frag_offsets,
      std::shared_ptr<RowSetMemoryOwner>,
      const bool output_columnar,
      const bool sort_on_gpu,
      RenderInfo*) const;

  const Executor* getExecutor() const { return executor_; }

  // shared, static methods attached to QMD for now
  static bool many_entries(const int64_t max_val, const int64_t min_val, const int64_t bucket) {
    return max_val - min_val > 10000 * std::max(bucket, int64_t(1));
  }
  
  static bool countDescriptorsLogicallyEmpty(const CountDistinctDescriptors& count_distinct_descriptors) {
    return std::all_of(
        count_distinct_descriptors.begin(), count_distinct_descriptors.end(), [](const CountDistinctDescriptor& desc) {
          return desc.impl_type_ == CountDistinctImplType::Invalid;
        });
  }

  static int8_t pick_target_compact_width(const RelAlgExecutionUnit& ra_exe_unit,
                                          const std::vector<InputTableInfo>& query_infos,
                                          const int8_t crt_min_byte_width);

  // TODO(adb): remove need for static version above
  bool countDistinctDescriptorsLogicallyEmpty() const {
    return countDescriptorsLogicallyEmpty(count_distinct_descriptors_);
  }

  // old / to be sorted

  bool allow_multifrag;
  GroupByColRangeType hash_type;

  bool keyless_hash;
  bool interleaved_bins_on_gpu;
  int32_t idx_target_as_key;
  int64_t init_val;

  std::vector<int8_t> group_col_widths;
#ifdef ENABLE_KEY_COMPACTION
  int8_t group_col_compact_width;  // compact width for all group cols if able to
                                   // be consistent otherwise 0
#endif

  std::vector<ColWidths> agg_col_widths;
  std::vector<ssize_t> target_groupby_indices;
  size_t entry_count;        // the number of entries in the main buffer
  size_t entry_count_small;  // the number of entries in the small buffer
  int64_t min_val;           // meaningful for OneColKnownRange, MultiColPerfectHash only
  int64_t max_val;
  int64_t bucket;
  bool has_nulls;
  GroupByMemSharing sharing;  // meaningful for GPU only
  CountDistinctDescriptors count_distinct_descriptors_;
  bool sort_on_gpu_;
  bool is_sort_plan;  // TODO(alex): remove
  bool output_columnar;
  bool render_output;
  std::vector<int8_t> key_column_pad_bytes;
  std::vector<int8_t> target_column_pad_bytes;
  bool must_use_baseline_sort;

  bool force_4byte_float_ = false;

  size_t getBufferSizeQuad(const ExecutorDeviceType device_type) const;
  size_t getSmallBufferSizeQuad() const;

  size_t getBufferSizeBytes(const RelAlgExecutionUnit& ra_exe_unit,
                            const unsigned thread_count,
                            const ExecutorDeviceType device_type) const;
  size_t getBufferSizeBytes(const ExecutorDeviceType device_type) const;
  size_t getSmallBufferSizeBytes() const;

  // TODO(alex): remove
  bool usesGetGroupValueFast() const;

  // TODO(alex): remove
  bool usesCachedContext() const;

  bool blocksShareMemory() const;
  bool threadsShareMemory() const;

  bool lazyInitGroups(const ExecutorDeviceType) const;

  bool interleavedBins(const ExecutorDeviceType) const;

  size_t sharedMemBytes(const ExecutorDeviceType) const;

  bool canOutputColumnar() const;

  bool sortOnGpu() const;

  size_t getKeyOffInBytes(const size_t bin, const size_t key_idx = 0) const;
  size_t getNextKeyOffInBytes(const size_t key_idx) const;
  size_t getColOffInBytes(const size_t bin, const size_t col_idx) const;
  size_t getColOffInBytesInNextBin(const size_t col_idx) const;
  size_t getNextColOffInBytes(const int8_t* col_ptr,
                              const size_t bin,
                              const size_t col_idx) const;
  size_t getColOnlyOffInBytes(const size_t col_idx) const;
  size_t getRowSize() const;
  size_t getColsSize() const;
  size_t getWarpCount() const;

  size_t getCompactByteWidth() const;
  bool isCompactLayoutIsometric() const;
  size_t getConsistColOffInBytes(const size_t bin, const size_t col_idx) const;

  inline size_t getEffectiveKeyWidth() const {
#ifdef ENABLE_KEY_COMPACTION
    return group_col_compact_width ? group_col_compact_width : sizeof(int64_t);
#else
    return sizeof(int64_t);
#endif
  }

  bool isWarpSyncRequired(const ExecutorDeviceType) const;

 private:
  const Executor* executor_;

  size_t getTotalBytesOfColumnarBuffers(const std::vector<ColWidths>& col_widths) const;
};

#endif  // QUERYENGINE_QUERYMEMORYDESCRIPTOR_H
