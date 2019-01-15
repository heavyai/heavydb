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

class Executor;
class QueryExecutionContext;
class RenderInfo;
class RowSetMemoryOwner;
struct InputTableInfo;

enum class QueryDescriptionType {
  GroupByPerfectHash,
  GroupByBaselineHash,
  Projection,
  NonGroupedAggregate,
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
enum class GroupByMemSharing { Shared, SharedForKeylessOneColumnKnownRange };

struct RelAlgExecutionUnit;
class TResultSetBufferDescriptor;
class GroupByAndAggregate;
struct ColRangeInfo;

class QueryMemoryDescriptor {
 public:
  QueryMemoryDescriptor();

  // Constructor for non-group by queries
  QueryMemoryDescriptor(const Executor* executor,
                        const RelAlgExecutionUnit& ra_exe_unit,
                        const std::vector<InputTableInfo>& query_infos,
                        const std::vector<int8_t>& group_col_widths,
                        const bool allow_multifrag,
                        const bool is_group_by,
                        const int8_t crt_min_byte_width,
                        RenderInfo* render_info,
                        const CountDistinctDescriptors count_distinct_descriptors,
                        const bool must_use_baseline_sort);

  // Constructor for group-by queries
  QueryMemoryDescriptor(const Executor* executor,
                        const RelAlgExecutionUnit& ra_exe_unit,
                        const std::vector<InputTableInfo>& query_infos,
                        const GroupByAndAggregate* group_by_and_agg,
                        const std::vector<int8_t>& group_col_widths,
                        const ColRangeInfo& col_range_info,
                        const bool allow_multifrag,
                        const bool is_group_by,
                        const int8_t crt_min_byte_width,
                        const bool sort_on_gpu_hint,
                        const size_t shard_count,
                        const size_t max_groups_buffer_entry_count,
                        RenderInfo* render_info,
                        const CountDistinctDescriptors count_distinct_descriptors,
                        const bool must_use_baseline_sort,
                        const bool output_columanr_hint);

  QueryMemoryDescriptor(const Executor* executor,
                        const size_t entry_count,
                        const QueryDescriptionType query_desc_type);

  QueryMemoryDescriptor(const QueryDescriptionType query_desc_type,
                        const int64_t min_val,
                        const int64_t max_val,
                        const bool has_nulls,
                        const std::vector<int8_t>& group_col_widths);

  // Serialization
  QueryMemoryDescriptor(const TResultSetBufferDescriptor& thrift_query_memory_descriptor);
  static TResultSetBufferDescriptor toThrift(const QueryMemoryDescriptor&);

  bool operator==(const QueryMemoryDescriptor& other) const;

  std::unique_ptr<QueryExecutionContext> getQueryExecutionContext(
      const RelAlgExecutionUnit&,
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

  static bool many_entries(const int64_t max_val,
                           const int64_t min_val,
                           const int64_t bucket) {
    return max_val - min_val > 10000 * std::max(bucket, int64_t(1));
  }

  static bool countDescriptorsLogicallyEmpty(
      const CountDistinctDescriptors& count_distinct_descriptors) {
    return std::all_of(count_distinct_descriptors.begin(),
                       count_distinct_descriptors.end(),
                       [](const CountDistinctDescriptor& desc) {
                         return desc.impl_type_ == CountDistinctImplType::Invalid;
                       });
  }

  bool countDistinctDescriptorsLogicallyEmpty() const {
    return countDescriptorsLogicallyEmpty(count_distinct_descriptors_);
  }

  static int8_t pick_target_compact_width(const RelAlgExecutionUnit& ra_exe_unit,
                                          const std::vector<InputTableInfo>& query_infos,
                                          const int8_t crt_min_byte_width);

  // Getters and Setters
  const Executor* getExecutor() const { return executor_; }

  QueryDescriptionType getQueryDescriptionType() const { return query_desc_type_; }
  void setQueryDescriptionType(const QueryDescriptionType val) { query_desc_type_ = val; }
  bool isSingleColumnGroupByWithPerfectHash() const {
    return getQueryDescriptionType() == QueryDescriptionType::GroupByPerfectHash &&
           getGroupbyColCount() == 1;
  }

  bool hasKeylessHash() const { return keyless_hash_; }
  void setHasKeylessHash(const bool val) { keyless_hash_ = val; }

  bool hasInterleavedBinsOnGpu() const { return interleaved_bins_on_gpu_; }
  void setHasInterleavedBinsOnGpu(const bool val) { interleaved_bins_on_gpu_ = val; }

  int32_t getTargetIdxForKey() const { return idx_target_as_key_; }
  void setTargetIdxForKey(const int32_t val) { idx_target_as_key_ = val; }

  int64_t getInitVal() const { return init_val_; }

  size_t groupColWidthsSize() const { return group_col_widths_.size(); }
  int8_t groupColWidth(const size_t key_idx) const {
    CHECK_LT(key_idx, group_col_widths_.size());
    return group_col_widths_[key_idx];
  }
  size_t getPrependedGroupColOffInBytes(const size_t group_idx) const;
  size_t getPrependedGroupBufferSizeInBytes() const;

  const auto groupColWidthsBegin() const { return group_col_widths_.begin(); }
  const auto groupColWidthsEnd() const { return group_col_widths_.end(); }
  void clearGroupColWidths() { group_col_widths_.clear(); }

  bool isGroupBy() const { return !group_col_widths_.empty(); }

  void setGroupColCompactWidth(const int8_t val) { group_col_compact_width_ = val; }

  size_t getColCount() const { return agg_col_widths_.size(); }
  const ColWidths getColumnWidth(const size_t idx) const {
    CHECK_LT(idx, agg_col_widths_.size());
    return agg_col_widths_[idx];
  }

  const size_t getPaddedColumnWidthBytes(const size_t idx) const {
    CHECK_LT(idx, padded_agg_col_widths_.size());
    return padded_agg_col_widths_[idx];
  }

  /*
   * This function recompute and set all column widths used in the IR and result set.
   * padded_agg_col_widths_ are all initialized to be agg_col_widths_'s compact values,
   * unless they are re-evaluated using this method.
   * Currently, it is enabled when we have columnar output and not in distributed mode.
   * NOTE Re. distributed: since result sets are serialized using rowwise
   * iterators, we should only use actual column widths when we have full support of it
   * for both columnar and rowwise.
   * TODO(Saman): enable using actual widths in distributed mode.
   */
  void recomputePaddedColumnWidthBytes() {
    padded_agg_col_widths_.clear();
    padded_agg_col_widths_.reserve(agg_col_widths_.size());
    for (size_t col_idx = 0; col_idx < agg_col_widths_.size(); col_idx++) {
      padded_agg_col_widths_.push_back(
          (output_columnar_ && !g_cluster &&
           query_desc_type_ == QueryDescriptionType::Projection)
              ? getColumnWidth(col_idx).actual
              : getColumnWidth(col_idx).compact);
    }
  }

  size_t getPaddedColWidthForRange(const size_t offset, const size_t range) const {
    CHECK_LE(offset + range, agg_col_widths_.size());
    size_t ret = 0;
    for (size_t i = offset; i < offset + range; i++) {
      ret += agg_col_widths_[i].compact;
    }
    return ret;
  }
  size_t getRowWidth() const {
    // Note: Actual row size may include padding (see ResultSetBufferAccessors.h)
    size_t ret = 0;
    for (const auto& target_width : agg_col_widths_) {
      ret += target_width.compact;
    }
    return ret;
  }
  int8_t updateActualMinByteWidth(const int8_t actual_min_byte_width) const {
    int8_t ret = actual_min_byte_width;
    for (auto wids : agg_col_widths_) {
      ret = std::min(ret, wids.compact);
    }
    return ret;
  }

  void addAggColWidth(const ColWidths& col_width) {
    agg_col_widths_.push_back(col_width);
    padded_agg_col_widths_.push_back(col_width.compact);
  }

  void clearAggColWidths() {
    agg_col_widths_.clear();
    padded_agg_col_widths_.clear();
  }

  ssize_t getTargetGroupbyIndex(const size_t target_idx) const {
    CHECK_LT(target_idx, target_groupby_indices_.size());
    return target_groupby_indices_[target_idx];
  }
  size_t targetGroupbyIndicesSize() const { return target_groupby_indices_.size(); }
  void clearTargetGroupbyIndices() { target_groupby_indices_.clear(); }

  size_t getEntryCount() const { return entry_count_; }
  void setEntryCount(const size_t val) { entry_count_ = val; }

  int64_t getMinVal() const { return min_val_; }
  int64_t getMaxVal() const { return max_val_; }
  int64_t getBucket() const { return bucket_; }

  bool hasNulls() const { return has_nulls_; }
  GroupByMemSharing getGpuMemSharing() const { return sharing_; }

  const CountDistinctDescriptor getCountDistinctDescriptor(const size_t idx) const {
    CHECK_LT(idx, count_distinct_descriptors_.size());
    return count_distinct_descriptors_[idx];
  }
  size_t getCountDistinctDescriptorsSize() const {
    return count_distinct_descriptors_.size();
  }

  bool sortOnGpu() const { return sort_on_gpu_; }
  void setSortOnGpu(const bool val) { sort_on_gpu_ = val; }

  bool canOutputColumnar() const;
  bool didOutputColumnar() const { return output_columnar_; }
  void setOutputColumnar(const bool val) {
    output_columnar_ = val;
    // padded column widths are used for all columnar formats
    recomputePaddedColumnWidthBytes();
  }
  bool shouldOutputColumnar(const bool output_columnar_hint,
                            const bool has_count_distinct_target);

  bool mustUseBaselineSort() const { return must_use_baseline_sort_; }

  // TODO(adb): remove and store this info more naturally in another
  // member
  bool forceFourByteFloat() const { return force_4byte_float_; }
  void setForceFourByteFloat(const bool val) { force_4byte_float_ = val; }

  // Getters derived from state
  size_t getGroupbyColCount() const { return group_col_widths_.size(); }
  size_t getKeyCount() const { return keyless_hash_ ? 0 : getGroupbyColCount(); }
  size_t getBufferColSlotCount() const {
    if (target_groupby_indices_.empty()) {
      return agg_col_widths_.size();
    }
    return agg_col_widths_.size() - std::count_if(target_groupby_indices_.begin(),
                                                  target_groupby_indices_.end(),
                                                  [](const ssize_t i) { return i >= 0; });
  }

  size_t getBufferSizeBytes(const RelAlgExecutionUnit& ra_exe_unit,
                            const unsigned thread_count,
                            const ExecutorDeviceType device_type) const;
  size_t getBufferSizeBytes(const ExecutorDeviceType device_type) const;

  // TODO(alex): remove
  bool usesGetGroupValueFast() const;

  // TODO(alex): remove
  bool usesCachedContext() const;

  bool blocksShareMemory() const;
  bool threadsShareMemory() const;

  bool lazyInitGroups(const ExecutorDeviceType) const;

  bool interleavedBins(const ExecutorDeviceType) const;

  size_t sharedMemBytes(const ExecutorDeviceType) const;

  size_t getColOffInBytes(const size_t col_idx) const;
  size_t getColOffInBytesInNextBin(const size_t col_idx) const;
  size_t getNextColOffInBytes(const int8_t* col_ptr,
                              const size_t bin,
                              const size_t col_idx) const;
  size_t getColOnlyOffInBytes(const size_t col_idx) const;
  size_t getRowSize() const;
  size_t getColsSize() const;
  size_t getWarpCount() const;

  size_t getCompactByteWidth() const;

  inline size_t getEffectiveKeyWidth() const {
    return group_col_compact_width_ ? group_col_compact_width_ : sizeof(int64_t);
  }

  bool isWarpSyncRequired(const ExecutorDeviceType) const;

  std::string toString() const;

 protected:
  void resetGroupColWidths(const std::vector<int8_t>& new_group_col_widths) {
    group_col_widths_ = new_group_col_widths;
  }
  std::vector<ColWidths> agg_col_widths_;
  std::vector<size_t> padded_agg_col_widths_;

 private:
  const Executor* executor_;
  bool allow_multifrag_;
  QueryDescriptionType query_desc_type_;
  bool keyless_hash_;
  bool interleaved_bins_on_gpu_;
  int32_t idx_target_as_key_;
  int64_t init_val_;
  std::vector<int8_t> group_col_widths_;
  int8_t group_col_compact_width_;  // compact width for all group
                                    // cols if able to be consistent
                                    // otherwise 0
  std::vector<ssize_t> target_groupby_indices_;
  size_t entry_count_;  // the number of entries in the main buffer
  int64_t min_val_;     // meaningful for OneColKnownRange,
                        // MultiColPerfectHash only
  int64_t max_val_;
  int64_t bucket_;
  bool has_nulls_;
  GroupByMemSharing sharing_;  // meaningful for GPU only
  CountDistinctDescriptors count_distinct_descriptors_;
  bool sort_on_gpu_;
  bool output_columnar_;
  bool render_output_;
  bool must_use_baseline_sort_;

  bool force_4byte_float_;

  size_t getTotalBytesOfColumnarBuffers(const std::vector<ColWidths>& col_widths) const;
  size_t getTotalBytesOfColumnarBuffers(const std::vector<ColWidths>& col_widths,
                                        const size_t num_entries_per_column) const;
  size_t getTotalBytesOfColumnarProjections(const size_t projection_count) const;

  friend class ResultSet;
  friend class QueryExecutionContext;

  template <typename META_CLASS_TYPE>
  friend class AggregateReductionEgress;
};

#endif  // QUERYENGINE_QUERYMEMORYDESCRIPTOR_H
