/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    ResultSet.h
 * @brief   Basic constructors and methods of the row set interface.
 *
 */

#ifndef QUERYENGINE_RESULTSET_H
#define QUERYENGINE_RESULTSET_H

#include "CardinalityEstimator.h"
#include "ColumnBufferLayout.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "DataMgr/Chunk/Chunk.h"
#include "ResultSetBufferAccessors.h"
#include "ResultSetEntryFilter.h"
#include "ResultSetStorage.h"
#include "Shared/quantile.h"
#include "TargetValue.h"
#include "ThirdParty/robin_hood/robin_hood.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

/*
 * Stores the underlying buffer and the meta-data for a result set. The buffer
 * format reflects the main requirements for result sets. Not all queries
 * specify a GROUP BY clause, but since it's the most important and challenging
 * case we'll focus on it. Note that the meta-data is stored separately from
 * the buffer and it's not transferred to GPU.
 *
 * 1. It has to be efficient for reduction of partial GROUP BY query results
 *    from multiple devices / cores, the cardinalities can be high. Reduction
 *    currently happens on the host.
 * 2. No conversions should be needed when buffers are transferred from GPU to
 *    host for reduction. This implies the buffer needs to be "flat", with no
 *    pointers to chase since they have no meaning in a different address space.
 * 3. Must be size-efficient.
 *
 * There are several variations of the format of a result set buffer, but the
 * most common is a sequence of entries which represent a row in the result or
 * an empty slot. One entry looks as follows:
 *
 * +-+-+-+-+-+-+-+-+-+-+-+--?--+-+-+-+-+-+-+-+-+-+-+-+-+
 * |key_0| ... |key_N-1| padding |value_0|...|value_N-1|
 * +-+-+-+-+-+-+-+-+-+-+-+--?--+-+-+-+-+-+-+-+-+-+-+-+-+
 *
 * (key_0 ... key_N-1) is a multiple component key, unique within the buffer.
 * It stores the tuple specified by the GROUP BY clause. All components have
 * the same width, 4 or 8 bytes. For the 4-byte components, 4-byte padding is
 * added if the number of components is odd. Not all entries in the buffer are
 * valid; an empty entry contains EMPTY_KEY_{64, 32} for 8-byte / 4-byte width,
 * respectively. An empty entry is ignored by subsequent operations on the
 * result set (reduction, iteration, sort etc).
 *
 * value_0 through value_N-1 are 8-byte fields which hold the columns of the
 * result, like aggregates and projected expressions. They're reduced between
 * multiple partial results for identical (key_0 ... key_N-1) tuples.
 *
 * The order of entries is decided by the type of hash used, which depends on
 * the range of the keys. For small enough ranges, a perfect hash is used. When
 * a perfect hash isn't feasible, open addressing (using MurmurHash) with linear
 * probing is used instead, with a 50% fill rate.
 */

struct ReductionCode;

namespace Analyzer {

class Expr;
class Estimator;
struct OrderEntry;

}  // namespace Analyzer

class Executor;
class StringDictionaryProxy;

class ResultSet;

class ResultSetRowIterator {
 public:
  using value_type = std::vector<TargetValue>;
  using difference_type = std::ptrdiff_t;
  using pointer = std::vector<TargetValue>*;
  using reference = std::vector<TargetValue>&;
  using iterator_category = std::input_iterator_tag;

  bool operator==(const ResultSetRowIterator& other) const {
    return result_set_ == other.result_set_ &&
           crt_row_buff_idx_ == other.crt_row_buff_idx_;
  }
  bool operator!=(const ResultSetRowIterator& other) const { return !(*this == other); }

  inline value_type operator*() const;
  inline ResultSetRowIterator& operator++(void);
  ResultSetRowIterator operator++(int) {
    ResultSetRowIterator iter(*this);
    ++(*this);
    return iter;
  }

  size_t getCurrentRowBufferIndex() const {
    if (crt_row_buff_idx_ == 0) {
      throw std::runtime_error("current row buffer iteration index is undefined");
    }
    return crt_row_buff_idx_ - 1;
  }

 private:
  const ResultSet* result_set_;
  size_t crt_row_buff_idx_;
  size_t global_entry_idx_;
  bool global_entry_idx_valid_;
  size_t fetched_so_far_;
  bool translate_strings_;
  bool decimal_to_double_;

  ResultSetRowIterator(const ResultSet* rs,
                       bool translate_strings,
                       bool decimal_to_double)
      : result_set_(rs)
      , crt_row_buff_idx_(0)
      , global_entry_idx_(0)
      , global_entry_idx_valid_(false)
      , fetched_so_far_(0)
      , translate_strings_(translate_strings)
      , decimal_to_double_(decimal_to_double) {}

  ResultSetRowIterator(const ResultSet* rs) : ResultSetRowIterator(rs, false, false) {}

  friend class ResultSet;
};

class TSerializedRows;
class ResultSetBuilder;

using AppendedStorage = std::vector<std::unique_ptr<ResultSetStorage>>;
using PermutationIdx = uint32_t;
using Permutation = std::vector<PermutationIdx>;
using PermutationView = VectorView<PermutationIdx>;

struct DeferredLazyFetchChunkSource {
  ChunkKey chunk_key;
  size_t num_bytes;
  size_t num_elements;
};

class DeferredLazyFetchChunk {
 public:
  DeferredLazyFetchChunk(const ColumnDescriptor& column_descriptor,
                         Data_Namespace::DataMgr* data_mgr,
                         ChunkKey chunk_key,
                         size_t num_bytes,
                         size_t num_elements);
  DeferredLazyFetchChunk(const ColumnDescriptor& column_descriptor,
                         Data_Namespace::DataMgr* data_mgr,
                         std::vector<DeferredLazyFetchChunkSource> sources);
  ~DeferredLazyFetchChunk();

  void materialize(const int8_t*& buffer_slot) const;
  void materializeRows(const std::vector<int64_t>& local_row_indices,
                       const int8_t*& buffer_slot) const;
  void materializeRow(const int64_t local_row_idx, const int8_t*& buffer_slot) const;

 private:
  struct SourceState {
    ChunkKey chunk_key;
    size_t num_bytes;
    size_t num_elements;
    size_t byte_offset;
    size_t row_offset;
    std::shared_ptr<Chunk_NS::Chunk> chunk;
    size_t sparse_frame_size{0};
    std::vector<bool> materialized_frames;
  };

  void materializeFullyLocked(const int8_t*& buffer_slot) const;
  bool materializeFullyViaGpuLocked() const;
  bool materializeRowsLocked(const int64_t* local_row_indices,
                             size_t row_count,
                             const int8_t*& buffer_slot) const;

  const ColumnDescriptor column_descriptor_;
  Data_Namespace::DataMgr* const data_mgr_;
  mutable std::vector<SourceState> sources_;
  size_t num_bytes_{0};
  size_t num_elements_{0};
  mutable std::mutex mutex_;
  mutable std::vector<int8_t> materialized_buffer_;
  mutable const int8_t* buffer_{nullptr};
  mutable int8_t* sparse_buffer_{nullptr};
  mutable std::future<void> cpu_cache_future_;
  mutable std::atomic<bool> fully_materialized_{false};
};

using DeferredLazyFetchChunkPtr = std::shared_ptr<DeferredLazyFetchChunk>;
using DeferredLazyFetchChunkFragment = std::vector<DeferredLazyFetchChunkPtr>;
using DeferredLazyFetchChunks = std::vector<DeferredLazyFetchChunkFragment>;
using DeferredLazyFetchChunkStorages = std::vector<DeferredLazyFetchChunks>;

struct LazyFetchSourceMetadataEntry {
  std::shared_ptr<const ChunkMetadata> chunk_metadata;
  SQLTypeInfo source_type;
};
using LazyFetchSourceMetadata =
    std::unordered_map<size_t, std::vector<LazyFetchSourceMetadataEntry>>;

// Common base class to ResultSetComparator template specializations.
class ResultSetComparatorBase {
 public:
  virtual ~ResultSetComparatorBase() = default;
};

class ResultSet {
 public:
  friend ResultSetBuilder;
  // Can use derivatives of the builder class to construct a ResultSet

  ResultSet(const std::vector<TargetInfo>& targets,
            const ExecutorDeviceType device_type,
            const QueryMemoryDescriptor& query_mem_desc,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
            const unsigned block_size,
            const unsigned grid_size);

  ResultSet(const std::vector<TargetInfo>& targets,
            const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
            const std::vector<std::vector<const int8_t*>>& col_buffers,
            const ColumnBufferLayouts& col_buffer_layouts,
            const std::vector<std::vector<int64_t>>& frag_offsets,
            const std::vector<int64_t>& consistent_frag_sizes,
            const ExecutorDeviceType device_type,
            const int device_id,
            const int thread_idx,
            const QueryMemoryDescriptor& query_mem_desc,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
            const unsigned block_size,
            const unsigned grid_size);

  ResultSet(const std::shared_ptr<const Analyzer::Estimator>,
            const ExecutorDeviceType device_type,
            const int device_id,
            Data_Namespace::DataMgr* data_mgr,
            std::shared_ptr<CudaAllocator> device_allocator);

  ResultSet(const std::string& explanation);

  ResultSet(int64_t queue_time_ms,
            int64_t render_time_ms,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  ~ResultSet();

  std::string toString() const {
    return typeName(this) + "(targets=" + ::toString(targets_) +
           ", query_mem_desc=" + ::toString(query_mem_desc_) + ")";
  }

  std::string summaryToString() const;

  inline ResultSetRowIterator rowIterator(size_t from_logical_index,
                                          bool translate_strings,
                                          bool decimal_to_double) const {
    ResultSetRowIterator rowIterator(this, translate_strings, decimal_to_double);

    // move to first logical position
    ++rowIterator;

    for (size_t index = 0; index < from_logical_index; index++) {
      ++rowIterator;
    }

    return rowIterator;
  }

  inline ResultSetRowIterator rowIterator(bool translate_strings,
                                          bool decimal_to_double) const {
    return rowIterator(0, translate_strings, decimal_to_double);
  }

  ExecutorDeviceType getDeviceType() const;

  const ResultSetStorage* allocateStorage() const;

  const ResultSetStorage* allocateStorage(int8_t*,
                                          const std::vector<int64_t>&,
                                          std::shared_ptr<VarlenOutputInfo> = nullptr,
                                          size_t provided_buffer_size_bytes = 0) const;

  const ResultSetStorage* allocateStorage(const std::vector<int64_t>&) const;

  void updateStorageEntryCount(const size_t new_entry_count) {
    CHECK(query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection ||
          query_mem_desc_.getQueryDescriptionType() ==
              QueryDescriptionType::TableFunction ||
          query_mem_desc_.getQueryDescriptionType() ==
              QueryDescriptionType::GroupByBaselineHash);
    query_mem_desc_.setEntryCount(new_entry_count);
    CHECK(storage_);
    storage_->updateEntryCount(new_entry_count);
    invalidateCachedRowCount();
  }

  std::vector<TargetValue> getNextRow(const bool translate_strings,
                                      const bool decimal_to_double) const;

  size_t getCurrentRowBufferIndex() const;

  std::vector<TargetValue> getRowAt(const size_t index) const;

  TargetValue getRowAt(const size_t row_idx,
                       const size_t col_idx,
                       const bool translate_strings,
                       const bool decimal_to_double = true) const;

  // Specialized random access getter for result sets with a single column to
  // avoid the overhead of building a std::vector<TargetValue> result with only
  // one element. Only used by RelAlgTranslator::getInIntegerSetExpr currently.
  OneIntegerColumnRow getOneColRow(const size_t index) const;

  std::vector<TargetValue> getRowAtNoTranslations(
      const size_t index,
      const std::vector<bool>& targets_to_skip = {}) const;

  bool isRowAtEmpty(const size_t index) const;

  void sort(const std::list<Analyzer::OrderEntry>& order_entries,
            size_t top_n,
            const ExecutorDeviceType device_type,
            Executor* executor,
            bool need_to_initialize_device_ids_to_use = false);

  void keepFirstN(const size_t n);

  void dropFirstN(const size_t n);

  void append(ResultSet& that);

  bool hasStorage() const { return static_cast<bool>(storage_); }

  bool hasMultipleStorages() const {
    return (storage_ ? size_t(1) : size_t(0)) + appended_storage_.size() > size_t(1);
  }

  const ResultSetStorage* getStorage() const;

  size_t colCount() const;

  SQLTypeInfo getColType(const size_t col_idx) const;

  /**
   * @brief Returns the number of valid entries in the result set (i.e that will
   * be returned from the SQL query or inputted into the next query step)
   *
   * Note that this can be less than or equal to the value returned by
   * ResultSet::getEntries(), whether due to a SQL LIMIT/OFFSET applied or because
   * the result set representation is inherently sparse (i.e. baseline hash group by).
   *
   * Internally this function references/sets a cached value (`cached_row_count_`)
   * so that the cost of computing the result is only paid once per result set.
   *
   * If the actual row count is not cached and needs to be computed, in some cases
   * that can be O(1) (i.e. if limits and offsets are present, or for the output
   * of a table function). For projections, we use a binary search, so it is
   * O(log n), otherwise it is O(n) (with n being ResultSet::entryCount()),
   * which will be run in parallel if the entry count >= the default of 20000
   * or if `force_parallel` is set to true
   *
   * Note that we currently do not invalidate the cache if the result set is changed
   * (i.e appended to), so this function should only be called after the result
   * set is finalized.
   *
   * @param force_parallel Forces the row count to be computed in parallel if
   * the row count cannot be otherwise be computed from metadata or via a binary
   * search (otherwise parallel search is automatically used for result sets
   * with `entryCount() >= 20000`)
   *
   */

  size_t rowCount(const bool force_parallel = false) const;

  void invalidateCachedRowCount() const;

  void setCachedRowCount(const size_t row_count) const;

  /**
   * @brief Returns a boolean signifying whether there are valid entries
   * in the result set.
   *
   * Note a result set can be logically empty even if the value returned by
   * `ResultSet::entryCount()` is > 0, whether due to a SQL LIMIT/OFFSET applied or
   * because the result set representation is inherently sparse (i.e. baseline hash group
   * by).
   *
   * Internally this function is just implemented as `ResultSet::rowCount() == 0`, which
   * caches it's value so the row count will only be computed once per finalized result
   * set.
   *
   */

  bool isEmpty() const;

  /**
   * @brief Returns the number of entries the result set is allocated to hold.
   *
   * Note that this can be greater than or equal to the actual number of valid rows
   * in the result set, whether due to a SQL LIMIT/OFFSET applied or because
   * the result set representation is inherently sparse (i.e. baseline hash group by)
   *
   * For getting the number of valid rows in the result set (inclusive
   * of any applied LIMIT and/or OFFSET), use `ResultSet::rowCount().` Or
   * to just test if there are any valid rows, use `ResultSet::entryCount()`,
   * as a return value from `entryCount()` greater than 0 does not neccesarily
   * mean the result set is empty.
   *
   */

  size_t entryCount() const;

  size_t getBufferSizeBytes(const ExecutorDeviceType device_type) const;

  bool definitelyHasNoRows() const;

  const QueryMemoryDescriptor& getQueryMemDesc() const;

  const std::vector<TargetInfo>& getTargetInfos() const;

  const std::vector<int64_t>& getTargetInitVals() const;

  int8_t* getDeviceEstimatorBuffer() const;

  int8_t* getHostEstimatorBuffer() const;

  void syncEstimatorBuffer() const;

  size_t getNDVEstimator() const;

  struct QueryExecutionTimings {
    // all in ms
    int64_t executor_queue_time{0};
    int64_t render_time{0};
    int64_t compilation_queue_time{0};
    int64_t kernel_queue_time{0};
  };

  void setQueueTime(const int64_t queue_time);
  void setKernelQueueTime(const int64_t kernel_queue_time);
  void addCompilationQueueTime(const int64_t compilation_queue_time);

  int64_t getQueueTime() const;
  int64_t getRenderTime() const;

  void moveToBegin() const;

  bool isTruncated() const;

  bool isExplain() const;

  void setValidationOnlyRes();
  bool isValidationOnlyRes() const;

  std::string getExplanation() const {
    if (just_explain_) {
      return explanation_;
    }
    return {};
  }

  bool isGeoColOnGpu(const size_t col_idx) const;
  int getDeviceId() const;
  int getThreadIdx() const;

  // Materialize string from StringDictionaryProxy
  std::string getString(SQLTypeInfo const&, int64_t const ival) const;

  // Called from the executor because in the new ResultSet we assume the 'padded' field
  // in SlotSize already contains the padding, whereas in the executor it's computed.
  // Once the buffer initialization moves to ResultSet we can remove this method.
  static QueryMemoryDescriptor fixupQueryMemoryDescriptor(const QueryMemoryDescriptor&);

  // Convert int64_t to ScalarTargetValue based on SQLTypeInfo and translate_strings.
  ScalarTargetValue convertToScalarTargetValue(SQLTypeInfo const&,
                                               bool const translate_strings,
                                               int64_t const val) const;

  // Called from ResultSetComparator<>::operator().
  bool isLessThan(SQLTypeInfo const&, int64_t const lhs, int64_t const rhs) const;

  // Required for sql_validate calls.
  static bool isNullIval(SQLTypeInfo const&,
                         bool const translate_strings,
                         int64_t const ival);

  // Return NULL ScalarTargetValue based on SQLTypeInfo and translate_strings.
  static ScalarTargetValue nullScalarTargetValue(SQLTypeInfo const&,
                                                 bool const translate_strings);

  void fillOneEntry(const std::vector<int64_t>& entry) {
    CHECK(storage_);
    if (storage_->query_mem_desc_.didOutputColumnar()) {
      storage_->fillOneEntryColWise(entry);
    } else {
      storage_->fillOneEntryRowWise(entry);
    }
  }

  void initializeStorage() const;

  void holdChunks(const std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks) {
    chunks_ = chunks;
  }
  void setDeferredLazyFetchChunks(
      const DeferredLazyFetchChunks& deferred_lazy_fetch_chunks);
  bool hasDeferredLazyFetchChunks() const { return !deferred_lazy_fetch_chunks_.empty(); }
  void materializeDeferredLazyFetchColumns(
      const std::vector<size_t>& target_logical_indices) const;
  // Materialize only source rows referenced by non-empty physical result entries.
  void materializeDeferredLazyFetchColumnsForAllRows(
      const std::vector<size_t>& target_logical_indices) const;
  void materializeDeferredLazyFetchColumnsForOutputRows(
      const std::vector<size_t>& target_logical_indices) const;
  void materializeDeferredLazyFetchColumn(const size_t target_logical_idx) const;
  void setLazyFetchSourceMetadata(
      const LazyFetchSourceMetadata& lazy_fetch_source_metadata);
  bool hasLazyFetchSourceMetadata() const { return !lazy_fetch_source_metadata_.empty(); }
  std::vector<LazyFetchSourceMetadataEntry> getLazyFetchSourceMetadata(
      const size_t target_logical_idx) const;
  void holdChunkIterators(const std::shared_ptr<std::list<ChunkIter>> chunk_iters) {
    chunk_iters_.push_back(chunk_iters);
  }
  void holdLiterals(std::vector<int8_t>& literal_buff) {
    literal_buffers_.push_back(std::move(literal_buff));
  }

  std::shared_ptr<RowSetMemoryOwner> getRowSetMemOwner() const {
    return row_set_mem_owner_;
  }

  const Permutation& getPermutationBuffer() const;
  const bool isPermutationBufferEmpty() const { return permutation_.empty(); };

  void serialize(TSerializedRows& serialized_rows) const;

  static std::unique_ptr<ResultSet> unserialize(const TSerializedRows& serialized_rows,
                                                const Executor*);

  size_t getLimit() const;

  // APIs for data recycler
  ResultSetPtr copyForCacheInsertion();
  ResultSetPtr copyForCacheRetrieval();

  ResultSetPtr compactBaselineHashForReduction(
      const size_t min_compaction_entry_count = 1000000000,
      const ResultSetEntryFilter* entry_filter = nullptr) const;
  ResultSetPtr extractAndClearBaselineHashEntries(
      const std::vector<int64_t>& keys,
      const bool retain_device_rowwise_for_post_filter = false);
  void markBaselineHashDenseForReduction(const size_t row_count) {
    CHECK(query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash);
    baseline_hash_dense_for_reduction_ = true;
    setCachedRowCount(row_count);
  }
  bool isBaselineHashDenseForReduction() const {
    return baseline_hash_dense_for_reduction_;
  }
  void clearBaselineHashDenseForReduction() {
    baseline_hash_dense_for_reduction_ = false;
  }

  void clearPermutation() {
    if (!permutation_.empty()) {
      permutation_.clear();
    }
  }

  void initStatus() {
    // todo(yoonmin): what else we additionally need to consider
    // to make completely clear status of the resultset for reuse?
    crt_row_buff_idx_ = 0;
    fetched_so_far_ = 0;
    clearPermutation();
    setGeoReturnType(ResultSet::GeoReturnType::WktString);
    invalidateCachedRowCount();
    drop_first_ = 0;
    keep_first_ = 0;
  }

  void invalidateResultSetChunks() {
    if (!chunks_.empty()) {
      chunks_.clear();
    }
    if (!chunk_iters_.empty()) {
      chunk_iters_.clear();
    }
  };

  const bool isEstimator() const { return !estimator_; }

  void setCached(bool val) { cached_ = val; }

  const bool isCached() const { return cached_; }

  void setExecTime(const long exec_time) { query_exec_time_ = exec_time; }

  const long getExecTime() const { return query_exec_time_; }

  void setQueryPlanHash(const QueryPlanHash query_plan) { query_plan_ = query_plan; }

  const QueryPlanHash getQueryPlanHash() { return query_plan_; }

  std::unordered_set<size_t> getInputTableKeys() const { return input_table_keys_; }

  void setInputTableKeys(std::unordered_set<size_t>&& intput_table_keys) {
    input_table_keys_ = std::move(intput_table_keys);
  }

  void setTargetMetaInfo(const std::vector<TargetMetaInfo>& target_meta_info) {
    std::copy(target_meta_info.begin(),
              target_meta_info.end(),
              std::back_inserter(target_meta_info_));
  }

  std::vector<TargetMetaInfo> getTargetMetaInfo() { return target_meta_info_; }

  std::optional<bool> canUseSpeculativeTopNSort() const {
    return can_use_speculative_top_n_sort;
  }

  void setUseSpeculativeTopNSort(bool value) { can_use_speculative_top_n_sort = value; }

  const bool hasValidBuffer() const {
    if (storage_) {
      return true;
    }
    return false;
  }

  unsigned getBlockSize() const { return block_size_; }

  unsigned getGridSize() const { return grid_size_; }

  /**
   * Geo return type options when accessing geo columns from a result set.
   */
  enum class GeoReturnType {
    GeoTargetValue,      /**< Copies the geo data into a struct of vectors - coords are
                            uncompressed */
    WktString,           /**< Returns the geo data as a WKT string */
    GeoTargetValuePtr,   /**< Returns only the pointers of the underlying buffers for the
                            geo data. */
    GeoTargetValueGpuPtr /**< If geo data is currently on a device, keep the data on the
                            device and return the device ptrs */
  };
  GeoReturnType getGeoReturnType() const { return geo_return_type_; }
  void setGeoReturnType(const GeoReturnType val) { geo_return_type_ = val; }

  void copyColumnIntoBuffer(const size_t column_idx,
                            int8_t* output_buffer,
                            const size_t output_buffer_size) const;

  bool isDirectColumnarConversionPossible() const;

  bool didOutputColumnar() const { return this->query_mem_desc_.didOutputColumnar(); }

  bool isZeroCopyColumnarConversionPossible(size_t column_idx) const;
  const int8_t* getColumnarBuffer(size_t column_idx) const;
  const size_t getColumnarBufferSize(size_t column_idx) const;
  using ColumnarBufferFragment = std::pair<const int8_t*, size_t>;
  struct DeviceColumnarBufferFragment {
    const int8_t* buffer;
    size_t entry_count;
    int device_id;
    std::shared_ptr<CudaAllocator> owner;
    std::shared_ptr<CudaStreamReadyEvent> ready_event;
  };
  struct DeviceColumnarFragmentInfo {
    size_t entry_count;
    int device_id;
  };
  struct DeviceRowwiseBufferFragment {
    const int8_t* buffer;
    size_t entry_count;
    int device_id;
    std::shared_ptr<CudaAllocator> owner;
    std::shared_ptr<CudaStreamReadyEvent> ready_event;
  };
  void addDeviceColumnarBufferFragment(size_t column_idx,
                                       const int device_id,
                                       const int8_t* buffer,
                                       const size_t entry_count);
  void clearDeviceColumnarBufferFragments();
  void markDeviceColumnarFragmentsCoverLogicalRows() const;
  void markDeviceColumnarFragmentsFormDenseCpuRows() const;
  void markDeviceColumnarFragmentsExcludeBaselineBoundaryKeys() const;
  void addDeviceRowwiseBufferFragment(const int device_id,
                                      const int8_t* buffer,
                                      const size_t entry_count);
  void clearDeviceRowwiseBufferFragments();
  bool getDeviceRowwiseBufferFragments(
      std::vector<DeviceRowwiseBufferFragment>& fragments) const;
  bool canDeferDeviceColumnarCpuMaterialization() const;
  void markDeviceColumnarCpuStorageInvalid() const;
  void markDeviceColumnarCpuStorageValid() const;
  void markEntryFilterApplied() { entry_filter_applied_ = true; }
  bool isEntryFilterApplied() const { return entry_filter_applied_; }
  void markSparseBaselineEntryFilterAppliedBeforeCopy() {
    sparse_baseline_entry_filter_applied_before_copy_ = true;
  }
  bool wasSparseBaselineEntryFilterAppliedBeforeCopy() const {
    return sparse_baseline_entry_filter_applied_before_copy_;
  }
  void markDeviceColumnarFragmentsCoverCpuBaselineBoundaryRows() const;
  void materializeDeviceColumnarCpuStorageIfNeeded() const;
  bool getDeviceColumnarBufferFragments(
      size_t column_idx,
      size_t elem_size,
      std::vector<DeviceColumnarBufferFragment>& fragments) const;
  bool appendDeviceColumnarFragmentsFromCpuBaselineHashResult(const ResultSet& source);
  bool appendDeviceOnlyColumnarFragmentsFromCpuBaselineHashResult(
      const ResultSet& source);
  bool getDeviceColumnarFragmentInfo(
      std::vector<DeviceColumnarFragmentInfo>& fragments) const;
  bool getColumnarFragmentRowCounts(std::vector<size_t>& row_counts) const;
  bool getColumnarBufferFragments(size_t column_idx,
                                  size_t elem_size,
                                  std::vector<ColumnarBufferFragment>& fragments) const;

  QueryDescriptionType getQueryDescriptionType() const {
    return query_mem_desc_.getQueryDescriptionType();
  }

  const int8_t getPaddedSlotWidthBytes(const size_t slot_idx) const {
    return query_mem_desc_.getPaddedSlotWidthBytes(slot_idx);
  }

  // returns a bitmap of all single-slot targets, as well as its count
  std::tuple<std::vector<bool>, size_t> getSingleSlotTargetBitmap() const;

  std::tuple<std::vector<bool>, size_t> getSupportedSingleSlotTargetBitmap() const;

  std::vector<size_t> getSlotIndicesForTargetIndices() const;

  const std::vector<ColumnLazyFetchInfo>& getLazyFetchInfo() const {
    return lazy_fetch_info_;
  }

  bool areAnyColumnsLazyFetched() const {
    auto is_lazy = [](auto const& info) { return info.is_lazily_fetched; };
    return std::any_of(lazy_fetch_info_.begin(), lazy_fetch_info_.end(), is_lazy);
  }

  size_t getNumColumnsLazyFetched() const {
    auto is_lazy = [](auto const& info) { return info.is_lazily_fetched; };
    return std::count_if(lazy_fetch_info_.begin(), lazy_fetch_info_.end(), is_lazy);
  }

  void setSeparateVarlenStorageValid(const bool val) {
    separate_varlen_storage_valid_ = val;
  }

  const std::vector<std::string> getStringDictionaryPayloadCopy(
      const shared::StringDictKey& dict_key) const;

  using UniqueStringsForDictEncodedTargetCol =
      std::pair<std::vector<int32_t>, std::vector<std::string>>;

  const UniqueStringsForDictEncodedTargetCol getUniqueStringsForDictEncodedTargetCol(
      const size_t col_idx) const;

  std::vector<UniqueStringsForDictEncodedTargetCol>
  getUniqueStringsForDictEncodedTargetCols(const std::vector<size_t>& col_indices) const;

  StringDictionaryProxy* getStringDictionaryProxy(
      const shared::StringDictKey& dict_key) const;

  template <typename ENTRY_TYPE, QueryDescriptionType QUERY_TYPE, bool COLUMNAR_FORMAT>
  ENTRY_TYPE getEntryAt(const size_t row_idx,
                        const size_t target_idx,
                        const size_t slot_idx) const;

  ChunkStats getTableFunctionChunkStats(const size_t target_idx) const;

  static double calculateQuantile(quantile::TDigest* const t_digest);

  void translateDictEncodedColumns(std::vector<TargetInfo> const&,
                                   size_t const start_idx);

  struct RowIterationState {
    size_t prev_target_idx_{0};
    size_t cur_target_idx_;
    size_t agg_idx_{0};
    int8_t const* buf_ptr_{nullptr};
    int8_t compact_sz1_;
  };

  class CellCallback;
  void eachCellInColumn(RowIterationState&, CellCallback const&);

  const Executor* getExecutor() const { return query_mem_desc_.getExecutor(); }

  bool checkSlotUsesFlatBufferFormat(const size_t slot_idx) const {
    return query_mem_desc_.checkSlotUsesFlatBufferFormat(slot_idx);
  }

  void setCudaAllocator(const Executor* executor, int device_id);
  void setCudaAllocator(std::shared_ptr<CudaAllocator> cuda_allocator);

  CudaAllocator* getCudaAllocator() const;

  bool hasDeviceBufferOwnership() const;

  void setCudaStream(const Executor* executor, int device_id);

  CUstream getCudaStream() const;

  /**
   * @brief Fetches and materializes a lazily-fetched column value into a provided buffer.
   *
   * This function retrieves a lazily-fetched column value for a specific entry and
   * column, decodes it if necessary, and writes the result to the provided output buffer.
   * It is meant as a faster alternative to normal result fetching with the
   * ResultSet::getRowAt function, which has significant overhead by going through the
   * boost variant interface to access data.
   *
   * @param global_entry_idx The global index of the entry to fetch.
   * @param col_idx The index of the column to fetch.
   * @param output_ptr Pointer to the buffer where the fetched value will be written.
   *
   * @note This function supports various data types including boolean, integer types,
   *       floating-point types, temporal types, and dictionary-encoded text types.
   *       For non-dictionary-encoded text types, it will throw an exception.
   *       It also does not support flatbuffer storage.
   *
   */

  template <typename T>
  void fetchLazyColumnValue(const size_t global_entry_idx,
                            const size_t col_index,
                            T* output_ptr) const;

 private:
  enum class DeferredLazyFetchRowSelection { AllNonEmptyRows, OutputRows };

  void materializeDeferredLazyFetchColumnsForRows(
      const std::vector<size_t>& target_logical_indices,
      const DeferredLazyFetchRowSelection row_selection) const;
  void markDeferredLazyFetchColumnsMaterializedForAllRows(
      const std::vector<size_t>& target_logical_indices) const;
  bool isDeferredLazyFetchColumnMaterializedForAllRows(size_t target_logical_idx) const;

  ResultSetPtr copyForCache();

  void advanceCursorToNextEntry(ResultSetRowIterator& iter) const;

  std::vector<TargetValue> getNextRowImpl(const bool translate_strings,
                                          const bool decimal_to_double) const;

  std::vector<TargetValue> getNextRowUnlocked(const bool translate_strings,
                                              const bool decimal_to_double) const;

  std::vector<TargetValue> getRowAt(const size_t index,
                                    const bool translate_strings,
                                    const bool decimal_to_double,
                                    const bool fixup_count_distinct_pointers,
                                    const std::vector<bool>& targets_to_skip = {}) const;

  // NOTE: just for direct columnarization use at the moment
  template <typename ENTRY_TYPE>
  ENTRY_TYPE getColumnarPerfectHashEntryAt(const size_t row_idx,
                                           const size_t target_idx,
                                           const size_t slot_idx) const;

  template <typename ENTRY_TYPE>
  ENTRY_TYPE getRowWisePerfectHashEntryAt(const size_t row_idx,
                                          const size_t target_idx,
                                          const size_t slot_idx) const;

  template <typename ENTRY_TYPE>
  ENTRY_TYPE getRowWiseBaselineEntryAt(const size_t row_idx,
                                       const size_t target_idx,
                                       const size_t slot_idx) const;

  template <typename ENTRY_TYPE>
  ENTRY_TYPE getColumnarBaselineEntryAt(const size_t row_idx,
                                        const size_t target_idx,
                                        const size_t slot_idx) const;

  size_t binSearchRowCount() const;

  size_t parallelRowCount() const;

  size_t advanceCursorToNextEntry() const;

  void radixSortOnGpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  void radixSortOnCpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  static bool isNull(const SQLTypeInfo& ti,
                     const InternalTargetValue& val,
                     const bool float_argument_input);

  TargetValue getTargetValueFromBufferRowwise(
      int8_t* rowwise_target_ptr,
      int8_t* keys_ptr,
      const QueryMemoryDescriptor& query_mem_desc,
      const size_t entry_buff_idx,
      const TargetInfo& target_info,
      const size_t target_logical_idx,
      const size_t slot_idx,
      const bool translate_strings,
      const bool decimal_to_double,
      const bool fixup_count_distinct_pointers) const;

  TargetValue getTargetValueFromBufferColwise(const int8_t* col_ptr,
                                              const int8_t* keys_ptr,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              const size_t local_entry_idx,
                                              const size_t global_entry_idx,
                                              const TargetInfo& target_info,
                                              const size_t target_logical_idx,
                                              const size_t slot_idx,
                                              const bool translate_strings,
                                              const bool decimal_to_double) const;

  TargetValue makeTargetValue(const int8_t* ptr,
                              const int8_t compact_sz,
                              const QueryMemoryDescriptor& query_mem_desc,
                              const TargetInfo& target_info,
                              const size_t target_logical_idx,
                              const bool translate_strings,
                              const bool decimal_to_double,
                              const size_t entry_buff_idx) const;

  ScalarTargetValue makeStringTargetValue(
      SQLTypeInfo const& chosen_type,
      bool const translate_strings,
      int64_t const ival,
      std::optional<size_t> target_logical_idx = std::nullopt) const;

  TargetValue makeVarlenTargetValue(const int8_t* ptr1,
                                    const int8_t compact_sz1,
                                    const int8_t* ptr2,
                                    const int8_t compact_sz2,
                                    const TargetInfo& target_info,
                                    const size_t target_logical_idx,
                                    const bool translate_strings,
                                    const size_t entry_buff_idx) const;

  struct VarlenTargetPtrPair {
    int8_t* ptr1;
    int8_t compact_sz1;
    int8_t* ptr2;
    int8_t compact_sz2;

    VarlenTargetPtrPair()
        : ptr1(nullptr), compact_sz1(0), ptr2(nullptr), compact_sz2(0) {}
  };
  TargetValue makeGeoTargetValue(const int8_t* geo_target_ptr,
                                 const size_t slot_idx,
                                 const TargetInfo& target_info,
                                 const size_t target_logical_idx,
                                 const size_t entry_buff_idx) const;

  struct StorageLookupResult {
    const ResultSetStorage* storage_ptr;
    const size_t fixedup_entry_idx;
    const size_t storage_idx;
  };

  struct ColumnFragmentLookupResult {
    size_t storage_idx;
    size_t fragment_idx;
    int64_t local_row_idx;
  };

  InternalTargetValue getVarlenOrderEntry(const int64_t str_ptr,
                                          const size_t str_len) const;

  int64_t lazyReadInt(const int64_t ival,
                      const size_t target_logical_idx,
                      const StorageLookupResult& storage_lookup_result) const;

  /// Returns (storageIdx, entryIdx) pair, where:
  /// storageIdx : 0 is storage_, storageIdx-1 is index into appended_storage_.
  /// entryIdx   : local index into the storage object.
  std::pair<size_t, size_t> getStorageIndex(const size_t entry_idx) const;

  const std::vector<const int8_t*>& getColumnFrag(const size_t storge_idx,
                                                  const size_t col_logical_idx,
                                                  const int local_col_id,
                                                  int64_t& global_idx) const;
  ColumnFragmentLookupResult resolveColumnFragment(const size_t storage_idx,
                                                   const size_t col_logical_idx,
                                                   const int local_col_id,
                                                   const int64_t global_idx) const;

  const VarlenOutputInfo* getVarlenOutputInfo(const size_t entry_idx) const;

  StorageLookupResult findStorage(const size_t entry_idx) const;

  struct TargetOffsets {
    const int8_t* ptr1;
    const size_t compact_sz1;
    const int8_t* ptr2;
    const size_t compact_sz2;
    const size_t slot_idx;
  };

  struct RowWiseStorageOffsets {
    std::vector<TargetOffsets> target_offsets;
    size_t row_bytes;
    size_t key_width;
    size_t key_bytes_with_padding;
  };

  struct RowWiseTargetAccessor {
    RowWiseTargetAccessor(const ResultSet* result_set) : result_set_(result_set) {
      initializeOffsetsForStorage();
    }

    InternalTargetValue getColumnInternal(
        const int8_t* buff,
        const size_t entry_idx,
        const size_t target_logical_idx,
        const StorageLookupResult& storage_lookup_result) const;

    void initializeOffsetsForStorage();

    inline const int8_t* get_rowwise_ptr(
        const int8_t* buff,
        const size_t entry_idx,
        const RowWiseStorageOffsets& storage_offsets) const {
      return buff + entry_idx * storage_offsets.row_bytes;
    }

    std::vector<RowWiseStorageOffsets> offsets_for_storage_;

    const ResultSet* result_set_;
  };

  struct ColumnWiseTargetAccessor {
    ColumnWiseTargetAccessor(const ResultSet* result_set) : result_set_(result_set) {
      initializeOffsetsForStorage();
    }

    void initializeOffsetsForStorage();

    InternalTargetValue getColumnInternal(
        const int8_t* buff,
        const size_t entry_idx,
        const size_t target_logical_idx,
        const StorageLookupResult& storage_lookup_result) const;

    std::vector<std::vector<TargetOffsets>> offsets_for_storage_;

    const ResultSet* result_set_;
  };

  using ApproxQuantileBuffers = std::vector<std::vector<double>>;
  using ModeBuffers = std::vector<std::vector<int64_t>>;

  /**
   * @brief Base class for materialized sort buffers
   * We need a base class so we can store a pointer to the
   * non-BUFFER_ITERATOR_TYPE templated base class in the ResultSet class
   */
  class MaterializedSortBuffersBase {
   public:
    struct TopNDictionarySortContext {
      size_t candidate_count;
      size_t top_n;
    };

    MaterializedSortBuffersBase(
        const ResultSet* result_set,
        const std::list<Analyzer::OrderEntry>& order_entries,
        bool single_threaded,
        std::optional<size_t> compact_permutation_size,
        std::optional<TopNDictionarySortContext> top_n_dictionary_sort_context)
        : result_set_(result_set)
        , order_entries_(order_entries)
        , single_threaded_(single_threaded)
        , compact_permutation_size_(compact_permutation_size)
        , top_n_dictionary_sort_context_(top_n_dictionary_sort_context) {}

    virtual ~MaterializedSortBuffersBase() = default;

    struct DictionaryStringSortPermutation {
      using LocalStringRankMap = robin_hood::unordered_flat_map<int32_t, int32_t>;

      explicit DictionaryStringSortPermutation(SortedStringPermutation global_permutation)
          : global_permutation_(std::move(global_permutation)) {}

      explicit DictionaryStringSortPermutation(LocalStringRankMap local_string_id_to_rank)
          : local_string_id_to_rank_(std::move(local_string_id_to_rank)) {}

      DictionaryStringSortPermutation(
          const StringDictionaryProxy* string_dictionary_proxy,
          const bool notnull,
          std::optional<int64_t> translated_null)
          : string_dictionary_proxy_(string_dictionary_proxy)
          , notnull_(notnull)
          , translated_null_(translated_null) {}

      bool operator()(int32_t lhs,
                      int32_t rhs,
                      bool sort_descending,
                      bool nulls_first) const;

     private:
      std::optional<SortedStringPermutation> global_permutation_;
      LocalStringRankMap local_string_id_to_rank_;
      const StringDictionaryProxy* string_dictionary_proxy_{nullptr};
      bool notnull_{false};
      std::optional<int64_t> translated_null_;
    };

    const std::vector<DictionaryStringSortPermutation>&
    getDictionaryEncodedSortPermutations() const {
      return dictionary_string_sorted_permutations_;
    }
    const std::vector<std::vector<int64_t>>& getCountDistinctBuffers() const {
      return count_distinct_materialized_buffers_;
    }
    const ApproxQuantileBuffers& getApproxQuantileBuffers() const {
      return approx_quantile_materialized_buffers_;
    }
    const ModeBuffers& getModeBuffers() const { return mode_buffers_; }

   protected:
    virtual void materializeBuffers() = 0;

    const ResultSet* result_set_;
    const std::list<Analyzer::OrderEntry>& order_entries_;
    const bool single_threaded_;
    const std::optional<size_t> compact_permutation_size_;
    const std::optional<TopNDictionarySortContext> top_n_dictionary_sort_context_;

    std::vector<DictionaryStringSortPermutation> dictionary_string_sorted_permutations_;
    std::vector<std::vector<int64_t>> count_distinct_materialized_buffers_;
    ApproxQuantileBuffers approx_quantile_materialized_buffers_;
    ModeBuffers mode_buffers_;
  };

  /**
   * @brief Templated class that actually handles the materialization of sort buffers,
   * templated by BUFFER_ITERATOR_TYPE, which is either RowWiseTargetAccessor or
   * ColumnWiseTargetAccessor
   */
  template <typename BUFFER_ITERATOR_TYPE>
  class MaterializedSortBuffers : public ResultSet::MaterializedSortBuffersBase {
   public:
    using BufferIteratorType = BUFFER_ITERATOR_TYPE;

    MaterializedSortBuffers(
        const ResultSet* result_set,
        const std::list<Analyzer::OrderEntry>& order_entries,
        bool single_threaded,
        std::optional<size_t> compact_permutation_size,
        std::optional<TopNDictionarySortContext> top_n_dictionary_sort_context)
        : MaterializedSortBuffersBase(result_set,
                                      order_entries,
                                      single_threaded,
                                      compact_permutation_size,
                                      top_n_dictionary_sort_context)
        , buffer_itr_(result_set) {
      materializeBuffers();
    }

   protected:
    void materializeBuffers() override {
      dictionary_string_sorted_permutations_ =
          materializeDictionaryEncodedSortPermutations();
      count_distinct_materialized_buffers_ = materializeCountDistinctColumns();
      approx_quantile_materialized_buffers_ = materializeApproxQuantileColumns();
      mode_buffers_ = materializeModeColumns();
    }

   private:
    std::vector<MaterializedSortBuffersBase::DictionaryStringSortPermutation>
    materializeDictionaryEncodedSortPermutations() const;
    MaterializedSortBuffersBase::DictionaryStringSortPermutation
    materializeDictionaryEncodedSortPermutation(
        const Analyzer::OrderEntry& order_entry) const;
    std::vector<std::vector<int64_t>> materializeCountDistinctColumns() const;
    ResultSet::ApproxQuantileBuffers materializeApproxQuantileColumns() const;
    ResultSet::ModeBuffers materializeModeColumns() const;
    std::vector<int64_t> materializeCountDistinctColumn(
        const Analyzer::OrderEntry& order_entry) const;
    ApproxQuantileBuffers::value_type materializeApproxQuantileColumn(
        const Analyzer::OrderEntry& order_entry) const;
    ModeBuffers::value_type materializeModeColumn(
        const Analyzer::OrderEntry& order_entry) const;
    struct ModeScatter;  // Functor for setting mode_buffers_.

    const BufferIteratorType buffer_itr_;
  };

  /**
   * @brief Initialize materialized sort buffers for dictionary encoded sort
   *  permutations, count distinct/approx_count distinct, mode, and
   * quantile/percentile calculations
   */
  void initMaterializedSortBuffers(
      const std::list<Analyzer::OrderEntry>& order_entries,
      bool single_threaded,
      std::optional<size_t> compact_permutation_size = std::nullopt,
      std::optional<MaterializedSortBuffersBase::TopNDictionarySortContext>
          top_n_dictionary_sort_context = std::nullopt);

  template <typename BUFFER_ITERATOR_TYPE>
  struct ResultSetComparator : public ResultSetComparatorBase {
    using BufferIteratorType = BUFFER_ITERATOR_TYPE;

    ResultSetComparator(const std::list<Analyzer::OrderEntry>& order_entries,
                        const ResultSet* result_set,
                        const PermutationView permutation,
                        const Executor* executor,
                        const bool single_threaded)
        : order_entries_(order_entries)
        , result_set_(result_set)
        , permutation_(permutation)
        , buffer_itr_(result_set)
        , executor_(executor)
        , single_threaded_(single_threaded)
        , dictionary_string_sorted_permutations_(
              result_set->materialized_sort_buffers_
                  ->getDictionaryEncodedSortPermutations())
        , count_distinct_materialized_buffers_(
              result_set->materialized_sort_buffers_->getCountDistinctBuffers())
        , approx_quantile_materialized_buffers_(
              result_set->materialized_sort_buffers_->getApproxQuantileBuffers())
        , mode_buffers_(result_set->materialized_sort_buffers_->getModeBuffers()) {}

    ResultSetComparator(ResultSetComparator const&) = delete;
    ResultSetComparator& operator=(ResultSetComparator const&) = delete;

    bool operator()(const PermutationIdx lhs, const PermutationIdx rhs) const;

    const std::list<Analyzer::OrderEntry>& order_entries_;
    const ResultSet* result_set_;
    const PermutationView permutation_;
    const BufferIteratorType buffer_itr_;
    const Executor* executor_;
    const bool single_threaded_;
    const std::vector<MaterializedSortBuffersBase::DictionaryStringSortPermutation>&
        dictionary_string_sorted_permutations_;
    const std::vector<std::vector<int64_t>>& count_distinct_materialized_buffers_;
    const ApproxQuantileBuffers& approx_quantile_materialized_buffers_;
    const ModeBuffers& mode_buffers_;
  };

  std::unique_ptr<ResultSetComparatorBase> createComparator(
      const std::list<Analyzer::OrderEntry>& order_entries,
      const PermutationView permutation,
      const Executor* executor,
      const bool single_threaded) {
    if (query_mem_desc_.didOutputColumnar()) {
      return std::make_unique<ResultSetComparator<ColumnWiseTargetAccessor>>(
          order_entries, this, permutation, executor, single_threaded);
    } else {
      return std::make_unique<ResultSetComparator<RowWiseTargetAccessor>>(
          order_entries, this, permutation, executor, single_threaded);
    }
  }

  static PermutationView topPermutation(PermutationView,
                                        const size_t top_n,
                                        const ResultSetComparatorBase*);

  template <typename BUFFER_ITERATOR_TYPE>
  static PermutationView topPermutationImpl(
      PermutationView,
      const size_t top_n,
      const ResultSetComparator<BUFFER_ITERATOR_TYPE>*);

  PermutationView initPermutationBuffer(PermutationView permutation,
                                        PermutationIdx const begin,
                                        PermutationIdx const end) const;
  PermutationView parallelInitPermutationBuffer(size_t entry_count);

  void parallelTop(const std::list<Analyzer::OrderEntry>& order_entries,
                   const size_t top_n,
                   const size_t entry_count,
                   const Executor* executor);

  bool sortWithMaterializedNumericKey(
      const std::list<Analyzer::OrderEntry>& order_entries,
      PermutationView permutation) const;

  void baselineSort(const std::list<Analyzer::OrderEntry>& order_entries,
                    const size_t top_n,
                    const ExecutorDeviceType device_type,
                    const Executor* executor);

  void doBaselineSort(const ExecutorDeviceType device_type,
                      const std::list<Analyzer::OrderEntry>& order_entries,
                      const size_t top_n,
                      const Executor* executor);

  bool canUseFastBaselineSort(const std::list<Analyzer::OrderEntry>& order_entries,
                              const size_t top_n);

  size_t sortRowCountForWatchdog() const;

  size_t rowCountImpl(const bool force_parallel) const;

  Data_Namespace::DataMgr* getDataManager() const;

  int getGpuCount() const;

  void serializeProjection(TSerializedRows& serialized_rows) const;
  void serializeVarlenAggColumn(int8_t* buf,
                                std::vector<std::string>& varlen_bufer) const;

  void serializeCountDistinctColumns(TSerializedRows&) const;

  void unserializeCountDistinctColumns(const TSerializedRows&);

  void fixupCountDistinctPointers();

  void create_active_buffer_set(CountDistinctSet& count_distinct_active_buffer_set) const;

  int64_t getDistinctBufferRefFromBufferRowwise(int8_t* rowwise_target_ptr,
                                                const TargetInfo& target_info) const;

  struct KeyInfo {
    const int8_t* key_ptr;
    const size_t key_width;
    KeyInfo(const int8_t* ptr, const size_t width) : key_ptr(ptr), key_width(width) {}
  };

  KeyInfo getKeyInfo(const ResultSetStorage* storage,
                     const int8_t* buff,
                     const size_t col_idx,
                     const size_t local_entry_idx) const;

  const std::vector<TargetInfo> targets_;
  const ExecutorDeviceType device_type_;
  const int device_id_;
  const int thread_idx_;
  mutable QueryMemoryDescriptor query_mem_desc_;
  mutable std::unique_ptr<ResultSetStorage> storage_;
  mutable size_t storage_buffer_size_bytes_{0};
  mutable AppendedStorage appended_storage_;
  mutable size_t crt_row_buff_idx_;
  mutable size_t fetched_so_far_;
  size_t drop_first_;
  size_t keep_first_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  Permutation permutation_;

  unsigned block_size_{0};
  unsigned grid_size_{0};
  QueryExecutionTimings timings_;

  std::list<std::shared_ptr<Chunk_NS::Chunk>> chunks_;
  std::vector<std::shared_ptr<std::list<ChunkIter>>> chunk_iters_;
  // TODO(miyu): refine by using one buffer and
  //   setting offset instead of ptr in group by buffer.
  std::vector<std::vector<int8_t>> literal_buffers_;
  std::vector<ColumnLazyFetchInfo> lazy_fetch_info_;
  mutable std::vector<std::vector<std::vector<const int8_t*>>> col_buffers_;
  DeferredLazyFetchChunkStorages deferred_lazy_fetch_chunks_;
  LazyFetchSourceMetadata lazy_fetch_source_metadata_;
  mutable std::mutex deferred_lazy_fetch_materialization_mutex_;
  mutable std::vector<uint8_t> deferred_lazy_fetch_columns_materialized_for_all_rows_;
  std::vector<std::vector<std::vector<ColumnBufferLayout>>> col_buffer_layouts_;
  std::vector<std::vector<std::vector<int64_t>>> frag_offsets_;
  std::vector<std::vector<int64_t>> consistent_frag_sizes_;
  std::vector<std::vector<DeviceColumnarBufferFragment>> device_columnar_fragments_;
  std::vector<DeviceRowwiseBufferFragment> device_rowwise_fragments_;
  mutable std::atomic<bool> device_columnar_cpu_storage_valid_{true};
  mutable bool device_columnar_fragments_cover_logical_rows_{false};
  mutable bool device_columnar_fragments_form_dense_cpu_rows_{false};
  mutable bool device_columnar_fragments_exclude_baseline_boundary_keys_{false};
  mutable bool device_columnar_fragments_cover_cpu_baseline_boundary_rows_{false};
  bool entry_filter_applied_{false};
  bool sparse_baseline_entry_filter_applied_before_copy_{false};
  mutable std::mutex device_columnar_cpu_storage_mutex_;

  const std::shared_ptr<const Analyzer::Estimator> estimator_;
  Data_Namespace::AbstractBuffer* device_estimator_buffer_{nullptr};
  mutable int8_t* host_estimator_buffer_{nullptr};
  Data_Namespace::DataMgr* data_mgr_;
  std::shared_ptr<CudaAllocator> cuda_allocator_{nullptr};
  CUstream cuda_stream_{nullptr};

  // only used by serialization
  using SerializedVarlenBufferStorage = std::vector<std::string>;

  std::vector<SerializedVarlenBufferStorage> serialized_varlen_buffer_;
  bool separate_varlen_storage_valid_;
  std::string explanation_;
  const bool just_explain_;
  bool for_validation_only_;
  bool baseline_hash_dense_for_reduction_{false};
  mutable std::atomic<int64_t> cached_row_count_;
  mutable std::mutex row_iteration_mutex_;

  // only used by geo
  mutable GeoReturnType geo_return_type_;

  // only used by data recycler
  bool cached_;  // indicator that this resultset is cached
  size_t
      query_exec_time_;  // an elapsed time to process the query for this resultset (ms)
  QueryPlanHash query_plan_;  // a hashed query plan DAG of this resultset
  std::unordered_set<size_t> input_table_keys_;  // input table signatures
  std::vector<TargetMetaInfo> target_meta_info_;
  std::unique_ptr<MaterializedSortBuffersBase> materialized_sort_buffers_;
  // if we recycle the resultset, we do not create work_unit of the query step
  // because we may skip its child query step(s)
  // so we try to keep whether this resultset is available to use speculative top n sort
  // when it is inserted to the recycler, and reuse this info when recycled
  std::optional<bool> can_use_speculative_top_n_sort;

  friend class ResultSetManager;
  friend class ResultSetRowIterator;
  friend class ColumnarResults;
};

ResultSetRowIterator::value_type ResultSetRowIterator::operator*() const {
  if (!global_entry_idx_valid_) {
    return {};
  }

  if (result_set_->just_explain_) {
    return {result_set_->explanation_};
  }

  return result_set_->getRowAt(
      global_entry_idx_, translate_strings_, decimal_to_double_, false);
}

inline ResultSetRowIterator& ResultSetRowIterator::operator++(void) {
  if (!result_set_->storage_ && !result_set_->just_explain_) {
    global_entry_idx_valid_ = false;
  } else if (result_set_->just_explain_) {
    global_entry_idx_valid_ = 0 == fetched_so_far_;
    fetched_so_far_ = 1;
  } else {
    result_set_->advanceCursorToNextEntry(*this);
  }
  return *this;
}

class ResultSetManager {
 public:
  ResultSet* reduce(std::vector<ResultSet*>&, const size_t executor_id);

  std::shared_ptr<ResultSet> getOwnResultSet();

  void rewriteVarlenAggregates(ResultSet*);

 private:
  std::shared_ptr<ResultSet> rs_;
};

class RowSortException : public std::runtime_error {
 public:
  RowSortException(const std::string& cause) : std::runtime_error(cause) {}
};

namespace result_set {

bool can_use_parallel_algorithms(const ResultSet& rows);

std::optional<size_t> first_dict_encoded_idx(std::vector<TargetInfo> const&);

bool use_parallel_algorithms(const ResultSet& rows);

}  // namespace result_set

#endif  // QUERYENGINE_RESULTSET_H
