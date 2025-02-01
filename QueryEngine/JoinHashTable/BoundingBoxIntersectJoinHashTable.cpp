/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "QueryEngine/JoinHashTable/BoundingBoxIntersectJoinHashTable.h"

#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/DataRecycler/DataRecycler.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/JoinHashTable/Builders/BaselineHashTableBuilder.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"
#include "QueryEngine/JoinHashTable/RangeJoinHashTable.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinKeyHandlers.h"
#include "QueryEngine/JoinHashTable/Runtime/JoinHashTableGpuUtils.h"
#include "QueryEngine/enums.h"

std::unique_ptr<HashtableRecycler> BoundingBoxIntersectJoinHashTable::hash_table_cache_ =
    std::make_unique<HashtableRecycler>(CacheItemType::BBOX_INTERSECT_HT,
                                        DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
std::unique_ptr<BoundingBoxIntersectTuningParamRecycler>
    BoundingBoxIntersectJoinHashTable::auto_tuner_cache_ =
        std::make_unique<BoundingBoxIntersectTuningParamRecycler>();

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<BoundingBoxIntersectJoinHashTable>
BoundingBoxIntersectJoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> condition,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const std::set<int>& device_ids,
    ColumnCacheMap& column_cache,
    Executor* executor,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const RegisteredQueryHint& query_hints,
    const TableIdToNodeMap& table_id_to_node_map) {
  decltype(std::chrono::steady_clock::now()) ts1, ts2;
  auto copied_query_hints = query_hints;
  if (query_hints.force_one_to_many_hash_join) {
    LOG(INFO) << "Ignoring query hint \'force_one_to_many_hash_join\' for bounding box "
                 "intersection";
    copied_query_hints.force_one_to_many_hash_join = false;
  }
  if (query_hints.force_baseline_hash_join) {
    LOG(INFO) << "Ignoring query hint \'force_baseline_hash_join\' for bounding box "
                 "intersection";
    copied_query_hints.force_baseline_hash_join = false;
  }
  std::vector<InnerOuter> inner_outer_pairs;
  if (const auto range_expr =
          dynamic_cast<const Analyzer::RangeOper*>(condition->get_right_operand())) {
    return RangeJoinHashTable::getInstance(condition,
                                           range_expr,
                                           query_infos,
                                           memory_level,
                                           join_type,
                                           device_ids,
                                           column_cache,
                                           executor,
                                           hashtable_build_dag_map,
                                           copied_query_hints,
                                           table_id_to_node_map);
  } else {
    inner_outer_pairs =
        HashJoin::normalizeColumnPairs(condition.get(), executor->getTemporaryTables())
            .first;
  }
  CHECK(!inner_outer_pairs.empty());

  const auto getHashTableType =
      [](const std::shared_ptr<Analyzer::BinOper> condition,
         const std::vector<InnerOuter>& inner_outer_pairs) -> HashType {
    HashType layout = HashType::OneToMany;
    if (condition->is_bbox_intersect_oper()) {
      CHECK_EQ(inner_outer_pairs.size(), size_t(1));
      if (inner_outer_pairs[0].first->get_type_info().is_array() &&
          inner_outer_pairs[0].second->get_type_info().is_array() &&
          // Bounds vs constructed points, former should yield ManyToMany
          inner_outer_pairs[0].second->get_type_info().get_size() == 32) {
        layout = HashType::ManyToMany;
      }
    }
    return layout;
  };

  const auto layout = getHashTableType(condition, inner_outer_pairs);

  if (VLOGGING(1)) {
    VLOG(1) << "Building geo hash table " << getHashTypeString(layout)
            << " for qual: " << condition->toString();
    ts1 = std::chrono::steady_clock::now();
  }

  const auto qi_0 = query_infos[0].info.getNumTuplesUpperBound();
  const auto qi_1 = query_infos[1].info.getNumTuplesUpperBound();

  VLOG(1) << "table_key = " << query_infos[0].table_key << " has " << qi_0 << " tuples.";
  VLOG(1) << "table_key = " << query_infos[1].table_key << " has " << qi_1 << " tuples.";

  const auto& query_info =
      get_inner_query_info(HashJoin::getInnerTableId(inner_outer_pairs), query_infos)
          .info;
  const auto total_entries = 2 * query_info.getNumTuplesUpperBound();
  if (total_entries > HashJoin::MAX_NUM_HASH_ENTRIES) {
    throw TooManyHashEntries();
  }

  auto join_hash_table =
      std::make_shared<BoundingBoxIntersectJoinHashTable>(condition,
                                                          join_type,
                                                          query_infos,
                                                          memory_level,
                                                          column_cache,
                                                          executor,
                                                          inner_outer_pairs,
                                                          device_ids,
                                                          copied_query_hints,
                                                          hashtable_build_dag_map,
                                                          table_id_to_node_map);
  try {
    join_hash_table->reify(layout);
  } catch (const HashJoinFail& e) {
    throw HashJoinFail(std::string("Could not build a 1-to-1 correspondence for columns "
                                   "involved in bounding box intersection | ") +
                       e.what());
  } catch (const ColumnarConversionNotSupported& e) {
    throw HashJoinFail(
        std::string("Could not build hash tables for bounding box intersection | "
                    "Inner table too big. Attempt manual table reordering "
                    "or create a single fragment inner table. | ") +
        e.what());
  } catch (const JoinHashTableTooBig& e) {
    throw e;
  } catch (const std::exception& e) {
    throw HashJoinFail(
        std::string("Failed to build hash tables for bounding box intersection | ") +
        e.what());
  }
  if (VLOGGING(1)) {
    ts2 = std::chrono::steady_clock::now();
    VLOG(1) << "Built geo hash table " << getHashTypeString(layout) << " in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count()
            << " ms";
  }
  return join_hash_table;
}

namespace {

std::vector<double> correct_uninitialized_bucket_sizes_to_thresholds(
    const std::vector<double>& bucket_sizes,
    const std::vector<double>& bucket_thresholds,
    const double initial_value) {
  std::vector<double> corrected_bucket_sizes(bucket_sizes);
  for (size_t i = 0; i != bucket_sizes.size(); ++i) {
    if (bucket_sizes[i] == initial_value) {
      corrected_bucket_sizes[i] = bucket_thresholds[i];
    }
  }
  return corrected_bucket_sizes;
}

std::vector<double> compute_bucket_sizes(
    const std::vector<double>& bucket_thresholds,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const JoinColumn& join_column,
    const JoinColumnTypeInfo& join_column_type,
    const std::vector<InnerOuter>& inner_outer_pairs,
    const Executor* executor) {
  // No coalesced keys for bounding box intersection yet
  CHECK_EQ(inner_outer_pairs.size(), 1u);

  const auto col = inner_outer_pairs[0].first;
  CHECK(col);
  const auto col_ti = col->get_type_info();
  CHECK(col_ti.is_array());

  // TODO: Compute the number of dimensions for keys used to perform bounding box
  // intersection
  const size_t num_dims{2};
  const double initial_bin_value{0.0};
  std::vector<double> bucket_sizes(num_dims, initial_bin_value);
  CHECK_EQ(bucket_thresholds.size(), num_dims);

  VLOG(1) << "Computing x and y bucket sizes for bounding box intersection with maximum "
             "bucket size "
          << std::to_string(bucket_thresholds[0]) << ", "
          << std::to_string(bucket_thresholds[1]);

  if (effective_memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
    const int thread_count = cpu_threads();
    compute_bucket_sizes_on_cpu(
        bucket_sizes, join_column, join_column_type, bucket_thresholds, thread_count);
  }
#ifdef HAVE_CUDA
  else {
    // Note that we compute the bucket sizes using only a single GPU
    auto const& device_ids = executor->getAvailableDevicesToProcessQuery();
    CHECK_GT(device_ids.size(), size_t(0));
    const int device_id = *device_ids.begin();
    auto device_allocator = executor->getCudaAllocator(device_id);
    CHECK(device_allocator);
    auto cuda_stream = executor->getCudaStream(device_id);
    auto device_bucket_sizes_gpu = transfer_vector_of_flat_objects_to_gpu(
        bucket_sizes, *device_allocator, "Boundingbox join hashtable bucket sizes");
    auto join_column_gpu = transfer_flat_object_to_gpu(
        join_column, *device_allocator, "Boundingbox hash join input column(s)");
    auto join_column_type_gpu =
        transfer_flat_object_to_gpu(join_column_type,
                                    *device_allocator,
                                    "Boundingbox hash join input column type(s)");
    auto device_bucket_thresholds_gpu = transfer_vector_of_flat_objects_to_gpu(
        bucket_thresholds,
        *device_allocator,
        "Boundingbox join hashtable bucket thresholds");
    compute_bucket_sizes_on_device(device_bucket_sizes_gpu,
                                   join_column_gpu,
                                   join_column_type_gpu,
                                   device_bucket_thresholds_gpu,
                                   cuda_stream);
    device_allocator->copyFromDevice(reinterpret_cast<int8_t*>(bucket_sizes.data()),
                                     reinterpret_cast<int8_t*>(device_bucket_sizes_gpu),
                                     bucket_sizes.size() * sizeof(double),
                                     "Boundingbox join hashtable bucket sizes");
  }
#endif
  const auto corrected_bucket_sizes = correct_uninitialized_bucket_sizes_to_thresholds(
      bucket_sizes, bucket_thresholds, initial_bin_value);

  VLOG(1) << "Computed x and y bucket sizes for bounding box intersection: ("
          << corrected_bucket_sizes[0] << ", " << corrected_bucket_sizes[1] << ")";

  return corrected_bucket_sizes;
}

struct HashTableProps {
  HashTableProps(const size_t entry_count,
                 const size_t emitted_keys_count,
                 const size_t hash_table_size,
                 const std::vector<double>& bucket_sizes)
      : entry_count(entry_count)
      , emitted_keys_count(emitted_keys_count)
      , keys_per_bin(entry_count == 0 ? std::numeric_limits<double>::max()
                                      : emitted_keys_count / (entry_count / 2.0))
      , hash_table_size(hash_table_size)
      , bucket_sizes(bucket_sizes) {}

  static HashTableProps invalid() { return HashTableProps(0, 0, 0, {}); }

  size_t entry_count;
  size_t emitted_keys_count;
  double keys_per_bin;
  size_t hash_table_size;
  std::vector<double> bucket_sizes;
};

std::ostream& operator<<(std::ostream& os, const HashTableProps& props) {
  os << " entry_count: " << props.entry_count << ", emitted_keys "
     << props.emitted_keys_count << ", hash table size " << props.hash_table_size
     << ", keys per bin " << props.keys_per_bin;
  return os;
}

struct TuningState {
  TuningState(const size_t bbox_intersect_max_table_size_bytes,
              const double bbox_intersect_target_entries_per_bin)
      : crt_props(HashTableProps::invalid())
      , prev_props(HashTableProps::invalid())
      , chosen_bbox_intersect_threshold(-1)
      , crt_step(0)
      , crt_reverse_search_iteration(0)
      , bbox_intersect_max_table_size_bytes(bbox_intersect_max_table_size_bytes)
      , bbox_intersect_target_entries_per_bin(bbox_intersect_target_entries_per_bin) {}

  // current and previous props, allows for easy backtracking
  HashTableProps crt_props;
  HashTableProps prev_props;

  // value we are tuning for
  double chosen_bbox_intersect_threshold;
  enum class TuningDirection { SMALLER, LARGER };
  TuningDirection tuning_direction{TuningDirection::SMALLER};

  // various constants / state
  size_t crt_step;                      // 1 indexed
  size_t crt_reverse_search_iteration;  // 1 indexed
  size_t bbox_intersect_max_table_size_bytes;
  double bbox_intersect_target_entries_per_bin;
  const size_t max_reverse_search_iterations{8};

  /**
   * Returns true to continue tuning, false to end the loop with the above threshold
   */
  bool operator()(const HashTableProps& new_props,
                  const bool new_bbox_intersect_threshold) {
    prev_props = crt_props;
    crt_props = new_props;
    crt_step++;

    if (hashTableTooBig() || keysPerBinIncreasing()) {
      if (hashTableTooBig()) {
        VLOG(1) << "Reached hash table size limit: "
                << bbox_intersect_max_table_size_bytes << " with "
                << crt_props.hash_table_size << " byte hash table, "
                << crt_props.keys_per_bin << " keys per bin.";
      } else if (keysPerBinIncreasing()) {
        VLOG(1) << "Keys per bin increasing from " << prev_props.keys_per_bin << " to "
                << crt_props.keys_per_bin;
        CHECK(previousIterationValid());
      }
      if (previousIterationValid()) {
        VLOG(1) << "Using previous threshold value " << chosen_bbox_intersect_threshold;
        crt_props = prev_props;
        return false;
      } else {
        CHECK(hashTableTooBig());
        crt_reverse_search_iteration++;
        chosen_bbox_intersect_threshold = new_bbox_intersect_threshold;

        if (crt_reverse_search_iteration == max_reverse_search_iterations) {
          VLOG(1) << "Hit maximum number (" << max_reverse_search_iterations
                  << ") of reverse tuning iterations. Aborting tuning";
          // use the crt props, but don't bother trying to tune any farther
          return false;
        }

        if (crt_reverse_search_iteration > 1 &&
            crt_props.hash_table_size == prev_props.hash_table_size) {
          // hash table size is not changing, bail
          VLOG(1) << "Hash table size not decreasing (" << crt_props.hash_table_size
                  << " bytes) and still above maximum allowed size ("
                  << bbox_intersect_max_table_size_bytes << " bytes). Aborting tuning";
          return false;
        }

        // if the hash table is too big on the very first step, change direction towards
        // larger bins to see if a slightly smaller hash table will fit
        if (crt_step == 1 && crt_reverse_search_iteration == 1) {
          VLOG(1)
              << "First iteration of tuning led to hash table size over "
                 "limit. Reversing search to try larger bin sizes (previous threshold: "
              << chosen_bbox_intersect_threshold << ")";
          // Need to change direction of tuning to tune "up" towards larger bins
          tuning_direction = TuningDirection::LARGER;
        }
        return true;
      }
      UNREACHABLE();
    }

    chosen_bbox_intersect_threshold = new_bbox_intersect_threshold;

    if (keysPerBinUnderThreshold()) {
      VLOG(1) << "Hash table reached size " << crt_props.hash_table_size
              << " with keys per bin " << crt_props.keys_per_bin << " under threshold "
              << bbox_intersect_target_entries_per_bin
              << ". Terminating bucket size loop.";
      return false;
    }

    if (crt_reverse_search_iteration > 0) {
      // We always take the first tuning iteration that succeeds when reversing
      // direction, as if we're here we haven't had a successful iteration and we're
      // "backtracking" our search by making bin sizes larger
      VLOG(1) << "On reverse (larger tuning direction) search found workable "
              << " hash table size of " << crt_props.hash_table_size
              << " with keys per bin " << crt_props.keys_per_bin
              << ". Terminating bucket size loop.";
      return false;
    }

    return true;
  }

  bool hashTableTooBig() const {
    return crt_props.hash_table_size > bbox_intersect_max_table_size_bytes;
  }

  bool keysPerBinIncreasing() const {
    return crt_props.keys_per_bin > prev_props.keys_per_bin;
  }

  bool previousIterationValid() const {
    return tuning_direction == TuningDirection::SMALLER && crt_step > 1;
  }

  bool keysPerBinUnderThreshold() const {
    return crt_props.keys_per_bin < bbox_intersect_target_entries_per_bin;
  }
};

class BucketSizeTuner {
 public:
  BucketSizeTuner(const double bucket_threshold,
                  const double step,
                  const double min_threshold,
                  const Data_Namespace::MemoryLevel effective_memory_level,
                  const std::unordered_map<int, ColumnsForDevice>& columns_per_device,
                  const std::vector<InnerOuter>& inner_outer_pairs,
                  const size_t table_tuple_count,
                  const Executor* executor)
      : num_dims_(2)  // Todo: allow varying number of dims
      , bucket_thresholds_(/*count=*/num_dims_, /*value=*/bucket_threshold)
      , step_(step)
      , min_threshold_(min_threshold)
      , effective_memory_level_(effective_memory_level)
      , columns_per_device_(columns_per_device)
      , inner_outer_pairs_(inner_outer_pairs)
      , table_tuple_count_(table_tuple_count)
      , executor_(executor) {
    CHECK(!columns_per_device_.empty());
  }

  bool tuneOneStep() { return tuneOneStep(TuningState::TuningDirection::SMALLER, step_); }

  bool tuneOneStep(const TuningState::TuningDirection tuning_direction) {
    return tuneOneStep(tuning_direction, step_);
  }

  bool tuneOneStep(const TuningState::TuningDirection tuning_direction,
                   const double step_overide) {
    if (table_tuple_count_ == 0) {
      return false;
    }
    if (tuning_direction == TuningState::TuningDirection::SMALLER) {
      return tuneSmallerOneStep(step_overide);
    }
    return tuneLargerOneStep(step_overide);
  }

  auto getMinBucketSize() const {
    return *std::min_element(bucket_thresholds_.begin(), bucket_thresholds_.end());
  }

  /**
   * Method to retrieve inverted bucket sizes, which are what are used elsewhere in the
   * hash join framework for bounding box intersection
   * @return the inverted bucket sizes, i.e. a set of that will place a raw value in a
   * bucket when multiplied by the raw value
   */
  std::vector<double> getInverseBucketSizes() {
    if (num_steps_ == 0) {
      CHECK_EQ(current_bucket_sizes_.size(), static_cast<size_t>(0));
      current_bucket_sizes_ = computeBucketSizes();
    }
    CHECK_EQ(current_bucket_sizes_.size(), num_dims_);
    std::vector<double> inverse_bucket_sizes;
    for (const auto s : current_bucket_sizes_) {
      inverse_bucket_sizes.emplace_back(1.0 / s);
    }
    return inverse_bucket_sizes;
  }

 private:
  bool bucketThresholdsBelowMinThreshold() const {
    for (const auto& t : bucket_thresholds_) {
      if (t < min_threshold_) {
        return true;
      }
    }
    return false;
  }

  std::vector<double> computeBucketSizes() const {
    if (table_tuple_count_ == 0) {
      return std::vector<double>(/*count=*/num_dims_, /*val=*/0);
    }
    return compute_bucket_sizes(bucket_thresholds_,
                                effective_memory_level_,
                                columns_per_device_.begin()->second.join_columns[0],
                                columns_per_device_.begin()->second.join_column_types[0],
                                inner_outer_pairs_,
                                executor_);
  }

  bool tuneSmallerOneStep(const double step_overide) {
    if (!current_bucket_sizes_.empty()) {
      CHECK_EQ(current_bucket_sizes_.size(), bucket_thresholds_.size());
      bucket_thresholds_ = current_bucket_sizes_;
      for (auto& t : bucket_thresholds_) {
        t /= step_overide;
      }
    }
    if (bucketThresholdsBelowMinThreshold()) {
      VLOG(1) << "Aborting tuning for bounding box intersection as at least one bucket "
                 "size is below min threshold";
      return false;
    }
    const auto next_bucket_sizes = computeBucketSizes();
    if (next_bucket_sizes == current_bucket_sizes_) {
      VLOG(1) << "Aborting tuning for bounding box intersection as bucket size is no "
                 "longer changing.";
      return false;
    }

    current_bucket_sizes_ = next_bucket_sizes;
    num_steps_++;
    return true;
  }

  bool tuneLargerOneStep(const double step_overide) {
    if (!current_bucket_sizes_.empty()) {
      CHECK_EQ(current_bucket_sizes_.size(), bucket_thresholds_.size());
      bucket_thresholds_ = current_bucket_sizes_;
    }
    // If current_bucket_sizes was empty, we will start from our initial threshold
    for (auto& t : bucket_thresholds_) {
      t *= step_overide;
    }
    // When tuning up, do not dynamically compute bucket_sizes, as compute_bucket_sizes as
    // written will pick the largest bin size below the threshold, meaning our bucket_size
    // will never increase beyond the size of the largest polygon. This could mean that we
    // can never make the bucket sizes large enough to get our hash table below the
    // maximum size Possible todo: enable templated version of compute_bucket_sizes that
    // allows for optionally finding smallest extent above threshold, to mirror default
    // behavior finding largest extent below threshold, and use former variant here
    current_bucket_sizes_ = bucket_thresholds_;
    num_steps_++;
    return true;
  }

  size_t num_dims_;
  std::vector<double> bucket_thresholds_;
  size_t num_steps_{0};
  const double step_;
  const double min_threshold_;
  const Data_Namespace::MemoryLevel effective_memory_level_;
  const std::unordered_map<int, ColumnsForDevice>& columns_per_device_;
  const std::vector<InnerOuter>& inner_outer_pairs_;
  const size_t table_tuple_count_;
  const Executor* executor_;

  std::vector<double> current_bucket_sizes_;

  friend std::ostream& operator<<(std::ostream& os, const BucketSizeTuner& tuner);
};

std::ostream& operator<<(std::ostream& os, const BucketSizeTuner& tuner) {
  os << "Step Num: " << tuner.num_steps_ << ", Threshold: " << std::fixed << "("
     << tuner.bucket_thresholds_[0] << ", " << tuner.bucket_thresholds_[1] << ")"
     << ", Step Size: " << std::fixed << tuner.step_ << ", Min: " << std::fixed
     << tuner.min_threshold_;
  return os;
}

}  // namespace

void BoundingBoxIntersectJoinHashTable::reifyWithLayout(const HashType layout) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(layoutRequiresAdditionalBuffers(layout));
  const auto& query_info =
      get_inner_query_info(HashJoin::getInnerTableId(inner_outer_pairs_), query_infos_)
          .info;
  auto [db_id, table_id] = HashJoin::getInnerTableId(inner_outer_pairs_);
  VLOG(1) << "Reify with layout " << getHashTypeString(layout) << "for db_id: " << db_id
          << ", table_id: " << table_id;
  if (query_info.fragments.empty()) {
    return;
  }

  auto bbox_intersect_max_table_size_bytes = g_bbox_intersect_max_table_size_bytes;
  std::optional<double> bbox_intersect_threshold_override;
  double bbox_intersect_target_entries_per_bin = g_bbox_intersect_target_entries_per_bin;
  auto skip_hashtable_caching = false;
  if (query_hints_.isHintRegistered(QueryHint::kBBoxIntersectBucketThreshold)) {
    VLOG(1) << "Setting bounding box intersection bucket threshold "
               "\'bbox_intersect_bucket_threshold\' via "
               "query hint: "
            << query_hints_.bbox_intersect_bucket_threshold;
    bbox_intersect_threshold_override = query_hints_.bbox_intersect_bucket_threshold;
  }
  if (query_hints_.isHintRegistered(QueryHint::kBBoxIntersectMaxSize)) {
    std::ostringstream oss;
    oss << "User requests to change a threshold \'bbox_intersect_max_table_size_bytes\' "
           "via "
           "query hint";
    if (!bbox_intersect_threshold_override.has_value()) {
      oss << ": " << bbox_intersect_max_table_size_bytes << " -> "
          << query_hints_.bbox_intersect_max_size;
      bbox_intersect_max_table_size_bytes = query_hints_.bbox_intersect_max_size;
    } else {
      oss << ", but is skipped since the query hint also changes the threshold "
             "\'bbox_intersect_bucket_threshold\'";
    }
    VLOG(1) << oss.str();
  }
  if (query_hints_.isHintRegistered(QueryHint::kBBoxIntersectNoCache)) {
    VLOG(1) << "User requests to skip caching join hashtable for bounding box "
               "intersection and its tuned "
               "parameters for this query";
    skip_hashtable_caching = true;
  }
  if (query_hints_.isHintRegistered(QueryHint::kBBoxIntersectKeysPerBin)) {
    VLOG(1) << "User requests to change a threshold \'bbox_intersect_keys_per_bin\' via "
               "query "
               "hint: "
            << bbox_intersect_target_entries_per_bin << " -> "
            << query_hints_.bbox_intersect_keys_per_bin;
    bbox_intersect_target_entries_per_bin = query_hints_.bbox_intersect_keys_per_bin;
  }

  auto data_mgr = executor_->getDataMgr();
  // we prioritize CPU when building a join hashtable for bounding box intersection, but
  // if we have GPU and user-given hint is given we selectively allow GPU to build it but
  // even if we have GPU but user foces to set CPU as execution device type we should not
  // allow to use GPU for building it
  auto allow_gpu_hashtable_build =
      query_hints_.isHintRegistered(QueryHint::kBBoxIntersectAllowGpuBuild) &&
      query_hints_.bbox_intersect_allow_gpu_build;
  if (allow_gpu_hashtable_build) {
    if (data_mgr->gpusPresent() &&
        memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      VLOG(1) << "A user forces to build GPU hash table for bounding box intersection";
    } else {
      allow_gpu_hashtable_build = false;
      VLOG(1) << "A user forces to build GPU hash table for bounding box intersection "
                 "but we skip it since either GPU is not presented or CPU execution mode "
                 "is set";
    }
  }

  std::unordered_map<int, ColumnsForDevice> columns_per_device;
  std::unordered_map<int, std::vector<Fragmenter_Namespace::FragmentInfo>>
      fragments_per_device;
  const auto shard_count = shardCount();
  size_t total_num_tuples = 0;
  auto const effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);
  for (auto device_id : device_ids_) {
    fragments_per_device.emplace(
        device_id,
        shard_count
            ? only_shards_for_device(
                  query_info.fragments, effective_memory_level, device_id, device_ids_)
            : query_info.fragments);
    const size_t crt_num_tuples =
        std::accumulate(fragments_per_device[device_id].begin(),
                        fragments_per_device[device_id].end(),
                        size_t(0),
                        [](const auto& sum, const auto& fragment) {
                          return sum + fragment.getNumTuples();
                        });
    total_num_tuples += crt_num_tuples;
    DeviceAllocator* device_allocator{nullptr};
    if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      device_allocator = executor_->getCudaAllocator(device_id);
      CHECK(device_allocator);
    }
    const auto columns_for_device = fetchColumnsForDevice(
        fragments_per_device[device_id], device_id, device_allocator);
    columns_per_device.emplace(device_id, columns_for_device);
  }

  // try to extract cache key for hash table and its relevant info
  auto hashtable_access_path_info =
      HashtableRecycler::getHashtableAccessPathInfo(inner_outer_pairs_,
                                                    {},
                                                    condition_->get_optype(),
                                                    join_type_,
                                                    hashtable_build_dag_map_,
                                                    device_ids_,
                                                    shard_count,
                                                    fragments_per_device,
                                                    executor_);
  hashtable_cache_key_ = hashtable_access_path_info.hashed_query_plan_dag;
  hashtable_cache_meta_info_ = hashtable_access_path_info.meta_info;
  table_keys_ = hashtable_access_path_info.table_keys;

  auto get_inner_table_key = [this]() {
    auto col_var = inner_outer_pairs_.front().first;
    return col_var->getTableKey();
  };

  if (table_keys_.empty()) {
    const auto& table_key = get_inner_table_key();
    table_keys_ = DataRecyclerUtil::getAlternativeTableKeys(
        composite_key_info_.cache_key_chunks, table_key);
  }
  CHECK(!table_keys_.empty());

  if (bbox_intersect_threshold_override) {
    // compute bucket sizes based on the user provided threshold
    BucketSizeTuner tuner(/*initial_threshold=*/*bbox_intersect_threshold_override,
                          /*step=*/1.0,
                          /*min_threshold=*/0.0,
                          effective_memory_level,
                          columns_per_device,
                          inner_outer_pairs_,
                          total_num_tuples,
                          executor_);
    const auto inverse_bucket_sizes = tuner.getInverseBucketSizes();

    auto [entry_count, emitted_keys_count] =
        computeHashTableCounts(shard_count,
                               inverse_bucket_sizes,
                               columns_per_device,
                               bbox_intersect_max_table_size_bytes,
                               *bbox_intersect_threshold_override);
    setInverseBucketSizeInfo(inverse_bucket_sizes, columns_per_device);
    // reifyImpl will check the hash table cache for an appropriate hash table w/ those
    // bucket sizes (or within tolerances) if a hash table exists use it, otherwise build
    // one
    generateCacheKey(bbox_intersect_max_table_size_bytes,
                     *bbox_intersect_threshold_override,
                     inverse_bucket_sizes,
                     fragments_per_device,
                     device_ids_);
    reifyImpl(columns_per_device,
              query_info,
              layout,
              shard_count,
              entry_count,
              emitted_keys_count,
              skip_hashtable_caching,
              bbox_intersect_max_table_size_bytes,
              *bbox_intersect_threshold_override);
  } else {
    double bbox_intersect_bucket_threshold = std::numeric_limits<double>::max();
    generateCacheKey(bbox_intersect_max_table_size_bytes,
                     bbox_intersect_bucket_threshold,
                     {},
                     fragments_per_device,
                     device_ids_);
    std::vector<size_t> per_device_chunk_key;
    if (HashtableRecycler::isInvalidHashTableCacheKey(hashtable_cache_key_) &&
        get_inner_table_key().table_id > 0) {
      for (auto device_id : device_ids_) {
        auto chunk_key_hash = boost::hash_value(composite_key_info_.cache_key_chunks);
        boost::hash_combine(
            chunk_key_hash,
            HashJoin::collectFragmentIds(fragments_per_device[device_id]));
        per_device_chunk_key.push_back(chunk_key_hash);
        AlternativeCacheKeyForBoundingBoxIntersection cache_key{
            inner_outer_pairs_,
            columns_per_device.begin()->second.join_columns.front().num_elems,
            chunk_key_hash,
            condition_->get_optype(),
            bbox_intersect_max_table_size_bytes,
            bbox_intersect_bucket_threshold,
            {}};
        hashtable_cache_key_[device_id] = getAlternativeCacheKey(cache_key);
        hash_table_cache_->addQueryPlanDagForTableKeys(hashtable_cache_key_[device_id],
                                                       table_keys_);
      }
    }

    auto cached_bucket_threshold = auto_tuner_cache_->getItemFromCache(
        hashtable_cache_key_.begin()->second,
        CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM,
        DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
    if (cached_bucket_threshold) {
      bbox_intersect_bucket_threshold = cached_bucket_threshold->bucket_threshold;
      auto inverse_bucket_sizes = cached_bucket_threshold->bucket_sizes;
      setBoundingBoxIntersectionMetaInfo(bbox_intersect_max_table_size_bytes,
                                         bbox_intersect_bucket_threshold,
                                         inverse_bucket_sizes);
      generateCacheKey(bbox_intersect_max_table_size_bytes,
                       bbox_intersect_bucket_threshold,
                       inverse_bucket_sizes,
                       fragments_per_device,
                       device_ids_);

      if (auto hash_table =
              hash_table_cache_->getItemFromCache(hashtable_cache_key_.begin()->second,
                                                  CacheItemType::BBOX_INTERSECT_HT,
                                                  DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                                                  std::nullopt)) {
        // if we already have a built hash table, we can skip the scans required for
        // computing bucket size and tuple count
        // reset as the hash table sizes can vary a bit
        setInverseBucketSizeInfo(inverse_bucket_sizes, columns_per_device);
        CHECK(hash_table);

        VLOG(1) << "Using cached hash table bucket size";

        reifyImpl(columns_per_device,
                  query_info,
                  layout,
                  shard_count,
                  hash_table->getEntryCount(),
                  hash_table->getEmittedKeysCount(),
                  skip_hashtable_caching,
                  bbox_intersect_max_table_size_bytes,
                  bbox_intersect_bucket_threshold);
      } else {
        VLOG(1) << "Computing bucket size for cached bucket threshold";
        // compute bucket size using our cached tuner value
        BucketSizeTuner tuner(/*initial_threshold=*/bbox_intersect_bucket_threshold,
                              /*step=*/1.0,
                              /*min_threshold=*/0.0,
                              effective_memory_level,
                              columns_per_device,
                              inner_outer_pairs_,
                              total_num_tuples,
                              executor_);

        const auto inverse_bucket_sizes = tuner.getInverseBucketSizes();

        auto [entry_count, emitted_keys_count] =
            computeHashTableCounts(shard_count,
                                   inverse_bucket_sizes,
                                   columns_per_device,
                                   bbox_intersect_max_table_size_bytes,
                                   bbox_intersect_bucket_threshold);
        setInverseBucketSizeInfo(inverse_bucket_sizes, columns_per_device);

        generateCacheKey(bbox_intersect_max_table_size_bytes,
                         bbox_intersect_bucket_threshold,
                         inverse_bucket_sizes,
                         fragments_per_device,
                         device_ids_);

        reifyImpl(columns_per_device,
                  query_info,
                  layout,
                  shard_count,
                  entry_count,
                  emitted_keys_count,
                  skip_hashtable_caching,
                  bbox_intersect_max_table_size_bytes,
                  bbox_intersect_bucket_threshold);
      }
    } else {
      // compute bucket size using the auto tuner
      BucketSizeTuner tuner(
          /*initial_threshold=*/bbox_intersect_bucket_threshold,
          /*step=*/2.0,
          /*min_threshold=*/1e-7,
          effective_memory_level,
          columns_per_device,
          inner_outer_pairs_,
          total_num_tuples,
          executor_);

      VLOG(1) << "Running auto tune logic for bounding box intersection with parameters: "
              << tuner;

      // manages the tuning state machine
      TuningState tuning_state(bbox_intersect_max_table_size_bytes,
                               bbox_intersect_target_entries_per_bin);
      while (tuner.tuneOneStep(tuning_state.tuning_direction)) {
        const auto inverse_bucket_sizes = tuner.getInverseBucketSizes();

        const auto [crt_entry_count, crt_emitted_keys_count] =
            computeHashTableCounts(shard_count,
                                   inverse_bucket_sizes,
                                   columns_per_device,
                                   tuning_state.bbox_intersect_max_table_size_bytes,
                                   tuning_state.chosen_bbox_intersect_threshold);
        const size_t hash_table_size = calculateHashTableSize(
            inverse_bucket_sizes.size(), crt_emitted_keys_count, crt_entry_count);
        HashTableProps crt_props(crt_entry_count,
                                 crt_emitted_keys_count,
                                 hash_table_size,
                                 inverse_bucket_sizes);
        VLOG(1) << "Tuner output: " << tuner << " with properties " << crt_props;

        const auto should_continue = tuning_state(crt_props, tuner.getMinBucketSize());
        setInverseBucketSizeInfo(tuning_state.crt_props.bucket_sizes, columns_per_device);
        if (!should_continue) {
          break;
        }
      }

      const auto& crt_props = tuning_state.crt_props;
      // sanity check that the hash table size has not changed. this is a fairly
      // inexpensive check to ensure the above algorithm is consistent
      const size_t hash_table_size =
          calculateHashTableSize(inverse_bucket_sizes_for_dimension_.size(),
                                 crt_props.emitted_keys_count,
                                 crt_props.entry_count);
      CHECK_EQ(crt_props.hash_table_size, hash_table_size);

      if (inverse_bucket_sizes_for_dimension_.empty() ||
          hash_table_size > bbox_intersect_max_table_size_bytes) {
        VLOG(1) << "Could not find suitable parameters to create hash "
                   "table for bounding box intersectionunder max allowed size ("
                << bbox_intersect_max_table_size_bytes << ") bytes.";
        throw TooBigHashTableForBoundingBoxIntersect(bbox_intersect_max_table_size_bytes);
      }

      VLOG(1) << "Final tuner output: " << tuner << " with properties " << crt_props;
      CHECK(!inverse_bucket_sizes_for_dimension_.empty());
      VLOG(1) << "Final bucket sizes: ";
      for (size_t dim = 0; dim < inverse_bucket_sizes_for_dimension_.size(); dim++) {
        VLOG(1) << "dim[" << dim
                << "]: " << 1.0 / inverse_bucket_sizes_for_dimension_[dim];
      }
      CHECK_GE(tuning_state.chosen_bbox_intersect_threshold, double(0));
      generateCacheKey(tuning_state.bbox_intersect_max_table_size_bytes,
                       tuning_state.chosen_bbox_intersect_threshold,
                       {},
                       fragments_per_device,
                       device_ids_);
      const auto candidate_auto_tuner_cache_key = hashtable_cache_key_.begin()->second;
      if (skip_hashtable_caching) {
        VLOG(1) << "Skip to add tuned parameters to auto tuner";
      } else {
        AutoTunerMetaInfo meta_info{tuning_state.bbox_intersect_max_table_size_bytes,
                                    tuning_state.chosen_bbox_intersect_threshold,
                                    inverse_bucket_sizes_for_dimension_};
        auto_tuner_cache_->putItemToCache(candidate_auto_tuner_cache_key,
                                          meta_info,
                                          CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM,
                                          DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                                          0,
                                          0);
      }
      bbox_intersect_bucket_threshold = tuning_state.chosen_bbox_intersect_threshold;
      reifyImpl(columns_per_device,
                query_info,
                layout,
                shard_count,
                crt_props.entry_count,
                crt_props.emitted_keys_count,
                skip_hashtable_caching,
                bbox_intersect_max_table_size_bytes,
                bbox_intersect_bucket_threshold);
    }
  }
}

size_t BoundingBoxIntersectJoinHashTable::calculateHashTableSize(
    size_t number_of_dimensions,
    size_t emitted_keys_count,
    size_t entry_count) const {
  const auto key_component_width = getKeyComponentWidth();
  const auto key_component_count = number_of_dimensions;
  const auto entry_size = key_component_count * key_component_width;
  const auto keys_for_all_rows = emitted_keys_count;
  const size_t one_to_many_hash_entries = 2 * entry_count + keys_for_all_rows;
  const size_t hash_table_size =
      entry_size * entry_count + one_to_many_hash_entries * sizeof(int32_t);
  return hash_table_size;
}

ColumnsForDevice BoundingBoxIntersectJoinHashTable::fetchColumnsForDevice(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    DeviceAllocator* dev_buff_owner) {
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);

  std::vector<JoinColumn> join_columns;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<JoinColumnTypeInfo> join_column_types;
  std::vector<std::shared_ptr<void>> malloc_owner;
  for (const auto& inner_outer_pair : inner_outer_pairs_) {
    const auto inner_col = inner_outer_pair.first;
    const auto inner_cd = get_column_descriptor_maybe(inner_col->getColumnKey());
    if (inner_cd && inner_cd->isVirtualCol) {
      throw FailedToJoinOnVirtualColumn();
    }
    join_columns.emplace_back(fetchJoinColumn(inner_col,
                                              fragments,
                                              effective_memory_level,
                                              device_id,
                                              chunks_owner,
                                              dev_buff_owner,
                                              malloc_owner,
                                              executor_,
                                              &column_cache_));
    const auto& ti = inner_col->get_type_info();
    join_column_types.emplace_back(JoinColumnTypeInfo{static_cast<size_t>(ti.get_size()),
                                                      0,
                                                      0,
                                                      inline_int_null_value<int64_t>(),
                                                      false,
                                                      0,
                                                      get_join_column_type_kind(ti)});
    CHECK(ti.is_array())
        << "Bounding box intersection currently only supported for arrays.";
  }
  return {join_columns, join_column_types, chunks_owner, {}, malloc_owner};
}

std::pair<size_t, size_t> BoundingBoxIntersectJoinHashTable::computeHashTableCounts(
    const size_t shard_count,
    const std::vector<double>& inverse_bucket_sizes_for_dimension,
    std::unordered_map<int, ColumnsForDevice>& columns_per_device,
    const size_t chosen_max_hashtable_size,
    const double chosen_bucket_threshold) {
  CHECK(!inverse_bucket_sizes_for_dimension.empty());
  const auto [tuple_count, emitted_keys_count] =
      approximateTupleCount(inverse_bucket_sizes_for_dimension,
                            columns_per_device,
                            chosen_max_hashtable_size,
                            chosen_bucket_threshold);
  const auto entry_count = 2 * std::max(tuple_count, size_t(1));

  return std::make_pair(
      get_entries_per_device(entry_count, shard_count, device_ids_, memory_level_),
      emitted_keys_count);
}

std::pair<size_t, size_t> BoundingBoxIntersectJoinHashTable::approximateTupleCount(
    const std::vector<double>& inverse_bucket_sizes_for_dimension,
    std::unordered_map<int, ColumnsForDevice>& columns_per_device,
    const size_t chosen_max_hashtable_size,
    const double chosen_bucket_threshold) {
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);
  CountDistinctDescriptor count_distinct_desc{
      CountDistinctImplType::Bitmap,
      0,
      0,
      11,
      true,
      effective_memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL
          ? ExecutorDeviceType::GPU
          : ExecutorDeviceType::CPU,
      1};
  const auto padded_size_bytes = count_distinct_desc.bitmapPaddedSizeBytes();

  CHECK(!columns_per_device.empty() &&
        !columns_per_device.begin()->second.join_columns.empty());
  if (columns_per_device.begin()->second.join_columns.front().num_elems == 0) {
    return std::make_pair(0, 0);
  }

  // TODO: state management in here should be revisited, but this should be safe enough
  // for now
  // re-compute bucket counts per device based on global bucket size
  for (auto& kv : columns_per_device) {
    auto& columns_for_device = kv.second;
    columns_for_device.setBucketInfo(inverse_bucket_sizes_for_dimension,
                                     inner_outer_pairs_);
  }

  // Number of keys must match dimension of buckets
  CHECK_EQ(columns_per_device.begin()->second.join_columns.size(),
           columns_per_device.begin()->second.join_buckets.size());
  if (effective_memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
    // Note that this path assumes each device has the same hash table (for GPU hash
    // join w/ hash table built on CPU)
    const auto cached_count_info =
        getApproximateTupleCountFromCache(hashtable_cache_key_.begin()->second,
                                          CacheItemType::BBOX_INTERSECT_HT,
                                          DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
    if (cached_count_info) {
      VLOG(1) << "Using a cached tuple count: " << cached_count_info->first
              << ", emitted keys count: " << cached_count_info->second;
      return *cached_count_info;
    }
    int thread_count = cpu_threads();
    std::vector<uint8_t> hll_buffer_all_cpus(thread_count * padded_size_bytes);
    auto hll_result = &hll_buffer_all_cpus[0];

    std::vector<int32_t> num_keys_for_row;
    // TODO(adb): support multi-column bounding box intersection
    num_keys_for_row.resize(columns_per_device.begin()->second.join_columns[0].num_elems);

    approximate_distinct_tuples_bbox_intersect(
        hll_result,
        num_keys_for_row,
        count_distinct_desc.bitmap_sz_bits,
        padded_size_bytes,
        columns_per_device.begin()->second.join_columns,
        columns_per_device.begin()->second.join_column_types,
        columns_per_device.begin()->second.join_buckets,
        thread_count);
    for (int i = 1; i < thread_count; ++i) {
      hll_unify(hll_result,
                hll_result + i * padded_size_bytes,
                size_t(1) << count_distinct_desc.bitmap_sz_bits);
    }
    return std::make_pair(
        hll_size(hll_result, count_distinct_desc.bitmap_sz_bits),
        static_cast<size_t>(num_keys_for_row.size() > 0 ? num_keys_for_row.back() : 0));
  }
#ifdef HAVE_CUDA
  std::unordered_map<int, std::vector<uint8_t>> host_hll_buffers;
  std::unordered_map<int, size_t> emitted_keys_count_device_threads;
  for (auto device_id : device_ids_) {
    std::vector<uint8_t> host_hll_buffer;
    host_hll_buffer.resize(count_distinct_desc.bitmapPaddedSizeBytes());
    host_hll_buffers.emplace(device_id, std::move(host_hll_buffer));
    emitted_keys_count_device_threads.emplace(device_id, 0u);
  }
  std::vector<std::future<void>> approximate_distinct_device_threads;
  for (auto device_id : device_ids_) {
    approximate_distinct_device_threads.emplace_back(std::async(
        std::launch::async,
        [this,
         device_id,
         &columns_per_device,
         &count_distinct_desc,
         &host_hll_buffers,
         &emitted_keys_count_device_threads] {
          auto device_allocator = executor_->getCudaAllocator(device_id);
          CHECK(device_allocator);
          auto device_hll_buffer =
              device_allocator->alloc(count_distinct_desc.bitmapPaddedSizeBytes());
          device_allocator->zeroDeviceMem(device_hll_buffer,
                                          count_distinct_desc.bitmapPaddedSizeBytes());
          const auto& columns_for_device = columns_per_device[device_id];
          auto join_columns_gpu = transfer_vector_of_flat_objects_to_gpu(
              columns_for_device.join_columns,
              *device_allocator,
              "Boundingbox hash join input column(s)");

          CHECK_GT(columns_for_device.join_buckets.size(), 0u);
          const auto& inverse_bucket_sizes_for_dimension =
              columns_for_device.join_buckets[0].inverse_bucket_sizes_for_dimension;
          auto inverse_bucket_sizes_gpu = device_allocator->alloc(
              inverse_bucket_sizes_for_dimension.size() * sizeof(double));
          device_allocator->copyToDevice(
              inverse_bucket_sizes_gpu,
              inverse_bucket_sizes_for_dimension.data(),
              inverse_bucket_sizes_for_dimension.size() * sizeof(double),
              "Boundingbox join hashtable inverse bucket sizes");
          const size_t row_counts_buffer_sz =
              columns_per_device.begin()->second.join_columns[0].num_elems *
              sizeof(int32_t);
          auto row_counts_buffer = device_allocator->alloc(row_counts_buffer_sz);
          device_allocator->zeroDeviceMem(row_counts_buffer, row_counts_buffer_sz);
          const auto key_handler = BoundingBoxIntersectKeyHandler(
              inverse_bucket_sizes_for_dimension.size(),
              join_columns_gpu,
              reinterpret_cast<double*>(inverse_bucket_sizes_gpu));
          const auto key_handler_gpu = transfer_flat_object_to_gpu(
              key_handler, *device_allocator, "Boundingbox hash join key handler");
          approximate_distinct_tuples_on_device_bbox_intersect(
              reinterpret_cast<uint8_t*>(device_hll_buffer),
              count_distinct_desc.bitmap_sz_bits,
              reinterpret_cast<int32_t*>(row_counts_buffer),
              key_handler_gpu,
              columns_for_device.join_columns[0].num_elems,
              executor_->getCudaStream(device_id));

          auto& host_emitted_keys_count = emitted_keys_count_device_threads[device_id];
          device_allocator->copyFromDevice(
              &host_emitted_keys_count,
              row_counts_buffer +
                  (columns_per_device.begin()->second.join_columns[0].num_elems - 1) *
                      sizeof(int32_t),
              sizeof(int32_t),
              "Boundingbox join hashtable emitted keys count");

          auto& host_hll_buffer = host_hll_buffers[device_id];
          device_allocator->copyFromDevice(
              &host_hll_buffer[0],
              device_hll_buffer,
              count_distinct_desc.bitmapPaddedSizeBytes(),
              "Boundingbox join hashtable hyperloglog buffer");
        }));
  }
  for (auto& child : approximate_distinct_device_threads) {
    child.get();
  }
  CHECK_EQ(Data_Namespace::MemoryLevel::GPU_LEVEL, effective_memory_level);
  auto it = host_hll_buffers.begin();
  auto hll_result = reinterpret_cast<int32_t*>(&(it->second));
  it++;
  for (; it != host_hll_buffers.end(); it++) {
    auto& host_hll_buffer = it->second;
    hll_unify(hll_result,
              reinterpret_cast<int32_t*>(&host_hll_buffer[0]),
              size_t(1) << count_distinct_desc.bitmap_sz_bits);
  }
  size_t emitted_keys_count = 0;
  for (auto const& kv : emitted_keys_count_device_threads) {
    emitted_keys_count += kv.second;
  }
  return std::make_pair(hll_size(hll_result, count_distinct_desc.bitmap_sz_bits),
                        emitted_keys_count);
#else
  UNREACHABLE();
  return {0, 0};
#endif  // HAVE_CUDA
}

void BoundingBoxIntersectJoinHashTable::setInverseBucketSizeInfo(
    const std::vector<double>& inverse_bucket_sizes,
    std::unordered_map<int, ColumnsForDevice>& columns_per_device) {
  // set global bucket size
  inverse_bucket_sizes_for_dimension_ = inverse_bucket_sizes;

  // re-compute bucket counts per device based on global bucket size
  CHECK_EQ(columns_per_device.size(), device_ids_.size());
  for (auto& kv : columns_per_device) {
    auto& columns_for_device = kv.second;
    columns_for_device.setBucketInfo(inverse_bucket_sizes_for_dimension_,
                                     inner_outer_pairs_);
  }
}

size_t BoundingBoxIntersectJoinHashTable::getKeyComponentWidth() const {
  return 8;
}

size_t BoundingBoxIntersectJoinHashTable::getKeyComponentCount() const {
  CHECK(!inverse_bucket_sizes_for_dimension_.empty());
  return inverse_bucket_sizes_for_dimension_.size();
}

void BoundingBoxIntersectJoinHashTable::reify(const HashType preferred_layout) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_LT(0u, device_ids_.size());
  composite_key_info_ = HashJoin::getCompositeKeyInfo(inner_outer_pairs_, executor_);

  CHECK(condition_->is_bbox_intersect_oper());
  CHECK_EQ(inner_outer_pairs_.size(), size_t(1));
  HashType layout;
  if (inner_outer_pairs_[0].second->get_type_info().is_fixlen_array() &&
      inner_outer_pairs_[0].second->get_type_info().get_size() == 32) {
    // bounds array
    layout = HashType::ManyToMany;
  } else {
    layout = HashType::OneToMany;
  }
  try {
    reifyWithLayout(layout);
    return;
  } catch (const JoinHashTableTooBig& e) {
    throw e;
  } catch (const std::exception& e) {
    VLOG(1) << "Caught exception while building baseline hash table for bounding box "
               "intersection: "
            << e.what();
    throw;
  }
}

void BoundingBoxIntersectJoinHashTable::reifyImpl(
    std::unordered_map<int, ColumnsForDevice>& columns_per_device,
    const Fragmenter_Namespace::TableInfo& query_info,
    const HashType layout,
    const size_t shard_count,
    const size_t entry_count,
    const size_t emitted_keys_count,
    const bool skip_hashtable_caching,
    const size_t chosen_max_hashtable_size,
    const double chosen_bucket_threshold) {
  std::vector<std::future<void>> init_threads;
  chosen_bbox_intersect_bucket_threshold_ = chosen_bucket_threshold;
  chosen_bbox_intersect_max_table_size_bytes_ = chosen_max_hashtable_size;
  setBoundingBoxIntersectionMetaInfo(chosen_bbox_intersect_bucket_threshold_,
                                     chosen_bbox_intersect_max_table_size_bytes_,
                                     inverse_bucket_sizes_for_dimension_);
  for (auto device_id : device_ids_) {
    init_threads.push_back(std::async(std::launch::async,
                                      &BoundingBoxIntersectJoinHashTable::reifyForDevice,
                                      this,
                                      columns_per_device[device_id],
                                      layout,
                                      entry_count,
                                      emitted_keys_count,
                                      skip_hashtable_caching,
                                      device_id,
                                      logger::thread_local_ids()));
  }
  for (auto& init_thread : init_threads) {
    init_thread.wait();
  }
  for (auto& init_thread : init_threads) {
    init_thread.get();
  }
}

void BoundingBoxIntersectJoinHashTable::reifyForDevice(
    const ColumnsForDevice& columns_for_device,
    const HashType layout,
    const size_t entry_count,
    const size_t emitted_keys_count,
    const bool skip_hashtable_caching,
    const int device_id,
    const logger::ThreadLocalIds parent_thread_local_ids) {
  logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
  DEBUG_TIMER_NEW_THREAD(parent_thread_local_ids.thread_id_);
  CHECK_EQ(getKeyComponentWidth(), size_t(8));
  CHECK(layoutRequiresAdditionalBuffers(layout));
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);
  BaselineHashTableEntryInfo hash_table_entry_info(entry_count,
                                                   emitted_keys_count,
                                                   sizeof(int32_t),
                                                   getKeyComponentCount(),
                                                   getKeyComponentWidth(),
                                                   layout,
                                                   false);
  if (effective_memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
    VLOG(1) << "Building join hash table for bounding box intersection on CPU.";
    auto hash_table = initHashTableOnCpu(columns_for_device.join_columns,
                                         columns_for_device.join_column_types,
                                         columns_for_device.join_buckets,
                                         hash_table_entry_info,
                                         skip_hashtable_caching);
    CHECK(hash_table);

#ifdef HAVE_CUDA
    if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      copyCpuHashTableToGpu(hash_table, device_id);
    } else {
#else
    CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
      moveHashTableForDevice(std::move(hash_table), device_id);
#ifdef HAVE_CUDA
    }
#endif
  } else {
#ifdef HAVE_CUDA
    auto hash_table = initHashTableOnGpu(columns_for_device.join_columns,
                                         columns_for_device.join_column_types,
                                         columns_for_device.join_buckets,
                                         hash_table_entry_info,
                                         device_id);
    moveHashTableForDevice(std::move(hash_table), device_id);
#else
    UNREACHABLE();
#endif
  }
}

std::shared_ptr<BaselineHashTable> BoundingBoxIntersectJoinHashTable::initHashTableOnCpu(
    const std::vector<JoinColumn>& join_columns,
    const std::vector<JoinColumnTypeInfo>& join_column_types,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const BaselineHashTableEntryInfo hash_table_entry_info,
    const bool skip_hashtable_caching) {
  auto timer = DEBUG_TIMER(__func__);
  decltype(std::chrono::steady_clock::now()) ts1, ts2;
  ts1 = std::chrono::steady_clock::now();
  CHECK(!join_columns.empty());
  CHECK(!join_bucket_info.empty());
  std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
  auto const hash_table_layout = hash_table_entry_info.getHashTableLayout();
  if (auto generic_hash_table =
          initHashTableOnCpuFromCache(hashtable_cache_key_.begin()->second,
                                      CacheItemType::BBOX_INTERSECT_HT,
                                      DataRecyclerUtil::CPU_DEVICE_IDENTIFIER)) {
    if (auto hash_table =
            std::dynamic_pointer_cast<BaselineHashTable>(generic_hash_table)) {
      VLOG(1) << "Using cached CPU hash table for initialization.";
      // See if a hash table of a different layout was returned.
      // If it was OneToMany, we can reuse it on ManyToMany.
      if (hash_table_layout == HashType::ManyToMany &&
          hash_table->getLayout() == HashType::OneToMany) {
        // use the cached hash table
        layout_override_ = HashType::ManyToMany;
        return hash_table;
      }
      if (hash_table_layout == hash_table->getLayout()) {
        return hash_table;
      }
    }
  }
  CHECK(layoutRequiresAdditionalBuffers(hash_table_layout));
  const auto key_component_count =
      join_bucket_info[0].inverse_bucket_sizes_for_dimension.size();

  const auto key_handler = BoundingBoxIntersectKeyHandler(
      key_component_count,
      &join_columns[0],
      join_bucket_info[0].inverse_bucket_sizes_for_dimension.data());
  BaselineJoinHashTableBuilder builder;
  const StrProxyTranslationMapsPtrsAndOffsets
      dummy_str_proxy_translation_maps_ptrs_and_offsets;
  const auto err =
      builder.initHashTableOnCpu(&key_handler,
                                 composite_key_info_,
                                 join_columns,
                                 join_column_types,
                                 join_bucket_info,
                                 dummy_str_proxy_translation_maps_ptrs_and_offsets,
                                 hash_table_entry_info,
                                 join_type_,
                                 executor_,
                                 query_hints_);
  ts2 = std::chrono::steady_clock::now();
  if (err) {
    throw HashJoinFail(std::string("Unrecognized error when initializing CPU hash table "
                                   "for bounding box intersection(") +
                       std::to_string(err) + std::string(")"));
  }
  std::shared_ptr<BaselineHashTable> hash_table = builder.getHashTable();
  if (skip_hashtable_caching) {
    VLOG(1) << "Skip to cache join hashtable for bounding box intersection";
  } else {
    auto hashtable_build_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count();
    putHashTableOnCpuToCache(hashtable_cache_key_.begin()->second,
                             CacheItemType::BBOX_INTERSECT_HT,
                             hash_table,
                             DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                             hashtable_build_time);
  }
  return hash_table;
}

#ifdef HAVE_CUDA

std::shared_ptr<BaselineHashTable> BoundingBoxIntersectJoinHashTable::initHashTableOnGpu(
    const std::vector<JoinColumn>& join_columns,
    const std::vector<JoinColumnTypeInfo>& join_column_types,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const BaselineHashTableEntryInfo hash_table_entry_info,
    const size_t device_id) {
  CHECK_EQ(memory_level_, Data_Namespace::MemoryLevel::GPU_LEVEL);

  VLOG(1) << "Building join hash table for bounding box intersection on GPU.";

  BaselineJoinHashTableBuilder builder;
  auto device_allocator = executor_->getCudaAllocator(device_id);
  CHECK(device_allocator);
  auto join_columns_gpu = transfer_vector_of_flat_objects_to_gpu(
      join_columns, *device_allocator, "Boundingbox hash join input column(s)");
  CHECK_EQ(join_columns.size(), 1u);
  CHECK(!join_bucket_info.empty());
  auto& inverse_bucket_sizes_for_dimension =
      join_bucket_info[0].inverse_bucket_sizes_for_dimension;
  auto inverse_bucket_sizes_gpu =
      transfer_vector_of_flat_objects_to_gpu(inverse_bucket_sizes_for_dimension,
                                             *device_allocator,
                                             "Boundingbox join hashtable bucket sizes");
  const auto key_handler =
      BoundingBoxIntersectKeyHandler(inverse_bucket_sizes_for_dimension.size(),
                                     join_columns_gpu,
                                     inverse_bucket_sizes_gpu);

  const auto err = builder.initHashTableOnGpu(&key_handler,
                                              join_columns,
                                              join_type_,
                                              hash_table_entry_info,
                                              device_id,
                                              executor_,
                                              query_hints_);
  if (err) {
    throw HashJoinFail(std::string("Unrecognized error when initializing GPU hash table "
                                   "for bounding box intersection (") +
                       std::to_string(err) + std::string(")"));
  }
  return builder.getHashTable();
}

void BoundingBoxIntersectJoinHashTable::copyCpuHashTableToGpu(
    std::shared_ptr<BaselineHashTable>& cpu_hash_table,
    const size_t device_id) {
  CHECK_EQ(memory_level_, Data_Namespace::MemoryLevel::GPU_LEVEL);

  // copy hash table to GPU
  BaselineJoinHashTableBuilder gpu_builder;
  gpu_builder.allocateDeviceMemory(
      cpu_hash_table->getHashTableEntryInfo(), device_id, executor_, query_hints_);
  std::shared_ptr<BaselineHashTable> gpu_hash_table = gpu_builder.getHashTable();
  CHECK(gpu_hash_table);
  auto gpu_buffer_ptr = gpu_hash_table->getGpuBuffer();
  CHECK(gpu_buffer_ptr);

  CHECK_LE(cpu_hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU),
           gpu_hash_table->getHashTableBufferSize(ExecutorDeviceType::GPU));
  auto device_allocator = executor_->getCudaAllocator(device_id);
  CHECK(device_allocator);
  device_allocator->copyToDevice(
      gpu_buffer_ptr,
      cpu_hash_table->getCpuBuffer(),
      cpu_hash_table->getHashTableBufferSize(ExecutorDeviceType::CPU),
      "Boundingbox join hashtable");
  moveHashTableForDevice(std::move(gpu_hash_table), device_id);
}

#endif  // HAVE_CUDA

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_INT(v) executor_->cgen_state_->llInt(v)
#define LL_FP(v) executor_->cgen_state_->llFp(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

llvm::Value* BoundingBoxIntersectJoinHashTable::codegenKey(const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto key_component_width = getKeyComponentWidth();
  CHECK(key_component_width == 4 || key_component_width == 8);
  const auto key_size_lv = LL_INT(getKeyComponentCount() * key_component_width);
  llvm::Value* key_buff_lv{nullptr};
  switch (key_component_width) {
    case 4:
      key_buff_lv =
          LL_BUILDER.CreateAlloca(llvm::Type::getInt32Ty(LL_CONTEXT), key_size_lv);
      break;
    case 8:
      key_buff_lv =
          LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
      break;
    default:
      CHECK(false);
  }

  const auto& inner_outer_pair = inner_outer_pairs_[0];
  const auto outer_geo = inner_outer_pair.second;
  const auto outer_geo_ti = outer_geo->get_type_info();

  llvm::Value* arr_ptr = nullptr;
  CodeGenerator code_generator(executor_);
  CHECK_EQ(inverse_bucket_sizes_for_dimension_.size(), static_cast<size_t>(2));

  if (outer_geo_ti.is_geometry()) {
    // TODO(adb): for points we will use the coords array, but for other geometries we
    // will need to use the bounding box. For now only support points.
    CHECK_EQ(outer_geo_ti.get_type(), kPOINT);

    if (const auto outer_geo_col = dynamic_cast<const Analyzer::ColumnVar*>(outer_geo)) {
      const auto outer_geo_col_lvs = code_generator.codegen(outer_geo_col, true, co);
      CHECK_EQ(outer_geo_col_lvs.size(), size_t(1));
      auto column_key = outer_geo_col->getColumnKey();
      column_key.column_id = column_key.column_id + 1;
      const auto coords_cd = Catalog_Namespace::get_metadata_for_column(column_key);
      CHECK(coords_cd);

      const auto array_ptr = executor_->cgen_state_->emitExternalCall(
          "array_buff",
          llvm::Type::getInt8PtrTy(executor_->cgen_state_->context_),
          {outer_geo_col_lvs.front(), code_generator.posArg(outer_geo_col)});
      CHECK(coords_cd->columnType.get_elem_type().get_type() == kTINYINT)
          << "Bounding box intersection only supports TINYINT coordinates columns.";
      arr_ptr = code_generator.castArrayPointer(array_ptr,
                                                coords_cd->columnType.get_elem_type());
    } else if (const auto outer_geo_function_operator =
                   dynamic_cast<const Analyzer::GeoOperator*>(outer_geo)) {
      // Process points dynamically constructed by geo function operators
      const auto outer_geo_function_operator_lvs =
          code_generator.codegen(outer_geo_function_operator, true, co);
      CHECK_EQ(outer_geo_function_operator_lvs.size(), size_t(2));
      arr_ptr = outer_geo_function_operator_lvs.front();
    } else if (const auto outer_geo_expr =
                   dynamic_cast<const Analyzer::GeoExpr*>(outer_geo)) {
      UNREACHABLE() << outer_geo_expr->toString();
    }
  } else if (outer_geo_ti.is_fixlen_array()) {
    // Process dynamically constructed points
    const auto outer_geo_cast_coord_array =
        dynamic_cast<const Analyzer::UOper*>(outer_geo);
    CHECK_EQ(outer_geo_cast_coord_array->get_optype(), kCAST);
    const auto outer_geo_coord_array = dynamic_cast<const Analyzer::ArrayExpr*>(
        outer_geo_cast_coord_array->get_operand());
    CHECK(outer_geo_coord_array);
    CHECK(outer_geo_coord_array->isLocalAlloc());
    CHECK_EQ(outer_geo_coord_array->getElementCount(), 2);
    auto elem_size = (outer_geo_ti.get_compression() == kENCODING_GEOINT)
                         ? sizeof(int32_t)
                         : sizeof(double);
    CHECK_EQ(outer_geo_ti.get_size(), int(2 * elem_size));
    const auto outer_geo_constructed_lvs = code_generator.codegen(outer_geo, true, co);
    // CHECK_EQ(outer_geo_constructed_lvs.size(), size_t(2));     // Pointer and size
    const auto array_ptr = outer_geo_constructed_lvs.front();  // Just need the pointer
    arr_ptr = LL_BUILDER.CreateGEP(
        array_ptr->getType()->getScalarType()->getPointerElementType(),
        array_ptr,
        LL_INT(0));
    arr_ptr = code_generator.castArrayPointer(array_ptr, SQLTypeInfo(kTINYINT, true));
  }
  if (!arr_ptr) {
    LOG(FATAL)
        << "Bounding box intersection currently only supports geospatial columns and "
           "constructed points.";
  }

  for (size_t i = 0; i < 2; i++) {
    const auto key_comp_dest_lv = LL_BUILDER.CreateGEP(
        key_buff_lv->getType()->getScalarType()->getPointerElementType(),
        key_buff_lv,
        LL_INT(i));

    // Note that get_bucket_key_for_range_compressed will need to be specialized for
    // future compression schemes
    auto bucket_key =
        outer_geo_ti.get_compression() == kENCODING_GEOINT
            ? executor_->cgen_state_->emitExternalCall(
                  "get_bucket_key_for_range_compressed",
                  get_int_type(64, LL_CONTEXT),
                  {arr_ptr, LL_INT(i), LL_FP(inverse_bucket_sizes_for_dimension_[i])})
            : executor_->cgen_state_->emitExternalCall(
                  "get_bucket_key_for_range_double",
                  get_int_type(64, LL_CONTEXT),
                  {arr_ptr, LL_INT(i), LL_FP(inverse_bucket_sizes_for_dimension_[i])});
    const auto col_lv = LL_BUILDER.CreateSExt(
        bucket_key, get_int_type(key_component_width * 8, LL_CONTEXT));
    LL_BUILDER.CreateStore(col_lv, key_comp_dest_lv);
  }
  return key_buff_lv;
}

std::vector<llvm::Value*> BoundingBoxIntersectJoinHashTable::codegenManyKey(
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto key_component_width = getKeyComponentWidth();
  CHECK(key_component_width == 4 || key_component_width == 8);
  auto hash_table = getAnyHashTableForDevice();
  CHECK(hash_table);
  CHECK(getHashType() == HashType::ManyToMany);

  VLOG(1) << "Performing codgen for ManyToMany";
  const auto& inner_outer_pair = inner_outer_pairs_[0];
  const auto outer_col = inner_outer_pair.second;

  CodeGenerator code_generator(executor_);
  const auto col_lvs = code_generator.codegen(outer_col, true, co);
  CHECK_EQ(col_lvs.size(), size_t(1));

  const auto outer_col_var = dynamic_cast<const Analyzer::ColumnVar*>(outer_col);
  CHECK(outer_col_var);
  const auto coords_cd =
      Catalog_Namespace::get_metadata_for_column(outer_col_var->getColumnKey());
  CHECK(coords_cd);

  const auto array_ptr = executor_->cgen_state_->emitExternalCall(
      "array_buff",
      llvm::Type::getInt8PtrTy(executor_->cgen_state_->context_),
      {col_lvs.front(), code_generator.posArg(outer_col)});

  // TODO(jclay): this seems to cast to double, and causes the GPU build to fail.
  // const auto arr_ptr =
  //     code_generator.castArrayPointer(array_ptr,
  //     coords_cd->columnType.get_elem_type());
  array_ptr->setName("array_ptr");

  auto num_keys_lv = executor_->cgen_state_->emitExternalCall(
      "get_num_buckets_for_bounds",
      get_int_type(32, LL_CONTEXT),
      {array_ptr,
       LL_INT(0),
       LL_FP(inverse_bucket_sizes_for_dimension_[0]),
       LL_FP(inverse_bucket_sizes_for_dimension_[1])});
  num_keys_lv->setName("num_keys_lv");

  return {num_keys_lv, array_ptr};
}

HashJoinMatchingSet BoundingBoxIntersectJoinHashTable::codegenMatchingSet(
    const CompilationOptions& co,
    const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  if (getHashType() == HashType::ManyToMany) {
    VLOG(1) << "Building codegenMatchingSet for ManyToMany";
    const auto key_component_width = getKeyComponentWidth();
    CHECK(key_component_width == 4 || key_component_width == 8);
    auto many_to_many_args = codegenManyKey(co);
    auto hash_ptr = HashJoin::codegenHashTableLoad(index, executor_);
    const auto composite_dict_ptr_type =
        llvm::Type::getIntNPtrTy(LL_CONTEXT, key_component_width * 8);
    const auto composite_key_dict =
        hash_ptr->getType()->isPointerTy()
            ? LL_BUILDER.CreatePointerCast(hash_ptr, composite_dict_ptr_type)
            : LL_BUILDER.CreateIntToPtr(hash_ptr, composite_dict_ptr_type);
    const auto key_component_count = getKeyComponentCount();

    auto one_to_many_ptr = hash_ptr;

    if (one_to_many_ptr->getType()->isPointerTy()) {
      one_to_many_ptr =
          LL_BUILDER.CreatePtrToInt(hash_ptr, llvm::Type::getInt64Ty(LL_CONTEXT));
    } else {
      CHECK(one_to_many_ptr->getType()->isIntegerTy(64));
    }

    const auto composite_key_dict_size = offsetBufferOff();
    one_to_many_ptr =
        LL_BUILDER.CreateAdd(one_to_many_ptr, LL_INT(composite_key_dict_size));

    const auto arr_type = get_int_array_type(32, kMaxBBoxOverlapsCount, LL_CONTEXT);
    const auto out_arr_lv = LL_BUILDER.CreateAlloca(arr_type);
    out_arr_lv->setName("out_arr");

    const auto casted_out_arr_lv =
        LL_BUILDER.CreatePointerCast(out_arr_lv, arr_type->getPointerTo());

    const auto element_ptr = LL_BUILDER.CreateGEP(arr_type, casted_out_arr_lv, LL_INT(0));

    auto rowid_ptr_i32 =
        LL_BUILDER.CreatePointerCast(element_ptr, llvm::Type::getInt32PtrTy(LL_CONTEXT));

    const auto error_code_ptr = LL_BUILDER.CreateAlloca(
        get_int_type(32, LL_CONTEXT), nullptr, "candidate_rows_error_code");
    LL_BUILDER.CreateStore(LL_INT(int32_t(0)), error_code_ptr);

    const auto candidate_count_lv = executor_->cgen_state_->emitExternalCall(
        "get_candidate_rows",
        llvm::Type::getInt64Ty(LL_CONTEXT),
        {rowid_ptr_i32,
         error_code_ptr,
         LL_INT(kMaxBBoxOverlapsCount),
         many_to_many_args[1],
         LL_INT(0),
         LL_FP(inverse_bucket_sizes_for_dimension_[0]),
         LL_FP(inverse_bucket_sizes_for_dimension_[1]),
         many_to_many_args[0],
         LL_INT(key_component_count),                // key_component_count
         composite_key_dict,                         // ptr to hash table
         LL_INT(getEntryCount()),                    // entry_count
         LL_INT(composite_key_dict_size),            // offset_buffer_ptr_offset
         LL_INT(getEntryCount() * sizeof(int32_t)),  // sub_buff_size
         LL_INT(int32_t(heavyai::ErrorCode::BBOX_OVERLAPS_LIMIT_EXCEEDED))});

    const auto slot_lv = LL_INT(int64_t(0));
    auto error_code_lv = LL_BUILDER.CreateLoad(
        error_code_ptr->getType()->getPointerElementType(), error_code_ptr);
    return {rowid_ptr_i32, candidate_count_lv, slot_lv, error_code_lv};
  } else {
    VLOG(1) << "Building codegenMatchingSet for Baseline";
    // TODO: duplicated w/ BaselineJoinHashTable -- push into the hash table builder?
    const auto key_component_width = getKeyComponentWidth();
    CHECK(key_component_width == 4 || key_component_width == 8);
    auto key_buff_lv = codegenKey(co);
    CHECK(getHashType() == HashType::OneToMany);
    auto hash_ptr = HashJoin::codegenHashTableLoad(index, executor_);
    const auto composite_dict_ptr_type =
        llvm::Type::getIntNPtrTy(LL_CONTEXT, key_component_width * 8);
    const auto composite_key_dict =
        hash_ptr->getType()->isPointerTy()
            ? LL_BUILDER.CreatePointerCast(hash_ptr, composite_dict_ptr_type)
            : LL_BUILDER.CreateIntToPtr(hash_ptr, composite_dict_ptr_type);
    const auto key_component_count = getKeyComponentCount();
    const auto key = executor_->cgen_state_->emitExternalCall(
        "get_composite_key_index_" + std::to_string(key_component_width * 8),
        get_int_type(64, LL_CONTEXT),
        {key_buff_lv,
         LL_INT(key_component_count),
         composite_key_dict,
         LL_INT(getEntryCount())});
    auto one_to_many_ptr = hash_ptr;
    if (one_to_many_ptr->getType()->isPointerTy()) {
      one_to_many_ptr =
          LL_BUILDER.CreatePtrToInt(hash_ptr, llvm::Type::getInt64Ty(LL_CONTEXT));
    } else {
      CHECK(one_to_many_ptr->getType()->isIntegerTy(64));
    }
    const auto composite_key_dict_size = offsetBufferOff();
    one_to_many_ptr =
        LL_BUILDER.CreateAdd(one_to_many_ptr, LL_INT(composite_key_dict_size));
    return HashJoin::codegenMatchingSet(
        std::vector<llvm::Value*>{
            one_to_many_ptr, key, LL_INT(int64_t(0)), LL_INT(getEntryCount() - 1)},
        false,
        false,
        false,
        getComponentBufferSize(),
        executor_);
  }
  UNREACHABLE();
  return HashJoinMatchingSet{};
}

std::string BoundingBoxIntersectJoinHashTable::toString(
    const ExecutorDeviceType device_type,
    const int device_id,
    bool raw) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  if (!buffer) {
    return "EMPTY";
  }
  auto hash_table = getHashTableForDevice(device_id);
  auto buffer_size = hash_table->getHashTableBufferSize(device_type);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);
    CHECK(executor_);
    auto device_allocator = executor_->getCudaAllocator(device_id);
    CHECK(device_allocator);
    device_allocator->copyFromDevice(
        buffer_copy.get(), buffer, buffer_size, "Boundingbox join hashtable");
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : reinterpret_cast<const int8_t*>(buffer);
#else
  auto ptr1 = reinterpret_cast<const int8_t*>(buffer);
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  CHECK(hash_table);
  const auto layout = getHashType();
  return HashTable::toString(
      "geo",
      getHashTypeString(layout),
      getKeyComponentCount() + (layout == HashType::OneToOne ? 1 : 0),
      getKeyComponentWidth(),
      hash_table->getEntryCount(),
      ptr1,
      ptr2,
      ptr3,
      ptr4,
      buffer_size,
      raw);
}

std::set<DecodedJoinHashBufferEntry> BoundingBoxIntersectJoinHashTable::toSet(
    const ExecutorDeviceType device_type,
    const int device_id) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto hash_table = getHashTableForDevice(device_id);
  CHECK(hash_table);
  auto buffer_size = hash_table->getHashTableBufferSize(device_type);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);
    CHECK(executor_);
    auto device_allocator = executor_->getCudaAllocator(device_id);
    CHECK(device_allocator);
    device_allocator->copyFromDevice(
        buffer_copy.get(), buffer, buffer_size, "Boundingbox join hashtable");
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : reinterpret_cast<const int8_t*>(buffer);
#else
  auto ptr1 = reinterpret_cast<const int8_t*>(buffer);
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  const auto layout = getHashType();
  return HashTable::toSet(getKeyComponentCount() + (layout == HashType::OneToOne ? 1 : 0),
                          getKeyComponentWidth(),
                          hash_table->getEntryCount(),
                          ptr1,
                          ptr2,
                          ptr3,
                          ptr4,
                          buffer_size);
}

Data_Namespace::MemoryLevel BoundingBoxIntersectJoinHashTable::getEffectiveMemoryLevel(
    const std::vector<InnerOuter>& inner_outer_pairs) const {
  if (query_hints_.isHintRegistered(QueryHint::kBBoxIntersectAllowGpuBuild) &&
      query_hints_.bbox_intersect_allow_gpu_build &&
      this->executor_->getDataMgr()->gpusPresent() &&
      memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
    return Data_Namespace::MemoryLevel::GPU_LEVEL;
  }
  // otherwise, try to build on CPU
  return Data_Namespace::MemoryLevel::CPU_LEVEL;
}

shared::TableKey BoundingBoxIntersectJoinHashTable::getInnerTableId() const noexcept {
  try {
    return HashJoin::getInnerTableId(inner_outer_pairs_);
  } catch (...) {
    CHECK(false);
  }
  return {};
}

std::shared_ptr<HashTable> BoundingBoxIntersectJoinHashTable::initHashTableOnCpuFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier) {
  auto timer = DEBUG_TIMER(__func__);
  VLOG(1) << "Checking CPU hash table cache.";
  CHECK(hash_table_cache_);
  HashtableCacheMetaInfo meta_info;
  meta_info.bbox_intersect_meta_info = getBoundingBoxIntersectMetaInfo();
  auto cached_hashtable =
      hash_table_cache_->getItemFromCache(key, item_type, device_identifier, meta_info);
  if (cached_hashtable) {
    return cached_hashtable;
  }
  return nullptr;
}

std::optional<std::pair<size_t, size_t>>
BoundingBoxIntersectJoinHashTable::getApproximateTupleCountFromCache(
    QueryPlanHash key,
    CacheItemType item_type,
    DeviceIdentifier device_identifier) {
  CHECK(hash_table_cache_);
  HashtableCacheMetaInfo metaInfo;
  metaInfo.bbox_intersect_meta_info = getBoundingBoxIntersectMetaInfo();
  auto cached_hashtable =
      hash_table_cache_->getItemFromCache(key, item_type, device_identifier, metaInfo);
  if (cached_hashtable) {
    return std::make_pair(cached_hashtable->getEntryCount() / 2,
                          cached_hashtable->getEmittedKeysCount());
  }
  return std::nullopt;
}

void BoundingBoxIntersectJoinHashTable::putHashTableOnCpuToCache(
    QueryPlanHash key,
    CacheItemType item_type,
    std::shared_ptr<HashTable> hashtable_ptr,
    DeviceIdentifier device_identifier,
    size_t hashtable_building_time) {
  CHECK(hash_table_cache_);
  CHECK(hashtable_ptr && !hashtable_ptr->getGpuBuffer());
  HashtableCacheMetaInfo meta_info;
  meta_info.bbox_intersect_meta_info = getBoundingBoxIntersectMetaInfo();
  meta_info.registered_query_hint = query_hints_;
  hash_table_cache_->putItemToCache(
      key,
      hashtable_ptr,
      item_type,
      device_identifier,
      hashtable_ptr->getHashTableBufferSize(ExecutorDeviceType::CPU),
      hashtable_building_time,
      meta_info);
}

bool BoundingBoxIntersectJoinHashTable::isBitwiseEq() const {
  return condition_->get_optype() == kBW_EQ;
}
