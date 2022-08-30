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

#ifndef OMNISCI_QUERYHINT_H
#define OMNISCI_QUERYHINT_H

#include <algorithm>
#include <optional>

#include <boost/algorithm/string.hpp>

#include "ThriftHandler/CommandLineOptions.h"

// we expect query hint enum val starts with zero,
// and let remaining enum value to be auto-incremented
enum QueryHint {
  kCpuMode = 0,
  kColumnarOutput,
  kRowwiseOutput,
  kOverlapsBucketThreshold,
  kOverlapsMaxSize,
  kOverlapsAllowGpuBuild,
  kOverlapsNoCache,
  kOverlapsKeysPerBin,
  kKeepResult,
  kKeepTableFuncResult,
  kAggregateTreeFanout,
  kCudaBlockSize,
  kCudaGridSize,
  kOptCudaBlockAndGridSizes,
  kWatchdog,
  kDynamicWatchdog,
  kWatchdogOff,
  kDynamicWatchdogOff,
  kQueryTimeLimit,
  kAllowLoopJoin,
  kDisableLoopJoin,
  kLoopJoinInnerTableMaxNumRows,
  kMaxJoinHashTableSize,
  kHintCount,   // should be at the last elem before INVALID enum value to count #
                // supported hints correctly
  kInvalidHint  // this should be the last elem of this enum
};

static const std::unordered_map<std::string, QueryHint> SupportedQueryHints = {
    {"cpu_mode", QueryHint::kCpuMode},
    {"columnar_output", QueryHint::kColumnarOutput},
    {"rowwise_output", QueryHint::kRowwiseOutput},
    {"overlaps_bucket_threshold", QueryHint::kOverlapsBucketThreshold},
    {"overlaps_max_size", QueryHint::kOverlapsMaxSize},
    {"overlaps_allow_gpu_build", QueryHint::kOverlapsAllowGpuBuild},
    {"overlaps_no_cache", QueryHint::kOverlapsNoCache},
    {"overlaps_keys_per_bin", QueryHint::kOverlapsKeysPerBin},
    {"keep_result", QueryHint::kKeepResult},
    {"keep_table_function_result", QueryHint::kKeepTableFuncResult},
    {"aggregate_tree_fanout", QueryHint::kAggregateTreeFanout},
    {"cuda_block_size", QueryHint::kCudaBlockSize},
    {"cuda_grid_size_multiplier", QueryHint::kCudaGridSize},
    {"cuda_opt_block_and_grid_sizes", kOptCudaBlockAndGridSizes},
    {"watchdog", QueryHint::kWatchdog},
    {"dynamic_watchdog", QueryHint::kDynamicWatchdog},
    {"watchdog_off", QueryHint::kWatchdogOff},
    {"dynamic_watchdog_off", QueryHint::kDynamicWatchdogOff},
    {"query_time_limit", QueryHint::kQueryTimeLimit},
    {"allow_loop_join", QueryHint::kAllowLoopJoin},
    {"disable_loop_join", QueryHint::kDisableLoopJoin},
    {"loop_join_inner_table_max_num_rows", QueryHint::kLoopJoinInnerTableMaxNumRows},
    {"max_join_hashtable_size", QueryHint::kMaxJoinHashTableSize}};

struct HintIdentifier {
  bool global_hint;
  std::string hint_name;

  HintIdentifier(bool global_hint, const std::string& hint_name)
      : global_hint(global_hint), hint_name(hint_name){};
};

class ExplainedQueryHint {
  // this class represents parsed query hint's specification
  // our query AST analyzer translates query hint string to understandable form which we
  // called "ExplainedQueryHint"
 public:
  // default constructor used for deserialization only
  ExplainedQueryHint()
      : hint_{QueryHint::kInvalidHint}
      , global_hint_{false}
      , is_marker_{false}
      , has_kv_type_options_{false} {}

  ExplainedQueryHint(QueryHint hint,
                     bool global_hint,
                     bool is_marker,
                     bool has_kv_type_options)
      : hint_(hint)
      , global_hint_(global_hint)
      , is_marker_(is_marker)
      , has_kv_type_options_(has_kv_type_options) {}

  ExplainedQueryHint(QueryHint hint,
                     bool global_hint,
                     bool is_marker,
                     bool has_kv_type_options,
                     std::vector<std::string>& list_options)
      : hint_(hint)
      , global_hint_(global_hint)
      , is_marker_(is_marker)
      , has_kv_type_options_(has_kv_type_options)
      , list_options_(std::move(list_options)) {}

  ExplainedQueryHint(QueryHint hint,
                     bool global_hint,
                     bool is_marker,
                     bool has_kv_type_options,
                     std::unordered_map<std::string, std::string>& kv_options)
      : hint_(hint)
      , global_hint_(global_hint)
      , is_marker_(is_marker)
      , has_kv_type_options_(has_kv_type_options)
      , kv_options_(std::move(kv_options)) {}

  void setListOptions(std::vector<std::string>& list_options) {
    list_options_ = list_options;
  }

  void setKVOptions(std::unordered_map<std::string, std::string>& kv_options) {
    kv_options_ = kv_options;
  }

  void setInheritPaths(std::vector<int>& interit_paths) {
    inherit_paths_ = interit_paths;
  }

  const std::vector<std::string>& getListOptions() const { return list_options_; }

  const std::vector<int>& getInteritPath() const { return inherit_paths_; }

  const std::unordered_map<std::string, std::string>& getKVOptions() const {
    return kv_options_;
  }

  const QueryHint getHint() const { return hint_; }

  bool isGlobalHint() const { return global_hint_; }

  bool hasOptions() const { return is_marker_; }

  bool hasKvOptions() const { return has_kv_type_options_; }

 private:
  QueryHint hint_;
  // Set true if this hint affects globally
  // Otherwise it just affects the node which this hint is included (aka table hint)
  bool global_hint_;
  // set true if this has no extra options (neither list_options nor kv_options)
  bool is_marker_;
  // Set true if it is not a marker and has key-value type options
  // Otherwise (it is not a marker but has list type options), we set this be false
  bool has_kv_type_options_;
  std::vector<int> inherit_paths_;  // currently not used
  std::vector<std::string> list_options_;
  std::unordered_map<std::string, std::string> kv_options_;
};

struct RegisteredQueryHint {
  // for each query hint, we first translate the raw query hint info
  // to understandable form called "ExplainedQueryHint"
  // and we get all necessary info from it and organize it into "RegisteredQueryHint"
  // so by using "RegisteredQueryHint", we can know and access which query hint is
  // registered and its detailed info such as the hint's parameter values given by user
  // NOTE: after changing query hint fields, we "SHOULD" also update the corresponding
  // "QueryHintSerializer" accordingly
  RegisteredQueryHint()
      : cpu_mode(false)
      , columnar_output(false)
      , rowwise_output(false)
      , keep_result(false)
      , keep_table_function_result(false)
      , watchdog(std::nullopt)
      , dynamic_watchdog(std::nullopt)
      , query_time_limit(0)
      , cuda_block_size(0)
      , cuda_grid_size_multiplier(0.0)
      , opt_cuda_grid_and_block_size(false)
      , aggregate_tree_fanout(8)
      , overlaps_bucket_threshold(std::numeric_limits<double>::max())
      , overlaps_max_size(g_overlaps_max_table_size_bytes)
      , overlaps_allow_gpu_build(false)
      , overlaps_no_cache(false)
      , overlaps_keys_per_bin(g_overlaps_target_entries_per_bin)
      , use_loop_join(std::nullopt)
      , loop_join_inner_table_max_num_rows(g_trivial_loop_join_threshold)
      , max_join_hash_table_size(std::numeric_limits<size_t>::max())
      , registered_hint(QueryHint::kHintCount, false) {}

  RegisteredQueryHint operator||(const RegisteredQueryHint& global_hints) const {
    CHECK_EQ(registered_hint.size(), global_hints.registered_hint.size());
    // apply registered global hint to the local hint if necessary
    // we prioritize global hint when both side of hints are enabled simultaneously
    RegisteredQueryHint updated_query_hints(*this);

    constexpr int num_hints = static_cast<int>(QueryHint::kHintCount);
    for (int i = 0; i < num_hints; ++i) {
      if (global_hints.registered_hint.at(i)) {
        updated_query_hints.registered_hint.at(i) = true;
        switch (static_cast<QueryHint>(i)) {
          case QueryHint::kCpuMode:
            updated_query_hints.cpu_mode = true;
            break;
          case QueryHint::kColumnarOutput:
            updated_query_hints.columnar_output = true;
            break;
          case QueryHint::kRowwiseOutput:
            updated_query_hints.rowwise_output = true;
            break;
          case QueryHint::kCudaBlockSize:
            updated_query_hints.cuda_block_size = global_hints.cuda_block_size;
            break;
          case QueryHint::kCudaGridSize:
            updated_query_hints.cuda_grid_size_multiplier =
                global_hints.cuda_grid_size_multiplier;
            break;
          case QueryHint::kOptCudaBlockAndGridSizes:
            updated_query_hints.opt_cuda_grid_and_block_size = true;
            break;
          case QueryHint::kOverlapsBucketThreshold:
            updated_query_hints.overlaps_bucket_threshold =
                global_hints.overlaps_bucket_threshold;
            break;
          case QueryHint::kOverlapsMaxSize:
            updated_query_hints.overlaps_max_size = global_hints.overlaps_max_size;
            break;
          case QueryHint::kOverlapsAllowGpuBuild:
            updated_query_hints.overlaps_allow_gpu_build = true;
            break;
          case QueryHint::kOverlapsNoCache:
            updated_query_hints.overlaps_no_cache = true;
            break;
          case QueryHint::kOverlapsKeysPerBin:
            updated_query_hints.overlaps_keys_per_bin =
                global_hints.overlaps_keys_per_bin;
            break;
          case QueryHint::kKeepResult:
            updated_query_hints.keep_result = global_hints.keep_result;
            break;
          case QueryHint::kKeepTableFuncResult:
            updated_query_hints.keep_table_function_result =
                global_hints.keep_table_function_result;
            break;
          case QueryHint::kAggregateTreeFanout:
            updated_query_hints.aggregate_tree_fanout =
                global_hints.aggregate_tree_fanout;
            break;
          case QueryHint::kWatchdog:
          case QueryHint::kWatchdogOff:
            updated_query_hints.watchdog = global_hints.watchdog;
            break;
          case QueryHint::kDynamicWatchdog:
          case QueryHint::kDynamicWatchdogOff:
            updated_query_hints.dynamic_watchdog = global_hints.dynamic_watchdog;
            break;
          case QueryHint::kQueryTimeLimit:
            updated_query_hints.query_time_limit = global_hints.query_time_limit;
            break;
          case QueryHint::kAllowLoopJoin:
          case QueryHint::kDisableLoopJoin:
            updated_query_hints.use_loop_join = global_hints.use_loop_join;
            break;
          case QueryHint::kLoopJoinInnerTableMaxNumRows:
            updated_query_hints.loop_join_inner_table_max_num_rows =
                global_hints.loop_join_inner_table_max_num_rows;
            break;
          case QueryHint::kMaxJoinHashTableSize:
            updated_query_hints.max_join_hash_table_size =
                global_hints.max_join_hash_table_size;
            break;
          default:
            UNREACHABLE();
        }
      }
    }
    return updated_query_hints;
  }

  // general query execution
  bool cpu_mode;
  bool columnar_output;
  bool rowwise_output;
  bool keep_result;
  bool keep_table_function_result;
  std::optional<bool> watchdog;
  std::optional<bool> dynamic_watchdog;
  size_t query_time_limit;

  // control CUDA behavior
  size_t cuda_block_size;
  double cuda_grid_size_multiplier;
  bool opt_cuda_grid_and_block_size;

  // window function framing
  size_t aggregate_tree_fanout;

  // overlaps hash join
  double overlaps_bucket_threshold;  // defined in "OverlapsJoinHashTable.h"
  size_t overlaps_max_size;
  bool overlaps_allow_gpu_build;
  bool overlaps_no_cache;
  double overlaps_keys_per_bin;

  // generic hash join
  std::optional<bool> use_loop_join;
  size_t loop_join_inner_table_max_num_rows;
  size_t max_join_hash_table_size;

  std::vector<bool> registered_hint;

  static RegisteredQueryHint defaults() { return RegisteredQueryHint(); }

 public:
  static QueryHint translateQueryHint(const std::string& hint_name) {
    const auto lowered_hint_name = boost::algorithm::to_lower_copy(hint_name);
    auto it = SupportedQueryHints.find(lowered_hint_name);
    return it == SupportedQueryHints.end() ? QueryHint::kInvalidHint : it->second;
  }

  bool isAnyQueryHintDelivered() const {
    const auto identity = [](const bool b) { return b; };
    return std::any_of(registered_hint.begin(), registered_hint.end(), identity);
  }

  void registerHint(const QueryHint hint) {
    const auto hint_class = static_cast<int>(hint);
    registered_hint.at(hint_class) = true;
  }

  bool isHintRegistered(const QueryHint hint) const {
    const auto hint_class = static_cast<int>(hint);
    return registered_hint.at(hint_class);
  }
};

// a map from hint_name to its detailed info
using Hints = std::unordered_map<QueryHint, ExplainedQueryHint>;

#endif  // OMNISCI_QUERYHINT_H
