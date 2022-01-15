/*
 * Copyright 2020 OmniSci, Inc.
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
    {"overlaps_keys_per_bin", QueryHint::kOverlapsKeysPerBin}};

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

  const std::vector<std::string>& getListOptions() { return list_options_; }

  const std::vector<int>& getInteritPath() { return inherit_paths_; }

  const std::unordered_map<std::string, std::string>& getKVOptions() {
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
  RegisteredQueryHint()
      : cpu_mode(false)
      , columnar_output(false)
      , rowwise_output(false)
      , overlaps_bucket_threshold(std::numeric_limits<double>::max())
      , overlaps_max_size(g_overlaps_max_table_size_bytes)
      , overlaps_allow_gpu_build(false)
      , overlaps_no_cache(false)
      , overlaps_keys_per_bin(g_overlaps_target_entries_per_bin)
      , registered_hint(QueryHint::kHintCount, false) {}

  RegisteredQueryHint operator||(const RegisteredQueryHint& global_hints) const {
    CHECK_EQ(registered_hint.size(), global_hints.registered_hint.size());
    // apply registered global hint to the local hint if necessary
    // we prioritize global hint when both side of hints are enabled simultaneously
    RegisteredQueryHint updated_query_hints(*this);

    int num_hints = static_cast<int>(QueryHint::kHintCount);
    for (int i = 0; i < num_hints; ++i) {
      if (global_hints.registered_hint.at(i)) {
        updated_query_hints.registered_hint.at(i) = global_hints.registered_hint[i];
        switch (i) {
          case static_cast<int>(QueryHint::kCpuMode): {
            updated_query_hints.cpu_mode = true;
            break;
          }
          case static_cast<int>(QueryHint::kColumnarOutput): {
            updated_query_hints.columnar_output = true;
            break;
          }
          case static_cast<int>(QueryHint::kRowwiseOutput): {
            updated_query_hints.rowwise_output = true;
            break;
          }
          case static_cast<int>(QueryHint::kOverlapsBucketThreshold): {
            updated_query_hints.overlaps_bucket_threshold =
                global_hints.overlaps_bucket_threshold;
            break;
          }
          case static_cast<int>(QueryHint::kOverlapsMaxSize): {
            updated_query_hints.overlaps_max_size = global_hints.overlaps_max_size;
            break;
          }
          case static_cast<int>(QueryHint::kOverlapsAllowGpuBuild): {
            updated_query_hints.overlaps_allow_gpu_build = true;
            break;
          }
          case static_cast<int>(QueryHint::kOverlapsNoCache): {
            updated_query_hints.overlaps_no_cache = true;
            break;
          }
          case static_cast<int>(QueryHint::kOverlapsKeysPerBin): {
            updated_query_hints.overlaps_keys_per_bin =
                global_hints.overlaps_keys_per_bin;
            break;
          }
        }
      }
    }
    return updated_query_hints;
  }

  // general query execution
  bool cpu_mode;
  bool columnar_output;
  bool rowwise_output;

  // overlaps hash join
  double overlaps_bucket_threshold;  // defined in "OverlapsJoinHashTable.h"
  size_t overlaps_max_size;
  bool overlaps_allow_gpu_build;
  bool overlaps_no_cache;
  double overlaps_keys_per_bin;

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
