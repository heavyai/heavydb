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

#include <optional>

#include <boost/algorithm/string.hpp>

#include "Shared/Config.h"

// we expect query hint enum val starts with zero,
// and let remaining enum value to be auto-incremented
enum QueryHint {
  kCpuMode = 0,
  kColumnarOutput,
  kRowwiseOutput,
  kHintCount,   // should be at the last elem before INVALID enum value to count #
                // supported hints correctly
  kInvalidHint  // this should be the last elem of this enum
};

static const std::unordered_map<std::string, QueryHint> SupportedQueryHints = {
    {"cpu_mode", QueryHint::kCpuMode},
    {"columnar_output", QueryHint::kColumnarOutput},
    {"rowwise_output", QueryHint::kRowwiseOutput}};

class ExplainedQueryHint {
  // this class represents parsed query hint's specification
  // our query AST analyzer translates query hint string to understandable form which we
  // called "ExplainedQueryHint"
 public:
  ExplainedQueryHint(QueryHint hint,
                     bool query_hint,
                     bool is_marker,
                     bool has_kv_type_options)
      : hint_(hint)
      , query_hint_(query_hint)
      , is_marker_(is_marker)
      , has_kv_type_options_(has_kv_type_options) {}

  ExplainedQueryHint(QueryHint hint,
                     bool query_hint,
                     bool is_marker,
                     bool has_kv_type_options,
                     std::vector<std::string>& list_options)
      : hint_(hint)
      , query_hint_(query_hint)
      , is_marker_(is_marker)
      , has_kv_type_options_(has_kv_type_options)
      , list_options_(std::move(list_options)) {}

  ExplainedQueryHint(QueryHint hint,
                     bool query_hint,
                     bool is_marker,
                     bool has_kv_type_options,
                     std::unordered_map<std::string, std::string>& kv_options)
      : hint_(hint)
      , query_hint_(query_hint)
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

  bool isQueryHint() const { return query_hint_; }

  bool hasOptions() const { return is_marker_; }

  bool hasKvOptions() const { return has_kv_type_options_; }

 private:
  QueryHint hint_;
  // Set true if this hint affects globally
  // Otherwise it just affects the node which this hint is included (aka table hint)
  bool query_hint_;
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
  RegisteredQueryHint(bool enable_columnar_output)
      : cpu_mode(false)
      , columnar_output(enable_columnar_output)
      , rowwise_output(!enable_columnar_output)
      , registered_hint(QueryHint::kHintCount, false) {}

  RegisteredQueryHint& operator=(const RegisteredQueryHint& other) {
    cpu_mode = other.cpu_mode;
    columnar_output = other.columnar_output;
    rowwise_output = other.rowwise_output;
    registered_hint = other.registered_hint;
    return *this;
  }

  RegisteredQueryHint(const RegisteredQueryHint& other) {
    cpu_mode = other.cpu_mode;
    columnar_output = other.columnar_output;
    rowwise_output = other.rowwise_output;
    registered_hint = other.registered_hint;
  }

  // general query execution
  bool cpu_mode;
  bool columnar_output;
  bool rowwise_output;

  std::vector<bool> registered_hint;

  static RegisteredQueryHint fromConfig(const Config& config) {
    return RegisteredQueryHint(config.rs.enable_columnar_output);
  }

 public:
  static QueryHint translateQueryHint(const std::string& hint_name) {
    const auto lowered_hint_name = boost::algorithm::to_lower_copy(hint_name);
    auto it = SupportedQueryHints.find(hint_name);
    if (it != SupportedQueryHints.end()) {
      return it->second;
    }
    return QueryHint::kInvalidHint;
  }

  bool isAnyQueryHintDelivered() const {
    for (auto flag : registered_hint) {
      if (flag) {
        return true;
      }
    }
    return false;
  }

  void registerHint(const QueryHint hint) {
    const auto hint_class = static_cast<int>(hint);
    if (hint_class >= 0 && hint_class < QueryHint::kHintCount) {
      registered_hint[hint_class] = true;
    }
  }

  const bool isHintRegistered(const QueryHint hint) const {
    const auto hint_class = static_cast<int>(hint);
    if (hint_class >= 0 && hint_class < QueryHint::kHintCount) {
      return registered_hint[hint_class];
    }
    return false;
  }
};

// a map from hint_name to its detailed info
using Hints = std::unordered_map<QueryHint, ExplainedQueryHint>;

#endif  // OMNISCI_QUERYHINT_H
