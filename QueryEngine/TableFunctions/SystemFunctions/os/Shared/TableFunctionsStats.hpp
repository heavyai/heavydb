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

#pragma once

#ifndef __CUDACC__

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <rapidjson/document.h>

#include "QueryEngine/heavydbTypes.h"

enum class StatsRequestPredicateOp { NONE, LT, GT };

struct StatsRequestPredicate {
  StatsRequestPredicate()
      : predicate_op(StatsRequestPredicateOp::NONE)
      , filter_val(0.)
      , is_gt(false)
      , is_no_op(true) {}
  StatsRequestPredicate(const StatsRequestPredicateOp predicate_op,
                        const double filter_val)
      : predicate_op(predicate_op)
      , filter_val(filter_val)
      , is_gt(predicate_op == StatsRequestPredicateOp::GT)
      , is_no_op(predicate_op == StatsRequestPredicateOp::NONE) {}

  StatsRequestPredicate(const StatsRequestPredicate& other)
      : predicate_op(other.predicate_op)
      , filter_val(other.filter_val)
      , is_gt(other.is_gt)
      , is_no_op(other.is_no_op) {}

  std::string to_string() const {
    std::string str;
    switch (predicate_op) {
      case StatsRequestPredicateOp::NONE: {
        str += "NONE";
        break;
      }
      case StatsRequestPredicateOp::LT: {
        str += "LT";
        break;
      }
      case StatsRequestPredicateOp::GT: {
        str += "GT";
        break;
      }
    }
    str += "|" + std::to_string(filter_val);
    return str;
  }

  template <typename T>
  inline bool operator()(const T val) const {
    return is_no_op || (is_gt && (val >= filter_val));
  }

  bool operator==(StatsRequestPredicate const& rhs) const {
    return predicate_op == rhs.predicate_op && filter_val == rhs.filter_val;
  }

  StatsRequestPredicateOp predicate_op;
  double filter_val;
  bool is_gt;
  bool is_no_op;
};

template <typename T>
struct ColumnStats {
  int64_t total_count;
  int64_t non_null_or_filtered_count;
  T min;
  T max;
  T sum;
  double mean;

  ColumnStats()
      : total_count(0)
      , non_null_or_filtered_count(0)
      , min(std::numeric_limits<T>::max())
      , max(std::numeric_limits<T>::lowest())
      , sum(0.0)
      , mean(0.0) {}
};

enum class StatsRequestAggType {
  COUNT,
  MIN,
  MAX,
  SUM,
  AVG,
};

struct StatsRequest {
  std::string name;
  int32_t attr_id;
  StatsRequestAggType agg_type;
  StatsRequestPredicateOp filter_type;
  double filter_val;
  double result;
};

std::vector<StatsRequest> parse_stats_requests_json(
    const std::string& stats_requests_json_str,
    const int64_t num_attrs);

template <typename TA>
ColumnStats<TA> get_column_stats(
    const ColumnList<TA>& attrs,
    StatsRequest& stats_request,
    std::unordered_map<std::string, ColumnStats<TA>>& stats_map) {
  StatsRequestPredicate predicate(stats_request.filter_type, stats_request.filter_val);
  const std::string request_str_key =
      std::to_string(stats_request.attr_id) + "||" + predicate.to_string();
  auto stats_map_itr = stats_map.find(request_str_key);
  if (stats_map_itr != stats_map.end()) {
    return stats_map_itr->second;
  }
  const auto column_stats = get_column_stats(attrs[stats_request.attr_id], predicate);
  stats_map[request_str_key] = column_stats;
  return column_stats;
}

template <typename TA>
void compute_stats_requests(const ColumnList<TA>& attrs,
                            std::vector<StatsRequest>& stats_requests) {
  std::unordered_map<std::string, ColumnStats<TA>> stats_map;

  for (auto& stats_request : stats_requests) {
    const auto column_stats = get_column_stats(attrs, stats_request, stats_map);
    switch (stats_request.agg_type) {
      case StatsRequestAggType::COUNT: {
        stats_request.result = column_stats.non_null_or_filtered_count;
        break;
      }
      case StatsRequestAggType::MIN: {
        stats_request.result = column_stats.min;
        break;
      }
      case StatsRequestAggType::MAX: {
        stats_request.result = column_stats.max;
        break;
      }
      case StatsRequestAggType::SUM: {
        stats_request.result = column_stats.sum;
        break;
      }
      case StatsRequestAggType::AVG: {
        stats_request.result = column_stats.mean;
        break;
      }
    }
  }
}

template <typename TA>
void populate_output_stats_cols(Column<TextEncodingDict>& stat_names,
                                Column<TA>& stat_vals,
                                const std::vector<StatsRequest>& stats_requests) {
  const int64_t num_requests = static_cast<int64_t>(stats_requests.size());
  for (int64_t request_id = 0; request_id < num_requests; ++request_id) {
    stat_names[request_id] =
        stat_names.string_dict_proxy_->getOrAddTransient(stats_requests[request_id].name);
    stat_vals[request_id] = stats_requests[request_id].result;
  }
}

template <typename T>
NEVER_INLINE HOST ColumnStats<T> get_column_stats(
    const T* data,
    const int64_t num_rows,
    const StatsRequestPredicate& predicate = StatsRequestPredicate());

template <typename T>
NEVER_INLINE HOST ColumnStats<T> get_column_stats(
    const Column<T>& col,
    const StatsRequestPredicate& predicate = StatsRequestPredicate());

#endif  // __CUDACC__
