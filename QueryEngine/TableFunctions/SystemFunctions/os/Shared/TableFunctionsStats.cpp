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

#ifndef __CUDACC__

#include "TableFunctionsStats.hpp"

template <typename T>
NEVER_INLINE HOST ColumnStats<T> get_column_stats(
    const T* data,
    const int64_t num_rows,
    const StatsRequestPredicate& predicate) {
  // const int64_t num_rows = col.size();
  const size_t max_thread_count = std::thread::hardware_concurrency();
  const size_t max_inputs_per_thread = 20000;
  const size_t num_threads = std::min(
      max_thread_count, ((num_rows + max_inputs_per_thread - 1) / max_inputs_per_thread));

  std::vector<T> local_col_mins(num_threads, std::numeric_limits<T>::max());
  std::vector<T> local_col_maxes(num_threads, std::numeric_limits<T>::lowest());
  std::vector<double> local_col_sums(num_threads, 0.);
  std::vector<int64_t> local_col_non_null_or_filtered_counts(num_threads, 0L);
  tbb::task_arena limited_arena(num_threads);
  limited_arena.execute([&] {
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_rows),
                      [&](const tbb::blocked_range<int64_t>& r) {
                        const int64_t start_idx = r.begin();
                        const int64_t end_idx = r.end();
                        T local_col_min = std::numeric_limits<T>::max();
                        T local_col_max = std::numeric_limits<T>::lowest();
                        double local_col_sum = 0.;
                        int64_t local_col_non_null_or_filtered_count = 0;
                        for (int64_t r = start_idx; r < end_idx; ++r) {
                          const T val = data[r];
                          if (val == inline_null_value<T>()) {
                            continue;
                          }
                          if (!predicate(val)) {
                            continue;
                          }
                          if (val < local_col_min) {
                            local_col_min = val;
                          }
                          if (val > local_col_max) {
                            local_col_max = val;
                          }
                          local_col_sum += data[r];
                          local_col_non_null_or_filtered_count++;
                        }
                        size_t thread_idx = tbb::this_task_arena::current_thread_index();
                        if (local_col_min < local_col_mins[thread_idx]) {
                          local_col_mins[thread_idx] = local_col_min;
                        }
                        if (local_col_max > local_col_maxes[thread_idx]) {
                          local_col_maxes[thread_idx] = local_col_max;
                        }
                        local_col_sums[thread_idx] += local_col_sum;
                        local_col_non_null_or_filtered_counts[thread_idx] +=
                            local_col_non_null_or_filtered_count;
                      });
  });

  ColumnStats<T> column_stats;
  // Use separate double col_sum instead of column_stats.sum to avoid fp imprecision if T
  // is float
  double col_sum = 0.0;
  column_stats.total_count = num_rows;

  for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    if (local_col_mins[thread_idx] < column_stats.min) {
      column_stats.min = local_col_mins[thread_idx];
    }
    if (local_col_maxes[thread_idx] > column_stats.max) {
      column_stats.max = local_col_maxes[thread_idx];
    }
    col_sum += local_col_sums[thread_idx];
    column_stats.non_null_or_filtered_count +=
        local_col_non_null_or_filtered_counts[thread_idx];
  }

  if (column_stats.non_null_or_filtered_count > 0) {
    column_stats.sum = col_sum;
    column_stats.mean = col_sum / column_stats.non_null_or_filtered_count;
  }
  return column_stats;
}

template NEVER_INLINE HOST ColumnStats<int8_t> get_column_stats(
    const int8_t* data,
    const int64_t num_rows,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<int16_t> get_column_stats(
    const int16_t* data,
    const int64_t num_rows,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<int32_t> get_column_stats(
    const int32_t* data,
    const int64_t num_rows,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<int64_t> get_column_stats(
    const int64_t* data,
    const int64_t num_rows,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<float> get_column_stats(
    const float* data,
    const int64_t num_rows,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<double> get_column_stats(
    const double* data,
    const int64_t num_rows,
    const StatsRequestPredicate& predicate);

template <typename T>
NEVER_INLINE HOST ColumnStats<T> get_column_stats(
    const Column<T>& col,
    const StatsRequestPredicate& predicate) {
  return get_column_stats(col.getPtr(), col.size(), predicate);
}

template NEVER_INLINE HOST ColumnStats<int8_t> get_column_stats(
    const Column<int8_t>& col,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<int16_t> get_column_stats(
    const Column<int16_t>& col,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<int32_t> get_column_stats(
    const Column<int32_t>& col,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<int64_t> get_column_stats(
    const Column<int64_t>& col,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<float> get_column_stats(
    const Column<float>& col,
    const StatsRequestPredicate& predicate);
template NEVER_INLINE HOST ColumnStats<double> get_column_stats(
    const Column<double>& col,
    const StatsRequestPredicate& predicate);

StatsRequestAggType convert_string_to_stats_request_agg_type(const std::string& str) {
  if (str == "COUNT") {
    return StatsRequestAggType::COUNT;
  }
  if (str == "MIN") {
    return StatsRequestAggType::MIN;
  }
  if (str == "MAX") {
    return StatsRequestAggType::MAX;
  }
  if (str == "SUM") {
    return StatsRequestAggType::SUM;
  }
  if (str == "AVG") {
    return StatsRequestAggType::AVG;
  }
  throw std::runtime_error("Invalid StatsRequestAggType: " + str);
}

StatsRequestPredicateOp convert_string_to_stats_request_predicate_op(
    const std::string& str) {
  if (str == "NONE") {
    return StatsRequestPredicateOp::NONE;
  }
  if (str == "LT" || str == "<") {
    return StatsRequestPredicateOp::LT;
  }
  if (str == "GT" || str == ">") {
    return StatsRequestPredicateOp::GT;
  }
  throw std::runtime_error("Invalid StatsRequestPredicateOp: " + str);
}

std::string replace_substrings(const std::string& str,
                               const std::string& pattern_str,
                               const std::string& replacement_str) {
  std::string replaced_str(str);

  size_t search_start_index = 0;
  const auto pattern_str_len = pattern_str.size();
  const auto replacement_str_len = replacement_str.size();

  while (true) {
    search_start_index = replaced_str.find(pattern_str, search_start_index);
    if (search_start_index == std::string::npos) {
      break;
    }
    replaced_str.replace(search_start_index, pattern_str_len, replacement_str);
    search_start_index += replacement_str_len;
  }
  return replaced_str;
}

std::vector<StatsRequest> parse_stats_requests_json(
    const std::string& stats_requests_json_str,
    const int64_t num_attrs) {
  std::vector<StatsRequest> stats_requests;
  rapidjson::Document doc;

  // remove double double quotes our parser introduces
  const auto fixed_stats_requests_json_str =
      replace_substrings(stats_requests_json_str, "\"\"", "\"");

  if (doc.Parse(fixed_stats_requests_json_str.c_str()).HasParseError()) {
    // Not valid JSON
    std::cout << "DEBUG: Failed JSON: " << fixed_stats_requests_json_str << std::endl;
    throw std::runtime_error("Could not parse Stats Requests JSON.");
  }
  // Todo (todd): Enforce Schema
  if (!doc.IsArray()) {
    throw std::runtime_error("Stats Request JSON did not contain valid root Array.");
  }
  const std::vector<std::string> required_keys = {
      "name", "attr_id", "agg_type", "filter_type"};

  for (const auto& stat_request_obj : doc.GetArray()) {
    for (const auto& required_key : required_keys) {
      if (!stat_request_obj.HasMember(required_key)) {
        throw std::runtime_error("Stats Request JSON missing key " + required_key + ".");
      }
      if (required_key == "attr_id") {
        if (!stat_request_obj[required_key].IsUint()) {
          throw std::runtime_error(required_key + " must be int type");
        }
      } else {
        if (!stat_request_obj[required_key].IsString()) {
          throw std::runtime_error(required_key + " must be string type");
        }
      }
    }
    StatsRequest stats_request;
    stats_request.name = stat_request_obj["name"].GetString();
    stats_request.attr_id = stat_request_obj["attr_id"].GetInt() - 1;
    if (stats_request.attr_id < 0 || stats_request.attr_id >= num_attrs) {
      throw std::runtime_error("Invalid attr_id: " +
                               std::to_string(stats_request.attr_id));
    }

    std::string agg_type_str = stat_request_obj["agg_type"].GetString();
    std::transform(
        agg_type_str.begin(), agg_type_str.end(), agg_type_str.begin(), ::toupper);
    stats_request.agg_type = convert_string_to_stats_request_agg_type(agg_type_str);

    std::string filter_type_str = stat_request_obj["filter_type"].GetString();
    std::transform(filter_type_str.begin(),
                   filter_type_str.end(),
                   filter_type_str.begin(),
                   ::toupper);
    stats_request.filter_type =
        convert_string_to_stats_request_predicate_op(filter_type_str);
    if (stats_request.filter_type != StatsRequestPredicateOp::NONE) {
      if (!stat_request_obj.HasMember("filter_val")) {
        throw std::runtime_error("Stats Request JSON missing expected filter_val");
      }
      if (!stat_request_obj["filter_val"].IsNumber()) {
        throw std::runtime_error("Stats Request JSON filter_val should be numeric.");
      }
      stats_request.filter_val = stat_request_obj["filter_val"].GetDouble();
    }
    stats_requests.emplace_back(stats_request);
  }
  return stats_requests;
}

std::vector<std::pair<const char*, double>> get_stats_key_value_pairs(
    const std::vector<StatsRequest>& stats_requests) {
  std::vector<std::pair<const char*, double>> stats_key_value_pairs;
  for (const auto& stats_request : stats_requests) {
    stats_key_value_pairs.emplace_back(
        std::make_pair(stats_request.name.c_str(), stats_request.result));
  }
  return stats_key_value_pairs;
}

#endif  // __CUDACC__
