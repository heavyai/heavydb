/*
 * Copyright 2022 Intel Corporation.
 * Copyright 2021 OmniSci, Inc.
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

#include "QueryEngine/MemoryLayoutBuilder.h"

#include "QueryEngine/CardinalityEstimator.h"
#include "QueryEngine/ColRangeInfo.h"
#include "QueryEngine/OutputBufferInitialization.h"

MemoryLayoutBuilder::MemoryLayoutBuilder(const RelAlgExecutionUnit& ra_exe_unit)
    : ra_exe_unit_(ra_exe_unit) {
  for (const auto& groupby_expr : ra_exe_unit_.groupby_exprs) {
    if (!groupby_expr) {
      continue;
    }
    const auto& groupby_ti = groupby_expr->get_type_info();
    if (groupby_ti.is_bytes()) {
      throw std::runtime_error(
          "Cannot group by string columns which are not dictionary encoded.");
    }
    if (groupby_ti.is_buffer()) {
      throw std::runtime_error("Group by buffer not supported");
    }
  }
}

namespace {

bool has_count_distinct(const RelAlgExecutionUnit& ra_exe_unit, bool bigint_count) {
  for (const auto& target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = get_target_info(target_expr, bigint_count);
    if (agg_info.is_agg && is_distinct_target(agg_info)) {
      return true;
    }
  }
  return false;
}

bool is_column_range_too_big_for_perfect_hash(const ColRangeInfo& col_range_info,
                                              const int64_t max_entry_count) {
  try {
    return static_cast<int64_t>(checked_int64_t(col_range_info.max) -
                                checked_int64_t(col_range_info.min)) >= max_entry_count;
  } catch (...) {
    return true;
  }
}

bool cardinality_estimate_less_than_column_range(const int64_t cardinality_estimate,
                                                 const ColRangeInfo& col_range_info) {
  try {
    // the cardinality estimate is the size of the baseline hash table. further penalize
    // the baseline hash table by a factor of 2x due to overhead in computing baseline
    // hash. This has the overall effect of penalizing baseline hash over perfect hash by
    // 4x; i.e. if the cardinality of the filtered data is less than 25% of the entry
    // count of the column, we use baseline hash on the filtered set
    return checked_int64_t(cardinality_estimate) * 2 <
           static_cast<int64_t>(checked_int64_t(col_range_info.max) -
                                checked_int64_t(col_range_info.min));
  } catch (...) {
    return false;
  }
}

bool expr_is_rowid(const Analyzer::Expr* expr) {
  const auto col = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (!col) {
    return false;
  }
  return col->is_virtual();
}

ColRangeInfo get_col_range_info(const RelAlgExecutionUnit& ra_exe_unit,
                                const std::vector<InputTableInfo>& query_infos,
                                std::optional<int64_t> group_cardinality_estimation,
                                Executor* executor,
                                const ExecutorDeviceType device_type) {
  const Config& config = executor->getConfig();
  // Use baseline layout more eagerly on the GPU if the query uses count distinct,
  // because our HyperLogLog implementation is 4x less memory efficient on GPU.
  // Technically, this only applies to APPROX_COUNT_DISTINCT, but in practice we
  // can expect this to be true anyway for grouped queries since the precise version
  // uses significantly more memory.
  int64_t baseline_threshold = config.exec.group_by.baseline_threshold;
  if (has_count_distinct(ra_exe_unit, config.exec.group_by.bigint_count) &&
      device_type == ExecutorDeviceType::GPU) {
    baseline_threshold = baseline_threshold / 4;
  }
  if (ra_exe_unit.groupby_exprs.size() != 1) {
    try {
      checked_int64_t cardinality{1};
      bool has_nulls{false};
      for (const auto& groupby_expr : ra_exe_unit.groupby_exprs) {
        auto col_range_info =
            get_expr_range_info(ra_exe_unit, query_infos, groupby_expr.get(), executor);
        if (col_range_info.hash_type_ != QueryDescriptionType::GroupByPerfectHash) {
          // going through baseline hash if a non-integer type is encountered
          return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
        }
        auto crt_col_cardinality = col_range_info.getBucketedCardinality();
        CHECK_GE(crt_col_cardinality, 0);
        cardinality *= crt_col_cardinality;
        if (col_range_info.has_nulls) {
          has_nulls = true;
        }
      }
      // For zero or high cardinalities, use baseline layout.
      if (!cardinality || cardinality > baseline_threshold) {
        return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
      }
      return {QueryDescriptionType::GroupByPerfectHash,
              0,
              int64_t(cardinality),
              0,
              has_nulls};
    } catch (...) {  // overflow when computing cardinality
      return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
    }
  }
  // For single column groupby on high timestamps, force baseline hash due to wide ranges
  // we are likely to encounter when applying quals to the expression range
  // TODO: consider allowing TIMESTAMP(9) (nanoseconds) with quals to use perfect hash if
  // the range is small enough
  if (ra_exe_unit.groupby_exprs.front() &&
      ra_exe_unit.groupby_exprs.front()->get_type_info().is_high_precision_timestamp() &&
      ra_exe_unit.simple_quals.size() > 0) {
    return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
  }
  const auto col_range_info = get_expr_range_info(
      ra_exe_unit, query_infos, ra_exe_unit.groupby_exprs.front().get(), executor);
  if (!ra_exe_unit.groupby_exprs.front()) {
    return col_range_info;
  }
  static const int64_t MAX_BUFFER_SIZE = 1 << 30;
  const int64_t col_count =
      ra_exe_unit.groupby_exprs.size() + ra_exe_unit.target_exprs.size();
  int64_t max_entry_count = MAX_BUFFER_SIZE / (col_count * sizeof(int64_t));
  if (has_count_distinct(ra_exe_unit, config.exec.group_by.bigint_count)) {
    max_entry_count = std::min(max_entry_count, baseline_threshold);
  }
  const auto& groupby_expr_ti = ra_exe_unit.groupby_exprs.front()->get_type_info();
  if (groupby_expr_ti.is_string() && !col_range_info.bucket) {
    CHECK(groupby_expr_ti.get_compression() == kENCODING_DICT);

    const bool has_filters =
        !ra_exe_unit.quals.empty() || !ra_exe_unit.simple_quals.empty();
    if (has_filters &&
        is_column_range_too_big_for_perfect_hash(col_range_info, max_entry_count)) {
      // if filters are present, we can use the filter to narrow the cardinality of the
      // group by in the case of ranges too big for perfect hash. Otherwise, we are better
      // off attempting perfect hash (since we know the range will be made of
      // monotonically increasing numbers from min to max for dictionary encoded strings)
      // and failing later due to excessive memory use.
      // Check the conditions where baseline hash can provide a performance increase and
      // return baseline hash (potentially forcing an estimator query) as the range type.
      // Otherwise, return col_range_info which will likely be perfect hash, though could
      // be baseline from a previous call of this function prior to the estimator query.
      if (!ra_exe_unit.sort_info.order_entries.empty()) {
        // TODO(adb): allow some sorts to pass through this block by centralizing sort
        // algorithm decision making
        if (has_count_distinct(ra_exe_unit, config.exec.group_by.bigint_count) &&
            is_column_range_too_big_for_perfect_hash(col_range_info, max_entry_count)) {
          // always use baseline hash for column range too big for perfect hash with count
          // distinct descriptors. We will need 8GB of CPU memory minimum for the perfect
          // hash group by in this case.
          return {QueryDescriptionType::GroupByBaselineHash,
                  col_range_info.min,
                  col_range_info.max,
                  0,
                  col_range_info.has_nulls};
        } else {
          // use original col range for sort
          return col_range_info;
        }
      }
      // if filters are present and the filtered range is less than the cardinality of
      // the column, consider baseline hash
      if (!group_cardinality_estimation ||
          cardinality_estimate_less_than_column_range(*group_cardinality_estimation,
                                                      col_range_info)) {
        return {QueryDescriptionType::GroupByBaselineHash,
                col_range_info.min,
                col_range_info.max,
                0,
                col_range_info.has_nulls};
      }
    }
  } else if ((!expr_is_rowid(ra_exe_unit.groupby_exprs.front().get())) &&
             is_column_range_too_big_for_perfect_hash(col_range_info, max_entry_count) &&
             !col_range_info.bucket) {
    return {QueryDescriptionType::GroupByBaselineHash,
            col_range_info.min,
            col_range_info.max,
            0,
            col_range_info.has_nulls};
  }
  return col_range_info;
}

/**
 * This function goes through all target expressions and answers two questions:
 * 1. Is it possible to have keyless hash?
 * 2. If yes to 1, then what aggregate expression should be considered to represent the
 * key's presence, if needed (e.g., in detecting empty entries in the result set).
 *
 * NOTE: Keyless hash is only valid with single-column group by at the moment.
 *
 */
KeylessInfo get_keyless_info(const RelAlgExecutionUnit& ra_exe_unit,
                             const std::vector<InputTableInfo>& query_infos,
                             const bool is_group_by,
                             Executor* executor) {
  bool keyless{true}, found{false};
  int32_t num_agg_expr{0};
  int32_t index{0};
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info =
        get_target_info(target_expr, executor->getConfig().exec.group_by.bigint_count);
    const auto chosen_type = get_compact_type(agg_info);
    if (agg_info.is_agg) {
      num_agg_expr++;
    }
    if (!found && agg_info.is_agg && !is_distinct_target(agg_info)) {
      auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
      CHECK(agg_expr);
      const auto arg_expr = agg_arg(target_expr);
      const bool float_argument_input = takes_float_argument(agg_info);
      switch (agg_info.agg_kind) {
        case kAVG:
          ++index;
          if (arg_expr && !arg_expr->get_type_info().get_notnull()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos, executor);
            if (expr_range_info.getType() == ExpressionRangeType::Invalid ||
                expr_range_info.hasNulls()) {
              break;
            }
          }
          found = true;
          break;
        case kCOUNT:
          if (arg_expr && !arg_expr->get_type_info().get_notnull()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos, executor);
            if (expr_range_info.getType() == ExpressionRangeType::Invalid ||
                expr_range_info.hasNulls()) {
              break;
            }
          }
          found = true;
          break;
        case kSUM: {
          auto arg_ti = arg_expr->get_type_info();
          if (constrained_not_null(arg_expr, ra_exe_unit.quals)) {
            arg_ti.set_notnull(true);
          }
          if (!arg_ti.get_notnull()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos, executor);
            if (expr_range_info.getType() != ExpressionRangeType::Invalid &&
                !expr_range_info.hasNulls()) {
              found = true;
            }
          } else {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos, executor);
            switch (expr_range_info.getType()) {
              case ExpressionRangeType::Float:
              case ExpressionRangeType::Double:
                if (expr_range_info.getFpMax() < 0 || expr_range_info.getFpMin() > 0) {
                  found = true;
                }
                break;
              case ExpressionRangeType::Integer:
                if (expr_range_info.getIntMax() < 0 || expr_range_info.getIntMin() > 0) {
                  found = true;
                }
                break;
              default:
                break;
            }
          }
          break;
        }
        case kMIN: {
          CHECK(agg_expr && agg_expr->get_arg());
          const auto& arg_ti = agg_expr->get_arg()->get_type_info();
          if (arg_ti.is_string() || arg_ti.is_buffer()) {
            break;
          }
          auto expr_range_info =
              getExpressionRange(agg_expr->get_arg(), query_infos, executor);
          auto init_max = get_agg_initial_val(agg_info.agg_kind,
                                              chosen_type,
                                              is_group_by || float_argument_input,
                                              float_argument_input ? sizeof(float) : 8);
          switch (expr_range_info.getType()) {
            case ExpressionRangeType::Float:
            case ExpressionRangeType::Double: {
              auto double_max =
                  *reinterpret_cast<const double*>(may_alias_ptr(&init_max));
              if (expr_range_info.getFpMax() < double_max) {
                found = true;
              }
              break;
            }
            case ExpressionRangeType::Integer:
              if (expr_range_info.getIntMax() < init_max) {
                found = true;
              }
              break;
            default:
              break;
          }
          break;
        }
        case kMAX: {
          CHECK(agg_expr && agg_expr->get_arg());
          const auto& arg_ti = agg_expr->get_arg()->get_type_info();
          if (arg_ti.is_string() || arg_ti.is_buffer()) {
            break;
          }
          auto expr_range_info =
              getExpressionRange(agg_expr->get_arg(), query_infos, executor);
          // NULL sentinel and init value for kMAX are identical, which results in
          // ambiguity in detecting empty keys in presence of nulls.
          if (expr_range_info.getType() == ExpressionRangeType::Invalid ||
              expr_range_info.hasNulls()) {
            break;
          }
          auto init_min = get_agg_initial_val(agg_info.agg_kind,
                                              chosen_type,
                                              is_group_by || float_argument_input,
                                              float_argument_input ? sizeof(float) : 8);
          switch (expr_range_info.getType()) {
            case ExpressionRangeType::Float:
            case ExpressionRangeType::Double: {
              auto double_min =
                  *reinterpret_cast<const double*>(may_alias_ptr(&init_min));
              if (expr_range_info.getFpMin() > double_min) {
                found = true;
              }
              break;
            }
            case ExpressionRangeType::Integer:
              if (expr_range_info.getIntMin() > init_min) {
                found = true;
              }
              break;
            default:
              break;
          }
          break;
        }
        default:
          keyless = false;
          break;
      }
    }
    if (!keyless) {
      break;
    }
    if (!found) {
      ++index;
    }
  }

  // shouldn't use keyless for projection only
  return {
      keyless && found,
      index,
  };
}

CountDistinctDescriptors init_count_distinct_descriptors(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const ExecutorDeviceType device_type,
    Executor* executor,
    size_t group_by_slots_count,
    QueryDescriptionType group_by_hash_type) {
  CountDistinctDescriptors count_distinct_descriptors;
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    auto agg_info =
        get_target_info(target_expr, executor->getConfig().exec.group_by.bigint_count);
    if (is_distinct_target(agg_info)) {
      CHECK(agg_info.is_agg);
      CHECK(agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT);
      const auto agg_expr = static_cast<const Analyzer::AggExpr*>(target_expr);
      const auto& arg_ti = agg_expr->get_arg()->get_type_info();
      if (arg_ti.is_bytes()) {
        throw std::runtime_error(
            "Strings must be dictionary-encoded for COUNT(DISTINCT).");
      }
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT && arg_ti.is_buffer()) {
        throw std::runtime_error("APPROX_COUNT_DISTINCT on arrays not supported yet");
      }
      ColRangeInfo no_range_info{QueryDescriptionType::Projection, 0, 0, 0, false};
      auto arg_range_info =
          arg_ti.is_fp() ? no_range_info
                         : get_expr_range_info(
                               ra_exe_unit, query_infos, agg_expr->get_arg(), executor);
      CountDistinctImplType count_distinct_impl_type{CountDistinctImplType::HashSet};
      int64_t bitmap_sz_bits{0};
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
        const auto error_rate = agg_expr->get_arg1();
        if (error_rate) {
          CHECK(error_rate->get_type_info().get_type() == kINT);
          CHECK_GE(error_rate->get_constval().intval, 1);
          bitmap_sz_bits = hll_size_for_rate(error_rate->get_constval().smallintval);
        } else {
          bitmap_sz_bits = executor->getConfig().exec.group_by.hll_precision_bits;
        }
      }
      if (arg_range_info.isEmpty()) {
        count_distinct_descriptors.emplace_back(
            CountDistinctDescriptor{CountDistinctImplType::Bitmap,
                                    0,
                                    64,
                                    agg_info.agg_kind == kAPPROX_COUNT_DISTINCT,
                                    device_type,
                                    1});
        continue;
      }
      if (arg_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash &&
          !arg_ti.is_buffer()) {  // TODO(alex): allow bitmap
                                  // implementation for arrays
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
        if (agg_info.agg_kind == kCOUNT) {
          bitmap_sz_bits = arg_range_info.max - arg_range_info.min + 1;

          if (group_by_hash_type == QueryDescriptionType::GroupByBaselineHash) {
            const int64_t MAX_TOTAL_BITMAPS_BITS = 8 * 8 * 1000 * 1000 * 1000LL;  // 8GB
            int64_t total_bitmaps_size = bitmap_sz_bits * group_by_slots_count / 2;

            if (total_bitmaps_size <= 0 || total_bitmaps_size > MAX_TOTAL_BITMAPS_BITS) {
              count_distinct_impl_type = CountDistinctImplType::HashSet;
            }
          } else {
            const int64_t MAX_BITMAP_BITS{8 * 1000 * 1000 * 1000LL};
            if (bitmap_sz_bits <= 0 || bitmap_sz_bits > MAX_BITMAP_BITS) {
              count_distinct_impl_type = CountDistinctImplType::HashSet;
            }
          }
        }
      }
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT &&
          count_distinct_impl_type == CountDistinctImplType::HashSet &&
          !arg_ti.is_array()) {
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
      }

      if (executor->getConfig().exec.watchdog.enable && !(arg_range_info.isEmpty()) &&
          count_distinct_impl_type == CountDistinctImplType::HashSet) {
        throw WatchdogException("Cannot use a fast path for COUNT distinct");
      }
      const auto sub_bitmap_count =
          get_count_distinct_sub_bitmap_count(bitmap_sz_bits, ra_exe_unit, device_type);
      count_distinct_descriptors.emplace_back(
          CountDistinctDescriptor{count_distinct_impl_type,
                                  arg_range_info.min,
                                  bitmap_sz_bits,
                                  agg_info.agg_kind == kAPPROX_COUNT_DISTINCT,
                                  device_type,
                                  sub_bitmap_count});
    } else {
      count_distinct_descriptors.emplace_back(CountDistinctDescriptor{
          CountDistinctImplType::Invalid, 0, 0, false, device_type, 0});
    }
  }
  return count_distinct_descriptors;
}

std::unique_ptr<QueryMemoryDescriptor> build_query_memory_descriptor(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const bool allow_multifrag,
    const size_t max_groups_buffer_entry_count,
    const int8_t crt_min_byte_width,
    const bool sort_on_gpu_hint,
    const bool must_use_baseline_sort,
    const bool output_columnar_hint,
    std::optional<int64_t> group_cardinality_estimation,
    Executor* executor,
    const ExecutorDeviceType device_type) {
  const bool is_group_by{!ra_exe_unit.groupby_exprs.empty()};

  auto col_range_info = get_col_range_info(
      ra_exe_unit, query_infos, group_cardinality_estimation, executor, device_type);

  const auto count_distinct_descriptors =
      init_count_distinct_descriptors(ra_exe_unit,
                                      query_infos,
                                      device_type,
                                      executor,
                                      max_groups_buffer_entry_count,
                                      col_range_info.hash_type_);

  // Non-grouped aggregates do not support accessing aggregated ranges
  // Keyless hash is currently only supported with single-column perfect hash
  const auto keyless_info =
      !(is_group_by &&
        col_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash)
          ? KeylessInfo{false, -1}
          : get_keyless_info(ra_exe_unit, query_infos, is_group_by, executor);

  if (executor->getConfig().exec.watchdog.enable &&
      ((col_range_info.hash_type_ == QueryDescriptionType::GroupByBaselineHash &&
        max_groups_buffer_entry_count >
            executor->getConfig().exec.watchdog.baseline_max_groups) ||
       (col_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash &&
        ra_exe_unit.groupby_exprs.size() == 1 &&
        (col_range_info.max - col_range_info.min) /
                std::max(col_range_info.bucket, int64_t(1)) >
            130000000))) {
    throw WatchdogException("Query would use too much memory");
  }
  try {
    return QueryMemoryDescriptor::init(executor,
                                       ra_exe_unit,
                                       query_infos,
                                       col_range_info,
                                       keyless_info,
                                       allow_multifrag,
                                       device_type,
                                       crt_min_byte_width,
                                       sort_on_gpu_hint,
                                       max_groups_buffer_entry_count,
                                       count_distinct_descriptors,
                                       must_use_baseline_sort,
                                       output_columnar_hint,
                                       /*streaming_top_n_hint=*/true);
  } catch (const StreamingTopNOOM& e) {
    LOG(WARNING) << e.what() << " Disabling Streaming Top N.";
    return QueryMemoryDescriptor::init(executor,
                                       ra_exe_unit,
                                       query_infos,
                                       col_range_info,
                                       keyless_info,
                                       allow_multifrag,
                                       device_type,
                                       crt_min_byte_width,
                                       sort_on_gpu_hint,
                                       max_groups_buffer_entry_count,
                                       count_distinct_descriptors,
                                       must_use_baseline_sort,
                                       output_columnar_hint,
                                       /*streaming_top_n_hint=*/false);
  }
}

bool gpu_can_handle_order_entries(const RelAlgExecutionUnit& ra_exe_unit,
                                  const std::vector<InputTableInfo>& query_infos,
                                  const std::list<Analyzer::OrderEntry>& order_entries,
                                  Executor* executor) {
  if (order_entries.size() > 1) {  // TODO(alex): lift this restriction
    return false;
  }
  for (const auto& order_entry : order_entries) {
    CHECK_GE(order_entry.tle_no, 1);
    CHECK_LE(static_cast<size_t>(order_entry.tle_no), ra_exe_unit.target_exprs.size());
    const auto target_expr = ra_exe_unit.target_exprs[order_entry.tle_no - 1];
    if (!dynamic_cast<Analyzer::AggExpr*>(target_expr)) {
      return false;
    }
    // TODO(alex): relax the restrictions
    auto agg_expr = static_cast<Analyzer::AggExpr*>(target_expr);
    if (agg_expr->get_is_distinct() || agg_expr->get_aggtype() == kAVG ||
        agg_expr->get_aggtype() == kMIN || agg_expr->get_aggtype() == kMAX ||
        agg_expr->get_aggtype() == kAPPROX_COUNT_DISTINCT) {
      return false;
    }
    if (agg_expr->get_arg()) {
      const auto& arg_ti = agg_expr->get_arg()->get_type_info();
      if (arg_ti.is_fp()) {
        return false;
      }
      auto expr_range_info =
          get_expr_range_info(ra_exe_unit, query_infos, agg_expr->get_arg(), executor);
      // TOD(adb): QMD not actually initialized here?
      if ((!(expr_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash &&
             /* query_mem_desc.getGroupbyColCount() == 1 */ false) ||
           expr_range_info.has_nulls) &&
          order_entry.is_desc == order_entry.nulls_first) {
        return false;
      }
    }
    const auto& target_ti = target_expr->get_type_info();
    CHECK(!target_ti.is_buffer());
    if (!target_ti.is_integer()) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::unique_ptr<QueryMemoryDescriptor> MemoryLayoutBuilder::build(
    const std::vector<InputTableInfo>& query_infos,
    const bool allow_multifrag,
    const size_t max_groups_buffer_entry_count,
    const int8_t crt_min_byte_width,
    const bool output_columnar_hint,
    const bool just_explain,
    std::optional<int64_t> group_cardinality_estimation,
    Executor* executor,
    const ExecutorDeviceType device_type) {
  bool sort_on_gpu_hint =
      device_type == ExecutorDeviceType::GPU && allow_multifrag &&
      !ra_exe_unit_.sort_info.order_entries.empty() &&
      gpu_can_handle_order_entries(
          ra_exe_unit_, query_infos, ra_exe_unit_.sort_info.order_entries, executor);
  // must_use_baseline_sort is true iff we'd sort on GPU with the old algorithm
  // but the total output buffer size would be too big or it's a sharded top query.
  // For the sake of managing risk, use the new result set way very selectively for
  // this case only (alongside the baseline layout we've enabled for a while now).
  bool must_use_baseline_sort = false;
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc;
  while (true) {
    query_mem_desc = build_query_memory_descriptor(ra_exe_unit_,
                                                   query_infos,
                                                   allow_multifrag,
                                                   max_groups_buffer_entry_count,
                                                   crt_min_byte_width,
                                                   sort_on_gpu_hint,
                                                   must_use_baseline_sort,
                                                   output_columnar_hint,
                                                   group_cardinality_estimation,
                                                   executor,
                                                   device_type);
    CHECK(query_mem_desc);
    if (query_mem_desc->sortOnGpu() &&
        (query_mem_desc->getBufferSizeBytes(device_type) +
         align_to_int64(query_mem_desc->getEntryCount() * sizeof(int32_t))) >
            2 * 1024 * 1024 * 1024LL) {
      must_use_baseline_sort = true;
      sort_on_gpu_hint = false;
    } else {
      break;
    }
  }

  if (query_mem_desc->getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash &&
      !group_cardinality_estimation && !just_explain) {
    const auto col_range_info = get_col_range_info(
        ra_exe_unit_, query_infos, group_cardinality_estimation, executor, device_type);
    throw CardinalityEstimationRequired(col_range_info.max - col_range_info.min);
  }

  return query_mem_desc;
}

size_t MemoryLayoutBuilder::cudaSharedMemorySize(
    QueryMemoryDescriptor* query_mem_desc,
    const CudaMgr_Namespace::CudaMgr* cuda_mgr,
    Executor* executor,
    const ExecutorDeviceType device_type) const {
  if (device_type == ExecutorDeviceType::CPU) {
    return 0;
  }

  if (!cuda_mgr) {
    return 0;
  }
  const auto gpu_blocksize = executor->blockSize();
  const auto num_blocks_per_mp = executor->numBlocksPerMP();

  CHECK(query_mem_desc);
  if (query_mem_desc->didOutputColumnar()) {
    return 0;
  }

  const Config& config = executor->getConfig();
  /*
   * We only use shared memory strategy if GPU hardware provides native shared
   * memory atomics support. From CUDA Toolkit documentation:
   * https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#atomic-ops "Like
   * Maxwell, Pascal [and Volta] provides native shared memory atomic operations
   * for 32-bit integer arithmetic, along with native 32 or 64-bit compare-and-swap
   * (CAS)."
   *
   **/
  if (!cuda_mgr->isArchMaxwellOrLaterForAll()) {
    return 0;
  }

  if (query_mem_desc->getQueryDescriptionType() ==
          QueryDescriptionType::NonGroupedAggregate &&
      config.exec.group_by.enable_gpu_smem_non_grouped_agg &&
      query_mem_desc->countDistinctDescriptorsLogicallyEmpty()) {
    // TODO: relax this, if necessary
    if (gpu_blocksize < query_mem_desc->getEntryCount()) {
      return 0;
    }
    // skip shared memory usage when dealing with 1) variable length targets, 2)
    // not a COUNT aggregate
    const auto target_infos = target_exprs_to_infos(
        ra_exe_unit_.target_exprs, *query_mem_desc, config.exec.group_by.bigint_count);
    std::unordered_set<SQLAgg> supported_aggs{kCOUNT};
    if (std::find_if(target_infos.begin(),
                     target_infos.end(),
                     [&supported_aggs](const TargetInfo& ti) {
                       if (ti.sql_type.is_varlen() ||
                           !supported_aggs.count(ti.agg_kind)) {
                         return true;
                       } else {
                         return false;
                       }
                     }) == target_infos.end()) {
      return query_mem_desc->getRowSize() * query_mem_desc->getEntryCount();
    }
  }

  if (query_mem_desc->getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash &&
      config.exec.group_by.enable_gpu_smem_group_by) {
    /**
     * To simplify the implementation for practical purposes, we
     * initially provide shared memory support for cases where there are at most as many
     * entries in the output buffer as there are threads within each GPU device. In
     * order to relax this assumption later, we need to add a for loop in generated
     * codes such that each thread loops over multiple entries.
     * TODO: relax this if necessary
     */
    if (gpu_blocksize < query_mem_desc->getEntryCount()) {
      return 0;
    }

    // Fundamentally, we should use shared memory whenever the output buffer
    // is small enough so that we can fit it in the shared memory and yet expect
    // good occupancy.
    // For now, we allow keyless, row-wise layout, and only for perfect hash
    // group by operations.
    if (query_mem_desc->hasKeylessHash() &&
        query_mem_desc->countDistinctDescriptorsLogicallyEmpty() &&
        !query_mem_desc->useStreamingTopN()) {
      const size_t shared_memory_threshold_bytes = std::min(
          config.exec.group_by.gpu_smem_threshold == 0
              ? SIZE_MAX
              : config.exec.group_by.gpu_smem_threshold,
          cuda_mgr->getMinSharedMemoryPerBlockForAllDevices() / num_blocks_per_mp);
      const auto output_buffer_size =
          query_mem_desc->getRowSize() * query_mem_desc->getEntryCount();
      if (output_buffer_size > shared_memory_threshold_bytes) {
        return 0;
      }

      // skip shared memory usage when dealing with 1) variable length targets, 2)
      // non-basic aggregates (COUNT, SUM, MIN, MAX, AVG)
      // TODO: relax this if necessary
      const auto target_infos = target_exprs_to_infos(
          ra_exe_unit_.target_exprs, *query_mem_desc, config.exec.group_by.bigint_count);
      std::unordered_set<SQLAgg> supported_aggs{kCOUNT};
      if (config.exec.group_by.enable_gpu_smem_grouped_non_count_agg) {
        supported_aggs = {kCOUNT, kMIN, kMAX, kSUM, kAVG};
      }
      if (std::find_if(target_infos.begin(),
                       target_infos.end(),
                       [&supported_aggs](const TargetInfo& ti) {
                         if (ti.sql_type.is_varlen() ||
                             !supported_aggs.count(ti.agg_kind)) {
                           return true;
                         } else {
                           return false;
                         }
                       }) == target_infos.end()) {
        return query_mem_desc->getRowSize() * query_mem_desc->getEntryCount();
      }
    }
  }
  return 0;
}