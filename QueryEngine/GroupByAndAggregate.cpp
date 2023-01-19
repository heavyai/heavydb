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

#include "GroupByAndAggregate.h"
#include "AggregateUtils.h"

#include "CardinalityEstimator.h"
#include "CodeGenerator.h"
#include "Descriptors/QueryMemoryDescriptor.h"
#include "ExpressionRange.h"
#include "ExpressionRewrite.h"
#include "GpuInitGroups.h"
#include "InPlaceSort.h"
#include "LLVMFunctionAttributesUtil.h"
#include "MaxwellCodegenPatch.h"
#include "OutputBufferInitialization.h"
#include "TargetExprBuilder.h"

#include "../CudaMgr/CudaMgr.h"
#include "../Shared/checked_alloc.h"
#include "../Shared/funcannotations.h"
#include "../Utils/ChunkIter.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Execute.h"
#include "QueryTemplateGenerator.h"
#include "RuntimeFunctions.h"
#include "Shared/misc.h"
#include "StreamingTopN.h"
#include "TopKSort.h"
#include "WindowContext.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <cstring>  // strcat()
#include <limits>
#include <numeric>
#include <string_view>
#include <thread>

bool g_cluster{false};
bool g_bigint_count{false};
int g_hll_precision_bits{11};
size_t g_watchdog_baseline_max_groups{120000000};
extern int64_t g_bitmap_memory_limit;
extern size_t g_leaf_count;

bool ColRangeInfo::isEmpty() const {
  return min == 0 && max == -1;
}

std::ostream& operator<<(std::ostream& out, const ColRangeInfo& info) {
  out << "Hash Type = " << info.hash_type_ << " min = " << info.min
      << " max = " << info.max << " bucket = " << info.bucket
      << " has_nulls = " << info.has_nulls << "\n";
  return out;
}

std::ostream& operator<<(std::ostream& out, const CountDistinctImplType& type) {
  switch (type) {
    case CountDistinctImplType::Invalid:
      out << "Invalid";
      break;
    case CountDistinctImplType::Bitmap:
      out << "Bitmap";
      break;
    case CountDistinctImplType::UnorderedSet:
      out << "UnorderedSet";
      break;
    default:
      out << "<Unkown Type>";
      break;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const CountDistinctDescriptor& desc) {
  out << "Type = " << desc.impl_type_ << " min val = " << desc.min_val
      << " bitmap_sz_bits = " << desc.bitmap_sz_bits
      << " bool approximate = " << desc.approximate
      << " device_type = " << desc.device_type
      << " sub_bitmap_count = " << desc.sub_bitmap_count;
  return out;
}

namespace {

int32_t get_agg_count(const std::vector<Analyzer::Expr*>& target_exprs) {
  int32_t agg_count{0};
  for (auto target_expr : target_exprs) {
    CHECK(target_expr);
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr || agg_expr->get_aggtype() == kSAMPLE) {
      const auto& ti = target_expr->get_type_info();
      if (ti.is_buffer()) {
        agg_count += 2;
      } else if (ti.is_geometry()) {
        agg_count += ti.get_physical_coord_cols() * 2;
      } else {
        ++agg_count;
      }
      continue;
    }
    if (agg_expr && agg_expr->get_aggtype() == kAVG) {
      agg_count += 2;
    } else {
      ++agg_count;
    }
  }
  return agg_count;
}

bool expr_is_rowid(const Analyzer::Expr* expr) {
  const auto col = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (!col) {
    return false;
  }
  const auto cd = get_column_descriptor_maybe(col->getColumnKey());
  if (!cd || !cd->isVirtualCol) {
    return false;
  }
  CHECK_EQ("rowid", cd->columnName);
  return true;
}

bool has_count_distinct(const RelAlgExecutionUnit& ra_exe_unit) {
  for (const auto& target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
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

ColRangeInfo get_expr_range_info(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<InputTableInfo>& query_infos,
                                 const Analyzer::Expr* expr,
                                 Executor* executor) {
  if (!expr) {
    return {QueryDescriptionType::Projection, 0, 0, 0, false};
  }

  const auto expr_range = getExpressionRange(
      expr, query_infos, executor, boost::make_optional(ra_exe_unit.simple_quals));
  switch (expr_range.getType()) {
    case ExpressionRangeType::Integer: {
      if (expr_range.getIntMin() > expr_range.getIntMax()) {
        return {
            QueryDescriptionType::GroupByBaselineHash, 0, -1, 0, expr_range.hasNulls()};
      }
      return {QueryDescriptionType::GroupByPerfectHash,
              expr_range.getIntMin(),
              expr_range.getIntMax(),
              expr_range.getBucket(),
              expr_range.hasNulls()};
    }
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double: {
      if (expr_range.getFpMin() > expr_range.getFpMax()) {
        return {
            QueryDescriptionType::GroupByBaselineHash, 0, -1, 0, expr_range.hasNulls()};
      }
      return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
    }
    case ExpressionRangeType::Invalid:
      return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
    default:
      CHECK(false);
  }
  CHECK(false);
  return {QueryDescriptionType::NonGroupedAggregate, 0, 0, 0, false};
}

}  // namespace

ColRangeInfo GroupByAndAggregate::getColRangeInfo() {
  // Use baseline layout more eagerly on the GPU if the query uses count distinct,
  // because our HyperLogLog implementation is 4x less memory efficient on GPU.
  // Technically, this only applies to APPROX_COUNT_DISTINCT, but in practice we
  // can expect this to be true anyway for grouped queries since the precise version
  // uses significantly more memory.
  const int64_t baseline_threshold =
      Executor::getBaselineThreshold(has_count_distinct(ra_exe_unit_), device_type_);
  if (ra_exe_unit_.groupby_exprs.size() != 1) {
    try {
      checked_int64_t cardinality{1};
      bool has_nulls{false};
      for (const auto& groupby_expr : ra_exe_unit_.groupby_exprs) {
        auto col_range_info = get_expr_range_info(
            ra_exe_unit_, query_infos_, groupby_expr.get(), executor_);
        if (col_range_info.hash_type_ != QueryDescriptionType::GroupByPerfectHash) {
          // going through baseline hash if a non-integer type is encountered
          return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
        }
        auto crt_col_cardinality = getBucketedCardinality(col_range_info);
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
  if (ra_exe_unit_.groupby_exprs.front() &&
      ra_exe_unit_.groupby_exprs.front()->get_type_info().is_high_precision_timestamp() &&
      ra_exe_unit_.simple_quals.size() > 0) {
    return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
  }
  const auto col_range_info = get_expr_range_info(
      ra_exe_unit_, query_infos_, ra_exe_unit_.groupby_exprs.front().get(), executor_);
  if (!ra_exe_unit_.groupby_exprs.front()) {
    return col_range_info;
  }
  static const int64_t MAX_BUFFER_SIZE = 1 << 30;
  const int64_t col_count =
      ra_exe_unit_.groupby_exprs.size() + ra_exe_unit_.target_exprs.size();
  int64_t max_entry_count = MAX_BUFFER_SIZE / (col_count * sizeof(int64_t));
  if (has_count_distinct(ra_exe_unit_)) {
    max_entry_count = std::min(max_entry_count, baseline_threshold);
  }
  const auto& groupby_expr_ti = ra_exe_unit_.groupby_exprs.front()->get_type_info();
  if (groupby_expr_ti.is_string() && !col_range_info.bucket) {
    CHECK(groupby_expr_ti.get_compression() == kENCODING_DICT);

    const bool has_filters =
        !ra_exe_unit_.quals.empty() || !ra_exe_unit_.simple_quals.empty();
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
      if (!ra_exe_unit_.sort_info.order_entries.empty()) {
        // TODO(adb): allow some sorts to pass through this block by centralizing sort
        // algorithm decision making
        if (has_count_distinct(ra_exe_unit_) &&
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
      if (!group_cardinality_estimation_ ||
          cardinality_estimate_less_than_column_range(*group_cardinality_estimation_,
                                                      col_range_info)) {
        return {QueryDescriptionType::GroupByBaselineHash,
                col_range_info.min,
                col_range_info.max,
                0,
                col_range_info.has_nulls};
      }
    }
  } else if ((!expr_is_rowid(ra_exe_unit_.groupby_exprs.front().get())) &&
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

int64_t GroupByAndAggregate::getBucketedCardinality(const ColRangeInfo& col_range_info) {
  checked_int64_t crt_col_cardinality =
      checked_int64_t(col_range_info.max) - checked_int64_t(col_range_info.min);
  if (col_range_info.bucket) {
    crt_col_cardinality /= col_range_info.bucket;
  }
  return static_cast<int64_t>(crt_col_cardinality +
                              (1 + (col_range_info.has_nulls ? 1 : 0)));
}

namespace {
// Like getBucketedCardinality() without counting nulls.
int64_t get_bucketed_cardinality_without_nulls(const ColRangeInfo& col_range_info) {
  if (col_range_info.min <= col_range_info.max) {
    size_t size = col_range_info.max - col_range_info.min;
    if (col_range_info.bucket) {
      size /= col_range_info.bucket;
    }
    if (size >= static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      // try to use unordered_set instead of crashing due to CHECK failure
      // i.e., CHECK_LT(size, std::numeric_limits<int64_t>::max());
      return 0;
    }
    return static_cast<int64_t>(size + 1);
  } else {
    return 0;
  }
}
}  // namespace

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_BOOL(v) executor_->cgen_state_->llBool(v)
#define LL_INT(v) executor_->cgen_state_->llInt(v)
#define LL_FP(v) executor_->cgen_state_->llFp(v)
#define ROW_FUNC executor_->cgen_state_->row_func_
#define CUR_FUNC executor_->cgen_state_->current_func_

GroupByAndAggregate::GroupByAndAggregate(
    Executor* executor,
    const ExecutorDeviceType device_type,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const std::optional<int64_t>& group_cardinality_estimation)
    : executor_(executor)
    , ra_exe_unit_(ra_exe_unit)
    , query_infos_(query_infos)
    , row_set_mem_owner_(row_set_mem_owner)
    , device_type_(device_type)
    , group_cardinality_estimation_(group_cardinality_estimation) {
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
    if (groupby_ti.is_geometry()) {
      throw std::runtime_error("Group by geometry not supported");
    }
  }
}

int64_t GroupByAndAggregate::getShardedTopBucket(const ColRangeInfo& col_range_info,
                                                 const size_t shard_count) const {
  size_t device_count{0};
  if (device_type_ == ExecutorDeviceType::GPU) {
    device_count = executor_->cudaMgr()->getDeviceCount();
    CHECK_GT(device_count, 0u);
  }

  int64_t bucket{col_range_info.bucket};

  if (shard_count) {
    CHECK(!col_range_info.bucket);
    /*
      when a node has fewer devices than shard count,
      a) In a distributed setup, the minimum distance between two keys would be
      device_count because shards are stored consecutively across the physical tables,
      i.e if a shard column has values 0 to 9, and 3 shards on each leaf, then node 1
      would have values: 0,1,2,6,7,8 and node 2 would have values: 3,4,5,9. If each leaf
      node has only 1 device, in this case, all the keys from each node are loaded on
      the device each.

      b) In a single node setup, the distance would be minimum of device_count or
      difference of device_count - shard_count. For example: If a single node server
      running on 3 devices a shard column has values 0 to 9 in a table with 4 shards,
      device to fragment keys mapping would be: device 1 - 4,8,3,7 device 2 - 1,5,9
      device 3 - 2, 6 The bucket value would be 4(shards) - 3(devices) = 1 i.e. minimum
      of device_count or difference.

      When a node has device count equal to or more than shard count then the
      minimum distance is always at least shard_count * no of leaf nodes.
    */
    if (device_count < shard_count) {
      bucket = g_leaf_count ? std::max(device_count, static_cast<size_t>(1))
                            : std::min(device_count, shard_count - device_count);
    } else {
      bucket = shard_count * std::max(g_leaf_count, static_cast<size_t>(1));
    }
  }

  return bucket;
}

namespace {

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
    const auto agg_info = get_target_info(target_expr, g_bigint_count);
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
    const ColRangeInfo& group_by_range_info,
    const ExecutorDeviceType device_type,
    Executor* executor) {
  CountDistinctDescriptors count_distinct_descriptors;
  auto compute_bytes_per_group =
      [](size_t bitmap_sz, size_t sub_bitmap_count, ExecutorDeviceType device_type) {
        size_t effective_size_bytes = (bitmap_sz + 7) / 8;
        const auto padded_size =
            (device_type == ExecutorDeviceType::GPU || sub_bitmap_count > 1)
                ? align_to_int64(effective_size_bytes)
                : effective_size_bytes;
        return padded_size * sub_bitmap_count;
      };
  for (size_t i = 0; i < ra_exe_unit.target_exprs.size(); i++) {
    const auto target_expr = ra_exe_unit.target_exprs[i];
    auto agg_info = get_target_info(target_expr, g_bigint_count);
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
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT && arg_ti.is_geometry()) {
        throw std::runtime_error(
            "APPROX_COUNT_DISTINCT on geometry columns not supported");
      }
      if (agg_info.is_distinct && arg_ti.is_geometry()) {
        throw std::runtime_error("COUNT DISTINCT on geometry columns not supported");
      }
      ColRangeInfo no_range_info{QueryDescriptionType::Projection, 0, 0, 0, false};
      auto arg_range_info =
          arg_ti.is_fp() ? no_range_info
                         : get_expr_range_info(
                               ra_exe_unit, query_infos, agg_expr->get_arg(), executor);
      const auto it = ra_exe_unit.target_exprs_original_type_infos.find(i);
      if (it != ra_exe_unit.target_exprs_original_type_infos.end()) {
        const auto& original_target_expr_ti = it->second;
        if (arg_ti.is_integer() && original_target_expr_ti.get_type() == kDATE &&
            original_target_expr_ti.get_compression() == kENCODING_DATE_IN_DAYS) {
          // manually encode the col range of date col if necessary
          // (see conditionally_change_arg_to_int_type function in RelAlgExecutor.cpp)
          auto is_date_value_not_encoded = [&original_target_expr_ti](int64_t date_val) {
            if (original_target_expr_ti.get_comp_param() == 16) {
              return date_val < INT16_MIN || date_val > INT16_MAX;
            } else {
              return date_val < INT32_MIN || date_val > INT32_MIN;
            }
          };
          if (is_date_value_not_encoded(arg_range_info.min)) {
            // chunk metadata of the date column contains decoded value
            // so we manually encode it again here to represent its column range correctly
            arg_range_info.min =
                DateConverters::get_epoch_days_from_seconds(arg_range_info.min);
          }
          if (is_date_value_not_encoded(arg_range_info.max)) {
            arg_range_info.max =
                DateConverters::get_epoch_days_from_seconds(arg_range_info.max);
          }
          // now we manually encode the value, so we need to invalidate bucket value
          // i.e., 86000 -> 0, to correctly calculate the size of bitmap
          arg_range_info.bucket = 0;
        }
      }

      CountDistinctImplType count_distinct_impl_type{CountDistinctImplType::UnorderedSet};
      int64_t bitmap_sz_bits{0};
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
        const auto error_rate_expr = agg_expr->get_arg1();
        if (error_rate_expr) {
          CHECK(error_rate_expr->get_type_info().get_type() == kINT);
          auto const error_rate =
              dynamic_cast<Analyzer::Constant const*>(error_rate_expr.get());
          CHECK(error_rate);
          CHECK_GE(error_rate->get_constval().intval, 1);
          bitmap_sz_bits = hll_size_for_rate(error_rate->get_constval().smallintval);
        } else {
          bitmap_sz_bits = g_hll_precision_bits;
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
      const auto sub_bitmap_count =
          get_count_distinct_sub_bitmap_count(bitmap_sz_bits, ra_exe_unit, device_type);
      size_t worst_case_num_groups{1};
      if (arg_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash &&
          !(arg_ti.is_buffer() || arg_ti.is_geometry())) {  // TODO(alex): allow bitmap
                                                            // implementation for arrays
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
        if (shared::is_any<kCOUNT, kCOUNT_IF>(agg_info.agg_kind)) {
          bitmap_sz_bits = get_bucketed_cardinality_without_nulls(arg_range_info);
          if (bitmap_sz_bits <= 0 || g_bitmap_memory_limit <= bitmap_sz_bits) {
            count_distinct_impl_type = CountDistinctImplType::UnorderedSet;
          }
          // check a potential OOM when using bitmap-based approach
          const auto total_bytes_per_entry =
              compute_bytes_per_group(bitmap_sz_bits, sub_bitmap_count, device_type);
          const auto range_bucket = std::max(group_by_range_info.bucket, (int64_t)1);
          const auto maximum_num_groups =
              (group_by_range_info.max - group_by_range_info.min + 1) / range_bucket;
          const auto total_bitmap_bytes_for_groups =
              total_bytes_per_entry * maximum_num_groups;
          // we can estimate a potential OOM of bitmap-based count-distinct operator
          // by using the logic "check_total_bitmap_memory"
          if (total_bitmap_bytes_for_groups >=
              static_cast<size_t>(g_bitmap_memory_limit)) {
            const auto agg_expr_max_entry_count =
                arg_range_info.max - arg_range_info.min + 1;
            int64_t max_agg_expr_table_cardinality{1};
            std::set<const Analyzer::ColumnVar*,
                     bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)>
                colvar_set(Analyzer::ColumnVar::colvar_comp);
            agg_expr->collect_column_var(colvar_set, true);
            for (const auto cv : colvar_set) {
              auto it =
                  std::find_if(query_infos.begin(),
                               query_infos.end(),
                               [&](const auto& input_table_info) {
                                 return input_table_info.table_key == cv->getTableKey();
                               });
              int64_t cur_table_cardinality =
                  it != query_infos.end()
                      ? static_cast<int64_t>(it->info.getNumTuplesUpperBound())
                      : -1;
              max_agg_expr_table_cardinality =
                  std::max(max_agg_expr_table_cardinality, cur_table_cardinality);
              worst_case_num_groups *= cur_table_cardinality;
            }
            auto has_valid_stat = [agg_expr_max_entry_count, maximum_num_groups]() {
              return agg_expr_max_entry_count > 0 && maximum_num_groups > 0;
            };
            // if we have valid stats regarding input expr, we can try to relax the OOM
            if (has_valid_stat()) {
              // a threshold related to a ratio of a range of agg expr (let's say R)
              // and table cardinality (C), i.e., use unordered_set if the # bits to build
              // a bitmap based on R is four times larger than that of C
              const size_t unordered_set_threshold{2};
              // When we detect OOM of bitmap-based approach we selectively switch it to
              // hash set-based processing logic if one of the followings is satisfied:
              // 1) the column range is too wide compared with the table cardinality, or
              // 2) the column range is too wide compared with the avg of # unique values
              // per group by entry
              const auto bits_for_agg_entry = std::ceil(log(agg_expr_max_entry_count));
              const auto bits_for_agg_table =
                  std::ceil(log(max_agg_expr_table_cardinality));
              const auto avg_num_unique_entries_per_group =
                  std::ceil(max_agg_expr_table_cardinality / maximum_num_groups);
              // case a) given a range of entry count of agg_expr and the maximum
              // cardinality among source tables of the agg_expr , we try to detect the
              // misleading case of too sparse column range , i.e., agg_expr has 1M column
              // range but only has two tuples {1 and 1M} / case b) check whether
              // using bitmap is really beneficial when considering uniform distribution
              // of (unique) keys.
              if ((bits_for_agg_entry - bits_for_agg_table) >= unordered_set_threshold ||
                  agg_expr_max_entry_count >= avg_num_unique_entries_per_group) {
                count_distinct_impl_type = CountDistinctImplType::UnorderedSet;
              } else {
                throw std::runtime_error(
                    "Consider using approx_count_distinct operator instead of "
                    "count_distinct operator to lower the memory "
                    "requirements");
              }
            }
          }
        }
      }
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT &&
          count_distinct_impl_type == CountDistinctImplType::UnorderedSet &&
          !(arg_ti.is_array() || arg_ti.is_geometry())) {
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
      }
      const size_t too_many_entries{100000000};
      if (g_enable_watchdog && !(arg_range_info.isEmpty()) &&
          worst_case_num_groups > too_many_entries &&
          count_distinct_impl_type == CountDistinctImplType::UnorderedSet) {
        throw WatchdogException(
            "Detect too many input entries for set-based count distinct operator under "
            "the watchdog");
      }
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

}  // namespace

std::unique_ptr<QueryMemoryDescriptor> GroupByAndAggregate::initQueryMemoryDescriptor(
    const bool allow_multifrag,
    const size_t max_groups_buffer_entry_count,
    const int8_t crt_min_byte_width,
    RenderInfo* render_info,
    const bool output_columnar_hint) {
  const auto shard_count = device_type_ == ExecutorDeviceType::GPU
                               ? shard_count_for_top_groups(ra_exe_unit_)
                               : 0;
  bool sort_on_gpu_hint =
      device_type_ == ExecutorDeviceType::GPU && allow_multifrag &&
      !ra_exe_unit_.sort_info.order_entries.empty() &&
      gpuCanHandleOrderEntries(ra_exe_unit_.sort_info.order_entries) && !shard_count;
  // must_use_baseline_sort is true iff we'd sort on GPU with the old algorithm
  // but the total output buffer size would be too big or it's a sharded top query.
  // For the sake of managing risk, use the new result set way very selectively for
  // this case only (alongside the baseline layout we've enabled for a while now).
  bool must_use_baseline_sort = shard_count;
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc;
  while (true) {
    query_mem_desc = initQueryMemoryDescriptorImpl(allow_multifrag,
                                                   max_groups_buffer_entry_count,
                                                   crt_min_byte_width,
                                                   sort_on_gpu_hint,
                                                   render_info,
                                                   must_use_baseline_sort,
                                                   output_columnar_hint);
    CHECK(query_mem_desc);
    if (query_mem_desc->sortOnGpu() &&
        (query_mem_desc->getBufferSizeBytes(device_type_) +
         align_to_int64(query_mem_desc->getEntryCount() * sizeof(int32_t))) >
            2 * 1024 * 1024 * 1024LL) {
      must_use_baseline_sort = true;
      sort_on_gpu_hint = false;
    } else {
      break;
    }
  }
  return query_mem_desc;
}

std::unique_ptr<QueryMemoryDescriptor> GroupByAndAggregate::initQueryMemoryDescriptorImpl(
    const bool allow_multifrag,
    const size_t max_groups_buffer_entry_count,
    const int8_t crt_min_byte_width,
    const bool sort_on_gpu_hint,
    RenderInfo* render_info,
    const bool must_use_baseline_sort,
    const bool output_columnar_hint) {
  const bool is_group_by{!ra_exe_unit_.groupby_exprs.empty()};

  auto col_range_info_nosharding = getColRangeInfo();

  const auto shard_count = device_type_ == ExecutorDeviceType::GPU
                               ? shard_count_for_top_groups(ra_exe_unit_)
                               : 0;

  const auto col_range_info =
      ColRangeInfo{col_range_info_nosharding.hash_type_,
                   col_range_info_nosharding.min,
                   col_range_info_nosharding.max,
                   getShardedTopBucket(col_range_info_nosharding, shard_count),
                   col_range_info_nosharding.has_nulls};

  // Non-grouped aggregates do not support accessing aggregated ranges
  // Keyless hash is currently only supported with single-column perfect hash
  const auto keyless_info =
      !(is_group_by &&
        col_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash)
          ? KeylessInfo{false, -1}
          : get_keyless_info(ra_exe_unit_, query_infos_, is_group_by, executor_);

  if (g_enable_watchdog &&
      ((col_range_info.hash_type_ == QueryDescriptionType::GroupByBaselineHash &&
        max_groups_buffer_entry_count > g_watchdog_baseline_max_groups) ||
       (col_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash &&
        ra_exe_unit_.groupby_exprs.size() == 1 &&
        (col_range_info.max - col_range_info.min) /
                std::max(col_range_info.bucket, int64_t(1)) >
            130000000))) {
    throw WatchdogException("Query would use too much memory");
  }

  const auto count_distinct_descriptors = init_count_distinct_descriptors(
      ra_exe_unit_, query_infos_, col_range_info, device_type_, executor_);
  try {
    return QueryMemoryDescriptor::init(executor_,
                                       ra_exe_unit_,
                                       query_infos_,
                                       col_range_info,
                                       keyless_info,
                                       allow_multifrag,
                                       device_type_,
                                       crt_min_byte_width,
                                       sort_on_gpu_hint,
                                       shard_count,
                                       max_groups_buffer_entry_count,
                                       render_info,
                                       count_distinct_descriptors,
                                       must_use_baseline_sort,
                                       output_columnar_hint,
                                       /*streaming_top_n_hint=*/true);
  } catch (const StreamingTopNOOM& e) {
    LOG(WARNING) << e.what() << " Disabling Streaming Top N.";
    return QueryMemoryDescriptor::init(executor_,
                                       ra_exe_unit_,
                                       query_infos_,
                                       col_range_info,
                                       keyless_info,
                                       allow_multifrag,
                                       device_type_,
                                       crt_min_byte_width,
                                       sort_on_gpu_hint,
                                       shard_count,
                                       max_groups_buffer_entry_count,
                                       render_info,
                                       count_distinct_descriptors,
                                       must_use_baseline_sort,
                                       output_columnar_hint,
                                       /*streaming_top_n_hint=*/false);
  }
}

bool GroupByAndAggregate::gpuCanHandleOrderEntries(
    const std::list<Analyzer::OrderEntry>& order_entries) {
  if (order_entries.size() > 1) {  // TODO(alex): lift this restriction
    return false;
  }
  for (const auto& order_entry : order_entries) {
    CHECK_GE(order_entry.tle_no, 1);
    CHECK_LE(static_cast<size_t>(order_entry.tle_no), ra_exe_unit_.target_exprs.size());
    const auto target_expr = ra_exe_unit_.target_exprs[order_entry.tle_no - 1];
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
          get_expr_range_info(ra_exe_unit_, query_infos_, agg_expr->get_arg(), executor_);
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

bool GroupByAndAggregate::codegen(llvm::Value* filter_result,
                                  llvm::BasicBlock* sc_false,
                                  QueryMemoryDescriptor& query_mem_desc,
                                  const CompilationOptions& co,
                                  const GpuSharedMemoryContext& gpu_smem_context) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK(filter_result);

  bool can_return_error = false;
  llvm::BasicBlock* filter_false{nullptr};

  {
    const bool is_group_by = !ra_exe_unit_.groupby_exprs.empty();

    if (executor_->isArchMaxwell(co.device_type)) {
      prependForceSync();
    }
    DiamondCodegen filter_cfg(filter_result,
                              executor_,
                              !is_group_by || query_mem_desc.usesGetGroupValueFast(),
                              "filter",  // filter_true and filter_false basic blocks
                              nullptr,
                              false);
    filter_false = filter_cfg.cond_false_;

    if (is_group_by) {
      if (query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection &&
          !query_mem_desc.useStreamingTopN()) {
        const auto crt_matched = get_arg_by_name(ROW_FUNC, "crt_matched");
        LL_BUILDER.CreateStore(LL_INT(int32_t(1)), crt_matched);
        auto total_matched_ptr = get_arg_by_name(ROW_FUNC, "total_matched");
        llvm::Value* old_total_matched_val{nullptr};
        if (co.device_type == ExecutorDeviceType::GPU) {
          old_total_matched_val =
              LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add,
                                         total_matched_ptr,
                                         LL_INT(int32_t(1)),
#if LLVM_VERSION_MAJOR > 12
                                         LLVM_ALIGN(8),
#endif
                                         llvm::AtomicOrdering::Monotonic);
        } else {
          old_total_matched_val = LL_BUILDER.CreateLoad(
              total_matched_ptr->getType()->getPointerElementType(), total_matched_ptr);
          LL_BUILDER.CreateStore(
              LL_BUILDER.CreateAdd(old_total_matched_val, LL_INT(int32_t(1))),
              total_matched_ptr);
        }
        auto old_total_matched_ptr = get_arg_by_name(ROW_FUNC, "old_total_matched");
        LL_BUILDER.CreateStore(old_total_matched_val, old_total_matched_ptr);
      }

      auto agg_out_ptr_w_idx = codegenGroupBy(query_mem_desc, co, filter_cfg);
      auto varlen_output_buffer = codegenVarlenOutputBuffer(query_mem_desc);
      if (query_mem_desc.usesGetGroupValueFast() ||
          query_mem_desc.getQueryDescriptionType() ==
              QueryDescriptionType::GroupByPerfectHash) {
        if (query_mem_desc.getGroupbyColCount() > 1) {
          filter_cfg.setChainToNext();
        }
        // Don't generate null checks if the group slot is guaranteed to be non-null,
        // as it's the case for get_group_value_fast* family.
        can_return_error = codegenAggCalls(agg_out_ptr_w_idx,
                                           varlen_output_buffer,
                                           {},
                                           query_mem_desc,
                                           co,
                                           gpu_smem_context,
                                           filter_cfg);
      } else {
        {
          llvm::Value* nullcheck_cond{nullptr};
          if (query_mem_desc.didOutputColumnar()) {
            nullcheck_cond = LL_BUILDER.CreateICmpSGE(std::get<1>(agg_out_ptr_w_idx),
                                                      LL_INT(int32_t(0)));
          } else {
            nullcheck_cond = LL_BUILDER.CreateICmpNE(
                std::get<0>(agg_out_ptr_w_idx),
                llvm::ConstantPointerNull::get(
                    llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0)));
          }
          DiamondCodegen nullcheck_cfg(
              nullcheck_cond, executor_, false, "groupby_nullcheck", &filter_cfg, false);
          codegenAggCalls(agg_out_ptr_w_idx,
                          varlen_output_buffer,
                          {},
                          query_mem_desc,
                          co,
                          gpu_smem_context,
                          filter_cfg);
        }
        can_return_error = true;
        if (query_mem_desc.getQueryDescriptionType() ==
                QueryDescriptionType::Projection &&
            query_mem_desc.useStreamingTopN()) {
          // Ignore rejection on pushing current row to top-K heap.
          LL_BUILDER.CreateRet(LL_INT(int32_t(0)));
        } else {
          CodeGenerator code_generator(executor_);
          LL_BUILDER.CreateRet(LL_BUILDER.CreateNeg(LL_BUILDER.CreateTrunc(
              // TODO(alex): remove the trunc once pos is converted to 32 bits
              code_generator.posArg(nullptr),
              get_int_type(32, LL_CONTEXT))));
        }
      }
    } else {
      if (ra_exe_unit_.estimator) {
        std::stack<llvm::BasicBlock*> array_loops;
        codegenEstimator(array_loops, filter_cfg, query_mem_desc, co);
      } else {
        auto arg_it = ROW_FUNC->arg_begin();
        std::vector<llvm::Value*> agg_out_vec;
        for (int32_t i = 0; i < get_agg_count(ra_exe_unit_.target_exprs); ++i) {
          agg_out_vec.push_back(&*arg_it++);
        }
        can_return_error = codegenAggCalls(std::make_tuple(nullptr, nullptr),
                                           /*varlen_output_buffer=*/nullptr,
                                           agg_out_vec,
                                           query_mem_desc,
                                           co,
                                           gpu_smem_context,
                                           filter_cfg);
      }
    }
  }

  if (ra_exe_unit_.join_quals.empty()) {
    executor_->cgen_state_->ir_builder_.CreateRet(LL_INT(int32_t(0)));
  } else if (sc_false) {
    const auto saved_insert_block = LL_BUILDER.GetInsertBlock();
    LL_BUILDER.SetInsertPoint(sc_false);
    LL_BUILDER.CreateBr(filter_false);
    LL_BUILDER.SetInsertPoint(saved_insert_block);
  }

  return can_return_error;
}

llvm::Value* GroupByAndAggregate::codegenOutputSlot(
    llvm::Value* groups_buffer,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    DiamondCodegen& diamond_codegen) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK(query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection);
  CHECK_EQ(size_t(1), ra_exe_unit_.groupby_exprs.size());
  const auto group_expr = ra_exe_unit_.groupby_exprs.front();
  CHECK(!group_expr);
  if (!query_mem_desc.didOutputColumnar()) {
    CHECK_EQ(size_t(0), query_mem_desc.getRowSize() % sizeof(int64_t));
  }
  const int32_t row_size_quad = query_mem_desc.didOutputColumnar()
                                    ? 0
                                    : query_mem_desc.getRowSize() / sizeof(int64_t);
  CodeGenerator code_generator(executor_);
  if (query_mem_desc.useStreamingTopN()) {
    const auto& only_order_entry = ra_exe_unit_.sort_info.order_entries.front();
    CHECK_GE(only_order_entry.tle_no, int(1));
    const size_t target_idx = only_order_entry.tle_no - 1;
    CHECK_LT(target_idx, ra_exe_unit_.target_exprs.size());
    const auto order_entry_expr = ra_exe_unit_.target_exprs[target_idx];
    const auto chosen_bytes =
        static_cast<size_t>(query_mem_desc.getPaddedSlotWidthBytes(target_idx));
    auto order_entry_lv = executor_->cgen_state_->castToTypeIn(
        code_generator.codegen(order_entry_expr, true, co).front(), chosen_bytes * 8);
    const uint32_t n = ra_exe_unit_.sort_info.offset + ra_exe_unit_.sort_info.limit;
    std::string fname = "get_bin_from_k_heap";
    const auto& oe_ti = order_entry_expr->get_type_info();
    llvm::Value* null_key_lv = nullptr;
    if (oe_ti.is_integer() || oe_ti.is_decimal() || oe_ti.is_time()) {
      const size_t bit_width = order_entry_lv->getType()->getIntegerBitWidth();
      switch (bit_width) {
        case 32:
          null_key_lv = LL_INT(static_cast<int32_t>(inline_int_null_val(oe_ti)));
          break;
        case 64:
          null_key_lv = LL_INT(static_cast<int64_t>(inline_int_null_val(oe_ti)));
          break;
        default:
          CHECK(false);
      }
      fname += "_int" + std::to_string(bit_width) + "_t";
    } else {
      CHECK(oe_ti.is_fp());
      if (order_entry_lv->getType()->isDoubleTy()) {
        null_key_lv = LL_FP(static_cast<double>(inline_fp_null_val(oe_ti)));
      } else {
        null_key_lv = LL_FP(static_cast<float>(inline_fp_null_val(oe_ti)));
      }
      fname += order_entry_lv->getType()->isDoubleTy() ? "_double" : "_float";
    }
    const auto key_slot_idx =
        get_heap_key_slot_index(ra_exe_unit_.target_exprs, target_idx);
    return emitCall(
        fname,
        {groups_buffer,
         LL_INT(n),
         LL_INT(row_size_quad),
         LL_INT(static_cast<uint32_t>(query_mem_desc.getColOffInBytes(key_slot_idx))),
         LL_BOOL(only_order_entry.is_desc),
         LL_BOOL(!order_entry_expr->get_type_info().get_notnull()),
         LL_BOOL(only_order_entry.nulls_first),
         null_key_lv,
         order_entry_lv});
  } else {
    auto* arg = get_arg_by_name(ROW_FUNC, "max_matched");
    const auto output_buffer_entry_count_lv =
        LL_BUILDER.CreateLoad(arg->getType()->getPointerElementType(), arg);
    arg = get_arg_by_name(ROW_FUNC, "old_total_matched");
    const auto group_expr_lv =
        LL_BUILDER.CreateLoad(arg->getType()->getPointerElementType(), arg);
    std::vector<llvm::Value*> args{groups_buffer,
                                   output_buffer_entry_count_lv,
                                   group_expr_lv,
                                   code_generator.posArg(nullptr)};
    if (query_mem_desc.didOutputColumnar()) {
      const auto columnar_output_offset =
          emitCall("get_columnar_scan_output_offset", args);
      return columnar_output_offset;
    }
    args.push_back(LL_INT(row_size_quad));
    return emitCall("get_scan_output_slot", args);
  }
}

std::tuple<llvm::Value*, llvm::Value*> GroupByAndAggregate::codegenGroupBy(
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    DiamondCodegen& diamond_codegen) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto arg_it = ROW_FUNC->arg_begin();
  auto groups_buffer = arg_it++;

  std::stack<llvm::BasicBlock*> array_loops;

  // TODO(Saman): move this logic outside of this function.
  if (query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    if (query_mem_desc.didOutputColumnar()) {
      return std::make_tuple(
          &*groups_buffer,
          codegenOutputSlot(&*groups_buffer, query_mem_desc, co, diamond_codegen));
    } else {
      return std::make_tuple(
          codegenOutputSlot(&*groups_buffer, query_mem_desc, co, diamond_codegen),
          nullptr);
    }
  }

  CHECK(query_mem_desc.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByBaselineHash ||
        query_mem_desc.getQueryDescriptionType() ==
            QueryDescriptionType::GroupByPerfectHash);

  const int32_t row_size_quad = query_mem_desc.didOutputColumnar()
                                    ? 0
                                    : query_mem_desc.getRowSize() / sizeof(int64_t);

  const auto col_width_size = query_mem_desc.isSingleColumnGroupByWithPerfectHash()
                                  ? sizeof(int64_t)
                                  : query_mem_desc.getEffectiveKeyWidth();
  // for multi-column group by
  llvm::Value* group_key = nullptr;
  llvm::Value* key_size_lv = nullptr;

  if (!query_mem_desc.isSingleColumnGroupByWithPerfectHash()) {
    key_size_lv = LL_INT(static_cast<int32_t>(query_mem_desc.getGroupbyColCount()));
    if (query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash) {
      group_key =
          LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
    } else if (query_mem_desc.getQueryDescriptionType() ==
               QueryDescriptionType::GroupByBaselineHash) {
      group_key =
          col_width_size == sizeof(int32_t)
              ? LL_BUILDER.CreateAlloca(llvm::Type::getInt32Ty(LL_CONTEXT), key_size_lv)
              : LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
    }
    CHECK(group_key);
    CHECK(key_size_lv);
  }

  int32_t subkey_idx = 0;
  CHECK(query_mem_desc.getGroupbyColCount() == ra_exe_unit_.groupby_exprs.size());
  for (const auto& group_expr : ra_exe_unit_.groupby_exprs) {
    const auto col_range_info =
        get_expr_range_info(ra_exe_unit_, query_infos_, group_expr.get(), executor_);
    const auto translated_null_value = static_cast<int64_t>(
        query_mem_desc.isSingleColumnGroupByWithPerfectHash()
            ? checked_int64_t(query_mem_desc.getMaxVal()) +
                  (query_mem_desc.getBucket() ? query_mem_desc.getBucket() : 1)
            : checked_int64_t(col_range_info.max) +
                  (col_range_info.bucket ? col_range_info.bucket : 1));

    const bool col_has_nulls =
        query_mem_desc.getQueryDescriptionType() ==
                QueryDescriptionType::GroupByPerfectHash
            ? (query_mem_desc.isSingleColumnGroupByWithPerfectHash()
                   ? query_mem_desc.hasNulls()
                   : col_range_info.has_nulls)
            : false;

    const auto group_expr_lvs =
        executor_->groupByColumnCodegen(group_expr.get(),
                                        col_width_size,
                                        co,
                                        col_has_nulls,
                                        translated_null_value,
                                        diamond_codegen,
                                        array_loops,
                                        query_mem_desc.threadsShareMemory());
    const auto group_expr_lv = group_expr_lvs.translated_value;
    if (query_mem_desc.isSingleColumnGroupByWithPerfectHash()) {
      CHECK_EQ(size_t(1), ra_exe_unit_.groupby_exprs.size());
      return codegenSingleColumnPerfectHash(query_mem_desc,
                                            co,
                                            &*groups_buffer,
                                            group_expr_lv,
                                            group_expr_lvs.original_value,
                                            row_size_quad);
    } else {
      // store the sub-key to the buffer
      LL_BUILDER.CreateStore(
          group_expr_lv,
          LL_BUILDER.CreateGEP(
              group_key->getType()->getScalarType()->getPointerElementType(),
              group_key,
              LL_INT(subkey_idx++)));
    }
  }
  if (query_mem_desc.getQueryDescriptionType() ==
      QueryDescriptionType::GroupByPerfectHash) {
    CHECK(ra_exe_unit_.groupby_exprs.size() != 1);
    return codegenMultiColumnPerfectHash(
        &*groups_buffer, group_key, key_size_lv, query_mem_desc, row_size_quad);
  } else if (query_mem_desc.getQueryDescriptionType() ==
             QueryDescriptionType::GroupByBaselineHash) {
    return codegenMultiColumnBaselineHash(co,
                                          &*groups_buffer,
                                          group_key,
                                          key_size_lv,
                                          query_mem_desc,
                                          col_width_size,
                                          row_size_quad);
  }
  CHECK(false);
  return std::make_tuple(nullptr, nullptr);
}

llvm::Value* GroupByAndAggregate::codegenVarlenOutputBuffer(
    const QueryMemoryDescriptor& query_mem_desc) {
  if (!query_mem_desc.hasVarlenOutput()) {
    return nullptr;
  }

  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto arg_it = ROW_FUNC->arg_begin();
  arg_it++; /* groups_buffer */
  auto varlen_output_buffer = arg_it++;
  CHECK(varlen_output_buffer->getType() == llvm::Type::getInt64PtrTy(LL_CONTEXT));
  return varlen_output_buffer;
}

std::tuple<llvm::Value*, llvm::Value*>
GroupByAndAggregate::codegenSingleColumnPerfectHash(
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    llvm::Value* groups_buffer,
    llvm::Value* group_expr_lv_translated,
    llvm::Value* group_expr_lv_original,
    const int32_t row_size_quad) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK(query_mem_desc.usesGetGroupValueFast());
  std::string get_group_fn_name{query_mem_desc.didOutputColumnar()
                                    ? "get_columnar_group_bin_offset"
                                    : "get_group_value_fast"};
  if (!query_mem_desc.didOutputColumnar() && query_mem_desc.hasKeylessHash()) {
    get_group_fn_name += "_keyless";
  }
  if (query_mem_desc.interleavedBins(co.device_type)) {
    CHECK(!query_mem_desc.didOutputColumnar());
    CHECK(query_mem_desc.hasKeylessHash());
    get_group_fn_name += "_semiprivate";
  }
  std::vector<llvm::Value*> get_group_fn_args{&*groups_buffer,
                                              &*group_expr_lv_translated};
  if (group_expr_lv_original && get_group_fn_name == "get_group_value_fast" &&
      query_mem_desc.mustUseBaselineSort()) {
    get_group_fn_name += "_with_original_key";
    get_group_fn_args.push_back(group_expr_lv_original);
  }
  get_group_fn_args.push_back(LL_INT(query_mem_desc.getMinVal()));
  get_group_fn_args.push_back(LL_INT(query_mem_desc.getBucket()));
  if (!query_mem_desc.hasKeylessHash()) {
    if (!query_mem_desc.didOutputColumnar()) {
      get_group_fn_args.push_back(LL_INT(row_size_quad));
    }
  } else {
    if (!query_mem_desc.didOutputColumnar()) {
      get_group_fn_args.push_back(LL_INT(row_size_quad));
    }
    if (query_mem_desc.interleavedBins(co.device_type)) {
      auto warp_idx = emitCall("thread_warp_idx", {LL_INT(executor_->warpSize())});
      get_group_fn_args.push_back(warp_idx);
      get_group_fn_args.push_back(LL_INT(executor_->warpSize()));
    }
  }
  if (get_group_fn_name == "get_columnar_group_bin_offset") {
    return std::make_tuple(&*groups_buffer,
                           emitCall(get_group_fn_name, get_group_fn_args));
  }
  return std::make_tuple(emitCall(get_group_fn_name, get_group_fn_args), nullptr);
}

std::tuple<llvm::Value*, llvm::Value*> GroupByAndAggregate::codegenMultiColumnPerfectHash(
    llvm::Value* groups_buffer,
    llvm::Value* group_key,
    llvm::Value* key_size_lv,
    const QueryMemoryDescriptor& query_mem_desc,
    const int32_t row_size_quad) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK(query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByPerfectHash);
  // compute the index (perfect hash)
  auto perfect_hash_func = codegenPerfectHashFunction();
  auto hash_lv =
      LL_BUILDER.CreateCall(perfect_hash_func, std::vector<llvm::Value*>{group_key});

  if (query_mem_desc.didOutputColumnar()) {
    if (!query_mem_desc.hasKeylessHash()) {
      const std::string set_matching_func_name{
          "set_matching_group_value_perfect_hash_columnar"};
      const std::vector<llvm::Value*> set_matching_func_arg{
          groups_buffer,
          hash_lv,
          group_key,
          key_size_lv,
          llvm::ConstantInt::get(get_int_type(32, LL_CONTEXT),
                                 query_mem_desc.getEntryCount())};
      emitCall(set_matching_func_name, set_matching_func_arg);
    }
    return std::make_tuple(groups_buffer, hash_lv);
  } else {
    if (query_mem_desc.hasKeylessHash()) {
      return std::make_tuple(emitCall("get_matching_group_value_perfect_hash_keyless",
                                      {groups_buffer, hash_lv, LL_INT(row_size_quad)}),
                             nullptr);
    } else {
      return std::make_tuple(
          emitCall(
              "get_matching_group_value_perfect_hash",
              {groups_buffer, hash_lv, group_key, key_size_lv, LL_INT(row_size_quad)}),
          nullptr);
    }
  }
}

std::tuple<llvm::Value*, llvm::Value*>
GroupByAndAggregate::codegenMultiColumnBaselineHash(
    const CompilationOptions& co,
    llvm::Value* groups_buffer,
    llvm::Value* group_key,
    llvm::Value* key_size_lv,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t key_width,
    const int32_t row_size_quad) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  if (group_key->getType() != llvm::Type::getInt64PtrTy(LL_CONTEXT)) {
    CHECK(key_width == sizeof(int32_t));
    group_key =
        LL_BUILDER.CreatePointerCast(group_key, llvm::Type::getInt64PtrTy(LL_CONTEXT));
  }
  std::vector<llvm::Value*> func_args{
      groups_buffer,
      LL_INT(static_cast<int32_t>(query_mem_desc.getEntryCount())),
      &*group_key,
      &*key_size_lv,
      LL_INT(static_cast<int32_t>(key_width))};
  std::string func_name{"get_group_value"};
  if (query_mem_desc.didOutputColumnar()) {
    func_name += "_columnar_slot";
  } else {
    func_args.push_back(LL_INT(row_size_quad));
  }
  if (co.with_dynamic_watchdog) {
    func_name += "_with_watchdog";
  }
  if (query_mem_desc.didOutputColumnar()) {
    return std::make_tuple(groups_buffer, emitCall(func_name, func_args));
  } else {
    return std::make_tuple(emitCall(func_name, func_args), nullptr);
  }
}

llvm::Function* GroupByAndAggregate::codegenPerfectHashFunction() {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CHECK_GT(ra_exe_unit_.groupby_exprs.size(), size_t(1));
  auto ft = llvm::FunctionType::get(
      get_int_type(32, LL_CONTEXT),
      std::vector<llvm::Type*>{llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0)},
      false);
  auto key_hash_func = llvm::Function::Create(ft,
                                              llvm::Function::ExternalLinkage,
                                              "perfect_key_hash",
                                              executor_->cgen_state_->module_);
  executor_->cgen_state_->helper_functions_.push_back(key_hash_func);
  mark_function_always_inline(key_hash_func);
  auto& key_buff_arg = *key_hash_func->args().begin();
  llvm::Value* key_buff_lv = &key_buff_arg;
  auto bb = llvm::BasicBlock::Create(LL_CONTEXT, "entry", key_hash_func);
  llvm::IRBuilder<> key_hash_func_builder(bb);
  llvm::Value* hash_lv{llvm::ConstantInt::get(get_int_type(64, LL_CONTEXT), 0)};
  std::vector<int64_t> cardinalities;
  for (const auto& groupby_expr : ra_exe_unit_.groupby_exprs) {
    auto col_range_info =
        get_expr_range_info(ra_exe_unit_, query_infos_, groupby_expr.get(), executor_);
    CHECK(col_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash);
    cardinalities.push_back(getBucketedCardinality(col_range_info));
  }
  size_t dim_idx = 0;
  for (const auto& groupby_expr : ra_exe_unit_.groupby_exprs) {
    auto* gep = key_hash_func_builder.CreateGEP(
        key_buff_lv->getType()->getScalarType()->getPointerElementType(),
        key_buff_lv,
        LL_INT(dim_idx));
    auto key_comp_lv =
        key_hash_func_builder.CreateLoad(gep->getType()->getPointerElementType(), gep);
    auto col_range_info =
        get_expr_range_info(ra_exe_unit_, query_infos_, groupby_expr.get(), executor_);
    auto crt_term_lv =
        key_hash_func_builder.CreateSub(key_comp_lv, LL_INT(col_range_info.min));
    if (col_range_info.bucket) {
      crt_term_lv =
          key_hash_func_builder.CreateSDiv(crt_term_lv, LL_INT(col_range_info.bucket));
    }
    for (size_t prev_dim_idx = 0; prev_dim_idx < dim_idx; ++prev_dim_idx) {
      crt_term_lv = key_hash_func_builder.CreateMul(crt_term_lv,
                                                    LL_INT(cardinalities[prev_dim_idx]));
    }
    hash_lv = key_hash_func_builder.CreateAdd(hash_lv, crt_term_lv);
    ++dim_idx;
  }
  key_hash_func_builder.CreateRet(
      key_hash_func_builder.CreateTrunc(hash_lv, get_int_type(32, LL_CONTEXT)));
  return key_hash_func;
}

llvm::Value* GroupByAndAggregate::convertNullIfAny(const SQLTypeInfo& arg_type,
                                                   const TargetInfo& agg_info,
                                                   llvm::Value* target) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto& agg_type = agg_info.sql_type;
  const size_t chosen_bytes = agg_type.get_size();

  bool need_conversion{false};
  llvm::Value* arg_null{nullptr};
  llvm::Value* agg_null{nullptr};
  llvm::Value* target_to_cast{target};
  if (arg_type.is_fp()) {
    arg_null = executor_->cgen_state_->inlineFpNull(arg_type);
    if (agg_type.is_fp()) {
      agg_null = executor_->cgen_state_->inlineFpNull(agg_type);
      if (!static_cast<llvm::ConstantFP*>(arg_null)->isExactlyValue(
              static_cast<llvm::ConstantFP*>(agg_null)->getValueAPF())) {
        need_conversion = true;
      }
    } else {
      CHECK(agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT);
      return target;
    }
  } else {
    arg_null = executor_->cgen_state_->inlineIntNull(arg_type);
    if (agg_type.is_fp()) {
      agg_null = executor_->cgen_state_->inlineFpNull(agg_type);
      need_conversion = true;
      target_to_cast = executor_->castToFP(target, arg_type, agg_type);
    } else {
      agg_null = executor_->cgen_state_->inlineIntNull(agg_type);
      if ((static_cast<llvm::ConstantInt*>(arg_null)->getBitWidth() !=
           static_cast<llvm::ConstantInt*>(agg_null)->getBitWidth()) ||
          (static_cast<llvm::ConstantInt*>(arg_null)->getValue() !=
           static_cast<llvm::ConstantInt*>(agg_null)->getValue())) {
        need_conversion = true;
      }
    }
  }
  if (need_conversion) {
    auto cmp = arg_type.is_fp() ? LL_BUILDER.CreateFCmpOEQ(target, arg_null)
                                : LL_BUILDER.CreateICmpEQ(target, arg_null);
    return LL_BUILDER.CreateSelect(
        cmp,
        agg_null,
        executor_->cgen_state_->castToTypeIn(target_to_cast, chosen_bytes << 3));
  } else {
    return target;
  }
}

llvm::Value* GroupByAndAggregate::codegenWindowRowPointer(
    const Analyzer::WindowFunction* window_func,
    const QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    DiamondCodegen& diamond_codegen) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(executor_);
  if (window_func_context && window_function_is_aggregate(window_func->getKind())) {
    const int32_t row_size_quad = query_mem_desc.didOutputColumnar()
                                      ? 0
                                      : query_mem_desc.getRowSize() / sizeof(int64_t);
    auto arg_it = ROW_FUNC->arg_begin();
    auto groups_buffer = arg_it++;
    CodeGenerator code_generator(executor_);
    auto window_pos_lv = code_generator.codegenWindowPosition(
        window_func_context, code_generator.posArg(nullptr));
    const auto pos_in_window =
        LL_BUILDER.CreateTrunc(window_pos_lv, get_int_type(32, LL_CONTEXT));
    llvm::Value* entry_count_lv =
        LL_INT(static_cast<int32_t>(query_mem_desc.getEntryCount()));
    std::vector<llvm::Value*> args{
        &*groups_buffer, entry_count_lv, pos_in_window, code_generator.posArg(nullptr)};
    if (query_mem_desc.didOutputColumnar()) {
      const auto columnar_output_offset =
          emitCall("get_columnar_scan_output_offset", args);
      return LL_BUILDER.CreateSExt(columnar_output_offset, get_int_type(64, LL_CONTEXT));
    }
    args.push_back(LL_INT(row_size_quad));
    return emitCall("get_scan_output_slot", args);
  }
  auto arg_it = ROW_FUNC->arg_begin();
  auto groups_buffer = arg_it++;
  return codegenOutputSlot(&*groups_buffer, query_mem_desc, co, diamond_codegen);
}

bool GroupByAndAggregate::codegenAggCalls(
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx_in,
    llvm::Value* varlen_output_buffer,
    const std::vector<llvm::Value*>& agg_out_vec,
    QueryMemoryDescriptor& query_mem_desc,
    const CompilationOptions& co,
    const GpuSharedMemoryContext& gpu_smem_context,
    DiamondCodegen& diamond_codegen) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto agg_out_ptr_w_idx = agg_out_ptr_w_idx_in;
  // TODO(alex): unify the two cases, the output for non-group by queries
  //             should be a contiguous buffer
  const bool is_group_by = std::get<0>(agg_out_ptr_w_idx);
  bool can_return_error = false;
  if (is_group_by) {
    CHECK(agg_out_vec.empty());
  } else {
    CHECK(!agg_out_vec.empty());
  }

  // output buffer is casted into a byte stream to be able to handle data elements of
  // different sizes (only used when actual column width sizes are used)
  llvm::Value* output_buffer_byte_stream{nullptr};
  llvm::Value* out_row_idx{nullptr};
  if (query_mem_desc.didOutputColumnar() && !g_cluster &&
      query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    output_buffer_byte_stream = LL_BUILDER.CreateBitCast(
        std::get<0>(agg_out_ptr_w_idx),
        llvm::PointerType::get(llvm::Type::getInt8Ty(LL_CONTEXT), 0));
    output_buffer_byte_stream->setName("out_buff_b_stream");
    CHECK(std::get<1>(agg_out_ptr_w_idx));
    out_row_idx = LL_BUILDER.CreateZExt(std::get<1>(agg_out_ptr_w_idx),
                                        llvm::Type::getInt64Ty(LL_CONTEXT));
    out_row_idx->setName("out_row_idx");
  }

  TargetExprCodegenBuilder target_builder(ra_exe_unit_, is_group_by);
  for (size_t target_idx = 0; target_idx < ra_exe_unit_.target_exprs.size();
       ++target_idx) {
    auto target_expr = ra_exe_unit_.target_exprs[target_idx];
    CHECK(target_expr);

    target_builder(target_expr, executor_, query_mem_desc, co);
  }

  target_builder.codegen(this,
                         executor_,
                         query_mem_desc,
                         co,
                         gpu_smem_context,
                         agg_out_ptr_w_idx,
                         agg_out_vec,
                         output_buffer_byte_stream,
                         out_row_idx,
                         varlen_output_buffer,
                         diamond_codegen);

  return can_return_error;
}

/**
 * @brief: returns the pointer to where the aggregation should be stored.
 */
llvm::Value* GroupByAndAggregate::codegenAggColumnPtr(
    llvm::Value* output_buffer_byte_stream,
    llvm::Value* out_row_idx,
    const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t chosen_bytes,
    const size_t agg_out_off,
    const size_t target_idx) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  llvm::Value* agg_col_ptr{nullptr};
  if (query_mem_desc.didOutputColumnar()) {
    // TODO(Saman): remove the second columnar branch, and support all query description
    // types through the first branch. Then, input arguments should also be cleaned up
    if (!g_cluster &&
        query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
      CHECK(chosen_bytes == 1 || chosen_bytes == 2 || chosen_bytes == 4 ||
            chosen_bytes == 8);
      CHECK(output_buffer_byte_stream);
      CHECK(out_row_idx);
      size_t col_off = query_mem_desc.getColOffInBytes(agg_out_off);
      // multiplying by chosen_bytes, i.e., << log2(chosen_bytes)
      auto out_per_col_byte_idx =
#ifdef _WIN32
          LL_BUILDER.CreateShl(out_row_idx, __lzcnt(chosen_bytes) - 1);
#else
          LL_BUILDER.CreateShl(out_row_idx, __builtin_ffs(chosen_bytes) - 1);
#endif
      auto byte_offset = LL_BUILDER.CreateAdd(out_per_col_byte_idx,
                                              LL_INT(static_cast<int64_t>(col_off)));
      byte_offset->setName("out_byte_off_target_" + std::to_string(target_idx));
      auto output_ptr = LL_BUILDER.CreateGEP(
          output_buffer_byte_stream->getType()->getScalarType()->getPointerElementType(),
          output_buffer_byte_stream,
          byte_offset);
      agg_col_ptr = LL_BUILDER.CreateBitCast(
          output_ptr,
          llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0));
      agg_col_ptr->setName("out_ptr_target_" + std::to_string(target_idx));
    } else {
      auto const col_off_in_bytes = query_mem_desc.getColOffInBytes(agg_out_off);
      auto const col_off = col_off_in_bytes / chosen_bytes;
      auto const col_rem = col_off_in_bytes % chosen_bytes;
      CHECK_EQ(col_rem, 0u) << col_off_in_bytes << " % " << chosen_bytes;
      CHECK(std::get<1>(agg_out_ptr_w_idx));
      auto* agg_out_idx = LL_BUILDER.CreateZExt(
          std::get<1>(agg_out_ptr_w_idx),
          get_int_type(8 * sizeof(col_off), executor_->cgen_state_->context_));
      auto* offset = LL_BUILDER.CreateAdd(agg_out_idx, LL_INT(col_off));
      auto* bit_cast = LL_BUILDER.CreateBitCast(
          std::get<0>(agg_out_ptr_w_idx),
          llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0));
      agg_col_ptr = LL_BUILDER.CreateGEP(
          bit_cast->getType()->getScalarType()->getPointerElementType(),
          bit_cast,
          offset);
    }
  } else {
    auto const col_off_in_bytes = query_mem_desc.getColOnlyOffInBytes(agg_out_off);
    auto const col_off = col_off_in_bytes / chosen_bytes;
    auto const col_rem = col_off_in_bytes % chosen_bytes;
    CHECK_EQ(col_rem, 0u) << col_off_in_bytes << " % " << chosen_bytes;
    auto* bit_cast = LL_BUILDER.CreateBitCast(
        std::get<0>(agg_out_ptr_w_idx),
        llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0));
    agg_col_ptr = LL_BUILDER.CreateGEP(
        bit_cast->getType()->getScalarType()->getPointerElementType(),
        bit_cast,
        LL_INT(col_off));
  }
  CHECK(agg_col_ptr);
  return agg_col_ptr;
}

void GroupByAndAggregate::codegenEstimator(std::stack<llvm::BasicBlock*>& array_loops,
                                           DiamondCodegen& diamond_codegen,
                                           const QueryMemoryDescriptor& query_mem_desc,
                                           const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto& estimator_arg = ra_exe_unit_.estimator->getArgument();
  auto estimator_comp_count_lv = LL_INT(static_cast<int32_t>(estimator_arg.size()));
  auto estimator_key_lv = LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT),
                                                  estimator_comp_count_lv);
  int32_t subkey_idx = 0;
  for (const auto& estimator_arg_comp : estimator_arg) {
    const auto estimator_arg_comp_lvs =
        executor_->groupByColumnCodegen(estimator_arg_comp.get(),
                                        query_mem_desc.getEffectiveKeyWidth(),
                                        co,
                                        false,
                                        0,
                                        diamond_codegen,
                                        array_loops,
                                        true);
    CHECK(!estimator_arg_comp_lvs.original_value);
    const auto estimator_arg_comp_lv = estimator_arg_comp_lvs.translated_value;
    // store the sub-key to the buffer
    LL_BUILDER.CreateStore(
        estimator_arg_comp_lv,
        LL_BUILDER.CreateGEP(
            estimator_key_lv->getType()->getScalarType()->getPointerElementType(),
            estimator_key_lv,
            LL_INT(subkey_idx++)));
  }
  const auto int8_ptr_ty = llvm::PointerType::get(get_int_type(8, LL_CONTEXT), 0);
  const auto bitmap = LL_BUILDER.CreateBitCast(&*ROW_FUNC->arg_begin(), int8_ptr_ty);
  const auto key_bytes = LL_BUILDER.CreateBitCast(estimator_key_lv, int8_ptr_ty);
  const auto estimator_comp_bytes_lv =
      LL_INT(static_cast<int32_t>(estimator_arg.size() * sizeof(int64_t)));
  const auto bitmap_size_lv =
      LL_INT(static_cast<uint32_t>(ra_exe_unit_.estimator->getBufferSize()));
  emitCall(ra_exe_unit_.estimator->getRuntimeFunctionName(),
           {bitmap, &*bitmap_size_lv, key_bytes, &*estimator_comp_bytes_lv});
}

extern "C" RUNTIME_EXPORT void agg_count_distinct(int64_t* agg, const int64_t val) {
  reinterpret_cast<CountDistinctSet*>(*agg)->insert(val);
}

extern "C" RUNTIME_EXPORT void agg_count_distinct_skip_val(int64_t* agg,
                                                           const int64_t val,
                                                           const int64_t skip_val) {
  if (val != skip_val) {
    agg_count_distinct(agg, val);
  }
}

extern "C" RUNTIME_EXPORT void agg_approx_quantile(int64_t* agg, const double val) {
  auto* t_digest = reinterpret_cast<quantile::TDigest*>(*agg);
  t_digest->allocate();
  t_digest->add(val);
}

extern "C" RUNTIME_EXPORT void agg_mode_func(int64_t* agg, const int64_t val) {
  auto* mode_map = reinterpret_cast<AggMode*>(*agg);
  mode_map->add(val);
}

void GroupByAndAggregate::codegenCountDistinct(
    const size_t target_idx,
    const Analyzer::Expr* target_expr,
    std::vector<llvm::Value*>& agg_args,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto agg_info = get_target_info(target_expr, g_bigint_count);
  const auto& arg_ti =
      static_cast<const Analyzer::AggExpr*>(target_expr)->get_arg()->get_type_info();
  if (arg_ti.is_fp()) {
    agg_args.back() = executor_->cgen_state_->ir_builder_.CreateBitCast(
        agg_args.back(), get_int_type(64, executor_->cgen_state_->context_));
  }
  const auto& count_distinct_descriptor =
      query_mem_desc.getCountDistinctDescriptor(target_idx);
  CHECK(count_distinct_descriptor.impl_type_ != CountDistinctImplType::Invalid);
  if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
    CHECK(count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap);
    agg_args.push_back(LL_INT(int32_t(count_distinct_descriptor.bitmap_sz_bits)));
    if (device_type == ExecutorDeviceType::GPU) {
      const auto base_dev_addr = getAdditionalLiteral(-1);
      const auto base_host_addr = getAdditionalLiteral(-2);
      agg_args.push_back(base_dev_addr);
      agg_args.push_back(base_host_addr);
      emitCall("agg_approximate_count_distinct_gpu", agg_args);
    } else {
      emitCall("agg_approximate_count_distinct", agg_args);
    }
    return;
  }
  std::string agg_fname{"agg_count_distinct"};
  if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap) {
    agg_fname += "_bitmap";
    agg_args.push_back(LL_INT(static_cast<int64_t>(count_distinct_descriptor.min_val)));
  }
  if (agg_info.skip_null_val) {
    auto null_lv = executor_->cgen_state_->castToTypeIn(
        (arg_ti.is_fp()
             ? static_cast<llvm::Value*>(executor_->cgen_state_->inlineFpNull(arg_ti))
             : static_cast<llvm::Value*>(executor_->cgen_state_->inlineIntNull(arg_ti))),
        64);
    null_lv = executor_->cgen_state_->ir_builder_.CreateBitCast(
        null_lv, get_int_type(64, executor_->cgen_state_->context_));
    agg_fname += "_skip_val";
    agg_args.push_back(null_lv);
  }
  if (device_type == ExecutorDeviceType::GPU) {
    CHECK(count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap);
    agg_fname += "_gpu";
    const auto base_dev_addr = getAdditionalLiteral(-1);
    const auto base_host_addr = getAdditionalLiteral(-2);
    agg_args.push_back(base_dev_addr);
    agg_args.push_back(base_host_addr);
    agg_args.push_back(LL_INT(int64_t(count_distinct_descriptor.sub_bitmap_count)));
    CHECK_EQ(size_t(0),
             count_distinct_descriptor.bitmapPaddedSizeBytes() %
                 count_distinct_descriptor.sub_bitmap_count);
    agg_args.push_back(LL_INT(int64_t(count_distinct_descriptor.bitmapPaddedSizeBytes() /
                                      count_distinct_descriptor.sub_bitmap_count)));
  }
  if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap) {
    emitCall(agg_fname, agg_args);
  } else {
    executor_->cgen_state_->emitExternalCall(
        agg_fname, llvm::Type::getVoidTy(LL_CONTEXT), agg_args);
  }
}

void GroupByAndAggregate::codegenApproxQuantile(
    const size_t target_idx,
    const Analyzer::Expr* target_expr,
    std::vector<llvm::Value*>& agg_args,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type) {
  if (device_type == ExecutorDeviceType::GPU) {
    throw QueryMustRunOnCpu();
  }
  llvm::BasicBlock *calc, *skip{nullptr};
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto const arg_ti =
      static_cast<const Analyzer::AggExpr*>(target_expr)->get_arg()->get_type_info();
  bool const nullable = !arg_ti.get_notnull();

  auto* cs = executor_->cgen_state_.get();
  auto& irb = cs->ir_builder_;
  if (nullable) {
    auto* const null_value = cs->castToTypeIn(cs->inlineNull(arg_ti), 64);
    auto* const skip_cond = arg_ti.is_fp()
                                ? irb.CreateFCmpOEQ(agg_args.back(), null_value)
                                : irb.CreateICmpEQ(agg_args.back(), null_value);
    calc = llvm::BasicBlock::Create(cs->context_, "calc_approx_quantile");
    skip = llvm::BasicBlock::Create(cs->context_, "skip_approx_quantile");
    irb.CreateCondBr(skip_cond, skip, calc);
    cs->current_func_->getBasicBlockList().push_back(calc);
    irb.SetInsertPoint(calc);
  }
  if (!arg_ti.is_fp()) {
    auto const agg_info = get_target_info(target_expr, g_bigint_count);
    agg_args.back() = executor_->castToFP(agg_args.back(), arg_ti, agg_info.sql_type);
  }
  cs->emitExternalCall(
      "agg_approx_quantile", llvm::Type::getVoidTy(cs->context_), agg_args);
  if (nullable) {
    irb.CreateBr(skip);
    cs->current_func_->getBasicBlockList().push_back(skip);
    irb.SetInsertPoint(skip);
  }
}

void GroupByAndAggregate::codegenMode(const size_t target_idx,
                                      const Analyzer::Expr* target_expr,
                                      std::vector<llvm::Value*>& agg_args,
                                      const QueryMemoryDescriptor& query_mem_desc,
                                      const ExecutorDeviceType device_type) {
  if (device_type == ExecutorDeviceType::GPU) {
    throw QueryMustRunOnCpu();
  }
  llvm::BasicBlock *calc, *skip{nullptr};
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto const arg_ti =
      static_cast<const Analyzer::AggExpr*>(target_expr)->get_arg()->get_type_info();
  bool const nullable = !arg_ti.get_notnull();
  bool const is_fp = arg_ti.is_fp();
  auto* cs = executor_->cgen_state_.get();
  auto& irb = cs->ir_builder_;
  if (nullable) {
    auto* const null_value =
        is_fp ? cs->inlineNull(arg_ti) : cs->castToTypeIn(cs->inlineNull(arg_ti), 64);
    auto* const skip_cond = is_fp ? irb.CreateFCmpOEQ(agg_args.back(), null_value)
                                  : irb.CreateICmpEQ(agg_args.back(), null_value);
    calc = llvm::BasicBlock::Create(cs->context_, "calc_mode");
    skip = llvm::BasicBlock::Create(cs->context_, "skip_mode");
    irb.CreateCondBr(skip_cond, skip, calc);
    cs->current_func_->getBasicBlockList().push_back(calc);
    irb.SetInsertPoint(calc);
  }
  if (is_fp) {
    auto* const int_type = get_int_type(8 * arg_ti.get_size(), cs->context_);
    agg_args.back() = irb.CreateBitCast(agg_args.back(), int_type);
  }
  // "agg_mode" collides with existing names, so non-standard suffix "_func" is added.
  cs->emitExternalCall("agg_mode_func", llvm::Type::getVoidTy(cs->context_), agg_args);
  if (nullable) {
    irb.CreateBr(skip);
    cs->current_func_->getBasicBlockList().push_back(skip);
    irb.SetInsertPoint(skip);
  }
}

llvm::Value* GroupByAndAggregate::getAdditionalLiteral(const int32_t off) {
  CHECK_LT(off, 0);
  const auto lit_buff_lv = get_arg_by_name(ROW_FUNC, "literals");
  auto* bit_cast = LL_BUILDER.CreateBitCast(
      lit_buff_lv, llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0));
  auto* gep =
      LL_BUILDER.CreateGEP(bit_cast->getType()->getScalarType()->getPointerElementType(),
                           bit_cast,
                           LL_INT(off));
  return LL_BUILDER.CreateLoad(gep->getType()->getPointerElementType(), gep);
}

std::vector<llvm::Value*> GroupByAndAggregate::codegenAggArg(
    const Analyzer::Expr* target_expr,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
  const auto func_expr = dynamic_cast<const Analyzer::FunctionOper*>(target_expr);
  const auto arr_expr = dynamic_cast<const Analyzer::ArrayExpr*>(target_expr);

  // TODO(alex): handle arrays uniformly?
  CodeGenerator code_generator(executor_);
  if (target_expr) {
    const auto& target_ti = target_expr->get_type_info();
    if (target_ti.is_buffer() &&
        !executor_->plan_state_->isLazyFetchColumn(target_expr)) {
      const auto target_lvs =
          agg_expr ? code_generator.codegen(agg_expr->get_arg(), true, co)
                   : code_generator.codegen(
                         target_expr, !executor_->plan_state_->allow_lazy_fetch_, co);
      if (!func_expr && !arr_expr) {
        // Something with the chunk transport is code that was generated from a source
        // other than an ARRAY[] expression
        if (target_ti.is_bytes()) {
          CHECK_EQ(size_t(3), target_lvs.size());
          return {target_lvs[1], target_lvs[2]};
        }
        CHECK(target_ti.is_array());
        CHECK_EQ(size_t(1), target_lvs.size());
        CHECK(!agg_expr || agg_expr->get_aggtype() == kSAMPLE);
        const auto i32_ty = get_int_type(32, executor_->cgen_state_->context_);
        const auto i8p_ty =
            llvm::PointerType::get(get_int_type(8, executor_->cgen_state_->context_), 0);
        const auto& elem_ti = target_ti.get_elem_type();
        return {
            executor_->cgen_state_->emitExternalCall(
                "array_buff",
                i8p_ty,
                {target_lvs.front(), code_generator.posArg(target_expr)}),
            executor_->cgen_state_->emitExternalCall(
                "array_size",
                i32_ty,
                {target_lvs.front(),
                 code_generator.posArg(target_expr),
                 executor_->cgen_state_->llInt(log2_bytes(elem_ti.get_logical_size()))})};
      } else {
        if (agg_expr) {
          throw std::runtime_error(
              "Using array[] operator as argument to an aggregate operator is not "
              "supported");
        }
        CHECK(func_expr || arr_expr);
        if (dynamic_cast<const Analyzer::FunctionOper*>(target_expr)) {
          CHECK_EQ(size_t(1), target_lvs.size());
          const auto prefix = target_ti.get_buffer_name();
          CHECK(target_ti.is_array() || target_ti.is_bytes());
          const auto target_lv = LL_BUILDER.CreateLoad(
              target_lvs[0]->getType()->getPointerElementType(), target_lvs[0]);
          // const auto target_lv_type = target_lvs[0]->getType();
          // CHECK(target_lv_type->isStructTy());
          // CHECK_EQ(target_lv_type->getNumContainedTypes(), 3u);
          const auto i8p_ty = llvm::PointerType::get(
              get_int_type(8, executor_->cgen_state_->context_), 0);
          const auto ptr = LL_BUILDER.CreatePointerCast(
              LL_BUILDER.CreateExtractValue(target_lv, 0), i8p_ty);
          const auto size = LL_BUILDER.CreateExtractValue(target_lv, 1);
          const auto null_flag = LL_BUILDER.CreateExtractValue(target_lv, 2);
          const auto nullcheck_ok_bb =
              llvm::BasicBlock::Create(LL_CONTEXT, prefix + "_nullcheck_ok_bb", CUR_FUNC);
          const auto nullcheck_fail_bb = llvm::BasicBlock::Create(
              LL_CONTEXT, prefix + "_nullcheck_fail_bb", CUR_FUNC);

          // TODO(adb): probably better to zext the bool
          const auto nullcheck = LL_BUILDER.CreateICmpEQ(
              null_flag, executor_->cgen_state_->llInt(static_cast<int8_t>(1)));
          LL_BUILDER.CreateCondBr(nullcheck, nullcheck_fail_bb, nullcheck_ok_bb);

          const auto ret_bb =
              llvm::BasicBlock::Create(LL_CONTEXT, prefix + "_return", CUR_FUNC);
          LL_BUILDER.SetInsertPoint(ret_bb);
          auto result_phi = LL_BUILDER.CreatePHI(i8p_ty, 2, prefix + "_ptr_return");
          result_phi->addIncoming(ptr, nullcheck_ok_bb);
          const auto null_arr_sentinel = LL_BUILDER.CreateIntToPtr(
              executor_->cgen_state_->llInt(static_cast<int8_t>(0)), i8p_ty);
          result_phi->addIncoming(null_arr_sentinel, nullcheck_fail_bb);
          LL_BUILDER.SetInsertPoint(nullcheck_ok_bb);
          executor_->cgen_state_->emitExternalCall(
              "register_buffer_with_executor_rsm",
              llvm::Type::getVoidTy(executor_->cgen_state_->context_),
              {executor_->cgen_state_->llInt(reinterpret_cast<int64_t>(executor_)), ptr});
          LL_BUILDER.CreateBr(ret_bb);
          LL_BUILDER.SetInsertPoint(nullcheck_fail_bb);
          LL_BUILDER.CreateBr(ret_bb);

          LL_BUILDER.SetInsertPoint(ret_bb);
          return {result_phi, size};
        }
        CHECK_EQ(size_t(2), target_lvs.size());
        return {target_lvs[0], target_lvs[1]};
      }
    }
    if (target_ti.is_geometry() &&
        !executor_->plan_state_->isLazyFetchColumn(target_expr)) {
      auto generate_coord_lvs =
          [&](auto* selected_target_expr,
              bool const fetch_columns) -> std::vector<llvm::Value*> {
        const auto target_lvs =
            code_generator.codegen(selected_target_expr, fetch_columns, co);
        if (dynamic_cast<const Analyzer::GeoOperator*>(target_expr) &&
            target_expr->get_type_info().is_geometry()) {
          // return a pointer to the temporary alloca
          return target_lvs;
        }
        const auto geo_uoper = dynamic_cast<const Analyzer::GeoUOper*>(target_expr);
        const auto geo_binoper = dynamic_cast<const Analyzer::GeoBinOper*>(target_expr);
        if (geo_uoper || geo_binoper) {
          CHECK(target_expr->get_type_info().is_geometry());
          CHECK_EQ(2 * static_cast<size_t>(target_ti.get_physical_coord_cols()),
                   target_lvs.size());
          return target_lvs;
        }
        CHECK_EQ(static_cast<size_t>(target_ti.get_physical_coord_cols()),
                 target_lvs.size());

        const auto i32_ty = get_int_type(32, executor_->cgen_state_->context_);
        const auto i8p_ty =
            llvm::PointerType::get(get_int_type(8, executor_->cgen_state_->context_), 0);
        std::vector<llvm::Value*> coords;
        size_t ctr = 0;
        for (const auto& target_lv : target_lvs) {
          // TODO(adb): consider adding a utility to sqltypes so we can get the types of
          // the physical coords cols based on the sqltype (e.g. TINYINT for col 0, INT
          // for col 1 for pols / mpolys, etc). Hardcoding for now. first array is the
          // coords array (TINYINT). Subsequent arrays are regular INT.

          const size_t elem_sz = ctr == 0 ? 1 : 4;
          ctr++;
          int32_t fixlen = -1;
          if (target_ti.get_type() == kPOINT) {
            const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
            if (col_var) {
              const auto coords_cd = executor_->getPhysicalColumnDescriptor(col_var, 1);
              if (coords_cd && coords_cd->columnType.get_type() == kARRAY) {
                fixlen = coords_cd->columnType.get_size();
              }
            }
          }
          if (fixlen > 0) {
            coords.push_back(executor_->cgen_state_->emitExternalCall(
                "fast_fixlen_array_buff",
                i8p_ty,
                {target_lv, code_generator.posArg(selected_target_expr)}));
            coords.push_back(executor_->cgen_state_->llInt(int64_t(fixlen)));
            continue;
          }
          coords.push_back(executor_->cgen_state_->emitExternalCall(
              "array_buff",
              i8p_ty,
              {target_lv, code_generator.posArg(selected_target_expr)}));
          coords.push_back(executor_->cgen_state_->emitExternalCall(
              "array_size",
              i32_ty,
              {target_lv,
               code_generator.posArg(selected_target_expr),
               executor_->cgen_state_->llInt(log2_bytes(elem_sz))}));
        }
        return coords;
      };

      if (agg_expr) {
        return generate_coord_lvs(agg_expr->get_arg(), true);
      } else {
        return generate_coord_lvs(target_expr,
                                  !executor_->plan_state_->allow_lazy_fetch_);
      }
    }
  }
  bool fetch_column = !executor_->plan_state_->allow_lazy_fetch_;
  auto cv = dynamic_cast<Analyzer::ColumnVar const*>(target_expr);
  if (cv && !fetch_column &&
      executor_->plan_state_->force_non_lazy_fetch_columns_.count(cv->getColumnKey())) {
    VLOG(2) << "Force to fetch the target expression: " << cv->toString();
    fetch_column = true;
  }
  return agg_expr ? code_generator.codegen(agg_expr->get_arg(), true, co)
                  : code_generator.codegen(target_expr, fetch_column, co);
}

llvm::Value* GroupByAndAggregate::emitCall(const std::string& fname,
                                           const std::vector<llvm::Value*>& args) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  return executor_->cgen_state_->emitCall(fname, args);
}

void GroupByAndAggregate::checkErrorCode(llvm::Value* retCode) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  auto zero_const = llvm::ConstantInt::get(retCode->getType(), 0, true);
  auto rc_check_condition = executor_->cgen_state_->ir_builder_.CreateICmp(
      llvm::ICmpInst::ICMP_EQ, retCode, zero_const);

  executor_->cgen_state_->emitErrorCheck(rc_check_condition, retCode, "rc");
}

#undef CUR_FUNC
#undef ROW_FUNC
#undef LL_FP
#undef LL_INT
#undef LL_BOOL
#undef LL_BUILDER
#undef LL_CONTEXT

size_t GroupByAndAggregate::shard_count_for_top_groups(
    const RelAlgExecutionUnit& ra_exe_unit) {
  if (ra_exe_unit.sort_info.order_entries.size() != 1 || !ra_exe_unit.sort_info.limit) {
    return 0;
  }
  for (const auto& group_expr : ra_exe_unit.groupby_exprs) {
    const auto grouped_col_expr =
        dynamic_cast<const Analyzer::ColumnVar*>(group_expr.get());
    if (!grouped_col_expr) {
      continue;
    }
    const auto& column_key = grouped_col_expr->getColumnKey();
    if (column_key.table_id <= 0) {
      return 0;
    }
    const auto td = Catalog_Namespace::get_metadata_for_table(
        {column_key.db_id, column_key.table_id});
    if (td->shardedColumnId == column_key.column_id) {
      return td->nShards;
    }
  }
  return 0;
}
