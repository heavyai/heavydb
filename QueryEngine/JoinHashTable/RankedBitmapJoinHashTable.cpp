/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/JoinHashTable/RankedBitmapJoinHashTable.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <llvm/IR/Intrinsics.h>
#include "CudaMgr/CudaMgr.h"
#include "DataMgr/DataMgr.h"

#include "DataMgr/Allocators/CudaAllocator.h"
#include "DataMgr/ChunkMetadata.h"
#include "DataMgr/Encoder.h"
#include "Logger/Logger.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/DataRecycler/HashtableRecycler.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRewrite.h"
#ifdef HAVE_CUDA
#include "QueryEngine/GpuInitGroups.h"
#endif
#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"
#include "QueryEngine/JoinHashTable/RankedBitmapHashTable.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryEngine/QueryPlanDagCache.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "QueryEngine/enums.h"
#include "Shared/InlineNullValues.h"
#include "StringDictionary/StringDictionary.h"

std::unique_ptr<HashtableRecycler> RankedBitmapJoinHashTable::hash_table_cache_ =
    std::make_unique<HashtableRecycler>(CacheItemType::RANKED_BITMAP_HT,
                                        DataRecyclerUtil::MAX_GPU_CACHE_DEVICE_COUNT);

namespace {

DeviceIdentifier gpu_cache_device_identifier(const int device_id) {
  CHECK_GE(device_id, 0);
  return static_cast<DeviceIdentifier>(device_id + 1);
}

QueryPlanHash hash_ranked_bitmap_cache_key(const std::string& key) {
  auto hashed_key = boost::hash_value(key);
  if (hashed_key == EMPTY_HASHED_PLAN_DAG_KEY) {
    boost::hash_combine(hashed_key, key.size());
  }
  return hashed_key;
}

void mark_ranked_bitmap_cache_key_for_tables(
    const QueryPlanHash key,
    const std::unordered_set<size_t>& table_keys) {
  if (key == EMPTY_HASHED_PLAN_DAG_KEY || table_keys.empty()) {
    return;
  }
  RankedBitmapJoinHashTable::getHashTableCache()->addQueryPlanDagForTableKeys(key,
                                                                              table_keys);
}

bool supported_ranked_bitmap_type(const SQLTypeInfo& ti) {
  return (ti.is_integer() || ti.is_time() || ti.is_boolean()) && !ti.is_string() &&
         !ti.is_array() && !ti.is_fp();
}

struct FragmentKeyRange {
  int64_t min;
  int64_t max;
};

bool fragment_key_ranges_overlap(const FragmentKeyRange& lhs,
                                 const FragmentKeyRange& rhs) {
  return lhs.min <= rhs.max && rhs.min <= lhs.max;
}

#ifdef HAVE_CUDA
std::optional<std::shared_ptr<ChunkMetadata>>
synthesize_device_columnar_fragment_column_metadata(const ResultSet* rows,
                                                    const int fragment_id,
                                                    const int column_id) {
  if (!rows || fragment_id < 0 || column_id < 0 ||
      static_cast<size_t>(column_id) >= rows->colCount() || !rows->didOutputColumnar() ||
      rows->areAnyColumnsLazyFetched() || rows->isTruncated()) {
    return std::nullopt;
  }
  const auto col_ti = rows->getColType(column_id);
  const auto logical_ti = get_logical_type_info(col_ti);
  if (!supported_ranked_bitmap_type(logical_ti)) {
    return std::nullopt;
  }
  const auto elem_size = logical_ti.get_size();
  if (elem_size <= 0 || logical_ti.is_varlen()) {
    return std::nullopt;
  }

  std::vector<ResultSet::DeviceColumnarBufferFragment> fragments;
  if (!rows->getDeviceColumnarBufferFragments(
          column_id, static_cast<size_t>(elem_size), fragments) ||
      static_cast<size_t>(fragment_id) >= fragments.size()) {
    return std::nullopt;
  }
  const auto& fragment = fragments[fragment_id];
  if (!fragment.entry_count) {
    return std::nullopt;
  }

  auto encoder = Encoder::Create(nullptr, col_ti);
  CHECK(encoder);
  const auto null_val = inline_int_null_val(col_ti);
  DeviceColumnFragmentStats stats;
  fragment.owner->waitForReadyEvent(fragment.ready_event);
  if (!compute_columnar_fragment_int_stats_on_device(fragment.buffer,
                                                     fragment.entry_count,
                                                     static_cast<size_t>(elem_size),
                                                     null_val,
                                                     fragment.device_id,
                                                     stats,
                                                     fragment.owner->getCudaStream())) {
    return std::nullopt;
  }
  if (stats.has_values) {
    encoder->updateStats(stats.int_min, false);
    encoder->updateStats(stats.int_max, false);
  }
  if (stats.has_nulls) {
    encoder->updateStats(null_val, true);
  }

  auto chunk_metadata = std::make_shared<ChunkMetadata>();
  chunk_metadata->sqlType = col_ti;
  chunk_metadata->numElements = fragment.entry_count;
  chunk_metadata->numBytes = fragment.entry_count * static_cast<size_t>(elem_size);
  chunk_metadata->chunkStats = encoder->synthesizeChunkStats(col_ti);
  if (!stats.has_values) {
    chunk_metadata->chunkStats.has_nulls = stats.has_nulls;
  }
  return chunk_metadata;
}
#endif

std::shared_ptr<ChunkMetadata> get_fragment_column_metadata(
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const int column_id) {
  if (!fragment.resultSet) {
    const auto& metadata_map = fragment.getChunkMetadataMap();
    const auto metadata_it = metadata_map.find(column_id);
    return metadata_it == metadata_map.end() ? nullptr : metadata_it->second;
  }

  std::unique_ptr<std::lock_guard<std::mutex>> lock;
  if (fragment.resultSetMutex) {
    lock = std::make_unique<std::lock_guard<std::mutex>>(*fragment.resultSetMutex);
  }
  auto metadata_it = fragment.getChunkMetadataMapPhysical().find(column_id);
  if (metadata_it != fragment.getChunkMetadataMapPhysical().end()) {
    return metadata_it->second;
  }
#ifdef HAVE_CUDA
  auto metadata = synthesize_device_columnar_fragment_column_metadata(
      fragment.resultSet, fragment.fragmentId, column_id);
  if (!metadata) {
    return nullptr;
  }
  auto& mutable_fragment = const_cast<Fragmenter_Namespace::FragmentInfo&>(fragment);
  mutable_fragment.setChunkMetadata(column_id, *metadata);
  return *metadata;
#else
  return nullptr;
#endif
}

std::optional<FragmentKeyRange> get_fragment_key_range(
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const int column_id) {
  if (fragment.isEmptyPhysicalFragment() || !fragment.getNumTuples()) {
    return std::nullopt;
  }
  const auto metadata_ptr = get_fragment_column_metadata(fragment, column_id);
  if (!metadata_ptr) {
    return std::nullopt;
  }
  const auto& metadata = *metadata_ptr;
  if (metadata.isPlaceholder() || metadata.chunkStats.has_nulls ||
      !supported_ranked_bitmap_type(metadata.sqlType)) {
    return std::nullopt;
  }
  const auto min = extract_min_stat_int_type(metadata.chunkStats, metadata.sqlType);
  const auto max = extract_max_stat_int_type(metadata.chunkStats, metadata.sqlType);
  if (min > max) {
    return std::nullopt;
  }
  return FragmentKeyRange{min, max};
}

const Analyzer::ColumnVar* as_simple_column(const Analyzer::Expr* expr) {
  return dynamic_cast<const Analyzer::ColumnVar*>(expr);
}

const InputTableInfo* find_query_info(const shared::TableKey& table_key,
                                      const std::vector<InputTableInfo>& query_infos) {
  const auto info_it =
      std::find_if(query_infos.begin(), query_infos.end(), [&](const auto& info) {
        return info.table_key == table_key;
      });
  return info_it == query_infos.end() ? nullptr : &*info_it;
}

std::optional<std::vector<Fragmenter_Namespace::FragmentInfo>>
range_pruned_ranked_bitmap_fragments_for_device(
    const Analyzer::ColumnVar* inner_col,
    const Analyzer::Expr* outer_expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& inner_fragments,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) {
  CHECK(inner_col);
  CHECK(outer_expr);
  if (memory_level != Data_Namespace::GPU_LEVEL || inner_fragments.size() <= size_t(1)) {
    return std::nullopt;
  }
  if (inner_col->getTableKey().table_id >= 0 &&
      std::any_of(
          inner_fragments.begin(), inner_fragments.end(), [&](const auto& fragment) {
            return fragment.physicalTableId != inner_col->getTableKey().table_id;
          })) {
    return std::nullopt;
  }
  const auto outer_col = as_simple_column(outer_expr);
  if (!outer_col || outer_col->getTableKey() == inner_col->getTableKey() ||
      !supported_ranked_bitmap_type(outer_col->get_type_info())) {
    return std::nullopt;
  }
  const auto outer_info = find_query_info(outer_col->getTableKey(), query_infos);
  if (!outer_info || outer_info->info.fragments.empty()) {
    return std::nullopt;
  }

  std::vector<FragmentKeyRange> outer_ranges;
  for (const auto& outer_fragment : outer_info->info.fragments) {
    if (outer_fragment.deviceIds.size() <=
            static_cast<size_t>(Data_Namespace::GPU_LEVEL) ||
        outer_fragment.deviceIds[Data_Namespace::GPU_LEVEL] != device_id) {
      continue;
    }
    const auto range =
        get_fragment_key_range(outer_fragment, outer_col->getColumnKey().column_id);
    if (!range) {
      return std::nullopt;
    }
    outer_ranges.push_back(*range);
  }
  if (outer_ranges.empty()) {
    return std::nullopt;
  }

  std::vector<Fragmenter_Namespace::FragmentInfo> pruned_fragments;
  pruned_fragments.reserve(inner_fragments.size());
  for (const auto& inner_fragment : inner_fragments) {
    const auto inner_range =
        get_fragment_key_range(inner_fragment, inner_col->getColumnKey().column_id);
    if (!inner_range) {
      return std::nullopt;
    }
    const bool overlaps_outer_range =
        std::any_of(outer_ranges.begin(), outer_ranges.end(), [&](const auto& range) {
          return fragment_key_ranges_overlap(*inner_range, range);
        });
    if (overlaps_outer_range) {
      pruned_fragments.push_back(inner_fragment);
    }
  }
  if (pruned_fragments.empty() || pruned_fragments.size() == inner_fragments.size()) {
    return std::nullopt;
  }

  return pruned_fragments;
}

bool may_range_prune_ranked_bitmap_build(const Analyzer::ColumnVar* inner_col,
                                         const Analyzer::Expr* outer_expr,
                                         const std::vector<InputTableInfo>& query_infos,
                                         const Data_Namespace::MemoryLevel memory_level) {
  CHECK(inner_col);
  CHECK(outer_expr);
  if (memory_level != Data_Namespace::GPU_LEVEL) {
    return false;
  }
  const auto outer_col = as_simple_column(outer_expr);
  if (!outer_col || outer_col->getTableKey() == inner_col->getTableKey()) {
    return false;
  }
  const auto inner_info = find_query_info(inner_col->getTableKey(), query_infos);
  if (!inner_info || inner_info->info.fragments.size() <= size_t(1)) {
    return false;
  }
  return inner_col->getTableKey().table_id < 0 ||
         std::none_of(inner_info->info.fragments.begin(),
                      inner_info->info.fragments.end(),
                      [&](const auto& fragment) {
                        return fragment.physicalTableId !=
                               inner_col->getTableKey().table_id;
                      });
}

size_t checked_bit_count_for_ranked_bitmap(const ExpressionRange& range) {
  if (range.getIntMin() > range.getIntMax()) {
    return 0;
  }
  const auto bit_count = static_cast<__int128>(range.getIntMax()) -
                         static_cast<__int128>(range.getIntMin()) + 1;
  if (bit_count < 0) {
    throw TooManyHashEntries("Ranked bitmap join range is invalid");
  }
  if (bit_count > std::numeric_limits<size_t>::max()) {
    throw TooManyHashEntries("Ranked bitmap join range exceeds addressable host size");
  }
  return static_cast<size_t>(bit_count);
}

size_t checked_ranked_bitmap_bytes(const size_t bit_count, const size_t payload_count) {
  const auto bitmap_words = RankedBitmapHashTable::wordsForBits(bit_count);
  const auto rank_blocks = RankedBitmapHashTable::blocksForWords(bitmap_words);
  const auto total_words = static_cast<__int128>(bitmap_words) + rank_blocks +
                           static_cast<__int128>(payload_count);
  const auto total_bytes = total_words * sizeof(uint32_t);
  if (total_bytes > std::numeric_limits<size_t>::max()) {
    throw TooManyHashEntries("Ranked bitmap join table exceeds addressable host size");
  }
  return static_cast<size_t>(total_bytes);
}

void check_ranked_bitmap_size(const size_t table_bytes,
                              const size_t tuple_count,
                              const size_t key_width) {
  const size_t baseline_entry_width = 2 * key_width;
  const auto baseline_size_estimate =
      tuple_count > std::numeric_limits<size_t>::max() / (2 * baseline_entry_width)
          ? std::numeric_limits<size_t>::max()
          : 2 * tuple_count * baseline_entry_width;
  if (table_bytes >= baseline_size_estimate) {
    std::ostringstream oss;
    oss << "Ranked bitmap join table is not smaller than baseline (# ranked bytes: "
        << table_bytes << ", estimated baseline bytes: " << baseline_size_estimate
        << ", # input rows: " << tuple_count << ")";
    throw TooManyHashEntries(oss.str());
  }
}

bool supported_ranked_bitmap_filter_type(const SQLTypeInfo& ti) {
  return supported_ranked_bitmap_type(ti);
}

std::optional<int64_t> constant_to_ranked_bitmap_filter_value(
    const Analyzer::Constant* constant,
    const SQLTypeInfo& column_ti) {
  if (!constant || constant->get_is_null() ||
      !supported_ranked_bitmap_filter_type(column_ti)) {
    return std::nullopt;
  }
  const auto& constant_ti = constant->get_type_info();
  if (constant_ti.is_decimal() && constant_ti.get_scale() != 0) {
    return std::nullopt;
  }
  if (!(constant_ti.is_integer() || constant_ti.is_time() || constant_ti.is_boolean() ||
        (constant_ti.is_decimal() && constant_ti.get_scale() == 0))) {
    return std::nullopt;
  }
  if ((column_ti.is_time() || constant_ti.is_time()) &&
      (column_ti.get_type() != constant_ti.get_type() ||
       column_ti.get_dimension() != constant_ti.get_dimension())) {
    return std::nullopt;
  }
  return get_value_from_datum<int64_t>(constant->get_constval(), constant_ti.get_type());
}

struct BuildSideFilterCandidate {
  std::shared_ptr<Analyzer::ColumnVar> column;
  SQLOps op;
  int64_t value;
  std::shared_ptr<Analyzer::Expr> qual;
};

std::optional<BuildSideFilterCandidate> build_side_filter_candidate(
    const std::shared_ptr<Analyzer::Expr>& qual,
    const Analyzer::ColumnVar* inner_col) {
  CHECK(inner_col);
  const auto bin_oper = dynamic_cast<const Analyzer::BinOper*>(qual.get());
  if (!bin_oper || bin_oper->get_qualifier() != kONE) {
    return std::nullopt;
  }
  auto op = bin_oper->get_optype();
  if (op != kEQ && op != kLT && op != kLE && op != kGT && op != kGE) {
    return std::nullopt;
  }

  const Analyzer::ColumnVar* column =
      dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_left_operand());
  const Analyzer::Constant* constant =
      dynamic_cast<const Analyzer::Constant*>(bin_oper->get_right_operand());
  if (!column || !constant) {
    column = dynamic_cast<const Analyzer::ColumnVar*>(bin_oper->get_right_operand());
    constant = dynamic_cast<const Analyzer::Constant*>(bin_oper->get_left_operand());
    if (!column || !constant) {
      return std::nullopt;
    }
    op = COMMUTE_COMPARISON(op);
  }

  if (column->getTableKey() != inner_col->getTableKey() ||
      column->get_rte_idx() != inner_col->get_rte_idx()) {
    return std::nullopt;
  }
  const auto value =
      constant_to_ranked_bitmap_filter_value(constant, column->get_type_info());
  if (!value) {
    return std::nullopt;
  }
  return BuildSideFilterCandidate{
      std::dynamic_pointer_cast<Analyzer::ColumnVar>(column->deep_copy()),
      op,
      *value,
      qual};
}

bool same_build_side_filter_column(const Analyzer::ColumnVar& lhs,
                                   const Analyzer::ColumnVar& rhs) {
  return lhs.getColumnKey() == rhs.getColumnKey() &&
         lhs.get_rte_idx() == rhs.get_rte_idx() &&
         lhs.get_type_info() == rhs.get_type_info();
}

bool update_lower_bound(RankedBitmapJoinHashTable::BuildSideFilter& filter,
                        const int64_t value,
                        const bool inclusive) {
  if (!filter.has_lower_bound || value > filter.lower_bound ||
      (value == filter.lower_bound && !inclusive && filter.lower_bound_inclusive)) {
    filter.has_lower_bound = true;
    filter.lower_bound = value;
    filter.lower_bound_inclusive = inclusive;
  }
  return true;
}

bool update_upper_bound(RankedBitmapJoinHashTable::BuildSideFilter& filter,
                        const int64_t value,
                        const bool inclusive) {
  if (!filter.has_upper_bound || value < filter.upper_bound ||
      (value == filter.upper_bound && !inclusive && filter.upper_bound_inclusive)) {
    filter.has_upper_bound = true;
    filter.upper_bound = value;
    filter.upper_bound_inclusive = inclusive;
  }
  return true;
}

bool apply_build_side_filter_candidate(RankedBitmapJoinHashTable::BuildSideFilter& filter,
                                       const BuildSideFilterCandidate& candidate) {
  switch (candidate.op) {
    case kEQ:
      update_lower_bound(filter, candidate.value, true);
      update_upper_bound(filter, candidate.value, true);
      return true;
    case kGT:
      return update_lower_bound(filter, candidate.value, false);
    case kGE:
      return update_lower_bound(filter, candidate.value, true);
    case kLT:
      return update_upper_bound(filter, candidate.value, false);
    case kLE:
      return update_upper_bound(filter, candidate.value, true);
    default:
      return false;
  }
}

bool build_side_filter_is_contradictory(
    const RankedBitmapJoinHashTable::BuildSideFilter& filter) {
  if (!filter.has_lower_bound || !filter.has_upper_bound) {
    return false;
  }
  if (filter.lower_bound > filter.upper_bound) {
    return true;
  }
  return filter.lower_bound == filter.upper_bound &&
         (!filter.lower_bound_inclusive || !filter.upper_bound_inclusive);
}

std::optional<RankedBitmapJoinHashTable::BuildSideFilter>
build_ranked_bitmap_build_side_filter(
    const Analyzer::ColumnVar* inner_col,
    const std::list<std::shared_ptr<Analyzer::Expr>>& build_side_quals,
    std::vector<std::shared_ptr<Analyzer::Expr>>& pushed_down_build_quals) {
  std::optional<RankedBitmapJoinHashTable::BuildSideFilter> filter;
  for (const auto& qual : build_side_quals) {
    auto candidate = build_side_filter_candidate(qual, inner_col);
    if (!candidate) {
      continue;
    }
    if (!filter) {
      filter = RankedBitmapJoinHashTable::BuildSideFilter{candidate->column};
    } else if (!same_build_side_filter_column(*filter->column, *candidate->column)) {
      continue;
    }
    auto updated_filter = *filter;
    if (!apply_build_side_filter_candidate(updated_filter, *candidate) ||
        build_side_filter_is_contradictory(updated_filter)) {
      continue;
    }
    filter = std::move(updated_filter);
    pushed_down_build_quals.push_back(candidate->qual);
  }
  return filter;
}

std::string ranked_bitmap_filter_cache_key(
    const std::optional<RankedBitmapJoinHashTable::BuildSideFilter>& build_side_filter) {
  if (!build_side_filter) {
    return "|filter=none";
  }
  const auto filter_key = build_side_filter->column->getColumnKey();
  std::ostringstream oss;
  oss << "|filter=db:" << filter_key.db_id << ",table:" << filter_key.table_id
      << ",col:" << filter_key.column_id
      << ",rte:" << build_side_filter->column->get_rte_idx()
      << ",lower_enabled:" << build_side_filter->has_lower_bound
      << ",lower_inclusive:" << build_side_filter->lower_bound_inclusive
      << ",lower:" << build_side_filter->lower_bound
      << ",upper_enabled:" << build_side_filter->has_upper_bound
      << ",upper_inclusive:" << build_side_filter->upper_bound_inclusive
      << ",upper:" << build_side_filter->upper_bound;
  return oss.str();
}

std::string ranked_bitmap_temp_build_cache_key(
    const Analyzer::ColumnVar* inner_col,
    const ExpressionRange& col_range,
    const size_t bit_count,
    const size_t payload_count,
    const Data_Namespace::MemoryLevel memory_level,
    const std::set<int>& device_ids,
    const std::optional<RankedBitmapJoinHashTable::BuildSideFilter>& build_side_filter) {
  CHECK(inner_col);
  const auto column_key = inner_col->getColumnKey();
  if (column_key.table_id >= 0) {
    return {};
  }

  std::ostringstream oss;
  oss << "ranked-bitmap-temp:v1"
      << "|db=" << column_key.db_id << "|table=" << column_key.table_id
      << "|col=" << column_key.column_id << "|range_min=" << col_range.getIntMin()
      << "|range_max=" << col_range.getIntMax() << "|bits=" << bit_count
      << "|payload=" << payload_count << "|memory=" << static_cast<int>(memory_level)
      << ranked_bitmap_filter_cache_key(build_side_filter) << "|devices=";
  for (const auto device_id : device_ids) {
    oss << device_id << ",";
  }
  return oss.str();
}

std::string ranked_bitmap_temp_source_semantic_key(
    const RelAlgNode* source_node,
    const Analyzer::ColumnVar* inner_col,
    const Analyzer::Expr* outer_expr,
    const ExpressionRange& col_range,
    size_t bit_count,
    size_t payload_count,
    Data_Namespace::MemoryLevel memory_level,
    const std::set<int>& device_ids,
    SQLOps op_type,
    JoinType join_type,
    Executor* executor,
    const std::optional<RankedBitmapJoinHashTable::BuildSideFilter>& build_side_filter,
    std::unordered_set<size_t>* table_keys);

std::string ranked_bitmap_semantic_build_cache_key(
    const Analyzer::ColumnVar* inner_col,
    const Analyzer::Expr* outer_expr,
    const ExpressionRange& col_range,
    const size_t bit_count,
    const size_t payload_count,
    const Data_Namespace::MemoryLevel memory_level,
    const std::set<int>& device_ids,
    const SQLOps op_type,
    const JoinType join_type,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const TableIdToNodeMap& table_id_to_node_map,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    Executor* executor,
    const std::optional<RankedBitmapJoinHashTable::BuildSideFilter>& build_side_filter,
    std::unordered_set<size_t>* table_keys) {
  CHECK(inner_col);
  CHECK(outer_expr);
  CHECK(executor);
  if (!HashtableRecycler::isSafeToCacheHashtable(
          table_id_to_node_map, false, {}, inner_col->getTableKey())) {
    const auto source_node =
        get_temporary_table_source_node(inner_col->getTableKey(), table_id_to_node_map);
    if (source_node) {
      auto source_key = ranked_bitmap_temp_source_semantic_key(source_node,
                                                               inner_col,
                                                               outer_expr,
                                                               col_range,
                                                               bit_count,
                                                               payload_count,
                                                               memory_level,
                                                               device_ids,
                                                               op_type,
                                                               join_type,
                                                               executor,
                                                               build_side_filter,
                                                               table_keys);
      if (!source_key.empty()) {
        return source_key;
      }
    }
    return {};
  }

  std::unordered_map<int, std::vector<Fragmenter_Namespace::FragmentInfo>>
      frags_for_device;
  for (const auto device_id : device_ids) {
    frags_for_device.emplace(device_id, fragments);
  }
  const std::vector<InnerOuter> inner_outer_pairs{{inner_col, outer_expr}};
  const auto access_path_info =
      HashtableRecycler::getHashtableAccessPathInfo(inner_outer_pairs,
                                                    {},
                                                    op_type,
                                                    join_type,
                                                    hashtable_build_dag_map,
                                                    device_ids,
                                                    0,
                                                    frags_for_device,
                                                    executor);
  if (HashtableRecycler::isInvalidHashTableCacheKey(
          access_path_info.hashed_query_plan_dag)) {
    const auto source_node =
        get_temporary_table_source_node(inner_col->getTableKey(), table_id_to_node_map);
    if (source_node) {
      auto source_key = ranked_bitmap_temp_source_semantic_key(source_node,
                                                               inner_col,
                                                               outer_expr,
                                                               col_range,
                                                               bit_count,
                                                               payload_count,
                                                               memory_level,
                                                               device_ids,
                                                               op_type,
                                                               join_type,
                                                               executor,
                                                               build_side_filter,
                                                               table_keys);
      if (!source_key.empty()) {
        return source_key;
      }
    }
    if (!source_node || source_node->getQueryPlanDagHash() == EMPTY_HASHED_PLAN_DAG_KEY) {
      return {};
    }
    auto outer_col_info = executor->getQueryPlanDagCache().getJoinColumnsInfoHash(
        outer_expr, JoinColumnSide::kDirect, false);
    if (outer_col_info == EMPTY_HASHED_PLAN_DAG_KEY) {
      return {};
    }
    std::ostringstream oss;
    oss << "ranked-bitmap-temp-source-semantic:v1"
        << "|source_plan=" << source_node->getQueryPlanDagHash()
        << "|source_col=" << inner_col->getColumnKey().column_id
        << "|source_type=" << inner_col->get_type_info().toString()
        << "|outer_col_info=" << outer_col_info
        << "|outer_type=" << outer_expr->get_type_info().toString()
        << "|op=" << static_cast<int>(op_type) << "|join=" << static_cast<int>(join_type)
        << "|range_min=" << col_range.getIntMin()
        << "|range_max=" << col_range.getIntMax() << "|bits=" << bit_count
        << "|payload=" << payload_count << "|memory=" << static_cast<int>(memory_level)
        << ranked_bitmap_filter_cache_key(build_side_filter) << "|devices=";
    for (const auto device_id : device_ids) {
      oss << device_id << ",";
    }
    return oss.str();
  }
  if (table_keys) {
    *table_keys = access_path_info.table_keys;
  }

  std::ostringstream oss;
  oss << "ranked-bitmap-semantic:v1";
  for (const auto device_id : device_ids) {
    oss << "|device=" << device_id
        << ":plan=" << access_path_info.hashed_query_plan_dag.at(device_id);
  }
  oss << "|range_min=" << col_range.getIntMin() << "|range_max=" << col_range.getIntMax()
      << "|bits=" << bit_count << "|payload=" << payload_count
      << "|memory=" << static_cast<int>(memory_level)
      << ranked_bitmap_filter_cache_key(build_side_filter) << "|devices=";
  for (const auto device_id : device_ids) {
    oss << device_id << ",";
  }
  return oss.str();
}

struct RankedBitmapCacheKey {
  std::string key;
  std::string source;
  std::unordered_set<size_t> table_keys;
  bool globally_recyclable{false};
};

struct RankedBitmapRangePruningCacheDependency {
  std::string key;
  std::unordered_set<size_t> table_keys;
};

std::optional<RankedBitmapRangePruningCacheDependency>
ranked_bitmap_range_pruning_cache_dependency(
    const Analyzer::Expr* outer_expr,
    const TableIdToNodeMap& table_id_to_node_map) {
  const auto outer_col = as_simple_column(outer_expr);
  if (!outer_col) {
    return std::nullopt;
  }

  const auto outer_table_key = outer_col->getTableKey();
  std::ostringstream oss;
  std::unordered_set<size_t> table_keys;
  if (outer_table_key.table_id >= 0) {
    oss << "|range-pruned-outer=stored:db:" << outer_table_key.db_id
        << ",table:" << outer_table_key.table_id
        << ",col:" << outer_col->getColumnKey().column_id;
    table_keys.insert(boost::hash_value(
        std::vector<int>{outer_table_key.db_id, outer_table_key.table_id}));
  } else {
    const auto source_node =
        get_temporary_table_source_node(outer_table_key, table_id_to_node_map);
    if (!source_node || source_node->getQueryPlanDagHash() == EMPTY_HASHED_PLAN_DAG_KEY) {
      return std::nullopt;
    }
    oss << "|range-pruned-outer=temp:plan:" << source_node->getQueryPlanDagHash()
        << ",col:" << outer_col->getColumnKey().column_id
        << ",type:" << outer_col->get_type_info().toString();
    table_keys = ScanNodeTableKeyCollector::getScanNodeTableKey(source_node);
  }
  return RankedBitmapRangePruningCacheDependency{oss.str(), std::move(table_keys)};
}

std::string ranked_bitmap_temp_source_semantic_key(
    const RelAlgNode* source_node,
    const Analyzer::ColumnVar* inner_col,
    const Analyzer::Expr* outer_expr,
    const ExpressionRange& col_range,
    const size_t bit_count,
    const size_t payload_count,
    const Data_Namespace::MemoryLevel memory_level,
    const std::set<int>& device_ids,
    const SQLOps op_type,
    const JoinType join_type,
    Executor* executor,
    const std::optional<RankedBitmapJoinHashTable::BuildSideFilter>& build_side_filter,
    std::unordered_set<size_t>* table_keys) {
  CHECK(source_node);
  CHECK(inner_col);
  CHECK(outer_expr);
  CHECK(executor);
  if (dynamic_cast<const RelSort*>(source_node) ||
      source_node->getQueryPlanDagHash() == EMPTY_HASHED_PLAN_DAG_KEY) {
    return {};
  }
  auto input_table_keys = ScanNodeTableKeyCollector::getScanNodeTableKey(source_node);
  if (input_table_keys.empty()) {
    return {};
  }
  auto outer_col_info = executor->getQueryPlanDagCache().getJoinColumnsInfoHash(
      outer_expr, JoinColumnSide::kDirect, false);
  if (outer_col_info == EMPTY_HASHED_PLAN_DAG_KEY) {
    return {};
  }
  if (table_keys) {
    *table_keys = std::move(input_table_keys);
  }

  std::ostringstream oss;
  oss << "ranked-bitmap-temp-source-semantic:v2"
      << "|source_plan=" << source_node->getQueryPlanDagHash()
      << "|source_col=" << inner_col->getColumnKey().column_id
      << "|source_type=" << inner_col->get_type_info().toString()
      << "|outer_col_info=" << outer_col_info
      << "|outer_type=" << outer_expr->get_type_info().toString()
      << "|op=" << static_cast<int>(op_type) << "|join=" << static_cast<int>(join_type)
      << "|range_min=" << col_range.getIntMin() << "|range_max=" << col_range.getIntMax()
      << "|bits=" << bit_count << "|payload=" << payload_count
      << "|memory=" << static_cast<int>(memory_level)
      << ranked_bitmap_filter_cache_key(build_side_filter) << "|devices=";
  for (const auto device_id : device_ids) {
    oss << device_id << ",";
  }
  return oss.str();
}

RankedBitmapCacheKey ranked_bitmap_build_cache_key(
    const Analyzer::ColumnVar* inner_col,
    const Analyzer::Expr* outer_expr,
    const ExpressionRange& col_range,
    const size_t bit_count,
    const size_t payload_count,
    const Data_Namespace::MemoryLevel memory_level,
    const std::set<int>& device_ids,
    const SQLOps op_type,
    const JoinType join_type,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const TableIdToNodeMap& table_id_to_node_map,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    Executor* executor,
    const std::optional<RankedBitmapJoinHashTable::BuildSideFilter>& build_side_filter) {
  std::unordered_set<size_t> semantic_table_keys;
  auto semantic_key = ranked_bitmap_semantic_build_cache_key(inner_col,
                                                             outer_expr,
                                                             col_range,
                                                             bit_count,
                                                             payload_count,
                                                             memory_level,
                                                             device_ids,
                                                             op_type,
                                                             join_type,
                                                             hashtable_build_dag_map,
                                                             table_id_to_node_map,
                                                             fragments,
                                                             executor,
                                                             build_side_filter,
                                                             &semantic_table_keys);
  if (!semantic_key.empty()) {
    if (semantic_table_keys.empty() && inner_col->getTableKey().table_id > 0) {
      semantic_table_keys.insert(inner_col->getTableKey().hash());
    }
    const bool globally_recyclable = !semantic_table_keys.empty();
    return {std::move(semantic_key),
            "semantic",
            std::move(semantic_table_keys),
            globally_recyclable};
  }
  return {ranked_bitmap_temp_build_cache_key(inner_col,
                                             col_range,
                                             bit_count,
                                             payload_count,
                                             memory_level,
                                             device_ids,
                                             build_side_filter),
          "temporary-table",
          {},
          false};
}

void reuse_ranked_bitmap_hash_tables(
    RankedBitmapJoinHashTable& target,
    const std::shared_ptr<RankedBitmapJoinHashTable>& cached,
    const std::set<int>& device_ids) {
  CHECK(cached);
  for (const auto device_id : device_ids) {
    target.putHashTableForDevice(cached->getHashTableForDevice(device_id), device_id);
  }
}

llvm::Value* codegen_popcount32(CgenState* cgen_state, llvm::Value* val) {
  auto popcount_func = llvm::Intrinsic::getDeclaration(
      cgen_state->module_, llvm::Intrinsic::ctpop, {val->getType()});
  return cgen_state->ir_builder_.CreateCall(popcount_func, {val});
}

uint32_t host_popcount32(const uint32_t word) {
  return static_cast<uint32_t>(__builtin_popcount(word));
}

size_t ranked_bitmap_distinct_count_cpu(RankedBitmapHashTable& hash_table) {
  if (hash_table.getRankBlockCount() == 0) {
    return 0;
  }
  const auto last_block_idx = hash_table.getRankBlockCount() - 1;
  const auto block_start_word =
      last_block_idx * RankedBitmapHashTable::kRankBlockWordCount;
  auto rank_blocks = hash_table.getCpuRankBlocks();
  auto bitmap = hash_table.getCpuBitmap();
  uint64_t distinct_count = rank_blocks[last_block_idx];
  for (size_t word_idx = block_start_word; word_idx < hash_table.getBitmapWordCount();
       ++word_idx) {
    distinct_count += host_popcount32(bitmap[word_idx]);
  }
  CHECK_LE(distinct_count, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
  return static_cast<size_t>(distinct_count);
}

size_t exclusive_scan_counts_cpu(uint32_t* counts,
                                 uint32_t* offsets,
                                 const size_t distinct_count,
                                 const size_t payload_count,
                                 const size_t payload_chunk_word_count) {
  CHECK_GT(payload_chunk_word_count, size_t(0));
  uint64_t running_count = 0;
  uint64_t total_count = 0;
  for (size_t idx = 0; idx < distinct_count; ++idx) {
    const auto count = counts[idx];
    total_count += count;
    if (count > payload_chunk_word_count) {
      throw HashJoinFail(
          "Ranked bitmap one-to-many payload range exceeds chunk capacity");
    }
    const auto chunk_offset = running_count % payload_chunk_word_count;
    if (count > 0 && chunk_offset > 0) {
      const auto chunk_remaining = payload_chunk_word_count - chunk_offset;
      if (count > chunk_remaining) {
        running_count += chunk_remaining;
      }
    }
    if (running_count > std::numeric_limits<uint32_t>::max()) {
      throw HashJoinFail("Ranked bitmap one-to-many payload offsets exceed 32 bits");
    }
    offsets[idx] = static_cast<uint32_t>(running_count);
    running_count += count;
  }
  // NULL build keys (and an optional pushed-down build filter) intentionally emit no
  // payload. The count pass is authoritative; the source row count is only an upper
  // bound used for the initial allocation.
  if (total_count > payload_count) {
    throw HashJoinFail(
        "Ranked bitmap one-to-many payload count exceeds source row count");
  }
  if (running_count > std::numeric_limits<uint32_t>::max()) {
    throw HashJoinFail("Ranked bitmap one-to-many payload offsets exceed 32 bits");
  }
  return running_count;
}

#ifdef HAVE_CUDA
void copy_ranked_bitmap_index_on_device(RankedBitmapHashTable& dst,
                                        RankedBitmapHashTable& src,
                                        CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                        const int device_id) {
  CHECK(cuda_mgr);
  const auto index_bytes = src.getIndexWordCount() * sizeof(uint32_t);
  if (index_bytes == 0) {
    return;
  }
  auto src_index = src.hasSegmentedLayout()
                       ? reinterpret_cast<int8_t*>(src.getGpuBitmap())
                       : src.getGpuBuffer();
  auto dst_index = reinterpret_cast<int8_t*>(dst.getGpuBitmap());
  CHECK(src_index);
  CHECK(dst_index);
  cuda_mgr->copyDeviceToDevice(
      dst_index, src_index, index_bytes, device_id, device_id, "ranked bitmap index");
}

size_t ranked_bitmap_distinct_count_gpu(RankedBitmapHashTable& hash_table,
                                        DeviceAllocator* device_allocator,
                                        const uint32_t* bitmap,
                                        const uint32_t* rank_blocks) {
  CHECK(device_allocator);
  if (hash_table.getRankBlockCount() == 0) {
    return 0;
  }
  const auto last_block_idx = hash_table.getRankBlockCount() - 1;
  const auto block_start_word =
      last_block_idx * RankedBitmapHashTable::kRankBlockWordCount;
  uint32_t last_block_prefix{0};
  device_allocator->copyFromDevice(&last_block_prefix,
                                   rank_blocks + last_block_idx,
                                   sizeof(last_block_prefix),
                                   "ranked bitmap last rank block");
  const auto word_count = hash_table.getBitmapWordCount() - block_start_word;
  std::array<uint32_t, RankedBitmapHashTable::kRankBlockWordCount> words{};
  device_allocator->copyFromDevice(words.data(),
                                   bitmap + block_start_word,
                                   word_count * sizeof(uint32_t),
                                   "ranked bitmap last bitmap block");
  uint64_t distinct_count = last_block_prefix;
  for (size_t idx = 0; idx < word_count; ++idx) {
    distinct_count += host_popcount32(words[idx]);
  }
  CHECK_LE(distinct_count, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
  return static_cast<size_t>(distinct_count);
}

uint32_t* ranked_bitmap_gpu_bitmap(RankedBitmapHashTable& hash_table) {
  return hash_table.hasSegmentedLayout()
             ? hash_table.getGpuBitmap()
             : reinterpret_cast<uint32_t*>(hash_table.getGpuBuffer());
}

uint32_t* ranked_bitmap_gpu_rank_blocks(RankedBitmapHashTable& hash_table) {
  return ranked_bitmap_gpu_bitmap(hash_table) + hash_table.getBitmapWordCount();
}

void allreduce_payload_free_ranked_bitmaps_with_resources(
    const std::vector<int>& device_ids,
    const std::vector<std::shared_ptr<RankedBitmapHashTable>>& hash_tables,
    CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const std::vector<DeviceAllocator*>& device_allocators,
    const std::vector<CUstream>& cuda_streams,
    const size_t expected_distinct_count) {
  CHECK_EQ(device_ids.size(), hash_tables.size());
  CHECK_EQ(device_ids.size(), device_allocators.size());
  CHECK_EQ(device_ids.size(), cuda_streams.size());
  CHECK_GT(device_ids.size(), size_t(1));
  CHECK(cuda_mgr);
  const auto bitmap_word_count = hash_tables.front()->getBitmapWordCount();
  CHECK_GT(bitmap_word_count, size_t(0));
  const auto device_count = device_ids.size();
  const auto chunk_start = [&](const size_t chunk_index) {
    CHECK_LE(chunk_index, device_count);
    return static_cast<size_t>(
        (static_cast<unsigned __int128>(bitmap_word_count) * chunk_index) / device_count);
  };
  const auto max_chunk_word_count = (bitmap_word_count - 1) / device_count + size_t(1);
  CHECK_GT(max_chunk_word_count, size_t(0));

  struct CollectiveState {
    int device_id;
    std::shared_ptr<RankedBitmapHashTable> hash_table;
    uint32_t* bitmap;
    uint32_t* receive_buffer;
    CUstream stream;
  };
  std::vector<CollectiveState> states;
  states.reserve(device_count);
  for (size_t rank = 0; rank < device_count; ++rank) {
    auto& hash_table = hash_tables[rank];
    CHECK(hash_table);
    CHECK(hash_table->isPayloadFree());
    CHECK_EQ(hash_table->getBitmapWordCount(), bitmap_word_count);
    auto allocator = device_allocators[rank];
    CHECK(allocator);
    states.push_back(CollectiveState{device_ids[rank],
                                     hash_table,
                                     ranked_bitmap_gpu_bitmap(*hash_table),
                                     reinterpret_cast<uint32_t*>(allocator->alloc(
                                         max_chunk_word_count * sizeof(uint32_t))),
                                     cuda_streams[rank]});
  }

  // Reduce-scatter. Each rank finishes with one globally reduced bitmap chunk.
  for (size_t step = 0; step + 1 < device_count; ++step) {
    std::vector<std::future<void>> copy_threads;
    copy_threads.reserve(device_count);
    for (size_t rank = 0; rank < device_count; ++rank) {
      copy_threads.emplace_back(std::async(std::launch::async, [&, rank, step] {
        const auto source_rank = (rank + device_count - 1) % device_count;
        const auto receive_chunk = (rank + device_count - step - 1) % device_count;
        const auto begin = chunk_start(receive_chunk);
        const auto word_count = chunk_start(receive_chunk + 1) - begin;
        cuda_mgr->copyDeviceToDevice(
            reinterpret_cast<int8_t*>(states[rank].receive_buffer),
            reinterpret_cast<int8_t*>(states[source_rank].bitmap + begin),
            word_count * sizeof(uint32_t),
            states[rank].device_id,
            states[source_rank].device_id,
            "ranked bitmap ring reduce-scatter",
            states[rank].stream,
            true);
      }));
    }
    for (auto& copy_thread : copy_threads) {
      copy_thread.get();
    }

    std::vector<std::future<void>> reduce_threads;
    reduce_threads.reserve(device_count);
    for (size_t rank = 0; rank < device_count; ++rank) {
      reduce_threads.emplace_back(std::async(std::launch::async, [&, rank, step] {
        const auto receive_chunk = (rank + device_count - step - 1) % device_count;
        const auto begin = chunk_start(receive_chunk);
        const auto word_count = chunk_start(receive_chunk + 1) - begin;
        cuda_mgr->setContext(states[rank].device_id);
        bitwise_or_uint32_on_device(states[rank].bitmap + begin,
                                    states[rank].receive_buffer,
                                    static_cast<int64_t>(word_count),
                                    states[rank].stream);
      }));
    }
    for (auto& reduce_thread : reduce_threads) {
      reduce_thread.get();
    }
  }

  // Allgather the reduced chunks so every GPU can probe the complete bitmap.
  for (size_t step = 0; step + 1 < device_count; ++step) {
    std::vector<std::future<void>> copy_threads;
    copy_threads.reserve(device_count);
    for (size_t rank = 0; rank < device_count; ++rank) {
      copy_threads.emplace_back(std::async(std::launch::async, [&, rank, step] {
        const auto source_rank = (rank + device_count - 1) % device_count;
        const auto receive_chunk = (rank + device_count - step) % device_count;
        const auto begin = chunk_start(receive_chunk);
        const auto word_count = chunk_start(receive_chunk + 1) - begin;
        cuda_mgr->copyDeviceToDevice(
            reinterpret_cast<int8_t*>(states[rank].bitmap + begin),
            reinterpret_cast<int8_t*>(states[source_rank].bitmap + begin),
            word_count * sizeof(uint32_t),
            states[rank].device_id,
            states[source_rank].device_id,
            "ranked bitmap ring allgather",
            states[rank].stream,
            true);
      }));
    }
    for (auto& copy_thread : copy_threads) {
      copy_thread.get();
    }
  }

  std::vector<std::future<void>> rank_threads;
  rank_threads.reserve(device_count);
  for (size_t rank = 0; rank < device_count; ++rank) {
    rank_threads.emplace_back(std::async(std::launch::async, [&, rank] {
      auto allocator = device_allocators[rank];
      CHECK(allocator);
      cuda_mgr->setContext(states[rank].device_id);
      auto rank_blocks = ranked_bitmap_gpu_rank_blocks(*states[rank].hash_table);
      build_ranked_bitmap_index_on_device(rank_blocks,
                                          states[rank].bitmap,
                                          bitmap_word_count,
                                          RankedBitmapHashTable::kRankBlockWordCount,
                                          states[rank].stream);
      const auto distinct_count = ranked_bitmap_distinct_count_gpu(
          *states[rank].hash_table, allocator, states[rank].bitmap, rank_blocks);
      if (distinct_count != expected_distinct_count) {
        throw HashJoinFail(
            "Distributed payload-free ranked bitmap requires globally unique keys");
      }
    }));
  }
  for (auto& rank_thread : rank_threads) {
    rank_thread.get();
  }
}

void allreduce_payload_free_ranked_bitmaps(
    const std::vector<int>& device_ids,
    const std::vector<std::shared_ptr<RankedBitmapHashTable>>& hash_tables,
    Executor* executor,
    const size_t expected_distinct_count) {
  CHECK(executor);
  std::vector<DeviceAllocator*> device_allocators;
  std::vector<CUstream> cuda_streams;
  device_allocators.reserve(device_ids.size());
  cuda_streams.reserve(device_ids.size());
  for (const auto device_id : device_ids) {
    device_allocators.push_back(executor->getCudaAllocator(device_id));
    cuda_streams.push_back(executor->getCudaStream(device_id));
  }
  allreduce_payload_free_ranked_bitmaps_with_resources(
      device_ids,
      hash_tables,
      executor->getDataMgr()->getCudaMgr(),
      device_allocators,
      cuda_streams,
      expected_distinct_count);
}

size_t exclusive_scan_counts_gpu(DeviceAllocator* device_allocator,
                                 const uint32_t* counts,
                                 uint32_t* offsets,
                                 const size_t distinct_count,
                                 const size_t payload_count,
                                 const size_t payload_chunk_word_count,
                                 CUstream cuda_stream) {
  CHECK(device_allocator);
  CHECK_GT(payload_chunk_word_count, size_t(0));
  if (distinct_count == 0) {
    if (payload_count != 0) {
      throw HashJoinFail("Ranked bitmap one-to-many payload count mismatch");
    }
    return 0;
  }
  if (payload_count <= payload_chunk_word_count) {
    exclusive_scan_uint32_on_device(
        counts, offsets, static_cast<int64_t>(distinct_count), cuda_stream);
    uint32_t last_offset{0};
    uint32_t last_count{0};
    device_allocator->copyFromDevice(&last_offset,
                                     offsets + distinct_count - 1,
                                     sizeof(last_offset),
                                     "ranked bitmap last one-to-many offset");
    device_allocator->copyFromDevice(&last_count,
                                     counts + distinct_count - 1,
                                     sizeof(last_count),
                                     "ranked bitmap last one-to-many count");
    const auto payload_capacity =
        static_cast<uint64_t>(last_offset) + static_cast<uint64_t>(last_count);
    if (payload_capacity > payload_count) {
      throw HashJoinFail(
          "Ranked bitmap one-to-many payload count exceeds source row count");
    }
    CHECK_LE(payload_capacity,
             static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
    return static_cast<size_t>(payload_capacity);
  }
  const auto count_bytes = distinct_count * sizeof(uint32_t);
  auto host_offsets = std::make_unique<uint32_t[]>(distinct_count);
  device_allocator->copyFromDevice(
      host_offsets.get(), counts, count_bytes, "ranked bitmap one-to-many counts");
  const auto payload_capacity = exclusive_scan_counts_cpu(host_offsets.get(),
                                                          host_offsets.get(),
                                                          distinct_count,
                                                          payload_count,
                                                          payload_chunk_word_count);
  device_allocator->copyToDevice(
      offsets, host_offsets.get(), count_bytes, "ranked bitmap one-to-many offsets");
  return payload_capacity;
}
#endif

std::shared_ptr<RankedBitmapHashTable> promote_ranked_bitmap_to_one_to_many_cpu(
    RankedBitmapHashTable& source,
    const size_t bit_count,
    const size_t payload_count,
    const size_t distinct_count,
    const size_t max_slab_size) {
  auto promoted = std::make_shared<RankedBitmapHashTable>(ExecutorDeviceType::CPU,
                                                          bit_count,
                                                          payload_count,
                                                          max_slab_size,
                                                          nullptr,
                                                          -1,
                                                          HashType::OneToMany,
                                                          distinct_count);
  std::memcpy(promoted->getCpuBitmap(),
              source.getCpuBitmap(),
              source.getIndexWordCount() * sizeof(uint32_t));
  return promoted;
}

#ifdef HAVE_CUDA
std::shared_ptr<RankedBitmapHashTable> promote_ranked_bitmap_to_one_to_many_gpu(
    RankedBitmapHashTable& source,
    const size_t bit_count,
    const size_t payload_count,
    const size_t distinct_count,
    const size_t max_slab_size,
    Data_Namespace::DataMgr* data_mgr,
    DeviceAllocator* device_allocator,
    CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const int device_id) {
  auto promoted = std::make_shared<RankedBitmapHashTable>(ExecutorDeviceType::GPU,
                                                          bit_count,
                                                          payload_count,
                                                          max_slab_size,
                                                          data_mgr,
                                                          device_id,
                                                          HashType::OneToMany,
                                                          distinct_count);
  copy_ranked_bitmap_index_on_device(*promoted, source, cuda_mgr, device_id);
  promoted->copyGpuHeaderToDevice(device_allocator);
  return promoted;
}
#endif

}  // namespace

#ifdef HAVE_CUDA
void allreduce_payload_free_ranked_bitmaps_for_test(
    const std::vector<int>& device_ids,
    const std::vector<std::shared_ptr<RankedBitmapHashTable>>& hash_tables,
    Data_Namespace::DataMgr* data_mgr,
    const size_t expected_distinct_count) {
  CHECK(data_mgr);
  std::vector<std::shared_ptr<CudaAllocator>> allocator_owners;
  std::vector<DeviceAllocator*> device_allocators;
  std::vector<CUstream> cuda_streams;
  allocator_owners.reserve(device_ids.size());
  device_allocators.reserve(device_ids.size());
  cuda_streams.reserve(device_ids.size());
  for (const auto device_id : device_ids) {
    const auto cuda_stream = getQueryEngineCudaStreamForDevice(device_id);
    allocator_owners.push_back(
        std::make_shared<CudaAllocator>(data_mgr, device_id, cuda_stream));
    device_allocators.push_back(allocator_owners.back().get());
    cuda_streams.push_back(cuda_stream);
  }
  allreduce_payload_free_ranked_bitmaps_with_resources(device_ids,
                                                       hash_tables,
                                                       data_mgr->getCudaMgr(),
                                                       device_allocators,
                                                       cuda_streams,
                                                       expected_distinct_count);
}
#endif

HashtableRecycler* RankedBitmapJoinHashTable::getHashTableCache() {
  CHECK(hash_table_cache_);
  return hash_table_cache_.get();
}

void RankedBitmapJoinHashTable::invalidateCache() {
  CHECK(hash_table_cache_);
  hash_table_cache_->clearCache();
}

void RankedBitmapJoinHashTable::markCachedItemAsDirty(size_t table_key) {
  CHECK(hash_table_cache_);
  auto candidate_table_keys =
      hash_table_cache_->getMappedQueryPlanDagsWithTableKey(table_key);
  if (!candidate_table_keys.has_value()) {
    return;
  }
  for (int device_identifier = 1;
       device_identifier <= DataRecyclerUtil::MAX_GPU_CACHE_DEVICE_COUNT;
       ++device_identifier) {
    hash_table_cache_->markCachedItemAsDirty(table_key,
                                             *candidate_table_keys,
                                             CacheItemType::RANKED_BITMAP_HT,
                                             device_identifier);
  }
}

std::shared_ptr<RankedBitmapJoinHashTable> RankedBitmapJoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> condition,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const std::set<int>& device_ids,
    ColumnCacheMap& column_cache,
    Executor* executor,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const TableIdToNodeMap& table_id_to_node_map,
    const RegisteredQueryHint& query_hint,
    const std::list<std::shared_ptr<Analyzer::Expr>>& build_side_quals,
    const bool payload_free_unique_probe) {
  if (join_type != JoinType::INNER) {
    throw HashJoinFail("Ranked bitmap join is only valid for INNER joins");
  }
  if (!IS_EQUIVALENCE(condition->get_optype()) || condition->get_optype() == kBW_EQ) {
    throw HashJoinFail("Ranked bitmap join only supports regular equality predicates");
  }
  if (dynamic_cast<const Analyzer::UOper*>(condition->get_left_operand()) ||
      dynamic_cast<const Analyzer::UOper*>(condition->get_right_operand())) {
    throw HashJoinFail("Ranked bitmap join does not support casted join keys");
  }
  const auto normalized =
      HashJoin::normalizeColumnPairs(condition.get(), executor->getTemporaryTables());
  if (normalized.first.size() != 1 || normalized.second.size() != 1 ||
      normalized.second.front().first.size() || normalized.second.front().second.size()) {
    throw HashJoinFail("Ranked bitmap join only supports one integer column pair");
  }
  const auto inner_col = normalized.first.front().first;
  const auto outer_expr = normalized.first.front().second;
  CHECK(inner_col);
  CHECK(outer_expr);
  if (const auto inner_cd = get_column_descriptor_maybe(inner_col->getColumnKey());
      inner_cd && inner_cd->isVirtualCol) {
    throw HashJoinFail("Ranked bitmap join does not support virtual build columns");
  }
  if (const auto inner_cd = get_column_descriptor_maybe(inner_col->getColumnKey());
      inner_cd && inner_cd->isGeoPhyCol) {
    throw HashJoinFail(
        "Ranked bitmap join does not support physical geospatial build columns");
  }
  const auto& ti = inner_col->get_type_info();
  if (!supported_ranked_bitmap_type(ti)) {
    throw HashJoinFail("Ranked bitmap join only supports integer-like join keys");
  }
  const auto outer_ti = HashJoin::getExpressionTypeForJoin(
      outer_expr, executor->getTemporaryTables(), table_id_to_node_map, true);
  if (!supported_ranked_bitmap_type(outer_ti) || ti.get_type() != outer_ti.get_type()) {
    throw HashJoinFail(
        "Ranked bitmap join requires matching integer-like join key types");
  }
  if (ti.is_boolean() && (!ti.get_notnull() || !outer_ti.get_notnull())) {
    throw HashJoinFail("Ranked bitmap join does not support nullable boolean keys");
  }
  const auto outer_col = HashJoin::getHashJoinColumn<Analyzer::ColumnVar>(outer_expr);
  if (!outer_col) {
    throw HashJoinFail("Ranked bitmap join only supports stored outer columns");
  }
  const auto outer_col_ti = HashJoin::getColumnTypeForJoin(
      outer_col, executor->getTemporaryTables(), table_id_to_node_map, true);
  if (ti.get_type() != outer_col_ti.get_type()) {
    throw HashJoinFail("Ranked bitmap join requires matching stored join key types");
  }
  if (const auto outer_cd = get_column_descriptor_maybe(outer_col->getColumnKey());
      outer_cd && outer_cd->isGeoPhyCol) {
    throw HashJoinFail(
        "Ranked bitmap join does not support physical geospatial probe columns");
  }

  auto col_range = getExpressionRange(inner_col, query_infos, executor);
  if (col_range.getType() == ExpressionRangeType::Invalid) {
    throw HashJoinFail("Could not compute range for ranked bitmap join key");
  }
  const auto bit_count = checked_bit_count_for_ranked_bitmap(col_range);
  if (bit_count == 0) {
    throw HashJoinFail("Ranked bitmap join input range is empty");
  }
  const auto& inner_table_info =
      get_inner_query_info(inner_col->getTableKey(), query_infos).info;
  const auto tuple_count = get_hash_join_table_num_tuples(inner_table_info);
  if (tuple_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw TooManyHashEntries(
        "Ranked bitmap join row ids require more than unsigned 32 bits");
  }
  const auto table_bytes = checked_ranked_bitmap_bytes(
      bit_count, payload_free_unique_probe ? size_t(0) : tuple_count);
  if (query_hint.isHintRegistered(QueryHint::kMaxJoinHashTableSize) &&
      table_bytes > query_hint.max_join_hash_table_size) {
    throw JoinHashTableTooBig(table_bytes, query_hint.max_join_hash_table_size);
  }
  const auto key_width = std::max<size_t>(sizeof(int32_t), ti.get_logical_size());
  check_ranked_bitmap_size(table_bytes, tuple_count, key_width);
  std::vector<std::shared_ptr<Analyzer::Expr>> pushed_down_build_quals;
  auto build_side_filter =
      payload_free_unique_probe
          ? std::optional<RankedBitmapJoinHashTable::BuildSideFilter>{}
          : build_ranked_bitmap_build_side_filter(
                inner_col, build_side_quals, pushed_down_build_quals);
  // An unfiltered source larger than its key domain cannot prove uniqueness. Keep the
  // established one-to-many path instead of trying the unique-only pruned layout and
  // falling all the way back to a different hash-table implementation.
  const bool allow_range_pruning =
      may_range_prune_ranked_bitmap_build(
          inner_col, outer_expr, query_infos, memory_level) &&
      (payload_free_unique_probe || build_side_filter || tuple_count <= bit_count);

  auto cache_key_info = ranked_bitmap_build_cache_key(inner_col,
                                                      outer_expr,
                                                      col_range,
                                                      bit_count,
                                                      tuple_count,
                                                      memory_level,
                                                      device_ids,
                                                      condition->get_optype(),
                                                      join_type,
                                                      hashtable_build_dag_map,
                                                      table_id_to_node_map,
                                                      inner_table_info.fragments,
                                                      executor,
                                                      build_side_filter);
  if (inner_col->getTableKey().table_id < 0 && !payload_free_unique_probe) {
    cache_key_info.globally_recyclable = false;
  }
  if (payload_free_unique_probe) {
    if (build_side_filter) {
      throw HashJoinFail(
          "Payload-free ranked bitmap joins do not support build-side filters");
    }
    if (!cache_key_info.key.empty()) {
      cache_key_info.key += "|payload-free-unique-probe=v1";
      cache_key_info.source += "+payload-free";
    }
  }
  if (allow_range_pruning) {
    if (!cache_key_info.key.empty()) {
      cache_key_info.key += inner_col->getTableKey().table_id < 0
                                ? "|range-pruned-device-ranges=v2"
                                : "|range-pruned-device-ranges=v4";
      cache_key_info.source += "+range-pruned";
      const auto outer_dependency =
          ranked_bitmap_range_pruning_cache_dependency(outer_expr, table_id_to_node_map);
      if (outer_dependency) {
        cache_key_info.key += outer_dependency->key;
        cache_key_info.table_keys.insert(outer_dependency->table_keys.begin(),
                                         outer_dependency->table_keys.end());
      } else {
        cache_key_info.globally_recyclable = false;
      }
    }
  }
  auto join_hash_table = std::shared_ptr<RankedBitmapJoinHashTable>(
      new RankedBitmapJoinHashTable(condition,
                                    inner_col,
                                    outer_expr,
                                    col_range,
                                    bit_count,
                                    tuple_count,
                                    table_bytes,
                                    query_infos,
                                    memory_level,
                                    join_type,
                                    device_ids,
                                    column_cache,
                                    executor,
                                    std::move(build_side_filter),
                                    std::move(pushed_down_build_quals),
                                    allow_range_pruning,
                                    payload_free_unique_probe));
  const auto row_set_mem_owner = executor->getRowSetMemoryOwner();
  if (!cache_key_info.key.empty() && row_set_mem_owner) {
    if (const auto cached_base =
            row_set_mem_owner->getCachedJoinHashTable(cache_key_info.key)) {
      auto cached = std::dynamic_pointer_cast<RankedBitmapJoinHashTable>(cached_base);
      if (cached) {
        CHECK_EQ(cached->bit_count_, bit_count);
        CHECK_EQ(cached->payload_count_, tuple_count);
        CHECK_EQ(cached->payload_free_unique_probe_, payload_free_unique_probe);
        CHECK_EQ(cached->col_range_.getIntMin(), col_range.getIntMin());
        CHECK_EQ(cached->col_range_.getIntMax(), col_range.getIntMax());
        CHECK_EQ(cached->memory_level_, memory_level);
        reuse_ranked_bitmap_hash_tables(*join_hash_table, cached, device_ids);
        join_hash_table->hash_type_.store(cached->getHashType(),
                                          std::memory_order_relaxed);
        return join_hash_table;
      }
    }
  }
  const bool can_use_global_gpu_cache =
      cache_key_info.globally_recyclable &&
      memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL &&
      !cache_key_info.key.empty();
  const auto global_cache_key = can_use_global_gpu_cache
                                    ? hash_ranked_bitmap_cache_key(cache_key_info.key)
                                    : EMPTY_HASHED_PLAN_DAG_KEY;
  if (can_use_global_gpu_cache) {
    bool found_all_devices = true;
    std::map<int, std::shared_ptr<HashTable>> cached_hash_tables;
    HashType cached_hash_type = HashType::OneToOne;
    for (const auto device_id : device_ids) {
      auto cached_hash_table =
          hash_table_cache_->getItemFromCache(global_cache_key,
                                              CacheItemType::RANKED_BITMAP_HT,
                                              gpu_cache_device_identifier(device_id));
      auto ranked_hash_table =
          std::dynamic_pointer_cast<RankedBitmapHashTable>(cached_hash_table);
      if (!ranked_hash_table) {
        found_all_devices = false;
        break;
      }
      cached_hash_type = ranked_hash_table->getLayout();
      cached_hash_tables.emplace(device_id, std::move(cached_hash_table));
    }
    if (found_all_devices) {
      for (auto& [device_id, hash_table] : cached_hash_tables) {
        join_hash_table->putHashTableForDevice(std::move(hash_table), device_id);
      }
      join_hash_table->hash_type_.store(cached_hash_type, std::memory_order_relaxed);
      return join_hash_table;
    }
  }
  join_hash_table->reify();
  if (can_use_global_gpu_cache) {
    mark_ranked_bitmap_cache_key_for_tables(global_cache_key, cache_key_info.table_keys);
    for (const auto device_id : device_ids) {
      auto hash_table = join_hash_table->getHashTableForDevice(device_id);
      hash_table_cache_->putItemToCache(
          global_cache_key,
          hash_table,
          CacheItemType::RANKED_BITMAP_HT,
          gpu_cache_device_identifier(device_id),
          hash_table->getHashTableBufferSize(ExecutorDeviceType::GPU),
          0);
    }
  }
  if (!cache_key_info.key.empty() && row_set_mem_owner) {
    row_set_mem_owner->putCachedJoinHashTable(cache_key_info.key, join_hash_table);
  }
  return join_hash_table;
}

RankedBitmapJoinHashTable::RankedBitmapJoinHashTable(
    const std::shared_ptr<Analyzer::BinOper> condition,
    const Analyzer::ColumnVar* inner_col,
    const Analyzer::Expr* outer_expr,
    const ExpressionRange& col_range,
    const size_t bit_count,
    const size_t payload_count,
    const size_t table_bytes,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const std::set<int>& device_ids,
    ColumnCacheMap& column_cache,
    Executor* executor,
    std::optional<BuildSideFilter> build_side_filter,
    std::vector<std::shared_ptr<Analyzer::Expr>> pushed_down_build_quals,
    const bool allow_range_pruning,
    const bool payload_free_unique_probe)
    : condition_(condition)
    , inner_col_(std::dynamic_pointer_cast<Analyzer::ColumnVar>(inner_col->deep_copy()))
    , outer_expr_(outer_expr->deep_copy())
    , col_range_(col_range)
    , bit_count_(bit_count)
    , payload_count_(payload_count)
    , table_bytes_(table_bytes)
    , query_infos_(query_infos)
    , memory_level_(memory_level)
    , join_type_(join_type)
    , device_ids_(device_ids)
    , column_cache_(column_cache)
    , executor_(executor)
    , build_side_filter_(std::move(build_side_filter))
    , pushed_down_build_quals_(std::move(pushed_down_build_quals))
    , allow_range_pruning_(allow_range_pruning)
    , payload_free_unique_probe_(payload_free_unique_probe) {
  CHECK(condition_);
  CHECK(inner_col_);
  CHECK(outer_expr_);
  CHECK_GT(device_ids_.size(), 0u);
}

bool RankedBitmapJoinHashTable::isBuildSideQualifierPushedDown(
    const Analyzer::Expr* qual) const {
  if (!qual) {
    return false;
  }
  return std::any_of(pushed_down_build_quals_.begin(),
                     pushed_down_build_quals_.end(),
                     [qual](const auto& pushed_qual) {
                       return pushed_qual.get() == qual ||
                              (pushed_qual && *pushed_qual == *qual);
                     });
}

void RankedBitmapJoinHashTable::reify() {
  const auto& query_info = get_inner_query_info(getInnerTableId(), query_infos_).info;
  if (query_info.fragments.empty()) {
    return;
  }

#ifdef HAVE_CUDA
  std::map<int, std::vector<Fragmenter_Namespace::FragmentInfo>> local_fragments;
  bool use_distributed_payload_free_bitmap =
      memory_level_ == Data_Namespace::GPU_LEVEL && payload_free_unique_probe_ &&
      getInnerTableId().table_id < 0 && device_ids_.size() > size_t(1);
  if (use_distributed_payload_free_bitmap) {
    for (const auto device_id : device_ids_) {
      local_fragments.emplace(device_id,
                              std::vector<Fragmenter_Namespace::FragmentInfo>{});
      if (allow_range_pruning_ &&
          range_pruned_ranked_bitmap_fragments_for_device(inner_col_.get(),
                                                          outer_expr_.get(),
                                                          query_info.fragments,
                                                          query_infos_,
                                                          memory_level_,
                                                          device_id)) {
        use_distributed_payload_free_bitmap = false;
        break;
      }
    }
  }
  size_t assigned_row_count{0};
  if (use_distributed_payload_free_bitmap) {
    for (const auto& fragment : query_info.fragments) {
      if (fragment.deviceIds.size() <= static_cast<size_t>(Data_Namespace::GPU_LEVEL)) {
        use_distributed_payload_free_bitmap = false;
        break;
      }
      const auto device_id = fragment.deviceIds[Data_Namespace::GPU_LEVEL];
      const auto local_it = local_fragments.find(device_id);
      if (local_it == local_fragments.end()) {
        use_distributed_payload_free_bitmap = false;
        break;
      }
      local_it->second.push_back(fragment);
      const auto fragment_row_count = fragment.getNumTuples();
      if (fragment_row_count > std::numeric_limits<size_t>::max() - assigned_row_count) {
        throw TooManyHashEntries("Distributed ranked bitmap row count overflow");
      }
      assigned_row_count += fragment_row_count;
    }
  }
  if (use_distributed_payload_free_bitmap) {
    const auto key_width = inner_col_->get_type_info().get_size();
    const auto bitmap_bytes =
        RankedBitmapHashTable::wordsForBits(bit_count_) * sizeof(uint32_t);
    const auto source_bytes = key_width > 0
                                  ? static_cast<unsigned __int128>(payload_count_) *
                                        static_cast<size_t>(key_width)
                                  : 0;
    use_distributed_payload_free_bitmap =
        key_width > 0 && assigned_row_count == payload_count_ && bitmap_bytes > 0 &&
        source_bytes * 2 > static_cast<unsigned __int128>(bitmap_bytes) * 5 &&
        std::all_of(local_fragments.begin(),
                    local_fragments.end(),
                    [](const auto& entry) { return !entry.second.empty(); });
  }
  if (use_distributed_payload_free_bitmap) {
    std::vector<std::future<void>> init_threads;
    for (const auto device_id : device_ids_) {
      init_threads.push_back(std::async(std::launch::async,
                                        &RankedBitmapJoinHashTable::reifyForDevice,
                                        this,
                                        local_fragments.at(device_id),
                                        device_id,
                                        logger::thread_local_ids(),
                                        true));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }
    std::vector<int> device_ids(device_ids_.begin(), device_ids_.end());
    std::vector<std::shared_ptr<RankedBitmapHashTable>> hash_tables;
    hash_tables.reserve(device_ids.size());
    for (const auto device_id : device_ids) {
      auto hash_table = std::dynamic_pointer_cast<RankedBitmapHashTable>(
          getHashTableForDevice(device_id));
      CHECK(hash_table);
      hash_tables.push_back(std::move(hash_table));
    }
    allreduce_payload_free_ranked_bitmaps(
        device_ids, hash_tables, executor_, payload_count_);
    return;
  }
#endif

  std::vector<std::future<void>> init_threads;
  for (const auto device_id : device_ids_) {
    init_threads.push_back(std::async(std::launch::async,
                                      &RankedBitmapJoinHashTable::reifyForDevice,
                                      this,
                                      query_info.fragments,
                                      device_id,
                                      logger::thread_local_ids(),
                                      false));
  }
  for (auto& init_thread : init_threads) {
    init_thread.wait();
  }
  for (auto& init_thread : init_threads) {
    init_thread.get();
  }
}

void RankedBitmapJoinHashTable::reifyForDevice(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    const logger::ThreadLocalIds parent_thread_local_ids,
    const bool defer_payload_free_rank_index) {
  logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();

  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<std::shared_ptr<void>> malloc_owner;
  DeviceAllocator* device_allocator{nullptr};
  const auto effective_memory_level = memory_level_;
#ifdef HAVE_CUDA
  if (effective_memory_level == Data_Namespace::GPU_LEVEL) {
    auto cuda_mgr = executor_->getDataMgr()->getCudaMgr();
    CHECK(cuda_mgr);
    cuda_mgr->setContext(device_id);
    device_allocator = executor_->getCudaAllocator(device_id);
    CHECK(device_allocator);
  }
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif

  const auto range_pruned_fragments =
      allow_range_pruning_
          ? range_pruned_ranked_bitmap_fragments_for_device(inner_col_.get(),
                                                            outer_expr_.get(),
                                                            fragments,
                                                            query_infos_,
                                                            effective_memory_level,
                                                            device_id)
          : std::nullopt;
  const auto& build_fragments =
      range_pruned_fragments ? *range_pruned_fragments : fragments;
  const bool preserve_result_set_fragment_offsets = range_pruned_fragments.has_value();
  std::map<std::pair<int, int>, size_t> physical_fragment_rowid_offsets;
  if (range_pruned_fragments && inner_col_->getTableKey().table_id >= 0) {
    size_t rowid_offset{0};
    for (const auto& fragment : fragments) {
      physical_fragment_rowid_offsets.emplace(
          std::make_pair(fragment.physicalTableId, fragment.fragmentId), rowid_offset);
      const auto fragment_row_count = fragment.getNumTuples();
      if (fragment_row_count > std::numeric_limits<size_t>::max() - rowid_offset) {
        throw TooManyHashEntries("Ranked bitmap build rowid offset overflow");
      }
      rowid_offset += fragment_row_count;
    }
  }
  const auto physical_rowid_offsets = physical_fragment_rowid_offsets.empty()
                                          ? nullptr
                                          : &physical_fragment_rowid_offsets;
  auto join_column = fetchJoinColumn(inner_col_.get(),
                                     build_fragments,
                                     effective_memory_level,
                                     device_id,
                                     chunks_owner,
                                     device_allocator,
                                     malloc_owner,
                                     executor_,
                                     &column_cache_,
                                     preserve_result_set_fragment_offsets,
                                     physical_rowid_offsets);
  const auto build_filter = [&]() {
    if (!build_side_filter_) {
      return make_empty_join_column_filter();
    }
    auto filter_column = fetchJoinColumn(build_side_filter_->column.get(),
                                         build_fragments,
                                         effective_memory_level,
                                         device_id,
                                         chunks_owner,
                                         device_allocator,
                                         malloc_owner,
                                         executor_,
                                         &column_cache_,
                                         preserve_result_set_fragment_offsets,
                                         physical_rowid_offsets);
    const auto& filter_ti = build_side_filter_->column->get_type_info();
    return JoinColumnFilter{true,
                            filter_column,
                            JoinColumnTypeInfo{static_cast<size_t>(filter_ti.get_size()),
                                               0,
                                               0,
                                               inline_fixed_encoding_null_val(filter_ti),
                                               false,
                                               0,
                                               get_join_column_type_kind(filter_ti)},
                            build_side_filter_->has_lower_bound,
                            build_side_filter_->lower_bound_inclusive,
                            build_side_filter_->lower_bound,
                            build_side_filter_->has_upper_bound,
                            build_side_filter_->upper_bound_inclusive,
                            build_side_filter_->upper_bound};
  }();
  const auto device_payload_count = join_column.num_elems;
  const auto device_table_bytes =
      range_pruned_fragments
          ? checked_ranked_bitmap_bytes(bit_count_, device_payload_count)
          : table_bytes_;
  const auto& build_ti = inner_col_->get_type_info();
  JoinColumnTypeInfo type_info{static_cast<size_t>(build_ti.get_size()),
                               col_range_.getIntMin(),
                               col_range_.getIntMax(),
                               inline_fixed_encoding_null_val(build_ti),
                               false,
                               0,
                               get_join_column_type_kind(build_ti)};

  std::shared_ptr<RankedBitmapHashTable> ranked_hash_table;
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    const auto max_cpu_slab_size = executor_->maxCpuSlabSize();
    const bool defer_filtered_payload =
        build_filter.enabled && device_table_bytes > max_cpu_slab_size;
    const size_t initial_payload_count =
        (payload_free_unique_probe_ || defer_filtered_payload) ? size_t(0)
                                                               : device_payload_count;
    ranked_hash_table =
        std::make_shared<RankedBitmapHashTable>(ExecutorDeviceType::CPU,
                                                bit_count_,
                                                initial_payload_count,
                                                max_cpu_slab_size,
                                                nullptr,
                                                -1,
                                                HashType::OneToOne,
                                                0,
                                                defer_filtered_payload,
                                                payload_free_unique_probe_);
    const int thread_count = cpu_threads();
    std::vector<std::future<void>> bitmap_threads;
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      bitmap_threads.emplace_back(std::async(std::launch::async,
                                             fill_join_bitmap,
                                             ranked_hash_table->getCpuBitmap(),
                                             join_column,
                                             type_info,
                                             build_filter,
                                             col_range_.getIntMin(),
                                             col_range_.getIntMax(),
                                             thread_idx,
                                             thread_count));
    }
    for (auto& child : bitmap_threads) {
      child.wait();
    }
    for (auto& child : bitmap_threads) {
      child.get();
    }

    std::vector<std::future<void>> rank_threads;
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      rank_threads.emplace_back(std::async(std::launch::async,
                                           build_ranked_bitmap_index,
                                           ranked_hash_table->getCpuRankBlocks(),
                                           ranked_hash_table->getCpuBitmap(),
                                           ranked_hash_table->getBitmapWordCount(),
                                           RankedBitmapHashTable::kRankBlockWordCount,
                                           thread_idx,
                                           thread_count));
    }
    for (auto& child : rank_threads) {
      child.wait();
    }
    for (auto& child : rank_threads) {
      child.get();
    }
    uint32_t running_count = 0;
    auto rank_blocks = ranked_hash_table->getCpuRankBlocks();
    for (size_t block_idx = 0; block_idx < ranked_hash_table->getRankBlockCount();
         ++block_idx) {
      const auto block_count = rank_blocks[block_idx];
      rank_blocks[block_idx] = running_count;
      running_count += block_count;
    }
    const auto distinct_count = ranked_bitmap_distinct_count_cpu(*ranked_hash_table);
    CHECK_EQ(static_cast<size_t>(running_count), distinct_count);
    if (payload_free_unique_probe_) {
      if (distinct_count != device_payload_count) {
        throw HashJoinFail(
            "Payload-free ranked bitmap join requires unique build-side keys");
      }
    } else if (build_filter.enabled && ranked_hash_table->hasSegmentedLayout() &&
               ranked_hash_table->getPayloadCount() != distinct_count) {
      ranked_hash_table->resizeOneToOnePayloadBuffer(distinct_count, max_cpu_slab_size);
    }

    if (!payload_free_unique_probe_ && range_pruned_fragments && !build_filter.enabled &&
        distinct_count < device_payload_count) {
      throw HashJoinFail(
          "Range-pruned ranked bitmap build requires unique build-side keys");
    }
    if (!payload_free_unique_probe_ && !build_filter.enabled &&
        distinct_count < device_payload_count) {
      if (ranked_hash_table->hasSegmentedLayout()) {
        ranked_hash_table->allocateOneToManyBuffers(distinct_count, max_cpu_slab_size);
      } else {
        ranked_hash_table = promote_ranked_bitmap_to_one_to_many_cpu(*ranked_hash_table,
                                                                     bit_count_,
                                                                     device_payload_count,
                                                                     distinct_count,
                                                                     max_cpu_slab_size);
      }
      auto counts = ranked_hash_table->getCpuCounts();
      auto offsets = ranked_hash_table->getCpuOffsets();
      std::fill(counts, counts + distinct_count, uint32_t(0));
      std::vector<std::future<void>> count_threads;
      for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        count_threads.emplace_back(std::async(std::launch::async,
                                              count_ranked_bitmap_matches,
                                              counts,
                                              ranked_hash_table->getCpuBitmap(),
                                              ranked_hash_table->getCpuRankBlocks(),
                                              join_column,
                                              type_info,
                                              col_range_.getIntMin(),
                                              col_range_.getIntMax(),
                                              RankedBitmapHashTable::kRankBlockWordCount,
                                              thread_idx,
                                              thread_count));
      }
      for (auto& child : count_threads) {
        child.wait();
      }
      for (auto& child : count_threads) {
        child.get();
      }
      const auto payload_capacity =
          exclusive_scan_counts_cpu(counts,
                                    offsets,
                                    distinct_count,
                                    device_payload_count,
                                    ranked_hash_table->getPayloadChunkWordCount());
      if (payload_capacity != ranked_hash_table->getPayloadCount()) {
        ranked_hash_table->resizeOneToManyPayloadBuffer(payload_capacity,
                                                        max_cpu_slab_size);
        counts = ranked_hash_table->getCpuCounts();
        offsets = ranked_hash_table->getCpuOffsets();
      }
      std::fill(counts, counts + distinct_count, uint32_t(0));

      std::vector<std::future<int>> payload_threads;
      for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        payload_threads.emplace_back(std::async(
            std::launch::async,
            fill_ranked_bitmap_payload_one_to_many_segmented,
            ranked_hash_table->getCpuPayloadPtrs(),
            static_cast<int64_t>(ranked_hash_table->getPayloadChunkWordCount()),
            counts,
            offsets,
            ranked_hash_table->getCpuBitmap(),
            ranked_hash_table->getCpuRankBlocks(),
            join_column,
            type_info,
            col_range_.getIntMin(),
            col_range_.getIntMax(),
            RankedBitmapHashTable::kRankBlockWordCount,
            ranked_hash_table->getPayloadCount(),
            thread_idx,
            thread_count));
      }
      for (auto& child : payload_threads) {
        child.wait();
      }
      for (auto& child : payload_threads) {
        if (child.get()) {
          throw HashJoinFail("Ranked bitmap one-to-many payload build failed");
        }
      }
      hash_type_.store(HashType::OneToMany, std::memory_order_relaxed);
    } else if (!payload_free_unique_probe_) {
      if (build_filter.enabled) {
        CHECK_LE(distinct_count, ranked_hash_table->getPayloadCount());
      } else {
        CHECK_EQ(distinct_count, ranked_hash_table->getPayloadCount());
      }
      const size_t payload_chunk_count = ranked_hash_table->hasSegmentedLayout()
                                             ? ranked_hash_table->getPayloadChunkCount()
                                             : size_t(1);
      for (size_t chunk_idx = 0; chunk_idx < payload_chunk_count; ++chunk_idx) {
        std::vector<std::future<void>> init_payload_threads;
        for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
          init_payload_threads.emplace_back(std::async(
              std::launch::async,
              init_hash_join_buff,
              ranked_hash_table->hasSegmentedLayout()
                  ? reinterpret_cast<int32_t*>(
                        ranked_hash_table->getCpuPayloadChunk(chunk_idx))
                  : reinterpret_cast<int32_t*>(ranked_hash_table->getCpuPayload()),
              ranked_hash_table->hasSegmentedLayout()
                  ? ranked_hash_table->getPayloadChunkWordCount(chunk_idx)
                  : ranked_hash_table->getPayloadCount(),
              -1,
              thread_idx,
              thread_count));
        }
        for (auto& child : init_payload_threads) {
          child.wait();
        }
        for (auto& child : init_payload_threads) {
          child.get();
        }
        if (!ranked_hash_table->hasSegmentedLayout()) {
          break;
        }
      }

      std::vector<std::future<int>> payload_threads;
      for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
        if (ranked_hash_table->hasSegmentedLayout()) {
          payload_threads.emplace_back(std::async(
              std::launch::async,
              fill_ranked_bitmap_payload_segmented,
              ranked_hash_table->getCpuPayloadPtrs(),
              static_cast<int64_t>(ranked_hash_table->getPayloadChunkWordCount()),
              ranked_hash_table->getCpuBitmap(),
              ranked_hash_table->getCpuRankBlocks(),
              join_column,
              type_info,
              build_filter,
              col_range_.getIntMin(),
              col_range_.getIntMax(),
              RankedBitmapHashTable::kRankBlockWordCount,
              ranked_hash_table->getPayloadCount(),
              thread_idx,
              thread_count));
        } else {
          payload_threads.emplace_back(
              std::async(std::launch::async,
                         fill_ranked_bitmap_payload,
                         ranked_hash_table->getCpuPayload(),
                         ranked_hash_table->getCpuBitmap(),
                         ranked_hash_table->getCpuRankBlocks(),
                         join_column,
                         type_info,
                         build_filter,
                         col_range_.getIntMin(),
                         col_range_.getIntMax(),
                         RankedBitmapHashTable::kRankBlockWordCount,
                         ranked_hash_table->getPayloadCount(),
                         thread_idx,
                         thread_count));
        }
      }
      for (auto& child : payload_threads) {
        child.wait();
      }
      for (auto& child : payload_threads) {
        if (child.get()) {
          throw HashJoinFail("Ranked bitmap join key is not unique");
        }
      }
    }
  } else {
#ifdef HAVE_CUDA
    const auto max_gpu_slab_size = executor_->maxGpuSlabSize();
    const bool defer_filtered_payload =
        build_filter.enabled && device_table_bytes > max_gpu_slab_size;
    const bool force_segmented_layout =
        defer_filtered_payload ||
        (range_pruned_fragments && inner_col_->getTableKey().table_id >= 0 &&
         table_bytes_ > max_gpu_slab_size);
    const size_t initial_payload_count =
        (payload_free_unique_probe_ || defer_filtered_payload) ? size_t(0)
                                                               : device_payload_count;
    uint32_t* bitmap{nullptr};
    uint32_t* rank_blocks{nullptr};
    uint32_t* payload{nullptr};
    {
      ranked_hash_table =
          std::make_shared<RankedBitmapHashTable>(ExecutorDeviceType::GPU,
                                                  bit_count_,
                                                  initial_payload_count,
                                                  max_gpu_slab_size,
                                                  executor_->getDataMgr(),
                                                  device_id,
                                                  HashType::OneToOne,
                                                  0,
                                                  force_segmented_layout,
                                                  payload_free_unique_probe_);
      if (ranked_hash_table->hasSegmentedLayout()) {
        ranked_hash_table->copyGpuHeaderToDevice(device_allocator);
        bitmap = ranked_hash_table->getGpuBitmap();
        rank_blocks = ranked_hash_table->getGpuRankBlocks();
        device_allocator->zeroDeviceMem(
            reinterpret_cast<int8_t*>(bitmap),
            ranked_hash_table->getIndexWordCount() * sizeof(uint32_t));
      } else {
        auto gpu_buffer = ranked_hash_table->getGpuBuffer();
        bitmap = reinterpret_cast<uint32_t*>(gpu_buffer);
        rank_blocks = bitmap + ranked_hash_table->getBitmapWordCount();
        payload = rank_blocks + ranked_hash_table->getRankBlockCount();
        device_allocator->zeroDeviceMem(gpu_buffer,
                                        ranked_hash_table->getAllocatedBytes());
      }
    }
    const auto cuda_stream = executor_->getCudaStream(device_id);
    fill_join_bitmap_on_device(bitmap,
                               join_column,
                               type_info,
                               build_filter,
                               col_range_.getIntMin(),
                               col_range_.getIntMax(),
                               cuda_stream);
    if (defer_payload_free_rank_index) {
      CHECK(payload_free_unique_probe_);
      std::shared_ptr<HashTable> hash_table = std::move(ranked_hash_table);
      moveHashTableForDevice(std::move(hash_table), device_id);
      return;
    }
    size_t distinct_count{0};
    {
      build_ranked_bitmap_index_on_device(rank_blocks,
                                          bitmap,
                                          ranked_hash_table->getBitmapWordCount(),
                                          RankedBitmapHashTable::kRankBlockWordCount,
                                          cuda_stream);
      distinct_count = ranked_bitmap_distinct_count_gpu(
          *ranked_hash_table, device_allocator, bitmap, rank_blocks);
    }
    if (payload_free_unique_probe_) {
      if (distinct_count != device_payload_count) {
        throw HashJoinFail(
            "Payload-free ranked bitmap join requires unique build-side keys");
      }
    } else if (build_filter.enabled && ranked_hash_table->hasSegmentedLayout() &&
               ranked_hash_table->getPayloadCount() != distinct_count) {
      ranked_hash_table->resizeOneToOnePayloadBuffer(distinct_count, max_gpu_slab_size);
      ranked_hash_table->copyGpuHeaderToDevice(device_allocator);
      bitmap = ranked_hash_table->getGpuBitmap();
      rank_blocks = ranked_hash_table->getGpuRankBlocks();
    }
    if (!payload_free_unique_probe_ && range_pruned_fragments && !build_filter.enabled &&
        distinct_count < device_payload_count) {
      throw HashJoinFail(
          "Range-pruned ranked bitmap build requires unique build-side keys");
    }
    if (!payload_free_unique_probe_ && !build_filter.enabled && distinct_count > 0 &&
        distinct_count < device_payload_count) {
      if (ranked_hash_table->hasSegmentedLayout()) {
        ranked_hash_table->allocateOneToManyBuffers(distinct_count, max_gpu_slab_size);
        ranked_hash_table->copyGpuHeaderToDevice(device_allocator);
      } else {
        auto cuda_mgr = executor_->getDataMgr()->getCudaMgr();
        CHECK(cuda_mgr);
        ranked_hash_table =
            promote_ranked_bitmap_to_one_to_many_gpu(*ranked_hash_table,
                                                     bit_count_,
                                                     device_payload_count,
                                                     distinct_count,
                                                     max_gpu_slab_size,
                                                     executor_->getDataMgr(),
                                                     device_allocator,
                                                     cuda_mgr,
                                                     device_id);
      }
      bitmap = ranked_hash_table->getGpuBitmap();
      rank_blocks = ranked_hash_table->getGpuRankBlocks();
      auto counts = ranked_hash_table->getGpuCounts();
      auto offsets = ranked_hash_table->getGpuOffsets();
      const auto count_bytes = distinct_count * sizeof(uint32_t);
      device_allocator->zeroDeviceMem(reinterpret_cast<int8_t*>(counts), count_bytes);
      count_ranked_bitmap_matches_on_device(counts,
                                            bitmap,
                                            rank_blocks,
                                            join_column,
                                            type_info,
                                            col_range_.getIntMin(),
                                            col_range_.getIntMax(),
                                            RankedBitmapHashTable::kRankBlockWordCount,
                                            cuda_stream);
      const auto payload_capacity =
          exclusive_scan_counts_gpu(device_allocator,
                                    counts,
                                    offsets,
                                    distinct_count,
                                    device_payload_count,
                                    ranked_hash_table->getPayloadChunkWordCount(),
                                    cuda_stream);
      if (payload_capacity != ranked_hash_table->getPayloadCount()) {
        ranked_hash_table->resizeOneToManyPayloadBuffer(payload_capacity,
                                                        max_gpu_slab_size);
        ranked_hash_table->copyGpuHeaderToDevice(device_allocator);
        counts = ranked_hash_table->getGpuCounts();
        offsets = ranked_hash_table->getGpuOffsets();
      }
      device_allocator->zeroDeviceMem(reinterpret_cast<int8_t*>(counts), count_bytes);
      int err{0};
      auto dev_err_buff = device_allocator->alloc(sizeof(int));
      device_allocator->copyToDevice(
          dev_err_buff, &err, sizeof(err), "Ranked bitmap join error buffer");
      fill_ranked_bitmap_payload_one_to_many_segmented_on_device(
          ranked_hash_table->getGpuPayloadPtrs(),
          ranked_hash_table->getPayloadChunkWordCount(),
          counts,
          offsets,
          bitmap,
          rank_blocks,
          join_column,
          type_info,
          col_range_.getIntMin(),
          col_range_.getIntMax(),
          RankedBitmapHashTable::kRankBlockWordCount,
          ranked_hash_table->getPayloadCount(),
          reinterpret_cast<int*>(dev_err_buff),
          cuda_stream);
      device_allocator->copyFromDevice(
          &err, dev_err_buff, sizeof(err), "Ranked bitmap join error code");
      if (err) {
        throw HashJoinFail("Ranked bitmap one-to-many payload build failed");
      }
      hash_type_.store(HashType::OneToMany, std::memory_order_relaxed);
    } else if (!payload_free_unique_probe_) {
      if (build_filter.enabled) {
        CHECK_LE(distinct_count, ranked_hash_table->getPayloadCount());
      } else {
        CHECK_EQ(distinct_count, ranked_hash_table->getPayloadCount());
      }
      const bool unique_unfiltered_payload = !build_filter.enabled;
      if (!unique_unfiltered_payload && ranked_hash_table->hasSegmentedLayout()) {
        for (size_t chunk_idx = 0; chunk_idx < ranked_hash_table->getPayloadChunkCount();
             ++chunk_idx) {
          init_hash_join_buff_on_device(
              reinterpret_cast<int32_t*>(
                  ranked_hash_table->getGpuPayloadChunk(chunk_idx)),
              ranked_hash_table->getPayloadChunkWordCount(chunk_idx),
              -1,
              cuda_stream);
        }
      } else if (!unique_unfiltered_payload) {
        init_hash_join_buff_on_device(reinterpret_cast<int32_t*>(payload),
                                      ranked_hash_table->getPayloadCount(),
                                      -1,
                                      cuda_stream);
      }
      int err{0};
      auto dev_err_buff = device_allocator->alloc(sizeof(int));
      device_allocator->copyToDevice(
          dev_err_buff, &err, sizeof(err), "Ranked bitmap join error buffer");
      if (unique_unfiltered_payload && ranked_hash_table->hasSegmentedLayout()) {
        fill_ranked_bitmap_payload_unique_segmented_on_device(
            ranked_hash_table->getGpuPayloadPtrs(),
            ranked_hash_table->getPayloadChunkWordCount(),
            bitmap,
            rank_blocks,
            join_column,
            type_info,
            col_range_.getIntMin(),
            col_range_.getIntMax(),
            RankedBitmapHashTable::kRankBlockWordCount,
            ranked_hash_table->getPayloadCount(),
            reinterpret_cast<int*>(dev_err_buff),
            cuda_stream);
      } else if (ranked_hash_table->hasSegmentedLayout()) {
        fill_ranked_bitmap_payload_segmented_on_device(
            ranked_hash_table->getGpuPayloadPtrs(),
            ranked_hash_table->getPayloadChunkWordCount(),
            bitmap,
            rank_blocks,
            join_column,
            type_info,
            build_filter,
            col_range_.getIntMin(),
            col_range_.getIntMax(),
            RankedBitmapHashTable::kRankBlockWordCount,
            ranked_hash_table->getPayloadCount(),
            reinterpret_cast<int*>(dev_err_buff),
            cuda_stream);
      } else if (unique_unfiltered_payload) {
        fill_ranked_bitmap_payload_unique_on_device(
            payload,
            bitmap,
            rank_blocks,
            join_column,
            type_info,
            col_range_.getIntMin(),
            col_range_.getIntMax(),
            RankedBitmapHashTable::kRankBlockWordCount,
            ranked_hash_table->getPayloadCount(),
            reinterpret_cast<int*>(dev_err_buff),
            cuda_stream);
      } else {
        fill_ranked_bitmap_payload_on_device(payload,
                                             bitmap,
                                             rank_blocks,
                                             join_column,
                                             type_info,
                                             build_filter,
                                             col_range_.getIntMin(),
                                             col_range_.getIntMax(),
                                             RankedBitmapHashTable::kRankBlockWordCount,
                                             ranked_hash_table->getPayloadCount(),
                                             reinterpret_cast<int*>(dev_err_buff),
                                             cuda_stream);
      }
      device_allocator->copyFromDevice(
          &err, dev_err_buff, sizeof(err), "Ranked bitmap join error code");
      if (err) {
        throw HashJoinFail("Ranked bitmap join key is not unique");
      }
    }
#else
    UNREACHABLE();
#endif
  }
  std::shared_ptr<HashTable> hash_table = std::move(ranked_hash_table);
  moveHashTableForDevice(std::move(hash_table), device_id);
}

llvm::Value* RankedBitmapJoinHashTable::codegenSlot(const CompilationOptions& co,
                                                    const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->getCgenStatePtr());
  CodeGenerator code_generator(executor_);
  auto key_lv =
      HashJoin::codegenColOrStringOper(outer_expr_.get(), {}, code_generator, co);
  auto cgen_state = executor_->getCgenStatePtr();
  auto hash_ptr = HashJoin::codegenHashTableLoad(index, executor_);
  auto hash_ptr_ty = llvm::Type::getInt8PtrTy(cgen_state->context_);
  llvm::Value* hash_buff{nullptr};
  if (hash_ptr->getType()->isPointerTy()) {
    hash_buff = cgen_state->ir_builder_.CreatePointerCast(hash_ptr, hash_ptr_ty);
  } else {
    hash_buff = cgen_state->ir_builder_.CreateIntToPtr(hash_ptr, hash_ptr_ty);
  }

  const auto key_i64 = cgen_state->castToTypeIn(key_lv, 64);
  const auto key_ti = get_logical_type_info(outer_expr_->get_type_info());
  const auto hash_table = std::dynamic_pointer_cast<RankedBitmapHashTable>(
      getHashTableForDevice(*device_ids_.begin()));
  CHECK(hash_table);

  auto& builder = cgen_state->ir_builder_;
  auto& context = cgen_state->context_;
  auto i32_ty = get_int_type(32, context);
  auto i64_ty = get_int_type(64, context);
  auto i8_ptr_ty = llvm::Type::getInt8PtrTy(context);
  auto i32_ptr_ty = llvm::Type::getInt32PtrTy(context);
  auto i64_ptr_ty = llvm::Type::getInt64PtrTy(context);
  auto invalid_slot = cgen_state->llInt(int64_t(-1));
  auto min_val = cgen_state->llInt(col_range_.getIntMin());
  auto max_val = cgen_state->llInt(col_range_.getIntMax());
  auto null_val = cgen_state->llInt(inline_fixed_encoding_null_val(key_ti));
  auto hash_null = builder.CreateICmpEQ(
      hash_buff,
      llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(i8_ptr_ty)));
  auto key_is_null = builder.CreateICmpEQ(key_i64, null_val);
  auto key_lt_min = builder.CreateICmpSLT(key_i64, min_val);
  auto key_gt_max = builder.CreateICmpSGT(key_i64, max_val);
  auto invalid_key = builder.CreateOr(builder.CreateOr(hash_null, key_is_null),
                                      builder.CreateOr(key_lt_min, key_gt_max));

  auto lookup_bb = llvm::BasicBlock::Create(
      context, "ranked_bitmap_lookup", cgen_state->current_func_);
  auto missing_bb = llvm::BasicBlock::Create(
      context, "ranked_bitmap_missing", cgen_state->current_func_);
  auto present_bb = llvm::BasicBlock::Create(
      context, "ranked_bitmap_present", cgen_state->current_func_);
  auto exit_bb =
      llvm::BasicBlock::Create(context, "ranked_bitmap_exit", cgen_state->current_func_);

  auto invalid_bb = builder.GetInsertBlock();
  builder.CreateCondBr(invalid_key, exit_bb, lookup_bb);

  builder.SetInsertPoint(lookup_bb);
  llvm::Value* bitmap{nullptr};
  llvm::Value* rank_blocks{nullptr};
  llvm::Value* payload{nullptr};
  llvm::Value* payload_ptrs{nullptr};
  llvm::Value* payload_chunk_word_count{nullptr};
  if (hash_table->hasSegmentedLayout()) {
    auto header = builder.CreatePointerCast(hash_buff, i64_ptr_ty);
    auto load_header_ptr = [&](const size_t header_idx, llvm::Type* ptr_ty) {
      auto addr = builder.CreateLoad(
          i64_ty,
          builder.CreateGEP(
              i64_ty, header, cgen_state->llInt(static_cast<int64_t>(header_idx))));
      return builder.CreateIntToPtr(addr, ptr_ty);
    };
    bitmap = load_header_ptr(RankedBitmapHashTable::kHeaderBitmapPtr, i32_ptr_ty);
    rank_blocks =
        load_header_ptr(RankedBitmapHashTable::kHeaderRankBlocksPtr, i32_ptr_ty);
    payload_ptrs =
        load_header_ptr(RankedBitmapHashTable::kHeaderPayloadPtrsPtr, i64_ptr_ty);
    payload_chunk_word_count = builder.CreateLoad(
        i64_ty,
        builder.CreateGEP(i64_ty,
                          header,
                          cgen_state->llInt(static_cast<int64_t>(
                              RankedBitmapHashTable::kHeaderPayloadChunkWords))));
  } else {
    bitmap = builder.CreatePointerCast(hash_buff, i32_ptr_ty);
    rank_blocks = builder.CreateGEP(
        i32_ty,
        bitmap,
        cgen_state->llInt(static_cast<int64_t>(hash_table->getBitmapWordCount())));
    payload = builder.CreateGEP(
        i32_ty,
        rank_blocks,
        cgen_state->llInt(static_cast<int64_t>(hash_table->getRankBlockCount())));
  }
  auto bitmap_idx = builder.CreateSub(key_i64, min_val);
  auto word_idx = builder.CreateLShr(bitmap_idx, cgen_state->llInt(int64_t(5)));
  auto word_ptr = builder.CreateGEP(i32_ty, bitmap, word_idx);
  auto word = builder.CreateLoad(i32_ty, word_ptr);
  auto bit_idx_i64 = builder.CreateAnd(bitmap_idx, cgen_state->llInt(int64_t(31)));
  auto bit_idx_i32 = builder.CreateTrunc(bit_idx_i64, i32_ty);
  auto bit_mask = builder.CreateShl(llvm::ConstantInt::get(i32_ty, 1), bit_idx_i32);
  auto word_has_key = builder.CreateICmpNE(builder.CreateAnd(word, bit_mask),
                                           llvm::ConstantInt::get(i32_ty, 0));
  builder.CreateCondBr(word_has_key, present_bb, missing_bb);

  builder.SetInsertPoint(missing_bb);
  builder.CreateBr(exit_bb);
  auto missing_exit_bb = builder.GetInsertBlock();

  builder.SetInsertPoint(present_bb);
  if (hash_table->isPayloadFree()) {
    builder.CreateBr(exit_bb);
    auto present_exit_bb = builder.GetInsertBlock();

    builder.SetInsertPoint(exit_bb);
    auto slot = builder.CreatePHI(i64_ty, 3);
    slot->addIncoming(invalid_slot, invalid_bb);
    slot->addIncoming(invalid_slot, missing_exit_bb);
    slot->addIncoming(cgen_state->llInt(int64_t(0)), present_exit_bb);
    return slot;
  }
  auto rank_block_word_count =
      cgen_state->llInt(static_cast<int64_t>(RankedBitmapHashTable::kRankBlockWordCount));
  auto block_idx = builder.CreateUDiv(word_idx, rank_block_word_count);
  auto block_start_word = builder.CreateMul(block_idx, rank_block_word_count);
  auto rank_ptr = builder.CreateGEP(i32_ty, rank_blocks, block_idx);
  llvm::Value* rank = builder.CreateLoad(i32_ty, rank_ptr);
  for (size_t lane = 0; lane < RankedBitmapHashTable::kRankBlockWordCount; ++lane) {
    auto lane_word_idx = builder.CreateAdd(block_start_word,
                                           cgen_state->llInt(static_cast<int64_t>(lane)));
    auto needs_lane = builder.CreateICmpULT(lane_word_idx, word_idx);
    auto safe_lane_word_idx = builder.CreateSelect(needs_lane, lane_word_idx, word_idx);
    auto lane_word_ptr = builder.CreateGEP(i32_ty, bitmap, safe_lane_word_idx);
    auto lane_word = builder.CreateLoad(i32_ty, lane_word_ptr);
    auto lane_popcount = codegen_popcount32(cgen_state, lane_word);
    rank = builder.CreateAdd(
        rank,
        builder.CreateSelect(
            needs_lane, lane_popcount, llvm::ConstantInt::get(i32_ty, 0)));
  }
  auto lower_bits_mask =
      builder.CreateSub(builder.CreateShl(llvm::ConstantInt::get(i32_ty, 1), bit_idx_i32),
                        llvm::ConstantInt::get(i32_ty, 1));
  rank = builder.CreateAdd(
      rank, codegen_popcount32(cgen_state, builder.CreateAnd(word, lower_bits_mask)));
  auto payload_idx = builder.CreateZExt(rank, i64_ty);
  llvm::Value* payload_ptr{nullptr};
  if (hash_table->hasSegmentedLayout()) {
    auto chunk_idx = builder.CreateUDiv(payload_idx, payload_chunk_word_count);
    auto chunk_offset = builder.CreateURem(payload_idx, payload_chunk_word_count);
    auto chunk_addr =
        builder.CreateLoad(i64_ty, builder.CreateGEP(i64_ty, payload_ptrs, chunk_idx));
    auto chunk_ptr = builder.CreateIntToPtr(chunk_addr, i32_ptr_ty);
    payload_ptr = builder.CreateGEP(i32_ty, chunk_ptr, chunk_offset);
  } else {
    payload_ptr = builder.CreateGEP(i32_ty, payload, payload_idx);
  }
  auto payload_i32 = builder.CreateLoad(i32_ty, payload_ptr);
  auto payload_i64 = builder.CreateZExt(payload_i32, i64_ty);
  builder.CreateBr(exit_bb);
  auto present_exit_bb = builder.GetInsertBlock();

  builder.SetInsertPoint(exit_bb);
  auto slot = builder.CreatePHI(i64_ty, 3);
  slot->addIncoming(invalid_slot, invalid_bb);
  slot->addIncoming(invalid_slot, missing_exit_bb);
  slot->addIncoming(payload_i64, present_exit_bb);
  return slot;
}

HashJoinMatchingSet RankedBitmapJoinHashTable::codegenMatchingSet(
    const CompilationOptions& co,
    const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->getCgenStatePtr());
  CodeGenerator code_generator(executor_);
  auto key_lv =
      HashJoin::codegenColOrStringOper(outer_expr_.get(), {}, code_generator, co);
  auto cgen_state = executor_->getCgenStatePtr();
  auto hash_ptr = HashJoin::codegenHashTableLoad(index, executor_);
  auto hash_ptr_ty = llvm::Type::getInt8PtrTy(cgen_state->context_);
  llvm::Value* hash_buff{nullptr};
  if (hash_ptr->getType()->isPointerTy()) {
    hash_buff = cgen_state->ir_builder_.CreatePointerCast(hash_ptr, hash_ptr_ty);
  } else {
    hash_buff = cgen_state->ir_builder_.CreateIntToPtr(hash_ptr, hash_ptr_ty);
  }

  const auto hash_table = std::dynamic_pointer_cast<RankedBitmapHashTable>(
      getHashTableForDevice(*device_ids_.begin()));
  CHECK(hash_table);
  CHECK(hash_table->getLayout() == HashType::OneToMany);
  CHECK(hash_table->hasSegmentedLayout());

  auto& builder = cgen_state->ir_builder_;
  auto& context = cgen_state->context_;
  auto i32_ty = get_int_type(32, context);
  auto i64_ty = get_int_type(64, context);
  auto i8_ptr_ty = llvm::Type::getInt8PtrTy(context);
  auto i32_ptr_ty = llvm::Type::getInt32PtrTy(context);
  auto i64_ptr_ty = llvm::Type::getInt64PtrTy(context);
  auto min_val = cgen_state->llInt(col_range_.getIntMin());
  auto max_val = cgen_state->llInt(col_range_.getIntMax());
  const auto key_i64 = cgen_state->castToTypeIn(key_lv, 64);
  const auto key_ti = get_logical_type_info(outer_expr_->get_type_info());
  auto null_val = cgen_state->llInt(inline_fixed_encoding_null_val(key_ti));
  auto hash_null = builder.CreateICmpEQ(
      hash_buff,
      llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(i8_ptr_ty)));
  auto key_is_null = builder.CreateICmpEQ(key_i64, null_val);
  auto key_lt_min = builder.CreateICmpSLT(key_i64, min_val);
  auto key_gt_max = builder.CreateICmpSGT(key_i64, max_val);
  auto invalid_key = builder.CreateOr(builder.CreateOr(hash_null, key_is_null),
                                      builder.CreateOr(key_lt_min, key_gt_max));

  auto safe_elements = builder.CreateAlloca(i32_ty, nullptr, "ranked_bitmap_empty_rows");
  builder.CreateStore(llvm::ConstantInt::get(i32_ty, 0), safe_elements);

  auto lookup_bb = llvm::BasicBlock::Create(
      context, "ranked_bitmap_set_lookup", cgen_state->current_func_);
  auto missing_bb = llvm::BasicBlock::Create(
      context, "ranked_bitmap_set_missing", cgen_state->current_func_);
  auto present_bb = llvm::BasicBlock::Create(
      context, "ranked_bitmap_set_present", cgen_state->current_func_);
  auto exit_bb = llvm::BasicBlock::Create(
      context, "ranked_bitmap_set_exit", cgen_state->current_func_);

  auto invalid_bb = builder.GetInsertBlock();
  builder.CreateCondBr(invalid_key, exit_bb, lookup_bb);

  builder.SetInsertPoint(lookup_bb);
  auto header = builder.CreatePointerCast(hash_buff, i64_ptr_ty);
  auto load_header_ptr = [&](const size_t header_idx, llvm::Type* ptr_ty) {
    auto addr = builder.CreateLoad(
        i64_ty,
        builder.CreateGEP(
            i64_ty, header, cgen_state->llInt(static_cast<int64_t>(header_idx))));
    return builder.CreateIntToPtr(addr, ptr_ty);
  };
  auto bitmap = load_header_ptr(RankedBitmapHashTable::kHeaderBitmapPtr, i32_ptr_ty);
  auto rank_blocks =
      load_header_ptr(RankedBitmapHashTable::kHeaderRankBlocksPtr, i32_ptr_ty);
  auto payload_ptrs =
      load_header_ptr(RankedBitmapHashTable::kHeaderPayloadPtrsPtr, i64_ptr_ty);
  auto payload_chunk_word_count = builder.CreateLoad(
      i64_ty,
      builder.CreateGEP(i64_ty,
                        header,
                        cgen_state->llInt(static_cast<int64_t>(
                            RankedBitmapHashTable::kHeaderPayloadChunkWords))));
  auto counts = load_header_ptr(RankedBitmapHashTable::kHeaderCountsPtr, i32_ptr_ty);
  auto offsets = load_header_ptr(RankedBitmapHashTable::kHeaderOffsetsPtr, i32_ptr_ty);

  auto bitmap_idx = builder.CreateSub(key_i64, min_val);
  auto word_idx = builder.CreateLShr(bitmap_idx, cgen_state->llInt(int64_t(5)));
  auto word_ptr = builder.CreateGEP(i32_ty, bitmap, word_idx);
  auto word = builder.CreateLoad(i32_ty, word_ptr);
  auto bit_idx_i64 = builder.CreateAnd(bitmap_idx, cgen_state->llInt(int64_t(31)));
  auto bit_idx_i32 = builder.CreateTrunc(bit_idx_i64, i32_ty);
  auto bit_mask = builder.CreateShl(llvm::ConstantInt::get(i32_ty, 1), bit_idx_i32);
  auto word_has_key = builder.CreateICmpNE(builder.CreateAnd(word, bit_mask),
                                           llvm::ConstantInt::get(i32_ty, 0));
  builder.CreateCondBr(word_has_key, present_bb, missing_bb);

  builder.SetInsertPoint(missing_bb);
  builder.CreateBr(exit_bb);
  auto missing_exit_bb = builder.GetInsertBlock();

  builder.SetInsertPoint(present_bb);
  auto rank_block_word_count =
      cgen_state->llInt(static_cast<int64_t>(RankedBitmapHashTable::kRankBlockWordCount));
  auto block_idx = builder.CreateUDiv(word_idx, rank_block_word_count);
  auto block_start_word = builder.CreateMul(block_idx, rank_block_word_count);
  auto rank_ptr = builder.CreateGEP(i32_ty, rank_blocks, block_idx);
  llvm::Value* rank = builder.CreateLoad(i32_ty, rank_ptr);
  for (size_t lane = 0; lane < RankedBitmapHashTable::kRankBlockWordCount; ++lane) {
    auto lane_word_idx = builder.CreateAdd(block_start_word,
                                           cgen_state->llInt(static_cast<int64_t>(lane)));
    auto needs_lane = builder.CreateICmpULT(lane_word_idx, word_idx);
    auto safe_lane_word_idx = builder.CreateSelect(needs_lane, lane_word_idx, word_idx);
    auto lane_word_ptr = builder.CreateGEP(i32_ty, bitmap, safe_lane_word_idx);
    auto lane_word = builder.CreateLoad(i32_ty, lane_word_ptr);
    auto lane_popcount = codegen_popcount32(cgen_state, lane_word);
    rank = builder.CreateAdd(
        rank,
        builder.CreateSelect(
            needs_lane, lane_popcount, llvm::ConstantInt::get(i32_ty, 0)));
  }
  auto lower_bits_mask =
      builder.CreateSub(builder.CreateShl(llvm::ConstantInt::get(i32_ty, 1), bit_idx_i32),
                        llvm::ConstantInt::get(i32_ty, 1));
  rank = builder.CreateAdd(
      rank, codegen_popcount32(cgen_state, builder.CreateAnd(word, lower_bits_mask)));
  auto rank_i64 = builder.CreateZExt(rank, i64_ty);
  auto count_i32 =
      builder.CreateLoad(i32_ty, builder.CreateGEP(i32_ty, counts, rank_i64));
  auto count_i64 = builder.CreateZExt(count_i32, i64_ty);
  auto payload_idx = builder.CreateZExt(
      builder.CreateLoad(i32_ty, builder.CreateGEP(i32_ty, offsets, rank_i64)), i64_ty);
  auto chunk_idx = builder.CreateUDiv(payload_idx, payload_chunk_word_count);
  auto chunk_offset = builder.CreateURem(payload_idx, payload_chunk_word_count);
  auto chunk_addr =
      builder.CreateLoad(i64_ty, builder.CreateGEP(i64_ty, payload_ptrs, chunk_idx));
  auto chunk_ptr = builder.CreateIntToPtr(chunk_addr, i32_ptr_ty);
  auto elements = builder.CreateGEP(i32_ty, chunk_ptr, chunk_offset);
  auto chunk_remaining = builder.CreateSub(payload_chunk_word_count, chunk_offset);
  auto crosses_chunk = builder.CreateICmpUGT(count_i64, chunk_remaining);
  auto error_code = builder.CreateSelect(
      crosses_chunk,
      cgen_state->llInt(int32_t(heavyai::ErrorCode::OVERFLOW_OR_UNDERFLOW)),
      cgen_state->llInt(int32_t(0)));
  builder.CreateBr(exit_bb);
  auto present_exit_bb = builder.GetInsertBlock();

  builder.SetInsertPoint(exit_bb);
  auto elements_phi = builder.CreatePHI(i32_ptr_ty, 3);
  elements_phi->addIncoming(safe_elements, invalid_bb);
  elements_phi->addIncoming(safe_elements, missing_exit_bb);
  elements_phi->addIncoming(elements, present_exit_bb);

  auto count_phi = builder.CreatePHI(i64_ty, 3);
  count_phi->addIncoming(cgen_state->llInt(int64_t(0)), invalid_bb);
  count_phi->addIncoming(cgen_state->llInt(int64_t(0)), missing_exit_bb);
  count_phi->addIncoming(count_i64, present_exit_bb);

  auto slot_phi = builder.CreatePHI(i64_ty, 3);
  slot_phi->addIncoming(cgen_state->llInt(int64_t(-1)), invalid_bb);
  slot_phi->addIncoming(cgen_state->llInt(int64_t(-1)), missing_exit_bb);
  slot_phi->addIncoming(rank_i64, present_exit_bb);

  auto error_phi = builder.CreatePHI(i32_ty, 3);
  error_phi->addIncoming(cgen_state->llInt(int32_t(0)), invalid_bb);
  error_phi->addIncoming(cgen_state->llInt(int32_t(0)), missing_exit_bb);
  error_phi->addIncoming(error_code, present_exit_bb);

  return {elements_phi, count_phi, slot_phi, error_phi};
}

std::string RankedBitmapJoinHashTable::toString(const ExecutorDeviceType device_type,
                                                const int device_id,
                                                bool) const {
  auto hash_table = getHashTableForDevice(device_id);
  std::ostringstream oss;
  oss << "ranked bitmap " << ::toString(device_type) << " join hash table"
      << ", bits: " << bit_count_ << ", payload rows: " << payload_count_ << ", bytes: "
      << (hash_table ? hash_table->getHashTableBufferSize(device_type) : table_bytes_)
      << ", range: [" << col_range_.getIntMin() << ", " << col_range_.getIntMax() << "]";
  return oss.str();
}
