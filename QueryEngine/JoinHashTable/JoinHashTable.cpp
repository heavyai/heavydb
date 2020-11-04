/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "QueryEngine/JoinHashTable/JoinHashTable.h"

#include <atomic>
#include <future>
#include <numeric>
#include <thread>

#include "Logger/Logger.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/JoinHashTable/HashJoinRuntime.h"
#include "QueryEngine/JoinHashTable/PerfectHashTableBuilder.h"
#include "QueryEngine/RangeTableIndexVisitor.h"
#include "QueryEngine/RuntimeFunctions.h"

InnerOuter normalize_column_pair(const Analyzer::Expr* lhs,
                                 const Analyzer::Expr* rhs,
                                 const Catalog_Namespace::Catalog& cat,
                                 const TemporaryTables* temporary_tables,
                                 const bool is_overlaps_join) {
  const auto& lhs_ti = lhs->get_type_info();
  const auto& rhs_ti = rhs->get_type_info();
  if (!is_overlaps_join) {
    if (lhs_ti.get_type() != rhs_ti.get_type()) {
      throw HashJoinFail("Equijoin types must be identical, found: " +
                         lhs_ti.get_type_name() + ", " + rhs_ti.get_type_name());
    }
    if (!lhs_ti.is_integer() && !lhs_ti.is_time() && !lhs_ti.is_string() &&
        !lhs_ti.is_decimal()) {
      throw HashJoinFail("Cannot apply hash join to inner column type " +
                         lhs_ti.get_type_name());
    }
    // Decimal types should be identical.
    if (lhs_ti.is_decimal() && (lhs_ti.get_scale() != rhs_ti.get_scale() ||
                                lhs_ti.get_precision() != rhs_ti.get_precision())) {
      throw HashJoinFail("Equijoin with different decimal types");
    }
  }

  const auto lhs_cast = dynamic_cast<const Analyzer::UOper*>(lhs);
  const auto rhs_cast = dynamic_cast<const Analyzer::UOper*>(rhs);
  if (lhs_ti.is_string() && (static_cast<bool>(lhs_cast) != static_cast<bool>(rhs_cast) ||
                             (lhs_cast && lhs_cast->get_optype() != kCAST) ||
                             (rhs_cast && rhs_cast->get_optype() != kCAST))) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  // Casts to decimal are not suported.
  if (lhs_ti.is_decimal() && (lhs_cast || rhs_cast)) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  const auto lhs_col =
      lhs_cast ? dynamic_cast<const Analyzer::ColumnVar*>(lhs_cast->get_operand())
               : dynamic_cast<const Analyzer::ColumnVar*>(lhs);
  const auto rhs_col =
      rhs_cast ? dynamic_cast<const Analyzer::ColumnVar*>(rhs_cast->get_operand())
               : dynamic_cast<const Analyzer::ColumnVar*>(rhs);
  if (!lhs_col && !rhs_col) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  const Analyzer::ColumnVar* inner_col{nullptr};
  const Analyzer::ColumnVar* outer_col{nullptr};
  auto outer_ti = lhs_ti;
  auto inner_ti = rhs_ti;
  const Analyzer::Expr* outer_expr{lhs};
  if ((!lhs_col || (rhs_col && lhs_col->get_rte_idx() < rhs_col->get_rte_idx())) &&
      (!rhs_col || (!lhs_col || lhs_col->get_rte_idx() < rhs_col->get_rte_idx()))) {
    inner_col = rhs_col;
    outer_col = lhs_col;
  } else {
    if (lhs_col && lhs_col->get_rte_idx() == 0) {
      throw HashJoinFail("Cannot use hash join for given expression");
    }
    inner_col = lhs_col;
    outer_col = rhs_col;
    std::swap(outer_ti, inner_ti);
    outer_expr = rhs;
  }
  if (!inner_col) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  if (!outer_col) {
    MaxRangeTableIndexVisitor rte_idx_visitor;
    int outer_rte_idx = rte_idx_visitor.visit(outer_expr);
    // The inner column candidate is not actually inner; the outer
    // expression contains columns which are at least as deep.
    if (inner_col->get_rte_idx() <= outer_rte_idx) {
      throw HashJoinFail("Cannot use hash join for given expression");
    }
  }
  // We need to fetch the actual type information from the catalog since Analyzer
  // always reports nullable as true for inner table columns in left joins.
  const auto inner_col_cd = get_column_descriptor_maybe(
      inner_col->get_column_id(), inner_col->get_table_id(), cat);
  const auto inner_col_real_ti = get_column_type(inner_col->get_column_id(),
                                                 inner_col->get_table_id(),
                                                 inner_col_cd,
                                                 temporary_tables);
  const auto& outer_col_ti =
      !(dynamic_cast<const Analyzer::FunctionOper*>(lhs)) && outer_col
          ? outer_col->get_type_info()
          : outer_ti;
  // Casts from decimal are not supported.
  if ((inner_col_real_ti.is_decimal() || outer_col_ti.is_decimal()) &&
      (lhs_cast || rhs_cast)) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  if (is_overlaps_join) {
    if (!inner_col_real_ti.is_array()) {
      throw HashJoinFail(
          "Overlaps join only supported for inner columns with array type");
    }
    auto is_bounds_array = [](const auto ti) {
      return ti.is_fixlen_array() && ti.get_size() == 32;
    };
    if (!is_bounds_array(inner_col_real_ti)) {
      throw HashJoinFail(
          "Overlaps join only supported for 4-element double fixed length arrays");
    }
    if (!(outer_col_ti.get_type() == kPOINT || is_bounds_array(outer_col_ti))) {
      throw HashJoinFail(
          "Overlaps join only supported for geometry outer columns of type point or "
          "geometry columns with bounds");
    }
  } else {
    if (!(inner_col_real_ti.is_integer() || inner_col_real_ti.is_time() ||
          inner_col_real_ti.is_decimal() ||
          (inner_col_real_ti.is_string() &&
           inner_col_real_ti.get_compression() == kENCODING_DICT))) {
      throw HashJoinFail(
          "Can only apply hash join to integer-like types and dictionary encoded "
          "strings");
    }
  }

  auto normalized_inner_col = inner_col;
  auto normalized_outer_col = outer_col ? outer_col : outer_expr;

  const auto& normalized_inner_ti = normalized_inner_col->get_type_info();
  const auto& normalized_outer_ti = normalized_outer_col->get_type_info();

  if (normalized_inner_ti.is_string() != normalized_outer_ti.is_string()) {
    throw HashJoinFail(std::string("Could not build hash tables for incompatible types " +
                                   normalized_inner_ti.get_type_name() + " and " +
                                   normalized_outer_ti.get_type_name()));
  }

  return {normalized_inner_col, normalized_outer_col};
}

std::vector<InnerOuter> normalize_column_pairs(const Analyzer::BinOper* condition,
                                               const Catalog_Namespace::Catalog& cat,
                                               const TemporaryTables* temporary_tables) {
  std::vector<InnerOuter> result;
  const auto lhs_tuple_expr =
      dynamic_cast<const Analyzer::ExpressionTuple*>(condition->get_left_operand());
  const auto rhs_tuple_expr =
      dynamic_cast<const Analyzer::ExpressionTuple*>(condition->get_right_operand());

  CHECK_EQ(static_cast<bool>(lhs_tuple_expr), static_cast<bool>(rhs_tuple_expr));
  if (lhs_tuple_expr) {
    const auto& lhs_tuple = lhs_tuple_expr->getTuple();
    const auto& rhs_tuple = rhs_tuple_expr->getTuple();
    CHECK_EQ(lhs_tuple.size(), rhs_tuple.size());
    for (size_t i = 0; i < lhs_tuple.size(); ++i) {
      result.push_back(normalize_column_pair(lhs_tuple[i].get(),
                                             rhs_tuple[i].get(),
                                             cat,
                                             temporary_tables,
                                             condition->is_overlaps_oper()));
    }
  } else {
    CHECK(!lhs_tuple_expr && !rhs_tuple_expr);
    result.push_back(normalize_column_pair(condition->get_left_operand(),
                                           condition->get_right_operand(),
                                           cat,
                                           temporary_tables,
                                           condition->is_overlaps_oper()));
  }

  return result;
}

namespace {

InnerOuter get_cols(const Analyzer::BinOper* qual_bin_oper,
                    const Catalog_Namespace::Catalog& cat,
                    const TemporaryTables* temporary_tables) {
  const auto lhs = qual_bin_oper->get_left_operand();
  const auto rhs = qual_bin_oper->get_right_operand();
  return normalize_column_pair(lhs, rhs, cat, temporary_tables);
}

HashEntryInfo get_bucketized_hash_entry_info(SQLTypeInfo const& context_ti,
                                             ExpressionRange const& col_range,
                                             bool const is_bw_eq) {
  using EmptyRangeSize = boost::optional<size_t>;
  auto empty_range_check = [](ExpressionRange const& col_range,
                              bool const is_bw_eq) -> EmptyRangeSize {
    if (col_range.getIntMin() > col_range.getIntMax()) {
      CHECK_EQ(col_range.getIntMin(), int64_t(0));
      CHECK_EQ(col_range.getIntMax(), int64_t(-1));
      if (is_bw_eq) {
        return size_t(1);
      }
      return size_t(0);
    }
    return EmptyRangeSize{};
  };

  auto empty_range = empty_range_check(col_range, is_bw_eq);
  if (empty_range) {
    return {size_t(*empty_range), 1};
  }

  int64_t bucket_normalization =
      context_ti.get_type() == kDATE ? col_range.getBucket() : 1;
  CHECK_GT(bucket_normalization, 0);
  return {size_t(col_range.getIntMax() - col_range.getIntMin() + 1 + (is_bw_eq ? 1 : 0)),
          bucket_normalization};
}

size_t get_hash_entry_count(const ExpressionRange& col_range, const bool is_bw_eq) {
  if (col_range.getIntMin() > col_range.getIntMax()) {
    CHECK_EQ(col_range.getIntMin(), int64_t(0));
    CHECK_EQ(col_range.getIntMax(), int64_t(-1));
    return is_bw_eq ? 1 : 0;
  }
  return col_range.getIntMax() - col_range.getIntMin() + 1 + (is_bw_eq ? 1 : 0);
}

}  // namespace

std::vector<
    std::pair<JoinHashTable::JoinHashTableCacheKey, JoinHashTable::HashTableCacheValue>>
    JoinHashTable::join_hash_table_cache_;
std::mutex JoinHashTable::join_hash_table_cache_mutex_;

size_t get_shard_count(const Analyzer::BinOper* join_condition,
                       const Executor* executor) {
  const Analyzer::ColumnVar* inner_col{nullptr};
  const Analyzer::Expr* outer_col{nullptr};
  std::shared_ptr<Analyzer::BinOper> redirected_bin_oper;
  try {
    std::tie(inner_col, outer_col) =
        get_cols(join_condition, *executor->getCatalog(), executor->getTemporaryTables());
  } catch (...) {
    return 0;
  }
  if (!inner_col || !outer_col) {
    return 0;
  }
  return get_shard_count({inner_col, outer_col}, executor);
}

namespace {

bool shard_count_less_or_equal_device_count(const int inner_table_id,
                                            const Executor* executor) {
  const auto inner_table_info = executor->getTableInfo(inner_table_id);
  std::unordered_set<int> device_holding_fragments;
  auto cuda_mgr = executor->getCatalog()->getDataMgr().getCudaMgr();
  const int device_count = cuda_mgr ? cuda_mgr->getDeviceCount() : 1;
  for (const auto& fragment : inner_table_info.fragments) {
    if (fragment.shard != -1) {
      const auto it_ok = device_holding_fragments.emplace(fragment.shard % device_count);
      if (!it_ok.second) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

size_t get_shard_count(
    std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*> equi_pair,
    const Executor* executor) {
  const auto inner_col = equi_pair.first;
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(equi_pair.second);
  if (!outer_col || inner_col->get_table_id() < 0 || outer_col->get_table_id() < 0) {
    return 0;
  }
  if (outer_col->get_rte_idx()) {
    return 0;
  }
  if (inner_col->get_type_info() != outer_col->get_type_info()) {
    return 0;
  }
  const auto catalog = executor->getCatalog();
  const auto inner_td = catalog->getMetadataForTable(inner_col->get_table_id());
  CHECK(inner_td);
  const auto outer_td = catalog->getMetadataForTable(outer_col->get_table_id());
  CHECK(outer_td);
  if (inner_td->shardedColumnId == 0 || outer_td->shardedColumnId == 0 ||
      inner_td->nShards != outer_td->nShards) {
    return 0;
  }
  if (!shard_count_less_or_equal_device_count(inner_td->tableId, executor)) {
    return 0;
  }
  // The two columns involved must be the ones on which the tables have been sharded on.
  return (inner_td->shardedColumnId == inner_col->get_column_id() &&
          outer_td->shardedColumnId == outer_col->get_column_id()) ||
                 (outer_td->shardedColumnId == inner_col->get_column_id() &&
                  inner_td->shardedColumnId == inner_col->get_column_id())
             ? inner_td->nShards
             : 0;
}

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<JoinHashTable> JoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  decltype(std::chrono::steady_clock::now()) ts1, ts2;
  if (VLOGGING(1)) {
    VLOG(1) << "Building perfect hash table " << getHashTypeString(preferred_hash_type)
            << " for qual: " << qual_bin_oper->toString();
    ts1 = std::chrono::steady_clock::now();
  }
  CHECK(IS_EQUIVALENCE(qual_bin_oper->get_optype()));
  const auto cols =
      get_cols(qual_bin_oper.get(), *executor->getCatalog(), executor->temporary_tables_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& ti = inner_col->get_type_info();
  auto col_range =
      getExpressionRange(ti.is_string() ? cols.second : inner_col, query_infos, executor);
  if (col_range.getType() == ExpressionRangeType::Invalid) {
    throw HashJoinFail(
        "Could not compute range for the expressions involved in the equijoin");
  }
  if (ti.is_string()) {
    // The nullable info must be the same as the source column.
    const auto source_col_range = getExpressionRange(inner_col, query_infos, executor);
    if (source_col_range.getType() == ExpressionRangeType::Invalid) {
      throw HashJoinFail(
          "Could not compute range for the expressions involved in the equijoin");
    }
    if (source_col_range.getIntMin() > source_col_range.getIntMax()) {
      // If the inner column expression range is empty, use the inner col range
      CHECK_EQ(source_col_range.getIntMin(), int64_t(0));
      CHECK_EQ(source_col_range.getIntMax(), int64_t(-1));
      col_range = source_col_range;
    } else {
      col_range = ExpressionRange::makeIntRange(
          std::min(source_col_range.getIntMin(), col_range.getIntMin()),
          std::max(source_col_range.getIntMax(), col_range.getIntMax()),
          0,
          source_col_range.hasNulls());
    }
  }
  // We can't allocate more than 2GB contiguous memory on GPU and each entry is 4 bytes.
  const auto max_hash_entry_count =
      memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL
          ? static_cast<size_t>(std::numeric_limits<int32_t>::max() / sizeof(int32_t))
          : static_cast<size_t>(std::numeric_limits<int32_t>::max());

  auto bucketized_entry_count_info = get_bucketized_hash_entry_info(
      ti, col_range, qual_bin_oper->get_optype() == kBW_EQ);
  auto bucketized_entry_count = bucketized_entry_count_info.getNormalizedHashEntryCount();

  if (bucketized_entry_count > max_hash_entry_count) {
    throw TooManyHashEntries();
  }

  if (qual_bin_oper->get_optype() == kBW_EQ &&
      col_range.getIntMax() >= std::numeric_limits<int64_t>::max()) {
    throw HashJoinFail("Cannot translate null value for kBW_EQ");
  }
  auto join_hash_table =
      std::shared_ptr<JoinHashTable>(new JoinHashTable(qual_bin_oper,
                                                       inner_col,
                                                       query_infos,
                                                       memory_level,
                                                       preferred_hash_type,
                                                       col_range,
                                                       column_cache,
                                                       executor,
                                                       device_count));
  try {
    join_hash_table->reify();
  } catch (const TableMustBeReplicated& e) {
    // Throw a runtime error to abort the query
    join_hash_table->freeHashBufferMemory();
    throw std::runtime_error(e.what());
  } catch (const HashJoinFail& e) {
    // HashJoinFail exceptions log an error and trigger a retry with a join loop (if
    // possible)
    join_hash_table->freeHashBufferMemory();
    throw HashJoinFail(std::string("Could not build a 1-to-1 correspondence for columns "
                                   "involved in equijoin | ") +
                       e.what());
  } catch (const ColumnarConversionNotSupported& e) {
    throw HashJoinFail(std::string("Could not build hash tables for equijoin | ") +
                       e.what());
  } catch (const OutOfMemory& e) {
    throw HashJoinFail(
        std::string("Ran out of memory while building hash tables for equijoin | ") +
        e.what());
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("Fatal error while attempting to build hash tables for join: ") +
        e.what());
  }
  if (VLOGGING(1)) {
    ts2 = std::chrono::steady_clock::now();
    VLOG(1) << "Built perfect hash table "
            << getHashTypeString(join_hash_table->getHashType()) << " in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(ts2 - ts1).count()
            << " ms";
  }
  return join_hash_table;
}

bool needs_dictionary_translation(const Analyzer::ColumnVar* inner_col,
                                  const Analyzer::Expr* outer_col_expr,
                                  const Executor* executor) {
  const auto catalog = executor->getCatalog();
  CHECK(catalog);
  const auto inner_cd = get_column_descriptor_maybe(
      inner_col->get_column_id(), inner_col->get_table_id(), *catalog);
  const auto& inner_ti = get_column_type(inner_col->get_column_id(),
                                         inner_col->get_table_id(),
                                         inner_cd,
                                         executor->getTemporaryTables());
  // Only strings may need dictionary translation.
  if (!inner_ti.is_string()) {
    return false;
  }
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(outer_col_expr);
  CHECK(outer_col);
  const auto outer_cd = get_column_descriptor_maybe(
      outer_col->get_column_id(), outer_col->get_table_id(), *catalog);
  // Don't want to deal with temporary tables for now, require translation.
  if (!inner_cd || !outer_cd) {
    return true;
  }
  const auto& outer_ti = get_column_type(outer_col->get_column_id(),
                                         outer_col->get_table_id(),
                                         outer_cd,
                                         executor->getTemporaryTables());
  CHECK_EQ(inner_ti.is_string(), outer_ti.is_string());
  // If the two columns don't share the dictionary, translation is needed.
  return outer_ti.get_comp_param() != inner_ti.get_comp_param();
}

std::vector<Fragmenter_Namespace::FragmentInfo> only_shards_for_device(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    const int device_count) {
  std::vector<Fragmenter_Namespace::FragmentInfo> shards_for_device;
  for (const auto& fragment : fragments) {
    CHECK_GE(fragment.shard, 0);
    if (fragment.shard % device_count == device_id) {
      shards_for_device.push_back(fragment);
    }
  }
  return shards_for_device;
}

void JoinHashTable::reify() {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_LT(0, device_count_);
  catalog_ = const_cast<Catalog_Namespace::Catalog*>(executor_->getCatalog());
  const auto cols =
      get_cols(qual_bin_oper_.get(), *catalog_, executor_->temporary_tables_);
  const auto inner_col = cols.first;
  checkHashJoinReplicationConstraint(inner_col->get_table_id());
  const auto& query_info = getInnerQueryInfo(inner_col).info;
  if (query_info.fragments.empty()) {
    return;
  }
  if (query_info.getNumTuplesUpperBound() >
      static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw TooManyHashEntries();
  }
  std::vector<std::future<void>> init_threads;
  const int shard_count = shardCount();

  inner_outer_pairs_.push_back(cols);
  CHECK_EQ(inner_outer_pairs_.size(), size_t(1));

  std::vector<ColumnsForDevice> columns_per_device;
  std::vector<std::unique_ptr<CudaAllocator>> dev_buff_owners;
  try {
    auto& data_mgr = catalog_->getDataMgr();
    if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      for (int device_id = 0; device_id < device_count_; ++device_id) {
        dev_buff_owners.emplace_back(
            std::make_unique<CudaAllocator>(&data_mgr, device_id));
      }
    }
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      const auto fragments =
          shard_count
              ? only_shards_for_device(query_info.fragments, device_id, device_count_)
              : query_info.fragments;
      const auto columns_for_device =
          fetchColumnsForDevice(fragments,
                                device_id,
                                memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL
                                    ? dev_buff_owners[device_id].get()
                                    : nullptr);
      columns_per_device.push_back(columns_for_device);
      const auto hash_table_key = genHashTableKey(
          fragments, inner_outer_pairs_.front().second, inner_outer_pairs_.front().first);
      init_threads.push_back(std::async(std::launch::async,
                                        &JoinHashTable::reifyForDevice,
                                        this,
                                        hash_table_key,
                                        columns_per_device[device_id],
                                        hash_type_,
                                        device_id,
                                        logger::thread_id()));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }

  } catch (const NeedsOneToManyHash& e) {
    hash_type_ = JoinHashTableInterface::HashType::OneToMany;
    freeHashBufferMemory();
    init_threads.clear();
    if (memory_level_ == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      CHECK_EQ(dev_buff_owners.size(), size_t(device_count_));
    }
    CHECK_EQ(columns_per_device.size(), size_t(device_count_));
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      const auto fragments =
          shard_count
              ? only_shards_for_device(query_info.fragments, device_id, device_count_)
              : query_info.fragments;
      const auto hash_table_key = genHashTableKey(
          fragments, inner_outer_pairs_.front().second, inner_outer_pairs_.front().first);
      init_threads.push_back(std::async(std::launch::async,
                                        &JoinHashTable::reifyForDevice,
                                        this,
                                        hash_table_key,
                                        columns_per_device[device_id],
                                        hash_type_,
                                        device_id,
                                        logger::thread_id()));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }
  }
}

Data_Namespace::MemoryLevel JoinHashTable::getEffectiveMemoryLevel(
    const std::vector<InnerOuter>& inner_outer_pairs) const {
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    if (needs_dictionary_translation(
            inner_outer_pair.first, inner_outer_pair.second, executor_)) {
      return Data_Namespace::CPU_LEVEL;
    }
  }
  return memory_level_;
}

ColumnsForDevice JoinHashTable::fetchColumnsForDevice(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    DeviceAllocator* dev_buff_owner) {
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);

  std::vector<JoinColumn> join_columns;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<JoinColumnTypeInfo> join_column_types;
  std::vector<JoinBucketInfo> join_bucket_info;
  std::vector<std::shared_ptr<void>> malloc_owner;
  for (const auto& inner_outer_pair : inner_outer_pairs_) {
    const auto inner_col = inner_outer_pair.first;
    const auto inner_cd = get_column_descriptor_maybe(
        inner_col->get_column_id(), inner_col->get_table_id(), *catalog_);
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
                                                      inline_fixed_encoding_null_val(ti),
                                                      isBitwiseEq(),
                                                      0,
                                                      get_join_column_type_kind(ti)});
  }
  return {join_columns, join_column_types, chunks_owner, join_bucket_info, malloc_owner};
}

void JoinHashTable::reifyForDevice(const ChunkKey& hash_table_key,
                                   const ColumnsForDevice& columns_for_device,
                                   const JoinHashTableInterface::HashType layout,
                                   const int device_id,
                                   const logger::ThreadId parent_thread_id) {
  DEBUG_TIMER_NEW_THREAD(parent_thread_id);
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs_);

  CHECK_EQ(columns_for_device.join_columns.size(), size_t(1));
  CHECK_EQ(inner_outer_pairs_.size(), size_t(1));
  auto& join_column = columns_for_device.join_columns.front();
  if (layout == JoinHashTableInterface::HashType::OneToOne) {
    // TODO: check ret value
    const auto err = initHashTableForDevice(hash_table_key,
                                            join_column,
                                            inner_outer_pairs_.front(),
                                            JoinHashTableInterface::HashType::OneToOne,
                                            effective_memory_level,
                                            device_id);
    if (err) {
      throw NeedsOneToManyHash();
    }
  } else {
    const auto err = initHashTableForDevice(hash_table_key,
                                            join_column,
                                            inner_outer_pairs_.front(),
                                            JoinHashTableInterface::HashType::OneToMany,
                                            effective_memory_level,
                                            device_id);
    if (err) {
      throw std::runtime_error("Unexpected error building one to many hash table: " +
                               std::to_string(err));
    }
  }
}

int JoinHashTable::initHashTableForDevice(
    const ChunkKey& chunk_key,
    const JoinColumn& join_column,
    const InnerOuter& cols,
    const JoinHashTableInterface::HashType layout,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id) {
  auto timer = DEBUG_TIMER(__func__);
  const auto inner_col = cols.first;
  CHECK(inner_col);

  auto hash_entry_info = get_bucketized_hash_entry_info(
      inner_col->get_type_info(), col_range_, isBitwiseEq());
  if (!hash_entry_info && layout == JoinHashTableInterface::HashType::OneToOne) {
    // TODO: what is this for?
    return 0;
  }
  PerfectJoinHashTableBuilder builder(executor_->catalog_);
#ifndef HAVE_CUDA
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  if (!device_id) {
    hash_entry_count_ = hash_entry_info.getNormalizedHashEntryCount();
  }

  int err{0};
  const int32_t hash_join_invalid_val{-1};
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    CHECK(!chunk_key.empty());

    if (memory_level_ == Data_Namespace::GPU_LEVEL) {
#ifdef HAVE_CUDA
      // TODO: this ends up allocating gpu memory in the same hash table object as cpu
      // memory, which is somewhat wasteful. unify + cleanup w/ baseline code
      builder.allocateDeviceMemory(
          join_column, layout, hash_entry_info, shardCount(), device_id, device_count_);
#else
      UNREACHABLE();
#endif
    }

    auto hash_table = initHashTableOnCpuFromCache(chunk_key, join_column.num_elems, cols);
    {
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      if (!hash_table) {
        if (layout == JoinHashTableInterface::HashType::OneToOne) {
          builder.initOneToOneHashTableOnCpu(join_column,
                                             col_range_,
                                             isBitwiseEq(),
                                             cols,
                                             hash_entry_info,
                                             hash_join_invalid_val,
                                             executor_);
          hash_table = builder.getHashTable();
        } else {
          builder.initOneToManyHashTableOnCpu(join_column,
                                              col_range_,
                                              isBitwiseEq(),
                                              cols,
                                              hash_entry_info,
                                              hash_join_invalid_val,
                                              executor_);
          hash_table = builder.getHashTable();
        }
      } else {
        if (layout == JoinHashTableInterface::HashType::OneToOne &&
            hash_table->size() > hash_entry_info.getNormalizedHashEntryCount()) {
          // Too many hash entries, need to retry with a 1:many table
          throw NeedsOneToManyHash();
        }
      }
    }
    if (inner_col->get_table_id() > 0) {
      putHashTableOnCpuToCache(chunk_key, join_column.num_elems, hash_table, cols);
    }
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    if (memory_level_ == Data_Namespace::GPU_LEVEL) {
#ifdef HAVE_CUDA
      const auto& ti = inner_col->get_type_info();
      CHECK(ti.is_string());
      auto catalog = executor_->getCatalog();
      CHECK(catalog);
      auto& data_mgr = catalog->getDataMgr();
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);

      std::shared_ptr<PerfectHashTable> gpu_hash_table = builder.getHashTable();
      if (!gpu_hash_table) {
        // constructed the hash table above, so the same hash table will contain both CPU
        // and GPU data
        gpu_hash_table = hash_table;
      }
      CHECK(gpu_hash_table);
      auto gpu_buffer_ptr = gpu_hash_table->gpuBufferPtr();
      CHECK(gpu_buffer_ptr);

      CHECK(hash_table);
      copy_to_gpu(&data_mgr,
                  reinterpret_cast<CUdeviceptr>(gpu_buffer_ptr),
                  hash_table->data(),
                  hash_table->size() * hash_table->elementSize(),
                  device_id);
      CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
      hash_tables_for_device_[device_id] = std::move(gpu_hash_table);
#else
      UNREACHABLE();
#endif
    } else {
      CHECK(hash_table);
      CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
      hash_tables_for_device_[device_id] = hash_table;
    }
  } else {
#ifdef HAVE_CUDA
    CHECK_EQ(Data_Namespace::GPU_LEVEL, effective_memory_level);
    builder.allocateDeviceMemory(
        join_column, layout, hash_entry_info, shardCount(), device_id, device_count_);
    builder.initHashTableOnGpu(chunk_key,
                               join_column,
                               col_range_,
                               isBitwiseEq(),
                               cols,
                               layout,
                               hash_entry_info,
                               shardCount(),
                               hash_join_invalid_val,
                               device_id,
                               device_count_,
                               executor_);
    CHECK_LT(size_t(device_id), hash_tables_for_device_.size());
    hash_tables_for_device_[device_id] = builder.getHashTable();
#else
    UNREACHABLE();
#endif
  }

  return err;
}

ChunkKey JoinHashTable::genHashTableKey(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const Analyzer::Expr* outer_col_expr,
    const Analyzer::ColumnVar* inner_col) const {
  ChunkKey hash_table_key{executor_->getCatalog()->getCurrentDB().dbId,
                          inner_col->get_table_id(),
                          inner_col->get_column_id()};
  const auto& ti = inner_col->get_type_info();
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    size_t outer_elem_count = 0;
    const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(outer_col_expr);
    CHECK(outer_col);
    const auto& outer_query_info = getInnerQueryInfo(outer_col).info;
    for (auto& frag : outer_query_info.fragments) {
      outer_elem_count = frag.getNumTuples();
    }
    hash_table_key.push_back(outer_elem_count);
  }
  if (fragments.size() < 2) {
    hash_table_key.push_back(fragments.front().fragmentId);
  }
  return hash_table_key;
}

void JoinHashTable::checkHashJoinReplicationConstraint(const int table_id) const {
  if (!g_cluster) {
    return;
  }
  if (table_id >= 0) {
    const auto inner_td = executor_->getCatalog()->getMetadataForTable(table_id);
    CHECK(inner_td);
    size_t shard_count{0};
    shard_count = get_shard_count(qual_bin_oper_.get(), executor_);
    if (!shard_count && !table_is_replicated(inner_td)) {
      throw TableMustBeReplicated(inner_td->tableName);
    }
  }
}

std::shared_ptr<PerfectHashTable> JoinHashTable::initHashTableOnCpuFromCache(
    const ChunkKey& chunk_key,
    const size_t num_elements,
    const InnerOuter& cols) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_GE(chunk_key.size(), size_t(2));
  if (chunk_key[1] < 0) {
    // Do not cache hash tables over intermediate results
    return nullptr;
  }
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(cols.second);
  JoinHashTableCacheKey cache_key{col_range_,
                                  *cols.first,
                                  outer_col ? *outer_col : *cols.first,
                                  num_elements,
                                  chunk_key,
                                  qual_bin_oper_->get_optype()};
  std::lock_guard<std::mutex> join_hash_table_cache_lock(join_hash_table_cache_mutex_);
  for (const auto& kv : join_hash_table_cache_) {
    if (kv.first == cache_key) {
      return kv.second;
    }
  }
  return nullptr;
}

void JoinHashTable::putHashTableOnCpuToCache(const ChunkKey& chunk_key,
                                             const size_t num_elements,
                                             HashTableCacheValue hash_table,
                                             const InnerOuter& cols) {
  CHECK_GE(chunk_key.size(), size_t(2));
  if (chunk_key[1] < 0) {
    // Do not cache hash tables over intermediate results
    return;
  }
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(cols.second);
  JoinHashTableCacheKey cache_key{col_range_,
                                  *cols.first,
                                  outer_col ? *outer_col : *cols.first,
                                  num_elements,
                                  chunk_key,
                                  qual_bin_oper_->get_optype()};
  std::lock_guard<std::mutex> join_hash_table_cache_lock(join_hash_table_cache_mutex_);
  for (const auto& kv : join_hash_table_cache_) {
    if (kv.first == cache_key) {
      return;
    }
  }
  CHECK(hash_table);
  join_hash_table_cache_.emplace_back(cache_key, hash_table);
}

llvm::Value* JoinHashTable::codegenHashTableLoad(const size_t table_idx) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto hash_ptr = codegenHashTableLoad(table_idx, executor_);
  if (hash_ptr->getType()->isIntegerTy(64)) {
    return hash_ptr;
  }
  CHECK(hash_ptr->getType()->isPointerTy());
  return executor_->cgen_state_->ir_builder_.CreatePtrToInt(
      get_arg_by_name(executor_->cgen_state_->row_func_, "join_hash_tables"),
      llvm::Type::getInt64Ty(executor_->cgen_state_->context_));
}

llvm::Value* JoinHashTable::codegenHashTableLoad(const size_t table_idx,
                                                 Executor* executor) {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  llvm::Value* hash_ptr = nullptr;
  const auto total_table_count =
      executor->plan_state_->join_info_.join_hash_tables_.size();
  CHECK_LT(table_idx, total_table_count);
  if (total_table_count > 1) {
    auto hash_tables_ptr =
        get_arg_by_name(executor->cgen_state_->row_func_, "join_hash_tables");
    auto hash_pptr =
        table_idx > 0 ? executor->cgen_state_->ir_builder_.CreateGEP(
                            hash_tables_ptr,
                            executor->cgen_state_->llInt(static_cast<int64_t>(table_idx)))
                      : hash_tables_ptr;
    hash_ptr = executor->cgen_state_->ir_builder_.CreateLoad(hash_pptr);
  } else {
    hash_ptr = get_arg_by_name(executor->cgen_state_->row_func_, "join_hash_tables");
  }
  CHECK(hash_ptr);
  return hash_ptr;
}

std::vector<llvm::Value*> JoinHashTable::getHashJoinArgs(llvm::Value* hash_ptr,
                                                         const Analyzer::Expr* key_col,
                                                         const int shard_count,
                                                         const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  CodeGenerator code_generator(executor_);
  const auto key_lvs = code_generator.codegen(key_col, true, co);
  CHECK_EQ(size_t(1), key_lvs.size());
  auto const& key_col_ti = key_col->get_type_info();
  auto hash_entry_info =
      get_bucketized_hash_entry_info(key_col_ti, col_range_, isBitwiseEq());

  std::vector<llvm::Value*> hash_join_idx_args{
      hash_ptr,
      executor_->cgen_state_->castToTypeIn(key_lvs.front(), 64),
      executor_->cgen_state_->llInt(col_range_.getIntMin()),
      executor_->cgen_state_->llInt(col_range_.getIntMax())};
  if (shard_count) {
    const auto expected_hash_entry_count =
        get_hash_entry_count(col_range_, isBitwiseEq());
    const auto entry_count_per_shard =
        (expected_hash_entry_count + shard_count - 1) / shard_count;
    hash_join_idx_args.push_back(
        executor_->cgen_state_->llInt<uint32_t>(entry_count_per_shard));
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt<uint32_t>(shard_count));
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt<uint32_t>(device_count_));
  }
  auto key_col_logical_ti = get_logical_type_info(key_col->get_type_info());
  if (!key_col_logical_ti.get_notnull() || isBitwiseEq()) {
    hash_join_idx_args.push_back(executor_->cgen_state_->llInt(
        inline_fixed_encoding_null_val(key_col_logical_ti)));
  }
  auto special_date_bucketization_case = key_col_ti.get_type() == kDATE;
  if (isBitwiseEq()) {
    if (special_date_bucketization_case) {
      hash_join_idx_args.push_back(executor_->cgen_state_->llInt(
          col_range_.getIntMax() / hash_entry_info.bucket_normalization + 1));
    } else {
      hash_join_idx_args.push_back(
          executor_->cgen_state_->llInt(col_range_.getIntMax() + 1));
    }
  }

  if (special_date_bucketization_case) {
    hash_join_idx_args.emplace_back(
        executor_->cgen_state_->llInt(hash_entry_info.bucket_normalization));
  }

  return hash_join_idx_args;
}

HashJoinMatchingSet JoinHashTable::codegenMatchingSet(const CompilationOptions& co,
                                                      const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  const auto cols = get_cols(
      qual_bin_oper_.get(), *executor_->getCatalog(), executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  auto pos_ptr = codegenHashTableLoad(index);
  CHECK(pos_ptr);
  const int shard_count = shardCount();
  const auto key_col_var = dynamic_cast<const Analyzer::ColumnVar*>(key_col);
  const auto val_col_var = dynamic_cast<const Analyzer::ColumnVar*>(val_col);
  if (key_col_var && val_col_var &&
      self_join_not_covered_by_left_deep_tree(
          key_col_var,
          val_col_var,
          get_max_rte_scan_table(executor_->cgen_state_->scan_idx_to_hash_pos_))) {
    throw std::runtime_error(
        "Query execution fails because the query contains not supported self-join "
        "pattern. We suspect the query requires multiple left-deep join tree due to "
        "the "
        "join condition of the self-join and is not supported for now. Please consider "
        "rewriting table order in "
        "FROM clause.");
  }
  auto hash_join_idx_args = getHashJoinArgs(pos_ptr, key_col, shard_count, co);
  const int64_t sub_buff_size = getComponentBufferSize();
  const auto& key_col_ti = key_col->get_type_info();

  auto bucketize = (key_col_ti.get_type() == kDATE);
  return codegenMatchingSet(hash_join_idx_args,
                            shard_count,
                            !key_col_ti.get_notnull(),
                            isBitwiseEq(),
                            sub_buff_size,
                            executor_,
                            bucketize);
}

HashJoinMatchingSet JoinHashTable::codegenMatchingSet(
    const std::vector<llvm::Value*>& hash_join_idx_args_in,
    const bool is_sharded,
    const bool col_is_nullable,
    const bool is_bw_eq,
    const int64_t sub_buff_size,
    Executor* executor,
    bool is_bucketized) {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
  using namespace std::string_literals;

  std::string fname(is_bucketized ? "bucketized_hash_join_idx"s : "hash_join_idx"s);

  if (is_bw_eq) {
    fname += "_bitwise";
  }
  if (is_sharded) {
    fname += "_sharded";
  }
  if (!is_bw_eq && col_is_nullable) {
    fname += "_nullable";
  }

  const auto slot_lv = executor->cgen_state_->emitCall(fname, hash_join_idx_args_in);
  const auto slot_valid_lv = executor->cgen_state_->ir_builder_.CreateICmpSGE(
      slot_lv, executor->cgen_state_->llInt(int64_t(0)));

  auto pos_ptr = hash_join_idx_args_in[0];
  CHECK(pos_ptr);

  auto count_ptr = executor->cgen_state_->ir_builder_.CreateAdd(
      pos_ptr, executor->cgen_state_->llInt(sub_buff_size));
  auto hash_join_idx_args = hash_join_idx_args_in;
  hash_join_idx_args[0] = executor->cgen_state_->ir_builder_.CreatePtrToInt(
      count_ptr, llvm::Type::getInt64Ty(executor->cgen_state_->context_));

  const auto row_count_lv = executor->cgen_state_->ir_builder_.CreateSelect(
      slot_valid_lv,
      executor->cgen_state_->emitCall(fname, hash_join_idx_args),
      executor->cgen_state_->llInt(int64_t(0)));
  auto rowid_base_i32 = executor->cgen_state_->ir_builder_.CreateIntToPtr(
      executor->cgen_state_->ir_builder_.CreateAdd(
          pos_ptr, executor->cgen_state_->llInt(2 * sub_buff_size)),
      llvm::Type::getInt32PtrTy(executor->cgen_state_->context_));
  auto rowid_ptr_i32 =
      executor->cgen_state_->ir_builder_.CreateGEP(rowid_base_i32, slot_lv);
  return {rowid_ptr_i32, row_count_lv, slot_lv};
}

size_t JoinHashTable::offsetBufferOff() const noexcept {
  return 0;
}

size_t JoinHashTable::countBufferOff() const noexcept {
  return getComponentBufferSize();
}

size_t JoinHashTable::payloadBufferOff() const noexcept {
  return 2 * getComponentBufferSize();
}

size_t JoinHashTable::getComponentBufferSize() const noexcept {
  if (hash_type_ == JoinHashTableInterface::HashType::OneToMany) {
    return hash_entry_count_ * sizeof(int32_t);
  } else {
    return 0;
  }
}

int64_t JoinHashTable::getJoinHashBuffer(const ExecutorDeviceType device_type,
                                         const int device_id) const noexcept {
  CHECK(!hash_tables_for_device_.empty());
  if (device_type == ExecutorDeviceType::CPU && !hash_tables_for_device_.front()) {
    return 0;
  }
#ifdef HAVE_CUDA
  CHECK_LT(static_cast<size_t>(device_id), hash_tables_for_device_.size());
  if (device_type == ExecutorDeviceType::CPU) {
    CHECK(hash_tables_for_device_.front());
    return reinterpret_cast<int64_t>(hash_tables_for_device_.front()->data());
  } else {
    return hash_tables_for_device_[device_id]
               ? reinterpret_cast<CUdeviceptr>(
                     hash_tables_for_device_[device_id]->gpuBufferPtr())
               : reinterpret_cast<CUdeviceptr>(nullptr);
  }
#else
  CHECK(device_type == ExecutorDeviceType::CPU);
  CHECK(hash_tables_for_device_.front());
  return reinterpret_cast<int64_t>(hash_tables_for_device_.front()->data());
#endif
}

size_t JoinHashTable::getJoinHashBufferSize(const ExecutorDeviceType device_type,
                                            const int device_id) const noexcept {
  CHECK(!hash_tables_for_device_.empty());
  if (device_type == ExecutorDeviceType::CPU && !hash_tables_for_device_.front()) {
    return 0;
  }
#ifdef HAVE_CUDA
  CHECK_LT(static_cast<size_t>(device_id), hash_tables_for_device_.size());
  if (device_type == ExecutorDeviceType::CPU) {
    CHECK(hash_tables_for_device_.front());
    return hash_tables_for_device_.front()->size() *
           hash_tables_for_device_.front()->elementSize();
  } else {
    return hash_tables_for_device_[device_id]
               ? hash_tables_for_device_[device_id]->gpuReservedSize()
               : 0;
  }
#else
  CHECK(device_type == ExecutorDeviceType::CPU);
  CHECK(hash_tables_for_device_.front());
  return hash_tables_for_device_.front()->size() *
         hash_tables_for_device_.front()->elementSize();
#endif
}

std::string JoinHashTable::toString(const ExecutorDeviceType device_type,
                                    const int device_id,
                                    bool raw) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto buffer_size = getJoinHashBufferSize(device_type, device_id);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);

    copy_from_gpu(&executor_->getCatalog()->getDataMgr(),
                  buffer_copy.get(),
                  reinterpret_cast<CUdeviceptr>(reinterpret_cast<int8_t*>(buffer)),
                  buffer_size,
                  device_id);
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : reinterpret_cast<const int8_t*>(buffer);
#else
  auto ptr1 = reinterpret_cast<const int8_t*>(buffer);
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  return JoinHashTableInterface::toString("perfect",
                                          getHashTypeString(hash_type_),
                                          0,
                                          0,
                                          hash_entry_count_,
                                          ptr1,
                                          ptr2,
                                          ptr3,
                                          ptr4,
                                          buffer_size,
                                          raw);
}

std::set<DecodedJoinHashBufferEntry> JoinHashTable::toSet(
    const ExecutorDeviceType device_type,
    const int device_id) const {
  auto buffer = getJoinHashBuffer(device_type, device_id);
  auto buffer_size = getJoinHashBufferSize(device_type, device_id);
#ifdef HAVE_CUDA
  std::unique_ptr<int8_t[]> buffer_copy;
  if (device_type == ExecutorDeviceType::GPU) {
    buffer_copy = std::make_unique<int8_t[]>(buffer_size);

    copy_from_gpu(&executor_->getCatalog()->getDataMgr(),
                  buffer_copy.get(),
                  reinterpret_cast<CUdeviceptr>(reinterpret_cast<int8_t*>(buffer)),
                  buffer_size,
                  device_id);
  }
  auto ptr1 = buffer_copy ? buffer_copy.get() : reinterpret_cast<const int8_t*>(buffer);
#else
  auto ptr1 = reinterpret_cast<const int8_t*>(buffer);
#endif  // HAVE_CUDA
  auto ptr2 = ptr1 + offsetBufferOff();
  auto ptr3 = ptr1 + countBufferOff();
  auto ptr4 = ptr1 + payloadBufferOff();
  return JoinHashTableInterface::toSet(
      0, 0, hash_entry_count_, ptr1, ptr2, ptr3, ptr4, buffer_size);
}

llvm::Value* JoinHashTable::codegenSlot(const CompilationOptions& co,
                                        const size_t index) {
  AUTOMATIC_IR_METADATA(executor_->cgen_state_.get());
  using namespace std::string_literals;

  CHECK(getHashType() == JoinHashTableInterface::HashType::OneToOne);
  const auto cols = get_cols(
      qual_bin_oper_.get(), *executor_->getCatalog(), executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  CodeGenerator code_generator(executor_);
  const auto key_col_var = dynamic_cast<const Analyzer::ColumnVar*>(key_col);
  const auto val_col_var = dynamic_cast<const Analyzer::ColumnVar*>(val_col);
  if (key_col_var && val_col_var &&
      self_join_not_covered_by_left_deep_tree(
          key_col_var,
          val_col_var,
          get_max_rte_scan_table(executor_->cgen_state_->scan_idx_to_hash_pos_))) {
    throw std::runtime_error(
        "Query execution fails because the query contains not supported self-join "
        "pattern. We suspect the query requires multiple left-deep join tree due to "
        "the "
        "join condition of the self-join and is not supported for now. Please consider "
        "rewriting table order in "
        "FROM clause.");
  }
  const auto key_lvs = code_generator.codegen(key_col, true, co);
  CHECK_EQ(size_t(1), key_lvs.size());
  auto hash_ptr = codegenHashTableLoad(index);
  CHECK(hash_ptr);
  const int shard_count = shardCount();
  const auto hash_join_idx_args = getHashJoinArgs(hash_ptr, key_col, shard_count, co);

  const auto& key_col_ti = key_col->get_type_info();
  std::string fname((key_col_ti.get_type() == kDATE) ? "bucketized_hash_join_idx"s
                                                     : "hash_join_idx"s);

  if (isBitwiseEq()) {
    fname += "_bitwise";
  }
  if (shard_count) {
    fname += "_sharded";
  }

  if (!isBitwiseEq() && !key_col_ti.get_notnull()) {
    fname += "_nullable";
  }
  return executor_->cgen_state_->emitCall(fname, hash_join_idx_args);
}

const InputTableInfo& JoinHashTable::getInnerQueryInfo(
    const Analyzer::ColumnVar* inner_col) const {
  return get_inner_query_info(inner_col->get_table_id(), query_infos_);
}

const InputTableInfo& get_inner_query_info(
    const int inner_table_id,
    const std::vector<InputTableInfo>& query_infos) {
  std::optional<size_t> ti_idx;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (inner_table_id == query_infos[i].table_id) {
      ti_idx = i;
      break;
    }
  }
  CHECK(ti_idx);
  return query_infos[*ti_idx];
}

size_t get_entries_per_device(const size_t total_entries,
                              const size_t shard_count,
                              const size_t device_count,
                              const Data_Namespace::MemoryLevel memory_level) {
  const auto entries_per_shard =
      shard_count ? (total_entries + shard_count - 1) / shard_count : total_entries;
  size_t entries_per_device = entries_per_shard;
  if (memory_level == Data_Namespace::GPU_LEVEL && shard_count) {
    const auto shards_per_device = (shard_count + device_count - 1) / device_count;
    CHECK_GT(shards_per_device, 0u);
    entries_per_device = entries_per_shard * shards_per_device;
  }
  return entries_per_device;
}

// TODO(adb): unify with BaselineJoinHashTable
size_t JoinHashTable::shardCount() const {
  return memory_level_ == Data_Namespace::GPU_LEVEL
             ? get_shard_count(qual_bin_oper_.get(), executor_)
             : 0;
}

bool JoinHashTable::isBitwiseEq() const {
  return qual_bin_oper_->get_optype() == kBW_EQ;
}

void JoinHashTable::freeHashBufferMemory() {
  CHECK_GT(device_count_, 0);
  auto empty_hash_tables = decltype(hash_tables_for_device_)(device_count_);
  hash_tables_for_device_.swap(empty_hash_tables);
}
