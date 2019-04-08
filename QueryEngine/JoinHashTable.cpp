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

#include "JoinHashTable.h"
#include "Execute.h"
#include "ExpressionRewrite.h"
#include "HashJoinRuntime.h"
#include "RangeTableIndexVisitor.h"
#include "RuntimeFunctions.h"

#include <glog/logging.h>
#include <future>
#include <numeric>
#include <thread>

namespace {

class NeedsOneToManyHash : public HashJoinFail {
 public:
  NeedsOneToManyHash() : HashJoinFail("Needs one to many hash") {}
};

}  // namespace

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
    if (!lhs_ti.is_integer() && !lhs_ti.is_time() && !lhs_ti.is_string()) {
      throw HashJoinFail("Cannot apply hash join to inner column type " +
                         lhs_ti.get_type_name());
    }
  }

  const auto lhs_cast = dynamic_cast<const Analyzer::UOper*>(lhs);
  const auto rhs_cast = dynamic_cast<const Analyzer::UOper*>(rhs);
  if (lhs_ti.is_string() && (static_cast<bool>(lhs_cast) != static_cast<bool>(rhs_cast) ||
                             (lhs_cast && lhs_cast->get_optype() != kCAST) ||
                             (rhs_cast && rhs_cast->get_optype() != kCAST))) {
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
  if (is_overlaps_join) {
    if (!inner_col_real_ti.is_array()) {
      throw HashJoinFail(
          "Overlaps join only supported for inner columns with array type");
    }
    if (!(inner_col_real_ti.is_fixlen_array() && inner_col_real_ti.get_size() == 32)) {
      throw HashJoinFail(
          "Overlaps join only supported for 4-element double fixed length arrays");
    }
    if (!(outer_col_ti.get_type() == kPOINT)) {
      throw HashJoinFail(
          "Overlaps join only supported for geometry outer columns of type point");
    }
  } else {
    if (!(inner_col_real_ti.is_integer() || inner_col_real_ti.is_time() ||
          (inner_col_real_ti.is_string() &&
           inner_col_real_ti.get_compression() == kENCODING_DICT))) {
      throw HashJoinFail(
          "Can only apply hash join to integer-like types and dictionary encoded "
          "strings");
    }
  }
  return {inner_col, outer_col ? outer_col : outer_expr};
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

std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*> get_cols(
    const Analyzer::BinOper* qual_bin_oper,
    const Catalog_Namespace::Catalog& cat,
    const TemporaryTables* temporary_tables) {
  const auto lhs = qual_bin_oper->get_left_operand();
  const auto rhs = qual_bin_oper->get_right_operand();
  return normalize_column_pair(lhs, rhs, cat, temporary_tables);
}

// Number of entries required for the given range.
size_t get_hash_entry_count(const ExpressionRange& col_range, const bool is_bw_eq) {
  if (col_range.getIntMin() > col_range.getIntMax()) {
    CHECK_EQ(col_range.getIntMin(), int64_t(0));
    CHECK_EQ(col_range.getIntMax(), int64_t(-1));
    return is_bw_eq ? 1 : 0;
  }
  return col_range.getIntMax() - col_range.getIntMin() + 1 + (is_bw_eq ? 1 : 0);
}

}  // namespace

std::vector<std::pair<JoinHashTable::JoinHashTableCacheKey,
                      std::shared_ptr<std::vector<int32_t>>>>
    JoinHashTable::join_hash_table_cache_;
std::mutex JoinHashTable::join_hash_table_cache_mutex_;

size_t get_shard_count(const Analyzer::BinOper* join_condition,
                       const RelAlgExecutionUnit& ra_exe_unit,
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
  return get_shard_count({inner_col, outer_col}, ra_exe_unit, executor);
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
    const RelAlgExecutionUnit& ra_exe_unit,
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

std::shared_ptr<JoinHashTable> JoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
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
  if (get_hash_entry_count(col_range, qual_bin_oper->get_optype() == kBW_EQ) >
      max_hash_entry_count) {
    throw TooManyHashEntries();
  }
  if (qual_bin_oper->get_optype() == kBW_EQ &&
      col_range.getIntMax() >= std::numeric_limits<int64_t>::max()) {
    throw HashJoinFail("Cannot translate null value for kBW_EQ");
  }
  auto join_hash_table = std::shared_ptr<JoinHashTable>(new JoinHashTable(qual_bin_oper,
                                                                          inner_col,
                                                                          query_infos,
                                                                          ra_exe_unit,
                                                                          memory_level,
                                                                          col_range,
                                                                          column_cache,
                                                                          executor,
                                                                          device_count));
  try {
    join_hash_table->reify(device_count);
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
  } catch (const std::exception& e) {
    LOG(FATAL) << "Fatal error while attempting to build hash tables for join: "
               << e.what();
  }
  return join_hash_table;
}

std::pair<const int8_t*, size_t> JoinHashTable::getColumnFragment(
    const Analyzer::ColumnVar& hash_col,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const Data_Namespace::MemoryLevel effective_mem_lvl,
    const int device_id,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner) {
  return Executor::ExecutionDispatch::getColumnFragment(executor_,
                                                        hash_col,
                                                        fragment,
                                                        effective_mem_lvl,
                                                        device_id,
                                                        chunks_owner,
                                                        column_cache_);
}

std::pair<const int8_t*, size_t> JoinHashTable::getAllColumnFragments(
    const Analyzer::ColumnVar& hash_col,
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner) {
  std::lock_guard<std::mutex> linearized_multifrag_column_lock(
      linearized_multifrag_column_mutex_);
  if (linearized_multifrag_column_.first) {
    return linearized_multifrag_column_;
  }
  const int8_t* col_buff;
  size_t total_elem_count;
  std::tie(col_buff, total_elem_count) =
      Executor::ExecutionDispatch::getAllColumnFragments(
          executor_, hash_col, fragments, chunks_owner, column_cache_);
  linearized_multifrag_column_owner_.addColBuffer(col_buff);
  if (!shardCount()) {
    linearized_multifrag_column_ = {col_buff, total_elem_count};
  }
  return {col_buff, total_elem_count};
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

std::deque<Fragmenter_Namespace::FragmentInfo> only_shards_for_device(
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id,
    const int device_count) {
  std::deque<Fragmenter_Namespace::FragmentInfo> shards_for_device;
  for (const auto& fragment : fragments) {
    CHECK_GE(fragment.shard, 0);
    if (fragment.shard % device_count == device_id) {
      shards_for_device.push_back(fragment);
    }
  }
  return shards_for_device;
}

void JoinHashTable::reify(const int device_count) {
  CHECK_LT(0, device_count);
  const auto& catalog = *executor_->getCatalog();
  const auto cols = get_cols(qual_bin_oper_.get(), catalog, executor_->temporary_tables_);
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
#ifdef HAVE_CUDA
  gpu_hash_table_buff_.resize(device_count);
  gpu_hash_table_err_buff_.resize(device_count);
#endif  // HAVE_CUDA
  std::vector<std::future<void>> init_threads;
  const int shard_count = shardCount();

  try {
    for (int device_id = 0; device_id < device_count; ++device_id) {
      const auto fragments =
          shard_count
              ? only_shards_for_device(query_info.fragments, device_id, device_count)
              : query_info.fragments;
      init_threads.push_back(std::async(std::launch::async,
                                        &JoinHashTable::reifyOneToOneForDevice,
                                        this,
                                        fragments,
                                        device_id));
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
    for (int device_id = 0; device_id < device_count; ++device_id) {
      const auto fragments =
          shard_count
              ? only_shards_for_device(query_info.fragments, device_id, device_count)
              : query_info.fragments;

      init_threads.push_back(std::async(std::launch::async,
                                        &JoinHashTable::reifyOneToManyForDevice,
                                        this,
                                        fragments,
                                        device_id));
    }
    for (auto& init_thread : init_threads) {
      init_thread.wait();
    }
    for (auto& init_thread : init_threads) {
      init_thread.get();
    }
  }
}

std::pair<const int8_t*, size_t> JoinHashTable::fetchFragments(
    const Analyzer::ColumnVar* hash_col,
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragment_info,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    ThrustAllocator& dev_buff_owner) {
  static std::mutex fragment_fetch_mutex;
  const bool has_multi_frag = fragment_info.size() > 1;
  const auto& catalog = *executor_->getCatalog();
  auto& data_mgr = catalog.getDataMgr();
  const auto& first_frag = fragment_info.front();
  const int8_t* col_buff = nullptr;
  size_t elem_count = 0;

  const size_t elem_width = hash_col->get_type_info().get_size();
  if (has_multi_frag) {
    std::tie(col_buff, elem_count) =
        getAllColumnFragments(*hash_col, fragment_info, chunks_owner);
  }

  {
    std::lock_guard<std::mutex> fragment_fetch_lock(fragment_fetch_mutex);
    if (has_multi_frag) {
      if (effective_memory_level == Data_Namespace::GPU_LEVEL) {
        CHECK(col_buff != nullptr);
        CHECK_NE(elem_count, size_t(0));
        int8_t* dev_col_buff = nullptr;
        dev_col_buff = dev_buff_owner.allocate(elem_count * elem_width);
        copy_to_gpu(&data_mgr,
                    reinterpret_cast<CUdeviceptr>(dev_col_buff),
                    col_buff,
                    elem_count * elem_width,
                    device_id);
        col_buff = dev_col_buff;
      }
    } else {
      std::tie(col_buff, elem_count) = getColumnFragment(
          *hash_col, first_frag, effective_memory_level, device_id, chunks_owner);
    }
  }
  return {col_buff, elem_count};
}

ChunkKey JoinHashTable::genHashTableKey(
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
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

void JoinHashTable::reifyOneToOneForDevice(
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id) {
  const auto& catalog = *executor_->getCatalog();
  auto& data_mgr = catalog.getDataMgr();
  const auto cols = get_cols(qual_bin_oper_.get(), catalog, executor_->temporary_tables_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto inner_cd = get_column_descriptor_maybe(
      inner_col->get_column_id(), inner_col->get_table_id(), catalog);
  if (inner_cd && inner_cd->isVirtualCol) {
    throw FailedToJoinOnVirtualColumn();
  }
  CHECK(!inner_cd || !(inner_cd->isVirtualCol));
  // Since we don't have the string dictionary payloads on the GPU, we'll build
  // the join hash table on the CPU and transfer it to the GPU.
  const auto effective_memory_level =
      needs_dictionary_translation(inner_col, cols.second, executor_)
          ? Data_Namespace::CPU_LEVEL
          : memory_level_;
  if (fragments.empty()) {
    // No data in this fragment. Still need to create a hash table and initialize it
    // properly.
    ChunkKey empty_chunk;
    initHashTableForDevice(
        empty_chunk, nullptr, 0, cols, effective_memory_level, device_id);
  }

  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  ThrustAllocator dev_buff_owner(&data_mgr, device_id);
  const int8_t* col_buff = nullptr;
  size_t elem_count = 0;

  std::tie(col_buff, elem_count) = fetchFragments(inner_col,
                                                  fragments,
                                                  effective_memory_level,
                                                  device_id,
                                                  chunks_owner,
                                                  dev_buff_owner);

  initHashTableForDevice(genHashTableKey(fragments, cols.second, inner_col),
                         col_buff,
                         elem_count,
                         cols,
                         effective_memory_level,
                         device_id);
}

void JoinHashTable::reifyOneToManyForDevice(
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id) {
  const auto& catalog = *executor_->getCatalog();
  auto& data_mgr = catalog.getDataMgr();
  const auto cols = get_cols(qual_bin_oper_.get(), catalog, executor_->temporary_tables_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto inner_cd = get_column_descriptor_maybe(
      inner_col->get_column_id(), inner_col->get_table_id(), catalog);
  if (inner_cd && inner_cd->isVirtualCol) {
    throw FailedToJoinOnVirtualColumn();
  }
  CHECK(!inner_cd || !(inner_cd->isVirtualCol));
  // Since we don't have the string dictionary payloads on the GPU, we'll build
  // the join hash table on the CPU and transfer it to the GPU.
  const auto effective_memory_level =
      needs_dictionary_translation(inner_col, cols.second, executor_)
          ? Data_Namespace::CPU_LEVEL
          : memory_level_;
  if (fragments.empty()) {
    ChunkKey empty_chunk;
    initOneToManyHashTable(
        empty_chunk, nullptr, 0, cols, effective_memory_level, device_id);
    return;
  }

  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  ThrustAllocator dev_buff_owner(&data_mgr, device_id);
  const int8_t* col_buff = nullptr;
  size_t elem_count = 0;

  std::tie(col_buff, elem_count) = fetchFragments(inner_col,
                                                  fragments,
                                                  effective_memory_level,
                                                  device_id,
                                                  chunks_owner,
                                                  dev_buff_owner);

  initOneToManyHashTable(genHashTableKey(fragments, cols.second, inner_col),
                         col_buff,
                         elem_count,
                         cols,
                         effective_memory_level,
                         device_id);
}

void JoinHashTable::checkHashJoinReplicationConstraint(const int table_id) const {
  if (!g_cluster) {
    return;
  }
  if (table_id >= 0) {
    const auto inner_td = executor_->getCatalog()->getMetadataForTable(table_id);
    CHECK(inner_td);
    size_t shard_count{0};
    shard_count = get_shard_count(qual_bin_oper_.get(), ra_exe_unit_, executor_);
    if (!shard_count && !table_is_replicated(inner_td)) {
      throw TableMustBeReplicated(inner_td->tableName);
    }
  }
}

void JoinHashTable::initHashTableOnCpu(
    const int8_t* col_buff,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols,
    const size_t hash_entry_count,
    const int32_t hash_join_invalid_val) {
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& ti = inner_col->get_type_info();
  if (!cpu_hash_table_buff_) {
    cpu_hash_table_buff_ = std::make_shared<std::vector<int32_t>>(hash_entry_count);
    const StringDictionaryProxy* sd_inner_proxy{nullptr};
    const StringDictionaryProxy* sd_outer_proxy{nullptr};
    if (ti.is_string()) {
      CHECK_EQ(kENCODING_DICT, ti.get_compression());
      sd_inner_proxy = executor_->getStringDictionaryProxy(
          inner_col->get_comp_param(), executor_->row_set_mem_owner_, true);
      CHECK(sd_inner_proxy);
      const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(cols.second);
      CHECK(outer_col);
      sd_outer_proxy = executor_->getStringDictionaryProxy(
          outer_col->get_comp_param(), executor_->row_set_mem_owner_, true);
      CHECK(sd_outer_proxy);
    }
    int thread_count = cpu_threads();
    std::vector<std::thread> init_cpu_buff_threads;
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      init_cpu_buff_threads.emplace_back(
          [this, hash_entry_count, hash_join_invalid_val, thread_idx, thread_count] {
            init_hash_join_buff(&(*cpu_hash_table_buff_)[0],
                                hash_entry_count,
                                hash_join_invalid_val,
                                thread_idx,
                                thread_count);
          });
    }
    for (auto& t : init_cpu_buff_threads) {
      t.join();
    }
    init_cpu_buff_threads.clear();
    int err{0};
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      init_cpu_buff_threads.emplace_back([this,
                                          hash_join_invalid_val,
                                          col_buff,
                                          num_elements,
                                          sd_inner_proxy,
                                          sd_outer_proxy,
                                          thread_idx,
                                          thread_count,
                                          &ti,
                                          &err] {
        int partial_err = fill_hash_join_buff(&(*cpu_hash_table_buff_)[0],
                                              hash_join_invalid_val,
                                              {col_buff, num_elements},
                                              {static_cast<size_t>(ti.get_size()),
                                               col_range_.getIntMin(),
                                               inline_fixed_encoding_null_val(ti),
                                               isBitwiseEq(),
                                               col_range_.getIntMax() + 1,
                                               get_join_column_type_kind(ti)},
                                              sd_inner_proxy,
                                              sd_outer_proxy,
                                              thread_idx,
                                              thread_count);
        __sync_val_compare_and_swap(&err, 0, partial_err);
      });
    }
    for (auto& t : init_cpu_buff_threads) {
      t.join();
    }
    if (err) {
      cpu_hash_table_buff_.reset();
      // Too many hash entries, need to retry with a 1:many table
      throw NeedsOneToManyHash();
    }
  } else {
    if (cpu_hash_table_buff_->size() > hash_entry_count) {
      // Too many hash entries, need to retry with a 1:many table
      throw NeedsOneToManyHash();
    }
  }
}

void JoinHashTable::initOneToManyHashTableOnCpu(
    const int8_t* col_buff,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols,
    const size_t hash_entry_count,
    const int32_t hash_join_invalid_val) {
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& ti = inner_col->get_type_info();
  if (cpu_hash_table_buff_) {
    return;
  }
  cpu_hash_table_buff_ =
      std::make_shared<std::vector<int32_t>>(2 * hash_entry_count + num_elements);
  const StringDictionaryProxy* sd_inner_proxy{nullptr};
  const StringDictionaryProxy* sd_outer_proxy{nullptr};
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    sd_inner_proxy = executor_->getStringDictionaryProxy(
        inner_col->get_comp_param(), executor_->row_set_mem_owner_, true);
    CHECK(sd_inner_proxy);
    const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(cols.second);
    CHECK(outer_col);
    sd_outer_proxy = executor_->getStringDictionaryProxy(
        outer_col->get_comp_param(), executor_->row_set_mem_owner_, true);
    CHECK(sd_outer_proxy);
  }
  int thread_count = cpu_threads();
  std::vector<std::future<void>> init_threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    init_threads.emplace_back(std::async(std::launch::async,
                                         init_hash_join_buff,
                                         &(*cpu_hash_table_buff_)[0],
                                         hash_entry_count,
                                         hash_join_invalid_val,
                                         thread_idx,
                                         thread_count));
  }
  for (auto& child : init_threads) {
    child.wait();
  }
  for (auto& child : init_threads) {
    child.get();
  }
  fill_one_to_many_hash_table(&(*cpu_hash_table_buff_)[0],
                              hash_entry_count,
                              hash_join_invalid_val,
                              {col_buff, num_elements},
                              {static_cast<size_t>(ti.get_size()),
                               col_range_.getIntMin(),
                               inline_fixed_encoding_null_val(ti),
                               isBitwiseEq(),
                               col_range_.getIntMax() + 1,
                               get_join_column_type_kind(ti)},
                              sd_inner_proxy,
                              sd_outer_proxy,
                              thread_count);
}

namespace {

#ifdef HAVE_CUDA
// Number of entries per shard, rounded up.
size_t get_entries_per_shard(const size_t total_entry_count, const size_t shard_count) {
  CHECK_NE(size_t(0), shard_count);
  return (total_entry_count + shard_count - 1) / shard_count;
}
#endif  // HAVE_CUDA

}  // namespace

void JoinHashTable::initHashTableForDevice(
    const ChunkKey& chunk_key,
    const int8_t* col_buff,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id) {
  auto hash_entry_count = get_hash_entry_count(col_range_, isBitwiseEq());
  if (!hash_entry_count) {
    return;
  }
#ifdef HAVE_CUDA
  const auto shard_count = shardCount();
  const size_t entries_per_shard{
      shard_count ? get_entries_per_shard(hash_entry_count, shard_count) : 0};
  // Even if we join on dictionary encoded strings, the memory on the GPU is still
  // needed once the join hash table has been built on the CPU.
  const auto catalog = executor_->getCatalog();
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    auto& data_mgr = catalog->getDataMgr();
    if (shard_count) {
      const auto shards_per_device = (shard_count + device_count_ - 1) / device_count_;
      CHECK_GT(shards_per_device, 0);
      hash_entry_count = entries_per_shard * shards_per_device;
    }
    gpu_hash_table_buff_[device_id] = alloc_gpu_abstract_buffer(
        &data_mgr, hash_entry_count * sizeof(int32_t), device_id);
  }
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  const auto inner_col = cols.first;
  CHECK(inner_col);
#ifdef HAVE_CUDA
  const auto& ti = inner_col->get_type_info();
#endif
  const int32_t hash_join_invalid_val{-1};
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    CHECK(!chunk_key.empty() && col_buff);
    initHashTableOnCpuFromCache(chunk_key, num_elements, cols);
    {
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      initHashTableOnCpu(
          col_buff, num_elements, cols, hash_entry_count, hash_join_invalid_val);
    }
    if (inner_col->get_table_id() > 0) {
      putHashTableOnCpuToCache(chunk_key, num_elements, cols);
    }
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    if (memory_level_ == Data_Namespace::GPU_LEVEL) {
#ifdef HAVE_CUDA
      CHECK(ti.is_string());
      auto& data_mgr = catalog->getDataMgr();
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);

      copy_to_gpu(
          &data_mgr,
          reinterpret_cast<CUdeviceptr>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
          &(*cpu_hash_table_buff_)[0],
          cpu_hash_table_buff_->size() * sizeof((*cpu_hash_table_buff_)[0]),
          device_id);
#else
      CHECK(false);
#endif
    }
  } else {
#ifdef HAVE_CUDA
    int err{0};
    CHECK_EQ(Data_Namespace::GPU_LEVEL, effective_memory_level);
    auto& data_mgr = catalog->getDataMgr();
    gpu_hash_table_err_buff_[device_id] =
        alloc_gpu_abstract_buffer(&data_mgr, sizeof(int), device_id);
    auto dev_err_buff = reinterpret_cast<CUdeviceptr>(
        gpu_hash_table_err_buff_[device_id]->getMemoryPtr());
    copy_to_gpu(&data_mgr, dev_err_buff, &err, sizeof(err), device_id);
    init_hash_join_buff_on_device(
        reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
        hash_entry_count,
        hash_join_invalid_val,
        executor_->blockSize(),
        executor_->gridSize());
    if (chunk_key.empty()) {
      return;
    }
    JoinColumn join_column{col_buff, num_elements};
    JoinColumnTypeInfo type_info{static_cast<size_t>(ti.get_size()),
                                 col_range_.getIntMin(),
                                 inline_fixed_encoding_null_val(ti),
                                 isBitwiseEq(),
                                 col_range_.getIntMax() + 1,
                                 get_join_column_type_kind(ti)};
    if (shard_count) {
      CHECK_GT(device_count_, 0);
      for (size_t shard = device_id; shard < shard_count; shard += device_count_) {
        ShardInfo shard_info{shard, entries_per_shard, shard_count, device_count_};
        fill_hash_join_buff_on_device_sharded(
            reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
            hash_join_invalid_val,
            reinterpret_cast<int*>(dev_err_buff),
            join_column,
            type_info,
            shard_info,
            executor_->blockSize(),
            executor_->gridSize());
      }
    } else {
      fill_hash_join_buff_on_device(
          reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
          hash_join_invalid_val,
          reinterpret_cast<int*>(dev_err_buff),
          join_column,
          type_info,
          executor_->blockSize(),
          executor_->gridSize());
    }
    copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);

    if (err) {
      throw NeedsOneToManyHash();
    }
#else
    CHECK(false);
#endif
  }
}

void JoinHashTable::initOneToManyHashTable(
    const ChunkKey& chunk_key,
    const int8_t* col_buff,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id) {
  auto hash_entry_count = get_hash_entry_count(col_range_, isBitwiseEq());
#ifdef HAVE_CUDA
  const auto shard_count = get_shard_count(qual_bin_oper_.get(), ra_exe_unit_, executor_);
  const size_t entries_per_shard =
      (shard_count ? get_entries_per_shard(hash_entry_count, shard_count) : 0);
  // Even if we join on dictionary encoded strings, the memory on the GPU is still
  // needed once the join hash table has been built on the CPU.
  if (memory_level_ == Data_Namespace::GPU_LEVEL && shard_count) {
    const auto shards_per_device = (shard_count + device_count_ - 1) / device_count_;
    CHECK_GT(shards_per_device, 0);
    hash_entry_count = entries_per_shard * shards_per_device;
  }
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  if (!device_id) {
    hash_entry_count_ = hash_entry_count;
  }
  const auto inner_col = cols.first;
  CHECK(inner_col);
#ifdef HAVE_CUDA
  const auto& ti = inner_col->get_type_info();
  auto& data_mgr = executor_->getCatalog()->getDataMgr();
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    const size_t total_count = 2 * hash_entry_count + num_elements;
    OOM_TRACE_PUSH(+": total_count " + std::to_string(total_count));
    gpu_hash_table_buff_[device_id] =
        alloc_gpu_abstract_buffer(&data_mgr, total_count * sizeof(int32_t), device_id);
  }
#endif
  const int32_t hash_join_invalid_val{-1};
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    initHashTableOnCpuFromCache(chunk_key, num_elements, cols);
    {
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      initOneToManyHashTableOnCpu(
          col_buff, num_elements, cols, hash_entry_count, hash_join_invalid_val);
    }
    if (inner_col->get_table_id() > 0) {
      putHashTableOnCpuToCache(chunk_key, num_elements, cols);
    }
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    // Don't transfer the buffer if there was an error since we'll bail anyway.
    if (memory_level_ == Data_Namespace::GPU_LEVEL) {
#ifdef HAVE_CUDA
      CHECK(ti.is_string());
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      copy_to_gpu(
          &data_mgr,
          reinterpret_cast<CUdeviceptr>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
          &(*cpu_hash_table_buff_)[0],
          cpu_hash_table_buff_->size() * sizeof((*cpu_hash_table_buff_)[0]),
          device_id);
#else
      CHECK(false);
#endif
    }
  } else {
#ifdef HAVE_CUDA
    CHECK_EQ(Data_Namespace::GPU_LEVEL, effective_memory_level);
    data_mgr.getCudaMgr()->setContext(device_id);
    init_hash_join_buff_on_device(
        reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
        hash_entry_count,
        hash_join_invalid_val,
        executor_->blockSize(),
        executor_->gridSize());
    JoinColumn join_column{col_buff, num_elements};
    JoinColumnTypeInfo type_info{static_cast<size_t>(ti.get_size()),
                                 col_range_.getIntMin(),
                                 inline_fixed_encoding_null_val(ti),
                                 isBitwiseEq(),
                                 col_range_.getIntMax() + 1,
                                 get_join_column_type_kind(ti)};
    if (shard_count) {
      CHECK_GT(device_count_, 0);
      for (size_t shard = device_id; shard < shard_count; shard += device_count_) {
        ShardInfo shard_info{shard, entries_per_shard, shard_count, device_count_};
        fill_one_to_many_hash_table_on_device_sharded(
            reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
            hash_entry_count,
            hash_join_invalid_val,
            join_column,
            type_info,
            shard_info,
            executor_->blockSize(),
            executor_->gridSize());
      }
    } else {
      fill_one_to_many_hash_table_on_device(
          reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
          hash_entry_count,
          hash_join_invalid_val,
          join_column,
          type_info,
          executor_->blockSize(),
          executor_->gridSize());
    }
#else
    CHECK(false);
#endif
  }
}

void JoinHashTable::initHashTableOnCpuFromCache(
    const ChunkKey& chunk_key,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols) {
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
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      cpu_hash_table_buff_ = kv.second;
      break;
    }
  }
}

void JoinHashTable::putHashTableOnCpuToCache(
    const ChunkKey& chunk_key,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::Expr*>& cols) {
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
  join_hash_table_cache_.emplace_back(cache_key, cpu_hash_table_buff_);
}

llvm::Value* JoinHashTable::codegenHashTableLoad(const size_t table_idx) {
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
  llvm::Value* hash_ptr = nullptr;
  const auto total_table_count =
      executor->plan_state_->join_info_.join_hash_tables_.size();
  CHECK_LT(table_idx, total_table_count);
  if (total_table_count > 1) {
    auto hash_tables_ptr =
        get_arg_by_name(executor->cgen_state_->row_func_, "join_hash_tables");
    auto hash_pptr =
        table_idx > 0
            ? executor->cgen_state_->ir_builder_.CreateGEP(
                  hash_tables_ptr, executor->ll_int(static_cast<int64_t>(table_idx)))
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
  const auto key_lvs = executor_->codegen(key_col, true, co);
  CHECK_EQ(size_t(1), key_lvs.size());
  std::vector<llvm::Value*> hash_join_idx_args{
      hash_ptr,
      executor_->castToTypeIn(key_lvs.front(), 64),
      executor_->ll_int(col_range_.getIntMin()),
      executor_->ll_int(col_range_.getIntMax())};
  if (shard_count) {
    const auto expected_hash_entry_count =
        get_hash_entry_count(col_range_, isBitwiseEq());
    const auto entry_count_per_shard =
        (expected_hash_entry_count + shard_count - 1) / shard_count;
    hash_join_idx_args.push_back(executor_->ll_int<uint32_t>(entry_count_per_shard));
    hash_join_idx_args.push_back(executor_->ll_int<uint32_t>(shard_count));
    hash_join_idx_args.push_back(executor_->ll_int<uint32_t>(device_count_));
  }
  auto key_col_logical_ti = get_logical_type_info(key_col->get_type_info());
  if (!key_col_logical_ti.get_notnull() || isBitwiseEq()) {
    hash_join_idx_args.push_back(
        executor_->ll_int(inline_fixed_encoding_null_val(key_col_logical_ti)));
  }
  if (isBitwiseEq()) {
    hash_join_idx_args.push_back(executor_->ll_int(col_range_.getIntMax() + 1));
  }
  return hash_join_idx_args;
}

HashJoinMatchingSet JoinHashTable::codegenMatchingSet(const CompilationOptions& co,
                                                      const size_t index) {
  const auto cols = get_cols(
      qual_bin_oper_.get(), *executor_->getCatalog(), executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  auto pos_ptr = codegenHashTableLoad(index);
  CHECK(pos_ptr);
  const int shard_count = shardCount();
  auto hash_join_idx_args = getHashJoinArgs(pos_ptr, key_col, shard_count, co);
  const int64_t sub_buff_size = hash_entry_count_ * sizeof(int32_t);
  const auto& key_col_ti = key_col->get_type_info();
  return codegenMatchingSet(hash_join_idx_args,
                            shard_count,
                            !key_col_ti.get_notnull(),
                            isBitwiseEq(),
                            sub_buff_size,
                            executor_);
}

HashJoinMatchingSet JoinHashTable::codegenMatchingSet(
    const std::vector<llvm::Value*>& hash_join_idx_args_in,
    const bool is_sharded,
    const bool col_is_nullable,
    const bool is_bw_eq,
    const int64_t sub_buff_size,
    Executor* executor) {
  std::string fname{"hash_join_idx"};
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
      slot_lv, executor->ll_int(int64_t(0)));

  auto pos_ptr = hash_join_idx_args_in[0];
  CHECK(pos_ptr);

  auto count_ptr = executor->cgen_state_->ir_builder_.CreateAdd(
      pos_ptr, executor->ll_int(sub_buff_size));
  auto hash_join_idx_args = hash_join_idx_args_in;
  hash_join_idx_args[0] = executor->cgen_state_->ir_builder_.CreatePtrToInt(
      count_ptr, llvm::Type::getInt64Ty(executor->cgen_state_->context_));

  const auto row_count_lv = executor->cgen_state_->ir_builder_.CreateSelect(
      slot_valid_lv,
      executor->cgen_state_->emitCall(fname, hash_join_idx_args),
      executor->ll_int(int64_t(0)));
  auto rowid_base_i32 = executor->cgen_state_->ir_builder_.CreateIntToPtr(
      executor->cgen_state_->ir_builder_.CreateAdd(pos_ptr,
                                                   executor->ll_int(2 * sub_buff_size)),
      llvm::Type::getInt32PtrTy(executor->cgen_state_->context_));
  auto rowid_ptr_i32 =
      executor->cgen_state_->ir_builder_.CreateGEP(rowid_base_i32, slot_lv);
  return {rowid_ptr_i32, row_count_lv, slot_lv};
}

llvm::Value* JoinHashTable::codegenSlot(const CompilationOptions& co,
                                        const size_t index) {
  CHECK(getHashType() == JoinHashTableInterface::HashType::OneToOne);
  const auto cols = get_cols(
      qual_bin_oper_.get(), *executor_->getCatalog(), executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  const auto key_lvs = executor_->codegen(key_col, true, co);
  CHECK_EQ(size_t(1), key_lvs.size());
  auto hash_ptr = codegenHashTableLoad(index);
  CHECK(hash_ptr);
  const int shard_count = shardCount();
  const auto hash_join_idx_args = getHashJoinArgs(hash_ptr, key_col, shard_count, co);
  std::string fname{"hash_join_idx"};
  if (isBitwiseEq()) {
    fname += "_bitwise";
  }
  if (shard_count) {
    fname += "_sharded";
  }
  const auto& key_col_ti = key_col->get_type_info();
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
  ssize_t ti_idx = -1;
  for (size_t i = 0; i < query_infos.size(); ++i) {
    if (inner_table_id == query_infos[i].table_id) {
      ti_idx = i;
      break;
    }
  }
  CHECK_NE(ssize_t(-1), ti_idx);
  return query_infos[ti_idx];
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
    CHECK_GT(shards_per_device, 0);
    entries_per_device = entries_per_shard * shards_per_device;
  }
  return entries_per_device;
}

// TODO(adb): unify with BaselineJoinHashTable
size_t JoinHashTable::shardCount() const {
  return memory_level_ == Data_Namespace::GPU_LEVEL
             ? get_shard_count(qual_bin_oper_.get(), ra_exe_unit_, executor_)
             : 0;
}

bool JoinHashTable::isBitwiseEq() const {
  return qual_bin_oper_->get_optype() == kBW_EQ;
}

void JoinHashTable::freeHashBufferMemory() {
#ifdef HAVE_CUDA
  freeHashBufferGpuMemory();
#endif
  freeHashBufferCpuMemory();
}

void JoinHashTable::freeHashBufferGpuMemory() {
#ifdef HAVE_CUDA
  const auto& catalog = *executor_->getCatalog();
  auto& data_mgr = catalog.getDataMgr();
  for (auto& buf : gpu_hash_table_buff_) {
    if (buf) {
      free_gpu_abstract_buffer(&data_mgr, buf);
      buf = nullptr;
    }
  }
  for (auto& buf : gpu_hash_table_err_buff_) {
    if (buf) {
      free_gpu_abstract_buffer(&data_mgr, buf);
      buf = nullptr;
    }
  }
#else
  CHECK(false);
#endif  // HAVE_CUDA
}

void JoinHashTable::freeHashBufferCpuMemory() {
  cpu_hash_table_buff_.reset();
}
