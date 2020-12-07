/*
 * Copyright 2019 MapD Technologies, Inc.
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

#include "QueryEngine/JoinHashTable/HashJoin.h"

#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/EquiJoinCondition.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/JoinHashTable.h"
#include "QueryEngine/JoinHashTable/OverlapsJoinHashTable.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "QueryEngine/ScalarExprVisitor.h"

extern bool g_enable_overlaps_hashjoin;

//! fetchJoinColumn() calls ColumnFetcher::makeJoinColumn(), then copies the
//! JoinColumn's col_chunks_buff memory onto the GPU if required by the
//! effective_memory_level parameter. The dev_buff_owner parameter will
//! manage the GPU memory.
JoinColumn HashJoin::fetchJoinColumn(
    const Analyzer::ColumnVar* hash_col,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragment_info,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    DeviceAllocator* dev_buff_owner,
    std::vector<std::shared_ptr<void>>& malloc_owner,
    Executor* executor,
    ColumnCacheMap* column_cache) {
  static std::mutex fragment_fetch_mutex;
  std::lock_guard<std::mutex> fragment_fetch_lock(fragment_fetch_mutex);
  try {
    JoinColumn join_column = ColumnFetcher::makeJoinColumn(executor,
                                                           *hash_col,
                                                           fragment_info,
                                                           effective_memory_level,
                                                           device_id,
                                                           dev_buff_owner,
                                                           chunks_owner,
                                                           malloc_owner,
                                                           *column_cache);
    if (effective_memory_level == Data_Namespace::GPU_LEVEL) {
      CHECK(dev_buff_owner);
      auto device_col_chunks_buff = dev_buff_owner->alloc(join_column.col_chunks_buff_sz);
      dev_buff_owner->copyToDevice(device_col_chunks_buff,
                                   join_column.col_chunks_buff,
                                   join_column.col_chunks_buff_sz);
      join_column.col_chunks_buff = device_col_chunks_buff;
    }
    return join_column;
  } catch (...) {
    throw FailedToFetchColumn();
  }
}

namespace {

template <typename T>
std::string toStringFlat(const HashJoin* hash_table,
                         const ExecutorDeviceType device_type,
                         const int device_id) {
  auto mem =
      reinterpret_cast<const T*>(hash_table->getJoinHashBuffer(device_type, device_id));
  auto memsz = hash_table->getJoinHashBufferSize(device_type, device_id) / sizeof(T);
  std::string txt;
  for (size_t i = 0; i < memsz; ++i) {
    if (i > 0) {
      txt += ", ";
    }
    txt += std::to_string(mem[i]);
  }
  return txt;
}

}  // anonymous namespace

std::string HashJoin::toStringFlat64(const ExecutorDeviceType device_type,
                                     const int device_id) const {
  return toStringFlat<int64_t>(this, device_type, device_id);
}

std::string HashJoin::toStringFlat32(const ExecutorDeviceType device_type,
                                     const int device_id) const {
  return toStringFlat<int32_t>(this, device_type, device_id);
}

std::ostream& operator<<(std::ostream& os, const DecodedJoinHashBufferEntry& e) {
  os << "  {{";
  bool first = true;
  for (auto k : e.key) {
    if (!first) {
      os << ",";
    } else {
      first = false;
    }
    os << k;
  }
  os << "}, ";
  os << "{";
  first = true;
  for (auto p : e.payload) {
    if (!first) {
      os << ", ";
    } else {
      first = false;
    }
    os << p;
  }
  os << "}}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const DecodedJoinHashBufferSet& s) {
  os << "{\n";
  bool first = true;
  for (auto e : s) {
    if (!first) {
      os << ",\n";
    } else {
      first = false;
    }
    os << e;
  }
  if (!s.empty()) {
    os << "\n";
  }
  os << "}\n";
  return os;
}

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<HashJoin> HashJoin::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<HashJoin> join_hash_table;
  CHECK_GT(device_count, 0);
  if (!g_enable_overlaps_hashjoin && qual_bin_oper->is_overlaps_oper()) {
    throw std::runtime_error(
        "Overlaps hash join disabled, attempting to fall back to loop join");
  }
  if (qual_bin_oper->is_overlaps_oper()) {
    VLOG(1) << "Trying to build geo hash table:";
    join_hash_table = OverlapsJoinHashTable::getInstance(
        qual_bin_oper, query_infos, memory_level, device_count, column_cache, executor);
  } else if (dynamic_cast<const Analyzer::ExpressionTuple*>(
                 qual_bin_oper->get_left_operand())) {
    VLOG(1) << "Trying to build keyed hash table:";
    join_hash_table = BaselineJoinHashTable::getInstance(qual_bin_oper,
                                                         query_infos,
                                                         memory_level,
                                                         preferred_hash_type,
                                                         device_count,
                                                         column_cache,
                                                         executor);
  } else {
    try {
      VLOG(1) << "Trying to build perfect hash table:";
      join_hash_table = JoinHashTable::getInstance(qual_bin_oper,
                                                   query_infos,
                                                   memory_level,
                                                   preferred_hash_type,
                                                   device_count,
                                                   column_cache,
                                                   executor);
    } catch (TooManyHashEntries&) {
      const auto join_quals = coalesce_singleton_equi_join(qual_bin_oper);
      CHECK_EQ(join_quals.size(), size_t(1));
      const auto join_qual =
          std::dynamic_pointer_cast<Analyzer::BinOper>(join_quals.front());
      VLOG(1) << "Trying to build keyed hash table after perfect hash table:";
      join_hash_table = BaselineJoinHashTable::getInstance(join_qual,
                                                           query_infos,
                                                           memory_level,
                                                           preferred_hash_type,
                                                           device_count,
                                                           column_cache,
                                                           executor);
    }
  }
  CHECK(join_hash_table);
  if (VLOGGING(2)) {
    if (join_hash_table->getMemoryLevel() == Data_Namespace::MemoryLevel::GPU_LEVEL) {
      for (int device_id = 0; device_id < join_hash_table->getDeviceCount();
           ++device_id) {
        if (join_hash_table->getJoinHashBufferSize(ExecutorDeviceType::GPU, device_id) <=
            1000) {
          VLOG(2) << "Built GPU hash table: "
                  << join_hash_table->toString(ExecutorDeviceType::GPU, device_id);
        }
      }
    } else {
      if (join_hash_table->getJoinHashBufferSize(ExecutorDeviceType::CPU) <= 1000) {
        VLOG(2) << "Built CPU hash table: "
                << join_hash_table->toString(ExecutorDeviceType::CPU);
      }
    }
  }
  return join_hash_table;
}

CompositeKeyInfo HashJoin::getCompositeKeyInfo(
    const std::vector<InnerOuter>& inner_outer_pairs,
    const Executor* executor) {
  CHECK(executor);
  std::vector<const void*> sd_inner_proxy_per_key;
  std::vector<const void*> sd_outer_proxy_per_key;
  std::vector<ChunkKey> cache_key_chunks;  // used for the cache key
  const auto db_id = executor->getCatalog()->getCurrentDB().dbId;
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto inner_col = inner_outer_pair.first;
    const auto outer_col = inner_outer_pair.second;
    const auto& inner_ti = inner_col->get_type_info();
    const auto& outer_ti = outer_col->get_type_info();
    ChunkKey cache_key_chunks_for_column{
        db_id, inner_col->get_table_id(), inner_col->get_column_id()};
    if (inner_ti.is_string() &&
        !(inner_ti.get_comp_param() == outer_ti.get_comp_param())) {
      CHECK(outer_ti.is_string());
      CHECK(inner_ti.get_compression() == kENCODING_DICT &&
            outer_ti.get_compression() == kENCODING_DICT);
      const auto sd_inner_proxy = executor->getStringDictionaryProxy(
          inner_ti.get_comp_param(), executor->getRowSetMemoryOwner(), true);
      const auto sd_outer_proxy = executor->getStringDictionaryProxy(
          outer_ti.get_comp_param(), executor->getRowSetMemoryOwner(), true);
      CHECK(sd_inner_proxy && sd_outer_proxy);
      sd_inner_proxy_per_key.push_back(sd_inner_proxy);
      sd_outer_proxy_per_key.push_back(sd_outer_proxy);
      cache_key_chunks_for_column.push_back(sd_outer_proxy->getGeneration());
    } else {
      sd_inner_proxy_per_key.emplace_back();
      sd_outer_proxy_per_key.emplace_back();
    }
    cache_key_chunks.push_back(cache_key_chunks_for_column);
  }
  return {sd_inner_proxy_per_key, sd_outer_proxy_per_key, cache_key_chunks};
}

std::shared_ptr<Analyzer::ColumnVar> getSyntheticColumnVar(std::string_view table,
                                                           std::string_view column,
                                                           int rte_idx,
                                                           Executor* executor) {
  auto catalog = executor->getCatalog();
  CHECK(catalog);

  auto tmeta = catalog->getMetadataForTable(std::string(table));
  CHECK(tmeta);

  auto cmeta = catalog->getMetadataForColumn(tmeta->tableId, std::string(column));
  CHECK(cmeta);

  auto ti = cmeta->columnType;

  if (ti.is_geometry() && ti.get_type() != kPOINT) {
    int geoColumnId{0};
    switch (ti.get_type()) {
      case kLINESTRING: {
        geoColumnId = cmeta->columnId + 2;
        break;
      }
      case kPOLYGON: {
        geoColumnId = cmeta->columnId + 3;
        break;
      }
      case kMULTIPOLYGON: {
        geoColumnId = cmeta->columnId + 4;
        break;
      }
      default:
        CHECK(false);
    }
    cmeta = catalog->getMetadataForColumn(tmeta->tableId, geoColumnId);
    CHECK(cmeta);
    ti = cmeta->columnType;
  }

  auto cv =
      std::make_shared<Analyzer::ColumnVar>(ti, tmeta->tableId, cmeta->columnId, rte_idx);
  return cv;
}

class AllColumnVarsVisitor
    : public ScalarExprVisitor<std::set<const Analyzer::ColumnVar*>> {
 protected:
  std::set<const Analyzer::ColumnVar*> visitColumnVar(
      const Analyzer::ColumnVar* column) const override {
    return {column};
  }

  std::set<const Analyzer::ColumnVar*> visitColumnVarTuple(
      const Analyzer::ExpressionTuple* expr_tuple) const override {
    AllColumnVarsVisitor visitor;
    std::set<const Analyzer::ColumnVar*> result;
    for (const auto& expr_component : expr_tuple->getTuple()) {
      const auto component_rte_set = visitor.visit(expr_component.get());
      result.insert(component_rte_set.begin(), component_rte_set.end());
    }
    return result;
  }

  std::set<const Analyzer::ColumnVar*> aggregateResult(
      const std::set<const Analyzer::ColumnVar*>& aggregate,
      const std::set<const Analyzer::ColumnVar*>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

void setupSyntheticCaching(std::set<const Analyzer::ColumnVar*> cvs, Executor* executor) {
  std::unordered_set<int> phys_table_ids;
  for (auto cv : cvs) {
    phys_table_ids.insert(cv->get_table_id());
  }

  std::unordered_set<PhysicalInput> phys_inputs;
  for (auto cv : cvs) {
    phys_inputs.emplace(PhysicalInput{cv->get_column_id(), cv->get_table_id()});
  }

  executor->setupCaching(phys_inputs, phys_table_ids);
}

std::vector<InputTableInfo> getSyntheticInputTableInfo(
    std::set<const Analyzer::ColumnVar*> cvs,
    Executor* executor) {
  auto catalog = executor->getCatalog();
  CHECK(catalog);

  std::unordered_set<int> phys_table_ids;
  for (auto cv : cvs) {
    phys_table_ids.insert(cv->get_table_id());
  }

  // NOTE(sy): This vector ordering seems to work for now, but maybe we need to
  // review how rte_idx is assigned for ColumnVars. See for example Analyzer.h
  // and RelAlgExecutor.cpp and rte_idx there.
  std::vector<InputTableInfo> query_infos(phys_table_ids.size());
  size_t i = 0;
  for (auto id : phys_table_ids) {
    auto tmeta = catalog->getMetadataForTable(id);
    query_infos[i].table_id = id;
    query_infos[i].info = tmeta->fragmenter->getFragmentsForQuery();
    ++i;
  }

  return query_infos;
}

//! Make hash table from named tables and columns (such as for testing).
std::shared_ptr<HashJoin> HashJoin::getSyntheticInstance(
    std::string_view table1,
    std::string_view column1,
    std::string_view table2,
    std::string_view column2,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  auto a1 = getSyntheticColumnVar(table1, column1, 0, executor);
  auto a2 = getSyntheticColumnVar(table2, column2, 1, executor);

  auto qual_bin_oper = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, a1, a2);

  std::set<const Analyzer::ColumnVar*> cvs =
      AllColumnVarsVisitor().visit(qual_bin_oper.get());
  auto query_infos = getSyntheticInputTableInfo(cvs, executor);
  setupSyntheticCaching(cvs, executor);

  auto hash_table = HashJoin::getInstance(qual_bin_oper,
                                          query_infos,
                                          memory_level,
                                          preferred_hash_type,
                                          device_count,
                                          column_cache,
                                          executor);
  return hash_table;
}

//! Make hash table from named tables and columns (such as for testing).
std::shared_ptr<HashJoin> HashJoin::getSyntheticInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  std::set<const Analyzer::ColumnVar*> cvs =
      AllColumnVarsVisitor().visit(qual_bin_oper.get());
  auto query_infos = getSyntheticInputTableInfo(cvs, executor);
  setupSyntheticCaching(cvs, executor);

  auto hash_table = HashJoin::getInstance(qual_bin_oper,
                                          query_infos,
                                          memory_level,
                                          preferred_hash_type,
                                          device_count,
                                          column_cache,
                                          executor);
  return hash_table;
}

void HashJoin::checkHashJoinReplicationConstraint(const int table_id,
                                                  const size_t shard_count,
                                                  const Executor* executor) {
  if (!g_cluster) {
    return;
  }
  if (table_id >= 0) {
    CHECK(executor);
    const auto inner_td = executor->getCatalog()->getMetadataForTable(table_id);
    CHECK(inner_td);
    if (!shard_count && !table_is_replicated(inner_td)) {
      throw TableMustBeReplicated(inner_td->tableName);
    }
  }
}

namespace {

InnerOuter get_cols(const Analyzer::BinOper* qual_bin_oper,
                    const Catalog_Namespace::Catalog& cat,
                    const TemporaryTables* temporary_tables) {
  const auto lhs = qual_bin_oper->get_left_operand();
  const auto rhs = qual_bin_oper->get_right_operand();
  return normalize_column_pair(lhs, rhs, cat, temporary_tables);
}

}  // namespace

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
