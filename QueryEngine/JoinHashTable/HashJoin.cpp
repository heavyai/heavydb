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

#include "QueryEngine/JoinHashTable/HashJoin.h"

#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/EquiJoinCondition.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/OverlapsJoinHashTable.h"
#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"
#include "QueryEngine/RangeTableIndexVisitor.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "QueryEngine/ScalarExprVisitor.h"

#include <sstream>

extern bool g_enable_overlaps_hashjoin;

void ColumnsForDevice::setBucketInfo(
    const std::vector<double>& inverse_bucket_sizes_for_dimension,
    const std::vector<InnerOuter> inner_outer_pairs) {
  join_buckets.clear();

  CHECK_EQ(inner_outer_pairs.size(), join_columns.size());
  CHECK_EQ(join_columns.size(), join_column_types.size());
  for (size_t i = 0; i < join_columns.size(); i++) {
    const auto& inner_outer_pair = inner_outer_pairs[i];
    const auto inner_col = inner_outer_pair.first;
    const auto& ti = inner_col->get_type_info();
    const auto elem_ti = ti.get_elem_type();
    // CHECK(elem_ti.is_fp());

    join_buckets.emplace_back(JoinBucketInfo{inverse_bucket_sizes_for_dimension,
                                             elem_ti.get_type() == kDOUBLE});
  }
}

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
                                                           /*thread_idx=*/0,
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

std::ostream& operator<<(std::ostream& os,
                         const InnerOuterStringOpInfos& inner_outer_string_op_infos) {
  os << "(" << inner_outer_string_op_infos.first << ", "
     << inner_outer_string_op_infos.second << ")";
  return os;
}

std::string toString(const InnerOuterStringOpInfos& inner_outer_string_op_infos) {
  std::ostringstream os;
  os << inner_outer_string_op_infos;
  return os.str();
}

std::ostream& operator<<(
    std::ostream& os,
    const std::vector<InnerOuterStringOpInfos>& inner_outer_string_op_infos_pairs) {
  os << "[";
  bool first_elem = true;
  for (const auto& inner_outer_string_op_infos : inner_outer_string_op_infos_pairs) {
    if (!first_elem) {
      os << ", ";
    }
    first_elem = false;
    os << inner_outer_string_op_infos;
  }
  os << "]";
  return os;
}

std::string toString(
    const std::vector<InnerOuterStringOpInfos>& inner_outer_string_op_infos_pairs) {
  std::ostringstream os;
  os << inner_outer_string_op_infos_pairs;
  return os.str();
}

HashJoinMatchingSet HashJoin::codegenMatchingSet(
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
  auto rowid_ptr_i32 = executor->cgen_state_->ir_builder_.CreateGEP(
      rowid_base_i32->getType()->getScalarType()->getPointerElementType(),
      rowid_base_i32,
      slot_lv);
  return {rowid_ptr_i32, row_count_lv, slot_lv};
}

llvm::Value* HashJoin::codegenHashTableLoad(const size_t table_idx, Executor* executor) {
  AUTOMATIC_IR_METADATA(executor->cgen_state_.get());
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
                  hash_tables_ptr->getType()->getScalarType()->getPointerElementType(),
                  hash_tables_ptr,
                  executor->cgen_state_->llInt(static_cast<int64_t>(table_idx)))
            : hash_tables_ptr;
    hash_ptr = executor->cgen_state_->ir_builder_.CreateLoad(
        hash_pptr->getType()->getPointerElementType(), hash_pptr);
  } else {
    hash_ptr = get_arg_by_name(executor->cgen_state_->row_func_, "join_hash_tables");
  }
  CHECK(hash_ptr);
  return hash_ptr;
}

//! Make hash table from an in-flight SQL query's parse tree etc.
std::shared_ptr<HashJoin> HashJoin::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const std::vector<InputTableInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const JoinType join_type,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor,
    const HashTableBuildDagMap& hashtable_build_dag_map,
    const RegisteredQueryHint& query_hint,
    const TableIdToNodeMap& table_id_to_node_map) {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_ptr<HashJoin> join_hash_table;
  CHECK_GT(device_count, 0);
  if (!g_enable_overlaps_hashjoin && qual_bin_oper->is_overlaps_oper()) {
    throw std::runtime_error(
        "Overlaps hash join disabled, attempting to fall back to loop join");
  }
  if (qual_bin_oper->is_overlaps_oper()) {
    VLOG(1) << "Trying to build geo hash table:";
    join_hash_table = OverlapsJoinHashTable::getInstance(qual_bin_oper,
                                                         query_infos,
                                                         memory_level,
                                                         join_type,
                                                         device_count,
                                                         column_cache,
                                                         executor,
                                                         hashtable_build_dag_map,
                                                         query_hint,
                                                         table_id_to_node_map);
  } else if (dynamic_cast<const Analyzer::ExpressionTuple*>(
                 qual_bin_oper->get_left_operand())) {
    VLOG(1) << "Trying to build keyed hash table:";
    join_hash_table = BaselineJoinHashTable::getInstance(qual_bin_oper,
                                                         query_infos,
                                                         memory_level,
                                                         join_type,
                                                         preferred_hash_type,
                                                         device_count,
                                                         column_cache,
                                                         executor,
                                                         hashtable_build_dag_map,
                                                         query_hint,
                                                         table_id_to_node_map);
  } else {
    try {
      VLOG(1) << "Trying to build perfect hash table:";
      join_hash_table = PerfectJoinHashTable::getInstance(qual_bin_oper,
                                                          query_infos,
                                                          memory_level,
                                                          join_type,
                                                          preferred_hash_type,
                                                          device_count,
                                                          column_cache,
                                                          executor,
                                                          hashtable_build_dag_map,
                                                          query_hint,
                                                          table_id_to_node_map);
    } catch (JoinHashTableTooBig& e) {
      throw e;
    } catch (TooManyHashEntries&) {
      const auto join_quals = coalesce_singleton_equi_join(qual_bin_oper);
      CHECK_EQ(join_quals.size(), size_t(1));
      const auto join_qual =
          std::dynamic_pointer_cast<Analyzer::BinOper>(join_quals.front());
      VLOG(1) << "Trying to build keyed hash table after perfect hash table:";
      join_hash_table = BaselineJoinHashTable::getInstance(join_qual,
                                                           query_infos,
                                                           memory_level,
                                                           join_type,
                                                           preferred_hash_type,
                                                           device_count,
                                                           column_cache,
                                                           executor,
                                                           hashtable_build_dag_map,
                                                           query_hint,
                                                           table_id_to_node_map);
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

std::pair<const StringDictionaryProxy*, StringDictionaryProxy*>
HashJoin::getStrDictProxies(const InnerOuter& cols,
                            const Executor* executor,
                            const bool has_string_ops) {
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto inner_ti = inner_col->get_type_info();
  const auto outer_col = dynamic_cast<const Analyzer::ColumnVar*>(cols.second);
  std::pair<const StringDictionaryProxy*, StringDictionaryProxy*>
      inner_outer_str_dict_proxies{nullptr, nullptr};
  if (inner_ti.is_string() && outer_col) {
    CHECK(outer_col->get_type_info().is_string());
    inner_outer_str_dict_proxies.first =
        executor->getStringDictionaryProxy(inner_col->get_comp_param(), true);
    CHECK(inner_outer_str_dict_proxies.first);
    inner_outer_str_dict_proxies.second =
        executor->getStringDictionaryProxy(outer_col->get_comp_param(), true);
    CHECK(inner_outer_str_dict_proxies.second);
    if (!has_string_ops &&
        *inner_outer_str_dict_proxies.first == *inner_outer_str_dict_proxies.second) {
      // Dictionaries are the same - don't need to translate
      CHECK(inner_col->get_comp_param() == outer_col->get_comp_param());
      inner_outer_str_dict_proxies.first = nullptr;
      inner_outer_str_dict_proxies.second = nullptr;
    }
  }
  return inner_outer_str_dict_proxies;
}

const StringDictionaryProxy::IdMap* HashJoin::translateInnerToOuterStrDictProxies(
    const InnerOuter& cols,
    const InnerOuterStringOpInfos& inner_outer_string_op_infos,
    ExpressionRange& col_range,
    const Executor* executor) {
  const bool has_string_ops = inner_outer_string_op_infos.first.size() ||
                              inner_outer_string_op_infos.second.size();
  const auto inner_outer_proxies =
      HashJoin::getStrDictProxies(cols, executor, has_string_ops);
  const bool translate_dictionary =
      inner_outer_proxies.first && inner_outer_proxies.second;
  if (translate_dictionary) {
    const auto inner_dict_id = inner_outer_proxies.first->getDictId();
    const auto outer_dict_id = inner_outer_proxies.second->getDictId();
    CHECK(has_string_ops || inner_dict_id != outer_dict_id);
    const auto id_map = executor->getJoinIntersectionStringProxyTranslationMap(
        inner_outer_proxies.first,
        inner_outer_proxies.second,
        inner_outer_string_op_infos.first,
        inner_outer_string_op_infos.second,
        executor->getRowSetMemoryOwner());
    if (!inner_outer_string_op_infos.second.empty()) {
      // String op was applied to lhs table,
      // need to expand column range appropriately
      col_range = ExpressionRange::makeIntRange(
          std::min(col_range.getIntMin(),
                   static_cast<int64_t>(
                       inner_outer_proxies.second->transientEntryCount() + 1) *
                       -1),
          col_range.getIntMax(),
          0,
          col_range.hasNulls());
    }
    return id_map;
  }
  return nullptr;
}

std::vector<int> HashJoin::collectFragmentIds(
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments) {
  auto const fragment_id = [](auto const& frag_info) { return frag_info.fragmentId; };
  std::vector<int> frag_ids(fragments.size());
  std::transform(fragments.cbegin(), fragments.cend(), frag_ids.begin(), fragment_id);
  std::sort(frag_ids.begin(), frag_ids.end());
  return frag_ids;
}

CompositeKeyInfo HashJoin::getCompositeKeyInfo(
    const std::vector<InnerOuter>& inner_outer_pairs,
    const Executor* executor,
    const std::vector<InnerOuterStringOpInfos>& inner_outer_string_op_infos_pairs) {
  CHECK(executor);
  std::vector<const void*> sd_inner_proxy_per_key;
  std::vector<void*> sd_outer_proxy_per_key;
  std::vector<ChunkKey> cache_key_chunks;  // used for the cache key
  const auto db_id = executor->getCatalog()->getCurrentDB().dbId;
  const bool has_string_op_infos = inner_outer_string_op_infos_pairs.size();
  if (has_string_op_infos) {
    CHECK_EQ(inner_outer_pairs.size(), inner_outer_string_op_infos_pairs.size());
  }
  size_t string_op_info_pairs_idx = 0;
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto inner_col = inner_outer_pair.first;
    const auto outer_col = inner_outer_pair.second;
    const auto& inner_ti = inner_col->get_type_info();
    const auto& outer_ti = outer_col->get_type_info();
    ChunkKey cache_key_chunks_for_column{
        db_id, inner_col->get_table_id(), inner_col->get_column_id()};
    if (inner_ti.is_string() &&
        (!(inner_ti.get_comp_param() == outer_ti.get_comp_param()) ||
         (has_string_op_infos &&
          (inner_outer_string_op_infos_pairs[string_op_info_pairs_idx].first.size() ||
           inner_outer_string_op_infos_pairs[string_op_info_pairs_idx].second.size())))) {
      CHECK(outer_ti.is_string());
      CHECK(inner_ti.get_compression() == kENCODING_DICT &&
            outer_ti.get_compression() == kENCODING_DICT);
      const auto sd_inner_proxy = executor->getStringDictionaryProxy(
          inner_ti.get_comp_param(), executor->getRowSetMemoryOwner(), true);
      auto sd_outer_proxy = executor->getStringDictionaryProxy(
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
    string_op_info_pairs_idx++;
  }
  return {sd_inner_proxy_per_key, sd_outer_proxy_per_key, cache_key_chunks};
}

std::vector<const StringDictionaryProxy::IdMap*>
HashJoin::translateCompositeStrDictProxies(
    const CompositeKeyInfo& composite_key_info,
    const std::vector<InnerOuterStringOpInfos>& string_op_infos_for_keys,
    const Executor* executor) {
  const auto& inner_proxies = composite_key_info.sd_inner_proxy_per_key;
  const auto& outer_proxies = composite_key_info.sd_outer_proxy_per_key;
  const size_t num_proxies = inner_proxies.size();
  CHECK_EQ(num_proxies, outer_proxies.size());
  std::vector<const StringDictionaryProxy::IdMap*> proxy_translation_maps;
  proxy_translation_maps.reserve(num_proxies);
  for (size_t proxy_pair_idx = 0; proxy_pair_idx < num_proxies; ++proxy_pair_idx) {
    const bool translate_proxies =
        inner_proxies[proxy_pair_idx] && outer_proxies[proxy_pair_idx];
    if (translate_proxies) {
      const auto inner_proxy =
          reinterpret_cast<const StringDictionaryProxy*>(inner_proxies[proxy_pair_idx]);
      auto outer_proxy =
          reinterpret_cast<StringDictionaryProxy*>(outer_proxies[proxy_pair_idx]);
      CHECK(inner_proxy);
      CHECK(outer_proxy);

      CHECK_NE(inner_proxy->getDictId(), outer_proxy->getDictId());
      proxy_translation_maps.emplace_back(
          executor->getJoinIntersectionStringProxyTranslationMap(
              inner_proxy,
              outer_proxy,
              string_op_infos_for_keys[proxy_pair_idx].first,
              string_op_infos_for_keys[proxy_pair_idx].second,
              executor->getRowSetMemoryOwner()));
    } else {
      proxy_translation_maps.emplace_back(nullptr);
    }
  }
  return proxy_translation_maps;
}

llvm::Value* HashJoin::codegenColOrStringOper(
    const Analyzer::Expr* col_or_string_oper,
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos,
    CodeGenerator& code_generator,
    const CompilationOptions& co) {
  if (!string_op_infos.empty()) {
    const auto coerced_col_var =
        dynamic_cast<const Analyzer::ColumnVar*>(col_or_string_oper);
    CHECK(coerced_col_var);
    std::vector<llvm::Value*> codegen_val_vec{
        code_generator.codegenPseudoStringOper(coerced_col_var, string_op_infos, co)};
    return codegen_val_vec[0];
  }
  return code_generator.codegen(col_or_string_oper, true, co)[0];
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
  RegisteredQueryHint query_hint = RegisteredQueryHint::defaults();

  auto hash_table = HashJoin::getInstance(qual_bin_oper,
                                          query_infos,
                                          memory_level,
                                          JoinType::INNER,
                                          preferred_hash_type,
                                          device_count,
                                          column_cache,
                                          executor,
                                          {},
                                          query_hint,
                                          {});
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
  RegisteredQueryHint query_hint = RegisteredQueryHint::defaults();

  auto hash_table = HashJoin::getInstance(qual_bin_oper,
                                          query_infos,
                                          memory_level,
                                          JoinType::INNER,
                                          preferred_hash_type,
                                          device_count,
                                          column_cache,
                                          executor,
                                          {},
                                          query_hint,
                                          {});
  return hash_table;
}

std::pair<std::string, std::shared_ptr<HashJoin>> HashJoin::getSyntheticInstance(
    std::vector<std::shared_ptr<Analyzer::BinOper>> qual_bin_opers,
    const Data_Namespace::MemoryLevel memory_level,
    const HashType preferred_hash_type,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  std::set<const Analyzer::ColumnVar*> cvs;
  for (auto& qual : qual_bin_opers) {
    auto cv = AllColumnVarsVisitor().visit(qual.get());
    cvs.insert(cv.begin(), cv.end());
  }
  auto query_infos = getSyntheticInputTableInfo(cvs, executor);
  setupSyntheticCaching(cvs, executor);
  RegisteredQueryHint query_hint = RegisteredQueryHint::defaults();
  std::shared_ptr<HashJoin> hash_table;
  std::string error_msg;
  for (auto& qual : qual_bin_opers) {
    try {
      auto candidate_hash_table = HashJoin::getInstance(qual,
                                                        query_infos,
                                                        memory_level,
                                                        JoinType::INNER,
                                                        preferred_hash_type,
                                                        device_count,
                                                        column_cache,
                                                        executor,
                                                        {},
                                                        query_hint,
                                                        {});
      if (candidate_hash_table) {
        hash_table = candidate_hash_table;
      }
    } catch (HashJoinFail& e) {
      error_msg = e.what();
      continue;
    }
  }
  return std::make_pair(error_msg, hash_table);
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

template <typename T>
const T* HashJoin::getHashJoinColumn(const Analyzer::Expr* expr) {
  auto* target_expr = expr;
  if (auto cast_expr = dynamic_cast<const Analyzer::UOper*>(expr)) {
    target_expr = cast_expr->get_operand();
  }
  CHECK(target_expr);
  return dynamic_cast<const T*>(target_expr);
}

std::pair<InnerOuter, InnerOuterStringOpInfos> HashJoin::normalizeColumnPair(
    const Analyzer::Expr* lhs,
    const Analyzer::Expr* rhs,
    const Catalog_Namespace::Catalog& cat,
    const TemporaryTables* temporary_tables,
    const bool is_overlaps_join) {
  SQLTypeInfo lhs_ti = lhs->get_type_info();
  SQLTypeInfo rhs_ti = rhs->get_type_info();
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
    throw HashJoinFail(
        "Cannot use hash join for given expression (non-cast unary operator)");
  }
  // Casts to decimal are not suported.
  if (lhs_ti.is_decimal() && (lhs_cast || rhs_cast)) {
    throw HashJoinFail("Cannot use hash join for given expression (cast to decimal)");
  }
  auto lhs_col = getHashJoinColumn<Analyzer::ColumnVar>(lhs);
  auto rhs_col = getHashJoinColumn<Analyzer::ColumnVar>(rhs);

  const auto lhs_string_oper = getHashJoinColumn<Analyzer::StringOper>(lhs);
  const auto rhs_string_oper = getHashJoinColumn<Analyzer::StringOper>(rhs);

  auto process_string_op_infos = [](const auto& string_oper, auto& col, auto& ti) {
    std::vector<StringOps_Namespace::StringOpInfo> string_op_infos;
    if (string_oper) {
      col = dynamic_cast<const Analyzer::ColumnVar*>(string_oper->getArg(0));
      if (!col) {
        // Todo (todd): Allow for non-colvar inputs into string operators for
        // join predicates
        // We now guard against non constant/colvar/stringoper inputs
        // in Analyzer::StringOper::check_operand_types, but keeping this to not
        // depend on that logic if and when it changes as allowing non-colvar inputs
        // for hash joins will be additional work on top of allowing them
        // outside of join predicates
        throw HashJoinFail(
            "Hash joins involving string operators currently restricted to column inputs "
            "(i.e. not case statements).");
      }
      ti = col->get_type_info();
      CHECK(ti.is_dict_encoded_string());
      const auto chained_string_op_exprs = string_oper->getChainedStringOpExprs();
      CHECK_GT(chained_string_op_exprs.size(), 0UL);
      for (const auto& chained_string_op_expr : chained_string_op_exprs) {
        auto chained_string_op =
            dynamic_cast<const Analyzer::StringOper*>(chained_string_op_expr.get());
        CHECK(chained_string_op);
        StringOps_Namespace::StringOpInfo string_op_info(
            chained_string_op->get_kind(),
            chained_string_op->get_type_info(),
            chained_string_op->getLiteralArgs());
        string_op_infos.emplace_back(string_op_info);
      }
    }
    return string_op_infos;
  };

  auto outer_string_op_infos = process_string_op_infos(lhs_string_oper, lhs_col, lhs_ti);
  auto inner_string_op_infos = process_string_op_infos(rhs_string_oper, rhs_col, rhs_ti);

  if (!lhs_col && !rhs_col) {
    throw HashJoinFail(
        "Cannot use hash join for given expression (both lhs and rhs are invalid)",
        InnerQualDecision::UNKNOWN);
  }

  const Analyzer::ColumnVar* inner_col{nullptr};
  const Analyzer::ColumnVar* outer_col{nullptr};
  auto outer_ti = lhs_ti;
  auto inner_ti = rhs_ti;
  const Analyzer::Expr* outer_expr{lhs};
  InnerQualDecision inner_qual_decision = InnerQualDecision::UNKNOWN;
  if (!lhs_col || (rhs_col && lhs_col->get_rte_idx() < rhs_col->get_rte_idx())) {
    inner_qual_decision = InnerQualDecision::RHS;
    inner_col = rhs_col;
    outer_col = lhs_col;
  } else {
    inner_qual_decision = InnerQualDecision::LHS;
    if (lhs_col && lhs_col->get_rte_idx() == 0) {
      throw HashJoinFail(
          "Cannot use hash join for given expression (lhs' rte idx is zero)",
          inner_qual_decision);
    }
    inner_col = lhs_col;
    outer_col = rhs_col;
    std::swap(outer_ti, inner_ti);
    std::swap(outer_string_op_infos, inner_string_op_infos);
    outer_expr = rhs;
  }
  if (!inner_col) {
    throw HashJoinFail("Cannot use hash join for given expression (invalid inner col)",
                       inner_qual_decision);
  }
  if (!outer_col) {
    // check whether outer_col is a constant, i.e., inner_col = K;
    const auto outer_constant_col = dynamic_cast<const Analyzer::Constant*>(outer_expr);
    if (outer_constant_col) {
      throw HashJoinFail(
          "Cannot use hash join for given expression: try to join with a constant "
          "value",
          inner_qual_decision);
    }
    MaxRangeTableIndexVisitor rte_idx_visitor;
    int outer_rte_idx = rte_idx_visitor.visit(outer_expr);
    // The inner column candidate is not actually inner; the outer
    // expression contains columns which are at least as deep.
    if (inner_col->get_rte_idx() <= outer_rte_idx) {
      throw HashJoinFail(
          "Cannot use hash join for given expression (inner's rte <= outer's rte)",
          inner_qual_decision);
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
    throw HashJoinFail("Cannot use hash join for given expression (cast from decimal)");
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
    if (!(outer_col_ti.get_type() == kPOINT || is_bounds_array(outer_col_ti) ||
          is_constructed_point(outer_expr))) {
      throw HashJoinFail(
          "Overlaps join only supported for geometry outer columns of type point, "
          "geometry columns with bounds or constructed points");
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
  return std::make_pair(std::make_pair(normalized_inner_col, normalized_outer_col),
                        std::make_pair(inner_string_op_infos, outer_string_op_infos));
}

std::pair<std::vector<InnerOuter>, std::vector<InnerOuterStringOpInfos>>
HashJoin::normalizeColumnPairs(const Analyzer::BinOper* condition,
                               const Catalog_Namespace::Catalog& cat,
                               const TemporaryTables* temporary_tables) {
  std::pair<std::vector<InnerOuter>, std::vector<InnerOuterStringOpInfos>> result;
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
      const auto col_pair = normalizeColumnPair(lhs_tuple[i].get(),
                                                rhs_tuple[i].get(),
                                                cat,
                                                temporary_tables,
                                                condition->is_overlaps_oper());
      result.first.emplace_back(col_pair.first);
      result.second.emplace_back(col_pair.second);
    }
  } else {
    CHECK(!lhs_tuple_expr && !rhs_tuple_expr);
    const auto col_pair = normalizeColumnPair(condition->get_left_operand(),
                                              condition->get_right_operand(),
                                              cat,
                                              temporary_tables,
                                              condition->is_overlaps_oper());
    result.first.emplace_back(col_pair.first);
    result.second.emplace_back(col_pair.second);
  }

  return result;
}

bool HashJoin::canAccessHashTable(bool allow_hash_table_recycling,
                                  bool invalid_cache_key,
                                  JoinType join_type) {
  return g_enable_data_recycler && g_use_hashtable_cache && !invalid_cache_key &&
         allow_hash_table_recycling && join_type != JoinType::INVALID;
}

namespace {

InnerOuter get_cols(const Analyzer::BinOper* qual_bin_oper,
                    const Catalog_Namespace::Catalog& cat,
                    const TemporaryTables* temporary_tables) {
  const auto lhs = qual_bin_oper->get_left_operand();
  const auto rhs = qual_bin_oper->get_right_operand();
  return HashJoin::normalizeColumnPair(lhs, rhs, cat, temporary_tables).first;
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
