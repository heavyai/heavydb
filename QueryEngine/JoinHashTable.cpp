#include "JoinHashTable.h"
#include "Execute.h"
#include "ExpressionRewrite.h"
#include "HashJoinRuntime.h"
#include "RuntimeFunctions.h"

#include <glog/logging.h>
#include <thread>

namespace {

std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*> get_cols(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const Catalog_Namespace::Catalog& cat,
    const TemporaryTables* temporary_tables) {
  const auto lhs = qual_bin_oper->get_left_operand();
  const auto rhs = qual_bin_oper->get_right_operand();
  const auto& lhs_ti = lhs->get_type_info();
  const auto& rhs_ti = rhs->get_type_info();
  CHECK_EQ(kENCODING_NONE, lhs_ti.get_compression());
  CHECK_EQ(kENCODING_NONE, rhs_ti.get_compression());
  if (lhs_ti.get_type() != rhs_ti.get_type()) {
    throw HashJoinFail("Equijoin types must be identical, found: " + lhs_ti.get_type_name() + ", " +
                       rhs_ti.get_type_name());
  }
  if (!lhs_ti.is_integer() && !lhs_ti.is_string()) {
    throw HashJoinFail("Cannot apply hash join to " + lhs_ti.get_type_name());
  }
  const auto lhs_cast = dynamic_cast<const Analyzer::UOper*>(lhs);
  const auto rhs_cast = dynamic_cast<const Analyzer::UOper*>(rhs);
  if (static_cast<bool>(lhs_cast) != static_cast<bool>(rhs_cast) || (lhs_cast && lhs_cast->get_optype() != kCAST) ||
      (rhs_cast && rhs_cast->get_optype() != kCAST)) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  const auto lhs_col = lhs_cast ? dynamic_cast<const Analyzer::ColumnVar*>(lhs_cast->get_operand())
                                : dynamic_cast<const Analyzer::ColumnVar*>(lhs);
  const auto rhs_col = rhs_cast ? dynamic_cast<const Analyzer::ColumnVar*>(rhs_cast->get_operand())
                                : dynamic_cast<const Analyzer::ColumnVar*>(rhs);
  if (!lhs_col || !rhs_col) {
    throw HashJoinFail("Cannot use hash join for given expression");
  }
  const Analyzer::ColumnVar* inner_col{nullptr};
  const Analyzer::ColumnVar* outer_col{nullptr};
  if (lhs_col->get_rte_idx() == 0 && rhs_col->get_rte_idx() == 1) {
    inner_col = rhs_col;
    outer_col = lhs_col;
  } else {
    CHECK((lhs_col->get_rte_idx() == 1 && rhs_col->get_rte_idx() == 0));
    inner_col = lhs_col;
    outer_col = rhs_col;
  }
  // We need to fetch the actual type information from the catalog since Analyzer
  // always reports nullable as true for inner table columns in left joins.
  const auto inner_col_cd = get_column_descriptor_maybe(inner_col->get_column_id(), inner_col->get_table_id(), cat);
  const auto& inner_col_real_ti =
      get_column_type(inner_col->get_column_id(), inner_col->get_table_id(), inner_col_cd, temporary_tables);
  const auto& outer_col_ti = outer_col->get_type_info();
  if (outer_col_ti.get_notnull() != inner_col_real_ti.get_notnull()) {
    throw HashJoinFail("For hash join, both sides must have the same nullability");
  }
  if (!(inner_col_real_ti.is_integer() ||
        (inner_col_real_ti.is_string() && inner_col_real_ti.get_compression() == kENCODING_DICT))) {
    throw HashJoinFail("Can only apply hash join to integer-like types and dictionary encoded strings");
  }
  return {inner_col, outer_col};
}

std::string get_table_name(const int table_id, const Catalog_Namespace::Catalog& cat) {
  if (table_id >= 1) {
    const auto td = cat.getMetadataForTable(table_id);
    CHECK(td);
    return td->tableName;
  }
  return "$TEMPORARY_TABLE" + std::to_string(-table_id);
}

}  // namespace

std::vector<std::pair<JoinHashTable::JoinHashTableCacheKey, std::shared_ptr<std::vector<int32_t>>>>
    JoinHashTable::join_hash_table_cache_;
std::mutex JoinHashTable::join_hash_table_cache_mutex_;

std::shared_ptr<JoinHashTable> JoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const Catalog_Namespace::Catalog& cat,
    const std::vector<InputTableInfo>& query_infos,
    const std::list<std::shared_ptr<const InputColDescriptor>>& input_col_descs,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_count,
    Executor* executor) {
  CHECK_EQ(kEQ, qual_bin_oper->get_optype());
  const auto redirected_bin_oper =
      std::dynamic_pointer_cast<Analyzer::BinOper>(redirect_expr(qual_bin_oper.get(), input_col_descs));
  CHECK(redirected_bin_oper);
  const auto cols = get_cols(redirected_bin_oper, cat, executor->temporary_tables_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& ti = inner_col->get_type_info();
  auto col_range = getExpressionRange(ti.is_string() ? cols.second : inner_col, query_infos, executor);
  if (col_range.getType() == ExpressionRangeType::Invalid) {
    throw HashJoinFail("Could not compute range for the expressions involved in the equijoin");
  }
  if (ti.is_string()) {
    // The nullable info must be the same as the source column.
    const auto source_col_range = getExpressionRange(inner_col, query_infos, executor);
    if (source_col_range.getType() == ExpressionRangeType::Invalid) {
      throw HashJoinFail("Could not compute range for the expressions involved in the equijoin");
    }
    col_range = ExpressionRange::makeIntRange(std::min(source_col_range.getIntMin(), col_range.getIntMin()),
                                              std::max(source_col_range.getIntMax(), col_range.getIntMax()),
                                              0,
                                              source_col_range.hasNulls());
  }
  auto join_hash_table = std::shared_ptr<JoinHashTable>(
      new JoinHashTable(qual_bin_oper, inner_col, cat, query_infos, memory_level, col_range, executor));
  const int err = join_hash_table->reify(device_count);
  if (err) {
    if (err == ERR_MULTI_FRAG) {
      const auto cols = get_cols(qual_bin_oper, cat, executor->temporary_tables_);
      const auto inner_col = cols.first;
      CHECK(inner_col);
      const auto& table_info = join_hash_table->getInnerQueryInfo(inner_col);
      throw HashJoinFail("Multi-fragment inner table '" + get_table_name(table_info.table_id, cat) +
                         "' not supported yet");
    }
    throw HashJoinFail("Could not build a 1-to-1 correspondence for columns involved in equijoin");
  }
  return join_hash_table;
}

int JoinHashTable::reify(const int device_count) noexcept {
  CHECK_LT(0, device_count);
  const auto cols = get_cols(qual_bin_oper_, cat_, executor_->temporary_tables_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& query_info = getInnerQueryInfo(inner_col).info;
  if (query_info.fragments.empty()) {
    return 0;
  }
  if (query_info.fragments.size() != 1) {  // we don't support multiple fragment inner tables (yet)
    return ERR_MULTI_FRAG;
  }
  const auto& fragment = query_info.fragments.front();
  auto chunk_meta_it = fragment.chunkMetadataMap.find(inner_col->get_column_id());
  CHECK(chunk_meta_it != fragment.chunkMetadataMap.end());
  const auto cd = get_column_descriptor_maybe(inner_col->get_column_id(), inner_col->get_table_id(), cat_);
  CHECK(!cd || !(cd->isVirtualCol));
  const auto& ti =
      get_column_type(inner_col->get_column_id(), inner_col->get_table_id(), cd, executor_->temporary_tables_);
  // Since we don't have the string dictionary payloads on the GPU, we'll build
  // the join hash table on the CPU and transfer it to the GPU.
  const auto effective_memory_level = ti.is_string() ? Data_Namespace::CPU_LEVEL : memory_level_;
#ifdef HAVE_CUDA
  gpu_hash_table_buff_.resize(device_count);
#endif
  std::vector<int> errors(device_count);
  std::vector<std::thread> init_threads;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::unique_ptr<const ColumnarResults> columnar_results;
  if (!cd) {
    columnar_results.reset(
        columnarize_result(executor_->row_set_mem_owner_,
                           get_temporary_table(executor_->temporary_tables_, inner_col->get_table_id()),
                           fragment.fragmentId));
  }
  ChunkKey chunk_key{
      cat_.get_currentDB().dbId, inner_col->get_table_id(), inner_col->get_column_id(), fragment.fragmentId};
  const JoinHashTableCacheKey cache_key{col_range_, *inner_col, *cols.second, fragment.numTuples, chunk_key};
  for (int device_id = 0; device_id < device_count; ++device_id) {
    const int8_t* col_buff{nullptr};
    if (cd) {
      CHECK(!columnar_results);
      const auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                                   &cat_.get_dataMgr(),
                                                   chunk_key,
                                                   effective_memory_level,
                                                   effective_memory_level == Data_Namespace::CPU_LEVEL ? 0 : device_id,
                                                   chunk_meta_it->second.numBytes,
                                                   chunk_meta_it->second.numElements);
      chunks_owner.push_back(chunk);
      CHECK(chunk);
      auto ab = chunk->get_buffer();
      CHECK(ab->getMemoryPtr());
      col_buff = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
    } else {
      CHECK(columnar_results);
      col_buff =
          Executor::ExecutionDispatch::getColumn(columnar_results.get(),
                                                 inner_col->get_column_id(),
                                                 &cat_.get_dataMgr(),
                                                 effective_memory_level,
                                                 effective_memory_level == Data_Namespace::CPU_LEVEL ? 0 : device_id);
    }
    init_threads.emplace_back(
        [&errors, &chunk_key, &chunk_meta_it, &cols, &fragment, col_buff, effective_memory_level, device_id, this] {
          try {
            errors[device_id] = initHashTableForDevice(
                chunk_key, col_buff, fragment.numTuples, cols, effective_memory_level, device_id);
          } catch (...) {
            errors[device_id] = -1;
          }
        });
  }
  for (auto& init_thread : init_threads) {
    init_thread.join();
  }
  for (const int err : errors) {
    if (err) {
      return err;
    }
  }
  return 0;
}

int JoinHashTable::initHashTableOnCpu(const int8_t* col_buff,
                                      const size_t num_elements,
                                      const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols,
                                      const int32_t hash_entry_count,
                                      const int32_t hash_join_invalid_val) {
  const auto inner_col = cols.first;
  CHECK(inner_col);
  const auto& ti = inner_col->get_type_info();
  int err = 0;
  if (!cpu_hash_table_buff_) {
    cpu_hash_table_buff_ = std::make_shared<std::vector<int32_t>>(hash_entry_count);
    const StringDictionary* sd_inner{nullptr};
    const StringDictionary* sd_outer{nullptr};
    if (ti.is_string()) {
      CHECK_EQ(kENCODING_DICT, ti.get_compression());
      sd_inner = executor_->getStringDictionary(inner_col->get_comp_param(), executor_->row_set_mem_owner_);
      CHECK(sd_inner);
      sd_outer = executor_->getStringDictionary(cols.second->get_comp_param(), executor_->row_set_mem_owner_);
      CHECK(sd_outer);
    }
    int thread_count = cpu_threads();
    std::vector<std::thread> init_cpu_buff_threads;
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      init_cpu_buff_threads.emplace_back([this, hash_entry_count, hash_join_invalid_val, thread_idx, thread_count] {
        init_hash_join_buff(
            &(*cpu_hash_table_buff_)[0], hash_entry_count, hash_join_invalid_val, thread_idx, thread_count);
      });
    }
    for (auto& t : init_cpu_buff_threads) {
      t.join();
    }
    init_cpu_buff_threads.clear();
    for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      init_cpu_buff_threads.emplace_back([this,
                                          hash_join_invalid_val,
                                          hash_entry_count,
                                          col_buff,
                                          num_elements,
                                          sd_inner,
                                          sd_outer,
                                          thread_idx,
                                          thread_count,
                                          &ti,
                                          &err] {
        int partial_err = fill_hash_join_buff(&(*cpu_hash_table_buff_)[0],
                                              hash_join_invalid_val,
                                              col_buff,
                                              num_elements,
                                              ti.get_size(),
                                              col_range_.getIntMin(),
                                              inline_fixed_encoding_null_val(ti),
                                              col_range_.getIntMax() + 1,
                                              sd_inner,
                                              sd_outer,
                                              thread_idx,
                                              thread_count);
        __sync_val_compare_and_swap(&err, 0, partial_err);
      });
    }
    for (auto& t : init_cpu_buff_threads) {
      t.join();
    }
  }
  return err;
}

int JoinHashTable::initHashTableForDevice(const ChunkKey& chunk_key,
                                          const int8_t* col_buff,
                                          const size_t num_elements,
                                          const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols,
                                          const Data_Namespace::MemoryLevel effective_memory_level,
                                          const int device_id) {
  const int32_t hash_entry_count =
      col_range_.getIntMax() - col_range_.getIntMin() + 1 + (col_range_.hasNulls() ? 1 : 0);
#ifdef HAVE_CUDA
  // Even if we join on dictionary encoded strings, the memory on the GPU is still needed
  // once the join hash table has been built on the CPU.
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    auto& data_mgr = cat_.get_dataMgr();
    gpu_hash_table_buff_[device_id] = alloc_gpu_mem(&data_mgr, hash_entry_count * sizeof(int32_t), device_id, nullptr);
  }
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  const auto inner_col = cols.first;
  CHECK(inner_col);
#ifdef HAVE_CUDA
  const auto& ti = inner_col->get_type_info();
#endif
  int err = 0;
  const int32_t hash_join_invalid_val{-1};
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    initHashTableOnCpuFromCache(chunk_key, num_elements, cols);
    {
      std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
      err = initHashTableOnCpu(col_buff, num_elements, cols, hash_entry_count, hash_join_invalid_val);
    }
    if (!err && inner_col->get_table_id() > 0) {
      putHashTableOnCpuToCache(chunk_key, num_elements, cols);
    }
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    // Don't transfer the buffer if there was an error since we'll bail anyway.
    if (memory_level_ == Data_Namespace::GPU_LEVEL && !err) {
#ifdef HAVE_CUDA
      CHECK(ti.is_string());
      auto& data_mgr = cat_.get_dataMgr();
      copy_to_gpu(&data_mgr,
                  gpu_hash_table_buff_[device_id],
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
    auto& data_mgr = cat_.get_dataMgr();
    auto dev_err_buff = alloc_gpu_mem(&data_mgr, sizeof(int), device_id, nullptr);
    copy_to_gpu(&data_mgr, dev_err_buff, &err, sizeof(err), device_id);
    init_hash_join_buff_on_device(reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]),
                                  hash_entry_count,
                                  hash_join_invalid_val,
                                  executor_->blockSize(),
                                  executor_->gridSize());
    fill_hash_join_buff_on_device(reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]),
                                  hash_join_invalid_val,
                                  reinterpret_cast<int*>(dev_err_buff),
                                  col_buff,
                                  num_elements,
                                  ti.get_size(),
                                  col_range_.getIntMin(),
                                  inline_fixed_encoding_null_val(ti),
                                  col_range_.getIntMax() + 1,
                                  executor_->blockSize(),
                                  executor_->gridSize());
    copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);
#else
    CHECK(false);
#endif
  }
  return err;
}

void JoinHashTable::initHashTableOnCpuFromCache(
    const ChunkKey& chunk_key,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols) {
  JoinHashTableCacheKey cache_key{col_range_, *cols.first, *cols.second, num_elements, chunk_key};
  std::lock_guard<std::mutex> join_hash_table_cache_lock(join_hash_table_cache_mutex_);
  for (const auto& kv : join_hash_table_cache_) {
    if (kv.first == cache_key) {
      cpu_hash_table_buff_ = kv.second;
      break;
    }
  }
}

void JoinHashTable::putHashTableOnCpuToCache(
    const ChunkKey& chunk_key,
    const size_t num_elements,
    const std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*>& cols) {
  JoinHashTableCacheKey cache_key{col_range_, *cols.first, *cols.second, num_elements, chunk_key};
  std::lock_guard<std::mutex> join_hash_table_cache_lock(join_hash_table_cache_mutex_);
  for (const auto& kv : join_hash_table_cache_) {
    if (kv.first == cache_key) {
      return;
    }
  }
  join_hash_table_cache_.emplace_back(cache_key, cpu_hash_table_buff_);
}

llvm::Value* JoinHashTable::codegenSlot(const bool hoist_literals) noexcept {
  CHECK(executor_->plan_state_->join_info_.join_impl_type_ == Executor::JoinImplType::HashOneToOne);
  const auto cols = get_cols(qual_bin_oper_, cat_, executor_->temporary_tables_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  const auto key_lvs = executor_->codegen(key_col, true, hoist_literals);
  CHECK_EQ(size_t(1), key_lvs.size());
  CHECK(executor_->plan_state_->join_info_.join_hash_table_);
  auto hash_ptr = get_arg_by_name(executor_->cgen_state_->row_func_, "join_hash_table");
  std::vector<llvm::Value*> hash_join_idx_args{hash_ptr,
                                               executor_->castToTypeIn(key_lvs.front(), 64),
                                               executor_->ll_int(col_range_.getIntMin()),
                                               executor_->ll_int(col_range_.getIntMax())};
  if (col_range_.hasNulls()) {
    hash_join_idx_args.push_back(executor_->ll_int(inline_fixed_encoding_null_val(key_col->get_type_info())));
  }
  std::string fname{"hash_join_idx"};
  if (col_range_.hasNulls()) {
    fname += "_nullable";
  }
  const auto slot_lv = executor_->cgen_state_->emitCall(fname, hash_join_idx_args);
  const auto it_ok = executor_->cgen_state_->scan_idx_to_hash_pos_.emplace(val_col->get_rte_idx(), slot_lv);
  CHECK(it_ok.second);
  const auto slot_valid_lv =
      executor_->cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_SGE, slot_lv, executor_->ll_int(int64_t(0)));
  return slot_valid_lv;
}

const InputTableInfo& JoinHashTable::getInnerQueryInfo(const Analyzer::ColumnVar* inner_col) {
  ssize_t ti_idx = -1;
  for (size_t i = 0; i < query_infos_.size(); ++i) {
    if (inner_col->get_table_id() == query_infos_[i].table_id) {
      ti_idx = i;
      break;
    }
  }
  CHECK_NE(ssize_t(-1), ti_idx);
  return query_infos_[ti_idx];
}
