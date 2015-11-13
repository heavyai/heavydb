#include "JoinHashTable.h"
#include "Execute.h"
#include "HashJoinRuntime.h"
#include "RuntimeFunctions.h"

#include "../Chunk/Chunk.h"

#include <glog/logging.h>

namespace {

std::pair<const Analyzer::ColumnVar*, const Analyzer::ColumnVar*> get_cols(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper) {
  const auto lhs = qual_bin_oper->get_left_operand();
  const auto rhs = qual_bin_oper->get_right_operand();
  if (lhs->get_type_info() != rhs->get_type_info()) {
    return {nullptr, nullptr};
  }
  const auto lhs_cast = dynamic_cast<const Analyzer::UOper*>(lhs);
  const auto rhs_cast = dynamic_cast<const Analyzer::UOper*>(rhs);
  if (static_cast<bool>(lhs_cast) != static_cast<bool>(rhs_cast) || (lhs_cast && lhs_cast->get_optype() != kCAST) ||
      (rhs_cast && rhs_cast->get_optype() != kCAST)) {
    return {nullptr, nullptr};
  }
  const auto lhs_col = lhs_cast ? dynamic_cast<const Analyzer::ColumnVar*>(lhs_cast->get_operand())
                                : dynamic_cast<const Analyzer::ColumnVar*>(lhs);
  const auto rhs_col = rhs_cast ? dynamic_cast<const Analyzer::ColumnVar*>(rhs_cast->get_operand())
                                : dynamic_cast<const Analyzer::ColumnVar*>(rhs);
  if (!lhs_col || !rhs_col) {
    return {nullptr, nullptr};
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
  const auto& ti = inner_col->get_type_info();
  if (!(ti.is_integer() || (ti.is_string() && ti.get_compression() == kENCODING_DICT))) {
    return {nullptr, nullptr};
  }
  return {inner_col, outer_col};
}

}  // namespace

std::shared_ptr<JoinHashTable> JoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> qual_bin_oper,
    const Catalog_Namespace::Catalog& cat,
    const std::vector<Fragmenter_Namespace::QueryInfo>& query_infos,
    const Data_Namespace::MemoryLevel memory_level,
    const Executor* executor) {
  return nullptr;
  CHECK_EQ(kEQ, qual_bin_oper->get_optype());
  const auto cols = get_cols(qual_bin_oper);
  const auto inner_col = cols.first;
  if (!inner_col) {
    return nullptr;
  }
  const auto& ti = inner_col->get_type_info();
  const auto col_range = getExpressionRange(ti.is_string() ? cols.second : inner_col, query_infos, nullptr);
  if (col_range.has_nulls) {  // TODO(alex): lift this constraint
    return nullptr;
  }
  auto join_hash_table = std::shared_ptr<JoinHashTable>(
      new JoinHashTable(qual_bin_oper, inner_col, cat, query_infos, memory_level, col_range, executor));
  const int err = join_hash_table->reify();
  if (err) {
    return nullptr;
  }
  return join_hash_table;
}

int JoinHashTable::reify() {
  const auto cols = get_cols(qual_bin_oper_);
  const auto inner_col = cols.first;
  CHECK(inner_col);
  int err = 0;
  const auto& query_info = query_infos_[inner_col->get_rte_idx()];
  if (query_info.fragments.size() != 1) {  // we don't support multiple fragment inner tables (yet)
    return -1;
  }
  const auto& fragment = query_info.fragments.front();
  auto chunk_meta_it = fragment.chunkMetadataMap.find(inner_col->get_column_id());
  CHECK(chunk_meta_it != fragment.chunkMetadataMap.end());
  ChunkKey chunk_key{
      cat_.get_currentDB().dbId, inner_col->get_table_id(), inner_col->get_column_id(), fragment.fragmentId};
  const auto cd = cat_.getMetadataForColumn(inner_col->get_table_id(), inner_col->get_column_id());
  CHECK(!(cd->isVirtualCol));
  const auto& ti = inner_col->get_type_info();
  // Since we don't have the string dictionary payloads on the GPU, we'll build
  // the join hash table on the CPU and transfer it to the GPU.
  const auto effective_memory_level = ti.is_string() ? Data_Namespace::CPU_LEVEL : memory_level_;
  const int device_id = fragment.deviceIds[static_cast<int>(effective_memory_level)];
  const auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                               &cat_.get_dataMgr(),
                                               chunk_key,
                                               effective_memory_level,
                                               effective_memory_level == Data_Namespace::CPU_LEVEL ? 0 : device_id,
                                               chunk_meta_it->second.numBytes,
                                               chunk_meta_it->second.numElements);
  CHECK(chunk);
  auto ab = chunk->get_buffer();
  CHECK(ab->getMemoryPtr());
  const auto col_buff = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
  const int32_t groups_buffer_entry_count = col_range_.int_max - col_range_.int_min + 1;
#ifdef HAVE_CUDA
  // Even if we join on dictionary encoded strings, the memory on the GPU is still needed
  // once the join hash table has been built on the CPU.
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    auto& data_mgr = cat_.get_dataMgr();
    gpu_hash_table_buff_ = alloc_gpu_mem(&data_mgr, 2 * groups_buffer_entry_count * sizeof(int64_t), device_id);
  }
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    cpu_hash_table_buff_.resize(2 * groups_buffer_entry_count);
    const StringDictionary* sd_inner{nullptr};
    const StringDictionary* sd_outer{nullptr};
    if (ti.is_string()) {
      CHECK_EQ(kENCODING_DICT, ti.get_compression());
      sd_inner = executor_->getStringDictionary(inner_col->get_comp_param(), executor_->row_set_mem_owner_);
      CHECK(sd_inner);
      sd_outer = executor_->getStringDictionary(cols.second->get_comp_param(), executor_->row_set_mem_owner_);
      CHECK(sd_outer);
    }
    err = init_hash_join_buff(&cpu_hash_table_buff_[0],
                              groups_buffer_entry_count,
                              col_buff,
                              chunk_meta_it->second.numElements,
                              inner_col->get_type_info().get_size(),
                              col_range_.int_min,
                              sd_inner,
                              sd_outer);
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    // Don't transfer the buffer if there was an error since we'll bail anyway.
    if (memory_level_ == Data_Namespace::GPU_LEVEL && !err) {
#ifdef HAVE_CUDA
      CHECK(ti.is_string());
      auto& data_mgr = cat_.get_dataMgr();
      copy_to_gpu(&data_mgr,
                  gpu_hash_table_buff_,
                  &cpu_hash_table_buff_[0],
                  cpu_hash_table_buff_.size() * sizeof(cpu_hash_table_buff_[0]),
                  device_id);
#else
      CHECK(false);
#endif
    }
  } else {
#ifdef HAVE_CUDA
    CHECK_EQ(Data_Namespace::GPU_LEVEL, effective_memory_level);
    auto& data_mgr = cat_.get_dataMgr();
    auto dev_err_buff = alloc_gpu_mem(&data_mgr, sizeof(int), device_id);
    copy_to_gpu(&data_mgr, dev_err_buff, &err, sizeof(err), device_id);
    init_hash_join_buff_on_device(reinterpret_cast<int64_t*>(gpu_hash_table_buff_),
                                  reinterpret_cast<int*>(dev_err_buff),
                                  groups_buffer_entry_count,
                                  col_buff,
                                  chunk_meta_it->second.numElements,
                                  inner_col->get_type_info().get_size(),
                                  col_range_.int_min,
                                  executor_->blockSize(),
                                  executor_->gridSize());
    copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);
#else
    CHECK(false);
#endif
  }
  return err;
}

llvm::Value* JoinHashTable::codegenSlot(Executor* executor, const bool hoist_literals) {
  CHECK(executor->plan_state_->join_info_.join_impl_type_ == Executor::JoinImplType::HashOneToOne);
  const auto cols = get_cols(qual_bin_oper_);
  auto key_col = cols.second;
  CHECK(key_col);
  auto val_col = cols.first;
  CHECK(val_col);
  const auto key_lvs = executor->codegen(key_col, true, hoist_literals);
  CHECK_EQ(size_t(1), key_lvs.size());
  CHECK(executor->plan_state_->join_info_.join_hash_table_);
#ifdef HAVE_CUDA
  const int64_t join_hash_buff_ptr = memory_level_ == Data_Namespace::CPU_LEVEL
                                         ? reinterpret_cast<int64_t>(&cpu_hash_table_buff_[0])
                                         : gpu_hash_table_buff_;
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, memory_level_);
  const int64_t join_hash_buff_ptr = reinterpret_cast<int64_t>(&cpu_hash_table_buff_[0]);
#endif
  const auto i64_ty = get_int_type(64, executor->cgen_state_->context_);
  const auto hash_ptr = llvm::ConstantInt::get(i64_ty, join_hash_buff_ptr);
  // TODO(alex): maybe make the join hash table buffer a parameter (or a hoisted literal?),
  //             otoh once we fully set up the join hash table caching it won't change often
  const auto slot_lv = executor->cgen_state_->emitCall("hash_join_idx",
                                                       {hash_ptr,
                                                        executor->toDoublePrecision(key_lvs.front()),
                                                        executor->ll_int(col_range_.int_min),
                                                        executor->ll_int(col_range_.int_max)});
  const auto it_ok = executor->cgen_state_->scan_idx_to_hash_pos_.emplace(val_col->get_rte_idx(), slot_lv);
  CHECK(it_ok.second);
  const auto slot_valid_lv =
      executor->cgen_state_->ir_builder_.CreateICmp(llvm::ICmpInst::ICMP_SGE, slot_lv, executor->ll_int(int64_t(0)));
  return slot_valid_lv;
}
