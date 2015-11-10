#include "JoinHashTable.h"
#include "Execute.h"
#include "ExpressionRange.h"
#include "HashJoinRuntime.h"
#include "RuntimeFunctions.h"

#include "../Chunk/Chunk.h"

#include <glog/logging.h>

llvm::Value* JoinHashTable::reify(llvm::Value* key_val, const Executor* executor) {
  const auto& query_info = query_infos_[col_var_->get_rte_idx()];
  CHECK_EQ(size_t(1), query_info.fragments.size());  // we don't support multiple fragment inner tables yet
  const auto& fragment = query_info.fragments.front();
  auto chunk_meta_it = fragment.chunkMetadataMap.find(col_var_->get_column_id());
  CHECK(chunk_meta_it != fragment.chunkMetadataMap.end());
  ChunkKey chunk_key{
      cat_.get_currentDB().dbId, col_var_->get_table_id(), col_var_->get_column_id(), fragment.fragmentId};
  const auto cd = cat_.getMetadataForColumn(col_var_->get_table_id(), col_var_->get_column_id());
  CHECK(!(cd->isVirtualCol));
  const int device_id = fragment.deviceIds[static_cast<int>(memory_level_)];
  const auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                               &cat_.get_dataMgr(),
                                               chunk_key,
                                               memory_level_,
                                               memory_level_ == Data_Namespace::CPU_LEVEL ? 0 : device_id,
                                               chunk_meta_it->second.numBytes,
                                               chunk_meta_it->second.numElements);
  CHECK(chunk);
  auto ab = chunk->get_buffer();
  CHECK(ab->getMemoryPtr());
  const auto col_range = getExpressionRange(col_var_, query_infos_, nullptr);
  CHECK(col_range.type == ExpressionRangeType::Integer);  // TODO
  CHECK(!col_range.has_nulls);                            // TODO
  const auto col_buff = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
  const int32_t groups_buffer_entry_count = col_range.int_max - col_range.int_min + 1;
  if (memory_level_ == Data_Namespace::CPU_LEVEL) {
    cpu_hash_table_buff_.resize(2 * groups_buffer_entry_count);
    init_hash_join_buff(&cpu_hash_table_buff_[0],
                        groups_buffer_entry_count,
                        col_buff,
                        chunk_meta_it->second.numElements,
                        col_var_->get_type_info().get_size(),
                        col_range.int_min);
  } else {
#ifdef HAVE_CUDA
    CHECK_EQ(Data_Namespace::GPU_LEVEL, memory_level_);
    auto& data_mgr = cat_.get_dataMgr();
    gpu_hash_table_buff_ = alloc_gpu_mem(&data_mgr, 2 * groups_buffer_entry_count * sizeof(int64_t), device_id);
    init_hash_join_buff_on_device(reinterpret_cast<int64_t*>(gpu_hash_table_buff_),
                                  groups_buffer_entry_count,
                                  col_buff,
                                  chunk_meta_it->second.numElements,
                                  col_var_->get_type_info().get_size(),
                                  col_range.int_min,
                                  executor->blockSize(),
                                  executor->gridSize());
#else
    CHECK(false);
#endif
  }
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
  return executor->cgen_state_->emitCall(
      "hash_join_idx", {hash_ptr, key_val, executor->ll_int(col_range.int_min), executor->ll_int(col_range.int_max)});
}
