#include "JoinHashTable.h"
#include "Execute.h"
#include "ExpressionRange.h"
#include "RuntimeFunctions.h"

#include "../Chunk/Chunk.h"

#include <glog/logging.h>

namespace {

void init_groups(int64_t* groups_buffer,
                 const int32_t groups_buffer_entry_count,
                 const int32_t key_qw_count,
                 const int64_t* init_vals) {
  int32_t groups_buffer_entry_qw_count = groups_buffer_entry_count * (key_qw_count + 1);
  for (int32_t i = 0; i < groups_buffer_entry_qw_count; ++i) {
    groups_buffer[i] =
        (i % (key_qw_count + 1) < key_qw_count) ? EMPTY_KEY : init_vals[(i - key_qw_count) % (key_qw_count + 1)];
  }
}

}  // namespace

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
  if (memory_level_ == Data_Namespace::CPU_LEVEL) {
    const int32_t groups_buffer_entry_count = col_range.int_max - col_range.int_min + 1;
    cpu_hash_table_buff_.resize(2 * groups_buffer_entry_count);
    std::vector<int64_t> init_vals(1, -1);
    init_groups(&cpu_hash_table_buff_[0], groups_buffer_entry_count, 1, &init_vals[0]);
    const auto col_buff = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
    for (size_t i = 0; i < chunk_meta_it->second.numElements; ++i) {
      auto entry_ptr =
          get_group_value_fast(&cpu_hash_table_buff_[0],
                               fixed_width_int_decode_noinline(col_buff, col_var_->get_type_info().get_size(), i),
                               col_range.int_min,
                               1);
      // TODO; must check if it's one to one
      *entry_ptr = i;
    }
  } else {
    CHECK(false);  // TODO
  }
  const auto i64_ty = get_int_type(64, executor->cgen_state_->context_);
  const auto hash_ptr = llvm::ConstantInt::get(i64_ty, reinterpret_cast<int64_t>(&cpu_hash_table_buff_[0]));
  // TODO(alex): maybe make the join hash table buffer a parameter (or a hoisted literal?),
  //             otoh once we fully set up the join hash table caching it won't change often
  return executor->cgen_state_->emitCall(
      "hash_join_idx", {hash_ptr, key_val, executor->ll_int(col_range.int_min), executor->ll_int(col_range.int_max)});
}
