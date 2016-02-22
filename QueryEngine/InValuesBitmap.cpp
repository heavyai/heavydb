#include "Execute.h"
#include "InValuesBitmap.h"
#ifdef HAVE_CUDA
#include "GpuMemUtils.h"
#endif  // HAVE_CUDA
#include "GroupByAndAggregate.h"
#include "RuntimeFunctions.h"
#include "../Shared/checked_alloc.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <glog/logging.h>
#include <limits>

typedef boost::multiprecision::number<
    boost::multiprecision::
        cpp_int_backend<64, 64, boost::multiprecision::signed_magnitude, boost::multiprecision::checked, void>>
    checked_int64_t;

InValuesBitmap::InValuesBitmap(const std::vector<int64_t>& values,
                               const int64_t null_val,
                               const Data_Namespace::MemoryLevel memory_level,
                               const int device_id,
                               Data_Namespace::DataMgr* data_mgr)
    : bitset_(nullptr), has_nulls_(false), memory_level_(memory_level) {
#ifdef HAVE_CUDA
  CHECK(memory_level_ == Data_Namespace::CPU_LEVEL || memory_level == Data_Namespace::GPU_LEVEL);
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, memory_level_);
#endif  // HAVE_CUDA
  CHECK(!values.empty());
  min_val_ = std::numeric_limits<int64_t>::max();
  max_val_ = std::numeric_limits<int64_t>::min();
  for (const auto value : values) {
    if (value == null_val) {
      has_nulls_ = true;
      continue;
    }
    if (value < min_val_) {
      min_val_ = value;
    }
    if (value > max_val_) {
      max_val_ = value;
    }
  }
  if (max_val_ < min_val_) {
    CHECK_EQ(std::numeric_limits<int64_t>::max(), min_val_);
    CHECK_EQ(std::numeric_limits<int64_t>::min(), max_val_);
    return;
  }
  const int64_t MAX_BITMAP_BITS{8 * 1000 * 1000 * 1000L};
  const auto bitmap_sz_bits = static_cast<int64_t>(checked_int64_t(max_val_) - min_val_ + 1);
  if (bitmap_sz_bits > MAX_BITMAP_BITS) {
    throw FailedToCreateBitmap();
  }
  const auto bitmap_sz_bytes = bitmap_size_bytes(bitmap_sz_bits);
  bitset_ = static_cast<int8_t*>(checked_calloc(bitmap_sz_bytes, 1));
  for (const auto value : values) {
    if (value == null_val) {
      continue;
    }
    agg_count_distinct_bitmap(reinterpret_cast<int64_t*>(&bitset_), value, min_val_);
  }
#ifdef HAVE_CUDA
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    auto gpu_bitset = alloc_gpu_mem(data_mgr, bitmap_sz_bytes, device_id, nullptr);
    copy_to_gpu(data_mgr, gpu_bitset, bitset_, bitmap_sz_bytes, device_id);
    free(bitset_);
    bitset_ = reinterpret_cast<int8_t*>(gpu_bitset);
  }
#endif  // HAVE_CUDA
}

InValuesBitmap::~InValuesBitmap() {
  if (memory_level_ == Data_Namespace::CPU_LEVEL) {
    free(bitset_);
  }
}

llvm::Value* InValuesBitmap::codegen(llvm::Value* needle, Executor* executor, const bool hoist_literals) {
  const auto needle_i64 = executor->toDoublePrecision(needle);
  return executor->cgen_state_->emitCall("bit_is_set",
                                         {executor->ll_int(reinterpret_cast<int64_t>(bitset_)),
                                          needle_i64,
                                          executor->ll_int(min_val_),
                                          executor->ll_int(max_val_)});
}
