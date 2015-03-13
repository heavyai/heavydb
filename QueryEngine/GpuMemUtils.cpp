#include "GpuMemUtils.h"

#include "../CudaMgr/CudaMgr.h"
#include "GroupByAndAggregate.h"

#include <glog/logging.h>


CUdeviceptr alloc_gpu_mem(
    Data_Namespace::DataMgr* data_mgr,
    const size_t num_byes,
    const int device_id) {
  auto ab = data_mgr->alloc(Data_Namespace::GPU_LEVEL, device_id, num_byes);
  CHECK_EQ(ab->getPinCount(), 1);
  return reinterpret_cast<CUdeviceptr>(ab->getMemoryPtr());
}

void copy_to_gpu(
    Data_Namespace::DataMgr* data_mgr,
    CUdeviceptr dst,
    const void* src,
    const size_t num_byes,
    const int device_id) {
  CHECK(data_mgr->cudaMgr_);
  data_mgr->cudaMgr_->copyHostToDevice(
    reinterpret_cast<int8_t*>(dst), static_cast<const int8_t*>(src),
    num_byes, device_id);
}

namespace {

std::pair<CUdeviceptr, std::vector<CUdeviceptr>> create_dev_group_by_buffers(
    Data_Namespace::DataMgr* data_mgr,
    const std::vector<int64_t*>& group_by_buffers,
    const QueryMemoryDescriptor& query_mem_desc,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    const bool small_buffers) {
  if (group_by_buffers.empty()) {
    return std::make_pair(0, std::vector<CUdeviceptr> {});
  }
  size_t buffer_size {
    small_buffers ? query_mem_desc.getSmallBufferSize() : query_mem_desc.getBufferSize() };
  CHECK_GT(buffer_size, 0);
  std::vector<CUdeviceptr> group_by_dev_buffers;
  const size_t num_buffers { block_size_x * grid_size_x };
  for (size_t i = 0; i < num_buffers; ++i) {
    if (!query_mem_desc.threadsShareMemory() || (i % block_size_x == 0)) {
      auto group_by_dev_buffer = alloc_gpu_mem(
        data_mgr, buffer_size, device_id);
      copy_to_gpu(data_mgr, group_by_dev_buffer, group_by_buffers[i],
        buffer_size, device_id);
      for (size_t j = 0; j < (query_mem_desc.threadsShareMemory() ? block_size_x : 1); ++j) {
        group_by_dev_buffers.push_back(group_by_dev_buffer);
      }
    }
  }
  auto group_by_dev_ptr = alloc_gpu_mem(
    data_mgr, num_buffers * sizeof(CUdeviceptr), device_id);
  copy_to_gpu(data_mgr, group_by_dev_ptr, &group_by_dev_buffers[0],
    num_buffers * sizeof(CUdeviceptr), device_id);
  return std::make_pair(group_by_dev_ptr, group_by_dev_buffers);
}

}  // namespace

GpuQueryMemory create_dev_group_by_buffers(
    Data_Namespace::DataMgr* data_mgr,
    const std::vector<int64_t*>& group_by_buffers,
    const QueryMemoryDescriptor& query_mem_desc,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id) {
  auto dev_group_by_buffers = create_dev_group_by_buffers(
    data_mgr, group_by_buffers, query_mem_desc, block_size_x, grid_size_x, device_id, false);
  if (query_mem_desc.getSmallBufferSize()) {
    auto small_dev_group_by_buffers = create_dev_group_by_buffers(
      data_mgr, group_by_buffers, query_mem_desc, block_size_x, grid_size_x, device_id, true);
    return { dev_group_by_buffers, small_dev_group_by_buffers };
  }
  return GpuQueryMemory { dev_group_by_buffers };
}

void copy_from_gpu(
    Data_Namespace::DataMgr* data_mgr,
    void* dst,
    const CUdeviceptr src,
    const size_t num_byes,
    const int device_id) {
  CHECK(data_mgr->cudaMgr_);
  data_mgr->cudaMgr_->copyDeviceToHost(
    static_cast<int8_t*>(dst), reinterpret_cast<const int8_t*>(src),
    num_byes, device_id);
}

namespace {

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    const std::vector<int64_t*>& group_by_buffers,
                                    const size_t groups_buffer_size,
                                    const std::vector<CUdeviceptr>& group_by_dev_buffers,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id) {
  if (group_by_buffers.empty()) {
    return;
  }
  const size_t num_buffers { block_size_x * grid_size_x };
  for (size_t i = 0; i < num_buffers; ++i) {
    if (!query_mem_desc.threadsShareMemory() || (i % block_size_x == 0)) {
      copy_from_gpu(data_mgr, group_by_buffers[i], group_by_dev_buffers[i],
        groups_buffer_size, device_id);
    }
  }
}

}  // namespace

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    const QueryExecutionContext* query_exe_context,
                                    const GpuQueryMemory& gpu_query_mem,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id) {
  copy_group_by_buffers_from_gpu(
    data_mgr,
    query_exe_context->group_by_buffers_,
    query_exe_context->query_mem_desc_.getBufferSize(),
    gpu_query_mem.group_by_buffers.second,
    query_exe_context->query_mem_desc_,
    block_size_x, grid_size_x, device_id);
  if (query_exe_context->query_mem_desc_.getSmallBufferSize()) {
    CHECK(!query_exe_context->small_group_by_buffers_.empty());
    copy_group_by_buffers_from_gpu(
      data_mgr,
      query_exe_context->small_group_by_buffers_, query_exe_context->query_mem_desc_.getSmallBufferSize(),
      gpu_query_mem.small_group_by_buffers.second,
      query_exe_context->query_mem_desc_,
      block_size_x, grid_size_x, device_id);
  }
}
