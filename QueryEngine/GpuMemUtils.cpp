#include "GpuMemUtils.h"

#include "../CudaMgr/CudaMgr.h"

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

std::pair<CUdeviceptr, std::vector<CUdeviceptr>> create_dev_group_by_buffers(
    Data_Namespace::DataMgr* data_mgr,
    const std::vector<int64_t*>& group_by_buffers,
    const size_t groups_buffer_size,
    const bool fast_group_by,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id) {
  if (group_by_buffers.empty()) {
    return std::make_pair(0, std::vector<CUdeviceptr> {});
  }
  std::vector<CUdeviceptr> group_by_dev_buffers;
  const size_t num_buffers { block_size_x * grid_size_x };
  for (size_t i = 0; i < num_buffers; ++i) {
    if (!fast_group_by || (i % block_size_x == 0)) {
      auto group_by_dev_buffer = alloc_gpu_mem(
        data_mgr, groups_buffer_size, device_id);
      copy_to_gpu(data_mgr, group_by_dev_buffer, group_by_buffers[i],
        groups_buffer_size, device_id);
      for (size_t j = 0; j < (fast_group_by ? block_size_x : 1); ++j) {
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

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    std::vector<int64_t*> group_by_buffers,
                                    const size_t groups_buffer_size,
                                    const std::vector<CUdeviceptr>& group_by_dev_buffers,
                                    const bool fast_group_by,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id) {
  if (group_by_buffers.empty()) {
    return;
  }
  const size_t num_buffers { block_size_x * grid_size_x };
  for (size_t i = 0; i < num_buffers; ++i) {
    if (!fast_group_by || (i % block_size_x == 0)) {
      copy_from_gpu(data_mgr, group_by_buffers[i], group_by_dev_buffers[i],
        groups_buffer_size, device_id);
    }
  }
}
