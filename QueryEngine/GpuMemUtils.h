#ifndef QUERYENGINE_GPUMEMUTILS_H
#define QUERYENGINE_GPUMEMUTILS_H

#include "../DataMgr/DataMgr.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>


CUdeviceptr alloc_gpu_mem(
  Data_Namespace::DataMgr* data_mgr,
  const size_t num_byes,
  const int device_id);

void copy_to_gpu(
    Data_Namespace::DataMgr* data_mgr,
    CUdeviceptr dst,
    const void* src,
    const size_t num_byes,
    const int device_id);

std::pair<CUdeviceptr, std::vector<CUdeviceptr>> create_dev_group_by_buffers(
    Data_Namespace::DataMgr* data_mgr,
    const std::vector<int64_t*>& group_by_buffers,
    const size_t groups_buffer_size,
    const bool fast_group_by,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id);

void copy_from_gpu(
    Data_Namespace::DataMgr* data_mgr,
    void* dst,
    const CUdeviceptr src,
    const size_t num_byes,
    const int device_id);

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    std::vector<int64_t*> group_by_buffers,
                                    const size_t groups_buffer_size,
                                    const std::vector<CUdeviceptr>& group_by_dev_buffers,
                                    const bool fast_group_by,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id);

#endif // QUERYENGINE_GPUMEMUTILS_H
