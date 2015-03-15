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

void copy_from_gpu(
    Data_Namespace::DataMgr* data_mgr,
    void* dst,
    const CUdeviceptr src,
    const size_t num_byes,
    const int device_id);

struct GpuQueryMemory {
  std::pair<CUdeviceptr, std::vector<CUdeviceptr>> group_by_buffers;
  std::pair<CUdeviceptr, std::vector<CUdeviceptr>> small_group_by_buffers;
};

struct QueryMemoryDescriptor;

GpuQueryMemory create_dev_group_by_buffers(
    Data_Namespace::DataMgr* data_mgr,
    const std::vector<int64_t*>& group_by_buffers,
    const QueryMemoryDescriptor&,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id);

class QueryExecutionContext;

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    const QueryExecutionContext*,
                                    const GpuQueryMemory&,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id);

enum class ExecutorDeviceType {
  CPU,
  GPU,
  Auto
};

class QueryMemoryDescriptor;

// TODO(alex): remove
bool buffer_not_null(const QueryMemoryDescriptor& query_mem_desc,
                     const unsigned block_size_x,
                     const ExecutorDeviceType device_type,
                     size_t i);

#endif // QUERYENGINE_GPUMEMUTILS_H
