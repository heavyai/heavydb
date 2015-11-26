#ifndef QUERYENGINE_GPUMEMUTILS_H
#define QUERYENGINE_GPUMEMUTILS_H

#include "../DataMgr/DataMgr.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

class OutOfRenderMemory : public std::runtime_error {
 public:
  OutOfRenderMemory() : std::runtime_error("OutOfMemory") {}
};

class RenderAllocator {
 public:
  RenderAllocator(int8_t* preallocated_ptr,
                  const size_t preallocated_size,
                  const unsigned block_size_x,
                  const unsigned grid_size_x);

  CUdeviceptr alloc(const size_t bytes) {
    auto ptr = preallocated_ptr_ + crt_allocated_bytes_;
    crt_allocated_bytes_ += bytes;
    if (crt_allocated_bytes_ <= preallocated_size_) {
      return reinterpret_cast<CUdeviceptr>(ptr);
    }
    throw OutOfRenderMemory();
  }

  size_t getAllocatedSize() const { return crt_allocated_bytes_; }

  int8_t* getBasePtr() const { return preallocated_ptr_; }

 private:
  int8_t* preallocated_ptr_;
  const size_t preallocated_size_;
  size_t crt_allocated_bytes_;
};

CUdeviceptr alloc_gpu_mem(Data_Namespace::DataMgr* data_mgr,
                          const size_t num_bytes,
                          const int device_id,
                          RenderAllocator* render_allocator);

Data_Namespace::AbstractBuffer* alloc_gpu_abstract_buffer(Data_Namespace::DataMgr* data_mgr,
                                                          const size_t num_bytes,
                                                          const int device_id);

void free_gpu_abstract_buffer(Data_Namespace::DataMgr* data_mgr, Data_Namespace::AbstractBuffer* ab);

void copy_to_gpu(Data_Namespace::DataMgr* data_mgr,
                 CUdeviceptr dst,
                 const void* src,
                 const size_t num_bytes,
                 const int device_id);

void copy_from_gpu(Data_Namespace::DataMgr* data_mgr,
                   void* dst,
                   const CUdeviceptr src,
                   const size_t num_bytes,
                   const int device_id);

struct GpuQueryMemory {
  std::pair<CUdeviceptr, CUdeviceptr> group_by_buffers;
  std::pair<CUdeviceptr, CUdeviceptr> small_group_by_buffers;
};

struct QueryMemoryDescriptor;

GpuQueryMemory create_dev_group_by_buffers(Data_Namespace::DataMgr* data_mgr,
                                           const std::vector<int64_t*>& group_by_buffers,
                                           const std::vector<int64_t*>& small_group_by_buffers,
                                           const QueryMemoryDescriptor&,
                                           const unsigned block_size_x,
                                           const unsigned grid_size_x,
                                           const int device_id,
                                           const bool prepend_index_buffer,
                                           const bool always_init_group_by_on_host,
                                           RenderAllocator* render_allocator);

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    const std::vector<int64_t*>& group_by_buffers,
                                    const size_t groups_buffer_size,
                                    const CUdeviceptr group_by_dev_buffers_mem,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id,
                                    const bool prepend_index_buffer);

class QueryExecutionContext;

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    const QueryExecutionContext*,
                                    const GpuQueryMemory&,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id,
                                    const bool prepend_index_buffer);

// TODO(alex): remove
bool buffer_not_null(const QueryMemoryDescriptor& query_mem_desc,
                     const unsigned block_size_x,
                     const ExecutorDeviceType device_type,
                     size_t i);

#endif  // QUERYENGINE_GPUMEMUTILS_H
