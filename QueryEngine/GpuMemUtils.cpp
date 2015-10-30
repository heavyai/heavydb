#include "GpuMemUtils.h"

#include "../CudaMgr/CudaMgr.h"
#include "GroupByAndAggregate.h"

#include <glog/logging.h>

CUdeviceptr alloc_gpu_mem(Data_Namespace::DataMgr* data_mgr,
                          const size_t num_bytes,
                          const int device_id,
                          RenderAllocator* render_allocator) {
  if (render_allocator) {
    return render_allocator->alloc(num_bytes);
  }
  auto ab = alloc_gpu_abstract_buffer(data_mgr, num_bytes, device_id);
  return reinterpret_cast<CUdeviceptr>(ab->getMemoryPtr());
}

Data_Namespace::AbstractBuffer* alloc_gpu_abstract_buffer(Data_Namespace::DataMgr* data_mgr,
                                                          const size_t num_bytes,
                                                          const int device_id) {
  auto ab = data_mgr->alloc(Data_Namespace::GPU_LEVEL, device_id, num_bytes);
  CHECK_EQ(ab->getPinCount(), 1);
  return ab;
}

void free_gpu_abstract_buffer(Data_Namespace::DataMgr* data_mgr, Data_Namespace::AbstractBuffer* ab) {
  data_mgr->free(ab);
}

void copy_to_gpu(Data_Namespace::DataMgr* data_mgr,
                 CUdeviceptr dst,
                 const void* src,
                 const size_t num_bytes,
                 const int device_id) {
  CHECK(data_mgr->cudaMgr_);
  data_mgr->cudaMgr_->copyHostToDevice(
      reinterpret_cast<int8_t*>(dst), static_cast<const int8_t*>(src), num_bytes, device_id);
}

namespace {

size_t coalesced_size(const QueryMemoryDescriptor& query_mem_desc,
                      const size_t group_by_one_buffer_size,
                      const unsigned block_size_x,
                      const unsigned grid_size_x) {
  const size_t num_buffers{block_size_x * grid_size_x};
  return (query_mem_desc.threadsShareMemory() ? grid_size_x : num_buffers) * group_by_one_buffer_size;
}

std::pair<CUdeviceptr, CUdeviceptr> create_dev_group_by_buffers(Data_Namespace::DataMgr* data_mgr,
                                                                const std::vector<int64_t*>& group_by_buffers,
                                                                const QueryMemoryDescriptor& query_mem_desc,
                                                                const unsigned block_size_x,
                                                                const unsigned grid_size_x,
                                                                const int device_id,
                                                                const bool small_buffers,
                                                                const bool prepend_index_buffer,
                                                                const bool always_init_group_by_on_host,
                                                                RenderAllocator* render_allocator) {
  if (group_by_buffers.empty() && !render_allocator) {
    return std::make_pair(0, 0);
  }

  CHECK(!small_buffers || !prepend_index_buffer);

  size_t groups_buffer_size{small_buffers ? query_mem_desc.getSmallBufferSizeBytes()
                                          : query_mem_desc.getBufferSizeBytes(ExecutorDeviceType::GPU)};

  CHECK_GT(groups_buffer_size, 0);

  const size_t mem_size{coalesced_size(
      query_mem_desc, groups_buffer_size, block_size_x, query_mem_desc.blocksShareMemory() ? 1 : grid_size_x)};

  const size_t prepended_buff_size{prepend_index_buffer ? query_mem_desc.entry_count * sizeof(int64_t) : 0};

  CUdeviceptr group_by_dev_buffers_mem =
      alloc_gpu_mem(data_mgr, mem_size + prepended_buff_size, device_id, render_allocator) + prepended_buff_size;

  const size_t step{query_mem_desc.threadsShareMemory() ? block_size_x : 1};

  if (!render_allocator && (always_init_group_by_on_host || !query_mem_desc.lazyInitGroups(ExecutorDeviceType::GPU))) {
    std::vector<int8_t> buff_to_gpu(mem_size);
    auto buff_to_gpu_ptr = &buff_to_gpu[0];

    for (size_t i = 0; i < group_by_buffers.size(); i += step) {
      memcpy(buff_to_gpu_ptr, group_by_buffers[i], groups_buffer_size);
      buff_to_gpu_ptr += groups_buffer_size;
    }
    copy_to_gpu(data_mgr, group_by_dev_buffers_mem, &buff_to_gpu[0], buff_to_gpu.size(), device_id);
  }

  auto group_by_dev_buffer = group_by_dev_buffers_mem;

  const size_t num_ptrs{block_size_x * grid_size_x};

  std::vector<CUdeviceptr> group_by_dev_buffers(num_ptrs);

  for (size_t i = 0; i < num_ptrs; i += step) {
    for (size_t j = 0; j < step; ++j) {
      group_by_dev_buffers[i + j] = group_by_dev_buffer;
    }
    if (!query_mem_desc.blocksShareMemory()) {
      group_by_dev_buffer += groups_buffer_size;
    }
  }

  auto group_by_dev_ptr = alloc_gpu_mem(data_mgr, num_ptrs * sizeof(CUdeviceptr), device_id, nullptr);
  copy_to_gpu(data_mgr, group_by_dev_ptr, &group_by_dev_buffers[0], num_ptrs * sizeof(CUdeviceptr), device_id);

  return std::make_pair(group_by_dev_ptr, group_by_dev_buffers_mem);
}

}  // namespace

GpuQueryMemory create_dev_group_by_buffers(Data_Namespace::DataMgr* data_mgr,
                                           const std::vector<int64_t*>& group_by_buffers,
                                           const std::vector<int64_t*>& small_group_by_buffers,
                                           const QueryMemoryDescriptor& query_mem_desc,
                                           const unsigned block_size_x,
                                           const unsigned grid_size_x,
                                           const int device_id,
                                           const bool prepend_index_buffer,
                                           const bool always_init_group_by_on_host,
                                           RenderAllocator* render_allocator) {
  auto dev_group_by_buffers = create_dev_group_by_buffers(data_mgr,
                                                          group_by_buffers,
                                                          query_mem_desc,
                                                          block_size_x,
                                                          grid_size_x,
                                                          device_id,
                                                          false,
                                                          prepend_index_buffer,
                                                          always_init_group_by_on_host,
                                                          render_allocator);
  if (query_mem_desc.getSmallBufferSizeBytes()) {
    auto small_dev_group_by_buffers = create_dev_group_by_buffers(data_mgr,
                                                                  small_group_by_buffers,
                                                                  query_mem_desc,
                                                                  block_size_x,
                                                                  grid_size_x,
                                                                  device_id,
                                                                  true,
                                                                  prepend_index_buffer,
                                                                  always_init_group_by_on_host,
                                                                  render_allocator);
    return {dev_group_by_buffers, small_dev_group_by_buffers};
  }
  return GpuQueryMemory{dev_group_by_buffers};
}

void copy_from_gpu(Data_Namespace::DataMgr* data_mgr,
                   void* dst,
                   const CUdeviceptr src,
                   const size_t num_bytes,
                   const int device_id) {
  CHECK(data_mgr->cudaMgr_);
  data_mgr->cudaMgr_->copyDeviceToHost(
      static_cast<int8_t*>(dst), reinterpret_cast<const int8_t*>(src), num_bytes, device_id);
}

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    const std::vector<int64_t*>& group_by_buffers,
                                    const size_t groups_buffer_size,
                                    const CUdeviceptr group_by_dev_buffers_mem,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id,
                                    const bool prepend_index_buffer) {
  if (group_by_buffers.empty()) {
    return;
  }
  const unsigned block_buffer_count{query_mem_desc.blocksShareMemory() ? 1 : grid_size_x};
  const size_t num_buffers{block_size_x * block_buffer_count};
  const size_t index_buffer_sz{prepend_index_buffer ? query_mem_desc.entry_count * sizeof(int64_t) : 0};
  std::vector<int8_t> buff_from_gpu(
      coalesced_size(query_mem_desc, groups_buffer_size, block_size_x, block_buffer_count) + index_buffer_sz);
  copy_from_gpu(
      data_mgr, &buff_from_gpu[0], group_by_dev_buffers_mem - index_buffer_sz, buff_from_gpu.size(), device_id);
  auto buff_from_gpu_ptr = &buff_from_gpu[0];
  for (size_t i = 0; i < num_buffers; ++i) {
    if (buffer_not_null(query_mem_desc, block_size_x, ExecutorDeviceType::GPU, i)) {
      memcpy(group_by_buffers[i], buff_from_gpu_ptr, groups_buffer_size + index_buffer_sz);
      buff_from_gpu_ptr += groups_buffer_size;
    }
  }
}

void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                    const QueryExecutionContext* query_exe_context,
                                    const GpuQueryMemory& gpu_query_mem,
                                    const unsigned block_size_x,
                                    const unsigned grid_size_x,
                                    const int device_id,
                                    const bool prepend_index_buffer) {
  copy_group_by_buffers_from_gpu(data_mgr,
                                 query_exe_context->group_by_buffers_,
                                 query_exe_context->query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU),
                                 gpu_query_mem.group_by_buffers.second,
                                 query_exe_context->query_mem_desc_,
                                 block_size_x,
                                 grid_size_x,
                                 device_id,
                                 prepend_index_buffer);
  if (query_exe_context->query_mem_desc_.getSmallBufferSizeBytes()) {
    CHECK(!prepend_index_buffer);
    CHECK(!query_exe_context->small_group_by_buffers_.empty());
    copy_group_by_buffers_from_gpu(data_mgr,
                                   query_exe_context->small_group_by_buffers_,
                                   query_exe_context->query_mem_desc_.getSmallBufferSizeBytes(),
                                   gpu_query_mem.small_group_by_buffers.second,
                                   query_exe_context->query_mem_desc_,
                                   block_size_x,
                                   grid_size_x,
                                   device_id,
                                   false);
  }
}

// TODO(alex): remove
bool buffer_not_null(const QueryMemoryDescriptor& query_mem_desc,
                     const unsigned block_size_x,
                     const ExecutorDeviceType device_type,
                     size_t i) {
  if (device_type == ExecutorDeviceType::CPU) {
    return true;
  }
  return (!query_mem_desc.threadsShareMemory() || (i % block_size_x == 0));
}
