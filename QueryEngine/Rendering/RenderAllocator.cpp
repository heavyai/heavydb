/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/Rendering/RenderAllocator.h"
#include "Logger/Logger.h"
#include "Shared/checked_alloc.h"

#ifdef HAVE_RENDERING
#include "CudaMgr/CudaMgr.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/QueryRenderManager.h"
#endif  // HAVE_RENDERING

OutOfRenderMemory::OutOfRenderMemory(const size_t device_id,
                                     const size_t total_bytes,
                                     const size_t requested_bytes)
    : std::runtime_error(
          "Not enough buffer memory on device id " + std::to_string(device_id) +
          " to render the query results. Render buffer size: " +
          std::to_string(total_bytes) +
          " bytes. Requested size: " + std::to_string(requested_bytes) + " bytes.") {
  LOG(ERROR) << this->what();
}

RenderAllocator::RenderAllocator(int8_t* preallocated_ptr,
                                 const size_t preallocated_size,
                                 const size_t device_id)
    : preallocated_ptr_{preallocated_ptr}
    , preallocated_size_{preallocated_size}
    , device_id_{device_id}
    , crt_chunk_offset_bytes_{0}
    , crt_allocated_bytes_{0}
    , peak_allocated_bytes_{0}
    , alloc_mtx_ptr_{new std::mutex()} {}

int8_t* RenderAllocator::alloc(const size_t bytes) {
  CHECK(alloc_mtx_ptr_);
  std::lock_guard<std::mutex> alloc_lock(*alloc_mtx_ptr_);

  if (!preallocated_ptr_) {
    CHECK_EQ(crt_allocated_bytes_, 0u);
    throw OutOfRenderMemory(device_id_, preallocated_size_, bytes);
  }

  auto ptr = preallocated_ptr_ + crt_allocated_bytes_;
  crt_allocated_bytes_ += bytes;
  peak_allocated_bytes_ = std::max(crt_allocated_bytes_, peak_allocated_bytes_);

  if (crt_allocated_bytes_ <= preallocated_size_) {
    return ptr;
  }

  // reset the current allocated bytes for a proper
  // error resolution
  const auto used_bytes = crt_allocated_bytes_;
  crt_allocated_bytes_ = 0;
  throw OutOfRenderMemory(device_id_, preallocated_size_, used_bytes);
}

void RenderAllocator::markChunkComplete() {
  crt_chunk_offset_bytes_ = crt_allocated_bytes_;
}

size_t RenderAllocator::getCurrentChunkOffset() const {
  return crt_chunk_offset_bytes_;
}

size_t RenderAllocator::getCurrentChunkSize() const {
  return crt_allocated_bytes_ - crt_chunk_offset_bytes_;
}

size_t RenderAllocator::getAllocatedSize() const {
  return crt_allocated_bytes_;
}

size_t RenderAllocator::getPeakAllocatedSize() const {
  return peak_allocated_bytes_;
}

int8_t* RenderAllocator::getBasePtr() const {
  return preallocated_ptr_;
}

RenderAllocatorMap::RenderAllocatorMap(
    ::QueryRenderer::QueryRenderManager* render_manager)
    : render_manager_{render_manager} {
  CHECK(render_manager_);
#ifdef HAVE_RENDERING
  auto num_devices = static_cast<int>(render_manager_->getNumGpus());
  for (int i = 0; i < num_devices; ++i) {
    auto qob_descriptor = render_manager_->getQueryOutputBufferDescriptor(i);
    render_allocator_map_.emplace_back(
        qob_descriptor.handle, qob_descriptor.num_bytes, i);
  }
#endif  // HAVE_RENDERING
}

namespace {
#ifdef HAVE_RENDERING
void calculate_global_peak_usage(::QueryRenderer::QueryRenderManager& render_manager,
                                 const std::vector<RenderAllocator>& render_allocators) {
  if (render_allocators.size()) {
    size_t global_peak_usage = 0u;
    for (auto const& render_allocator : render_allocators) {
      global_peak_usage =
          std::max(render_allocator.getPeakAllocatedSize(), global_peak_usage);
    }
    render_manager.setRenderBufferPeakUsage(global_peak_usage);
  }
}
#endif  // HAVE_RENDERING
}  // namespace

RenderAllocatorMap::~RenderAllocatorMap() {
#ifdef HAVE_RENDERING
  // the render_manager could've been destroyed via a clear_gpu of some kind, so verify
  // we're not in that state before proceeding to unmap the cuda handles.
  const auto num_gpus = render_manager_->getNumGpus();
  if (num_gpus > 0 && render_allocator_map_.size() > 0) {
    CHECK_EQ(num_gpus, render_allocator_map_.size());
    calculate_global_peak_usage(*render_manager_, render_allocator_map_);
    for (size_t i = 0; i < render_allocator_map_.size(); ++i) {
      render_manager_->releaseQueryOutputBufferDescriptor(i, 0, nullptr);
    }
  }
#endif  // HAVE_RENDERING
}

RenderAllocator* RenderAllocatorMap::getRenderAllocator(size_t device_id) {
  return (*this)[device_id];
}

RenderAllocator* RenderAllocatorMap::operator[](size_t device_id) {
  CHECK(device_id < render_allocator_map_.size())
      << "Device id " << device_id << " not found in RenderAllocatorMap. Only "
      << render_allocator_map_.size() << " devices available.";

  return &render_allocator_map_[device_id];
}

void RenderAllocatorMap::bufferData(int8_t* data,
                                    const size_t num_data_bytes,
                                    const size_t device_id) {
#ifdef HAVE_RENDERING
  auto render_allocator = getRenderAllocator(device_id);
#ifdef HAVE_CUDA
  render_manager_->getCudaMgr()->setContext(device_id);
  cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(render_allocator->getBasePtr() +
                                             render_allocator->getCurrentChunkOffset()),
               data,
               num_data_bytes);
#else   // !HAVE_CUDA
  std::memcpy(render_allocator->getBasePtr() + render_allocator->getCurrentChunkOffset(),
              data,
              num_data_bytes);
#endif  // HAVE_CUDA
#endif  // HAVE_RENDERING
}

void RenderAllocatorMap::setDataLayout(
    const std::shared_ptr<::QueryRenderer::QueryDataLayout>& query_data_layout) {
#ifdef HAVE_RENDERING
  for (size_t i = 0; i < render_allocator_map_.size(); ++i) {
    render_manager_->setRenderBufferDataLayout(
        i,
        render_allocator_map_[i].getCurrentChunkOffset(),
        render_allocator_map_[i].getCurrentChunkSize(),
        query_data_layout);
    render_allocator_map_[i].markChunkComplete();
  }
#endif  // HAVE_RENDERING
}

void RenderAllocatorMap::prepForRendering(
    const std::shared_ptr<::QueryRenderer::QueryDataLayout>& query_data_layout) {
#ifdef HAVE_RENDERING
  // NOTE: the query_data_layout argument is deprecated
  // TODO(croot): remove the query_data_layout when deprecation is complete
  calculate_global_peak_usage(*render_manager_, render_allocator_map_);
  for (size_t i = 0; i < render_allocator_map_.size(); ++i) {
    render_manager_->releaseQueryOutputBufferDescriptor(
        i, render_allocator_map_[i].getAllocatedSize(), query_data_layout);
  }

  // NOTE: We need to do this clear so the data layout is maintained in the query output
  // buffers. If this clear isn't performed, the RenderAllocatorMap destructor would
  // eventually be called which sets the used bytes of the query output buffers to 0 which
  // in turn deletes all the layouts attached. Note: Without cuda, we don't need to unmap
  // buffers back for rendering so the clear is all that's needed.
  render_allocator_map_.clear();
#endif  // HAVE_RENDERINGG
}
