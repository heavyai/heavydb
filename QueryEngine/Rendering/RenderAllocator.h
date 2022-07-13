/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include "Shared/nocuda.h"
#endif  // HAVE_CUDA

#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "DataMgr/Allocators/DeviceAllocator.h"

namespace QueryRenderer {
class JSONLocation;
class QueryRenderManager;
struct QueryDataLayout;
}  // namespace QueryRenderer

class OutOfRenderMemory : public std::runtime_error {
 public:
  OutOfRenderMemory(const size_t device_id,
                    const size_t total_bytes,
                    const size_t requested_bytes);
};

class StreamingTopNNotSupportedInRenderQuery : public std::runtime_error {
 public:
  StreamingTopNNotSupportedInRenderQuery()
      : std::runtime_error("Streaming-Top-N not supported in Render Query") {}
};

class RenderAllocator : public Allocator {
 public:
  RenderAllocator(int8_t* preallocated_ptr,
                  const size_t preallocated_size,
                  const size_t device_id);

  int8_t* alloc(const size_t bytes) final;

  void markChunkComplete();

  size_t getCurrentChunkOffset() const;
  size_t getCurrentChunkSize() const;
  size_t getAllocatedSize() const;
  size_t getPeakAllocatedSize() const;

  int8_t* getBasePtr() const;

 private:
  int8_t* preallocated_ptr_;
  const size_t preallocated_size_;
  const size_t device_id_;
  size_t crt_chunk_offset_bytes_;
  size_t crt_allocated_bytes_;
  size_t peak_allocated_bytes_;

  std::unique_ptr<std::mutex> alloc_mtx_ptr_;
};

class RenderAllocatorMap {
 public:
  RenderAllocatorMap(::QueryRenderer::QueryRenderManager* render_manager);
  ~RenderAllocatorMap();

  RenderAllocator* getRenderAllocator(size_t device_id);
  RenderAllocator* operator[](size_t device_id);
  size_t size() const { return render_allocator_map_.size(); }

  void bufferData(int8_t* data, const size_t num_data_bytes, const size_t device_id);
  void setDataLayout(
      const std::shared_ptr<::QueryRenderer::QueryDataLayout>& query_data_layout);
  void prepForRendering(
      const std::shared_ptr<::QueryRenderer::QueryDataLayout>& query_data_layout);

 private:
  ::QueryRenderer::QueryRenderManager* render_manager_;
  std::vector<RenderAllocator> render_allocator_map_;
};
