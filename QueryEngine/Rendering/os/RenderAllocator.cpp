/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "../RenderAllocator.h"
#include "../../GpuInitGroups.h"
#include <glog/logging.h>

RenderAllocator::RenderAllocator(int8_t* preallocated_ptr,
                                 const size_t preallocated_size,
                                 const unsigned block_size_x,
                                 const unsigned grid_size_x,
                                 const RAExecutionPolicy execution_policy)
    : preallocated_size_(preallocated_size) {
  CHECK(false);
}

int8_t* RenderAllocator::alloc(const size_t bytes) {
  CHECK(false);
  return 0;
}

void RenderAllocator::markChunkComplete() {
  CHECK(false);
}

size_t RenderAllocator::getCurrentChunkOffset() const {
  CHECK(false);
  return 0;
}

size_t RenderAllocator::getCurrentChunkSize() const {
  CHECK(false);
  return 0;
}

size_t RenderAllocator::getAllocatedSize() const {
  CHECK(false);
  return 0;
}

int8_t* RenderAllocator::getBasePtr() const {
  CHECK(false);
  return nullptr;
}

RAExecutionPolicy RenderAllocator::getExecutionPolicy() const {
  CHECK(false);
  return RAExecutionPolicy::Host;
}

RenderAllocatorMap::RenderAllocatorMap(::QueryRenderer::QueryRenderManager* render_manager,
                                       const unsigned block_size_x,
                                       const unsigned grid_size_x) {
  CHECK(false);
}

RenderAllocatorMap::~RenderAllocatorMap() {}

RenderAllocator* RenderAllocatorMap::getRenderAllocator(size_t device_id) {
  CHECK(false);
  return nullptr;
}

RenderAllocator* RenderAllocatorMap::operator[](size_t device_id) {
  CHECK(false);
  return nullptr;
}

void RenderAllocatorMap::bufferData(int8_t* data, const size_t num_data_bytes, const size_t device_id) {
  CHECK(false);
}

void RenderAllocatorMap::setDataLayout(const std::shared_ptr<::QueryRenderer::QueryDataLayout>& query_data_layout) {
  CHECK(false);
}

void RenderAllocatorMap::prepForRendering(const std::shared_ptr<::QueryRenderer::QueryDataLayout>& query_data_layout) {
  CHECK(false);
}
