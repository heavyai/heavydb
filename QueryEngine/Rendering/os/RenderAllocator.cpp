/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../RenderAllocator.h"
#include "../../GpuInitGroups.h"
#include "Logger/Logger.h"

RenderAllocator::RenderAllocator(int8_t* preallocated_ptr,
                                 const size_t preallocated_size,
                                 const size_t device_id)
    : preallocated_size_{preallocated_size}, device_id_{device_id} {
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

RenderAllocatorMap::RenderAllocatorMap(
    ::QueryRenderer::QueryRenderManager* render_manager) {
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

void RenderAllocatorMap::bufferData(int8_t* data,
                                    const size_t num_data_bytes,
                                    const size_t device_id) {
  CHECK(false);
}

void RenderAllocatorMap::setDataLayout(
    const std::shared_ptr<::QueryRenderer::QueryDataLayout>& query_data_layout) {
  CHECK(false);
}

void RenderAllocatorMap::prepForRendering(
    const std::shared_ptr<::QueryRenderer::QueryDataLayout>& query_data_layout) {
  CHECK(false);
}
