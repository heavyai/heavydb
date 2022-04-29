/*
 * Copyright 2018 MapD Technologies, Inc.
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

#include "DataMgr/Allocators/GpuAllocator.h"

#include "Logger/Logger.h"
#include "Shared/types.h"

GpuAllocator::GpuAllocator(BufferProvider* buffer_provider, const int device_id)
    : buffer_provider_(buffer_provider), device_id_(device_id) {
  CHECK(buffer_provider_);
#ifdef HAVE_CUDA
  // TODO: remove as this context setting looks redundant (we pass an actual
  // context in every Allocator's method anyway)
  buffer_provider_->setContext(device_id);
#endif  // HAVE_CUDA
}

GpuAllocator::~GpuAllocator() {
  CHECK(buffer_provider_);
  for (auto& buffer_ptr : owned_buffers_) {
    buffer_provider_->free(buffer_ptr);
  }
}

Data_Namespace::AbstractBuffer* GpuAllocator::allocGpuAbstractBuffer(
    BufferProvider* buffer_provider,
    const size_t num_bytes,
    const int device_id) {
  CHECK(buffer_provider);
  auto ab = buffer_provider->alloc(Data_Namespace::GPU_LEVEL, device_id, num_bytes);
  CHECK_EQ(ab->getPinCount(), 1);
  return ab;
}

int8_t* GpuAllocator::alloc(const size_t num_bytes) {
  CHECK(buffer_provider_);
  owned_buffers_.emplace_back(
      GpuAllocator::allocGpuAbstractBuffer(buffer_provider_, num_bytes, device_id_));
  return owned_buffers_.back()->getMemoryPtr();
}

void GpuAllocator::free(Data_Namespace::AbstractBuffer* ab) const {
  buffer_provider_->free(ab);
}

void GpuAllocator::copyToDevice(int8_t* device_dst,
                                const int8_t* host_src,
                                const size_t num_bytes) const {
  buffer_provider_->copyToDevice(device_dst, host_src, num_bytes, device_id_);
}

void GpuAllocator::copyFromDevice(int8_t* host_dst,
                                  const int8_t* device_src,
                                  const size_t num_bytes) const {
  buffer_provider_->copyFromDevice(host_dst, device_src, num_bytes, device_id_);
}

void GpuAllocator::zeroDeviceMem(int8_t* device_ptr, const size_t num_bytes) const {
  buffer_provider_->zeroDeviceMem(device_ptr, num_bytes, device_id_);
}

void GpuAllocator::setDeviceMem(int8_t* device_ptr,
                                unsigned char uc,
                                const size_t num_bytes) const {
  buffer_provider_->setDeviceMem(device_ptr, uc, num_bytes, device_id_);
}
