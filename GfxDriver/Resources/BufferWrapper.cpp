/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/BufferWrapper.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Resources/Buffer.h"
#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/ResourceManager.h"

namespace gfx {

BufferWrapper::BufferWrapper(std::string_view resource_tracking_string,
                             BufferAllocatorShPtr buffer_allocator,
                             const BufferCreateInfo& create_info,
                             std::optional<LoggingCallback> oom_logging_cb)
    : resource_tracking_string_{resource_tracking_string}
    , buffer_allocator_{buffer_allocator}
    , buffer_type_{create_info.buffer_type}
    , buffer_usage_{create_info.usage} {
  RUNTIME_EX_ASSERT(create_info.size > 0u,
                    "Error constructing BufferWrapper: size must be > 0");
  if (any_bits_set(buffer_usage_ & BufferUsageBits::kLayoutBufferBit)) {
    layout_mgr_ = std::make_unique<BufferLayoutManager>(*this);
  }
  allocate(create_info.size, oom_logging_cb);
}

BufferWrapper::~BufferWrapper() {
  free();
}

const DeviceContext& BufferWrapper::getDeviceContext() const {
  CHECK(buffer_allocator_);
  return buffer_allocator_->getDeviceContext();
}

Buffer& BufferWrapper::getBuffer() const {
  CHECK(buffer_allocation_);
  return buffer_allocation_->buffer;
}

BufferType BufferWrapper::getType() const {
  return buffer_type_;
}

ResourceType BufferWrapper::getResourceType() const {
  return resource_type_from_buffer_type(buffer_type_);
}

ResourceHandle BufferWrapper::getResourceHandle() const {
  CHECK(buffer_allocation_);
  return buffer_allocation_->buffer.getResourceHandle();
}

BufferAccessType BufferWrapper::getAccessType() const {
  CHECK(buffer_allocation_);
  return buffer_allocation_->buffer.getAccessType();
}

BufferUsageBits BufferWrapper::getUsageBits() const {
  return buffer_usage_;
}

bool BufferWrapper::isMappable() const {
  CHECK(buffer_allocation_);
  return buffer_allocation_->buffer.isMappable();
}

const ResourceTrackingData& BufferWrapper::getTrackingData() {
  CHECK(buffer_allocation_);
  buffer_tracking_data_ = buffer_allocation_->buffer.getTrackingData();
  if (resource_tracking_string_ != buffer_tracking_data_.origin) {
    buffer_tracking_data_.origin = resource_tracking_string_ + " (allocated in '" +
                                   buffer_tracking_data_.origin + "', Offset " +
                                   std::to_string(buffer_allocation_->offset_bytes) + ")";
  }
  return buffer_tracking_data_;
}

const std::string& BufferWrapper::getTrackingDataNameOnly() const {
  CHECK(buffer_allocation_);
  return buffer_allocation_->buffer.getTrackingData().origin;
}

uint64_t BufferWrapper::getNumBytes() const {
  CHECK(buffer_allocation_);
  return buffer_allocation_->num_bytes;
}

uint64_t BufferWrapper::getAllocationBasePtr() const {
  CHECK(buffer_allocation_);
  return buffer_allocation_->base_ptr;
}

uint64_t BufferWrapper::getAllocationOffsetBytes() const {
  CHECK(buffer_allocation_);
  return buffer_allocation_->offset_bytes;
}

void BufferWrapper::validateUsability(const char* filename, int lineno) {
  CHECK(buffer_allocation_);
  buffer_allocation_->buffer.validateUsability(filename, lineno);
}

bool BufferWrapper::isUsable() const {
  if (buffer_allocation_) {
    return buffer_allocation_->buffer.isUsable();
  }
  return false;
}

void BufferWrapper::rebuild(const void* data,
                            uint64_t num_bytes,
                            std::optional<LoggingCallback> oom_logging_cb) {
  if (buffer_allocation_) {
    free();
  }
  allocate(num_bytes, oom_logging_cb);
  if (data) {
    CHECK(buffer_allocation_);
    CHECK_LE(num_bytes, buffer_allocation_->num_bytes);
    buffer_allocation_->buffer.updateSubData(
        data, num_bytes, buffer_allocation_->offset_bytes);
  }
}

void BufferWrapper::create(const void* data,
                           uint64_t num_bytes,
                           std::optional<LoggingCallback> oom_logging_cb) {
  allocate(num_bytes, oom_logging_cb);
  if (data) {
    CHECK(buffer_allocation_);
    CHECK_LE(num_bytes, buffer_allocation_->num_bytes);
    buffer_allocation_->buffer.updateSubData(
        data, num_bytes, buffer_allocation_->offset_bytes);
  }
}

void BufferWrapper::updateSubData(const void* data,
                                  uint64_t num_bytes,
                                  uint64_t byte_offset) {
  CHECK(data);
  CHECK(buffer_allocation_);
  CHECK_LE(num_bytes + byte_offset, buffer_allocation_->num_bytes);
  buffer_allocation_->buffer.updateSubData(
      data, num_bytes, buffer_allocation_->offset_bytes + byte_offset);
}

void BufferWrapper::getData(void* data, const uint64_t num_bytes) const {
  CHECK(data);
  CHECK(buffer_allocation_);
  CHECK_LE(num_bytes, buffer_allocation_->num_bytes);
  buffer_allocation_->buffer.getData(data, num_bytes, buffer_allocation_->offset_bytes);
}

DeviceAddress BufferWrapper::getDeviceAddress() const {
  CHECK(buffer_allocation_);
  return buffer_allocation_->buffer.getDeviceAddress() + buffer_allocation_->offset_bytes;
}

//
// Layout
//
bool BufferWrapper::hasLayout() const {
  return layout_mgr_ != nullptr;
}

BufferLayoutManager* BufferWrapper::getLayoutManager() const {
  return layout_mgr_.get();
}

void BufferWrapper::updateSubDataWithLayout(const void* data,
                                            const uint64_t num_bytes,
                                            const uint64_t offset_bytes,
                                            const BufferLayoutShPtr& layout) {
  CHECK(layout_mgr_) << "Buffer must be created with BufferUsageBits::kLayoutBuffer";
  CHECK(buffer_allocation_);
  layout_mgr_->validateBufferLayout(num_bytes, offset_bytes, layout);
  buffer_allocation_->buffer.updateSubData(data, num_bytes, offset_bytes);
  if (layout) {
    layout_mgr_->updateBufferLayouts(layout, num_bytes, offset_bytes);
  } else {
    layout_mgr_->deleteAllBufferLayouts();
  }
}

void BufferWrapper::allocate(const uint64_t num_bytes,
                             std::optional<LoggingCallback> oom_logging_cb) {
  // must not be a current allocation
  CHECK(buffer_allocator_);
  CHECK(!buffer_allocation_);

  // do the allocation
  buffer_allocation_ = buffer_allocator_->alloc(num_bytes);

  // report if failed
  // cannot log buffer name as tracking data does not exist
  if (!buffer_allocation_) {
    auto const& device = getDeviceContext();
    std::stringstream log_ss;
    log_ss << "Failed to allocate buffer of " << num_bytes << " bytes on GPU "
           << device.getGpuId() << "\n";
    device.logMemoryBudgetInfo(log_ss);
    log_ss << "\n";
    device.getResourceManager().logMemorySummary(log_ss);
    log_ss << "\n";
    if (oom_logging_cb) {
      (*oom_logging_cb)(log_ss);
    }
    LOG(ERROR) << log_ss.str();
  }
}

void BufferWrapper::free() {
  // there must be an allocator
  CHECK(buffer_allocator_);

  // nothing to do if there was no allocation
  // perhaps if shutting down the renderer after an OOM
  // cannot log buffer name as tracking data does not
  // exist, so just return
  if (!buffer_allocation_) {
    return;
  }

  // do the free
  buffer_allocator_->free(std::move(buffer_allocation_));
  CHECK(!buffer_allocation_);
}

}  // namespace gfx
