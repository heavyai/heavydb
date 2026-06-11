/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ForeignStorageBuffer.h"

namespace foreign_storage {
ForeignStorageBuffer::ForeignStorageBuffer() : AbstractBuffer(0) {}

void ForeignStorageBuffer::read(int8_t* const destination,
                                const size_t num_bytes,
                                const size_t offset,
                                const MemoryLevel destination_buffer_type,
                                const int destination_device_id) {
  memcpy(destination, buffer_.data() + offset, num_bytes);
}

void ForeignStorageBuffer::reserve(size_t total_num_bytes) {
  buffer_.reserve(total_num_bytes);
}

void ForeignStorageBuffer::append(int8_t* source,
                                  const size_t num_bytes,
                                  const MemoryLevel source_buffer_type,
                                  const int device_id) {
  buffer_.resize(size_ + num_bytes);
  std::copy(source, source + num_bytes, buffer_.begin() + size_);
  size_ += num_bytes;
}

int8_t* ForeignStorageBuffer::getMemoryPtr() {
  return buffer_.data();
}

size_t ForeignStorageBuffer::reservedSize() const {
  return buffer_.capacity();
}

MemoryLevel ForeignStorageBuffer::getType() const {
  return CPU_LEVEL;
}

void ForeignStorageBuffer::write(int8_t* source,
                                 const size_t num_bytes,
                                 const size_t offset,
                                 const MemoryLevel source_buffer_type,
                                 const int source_device_id) {
  UNREACHABLE();
}

size_t ForeignStorageBuffer::pageCount() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

size_t ForeignStorageBuffer::pageSize() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}
}  // namespace foreign_storage
