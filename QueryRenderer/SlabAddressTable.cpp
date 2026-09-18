/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <limits>

#include "GfxDriver/DeviceContext.h"
#include "QueryRenderer/SlabAddressTable.h"

namespace QueryRenderer {

SlabAddressTable::SlabAddressTable(gfx::ResourceManager& resource_mgr)
    : resource_mgr_{resource_mgr} {
  table_buffer_ = resource_mgr_.createBuffer("SLAB_ADDRESS_TABLE",
                                             {gfx::BufferType::kUnspecified,
                                              kTableBufferSize,
                                              gfx::BufferUsageBits::kUniformBufferBit,
                                              gfx::BufferAccessType::kHostVisible});
  CHECK(table_buffer_);
}

SlabAddressTable::~SlabAddressTable() {
  destroyResources();
}

const gfx::BufferWrapper& SlabAddressTable::getTableBuffer() const {
  // return the buffer wrapper for binding to a material
  CHECK(table_buffer_);
  return *table_buffer_;
}

void SlabAddressTable::reset() {
  // debugging
  VLOG(1) << "Resetting Slab Address Table for GPU "
          << resource_mgr_.getDeviceContext().getGpuId();

  // clear the table
  table_.clear();
}

void SlabAddressTable::addEntry(const uint64_t cuda_addr, const uint64_t vulkan_addr) {
  // debugging
  VLOG(1) << "  Adding Slab: CUDA " << (void*)cuda_addr << ", Vulkan "
          << (void*)vulkan_addr;

  // append to table
  table_.push_back({cuda_addr, vulkan_addr});
}

void SlabAddressTable::finalizeAndUpdateBuffer() {
  // debugging
  VLOG(1) << "  Found " << table_.size() << " slabs";

  // append terminating entry
  static constexpr uint64_t kTerminator = std::numeric_limits<uint64_t>::max();
  table_.push_back({kTerminator, kTerminator});

  // update uniform buffer
  CHECK(table_buffer_);
  auto const num_entries = std::min(kMaxSlabs, table_.size());
  table_buffer_->updateSubData(table_.data(), num_entries * sizeof(Entry), 0);
}

void SlabAddressTable::destroyResources() {
  // destroy uniform buffer
  if (table_buffer_) {
    resource_mgr_.destroyBuffer(std::move(table_buffer_));
  }

  // clear the table
  table_.clear();
}

//
// excessive boilerplate
// just flatten this out into QBM
//

const size_t SlabAddressTable::numEntries() const {
  return table_.size();
}

const std::pair<uint64_t, uint64_t> SlabAddressTable::getEntry(const size_t index) const {
  CHECK_LT(index, table_.size());
  auto const& entry = table_[index];
  return {entry.cuda, entry.vulkan};
}

}  // namespace QueryRenderer
