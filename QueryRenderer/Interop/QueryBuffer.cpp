/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Interop/QueryBuffer.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/ShaderBlockLayout.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/QueryBufferManager.h"

namespace QueryRenderer {

//
// QueryBuffer
//

QueryBuffer::QueryBuffer(QueryBufferManager& query_buffer_mgr,
                         QueryBufferType type,
                         const uint64_t num_bytes)
    : query_buffer_mgr_{query_buffer_mgr}, type_{type} {
  create(num_bytes);
}

QueryBuffer::~QueryBuffer() {
  destroy();
}

void QueryBuffer::create(const uint64_t num_bytes) {
  // buffer create info
  gfx::BufferCreateInfo ci;
  switch (type_) {
    case QueryBufferType::kVertex:
      ci.buffer_type = gfx::BufferType::kVertexBuffer;
      // cause QueryVertexBuffer (only) to be enabled for SSBO
      // binding to enable compute-shader QOB init
      // also BDA
      ci.usage = gfx::BufferUsageBits::kStorageBufferBit |
                 gfx::BufferUsageBits::kLayoutBufferBit |
                 gfx::BufferUsageBits::kDeviceAddressBit;
      break;
    case QueryBufferType::kIndex:
      ci.buffer_type = gfx::BufferType::kIndexBuffer;
      break;
    case QueryBufferType::kStorage:
      ci.buffer_type = gfx::BufferType::kUnspecified;
      ci.usage = gfx::BufferUsageBits::kStorageBufferBit |
                 gfx::BufferUsageBits::kLayoutBufferBit;
      break;
    case QueryBufferType::kIndirectVertex:
      ci.buffer_type = gfx::BufferType::kIndirectDrawVertexBuffer;
      break;
    case QueryBufferType::kIndirectIndex:
      ci.buffer_type = gfx::BufferType::kIndirectDrawIndexBuffer;
      break;
  }
  ci.size = num_bytes;

  // create the buffer
  buffers_and_descriptor_ =
      query_buffer_mgr_.createQueryBuffer("QueryBuffer (" + to_string(type_) + ")", ci);
}

void QueryBuffer::destroy() {
  query_buffer_mgr_.destroyQueryBuffer(std::move(buffers_and_descriptor_));
}

int32_t QueryBuffer::getGpuId() const {
  return getBufferWrapper()->getDeviceContext().getGpuId();
}

QueryBufferType QueryBuffer::getType() const {
  return type_;
}

BufferMemoryDescriptor QueryBuffer::getBufferMemoryDescriptor() {
  return buffers_and_descriptor_.descriptor;
}

void QueryBuffer::unmap() {
  // @TODO remove if OK to leave the host-visible buffer mapped all the time
}

bool QueryBuffer::isMapped() const {
  // @TODO remove if OK to leave the host-visible buffer mapped all the time
  return true;
}

uint64_t QueryBuffer::getNumBytes() const {
  return getBufferWrapper()->getNumBytes();
}

void QueryBuffer::rebuild(const uint64_t num_bytes) {
  destroy();
  create(num_bytes);
}

const gfx::BufferWrapper* QueryBuffer::getBufferWrapper() const {
  if (buffers_and_descriptor_.buffer) {
    return buffers_and_descriptor_.buffer.get();
  } else {
    return &buffers_and_descriptor_.host_visible_buffer->getSourceBufferWrapper();
  }
}

gfx::BufferWrapper* QueryBuffer::getBufferWrapper() {
  if (buffers_and_descriptor_.buffer) {
    return buffers_and_descriptor_.buffer.get();
  } else {
    return &buffers_and_descriptor_.host_visible_buffer->getSourceBufferWrapper();
  }
}

//
// QueryLayoutBuffer
//

QueryDataLayoutShPtr QueryLayoutBuffer::getQueryDataLayout() const {
  return query_data_layout_ptr_;
}

uint64_t QueryLayoutBuffer::getNumBytes() const {
  return src_interop_buffer_.getNumBytes();
}

uint64_t QueryLayoutBuffer::getNumUsedBytes(const QueryDataLayout& layout) const {
  RENDER_LOG_SCOPE() << "layout:" << &layout;
  return getLayoutManager().getNumUsedBytes(layout.getBufferLayout());
}

uint64_t QueryLayoutBuffer::getLayoutOffsetBytes(const QueryDataLayout& layout) const {
  return getLayoutManager().getBufferLayoutData(layout.getBufferLayout()).second;
}

BufferMemoryDescriptor QueryLayoutBuffer::getBufferMemoryDescriptor() {
  return src_interop_buffer_.getBufferMemoryDescriptor();
}

void QueryLayoutBuffer::unmap() {
  src_interop_buffer_.unmap();
}

void QueryLayoutBuffer::rebuild(const uint64_t num_bytes) {
  src_interop_buffer_.rebuild(num_bytes);
}

void QueryLayoutBuffer::reset() {
  deleteAllQueryDataLayouts();
}

QueryBufferType QueryLayoutBuffer::getQueryBufferType() const {
  return src_interop_buffer_.getType();
}

bool QueryLayoutBuffer::isMapped() const {
  return src_interop_buffer_.isMapped();
}

const gfx::BufferWrapper* QueryLayoutBuffer::getBufferWrapper() const {
  return src_interop_buffer_.getBufferWrapper();
}

bool QueryLayoutBuffer::hasAttribute(const std::string& attr_name) const {
  return getLayoutManager().hasAttribute(attr_name, nullptr);
}

bool QueryLayoutBuffer::hasAttribute(const std::string& attr_name,
                                     const QueryDataLayout& layout) const {
  return getLayoutManager().hasAttribute(attr_name, layout.getBufferLayout());
}

bool QueryLayoutBuffer::hasBufferLayout(const QueryDataLayout& layout) const {
  return getLayoutManager().hasBufferLayout(layout.getBufferLayout());
}

gfx::BufferAttrType QueryLayoutBuffer::getAttributeType(
    const std::string& attr_name) const {
  return getLayoutManager().getAttributeType(attr_name, nullptr);
}

gfx::BufferAttrType QueryLayoutBuffer::getAttributeType(
    const std::string& attr_name,
    const QueryDataLayout& layout) const {
  return getLayoutManager().getAttributeType(attr_name, layout.getBufferLayout());
}

gfx::TypeGLSLShPtr QueryLayoutBuffer::getAttributeTypeGLSL(
    const std::string& attr_name,
    const QueryDataLayout& layout) const {
  return getLayoutManager().getAttributeTypeGLSL(attr_name, layout.getBufferLayout());
}

void QueryLayoutBuffer::setQueryDataLayout(
    const QueryDataLayoutShPtr& query_data_layout_ptr,
    const uint64_t used_bytes,
    const uint64_t offset_bytes) {
  RENDER_LOG_SCOPE() << "query_data_layout_ptr=" << query_data_layout_ptr;
  CHECK(query_data_layout_ptr);

  // TODO(croot): the runtime logic based on buffer type could be pushed down a level or
  // two. For example, we could collapse QueryDataLayout::convertToBufferLayout &
  // QueryDataLayout::convertToSSBOLayout into a single method that takes the intended
  // cosumer buffer type as an argument and that could do the appropriate conversion on
  // the fly instead of here.
  auto gfx_compatible_layout = query_data_layout_ptr->getBufferLayout();
  const auto buffer_type = getBufferWrapper()->getType();
  const auto usage_bits = getBufferWrapper()->getUsageBits();
  if (buffer_type != gfx::BufferType::kVertexBuffer) {
    static constexpr gfx::BufferUsageBits kRequiredBitsMask =
        gfx::BufferUsageBits::kStorageBufferBit | gfx::BufferUsageBits::kUniformBufferBit;
    CHECK(any_bits_set(usage_bits & kRequiredBitsMask));
  }

  uint64_t bytes_per_vertex = gfx_compatible_layout->getNumBytesPerItem();
  RUNTIME_EX_ASSERT(
      used_bytes % bytes_per_vertex == 0,
      "QueryLayoutBuffer " + std::to_string(src_interop_buffer_.getGpuId()) +
          ": Buffer layout bytes-per-vertex " + std::to_string(bytes_per_vertex) +
          " does not align with the number of used bytes in the buffer: " +
          std::to_string(used_bytes) + ".");

  getLayoutManagerInternal().replaceBufferLayoutAtOffset(
      gfx_compatible_layout, used_bytes, offset_bytes);
  query_data_layout_ptr_ = query_data_layout_ptr;
}

void QueryLayoutBuffer::deleteAllQueryDataLayouts() {
  query_data_layout_ptr_ = nullptr;
  getLayoutManagerInternal().deleteAllBufferLayouts();
}

const gfx::BufferLayoutManager& QueryLayoutBuffer::getLayoutManager() const {
  CHECK(src_interop_buffer_.getBufferWrapper()->hasLayout());
  return *src_interop_buffer_.getBufferWrapper()->getLayoutManager();
}

gfx::BufferLayoutManager& QueryLayoutBuffer::getLayoutManagerInternal() {
  CHECK(getBufferWrapper()->hasLayout());
  return *getBufferWrapper()->getLayoutManager();
}

//
// QueryVertexBuffer
//

QueryVertexBuffer::QueryVertexBuffer(QueryBufferManager& query_buffer_mgr,
                                     const uint64_t num_bytes)
    : QueryLayoutBuffer(query_buffer_mgr, QueryBufferType::kVertex, num_bytes) {}

QueryVertexBuffer::QueryVertexBuffer(QueryBufferManager& query_buffer_mgr,
                                     void* data,
                                     const uint64_t num_bytes,
                                     const gfx::BufferLayoutShPtr& layout_ptr)
    : QueryLayoutBuffer(query_buffer_mgr,
                        QueryBufferType::kVertex,
                        data,
                        num_bytes,
                        layout_ptr) {}

gfx::VertexBuffer* QueryVertexBuffer::unmapForDraw() {
  unmap();
  return static_cast<gfx::VertexBuffer*>(src_interop_buffer_.getBufferWrapper());
}

uint32_t QueryVertexBuffer::numVertices(const QueryDataLayoutShPtr& layout_ptr) const {
  CHECK(layout_ptr);
  return getLayoutManager().numItems(layout_ptr->getBufferLayout());
}

void QueryVertexBuffer::updateSubDataWithLayout(const void* data,
                                                const uint64_t num_bytes,
                                                const uint64_t offset_bytes,
                                                const gfx::BufferLayoutShPtr& layout) {
  auto* vbo = static_cast<gfx::VertexBuffer*>(src_interop_buffer_.getBufferWrapper());
  vbo->updateSubDataWithLayout(data, num_bytes, offset_bytes, layout);
}

//
// QueryIndexBuffer
//

QueryIndexBuffer::QueryIndexBuffer(QueryBufferManager& query_buffer_mgr,
                                   const uint64_t num_bytes)
    : QueryBuffer(query_buffer_mgr, QueryBufferType::kIndex, num_bytes) {}

QueryIndexBuffer::QueryIndexBuffer(QueryBufferManager& query_buffer_mgr,
                                   const std::vector<uint32_t>& items)
    : QueryBuffer(query_buffer_mgr,
                  QueryBufferType::kIndex,
                  items.size() * sizeof(uint32_t)) {
  auto* source_buffer = getBufferWrapper();
  source_buffer->updateSubData(items.data(), source_buffer->getNumBytes(), 0);
}

gfx::IndexBuffer* QueryIndexBuffer::unmapForDraw() {
  if (isMapped()) {
    unmap();
  }
  return static_cast<gfx::IndexBuffer*>(getBufferWrapper());
}

uint32_t QueryIndexBuffer::numItems() const {
  auto* buffer_resource = static_cast<const gfx::IndexBuffer*>(getBufferWrapper());
  return (buffer_resource ? buffer_resource->numItems() : 0);
}

//
// QueryShaderStorageBuffer
//

QueryShaderStorageBuffer::QueryShaderStorageBuffer(QueryBufferManager& query_buffer_mgr,
                                                   const uint64_t num_bytes)
    : QueryLayoutBuffer(query_buffer_mgr, QueryBufferType::kStorage, num_bytes) {}

QueryShaderStorageBuffer::QueryShaderStorageBuffer(
    QueryBufferManager& query_buffer_mgr,
    void* data,
    const uint64_t num_bytes,
    const gfx::BufferLayoutShPtr& layout_ptr)
    : QueryLayoutBuffer(query_buffer_mgr,
                        QueryBufferType::kStorage,
                        data,
                        num_bytes,
                        layout_ptr) {}

gfx::BufferWrapper* QueryShaderStorageBuffer::unmapForDraw() {
  unmap();
  return src_interop_buffer_.getBufferWrapper();
}

//
// QueryIndirectVbo
//

QueryIndirectVbo::QueryIndirectVbo(QueryBufferManager& query_buffer_mgr,
                                   const uint64_t num_bytes)
    : QueryBuffer(query_buffer_mgr, QueryBufferType::kIndirectVertex, num_bytes) {}

QueryIndirectVbo::QueryIndirectVbo(QueryBufferManager& query_buffer_mgr,
                                   const std::vector<gfx::IndirectDrawVertexData>& items)
    : QueryBuffer(query_buffer_mgr, QueryBufferType::kIndirectVertex, 0) {
  auto* source_buffer = getBufferWrapper();
  CHECK_EQ(source_buffer->getType(), gfx::BufferType::kIndirectDrawVertexBuffer);
  auto& indirect_vbo = static_cast<gfx::IndirectDrawVertexBuffer&>(*source_buffer);
  indirect_vbo.create(items);
}

gfx::IndirectDrawVertexBuffer* QueryIndirectVbo::unmapForDraw() {
  if (isMapped()) {
    unmap();
  }
  return static_cast<gfx::IndirectDrawVertexBuffer*>(getBufferWrapper());
}

uint32_t QueryIndirectVbo::numItems() const {
  auto const* buffer_resource =
      static_cast<const gfx::IndirectDrawVertexBuffer*>(getBufferWrapper());
  return (buffer_resource ? buffer_resource->numItems() : 0);
}

//
// QueryIndirectIbo
//

QueryIndirectIbo::QueryIndirectIbo(QueryBufferManager& query_buffer_mgr,
                                   const uint64_t num_bytes)
    : QueryBuffer(query_buffer_mgr, QueryBufferType::kIndirectIndex, num_bytes) {}

QueryIndirectIbo::QueryIndirectIbo(QueryBufferManager& query_buffer_mgr,
                                   const std::vector<gfx::IndirectDrawIndexData>& items)
    : QueryBuffer(query_buffer_mgr, QueryBufferType::kIndirectIndex, 0) {
  auto* source_buffer = getBufferWrapper();
  CHECK_EQ(source_buffer->getType(), gfx::BufferType::kIndirectDrawIndexBuffer);
  auto& indirect_ibo = static_cast<gfx::IndirectDrawIndexBuffer&>(*source_buffer);
  indirect_ibo.create(items);
}

gfx::IndirectDrawIndexBuffer* QueryIndirectIbo::unmapForDraw() {
  if (isMapped()) {
    unmap();
  }
  return static_cast<gfx::IndirectDrawIndexBuffer*>(getBufferWrapper());
}

uint32_t QueryIndirectIbo::numItems() const {
  auto const* buffer_resource =
      static_cast<const gfx::IndirectDrawIndexBuffer*>(getBufferWrapper());
  return (buffer_resource ? buffer_resource->numItems() : 0);
}

}  // namespace QueryRenderer
