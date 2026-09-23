/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/IndexBuffer.h"
#include "GfxDriver/Resources/IndirectDrawBuffer.h"
#include "GfxDriver/Resources/VertexBuffer.h"
#include "QueryRenderer/Interop/Enums.h"
#include "QueryRenderer/QueryBufferManager.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/Types.h"
#include "Shared/ShapeDrawData.h"

namespace QueryRenderer {

struct PolyTableByteData {
  uint64_t num_vertex_bytes = 0;
  uint64_t num_line_indirect_draw_bytes = 0;
  uint64_t num_poly_indirect_draw_bytes = 0;
  uint64_t num_ssbo_bytes = 0;
  uint64_t num_poly_rowids_bytes = 0;
};

struct PolyTableDataInfo {
  uint32_t num_vertices = 0;
  uint32_t num_lines = 0;
  uint32_t num_polys = 0;
  uint64_t num_ssbo_bytes = 0;
  uint64_t num_poly_rowids_bytes = 0;
};

struct PolyTableLayoutInfo {
  QueryDataLayoutShPtr vboLayout;
  QueryDataLayoutShPtr ssboLayout;
};

struct LineTableByteData {
  uint64_t num_vertex_bytes = 0;
  uint64_t num_index_buffer_bytes = 0;
  uint64_t num_ssbo_bytes = 0;
  uint64_t num_indirect_vertex_bytes = 0;
  uint64_t num_indirect_index_bytes = 0;
};

struct LineTableDataInfo {
  uint32_t num_vertices = 0;
  uint32_t num_indices = 0;
  uint32_t num_line_segments = 0;
  uint64_t num_ssbo_bytes = 0;
};

class QueryBuffer {
 public:
  explicit QueryBuffer(QueryBufferManager& query_buffer_mgr,
                       const QueryBufferType type,
                       const uint64_t num_bytes);
  virtual ~QueryBuffer();

  int32_t getGpuId() const;
  QueryBufferType getType() const;

  bool isMapped() const;
  uint64_t getNumUsedBytes(const QueryDataLayoutShPtr& layout_ptr = nullptr) const;
  uint64_t getNumBytes() const;
  void rebuild(const uint64_t num_bytes);

  const gfx::BufferWrapper* getBufferWrapper() const;
  gfx::BufferWrapper* getBufferWrapper();

  gfx::BufferMemoryDescriptor getBufferMemoryDescriptor();
  void unmap();

 private:
  QueryBufferManager& query_buffer_mgr_;
  QueryBufferManager::BuffersAndDescriptor buffers_and_descriptor_;
  QueryBufferType type_;

  void create(const uint64_t num_bytes);
  void destroy();
};

class QueryLayoutBuffer {
 public:
  QueryLayoutBuffer() = delete;
  virtual ~QueryLayoutBuffer() = default;

  QueryDataLayoutShPtr getQueryDataLayout() const;

  uint64_t getNumBytes() const;
  uint64_t getNumUsedBytes(const QueryDataLayout& layout) const;
  uint64_t getLayoutOffsetBytes(const QueryDataLayout& layout_ptr) const;

  gfx::BufferMemoryDescriptor getBufferMemoryDescriptor();
  void unmap();
  void rebuild(const uint64_t num_bytes);
  void reset();

  QueryBufferType getQueryBufferType() const;

  bool isMapped() const;
  const gfx::BufferWrapper* getBufferWrapper() const;
  bool hasAttribute(const std::string& attr_name, const QueryDataLayout& layout) const;
  bool hasBufferLayout(const QueryDataLayout& layout) const;

  gfx::BufferAttrType getAttributeType(const std::string& attr_name,
                                       const QueryDataLayout& layout) const;
  gfx::TypeGLSLShPtr getAttributeTypeGLSL(const std::string& attr_name,
                                          const QueryDataLayout& layout) const;

  // Variants for Embedded data tables, which do not store the layout
  // TODO(scb) This can be fixed with dedicated work on the embedded table update paths,
  // as they do generate a layout, but don't store it
  bool hasAttribute(const std::string& attr_name) const;
  gfx::BufferAttrType getAttributeType(const std::string& attr_name) const;

  void setQueryDataLayout(const QueryDataLayoutShPtr& query_data_layout_ptr,
                          const uint64_t used_bytes = 0,
                          const uint64_t offset_bytes = 0);

  void deleteAllQueryDataLayouts();

  const gfx::BufferLayoutManager& getLayoutManager() const;

 protected:
  QueryBuffer src_interop_buffer_;
  QueryDataLayoutShPtr query_data_layout_ptr_;

  explicit QueryLayoutBuffer(QueryBufferManager& query_buffer_mgr,
                             QueryBufferType buffer_type,
                             const uint64_t num_bytes)
      : src_interop_buffer_(query_buffer_mgr, buffer_type, num_bytes) {}

  explicit QueryLayoutBuffer(QueryBufferManager& query_buffer_mgr,
                             QueryBufferType buffer_type,
                             void* data,
                             const uint64_t num_bytes,
                             const gfx::BufferLayoutShPtr& layout_ptr)
      : src_interop_buffer_(query_buffer_mgr, buffer_type, num_bytes) {
    src_interop_buffer_.getBufferWrapper()->updateSubDataWithLayout(
        data, num_bytes, 0, layout_ptr);
  }

  gfx::BufferLayoutManager& getLayoutManagerInternal();
};

class QueryVertexBuffer : public QueryLayoutBuffer {
 public:
  explicit QueryVertexBuffer(QueryBufferManager& query_buffer_mgr,
                             const uint64_t num_bytes = 0);

  explicit QueryVertexBuffer(QueryBufferManager& query_buffer_mgr,
                             void* data,
                             const uint64_t num_bytes,
                             const gfx::BufferLayoutShPtr& layout_ptr);

  ~QueryVertexBuffer() override = default;

  gfx::VertexBuffer* unmapForDraw();

  uint32_t numVertices(const QueryDataLayoutShPtr& layout_ptr = nullptr) const;
  void updateSubDataWithLayout(const void* data,
                               const uint64_t num_bytes,
                               const uint64_t offset_bytes,
                               const gfx::BufferLayoutShPtr& layout);
};

class QueryIndexBuffer : public QueryBuffer {
 public:
  explicit QueryIndexBuffer(QueryBufferManager& query_buffer_mgr,
                            const uint64_t num_bytes);
  explicit QueryIndexBuffer(QueryBufferManager& query_buffer_mgr,
                            const std::vector<uint32_t>& items);
  ~QueryIndexBuffer() override = default;

  gfx::IndexBuffer* unmapForDraw();

  uint32_t numItems() const;
};

class QueryShaderStorageBuffer : public QueryLayoutBuffer {
 public:
  explicit QueryShaderStorageBuffer(QueryBufferManager& query_buffer_mgr,
                                    const uint64_t num_bytes = 0);

  explicit QueryShaderStorageBuffer(QueryBufferManager& query_buffer_mgr,
                                    void* data,
                                    const uint64_t num_bytes,
                                    const gfx::BufferLayoutShPtr& layout_ptr);

  ~QueryShaderStorageBuffer() override = default;

  gfx::BufferWrapper* unmapForDraw();
};

class QueryIndirectVbo : public QueryBuffer {
 public:
  explicit QueryIndirectVbo(QueryBufferManager& query_buffer_mgr,
                            const uint64_t num_bytes);
  explicit QueryIndirectVbo(QueryBufferManager& query_buffer_mgr,
                            const std::vector<gfx::IndirectDrawVertexData>& items);
  ~QueryIndirectVbo() override = default;

  gfx::IndirectDrawVertexBuffer* unmapForDraw();

  uint32_t numItems() const;
};

class QueryIndirectIbo : public QueryBuffer {
 public:
  explicit QueryIndirectIbo(QueryBufferManager& query_buffer_mgr,
                            const uint64_t num_bytes);
  explicit QueryIndirectIbo(QueryBufferManager& query_buffer_mgr,
                            const std::vector<gfx::IndirectDrawIndexData>& items);
  ~QueryIndirectIbo() override = default;

  gfx::IndirectDrawIndexBuffer* unmapForDraw();

  uint32_t numItems() const;
};

}  // namespace QueryRenderer
