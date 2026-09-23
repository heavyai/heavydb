/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "GfxDriver/Resources/Types.h"

namespace gfx {

using VboAttrToShaderAttrPair = std::pair<std::string, std::string>;
using VboAttrToShaderAttrPairs = std::vector<VboAttrToShaderAttrPair>;

struct VboAndLayout {
  const VertexBuffer* vertex_buffer;
  BufferLayoutShPtr buffer_layout;
};

struct PrimitiveAssemblyAttrInfo {
  VboAndLayout vbo_and_layout{nullptr, nullptr};
  VboAttrToShaderAttrPairs attr_pairs;
};

class PrimitiveAssemblyDependency;

enum class PrimitiveTopology : uint8_t {
  kPointList,
  kLineList,
  kLineStrip,
  kTriangleList,
  kTriangleStrip,
  kTriangleFan,
  kLineListAdjacency,
  kLineStripAdjacency,
  kTriangleListAdjacency,
  kTriangleStripAdjacency
};

class PrimitiveAssembly {
 public:
  explicit PrimitiveAssembly(const PrimitiveTopology topology);
  explicit PrimitiveAssembly(const PrimitiveTopology topology,
                             const PrimitiveAssemblyAttrInfo& attr_info,
                             const IndexBuffer* index_buffer);
  PrimitiveAssembly() = delete;
  virtual ~PrimitiveAssembly();

  virtual uint32_t numVertices() const = 0;
  virtual uint64_t getVertexBufferOffsetBytes() const = 0;
  const VertexBuffer* getVertexBuffer() { return vertex_buffer_; }

  virtual uint32_t numIndices() const = 0;
  const IndexBuffer* getIndexBuffer() { return index_buffer_; }

  virtual uint32_t numInstances() const = 0;
  virtual bool isDirty() const = 0;
  virtual void markDirty() = 0;

  PrimitiveTopology getTopology() const { return topology_; }
  void addDependency(const PrimitiveAssemblyDependency* dependency);
  void removeDependency(const PrimitiveAssemblyDependency* dependency);

 private:
  std::unordered_set<const PrimitiveAssemblyDependency*> dependencies_;
  const PrimitiveTopology topology_;
  const VertexBuffer* vertex_buffer_;
  const IndexBuffer* index_buffer_;
};

}  // namespace gfx
