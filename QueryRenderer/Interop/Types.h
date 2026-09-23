/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

namespace QueryRenderer {

class QueryBuffer;
class QueryLayoutBuffer;
using QueryLayoutBufferShPtr = std::shared_ptr<QueryLayoutBuffer>;
using QueryLayoutBufferWkPtr = std::weak_ptr<QueryLayoutBuffer>;

class QueryVertexBuffer;
using QueryVertexBufferShPtr = std::shared_ptr<QueryVertexBuffer>;
using QueryVertexBufferWkPtr = std::weak_ptr<QueryVertexBuffer>;

class QueryIndexBuffer;
using QueryIndexBufferShPtr = std::shared_ptr<QueryIndexBuffer>;
using QueryIndexBufferWkPtr = std::weak_ptr<QueryIndexBuffer>;

class QueryShaderStorageBuffer;
using QueryShaderStorageBufferShPtr = std::shared_ptr<QueryShaderStorageBuffer>;
using QueryShaderStorageBufferWkPtr = std::weak_ptr<QueryShaderStorageBuffer>;

class QueryIndirectVbo;
using QueryIndirectVboShPtr = std::shared_ptr<QueryIndirectVbo>;
using QueryIndirectVboWkPtr = std::weak_ptr<QueryIndirectVbo>;

class QueryIndirectIbo;
using QueryIndirectIboShPtr = std::shared_ptr<QueryIndirectIbo>;
using QueryIndirectIboWkPtr = std::weak_ptr<QueryIndirectIbo>;

// extra data for poly rendering
struct PolyDrawBatchInfo {
  explicit PolyDrawBatchInfo(std::vector<uint32_t>&& nr, std::vector<uint32_t>&& np)
      : num_rows{std::move(nr)}, num_polys{std::move(np)} {}
  std::vector<uint32_t> num_rows;
  std::vector<uint32_t> num_polys;
};
using PolyDrawBatchInfoUqPtr = std::unique_ptr<PolyDrawBatchInfo>;

}  // namespace QueryRenderer
