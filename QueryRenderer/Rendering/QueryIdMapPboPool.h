/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Rendering/BaseQueryRsrcPool.h"
#include "QueryRenderer/Rendering/QueryIdMapPixelBuffer.h"
#include "QueryRenderer/Rendering/Types.h"

namespace QueryRenderer {

template <typename T = uint32_t>
class QueryIdMapPboPoolT
    : public BaseQueryRsrcPool<QueryIdMapPixelBuffer<T>, 300000, size_t, size_t> {
 public:
  QueryIdMapPboPoolT(gfx::ResourceManager& resource_mgr) : resource_mgr_(resource_mgr) {}
  ~QueryIdMapPboPoolT() override {}

 private:
  std::shared_ptr<QueryIdMapPixelBuffer<T>> initializeRsrc(size_t width,
                                                           size_t height) final {
    return std::make_shared<QueryIdMapPixelBuffer<T>>(this->resource_mgr_, width, height);
  }

  void updateRsrc(std::shared_ptr<QueryIdMapPixelBuffer<T>>& rsrc_ptr,
                  size_t width,
                  size_t height) final {
    rsrc_ptr->resize(width, height);
  }

  gfx::ResourceManager& resource_mgr_;
};

}  // namespace QueryRenderer
