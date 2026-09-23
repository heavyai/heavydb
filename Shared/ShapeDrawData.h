/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace gfx {

struct IndirectDrawVertexData {
  uint32_t vertex_count;
  uint32_t instance_count;
  uint32_t first_vertex;
  uint32_t first_instance;

  IndirectDrawVertexData()
      : vertex_count{0}, instance_count{0}, first_vertex{0}, first_instance{0} {}

  IndirectDrawVertexData(const uint32_t vertex_count,
                         const uint32_t instance_count = 1,
                         const uint32_t first_vertex = 0,
                         const uint32_t first_instance = 0)
      : vertex_count{vertex_count}
      , instance_count{instance_count}
      , first_vertex{first_vertex}
      , first_instance{first_instance} {}
};

struct IndirectDrawIndexData {
  uint32_t index_count;
  uint32_t instance_count;
  uint32_t first_index;
  uint32_t vertex_offset;
  uint32_t first_instance;

  IndirectDrawIndexData()
      : index_count{0}
      , instance_count{0}
      , first_index{0}
      , vertex_offset{0}
      , first_instance{0} {}

  IndirectDrawIndexData(const uint32_t index_count,
                        const uint32_t instance_count = 1,
                        const uint32_t first_index = 0,
                        const uint32_t vertex_offset = 0,
                        const uint32_t first_instance = 0)
      : index_count{index_count}
      , instance_count{instance_count}
      , first_index{first_index}
      , vertex_offset{vertex_offset}
      , first_instance{first_instance} {}
};

}  // namespace gfx
