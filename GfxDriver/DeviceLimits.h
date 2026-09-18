/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>

#include "Logger/Logger.h"

namespace gfx {

struct DeviceLimits {
  // Required
  uint32_t uniform_buffer_alignment = 0U;
  uint32_t shader_storage_buffer_alignment = 0U;
  uint64_t max_uniform_buffer_size = 0ULL;
  uint64_t max_shader_storage_buffer_size = 0ULL;

  uint32_t max_compute_workgroup_count[3] = {0U, 0U, 0U};
  uint32_t subgroup_size = 0U;
  uint32_t max_compute_workgroup_subgroups = 0U;
  uint32_t max_compute_shared_memory_size = 0U;

  uint32_t max_framebuffer_width = 0U;
  uint32_t max_framebuffer_height = 0U;

  float timestamp_period = 1.0f;

  // Raytracing (optional)
  uint32_t shader_group_handle_size = 0U;
  uint32_t shader_group_handle_alignment = 0U;
  uint32_t shader_group_base_alignment = 0U;

  // Mesh and Task shaders (optional)
  uint32_t max_task_workgroup_total_count = 0U;
  uint32_t max_task_workgroup_count[3] = {0U, 0U, 0U};
  uint32_t max_task_workgroup_invocations = 0U;
  uint32_t max_task_workgroup_size[3] = {0U, 0U, 0U};
  uint32_t max_task_payload_size = 0U;
  uint32_t max_task_shared_memory_size = 0U;
  uint32_t max_task_payload_and_shared_memory_size = 0U;
  uint32_t max_mesh_workgroup_total_count = 0U;
  uint32_t max_mesh_workgroup_count[3] = {0U, 0U, 0U};
  uint32_t max_mesh_workgroup_invocations = 0U;
  uint32_t max_mesh_workgroup_size[3] = {0U, 0U, 0U};
  uint32_t max_mesh_shared_memory_size = 0U;
  uint32_t max_mesh_payload_and_shared_memory_size = 0U;
  uint32_t max_mesh_output_memory_size = 0U;
  uint32_t max_mesh_payload_and_output_memory_size = 0U;
  uint32_t max_mesh_output_components = 0U;
  uint32_t max_mesh_output_vertices = 0U;
  uint32_t max_mesh_output_primitives = 0U;
  uint32_t max_mesh_output_layers = 0U;
  uint32_t max_mesh_multiview_view_count = 0U;
  uint32_t mesh_output_per_vertex_granularity = 0U;
  uint32_t mesh_output_per_primitive_granularity = 0U;
  uint32_t max_preferred_task_workgroup_invocations = 0U;
  uint32_t max_preferred_mesh_workgroup_invocations = 0U;

  inline bool areValid() const {
    return (uniform_buffer_alignment > 0U) && (shader_storage_buffer_alignment > 0U) &&
           (max_uniform_buffer_size > 0ULL) && (max_shader_storage_buffer_size > 0ULL) &&
           (max_compute_shared_memory_size > 0U) && (max_framebuffer_width > 0U) &&
           (max_framebuffer_height > 0U);
  }

  inline void combine(const DeviceLimits& other) {
    // We require all devices to return the same timestamp period
    CHECK_EQ(timestamp_period, other.timestamp_period);
    uniform_buffer_alignment =
        std::max(uniform_buffer_alignment, other.uniform_buffer_alignment);
    shader_storage_buffer_alignment =
        std::max(shader_storage_buffer_alignment, other.shader_storage_buffer_alignment);
    max_uniform_buffer_size =
        std::min(max_uniform_buffer_size, other.max_uniform_buffer_size);
    max_shader_storage_buffer_size =
        std::min(max_shader_storage_buffer_size, other.max_shader_storage_buffer_size);

    for (int i = 0; i < 3; ++i) {
      max_compute_workgroup_count[i] =
          std::min(max_compute_workgroup_count[i], other.max_compute_workgroup_count[i]);
    }
    subgroup_size = std::min(subgroup_size, other.subgroup_size);
    max_compute_workgroup_subgroups =
        std::min(max_compute_workgroup_subgroups, other.max_compute_workgroup_subgroups);
    max_compute_shared_memory_size =
        std::min(max_compute_shared_memory_size, other.max_compute_shared_memory_size);

    max_framebuffer_width = std::min(max_framebuffer_width, other.max_framebuffer_width);
    max_framebuffer_height =
        std::min(max_framebuffer_height, other.max_framebuffer_height);

    for (int i = 0; i < 3; i++) {
      max_task_workgroup_count[i] =
          std::min(max_task_workgroup_count[i], other.max_task_workgroup_count[i]);
      max_task_workgroup_size[i] =
          std::min(max_task_workgroup_size[i], other.max_task_workgroup_size[i]);
      max_mesh_workgroup_count[i] =
          std::min(max_mesh_workgroup_count[i], other.max_mesh_workgroup_count[i]);
      max_mesh_workgroup_size[i] =
          std::min(max_mesh_workgroup_size[i], other.max_mesh_workgroup_size[i]);
    }

    max_task_workgroup_total_count =
        std::min(max_task_workgroup_total_count, other.max_task_workgroup_total_count);
    max_task_workgroup_invocations =
        std::min(max_task_workgroup_invocations, other.max_task_workgroup_invocations);
    max_task_payload_size = std::min(max_task_payload_size, other.max_task_payload_size);
    max_task_shared_memory_size =
        std::min(max_task_shared_memory_size, other.max_task_shared_memory_size);
    max_task_payload_and_shared_memory_size =
        std::min(max_task_payload_and_shared_memory_size,
                 other.max_task_payload_and_shared_memory_size);
    max_mesh_workgroup_total_count =
        std::min(max_mesh_workgroup_total_count, other.max_mesh_workgroup_total_count);
    max_mesh_workgroup_invocations =
        std::min(max_mesh_workgroup_invocations, other.max_mesh_workgroup_invocations);
    max_mesh_shared_memory_size =
        std::min(max_mesh_shared_memory_size, other.max_mesh_shared_memory_size);
    max_mesh_payload_and_shared_memory_size =
        std::min(max_mesh_payload_and_shared_memory_size,
                 other.max_mesh_payload_and_shared_memory_size);
    max_mesh_output_memory_size =
        std::min(max_mesh_output_memory_size, other.max_mesh_output_memory_size);
    max_mesh_payload_and_output_memory_size =
        std::min(max_mesh_payload_and_output_memory_size,
                 other.max_mesh_payload_and_output_memory_size);
    max_mesh_output_components =
        std::min(max_mesh_output_components, other.max_mesh_output_components);
    max_mesh_output_vertices =
        std::min(max_mesh_output_vertices, other.max_mesh_output_vertices);
    max_mesh_output_primitives =
        std::min(max_mesh_output_primitives, other.max_mesh_output_primitives);
    max_mesh_output_layers =
        std::min(max_mesh_output_layers, other.max_mesh_output_layers);
    max_mesh_multiview_view_count =
        std::min(max_mesh_multiview_view_count, other.max_mesh_multiview_view_count);
    mesh_output_per_vertex_granularity = std::min(
        mesh_output_per_vertex_granularity, other.mesh_output_per_vertex_granularity);
    mesh_output_per_primitive_granularity =
        std::min(mesh_output_per_primitive_granularity,
                 other.mesh_output_per_primitive_granularity);
    max_preferred_task_workgroup_invocations =
        std::min(max_preferred_task_workgroup_invocations,
                 other.max_preferred_task_workgroup_invocations);
    max_preferred_mesh_workgroup_invocations =
        std::min(max_preferred_mesh_workgroup_invocations,
                 other.max_preferred_mesh_workgroup_invocations);

    // use max for granularity to ensure maximized memory calculation
    mesh_output_per_vertex_granularity = std::max(
        mesh_output_per_vertex_granularity, other.mesh_output_per_vertex_granularity);
    mesh_output_per_primitive_granularity =
        std::max(mesh_output_per_primitive_granularity,
                 other.mesh_output_per_primitive_granularity);
  }
};

}  // namespace gfx
