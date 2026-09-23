/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Commands/QueryPool.h"

#include <sstream>

#include "GfxDriver/Utils/LoggingUtils.h"

namespace gfx {

QueryPool::QueryPool(const DeviceContext& device_context,
                     std::string_view resource_tracking_string)
    : Resource(device_context, resource_tracking_string, ResourceType::kQueryPool) {}

std::string to_string(const gfx::QueryPool::Type type) {
  using e = gfx::QueryPool::Type;
  switch (type) {
    case e::kTimestamp:
      return "kTimestamp";
    case e::kPipelineStatistics:
      return "kPipelineStatistics";
    case e::kOcclusion:
      return "kOcclusion";
  }
  return "";
}

std::string to_string(const gfx::QueryPool::GetResultMode option) {
  using e = gfx::QueryPool::GetResultMode;
  switch (option) {
    case e::kNone:
      return "None";
    case e::kWait:
      return "Wait";
    case e::kWithAvailability:
      return "WithAvailability";
  }
  return "";
}

std::string to_string(const gfx::GraphicsPipelineStatistics& statistics) {
  std::stringstream ss;
  StreamStatFormatter sf(ss);
  sf("Input Assembly Vertices", statistics.input_assembly_vertices);
  sf("Input Assembly Primitives", statistics.input_assembly_primitives);
  sf("Vertex Shader Invocations", statistics.vertex_shader_invocations);
  sf("Geometry Shader Invocations", statistics.geometry_shader_invocations);
  sf("Geometry Shader Primitives", statistics.geometry_shader_primitives);
  sf("Clipping Invocations", statistics.clipping_invocations);
  sf("Clipping Primitives", statistics.clipping_primitives);
  sf("Fragment Shader Invocations", statistics.fragment_shader_invocations);
  sf("Task Shader Invocations", statistics.task_shader_invocations);
  sf("Mesh Shader Invocations", statistics.mesh_shader_invocations);
  sf("Availability", statistics.availability);
  return ss.str();
}

std::string to_string(const gfx::ComputePipelineStatistics& statistics) {
  std::stringstream ss;
  StreamStatFormatter sf(ss);
  sf("Compute Shader Invocations", statistics.compute_shader_invocations);
  sf("Availability", statistics.availability);
  return ss.str();
}

}  // end namespace gfx

std::ostream& operator<<(std::ostream& os, const gfx::QueryPool::Type type) {
  using e = gfx::QueryPool::Type;
  switch (type) {
    case e::kTimestamp:
      os << "Timestamp";
      break;
    case e::kPipelineStatistics:
      os << "PipelineStatistics";
      break;
    case e::kOcclusion:
      os << "Occlusion";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const gfx::QueryPool::GetResultMode option) {
  using e = gfx::QueryPool::GetResultMode;
  switch (option) {
    case e::kNone:
      os << "None";
      break;
    case e::kWait:
      os << "Wait";
      break;
    case e::kWithAvailability:
      os << "WithAvailability";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const gfx::GraphicsPipelineStatistics& statistics) {
  os << to_string(statistics);
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const gfx::ComputePipelineStatistics& statistics) {
  os << to_string(statistics);
  return os;
}
