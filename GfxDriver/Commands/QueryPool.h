/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <boost/noncopyable.hpp>

#include "GfxDriver/Resources/Resource.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

// There is no device limit available for this, so just pick a
// reasonably high number
static constexpr uint32_t kQueryPoolMaxSize = 512;

// Always use 64-bit timestamps as 32-bit isn't enough duration
using Timestamp = uint64_t;

// Use 32-bit for statistics
using Statistic = uint32_t;

struct GraphicsPipelineStatistics {
  // these fields must be in the order of VK_QUERY_PIPELINE_STATISTIC_*_BIT
  // bit values, as that is the order that the stats are written to this block
  Statistic input_assembly_vertices;
  Statistic input_assembly_primitives;
  Statistic vertex_shader_invocations;
  Statistic geometry_shader_invocations;
  Statistic geometry_shader_primitives;
  Statistic clipping_invocations;
  Statistic clipping_primitives;
  Statistic fragment_shader_invocations;
  Statistic task_shader_invocations;
  Statistic mesh_shader_invocations;
  Statistic availability;
};

struct ComputePipelineStatistics {
  Statistic compute_shader_invocations;
  Statistic availability;
};

struct OcclusionStatistics {
  Statistic num_samples;
  Statistic availability;
};

//
// QueryPool class
//
class QueryPool : public Resource {
 public:
  enum class Type { kTimestamp, kPipelineStatistics, kOcclusion };
  enum class GetResultMode { kNone, kWait, kWithAvailability };

  explicit QueryPool(const DeviceContext& device_context,
                     std::string_view resource_tracking_string);
  ~QueryPool() override = default;

  virtual Type getType() const = 0;
  virtual uint32_t getNumQueries() const = 0;

  // From Resource
  ResourceHandle getResourceHandle() const override = 0;

  // Reset all or a range of pool values
  virtual void reset(std::optional<uint32_t> first_query = std::nullopt,
                     std::optional<uint32_t> query_count = std::nullopt) = 0;

  //
  // Timestamps
  //
  // Get raw Timestamp values
  //
  virtual std::vector<Timestamp> getTimestampResults(
      std::optional<uint32_t> first_query = std::nullopt,
      std::optional<uint32_t> query_count = std::nullopt,
      std::optional<GetResultMode> get_result_mode = std::nullopt) = 0;

  // Convert a Timestamp value to a duration in microseconds
  // On NVidia this is 1 to 1
  virtual uint64_t timestampToMicroseconds(Timestamp timestamp) = 0;

  //
  // Statistics
  //
  virtual GraphicsPipelineStatistics getGraphicsPipelineStatistics(
      std::optional<GetResultMode> get_result_mode = std::nullopt) = 0;
  virtual ComputePipelineStatistics getComputePipelineStatistics(
      std::optional<GetResultMode> get_result_mode = std::nullopt) = 0;

  //
  // Occlusion
  //
  virtual OcclusionStatistics getOcclusionStatistics(
      std::optional<GetResultMode> get_result_mode = std::nullopt) = 0;
};

std::string to_string(const gfx::QueryPool::Type type);
std::string to_string(const gfx::QueryPool::GetResultMode option);
std::string to_string(const gfx::GraphicsPipelineStatistics& statistics);
std::string to_string(const gfx::ComputePipelineStatistics& statistics);

}  // namespace gfx

std::ostream& operator<<(std::ostream& os, const gfx::QueryPool::Type type);
std::ostream& operator<<(std::ostream& os, const gfx::QueryPool::GetResultMode option);
std::ostream& operator<<(std::ostream& os,
                         const gfx::GraphicsPipelineStatistics& statistics);
std::ostream& operator<<(std::ostream& os,
                         const gfx::ComputePipelineStatistics& statistics);
