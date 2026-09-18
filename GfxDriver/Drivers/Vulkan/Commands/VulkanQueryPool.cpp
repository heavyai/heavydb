/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanQueryPool.h"

#include "GfxDriver/Commands/QueryPool.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/Pipeline/Pipeline.h"

namespace gfx {

static constexpr std::array<VkQueryPipelineStatisticFlags, 2> kPipelineStatisticsFlags = {
    // graphics
    VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
        VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
        VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
        VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
        VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
        VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
        VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
        VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
        VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT |
        VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT,
    // compute
    VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT};

static constexpr std::array<uint32_t, 2> kPipelineStatisticsQueryPoolSizes = {
    // number of bits set in each of the above
    10u,
    1u};

static constexpr std::array<VkQueryType, 3> kQueryTypes = {
    VK_QUERY_TYPE_TIMESTAMP,
    VK_QUERY_TYPE_PIPELINE_STATISTICS,
    VK_QUERY_TYPE_OCCLUSION};

VulkanQueryPool::VulkanQueryPool(const DeviceContext& device_context,
                                 std::string_view resource_tracking_string,
                                 Type type,
                                 std::optional<Pipeline::Type> pipeline_type,
                                 std::optional<uint32_t> size)
    : QueryPool(device_context, resource_tracking_string)
    , type_{type}
    , pipeline_type_{pipeline_type}
    , vk_query_pool_{VK_NULL_HANDLE}
    , timestamp_period_{device_context.getLimits().timestamp_period} {
  // size and count depend on types
  switch (type_) {
    case Type::kTimestamp:
      CHECK(size) << "Query Pool size must be specified for Timestamp Queries";
      CHECK(!pipeline_type_);
      num_queries_ = num_results_ = *size;
      break;
    case Type::kPipelineStatistics:
      CHECK(!size) << "Query Pool size must NOT be specified for non-Timestamp Queries";
      num_queries_ = 1u;
      CHECK(pipeline_type_) << "Pipeline Type must be specified";
      CHECK_NE(*pipeline_type, Pipeline::Type::kRaytracing)
          << "Raytracing Pipeline Statistics not yet supported!";
      num_results_ = kPipelineStatisticsQueryPoolSizes[*pipeline_type];
      break;
    case Type::kOcclusion:
      CHECK(!size) << "Query Pool size must NOT be specified for non-Timestamp Queries";
      CHECK(pipeline_type_);
      CHECK_EQ(*pipeline_type_, Pipeline::Type::kGraphics)
          << "Occlusion Queries only supported for Graphics Pipelines";
      num_queries_ = num_results_ = 1u;
      break;
  }
  CHECK_LE(num_results_, kQueryPoolMaxSize);

  VkQueryPoolCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  ci.queryType = kQueryTypes[static_cast<int>(type_)];
  ci.queryCount = num_queries_;
  if (type == Type::kPipelineStatistics) {
    ci.pipelineStatistics = kPipelineStatisticsFlags[*pipeline_type_];
  }
  static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle();
  CHECK_VKRESULT(
      vkCreateQueryPool(
          static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle(),
          &ci,
          nullptr,
          &vk_query_pool_),
      "Creating QueryPool");
  setUsable();
  reset();
}

VulkanQueryPool::~VulkanQueryPool() {
  cleanupResource();
}

void VulkanQueryPool::cleanupResourceBase() {
  if (vk_query_pool_) {
    vkDestroyQueryPool(
        static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle(),
        vk_query_pool_,
        nullptr);
    vk_query_pool_ = VK_NULL_HANDLE;
  }
}

void VulkanQueryPool::makeEmpty() {}

void VulkanQueryPool::reset(std::optional<uint32_t> first_query,
                            std::optional<uint32_t> query_count) {
  uint32_t first = first_query ? *first_query : 0;
  uint32_t count = query_count ? *query_count : num_queries_;
  CHECK_LE(first + count, num_queries_);
  vkResetQueryPool(
      static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle(),
      vk_query_pool_,
      first,
      count);
}

std::vector<Timestamp> VulkanQueryPool::getTimestampResults(
    std::optional<uint32_t> first_query,
    std::optional<uint32_t> query_count,
    std::optional<GetResultMode> get_result_mode) {
  CHECK_EQ(type_, QueryPool::Type::kTimestamp);
  uint32_t first = first_query ? *first_query : 0;
  uint32_t count = query_count ? *query_count : num_queries_;
  CHECK_LE(first + count, num_queries_);

  size_t vector_size = count;
  uint32_t stride = sizeof(Timestamp);
  int flags = VK_QUERY_RESULT_64_BIT;
  if (get_result_mode) {
    switch (*get_result_mode) {
      case GetResultMode::kNone:
        break;
      case GetResultMode::kWait:
        flags = flags | VK_QUERY_RESULT_WAIT_BIT;
        break;
      case GetResultMode::kWithAvailability:
        flags = flags | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
        stride *= 2;
        vector_size *= 2;
        break;
    }
  }

  std::vector<Timestamp> results(vector_size, 0);

  CHECK_VKRESULT(
      vkGetQueryPoolResults(
          static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle(),
          vk_query_pool_,
          first,
          count,
          results.size() * sizeof(Timestamp),
          results.data(),
          stride,
          flags),
      "Getting Timestamp results");
  return results;
}

uint64_t VulkanQueryPool::timestampToMicroseconds(Timestamp timestamp) {
  return static_cast<uint64_t>(static_cast<double>(timestamp) * timestamp_period_) / 1000;
}

GraphicsPipelineStatistics VulkanQueryPool::getGraphicsPipelineStatistics(
    std::optional<GetResultMode> get_result_mode) {
  CHECK_EQ(type_, QueryPool::Type::kPipelineStatistics);
  CHECK(pipeline_type_);
  CHECK_EQ(*pipeline_type_, Pipeline::Type::kGraphics);
  GraphicsPipelineStatistics statistics{};
  getStatisticsInternal(reinterpret_cast<Statistic*>(&statistics), get_result_mode);
  return statistics;
}

ComputePipelineStatistics VulkanQueryPool::getComputePipelineStatistics(
    std::optional<GetResultMode> get_result_mode) {
  CHECK_EQ(type_, QueryPool::Type::kPipelineStatistics);
  CHECK(pipeline_type_);
  CHECK_EQ(*pipeline_type_, Pipeline::Type::kCompute);
  ComputePipelineStatistics statistics{};
  getStatisticsInternal(reinterpret_cast<Statistic*>(&statistics), get_result_mode);
  return statistics;
}

OcclusionStatistics VulkanQueryPool::getOcclusionStatistics(
    std::optional<GetResultMode> get_result_mode) {
  CHECK_EQ(type_, QueryPool::Type::kOcclusion);
  CHECK(pipeline_type_);
  CHECK_EQ(*pipeline_type_, Pipeline::Type::kGraphics);
  OcclusionStatistics statistics{};
  getStatisticsInternal(reinterpret_cast<Statistic*>(&statistics), get_result_mode);
  return statistics;
}

void VulkanQueryPool::getStatisticsInternal(
    Statistic* statistics,
    std::optional<GetResultMode> get_result_mode) {
  size_t vector_size = num_results_;
  VkQueryResultFlags flags{};
  if (get_result_mode) {
    switch (*get_result_mode) {
      case GetResultMode::kNone:
        break;
      case GetResultMode::kWait:
        flags = flags | VK_QUERY_RESULT_WAIT_BIT;
        break;
      case GetResultMode::kWithAvailability:
        flags = flags | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
        vector_size += 1;
        break;
    }
  }

  CHECK_VKRESULT(
      vkGetQueryPoolResults(
          static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle(),
          vk_query_pool_,
          0,
          1,
          vector_size * sizeof(Statistic),
          statistics,
          vector_size * sizeof(Statistic),
          flags),
      "Getting Statistics results");
}

}  // namespace gfx
