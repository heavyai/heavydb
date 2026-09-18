/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

#include <vulkan/vulkan.h>

#include "GfxDriver/Commands/QueryPool.h"
#include "GfxDriver/Pipeline/Pipeline.h"

namespace gfx {

class VulkanQueryPool : public QueryPool {
 public:
  explicit VulkanQueryPool(const DeviceContext& device_context,
                           std::string_view resource_tracking_string,
                           Type type,
                           std::optional<Pipeline::Type> pipeline_type,
                           std::optional<uint32_t> size);
  ~VulkanQueryPool() override;

  Type getType() const override { return type_; }
  uint32_t getNumQueries() const override { return num_queries_; }

  // Resource methods
  ResourceHandle getResourceHandle() const override {
    return reinterpret_cast<ResourceHandle>(vk_query_pool_);
  }
  // QueryPool methods
  void reset(std::optional<uint32_t> first_query = std::nullopt,
             std::optional<uint32_t> query_count = std::nullopt) override;

  std::vector<Timestamp> getTimestampResults(
      std::optional<uint32_t> first_query = std::nullopt,
      std::optional<uint32_t> query_count = std::nullopt,
      std::optional<GetResultMode> get_result_mode = std::nullopt) override;

  uint64_t timestampToMicroseconds(Timestamp timestamp) override;

  GraphicsPipelineStatistics getGraphicsPipelineStatistics(
      std::optional<GetResultMode> get_result_mode = std::nullopt) override;
  ComputePipelineStatistics getComputePipelineStatistics(
      std::optional<GetResultMode> get_result_mode = std::nullopt) override;

  OcclusionStatistics getOcclusionStatistics(
      std::optional<GetResultMode> get_result_mode = std::nullopt) override;

 protected:
  void cleanupResourceBase() override;
  void makeEmpty() override;

  void getStatisticsInternal(Statistic* statistics,
                             std::optional<GetResultMode> get_result_mode);

 private:
  Type type_;
  std::optional<Pipeline::Type> pipeline_type_;
  uint32_t num_queries_;
  uint32_t num_results_;
  VkQueryPool vk_query_pool_;
  double timestamp_period_;
};

}  // namespace gfx
