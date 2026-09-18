/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Pipeline/Pipeline.h"

#include <vulkan/vulkan.h>

#include "Logger/Logger.h"

namespace gfx {

class VulkanGraphicsPipeline : public GraphicsPipeline {
 public:
  explicit VulkanGraphicsPipeline(const DeviceContext& device_ctx,
                                  std::string_view resource_tracking_string,
                                  const Material& material,
                                  const PipelineDescriptor& pipeline_descriptor,
                                  const PrimitiveAssembly* primitive_assembly);
  ~VulkanGraphicsPipeline() override;

  ResourceHandle getResourceHandle() const override {
    return reinterpret_cast<ResourceHandle>(pipeline_);
  }
  ResourceHandle getPipelineHandle(uint32_t specialization_id) const override {
    // Allowing this method to be called with 0u simplifies generic usages
    // such as VulkanCommandExecutor::bindPipeline
    CHECK_EQ(specialization_id, 0u)
        << "Specialization unsupported for graphics pipelines";
    return reinterpret_cast<ResourceHandle>(pipeline_);
  }

  DynamicStateBits getDynamicStateBits() const override;
  ResourceHandle getLayout() const override;
  void create(const RenderPass& render_pass_handle) override;

 private:
  VkPipeline pipeline_;
  VkPipelineLayout pipeline_layout_;

  void cleanupResourceBase() override;
  void makeEmpty() override;
};

}  // namespace gfx
