/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_map>
#include <vector>

#include <boost/noncopyable.hpp>

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/Types.h"
#include "QueryRenderer/GlobalRenderContext.h"

namespace QueryRenderer {

class SeparateMultiSamplesPass : boost::noncopyable {
 public:
  enum class SourceFramebuffer { kRender, kComp };
  enum class OutputUsage { kShaderRead, kTransferSrc };

  SeparateMultiSamplesPass(const GlobalRenderContext& global_ctx, bool enable_api_export);
  ~SeparateMultiSamplesPass();

  void prepareRenderTargets(uint32_t width, uint32_t height);
  void postPrepareRenderTargets();

  const gfx::Texture& getTexture(GpuId gpu_id) const;

  void runPass(GpuId gpu_id,
               gfx::CommandList& command_list,
               SourceFramebuffer source_fbo,
               OutputUsage output_usage,
               uint32_t sample_index,
               const std::vector<gfx::SemaphoreHandle>& wait_semaphores,
               const std::vector<gfx::SemaphoreHandle>& signal_semaphores);

 private:
  class GpuData {
   public:
    explicit GpuData(const gfx::DeviceContext& device_ctx);
    ~GpuData();

    void prepareRenderTargets(uint32_t width, uint32_t height, bool enable_api_export);
    void destroyResources();
    void postPrepareRenderTargets(const GlobalRenderContext& global_ctx,
                                  const GpuId gpu_id);

   private:
    const gfx::DeviceContext& device_ctx_;

    // Separate multi-sample support
    gfx::resource_ptr<gfx::RenderPass> render_pass_shader_read_;
    gfx::resource_ptr<gfx::RenderPass> render_pass_transfer_src_;
    gfx::resource_ptr<gfx::Framebuffer> fbo_;
    gfx::AttachmentManager attachment_mgr_;
    gfx::resource_ptr<gfx::Texture> texture_;

    struct Resources {
      gfx::MaterialUqPtr material;
      gfx::resource_ptr<gfx::GraphicsPipeline> pipeline;
      gfx::Framebuffer* framebuffer = nullptr;
    };
    Resources resources_rndr_, resources_comp_;

    friend class SeparateMultiSamplesPass;
  };

  const GlobalRenderContext& global_ctx_;
  std::unordered_map<GpuId, GpuData> gpu_data_map_;

  bool enable_api_export_;

  bool initialized_;

  gfx::PipelineDescriptorUqPtr pipeline_descriptor_;

  // Initialize the gpu data map and build the material resources
  void initBaseResources();
  void destroyResources();
};

}  // namespace QueryRenderer
