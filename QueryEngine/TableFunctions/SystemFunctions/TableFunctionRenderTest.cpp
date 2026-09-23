/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/TableFunctions/SystemFunctions/TableFunctionRenderTest.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/GfxContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/BufferLayout.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/Resources/VertexBuffer.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "Shared/scope.h"

// copied from Tests/RenderTests/Utils/AttachmentUtils

namespace LocalAttachmentUtils {

using RenderTargetDesc = std::pair<gfx::PixelFormat, gfx::Framebuffer::Attachment>;
struct RenderTargetReturn {
  gfx::AttachmentManager attachment_mgr;
  std::vector<gfx::resource_ptr<gfx::Texture>> textures;
};

RenderTargetReturn build_attachments(
    gfx::ResourceManager& resource_mgr,
    const uint32_t width,
    const uint32_t height,
    const std::vector<RenderTargetDesc>& target_descs,
    const std::string& texture_name_base = "",
    const uint32_t num_samples = 1,
    const gfx::ImageUsageBits extra_color_usage_bits = gfx::ImageUsageBits::kNone) {
  gfx::AttachmentManager attachment_mgr;
  std::vector<gfx::resource_ptr<gfx::Texture>> textures;

  std::string name_base = texture_name_base.empty() ? "Framebuffer" : texture_name_base;
  for (auto const& [pixel_format, attachment] : target_descs) {
    gfx::ImageUsageBits extra_usage_bits =
        gfx::AttachmentManager::isColorAttachment(attachment)
            ? gfx::ImageUsageBits::kColorAttachmentBit | extra_color_usage_bits
            : gfx::ImageUsageBits::kDepthStencilAttachmentBit;
    std::stringstream ss;
    ss << name_base << " " << attachment;
    textures.emplace_back(resource_mgr.createTexture(
        ss.str(),
        width,
        height,
        1,
        pixel_format,
        num_samples,
        false,
        extra_usage_bits,
        gfx::get_default_sampler_state_for_format(pixel_format)));
    CHECK(textures.back());
    attachment_mgr.setAttachment(attachment, textures.back().get());
  }
  attachment_mgr.freezeLayout();
  return {attachment_mgr, std::move(textures)};
}

}  // namespace LocalAttachmentUtils

int TableFunctionRenderTest(TableFunctionManager& mgr) {
  auto& execution_context = *mgr.getGfxCommandExecutionContext();
  // Ensure the VkCommandPool is reset once execution completes. We could skip this for
  // normal transient TableFunctionManager instances, but it's almost zero cost
  ScopeGuard cleanup_guard([&execution_context]() { execution_context.resetPool(); });

  // get resource manager
  auto& resource_mgr = execution_context.getDeviceContext().getResourceManager();

  // create material
  auto caches = mgr.getGfxContext()->getShaderManager().createCacheVectorFromTemplate(
      {{"TableFunctions/tableFunctionTest.vert"},
       {"TableFunctions/tableFunctionTest.frag"}});
  auto material = resource_mgr.createMaterial("TableFunctionTest", caches);
  CHECK_NE(material.get(), nullptr);

  static constexpr uint32_t kRenderWidth = 320;
  static constexpr uint32_t kRenderHeight = 240;

  // attachment manager and textures
  auto [attachment_mgr, textures] = LocalAttachmentUtils::build_attachments(
      resource_mgr,
      kRenderWidth,
      kRenderHeight,
      {{gfx::PixelFormat::kRGBA8, gfx::Framebuffer::Attachment::kColor0}});

  // renderpass and framebuffer
  auto render_pass = resource_mgr.createRenderPass("Test",
                                                   attachment_mgr.getLayout(),
                                                   gfx::RenderPass::ClearBits::kAll,
                                                   gfx::ImageLayout::kUndefined,
                                                   gfx::ImageLayout::kAttachment);
  CHECK_NE(render_pass.get(), nullptr);
  auto framebuffer = resource_mgr.createFramebuffer(
      "Test", *render_pass, attachment_mgr, kRenderWidth, kRenderHeight, 1u);
  CHECK_NE(framebuffer.get(), nullptr);

  auto buffer_layout = std::make_shared<gfx::InterleavedBufferLayout>();
  CHECK_NE(buffer_layout.get(), nullptr);
  buffer_layout->addAttribute("in_position", gfx::BufferAttrType::kVec2f);
  buffer_layout->addAttribute("in_color", gfx::BufferAttrType::kVec3f);

  // clang-format off
    std::array<float, 15> vertex_data{-0.5f,  0.5f,  1.0f, 0.0f, 0.0f,
                                        0.0f, -0.5f,  0.0f, 1.0f, 0.0f,
                                        0.5f,  0.5f,  0.0f, 0.0f, 1.0f};
  // clang-format on

  // vertex buffer, layout, and attr map
  auto vertex_buffer_wrapper =
      resource_mgr.createBuffer("Test",
                                {gfx::BufferType::kVertexBuffer,
                                 vertex_data.size() * sizeof(float),
                                 gfx::BufferUsageBits::kLayoutBufferBit});
  CHECK_NE(vertex_buffer_wrapper.get(), nullptr);

  auto* vertex_buffer = static_cast<gfx::VertexBuffer*>(vertex_buffer_wrapper.get());
  vertex_buffer->updateSubDataWithLayout(
      vertex_data.data(), vertex_data.size() * sizeof(float), 0, buffer_layout);

  gfx::PrimitiveAssemblyAttrInfo attr_info{
      {vertex_buffer, buffer_layout},
      {{"in_position", "in_position"}, {"in_color", "in_color"}}};

  // pipeline
  auto primitive_assembly = resource_mgr.createPrimitiveAssembly(
      "Test", gfx::PrimitiveTopology::kTriangleList, *material, attr_info);
  gfx::PipelineDescriptor pipeline_desc;
  auto pipeline = resource_mgr.createGraphicsPipeline(
      "Test", *material, pipeline_desc, primitive_assembly.get());

  pipeline->create(*render_pass);

  // render via command list
  execution_context.getCommandExecutor().setDefaultViewportAndRenderArea(
      0, 0, kRenderWidth, kRenderHeight);
  execution_context.getCommandList()
      .beginRenderPass(*render_pass, *framebuffer)
      .drawVertices(*pipeline, *vertex_buffer, 3)
      .endRenderPass()
      .flush("Test");

  // extract image
  std::vector<uint8_t> pixels(kRenderWidth * kRenderHeight * 4);
  textures[0]->getPixels(kRenderWidth,
                         kRenderHeight,
                         1,
                         gfx::PixelFormat::kRGBA8,
                         pixels.data(),
                         pixels.size());

  // count non-transparent pixels
  auto const* p = pixels.data();
  int count = 0;
  for (uint32_t i = 0; i < kRenderWidth * kRenderHeight; i++, p += 4) {
    if (p[3] > 0) {
      count++;
    }
  }

  // destroy resources
  resource_mgr.destroyPipeline(std::move(pipeline));
  material = nullptr;
  resource_mgr.destroyBuffer(std::move(vertex_buffer_wrapper));
  resource_mgr.destroyFramebuffer(std::move(framebuffer));
  for (auto& texture : textures) {
    resource_mgr.destroyTexture(std::move(texture));
  }
  resource_mgr.destroyRenderPass(std::move(render_pass));

  // result
  return count;
}
