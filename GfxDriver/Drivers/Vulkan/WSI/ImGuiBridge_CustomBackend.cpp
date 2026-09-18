/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge_CustomBackend.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Resources/HostVisibleBufferWrapper.h"
#include "GfxDriver/Resources/IndexBuffer.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/VertexBuffer.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"

namespace gfx {

ImGuiBridge_CustomBackend::ImGuiBridge_CustomBackend(const DeviceContext& device)
    : device_{device} {}

void ImGuiBridge_CustomBackend::createFontAtlas() {
  int atlas_width = 0;
  int atlas_height = 0;
  unsigned char* font_data = nullptr;
  ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&font_data, &atlas_width, &atlas_height);

  static constexpr TextureSamplerState font_sampler_state = {SamplerFilterMode::kLinear,
                                                             SamplerFilterMode::kLinear,
                                                             SamplerWrapMode::kClampEdge,
                                                             SamplerWrapMode::kClampEdge};

  font_texture_atlas_ = device_.getResourceManager().createTexture("ImGui font atlas",
                                                                   atlas_width,
                                                                   atlas_height,
                                                                   1,
                                                                   PixelFormat::kRGBA8,
                                                                   1,
                                                                   false,
                                                                   ImageUsageBits::kNone,
                                                                   font_sampler_state,
                                                                   font_data);
}

void ImGuiBridge_CustomBackend::init(const RenderPass& render_pass,
                                     const RasterSampleCount num_samples) {
  auto& resource_mgr = device_.getResourceManager();

  createFontAtlas();

  //
  // Create draw material
  //
  auto caches = resource_mgr.getShaderManager().createCacheVectorFromTemplate(
      {{"ImGui/imgui_shader.vert"}, {"ImGui/imgui_shader.frag"}});
  material_ = resource_mgr.createMaterial("ImGui", caches);
  material_->setSamplerAttribute("sTexture", *font_texture_atlas_);
  material_->updateDescriptorSets();

  //
  // Vertex and index buffers
  //
  static constexpr uint32_t kInitialBufferItemCapacity = 2048;

  auto buffer_layout = std::make_shared<InterleavedBufferLayout>();
  buffer_layout->addAttribute("aPos", BufferAttrType::kVec2f);
  buffer_layout->addAttribute("aUV", BufferAttrType::kVec2f);
  buffer_layout->addAttribute("aColor", BufferAttrType::kVec4f);

  auto vertex_buffer_wrapper =
      resource_mgr.createBuffer("ImGui",
                                {BufferType::kVertexBuffer,
                                 kInitialBufferItemCapacity * sizeof(ImDrawVert),
                                 BufferUsageBits::kLayoutBufferBit,
                                 BufferAccessType::kHostVisible});

  auto* vertex_buffer = static_cast<VertexBuffer*>(vertex_buffer_wrapper.get());
  vertex_buffer->updateSubDataWithLayout(
      nullptr, vertex_buffer_wrapper->getNumBytes(), 0, buffer_layout);
  num_verts_ = kInitialBufferItemCapacity;
  vertex_buffer_ =
      resource_mgr.convertToHostVisibleBuffer(std::move(vertex_buffer_wrapper));

  BufferCreateInfo ibo_ci = {};
  ibo_ci.buffer_type = BufferType::kIndexBuffer;
  ibo_ci.access_type = BufferAccessType::kHostVisible;
  ibo_ci.size = 2048 * sizeof(ImDrawIdx);
  ibo_ci.index_buffer_data_type = IndexBufferDataType::kUnsigned16;
  auto index_buffer_wrapper = resource_mgr.createBuffer("ImGui", ibo_ci);
  num_indices_ = kInitialBufferItemCapacity;
  index_buffer_ =
      resource_mgr.convertToHostVisibleBuffer(std::move(index_buffer_wrapper));

  //
  // Pipeline
  //
  PrimitiveAssemblyAttrInfo attr_info{
      {vertex_buffer, buffer_layout},
      {{"aPos", "aPos"}, {"aUV", "aUV"}, {"aColor", "aColor"}}};

  auto primitive_assembly = resource_mgr.createPrimitiveAssembly(
      "ImGui", PrimitiveTopology::kTriangleList, *material_, attr_info);

  PipelineDescriptor pipeline_desc;
  pipeline_desc.setRasterSampleCount(num_samples);
  pipeline_desc.setFaceCullMode(FaceCullMode::kNone);
  pipeline_desc.setBlendFunc(BlendFunc::kSrcAlpha, BlendFunc::kOneMinusSrcAlpha);
  pipeline_desc.setAlphaBlendFunc(BlendFunc::kOne, BlendFunc::kOneMinusSrcAlpha);
  pipeline_desc.getPushConstantRanges().set(
      {PushConstantRange{ShaderStageBits::kVertex, 0, sizeof(PushConstants)}});

  pipeline_ = resource_mgr.createGraphicsPipeline(
      "ImGui", *material_, pipeline_desc, primitive_assembly.get());

  pipeline_->create(render_pass);
}

void ImGuiBridge_CustomBackend::shutdown() {
  auto& resource_mgr = device_.getResourceManager();
  if (pipeline_) {
    resource_mgr.destroyPipeline(std::move(pipeline_));
  }
  if (vertex_buffer_) {
    resource_mgr.destroyHostVisibleBuffer(std::move(vertex_buffer_));
  }
  if (index_buffer_) {
    resource_mgr.destroyHostVisibleBuffer(std::move(index_buffer_));
  }
  if (font_texture_atlas_) {
    resource_mgr.destroyTexture(std::move(font_texture_atlas_));
  }
  material_ = nullptr;
}

void ImGuiBridge_CustomBackend::updateDrawBuffers(const ImDrawData* imgui_draw_data) {
  CHECK(imgui_draw_data);
  auto imgui_vert_count = static_cast<uint32_t>(imgui_draw_data->TotalVtxCount);
  auto imgui_index_count = static_cast<uint32_t>(imgui_draw_data->TotalIdxCount);

  if ((imgui_draw_data->TotalVtxCount == 0) || (imgui_draw_data->TotalIdxCount == 0)) {
    return;
  }

  if (num_verts_ < imgui_vert_count) {
    // TODO: add rebuild() to HostVisibleBufferWrapper!
    vertex_buffer_mapped_ = nullptr;
    auto vertex_buffer_wrapper = vertex_buffer_->releaseSourceBuffer();
    vertex_buffer_ = nullptr;
    uint64_t num_bytes = imgui_vert_count * sizeof(ImDrawVert);
    vertex_buffer_wrapper->rebuild(nullptr, num_bytes);
    vertex_buffer_ = device_.getResourceManager().convertToHostVisibleBuffer(
        std::move(vertex_buffer_wrapper));
    num_verts_ = imgui_vert_count;
  }

  if (num_indices_ < imgui_index_count) {
    index_buffer_mapped_ = nullptr;
    auto index_buffer_wrapper = index_buffer_->releaseSourceBuffer();
    index_buffer_ = nullptr;
    uint64_t num_bytes = imgui_index_count * sizeof(ImDrawIdx);
    index_buffer_wrapper->rebuild(nullptr, num_bytes);
    index_buffer_ = device_.getResourceManager().convertToHostVisibleBuffer(
        std::move(index_buffer_wrapper));
    num_indices_ = imgui_index_count;
  }

  // TODO: persistent mapping
  vertex_buffer_->map(&vertex_buffer_mapped_);
  index_buffer_->map(&index_buffer_mapped_);

  ImDrawVert* dst_vertex = static_cast<ImDrawVert*>(vertex_buffer_mapped_);
  ImDrawIdx* dst_index = static_cast<ImDrawIdx*>(index_buffer_mapped_);

  for (int n = 0; n < imgui_draw_data->CmdListsCount; ++n) {
    const auto* cmd_list = imgui_draw_data->CmdLists[n];
    std::memcpy(dst_vertex,
                cmd_list->VtxBuffer.Data,
                cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
    std::memcpy(dst_index,
                cmd_list->IdxBuffer.Data,
                cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
    dst_vertex += cmd_list->VtxBuffer.Size;
    dst_index += cmd_list->IdxBuffer.Size;
  }

  vertex_buffer_->unmap();
  index_buffer_->unmap();
}

void ImGuiBridge_CustomBackend::draw(ImDrawData* draw_data,
                                     RenderPass& render_pass,
                                     Framebuffer& framebuffer) {
  updateDrawBuffers(draw_data);

  auto const& vbo =
      static_cast<const VertexBuffer&>(vertex_buffer_->getSourceBufferWrapper());
  auto const& ibo =
      static_cast<const IndexBuffer&>(index_buffer_->getSourceBufferWrapper());

  auto& cmd_list = device_.getCommandList();

  ImGuiIO& io = ImGui::GetIO();
  PushConstants.scale = glm::vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
  PushConstants.translate = glm::vec2(-1.0f);
  cmd_list.setPushConstants(*pipeline_,
                            "ImGui",
                            ShaderStageBits::kVertex,
                            &PushConstants,
                            sizeof(PushConstants));

  cmd_list.beginRenderPass(render_pass, framebuffer);

  int32_t global_vertex_offset = 0;
  uint32_t global_index_offset = 0;
  for (int32_t i = 0; i < draw_data->CmdListsCount; ++i) {
    const auto* imgui_cmd_list = draw_data->CmdLists[i];
    for (int32_t j = 0; j < imgui_cmd_list->CmdBuffer.Size; ++j) {
      const auto* cmd = &imgui_cmd_list->CmdBuffer[j];
      cmd_list
          .setScissor(std::max((int32_t)(cmd->ClipRect.x), 0),
                      std::max((int32_t)(cmd->ClipRect.y), 0),
                      (uint32_t)(cmd->ClipRect.z - cmd->ClipRect.x),
                      (uint32_t)(cmd->ClipRect.w - cmd->ClipRect.y))
          .drawIndexed(*pipeline_,
                       vbo,
                       ibo,
                       cmd->ElemCount,
                       cmd->IdxOffset + global_index_offset,
                       cmd->VtxOffset + global_vertex_offset);
    }
    global_vertex_offset += imgui_cmd_list->VtxBuffer.Size;
    global_index_offset += imgui_cmd_list->IdxBuffer.Size;
  }
  cmd_list.endRenderPass().flush("ImGui");
}

}  // namespace gfx
