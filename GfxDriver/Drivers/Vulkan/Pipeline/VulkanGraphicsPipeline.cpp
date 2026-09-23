/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanGraphicsPipeline.h"

#include "GfxDriver/Drivers/Vulkan/Pipeline/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanMaterial.h"
#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanPrimitiveAssembly.h"
#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanRenderPass.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanShaderModule.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"

namespace gfx {

VulkanGraphicsPipeline::VulkanGraphicsPipeline(
    const DeviceContext& device_ctx,
    std::string_view resource_tracking_string,
    const Material& material,
    const PipelineDescriptor& pipeline_descriptor,
    const PrimitiveAssembly* primitive_assembly)
    : GraphicsPipeline(device_ctx,
                       resource_tracking_string,
                       material,
                       pipeline_descriptor,
                       primitive_assembly)
    , pipeline_{VK_NULL_HANDLE}
    , pipeline_layout_{VK_NULL_HANDLE} {}

VulkanGraphicsPipeline::~VulkanGraphicsPipeline() {
  cleanupResource();
}

GraphicsPipeline::DynamicStateBits VulkanGraphicsPipeline::getDynamicStateBits() const {
  // All pipelines currently require dynamic viewports
  return kViewport;
}

ResourceHandle VulkanGraphicsPipeline::getLayout() const {
  return reinterpret_cast<ResourceHandle>(pipeline_layout_);
}

void VulkanGraphicsPipeline::cleanupResourceBase() {
  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  if (pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(vk_device.getHandle(), pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
  }
  if (pipeline_layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(vk_device.getHandle(), pipeline_layout_, nullptr);
    pipeline_layout_ = VK_NULL_HANDLE;
  }

  makeEmpty();
}

void VulkanGraphicsPipeline::makeEmpty() {}

void VulkanGraphicsPipeline::create(const RenderPass& render_pass) {
  // conversions from gfx enums to VK enums
  static constexpr std::array<VkCullModeFlagBits, 4> face_cull_mode_to_vk = {
      VK_CULL_MODE_NONE,
      VK_CULL_MODE_FRONT_BIT,
      VK_CULL_MODE_BACK_BIT,
      VK_CULL_MODE_FRONT_AND_BACK};
  static constexpr std::array<VkCompareOp, 2> depth_func_to_vk = {
      VK_COMPARE_OP_LESS_OR_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL};
  static constexpr std::array<VkBlendFactor, 12> blend_func_to_vk = {
      VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ZERO,
      VK_BLEND_FACTOR_SRC_COLOR,
      VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
      VK_BLEND_FACTOR_DST_COLOR,
      VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
      VK_BLEND_FACTOR_SRC_ALPHA,
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      VK_BLEND_FACTOR_DST_ALPHA,
      VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
      VK_BLEND_FACTOR_CONSTANT_ALPHA,
      VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA};
  static constexpr std::array<VkBlendOp, 2> blend_equation_to_vk = {VK_BLEND_OP_ADD,
                                                                    VK_BLEND_OP_MAX};
  static constexpr std::array<VkStencilOp, 3> stencil_op_to_vk = {
      VK_STENCIL_OP_KEEP, VK_STENCIL_OP_INVERT, VK_STENCIL_OP_ZERO};
  static constexpr std::array<VkCompareOp, 3> stencil_func_to_vk = {
      VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_EQUAL, VK_COMPARE_OP_NOT_EQUAL};

  auto const& pd = pipeline_descriptor_;

  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  // Destroy existing pipeline
  if (pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(vk_device.getHandle(), pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
  }
  // Destroy existing pipeline layout
  if (pipeline_layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(vk_device.getHandle(), pipeline_layout_, nullptr);
    pipeline_layout_ = VK_NULL_HANDLE;
  }

  auto const& vk_material = static_cast<const VulkanMaterial&>(material_);

  auto const* vk_primitive_assembly =
      static_cast<const VulkanPrimitiveAssembly*>(primitive_assembly_);

  VkResult result = VK_SUCCESS;

  //
  // color blend and blend attachment state
  //

  auto const& vk_render_pass = static_cast<const VulkanRenderPass&>(render_pass);
  auto num_color_attachments =
      vk_render_pass.getSubpassColorAttachmentCount(pd.getSubpassIndex());

  std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_states(
      num_color_attachments);

  for (uint32_t attachment_index = 0; attachment_index < num_color_attachments;
       ++attachment_index) {
    auto& color_blend_attachment_state = color_blend_attachment_states[attachment_index];

    color_blend_attachment_state = {};
    if (vk_material.hasFragmentShaderOutputLocation(attachment_index)) {
      color_blend_attachment_state.colorWriteMask =
          pd.getEnableColorWrites()
              ? (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
              : 0U;
    } else {
      color_blend_attachment_state.colorWriteMask = 0U;
    }
    if (pd.getEnableBlend() && vk_render_pass.subpassColorAttachmentIsBlendable(
                                   pd.getSubpassIndex(), attachment_index)) {
      color_blend_attachment_state.blendEnable = VK_TRUE;
      color_blend_attachment_state.srcColorBlendFactor =
          blend_func_to_vk[static_cast<int>(pd.getBlendFuncSrc())];
      color_blend_attachment_state.dstColorBlendFactor =
          blend_func_to_vk[static_cast<int>(pd.getBlendFuncDst())];
      color_blend_attachment_state.colorBlendOp =
          blend_equation_to_vk[static_cast<int>(pd.getBlendEquation())];
      if (pd.hasAlphaBlendFunc()) {
        color_blend_attachment_state.srcAlphaBlendFactor =
            blend_func_to_vk[static_cast<int>(pd.getAlphaBlendFuncSrc())];
        color_blend_attachment_state.dstAlphaBlendFactor =
            blend_func_to_vk[static_cast<int>(pd.getAlphaBlendFuncDst())];
      } else {
        color_blend_attachment_state.srcAlphaBlendFactor =
            blend_func_to_vk[static_cast<int>(pd.getBlendFuncSrc())];
        color_blend_attachment_state.dstAlphaBlendFactor =
            blend_func_to_vk[static_cast<int>(pd.getBlendFuncDst())];
      }
      if (pd.hasAlphaBlendEquation()) {
        color_blend_attachment_state.alphaBlendOp =
            blend_equation_to_vk[static_cast<int>(pd.getAlphaBlendEquation())];
      } else {
        color_blend_attachment_state.alphaBlendOp =
            blend_equation_to_vk[static_cast<int>(pd.getBlendEquation())];
      }
    } else {
      color_blend_attachment_state.blendEnable = VK_FALSE;
    }
  }

  VkPipelineColorBlendStateCreateInfo color_blend_state_ci = {};
  color_blend_state_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend_state_ci.logicOpEnable = VK_FALSE;    // logic blend, not used
  color_blend_state_ci.logicOp = VK_LOGIC_OP_COPY;  // logic blend, not used
  color_blend_state_ci.attachmentCount = num_color_attachments;
  color_blend_state_ci.pAttachments = color_blend_attachment_states.data();
  auto const& blend_color = pd.getBlendColor();
  color_blend_state_ci.blendConstants[0] = blend_color[0];  // unless dynamic?
  color_blend_state_ci.blendConstants[1] = blend_color[1];
  color_blend_state_ci.blendConstants[2] = blend_color[2];
  color_blend_state_ci.blendConstants[3] = blend_color[3];

  VkStencilOpState stencil_op_state = {};
  stencil_op_state.failOp =
      stencil_op_to_vk[static_cast<int>(pd.getStencilOpStencilFail())];
  stencil_op_state.passOp =
      stencil_op_to_vk[static_cast<int>(pd.getStencilOpDepthPass())];
  stencil_op_state.depthFailOp =
      stencil_op_to_vk[static_cast<int>(pd.getStencilOpDepthFail())];
  stencil_op_state.compareOp = stencil_func_to_vk[static_cast<int>(pd.getStencilFunc())];
  stencil_op_state.compareMask = pd.getStencilFuncMask();
  stencil_op_state.writeMask = pd.getStencilMask();
  stencil_op_state.reference = pd.getStencilFuncRef();

  //
  // depth stencil state
  //

  VkPipelineDepthStencilStateCreateInfo depth_stencil_state_ci = {};
  depth_stencil_state_ci.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil_state_ci.depthTestEnable = pd.getEnableDepthTest();
  depth_stencil_state_ci.depthWriteEnable = pd.getEnableDepthWrites();
  depth_stencil_state_ci.depthCompareOp =
      depth_func_to_vk[static_cast<int>(pd.getDepthFunc())];
  depth_stencil_state_ci.depthBoundsTestEnable = VK_FALSE;  // default
  depth_stencil_state_ci.minDepthBounds = 0.0f;             // default
  depth_stencil_state_ci.maxDepthBounds = 1.0f;             // default
  depth_stencil_state_ci.stencilTestEnable = pd.getEnableStencilTest();
  depth_stencil_state_ci.front = stencil_op_state;
  depth_stencil_state_ci.back = stencil_op_state;

  //
  // multisample state
  //

  VkPipelineMultisampleStateCreateInfo multisample_state_ci = {};
  multisample_state_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample_state_ci.rasterizationSamples =
      raster_sample_count_to_vk_sample_flag_bits(pd.getRasterSampleCount());
  multisample_state_ci.sampleShadingEnable = VK_FALSE;
  multisample_state_ci.minSampleShading = 0.0f;
  multisample_state_ci.pSampleMask = nullptr;  // enable all samples
  multisample_state_ci.alphaToCoverageEnable = VK_FALSE;
  multisample_state_ci.alphaToOneEnable = VK_FALSE;

  // Not yet handled:
  //   enable_program_point_size (no equivalent, always on in Vulkan, size must be >= 1.0)
  //   point_sprite_coord_origin (no equivalent, hopefully defaults to lower-left anyway)

  //
  // shader stages
  //
  std::vector<VkPipelineShaderStageCreateInfo> shader_stage_cis;
  auto const& shader_modules = vk_material.getShaderModuleMap();
  for (auto const& [shader_stage, shader_module] : shader_modules) {
    VkPipelineShaderStageCreateInfo shader_stage_ci = {};
    shader_stage_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_ci.stage = shader_stage_to_vk_shader_stage_flag(shader_stage);
    shader_stage_ci.module =
        reinterpret_cast<VkShaderModule>(shader_module->getResourceHandle());
    shader_stage_ci.pName = vk_material.getEntryPoint(shader_stage);
    shader_stage_cis.push_back(shader_stage_ci);
  }

  //
  // viewport state
  //
  // actual viewport and scissor will be set dynamically
  // but you still need the basic setup (counts etc.)
  //

  VkViewport static_viewport = {};
  static_viewport.x = 0.0f;
  static_viewport.y = 0.0f;
  static_viewport.width = 0.0f;
  static_viewport.height = 0.0f;
  static_viewport.minDepth = 0.0f;
  static_viewport.maxDepth = 1.0f;

  VkRect2D static_scissor = {};
  static_scissor.offset = {0, 0};
  static_scissor.extent = {0, 0};

  VkPipelineViewportStateCreateInfo viewport_state_ci = {};

  viewport_state_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state_ci.viewportCount = 1;
  viewport_state_ci.pViewports = &static_viewport;
  viewport_state_ci.scissorCount = 1;
  viewport_state_ci.pScissors = &static_scissor;

  //
  // rasterization state
  //

  // stolen from vksandbox
  // check valid

  VkPipelineRasterizationStateCreateInfo rasterization_state_ci = {};
  rasterization_state_ci.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization_state_ci.depthClampEnable = VK_FALSE;
  rasterization_state_ci.rasterizerDiscardEnable = VK_FALSE;
  rasterization_state_ci.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization_state_ci.lineWidth = 1.0f;
  rasterization_state_ci.cullMode =
      face_cull_mode_to_vk[static_cast<int>(pd.getFaceCullMode())];
  rasterization_state_ci.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterization_state_ci.depthBiasEnable = VK_FALSE;
  rasterization_state_ci.depthBiasConstantFactor = 0.0f;  // optional
  rasterization_state_ci.depthBiasClamp = 0.0f;           // optional
  rasterization_state_ci.depthBiasSlopeFactor = 0.0f;     // optional

  //
  // pipeline layout
  //

  VkDescriptorSetLayout vk_descriptor_set_layout = vk_material.getDescriptorSetLayout();

  VkPipelineLayoutCreateInfo pipeline_layout_ci = {};
  pipeline_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_ci.setLayoutCount = 1;
  pipeline_layout_ci.pSetLayouts = &vk_descriptor_set_layout;
  auto push_constant_ranges =
      push_constant_ranges_to_vk_push_constant_ranges(pd.getPushConstantRanges());
  pipeline_layout_ci.pushConstantRangeCount = push_constant_ranges.size();
  pipeline_layout_ci.pPushConstantRanges = push_constant_ranges.data();

  result = vkCreatePipelineLayout(
      vk_device.getHandle(), &pipeline_layout_ci, nullptr, &pipeline_layout_);
  CHECK_VKRESULT(result, "creating PipelineLayout");

  // name it
  vk_device.nameVulkanObject(
      VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline_layout_, getTrackingData().origin);

  //
  // pipeline dynamic state
  //

  constexpr static std::array<VkDynamicState, 2> pipeline_dynamic_states = {
      VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  // add more as required

  VkPipelineDynamicStateCreateInfo pipeline_dynamic_state_ci = {};
  pipeline_dynamic_state_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  pipeline_dynamic_state_ci.dynamicStateCount = pipeline_dynamic_states.size();
  pipeline_dynamic_state_ci.pDynamicStates = pipeline_dynamic_states.data();

  //
  // create the pipeline
  //

  // default vertex input state if no PrimitiveAssembly attached (for quad rendering)
  constexpr static VkPipelineVertexInputStateCreateInfo empty_vertex_input_state_ci{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      nullptr,
      0,
      0,
      nullptr,
      0,
      nullptr};

  constexpr static VkPipelineInputAssemblyStateCreateInfo empty_input_assembly_state_ci{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      nullptr,
      0,
      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      VK_FALSE};

  VkGraphicsPipelineCreateInfo pipeline_ci = {};
  pipeline_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_ci.stageCount = shader_stage_cis.size();
  pipeline_ci.pStages = shader_stage_cis.data();
  pipeline_ci.pVertexInputState =
      vk_primitive_assembly ? vk_primitive_assembly->getPipelineVertexInputStateCI()
                            : &empty_vertex_input_state_ci;
  pipeline_ci.pInputAssemblyState =
      vk_primitive_assembly ? vk_primitive_assembly->getPipelineInputAssemblyStateCI()
                            : &empty_input_assembly_state_ci;
  pipeline_ci.pViewportState = &viewport_state_ci;
  pipeline_ci.pRasterizationState = &rasterization_state_ci;
  pipeline_ci.pMultisampleState = &multisample_state_ci;
  pipeline_ci.pDepthStencilState = &depth_stencil_state_ci;
  pipeline_ci.pColorBlendState = &color_blend_state_ci;
  pipeline_ci.pDynamicState = &pipeline_dynamic_state_ci;
  pipeline_ci.layout = pipeline_layout_;
  pipeline_ci.renderPass =
      reinterpret_cast<VkRenderPass>(render_pass.getResourceHandle());
  pipeline_ci.subpass = pd.getSubpassIndex();
  pipeline_ci.basePipelineHandle = VK_NULL_HANDLE;  // optional
  pipeline_ci.basePipelineIndex = -1;               // optional

  result = vkCreateGraphicsPipelines(
      vk_device.getHandle(), VK_NULL_HANDLE, 1, &pipeline_ci, nullptr, &pipeline_);

  if (result != VK_SUCCESS) {
    vkDestroyPipelineLayout(vk_device.getHandle(), pipeline_layout_, nullptr);
    pipeline_layout_ = VK_NULL_HANDLE;
    CHECK_VKRESULT(result, "creating Graphics Pipeline");
  }

  // name it
  vk_device.nameVulkanObject(
      VK_OBJECT_TYPE_PIPELINE, pipeline_, getTrackingData().origin);

  //
  // done!
  //

  setUsable();
}

}  // namespace gfx
