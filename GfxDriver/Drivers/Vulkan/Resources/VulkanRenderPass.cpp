/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanRenderPass.h"

#include <set>

#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/Resources/AttachmentManager.h"

namespace gfx {

namespace {

struct SubpassData {
  std::vector<VkAttachmentReference> color_attachments;
  VkAttachmentReference depth_stencil_attachment = {};
  std::vector<VkAttachmentReference> input_attachments;
  std::vector<uint32_t> preserve_attachments;
  bool is_any_color_active = false;
  bool is_depth_active = false;
  bool is_any_input_active = false;
  SubpassDependencyBits dependencies = SubpassDependencyBits::kNone;

  bool hasDependencies() const {
    return is_any_color_active || is_depth_active || is_any_input_active ||
           any_bits_set(dependencies);
  }
};

// Get memory access and pipeline stage dependencies for a subpass
// This is conservative in that it will generate access and stage masks for
// any attachments used by the Subpass, even if there is no actual dependency.
// In most cases Pipeline stages returned are only those involved in fragment shading (or
// depth culling prior to fragment shading)
// It will also incorporate any explicit dependencies encoded in the
// SubpassDescriptor's SubpassDependencyBits
std::pair<VkAccessFlags, VkPipelineStageFlags> get_dependency_flags(
    const SubpassData& subpass_data,
    const bool is_src) {
  VkAccessFlags access_flags = {};
  VkPipelineStageFlags stage_flags = {};

  // Access flags

  // Automatic flags based on attachment usage in subpass
  if (is_src) {
    if (subpass_data.is_any_color_active || subpass_data.is_any_input_active) {
      access_flags |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    }
    if (subpass_data.is_depth_active) {
      access_flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
  } else {
    if (subpass_data.is_any_color_active) {
      access_flags |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    }
    if (subpass_data.is_any_input_active) {
      access_flags |= VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    }
    if (subpass_data.is_depth_active) {
      access_flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    }
  }

  // Explicit dependencies set in SubpassDescriptor
  if (any_bits_set(subpass_data.dependencies &
                   SubpassDependencyBits::kFragmentShaderRead)) {
    access_flags |= VK_ACCESS_SHADER_READ_BIT;
  }
  if (any_bits_set(subpass_data.dependencies &
                   SubpassDependencyBits::kFragmentShaderWrite)) {
    access_flags |= VK_ACCESS_SHADER_WRITE_BIT;
  }

  // Pipeline stages

  // Automatic flags based on attachment usage in subpass
  if (subpass_data.is_any_color_active) {
    stage_flags |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  }
  if (subpass_data.is_depth_active) {
    stage_flags |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  }
  if (subpass_data.is_any_input_active) {
    stage_flags |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }

  // Explicit dependencies set in SubpassDescriptor
  if (any_bits_set(subpass_data.dependencies &
                   (SubpassDependencyBits::kFragmentShaderRead |
                    SubpassDependencyBits::kFragmentShaderWrite))) {
    stage_flags |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }

  return {access_flags, stage_flags};
}
}  // namespace

VulkanRenderPass::VulkanRenderPass(
    const DeviceContext& device_ctx,
    std::string_view resource_tracking_string,
    const Framebuffer::Layout& framebuffer_layout,
    ClearBits clear_bits,
    ImageLayout initial_layout,
    ImageLayout final_layout,
    const std::vector<SubpassDescriptor>& subpass_descriptors,
    const AttachmentToImageLayoutMap& unused_attachment_layouts)
    : RenderPass(device_ctx, resource_tracking_string)
    , vk_render_pass_{VK_NULL_HANDLE}
    , final_layout_{final_layout}
    , num_subpasses_{0} {
  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(device_ctx);

  // attachment descs for renderpass createinfo
  std::vector<VkAttachmentDescription> vk_attachment_descs;
  // attachment refs for subpasses
  std::vector<VkAttachmentReference> vk_color_refs;
  VkAttachmentReference vk_depth_stencil_ref = {};
  bool have_depth_stencil = false;

  //
  // Create VkAttachmentDescriptors for each attachment
  // Create VkAttachmentReferences for all attachments to use with subpasses
  // Fill attachment index set to use for preserve attachment init
  //
  auto const& layout_attachments = framebuffer_layout.getAttachmentDescs();

  vk_attachment_descs.reserve(layout_attachments.size());
  std::set<uint32_t> attachment_id_set;

  // Build a set of bindpoints used by subpasses. If the SubpassDesc vector is empty
  // default to all bindpoints in the layout
  std::set<Framebuffer::Attachment> used_bindpoint_set;
  if (subpass_descriptors.empty()) {
    used_bindpoint_set = framebuffer_layout.getAttachmentBindingSet();
  } else {
    for (auto const& subpass_desc : subpass_descriptors) {
      used_bindpoint_set.insert(subpass_desc.attachments.begin(),
                                subpass_desc.attachments.end());
    }
  }

  // Select the loadOp to use if an attachment type is not marked for clearing
  // WARNING!
  // If initial layout is Undefined, and clearing is not set, the attachment
  // may contain garbage data, so this combination should only be used in
  // cases where the entire attachment will be overwritten (eg fullscreen passes)
  VkAttachmentLoadOp load_op = initial_layout == ImageLayout::kUndefined
                                   ? VK_ATTACHMENT_LOAD_OP_DONT_CARE
                                   : VK_ATTACHMENT_LOAD_OP_LOAD;

  std::vector<bool> all_color_attachments_blendable;
  for (auto const& item : layout_attachments) {
    auto attachment_index = framebuffer_layout.getIndex(item.bind_point);
    VkAttachmentDescription vk_desc = {};
    vk_desc.flags = 0;  // optional - alias bit
    vk_desc.format = vk_device.pixelFormatToVkFormat(item.format);
    vk_desc.samples = num_samples_to_vk_sample_flag_bits(item.num_samples);

    if (AttachmentManager::isColorAttachment(item.bind_point)) {
      // Check if the attachment is used by any subpasses
      if (used_bindpoint_set.count(item.bind_point)) {
        vk_desc.initialLayout = image_layout_to_vk_image_layout(initial_layout, true);
        vk_desc.finalLayout = image_layout_to_vk_image_layout(final_layout, true);
        vk_desc.loadOp = any_bits_set(clear_bits & ClearBits::kColor)
                             ? VK_ATTACHMENT_LOAD_OP_CLEAR
                             : load_op;
        vk_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      } else {
        // Attachment unused, check for a layout in the unused attachment layout map
        auto layout_itr = unused_attachment_layouts.find(item.bind_point);
        if (layout_itr != unused_attachment_layouts.end()) {
          // Layout specified, use it for both initial and final layout (this will not
          // work for undefined since we can't use VK_IMAGE_LAYOUT_UNDEFINED for
          // finalLayout). This allows unreferenced passthrough (preserved) attachments
          CHECK_NE(layout_itr->second, ImageLayout::kUndefined)
              << "Layout cannot be kUndefined when specifying unused attachment layouts";
          auto layout = image_layout_to_vk_image_layout(layout_itr->second, true);
          vk_desc.initialLayout = layout;
          vk_desc.finalLayout = layout;
          vk_desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
          vk_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        } else {
          // No layout explicitly specified for unused attachments
          vk_desc.initialLayout = image_layout_to_vk_image_layout(initial_layout, true);
          vk_desc.finalLayout = image_layout_to_vk_image_layout(final_layout, true);
          // If initialLayout is undefined, and the attachment is not used by any subpass,
          // then don't bother storing the result. The attachment will be transitioned to
          // finalLayout since we can't leave it in undefined but likely contains garbage.
          // A subsequent RenderPass will need to handle clearing the attachment
          if (initial_layout == ImageLayout::kUndefined) {
            vk_desc.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            vk_desc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
          } else {
            vk_desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            vk_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
          }
        }
      }

      VkAttachmentReference ref = {};
      ref.attachment = attachment_index;
      ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      vk_color_refs.push_back(ref);
      all_color_attachments_blendable.push_back(is_blendable_pixel_format(item.format));
    } else {
      CHECK_EQ(have_depth_stencil, false)
          << "Only one depth / stencil attachment supported";
      vk_desc.loadOp = any_bits_set(clear_bits & ClearBits::kDepth)
                           ? VK_ATTACHMENT_LOAD_OP_CLEAR
                           : load_op;
      vk_desc.stencilLoadOp = any_bits_set(clear_bits & ClearBits::kStencil)
                                  ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                  : load_op;

      // Set store ops. This determines if the values are retained after the
      // RenderPass completes
      // TODO: toss stencil? Make configurable?
      vk_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      vk_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;

      // TODO: stencil vs depth+stencil
      vk_desc.initialLayout = image_layout_to_vk_image_layout(initial_layout, false);
      vk_desc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

      vk_depth_stencil_ref.attachment = attachment_index;
      vk_depth_stencil_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      have_depth_stencil = true;
    }

    vk_attachment_descs.push_back(vk_desc);
    attachment_id_set.insert(attachment_index);
    if (any_bits_set(clear_bits)) {
      vk_clear_values_.push_back(pixel_format_to_vk_clear_value(item.format));
    }
  }

  //
  // Build SubpassDescriptions
  //
  uint32_t num_subpasses = subpass_descriptors.empty() ? 1 : subpass_descriptors.size();
  std::vector<VkSubpassDescription> vk_subpass_descs;
  vk_subpass_descs.reserve(num_subpasses);
  std::vector<SubpassData> subpass_data(num_subpasses);
  subpass_color_attachment_counts_.reserve(num_subpasses);
  subpass_color_attachments_blendable_.reserve(num_subpasses);

  if (subpass_descriptors.empty()) {
    // No subpass specified so use all attachments for required subpass 0
    VkSubpassDescription desc = {};
    desc.colorAttachmentCount = vk_color_refs.size();
    desc.pColorAttachments = vk_color_refs.data();
    subpass_color_attachment_counts_.push_back(desc.colorAttachmentCount);
    if (have_depth_stencil) {
      desc.pDepthStencilAttachment = &vk_depth_stencil_ref;
    }
    vk_subpass_descs.push_back(desc);
    subpass_color_attachments_blendable_.push_back(all_color_attachments_blendable);
  } else {
    // Iterate subpasses building SubpassData to hold temp vectors
    // and create VkSubpassDescriptions
    for (uint32_t i = 0; i < num_subpasses; ++i) {
      auto const& subpass = subpass_descriptors[i];
      auto& data = subpass_data[i];

      // Copy color refs for modifying (need to mark unused attachments)
      // We always reference all attachments in the subpass. This allows marking
      // arbitrary attachments as unused and also maintains the shader output
      // locations across all subpasses which is less insane
      data.color_attachments = vk_color_refs;
      data.depth_stencil_attachment = vk_depth_stencil_ref;

      // Loop over the AttachmentReferences and check if the bind point is used in the
      // subpass
      std::set<uint32_t> unused_attachments;
      for (auto& ref : data.color_attachments) {
        auto bind_point = layout_attachments[ref.attachment].bind_point;
        if (subpass.attachments.count(bind_point) == 0) {
          unused_attachments.insert(ref.attachment);
          ref.attachment = VK_ATTACHMENT_UNUSED;
        } else {
          data.is_any_color_active = true;
        }
      }

      // Check depth / stencil
      if (have_depth_stencil) {
        auto bind_point = layout_attachments[vk_depth_stencil_ref.attachment].bind_point;
        data.depth_stencil_attachment = vk_depth_stencil_ref;
        if (subpass.attachments.count(bind_point) == 0) {
          unused_attachments.insert(data.depth_stencil_attachment.attachment);
          data.depth_stencil_attachment.attachment = VK_ATTACHMENT_UNUSED;
        } else {
          data.is_depth_active = true;
        }
      }

      // TODO: This can be simplfied now since all subpasses reference all attachments
      // just marking unused ones explicitly
      subpass_color_attachment_counts_.push_back(data.color_attachments.size());
      subpass_color_attachments_blendable_.push_back(all_color_attachments_blendable);

      // Input attachments
      for (auto const& input_attachment : subpass_descriptors[i].input_attachments) {
        VkAttachmentReference input_ref = {};
        input_ref.attachment = framebuffer_layout.getIndex(input_attachment);
        input_ref.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        data.input_attachments.push_back(input_ref);
        data.is_any_input_active = true;
        unused_attachments.erase(input_ref.attachment);
      }

      // Copy explicit dependency flags
      data.dependencies = subpass.dependencies;

      // Populate VkSubpassDescription and add to vector
      VkSubpassDescription desc = {};
      desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      if (!data.color_attachments.empty()) {
        desc.colorAttachmentCount = data.color_attachments.size();
        desc.pColorAttachments = data.color_attachments.data();
      }
      if (have_depth_stencil) {
        desc.pDepthStencilAttachment = &data.depth_stencil_attachment;
      } else {
        desc.pDepthStencilAttachment = nullptr;
      }
      if (!unused_attachments.empty()) {
        data.preserve_attachments =
            std::vector<uint32_t>(unused_attachments.begin(), unused_attachments.end());
        desc.preserveAttachmentCount = data.preserve_attachments.size();
        desc.pPreserveAttachments = data.preserve_attachments.data();
      }
      if (!data.input_attachments.empty()) {
        desc.inputAttachmentCount = data.input_attachments.size();
        desc.pInputAttachments = data.input_attachments.data();
      }
      vk_subpass_descs.push_back(desc);
    }
  }

  //
  // Subpass Dependencies
  //
  std::vector<VkSubpassDependency> dependencies;

  for (uint32_t i = 0; i < num_subpasses; ++i) {
    auto const& this_subpass = subpass_data[i];
    auto const* next_subpass =
        (i >= (num_subpasses - 1)) ? nullptr : &subpass_data[i + 1];

    // Add external if first subpass
    if (i == 0) {
      VkSubpassDependency dep = {};
      dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
      dep.srcSubpass = VK_SUBPASS_EXTERNAL;
      dep.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
      dep.srcAccessMask = 0;
      dep.dstSubpass = 0;
      if (this_subpass.hasDependencies()) {
        std::tie(dep.dstAccessMask, dep.dstStageMask) =
            get_dependency_flags(this_subpass, false);
      } else {
        // If the first subpass has no dependencies
        // Just use top of pipe to create a barrier between the end of the previous
        // renderpass pipe and the start of this pipe
        dep.dstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      }
      dependencies.push_back(dep);
    }

    // Add subpass dependency from this subpass if this subpass has any dependencies
    // (attachments in use or explicit) AND the next subpass also has dependencies OR
    // if it's the last subpass (external)
    if (this_subpass.hasDependencies() &&
        (next_subpass == nullptr || next_subpass->hasDependencies())) {
      VkSubpassDependency dep = {};
      dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
      dep.srcSubpass = i;
      std::tie(dep.srcAccessMask, dep.srcStageMask) =
          get_dependency_flags(this_subpass, true);
      if (next_subpass) {
        dep.dstSubpass = i + 1;
        std::tie(dep.dstAccessMask, dep.dstStageMask) =
            get_dependency_flags(*next_subpass, false);
      } else {  // last subpass
        dep.dstSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dep.dstAccessMask = 0;
      }
      dependencies.push_back(dep);
    }
  }

  // Create VkRenderPass
  VkRenderPassCreateInfo render_pass_ci = {};
  render_pass_ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_ci.flags = 0;  // reserved
  render_pass_ci.attachmentCount = static_cast<uint32_t>(vk_attachment_descs.size());
  render_pass_ci.pAttachments = vk_attachment_descs.data();
  render_pass_ci.subpassCount = static_cast<uint32_t>(vk_subpass_descs.size());
  render_pass_ci.pSubpasses = vk_subpass_descs.data();
  render_pass_ci.dependencyCount = static_cast<uint32_t>(dependencies.size());
  render_pass_ci.pDependencies = dependencies.data();

  auto result = vkCreateRenderPass(
      vk_device.getHandle(), &render_pass_ci, nullptr, &vk_render_pass_);
  CHECK_VKRESULT(result, "Failed to create VkRenderPass");

  // name it
  vk_device.nameVulkanObject(
      VK_OBJECT_TYPE_RENDER_PASS, vk_render_pass_, getTrackingData().origin);
}

VulkanRenderPass::~VulkanRenderPass() {
  cleanupResourceBase();
}

void VulkanRenderPass::updateImageLayouts(Framebuffer& framebuffer) const {
  auto& image_layout_mgr =
      static_cast<VulkanResourceManager*>(&getDeviceContext().getResourceManager())
          ->getImageLayoutManager();
  auto& attachment_mgr = framebuffer.getAttachmentManager();
  auto const& attachments = attachment_mgr.getLayout().getAttachmentDescs();
  for (auto const& item : attachments) {
    auto vk_image =
        static_cast<VulkanTexture*>(attachment_mgr.getAttachmentTexture(item.bind_point))
            ->getImage();
    if (AttachmentManager::isColorAttachment(item.bind_point)) {
      image_layout_mgr.addOrSetLayout(vk_image, final_layout_);
    } else {
      image_layout_mgr.addOrSetLayout(vk_image, ImageLayout::kAttachment);
    }
  }
}

uint32_t VulkanRenderPass::getSubpassColorAttachmentCount(uint32_t subpass_index) const {
  CHECK_LT(subpass_index, subpass_color_attachment_counts_.size());
  return subpass_color_attachment_counts_[subpass_index];
}

bool VulkanRenderPass::subpassColorAttachmentIsBlendable(
    uint32_t subpass_index,
    uint32_t color_attachment_index) const {
  CHECK_LT(subpass_index, subpass_color_attachments_blendable_.size());
  CHECK_LT(color_attachment_index,
           subpass_color_attachments_blendable_[subpass_index].size());
  return subpass_color_attachments_blendable_[subpass_index][color_attachment_index];
}

void VulkanRenderPass::cleanupResourceBase() {
  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  if (vk_render_pass_ != VK_NULL_HANDLE) {
    vkDestroyRenderPass(vk_device.getHandle(), vk_render_pass_, nullptr);
  }
  makeEmpty();
}

void VulkanRenderPass::clearAttachment(Framebuffer& framebuffer,
                                       Framebuffer::Attachment attachment,
                                       VulkanCommandBuffer& cmd_buffer,
                                       const VkRect2D& render_area,
                                       uint32_t subpass_index) {
  //
  // TODO: subpass support
  //
  auto& attachment_mgr = framebuffer.getAttachmentManager();
  auto* texture = attachment_mgr.getAttachmentTexture(attachment);
  CHECK(texture);

  constexpr VkClearColorValue black = {};
  constexpr VkClearDepthStencilValue zero = {};
  VkClearAttachment clear_attachment_info = {};
  clear_attachment_info.clearValue.color = black;
  clear_attachment_info.clearValue.depthStencil = zero;

  if (AttachmentManager::isColorAttachment(attachment)) {
    clear_attachment_info.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear_attachment_info.colorAttachment =
        attachment_mgr.getLayout().getIndex(attachment);
  } else {
    clear_attachment_info.aspectMask =
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  }

  VkClearRect clear_rect = {};
  clear_rect.layerCount = 1;
  // Ensure clear area doesn't exceed the Framebuffer dimensions
  // This occurs due to the +1 width padding required to avoid artifacts
  // when multisampling. Some framebuffers (such as those in tests),
  // do not have this padding
  clear_rect.rect.extent.width =
      std::min(render_area.extent.width, framebuffer.getWidth());
  clear_rect.rect.extent.height =
      std::min(render_area.extent.height, framebuffer.getHeight());
  vkCmdClearAttachments(
      cmd_buffer.getHandle(), 1, &clear_attachment_info, 1, &clear_rect);
}

void VulkanRenderPass::makeEmpty() {
  vk_render_pass_ = VK_NULL_HANDLE;
  num_subpasses_ = 0;
}

}  // namespace gfx
