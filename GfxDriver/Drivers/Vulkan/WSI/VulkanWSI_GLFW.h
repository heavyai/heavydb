/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Drivers/Vulkan/WSI/VulkanWSI.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace gfx {

class VulkanWSI_GLFW : public VulkanWSI {
 public:
  explicit VulkanWSI_GLFW(const WindowSystemCreateInfo& wsi_ci);
  ~VulkanWSI_GLFW() override;

  std::vector<const char*> getRequiredInstanceExtensions() const override;
  VkSurfaceKHR createSurface(VkInstance instance) override;
  void destroySurface() override;

  VkSurfaceKHR getSurface() const override;

  std::pair<uint32_t, uint32_t> getWindowSize() const override;
  std::pair<float, float> getWindowContentScale() const override;

  void setWindowSize(uint32_t width, uint32_t height) override;
  void setWindowVisibility(bool is_visible) const override;
  void setWindowTitle(const std::string& title) override;

  bool windowShouldClose() const override;
  void pollEvents() const override;
  void waitEvents(float timeout_seconds) const override;

  GLFWwindow* getWindow() const { return glfw_window_; }

 private:
  GLFWwindow* glfw_window_;
  VkSurfaceKHR vk_surface_;
  std::string name_;
  bool show_resolution_in_title_;

  // GFLW callbacks
  void registerGLFWCallbacks();

  static void glfw_onResizeCallback(GLFWwindow* glfw_window, int width, int height);
  static void glfw_onKeyPressCallback(GLFWwindow* glfw_window,
                                      int key,
                                      int scancode,
                                      int action,
                                      int modifiers);
  static void glfw_onMouseButtonCallback(GLFWwindow* glfw_window,
                                         int button,
                                         int action,
                                         int mods);
  static void glfw_onCursorPosCallback(GLFWwindow* glfw_window, double x, double y);
  static void glfw_onScrollCallback(GLFWwindow* glfw_window, double x, double y);
};

}  // namespace gfx
