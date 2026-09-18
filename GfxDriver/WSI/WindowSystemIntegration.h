/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/Types.h"
#include "GfxDriver/WSI/Events.h"

namespace gfx {

//
// WindowSystemcreateInfo struct
//
// Pass to GfxContext during system start to ensure Window System Integration
// layer is created and available for use
//
struct WindowSystemCreateInfo {
  std::string window_name;
  bool is_visible_on_create{false};
  bool show_resolution_in_title{true};
  uint32_t width{0u};
  uint32_t height{0u};
};

//
// WindowSystemIntegration API
//
// Serves as an interface to the underlying windowing system for presentation
// of images on window surfaces
//
// Since input actions such as keyboard and mouse events are typically coupled
// with window integration, this API hosts those as well
//
class WindowSystemIntegration {
 public:
  WindowSystemIntegration();
  virtual ~WindowSystemIntegration();

  // shutdown destroy any device dependent resources and should be called
  // prior to tearing down logical devices
  virtual void shutdown() = 0;

  // Get the pixel format for the display surface. Used to set Framebuffer format
  // to match the display
  virtual PixelFormat getWindowPixelFormat() const = 0;

  // Get the x and y content scaling for the window (useful with hiDPI displays)
  virtual std::pair<float, float> getWindowContentScale() const = 0;

  // Set window properties
  virtual void setWindowSize(uint32_t width, uint32_t height) = 0;
  virtual void setWindowVisibility(bool is_visible) const = 0;
  virtual void setWindowTitle(const std::string& title) = 0;
  //
  // Presentation
  //
  // Acquire the next swapchain image, copy the Texture contents, and present using the
  // presentation device Will perform a multi-sample resolve if necessary
  virtual void copyAndPresentTexture(const Texture& texture) = 0;

  //
  // Window events
  //
  // pollEvents processes any waiting events and returns immediately, even if none are
  // pending. Used for continuous rendering
  virtual void pollEvents() const = 0;

  // waitEvents will wait until an event is received before returning, saving cpu cycles.
  // Useful for apps that do not need to constantly update
  virtual void waitEvents(float timeout_seconds) const = 0;

  // Determine if the window should close (close button)
  virtual bool windowShouldClose() const = 0;

  //
  // Event handlers (keyboard, mouse, etc)
  //

  // Register an event handler
  WSIEventHandlerID registerEventHandler(WSIEventHandler handler);

  // Unregister event handler
  // If called during notifyEvent, removal will be deferred until
  // notifyEvent completes
  void unregisterEventHandler(WSIEventHandlerID id);

  // Handlers are called in the order they are registered
  void notifyEvent(const WSIEvent& event);

 protected:
  bool hasEventHandlers() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gfx
