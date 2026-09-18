/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"
#include "GfxDriver/GfxContext.h"
#include "GfxDriver/ShaderCompiler/Library.h"
#include "GfxDriver/WSI/WindowSystemIntegration.h"
#include "Tests/RenderTests/Utils/AttachmentUtils.h"

namespace gfx {

//
// main declaration
// use DECLARE_WSI_APP_MAIN passing derived class to macro
//
template <class AppClass>
int AppMain(int argc, char* argv[]) {
  auto app = std::make_unique<AppClass>();
  if (!app->init(argc, argv)) {
    return EXIT_FAILURE;
  }
  if (!app->run()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#define DECLARE_WSI_APP_MAIN(AppClass)    \
  int main(int argc, char* argv[]) {      \
    return AppMain<AppClass>(argc, argv); \
  }

//
// WSIAppOptions class
//
static constexpr uint32_t kWindowWidth = 1024;
static constexpr uint32_t kWindowHeight = 768;

class WSIAppOptions {
 public:
  WSIAppOptions();
  ~WSIAppOptions();

  // Create a new options_description and set defaults to current class values
  void setOptions();

  // Get the current options_description to add new options or parse command line
  boost::program_options::options_description& getOptions() const;

  uint32_t window_width{kWindowWidth};
  uint32_t window_height{kWindowHeight};
  uint32_t num_samples{1};

 private:
  std::unique_ptr<boost::program_options::options_description> options_;
};

//
// WSIAppBase class
//
class WSIAppBase {
 public:
  enum class RenderFrequency { kContinuous, kWhenDirty };

  WSIAppBase();
  virtual ~WSIAppBase();

  bool init(int argc, char* argv[]);

  // Run the application main loop (called after appInit)
  virtual bool run();

 protected:
  // Required and used to set the window title (required)
  virtual std::string_view getAppName() const = 0;

  // Print supplemental help. Will be output prior to program option help
  virtual void appPrintHelp(std::ostream& os) {}

  // Create GfxContext
  virtual void gfxInit();

  // Modify or add program options
  virtual void addOrModifyProgramOptions(WSIAppOptions& options) {}

  // Called from gfxInit just prior to creating the GfxContext
  virtual void shaderLibraryInit(Library& library) {}

  // Initialize all resources needed to render
  virtual void appInit() {}
  // Destroy resources needed to render
  virtual void appShutdown() {}

  // Render and present an image (required)
  virtual void render() = 0;

  // Set the dirty flag to trigger a redraw
  void setRenderDirty();

  // Set flag to trigger app exit
  void setShouldExit();

  // System components
  std::unique_ptr<GfxContext> gfx_context_;
  std::unique_ptr<DeviceContext> device_;
  WindowSystemIntegration* wsi_;
  std::unique_ptr<ImGuiBridge> imgui_bridge_;

  // Render targets and framebuffer
  resource_ptr<Framebuffer> framebuffer_;
  RenderTargetReturn render_targets_;
  gfx::RasterSampleCount raster_sample_count_;
  uint32_t num_samples_;

  // Common state
  uint32_t window_width_;
  uint32_t window_height_;
  RenderFrequency render_frequency_;
  bool show_resolution_in_title_;

 private:
  std::unique_ptr<WSIAppOptions> options_;

  // Render loop state
  bool render_is_dirty_;
  bool should_exit_;

  // WSI Event
  void baseHandleWSIEvent(const WSIEvent& event);

  void shutdown();
};

}  // namespace gfx
