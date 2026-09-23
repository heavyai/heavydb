/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxWSI/WSIAppBase.h"

#include <boost/program_options.hpp>
#include <iostream>

#include "GfxDriver/DriverInstance.h"

namespace gfx {

WSIAppOptions::WSIAppOptions() {
  setOptions();
}

WSIAppOptions::~WSIAppOptions() {}

boost::program_options::options_description& WSIAppOptions::getOptions() const {
  return *options_;
}

void WSIAppOptions::setOptions() {
  namespace po = boost::program_options;
  // Create new options_description
  // Will overwrite existing options
  options_ = std::make_unique<po::options_description>("Options");
  options_->add_options()("help", "print help message");
  options_->add_options()("render-width,w",
                          po::value<unsigned>(&window_width)->default_value(window_width),
                          "Initial image width");
  options_->add_options()(
      "render-height,h",
      po::value<unsigned>(&window_height)->default_value(window_height),
      "Initial image height");
  options_->add_options()("samples,s",
                          po::value<unsigned>(&num_samples)->default_value(num_samples),
                          "Multisampling sample count");
}

WSIAppBase::WSIAppBase()
    : wsi_{nullptr}
    , raster_sample_count_{gfx::RasterSampleCount::k1}
    , window_width_{kWindowWidth}
    , window_height_{kWindowHeight}
    , render_frequency_{RenderFrequency::kContinuous}
    , show_resolution_in_title_{true}
    , render_is_dirty_{true}
    , should_exit_{false} {}

WSIAppBase::~WSIAppBase() {
  appShutdown();
  imgui_bridge_ = nullptr;
  gfx_context_->getPrimaryDriver().destroyDeviceContext(std::move(device_));
}

bool WSIAppBase::init(int argc, char* argv[]) {
  namespace po = boost::program_options;

  // Create options and set defaults
  options_ = std::make_unique<WSIAppOptions>();
  options_->setOptions();

  // App options
  addOrModifyProgramOptions(*options_);

  // Get final app options desc
  auto& desc = options_->getOptions();

  // ImGui options
  ImGuiOptions imgui_options;
  desc.add(imgui_options.getOptions());

  // Logger options
  logger::LogOptions log_options(argv[0]);
  // Change logger default options:
  // log INFO to console
  log_options.severity_clog_ = logger::Severity::INFO;
  // stderr only (no file logging)
  log_options.max_files_ = 0;  // stderr only by default
  log_options.set_options();

  // Add log options to program options for --help
  desc.add(log_options.get_options());

  // Process command line
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

    // explicitly parse imgui options to allow vm.count checks in ImGuiOptions
    imgui_options.parseCommandLine(argc, argv);

    if (vm.count("help")) {
      appPrintHelp(std::cout);
      std::cout << desc << "\n";
      return false;
    }
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return false;
  }

  logger::init(log_options);

  // Copy parsed options into class vars
  window_width_ = options_->window_width;
  window_height_ = options_->window_height;
  num_samples_ = options_->num_samples;
  raster_sample_count_ = value_to_raster_sample_count(num_samples_);

  try {
    LOG(INFO) << "Window settings W: " << window_width_ << "  H: " << window_height_
              << "  samples: " << options_->num_samples;

    // Create GfxContext and create a DeviceContext
    gfxInit();

    // TODO: ImGui .ini file support with program options
    // Pass an empty string for now to prevent any .ini usage
    imgui_bridge_ =
        std::make_unique<ImGuiBridge>(std::string(), *device_, *wsi_, imgui_options);

    // Init app resources
    appInit();
    CHECK(framebuffer_);

    // Broadcast initial window sizing
    wsi_->notifyEvent(WSIWindowResizeEvent(window_width_, window_height_));

    device_->getCommandExecutor().setDefaultViewportAndRenderArea(
        0, 0, window_width_, window_height_);
    imgui_bridge_->setDisplaySize(window_width_, window_height_);

    // Make window visible
    wsi_->setWindowVisibility(true);
  } catch (std::exception& e) {
    LOG(ERROR) << e.what() << std::endl;
    return false;
  }

  return true;
}

void WSIAppBase::gfxInit() {
  WindowSystemCreateInfo wsi_ci = {};
  wsi_ci.is_visible_on_create = false;
  wsi_ci.width = window_width_;
  wsi_ci.height = window_height_;
  wsi_ci.window_name = getAppName();
  wsi_ci.show_resolution_in_title = show_resolution_in_title_;

  auto library = std::make_unique<Library>();
  shaderLibraryInit(*library);

  gfx_context_ = std::make_unique<GfxContext>(
      DriverType::kVulkan, GfxUsage::kCudaInterop, std::move(library), 60000u, &wsi_ci);

  wsi_ = gfx_context_->getWSI();
  auto const& driver = gfx_context_->getPrimaryDriver();
  auto uuids = driver.getUUIDs();
  // TODO: ensure it can present
  device_ = driver.createDeviceContext(uuids[0], 0);

  // register action handler
  wsi_->registerEventHandler(
      [this](const WSIEvent& event) { baseHandleWSIEvent(event); });
}

void WSIAppBase::baseHandleWSIEvent(const WSIEvent& event) {
  switch (event.type()) {
    case WSIEvent::Type::kWindowResize: {
      auto re = static_cast<const WSIWindowResizeEvent&>(event);
      window_width_ = re.width();
      window_height_ = re.height();
      render_is_dirty_ = true;
      device_->getCommandExecutor().setDefaultViewportAndRenderArea(
          0, 0, window_width_, window_height_);
      imgui_bridge_->setDisplaySize(window_width_, window_height_);
      framebuffer_->resize(window_width_, window_height_);
    } break;
    case WSIEvent::Type::kKeyboard: {
      auto ke = static_cast<const WSIKeyboardEvent&>(event);
      if (ke.key() == WSIKeyboardKey::kEscape ||
          (ke.key() == WSIKeyboardKey::kC &&
           any_bits_set(ke.modBits() & WSIKeyboardModBits::kControl))) {
        should_exit_ = true;
      }
    } break;
    default:
      return;
  }
}

void WSIAppBase::setRenderDirty() {
  render_is_dirty_ = true;
}

void WSIAppBase::setShouldExit() {
  should_exit_ = true;
}

bool WSIAppBase::run() {
  try {
    // Loop until app window closes
    while (!wsi_->windowShouldClose() && !should_exit_) {
      switch (render_frequency_) {
        case RenderFrequency::kContinuous:
          render();
          wsi_->pollEvents();
          break;

        case RenderFrequency::kWhenDirty:
          if (render_is_dirty_) {
            render();
            render_is_dirty_ = false;
          }
          wsi_->waitEvents(1.0f);
          break;
      }
      device_->resetCommandPools();
    }

    // Wait until device is done
    device_->waitIdle();

    // Tear everything down
    shutdown();
  } catch (std::exception& e) {
    LOG(ERROR) << e.what();
    return false;
  }
  return true;
}

void WSIAppBase::shutdown() {
  appShutdown();
  auto& resource_mgr = device_->getResourceManager();
  resource_mgr.destroyFramebuffer(std::move(framebuffer_));
  for (auto& texture : render_targets_.textures) {
    resource_mgr.destroyTexture(std::move(texture));
  }
}

}  // namespace gfx
