/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxWSI/WSIAppBase.h"

#include <iostream>
#include <sstream>

#include <imgui/imgui.h>
#include <glm/vec4.hpp>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"
#include "GfxDriver/GfxContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/ShaderCompiler/Library.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "GfxDriver/WSI/WindowSystemIntegration.h"
#include "Logger/Logger.h"
#include "Tests/RenderTests/Utils/AttachmentUtils.h"

using namespace gfx;

class WSITestApp : public WSIAppBase {
 public:
  ~WSITestApp() override = default;

  std::string_view getAppName() const override { return "WSI Test"; }

  void shaderLibraryInit(Library& library) override {
    library.addFromManifestFile("ShaderManifest.json",
                                std::string(RENDER_TESTS_PATH) + "GfxWSI/shaders/");
  }

  void appInit() override {
    // Generate spirv
    auto caches = this->gfx_context_->getShaderManager().createCacheVectorFromTemplate(
        {{"WSITests/fullScreenTriangle.vert"}, {"WSITests/wsiTestGradient.frag"}});
    CHECK_NE(caches.size(), 0u);
    CHECK_NE(caches[0]->getSpirv().size(), 0u);
    CHECK_NE(caches[1]->getSpirv().size(), 0u);

    auto& resource_mgr = device_->getResourceManager();

    // material
    material_ = resource_mgr.createMaterial("Gradient", caches);
    CHECK(material_.get());
    material_->updateDescriptorSets();

    // attachment manager and textures
    auto color_format = wsi_->getWindowPixelFormat();
    render_targets_ =
        build_attachments(resource_mgr,
                          window_width_,
                          window_height_,
                          {{color_format, Framebuffer::Attachment::kColor0}});

    // renderpass and framebuffer
    render_pass_ =
        resource_mgr.createRenderPass("Gradient",
                                      render_targets_.attachment_mgr.getLayout(),
                                      gfx::RenderPass::ClearBits::kAll,
                                      ImageLayout::kUndefined,
                                      ImageLayout::kAttachment);
    CHECK(render_pass_.get());
    framebuffer_ = resource_mgr.createFramebuffer("Gradient",
                                                  *render_pass_,
                                                  render_targets_.attachment_mgr,
                                                  window_width_,
                                                  window_height_,
                                                  1u);
    CHECK(framebuffer_.get());

    // pipeline
    PipelineDescriptor pipeline_desc;
    pipeline_desc.setEnableBlend(true);
    pipeline_ =
        resource_mgr.createGraphicsPipeline("Gradient", *material_, pipeline_desc);
    pipeline_->create(*render_pass_);

    imgui_bridge_->initBackendAndResources(
        framebuffer_->getAttachmentManager().getLayout());

    wsi_->registerEventHandler([this](const WSIEvent& event) { handleWSIEvent(event); });
  }

  void drawUI() {
    //
    // Main panel
    //
    // NOTE: Change ImGui_Once to ImGuiCond_FirstUseEver to restore size/location
    // from the automatically written .ini file
    imgui_bridge_->setNextWindowPos(ImGuiBridge::kTopLeft, 5.0f, ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(640, 512), ImGuiCond_Once);
    ImGui::Begin("WSI Test", &show_ui_);

    if (ImGui::BeginTabBar("Stuff", ImGuiTabBarFlags_None)) {
      if (ImGui::BeginTabItem("Settings")) {
        if (ImGui::CollapsingHeader("UI Stuff", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::Checkbox("Echo Keyboard Events", &echo_keyboard_events_);
          imgui_bridge_->HelpMarker("Print keyboard events in the console");
          ImGui::Checkbox("Echo Mouse Events", &echo_mouse_events_);
          imgui_bridge_->HelpMarker("Print mouse events in the console");
          ImGui::Checkbox("Echo Mouse Position", &echo_mouse_pos_);
          imgui_bridge_->HelpMarker("Print mouse position in the console");
          ImGui::Checkbox("Echo Scroll", &echo_scroll_);
          imgui_bridge_->HelpMarker("Print scroll offset in the console");
          ImGui::Checkbox("Show ImGui Demo", &show_imgui_demo_);
          imgui_bridge_->HelpMarker("Show the ImGui demo window (Hotkey: D)");
          ImGui::Checkbox("Show Overlays", &show_overlays_);
          imgui_bridge_->HelpMarker("Show the test overlays (Hotkey: O)");
        }
        if (ImGui::CollapsingHeader("Colors", ImGuiTreeNodeFlags_DefaultOpen)) {
          static constexpr ImGuiColorEditFlags color_edit_flags =
              ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel |
              ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_AlphaBar;
          uniforms_dirty_ |= ImGui::ColorEdit4(
              "Gradient Color 1", (float*)&grad_color1, color_edit_flags);
          uniforms_dirty_ |= ImGui::ColorEdit4(
              "Gradient Color 2", (float*)&grad_color2, color_edit_flags);
        }
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Memory Summary")) {
        std::stringstream ss;
        device_->getResourceManager().logMemorySummary(ss);
        imgui_bridge_->pushFont(ImGuiBridge::Font::kFixedWidth);
        ImGui::Text("%s", ss.str().c_str());
        imgui_bridge_->popFont();
        ImGui::EndTabItem();
      }

      if (ImGui::BeginTabItem("Help")) {
        static HelpInputTable hotkeys = {{{"", "U", "Show UI"},
                                          {"", "D", "Show ImGui demo"},
                                          {"", "O", "Show overlays"}}};
        ImGui::CollapsingHeader("Hotkeys", ImGuiTreeNodeFlags_Leaf);
        draw_hotkey_help(hotkeys, false, false);
        ImGui::Spacing();
        ImGui::CollapsingHeader("Available Fonts", ImGuiTreeNodeFlags_Leaf);
        imgui_bridge_->pushFont(ImGuiBridge::Font::kPretty);
        ImGui::Text("Pretty font 012345 AaBbCcDd");
        imgui_bridge_->popFont();
        imgui_bridge_->pushFont(ImGuiBridge::Font::kFixedWidth);
        ImGui::Text("Fixed-width font 012345 AaBbCcDd");
        imgui_bridge_->popFont();
        imgui_bridge_->pushFont(ImGuiBridge::Font::kTinySans);
        ImGui::Text("Tiny-sans font 012345 AaBbCcDd");
        imgui_bridge_->popFont();
        imgui_bridge_->pushFont(ImGuiBridge::Font::kTinyRegular);
        ImGui::Text("Tiny-regular font 012345 AaBbCcDd");
        imgui_bridge_->popFont();
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
    ImGui::End();

    //
    // Overlays
    //
    static ImGuiBridge::Overlay static_overlay(*imgui_bridge_,
                                               "Static Overlay",
                                               ImGuiBridge::Font::kFixedWidth,
                                               ImGuiBridge::WindowPos::kTopRight,
                                               show_overlays_,
                                               []() { ImGui::Text("Just some text"); });

    auto cursor_pos_overlay_cb = [this]() {
      ImGui::Text("CursorPosition");
      ImGui::Text("X: %d", cursor_x_);
      ImGui::Text("Y: %d", cursor_y_);
    };
    static ImGuiBridge::Overlay cursor_pos_overlay(*imgui_bridge_,
                                                   "Another Overlay",
                                                   ImGuiBridge::Font::kPretty,
                                                   ImGuiBridge::WindowPos::kBottomRight,
                                                   show_overlays_,
                                                   cursor_pos_overlay_cb);

    if (show_overlays_) {
      static_overlay.draw();
      cursor_pos_overlay.draw();
    }
  }

  void render() override {
    // Update colors
    if (uniforms_dirty_) {
      material_->setUniformAttribute("gradColor1", grad_color1);
      material_->setUniformAttribute("gradColor2", grad_color2);
      uniforms_dirty_ = false;
    }

    // Draw gradient
    auto& cmd_list = device_->getCommandList();
    cmd_list.beginRenderPass(*render_pass_, *framebuffer_)
        .drawFullscreen(*pipeline_)
        .endRenderPass()
        .flush("Draw");

    if (show_ui_ || show_imgui_demo_) {
      imgui_bridge_->beginRecording();
      // Main UI window
      if (show_ui_) {
        drawUI();
        if (uniforms_dirty_) {
          setRenderDirty();
        }
      }

      // ImGui demo window
      if (show_imgui_demo_) {
        ImGui::ShowDemoWindow(&show_imgui_demo_);
      }
      imgui_bridge_->endRecordingAndDraw(*framebuffer_);
    }

    // Copy color to swapchain and present
    wsi_->copyAndPresentTexture(*render_targets_.textures[0]);
  }

  void appShutdown() override {
    auto& resource_mgr = device_->getResourceManager();
    resource_mgr.destroyPipeline(std::move(pipeline_));
    resource_mgr.destroyRenderPass(std::move(render_pass_));
    material_ = nullptr;
  }

  void handleWindowResizeEvent(const WSIWindowResizeEvent& event) {
    std::array<float, 2> dimensions{static_cast<float>(event.width()),
                                    static_cast<float>(event.height())};
    material_->setUniformAttribute("imageSize", dimensions);
  }

  void handleKeyboardEvent(const WSIKeyboardEvent& event) {
    auto action = event.action();
    auto key = event.key();
    auto const* key_name = event.keyName();
    // Handle keys we care about first
    if (action == WSIKeyboardAction::kPress) {
      switch (key) {
        case WSIKeyboardKey::kU:
          show_ui_ = !show_ui_;
          break;
        case WSIKeyboardKey::kD:
          show_imgui_demo_ = !show_imgui_demo_;
          break;
        case WSIKeyboardKey::kO:
          show_overlays_ = !show_overlays_;
          break;
        default:
          break;
      }
    }

    // Echo all keys to console
    if (echo_keyboard_events_) {
      if (key == WSIKeyboardKey::kUnknown) {
        std::cout << "Key: Unknown";
      } else if (key_name) {
        std::cout << "Key: '" << key_name << "'";
      } else {
        std::cout << "WSI code: " << static_cast<int>(key);
      }
      std::cout << "  action: " << action << "  mods: " << event.modBits() << std::endl;
    }
  }

  void handleMouseButtonEvent(const WSIMouseButtonEvent& event) {
    if (echo_mouse_events_) {
      std::cout << "Button: " << event.button() << "  action: " << event.action()
                << "  mods: " << event.modBits() << std::endl;
    }
  }

  void handleCursorEvent(const WSIMouseCursorEvent& event) {
    auto x = event.x();
    auto y = event.y();
    // Store the cursor position for the overlay
    cursor_x_ = static_cast<int>(x);
    cursor_y_ = static_cast<int>(y);
    if (echo_mouse_pos_) {
      std::cout << "Cursor (" << x << ", " << y << ")" << std::endl;
    }
  }

  void handleScrollEvent(const WSIScrollEvent& event) {
    if (echo_scroll_) {
      std::cout << "Scroll (" << event.x() << ", " << event.y() << ")" << std::endl;
    }
  }

  void handleWSIEvent(const WSIEvent& event) {
    using e = WSIEvent::Type;
    switch (event.type()) {
      case e::kWindowResize:
        handleWindowResizeEvent(static_cast<const WSIWindowResizeEvent&>(event));
        break;
      case e::kMouseButton:
        handleMouseButtonEvent(static_cast<const WSIMouseButtonEvent&>(event));
        break;
      case e::kCursor:
        handleCursorEvent(static_cast<const WSIMouseCursorEvent&>(event));
        break;
      case e::kScroll:
        handleScrollEvent(static_cast<const WSIScrollEvent&>(event));
        break;
      case e::kKeyboard:
        handleKeyboardEvent(static_cast<const WSIKeyboardEvent&>(event));
        break;
    }
  }

 private:
  // Test resources
  resource_ptr<RenderPass> render_pass_;
  resource_ptr<GraphicsPipeline> pipeline_;
  std::unique_ptr<Material> material_;

  // internal state
  bool show_ui_{true};
  bool echo_keyboard_events_{true};
  bool echo_mouse_events_{false};
  bool echo_mouse_pos_{false};
  bool echo_scroll_{false};
  bool show_overlays_{false};
  bool show_imgui_demo_{false};
  int cursor_x_{0};
  int cursor_y_{0};

  bool uniforms_dirty_{true};
  glm::vec4 grad_color1{1, 0, 0, 1};
  glm::vec4 grad_color2{0, 1, 0, 1};
};

DECLARE_WSI_APP_MAIN(WSITestApp)
