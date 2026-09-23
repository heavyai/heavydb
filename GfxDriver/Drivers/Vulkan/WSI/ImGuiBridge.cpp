/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"

#include <filesystem>

#include <GLFW/glfw3.h>
#include <imgui/backends/imgui_impl_glfw.h>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/WSI/VulkanWSI_GLFW.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "Logger/Logger.h"

namespace gfx {

namespace po = boost::program_options;

ImGuiOptions::ImGuiOptions() {
  options_ = std::make_unique<po::options_description>("ImGui");
  options_->add_options()("font-scale",
                          po::value<float>(&font_scale),
                          "Override automatic font scale factor (1.0 = no scaling)");
}

po::options_description const& ImGuiOptions::getOptions() const {
  return *options_;
}

void ImGuiOptions::parseCommandLine(int argc, char const* const* argv) {
  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(*options_).allow_unregistered().run(),
      vm);

  if (vm.count("font-scale")) {
    use_font_scale = true;
  }

  po::notify(vm);
}

ImGuiBridge::ImGuiBridge(const std::string& ini_file,
                         const DeviceContext& device,
                         const WindowSystemIntegration& wsi,
                         ImGuiOptions& options)
    : is_initialized_{false}
    , wsi_{wsi}
    , device_{device}
    , font_glyph_size_{kFontGlyphSize}
    , fonts_loaded_{false}
    , is_recording_{false} {
  if (!ini_file.empty()) {
    ini_filename_ = ini_file + "_UI.ini";
  }

  // Scale font
  // Use program option if set
  // otherwise scale using window content scale (typically 1.0 or 2.0)
  float ui_scale = 1.0f;
  if (options.use_font_scale) {
    ui_scale = options.font_scale;
  } else {
    auto [w_scale, h_scale] = wsi.getWindowContentScale();
    ui_scale = std::max(w_scale, h_scale);
  }
  font_glyph_size_ *= ui_scale;

  // Create an ImGui context
  ImGui::CreateContext();
  ImGui::GetStyle().ScaleAllSizes(ui_scale);
}

ImGuiBridge::~ImGuiBridge() {
  destroyBackendAndResources();
  if (ImGui::GetCurrentContext()) {
    ImGui::DestroyContext();
  }
}

void ImGuiBridge::initBackendAndResources(const Framebuffer::Layout& framebuffer_layout) {
  CHECK(!is_initialized_);

  // Initialize the draw backend
#if USE_CUSTOM_IMGUI_BACKEND
  bridge_backend_ = std::make_unique<ImGuiBridge_CustomBackend>(device_);
#else
  bridge_backend_ = std::make_unique<ImGuiBridge_DefaultBackend>(device_, wsi_);
#endif  // !USE_CUSTOM_IMGUI_BACKEND

  // Initialize the ImGui GLFW backend
  // Automatically captures GLFW input events
  ImGui_ImplGlfw_InitForVulkan(static_cast<const VulkanWSI_GLFW&>(wsi_).getWindow(),
                               true);

  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = ini_filename_.c_str();
  io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
  io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
#ifdef IMGUI_HAS_DOCK
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
#endif

  // Setup style (TODO: program option for Light vs Dark)
  ImGui::StyleColorsDark();
  //
  // Load Fonts
  //
  // Check if font path exists, if not use the built in default font
  // If font path exists but font files missing then CHECK
  std::string font_path(IMGUI_FONT_PATH);
  if (std::filesystem::exists(font_path)) {
    auto add_font = [this, &font_path, &io](const std::string& filename,
                                            float scale = 1.0f) -> ImFont* {
      auto font_file = font_path + filename;
      CHECK(std::filesystem::exists(font_file))
          << "Could not find font file: " << font_file;
      return io.Fonts->AddFontFromFileTTF(font_file.c_str(), font_glyph_size_ * scale);
    };

    fonts_[Font::kPretty] = add_font("Roboto-Medium.ttf");
    fonts_[Font::kFixedWidth] = add_font("Cousine-Regular.ttf");
    fonts_[Font::kTinySans] = add_font("DroidSans.ttf", 0.8f);
    fonts_[Font::kTinyRegular] = add_font("Karla-Regular.ttf", 0.8f);
    fonts_loaded_ = true;
  } else {
    LOG(WARNING) << "Using simple font, font path not found: " << font_path;
    ImFontConfig default_font_config;
    default_font_config.SizePixels = font_glyph_size_;
    io.Fonts->AddFontDefault(&default_font_config);
  }

  //
  // Renderpass (used by imgui and custom backends)
  //
  render_pass_ =
      device_.getResourceManager().createRenderPass("ImGui",
                                                    framebuffer_layout,
                                                    gfx::RenderPass::ClearBits::kNone,
                                                    ImageLayout::kAttachment,
                                                    ImageLayout::kAttachment);

  bridge_backend_->init(
      *render_pass_,
      framebuffer_layout.getAttachmentDesc(Framebuffer::Attachment::kColor0).num_samples);
  is_initialized_ = true;
}

void ImGuiBridge::destroyBackendAndResources() {
  if (bridge_backend_) {
    bridge_backend_->shutdown();
  }
  if (render_pass_) {
    device_.getResourceManager().destroyRenderPass(std::move(render_pass_));
  }
  ImGui_ImplGlfw_Shutdown();
}

void ImGuiBridge::beginRecording() {
  CHECK(is_initialized_);
  CHECK(!is_recording_)
      << "Error recording imGui commands, beginRecording called before endRecording";
  is_recording_ = true;

  bridge_backend_->newFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void ImGuiBridge::endRecordingAndDraw(Framebuffer& framebuffer) {
  CHECK(is_initialized_);
  CHECK(is_recording_)
      << "Error recording imGui commands, endRecording called before beginRecording";
  is_recording_ = false;
  ImGui::Render();
  ImDrawData* draw_data = ImGui::GetDrawData();
  if (draw_data->CmdListsCount > 0) {
    bridge_backend_->draw(draw_data, *render_pass_, framebuffer);
  }
}

void ImGuiBridge::pushFont(ImGuiBridge::Font font) {
  if (fonts_loaded_) {
    ImGui::PushFont(fonts_[font]);
  }
}

void ImGuiBridge::popFont() {
  if (fonts_loaded_) {
    ImGui::PopFont();
  }
}

void ImGuiBridge::setDisplaySize(uint32_t width, uint32_t height) {
  ImGui::GetIO().DisplaySize = ImVec2(width, height);
}

void ImGuiBridge::setNextWindowPos(WindowPos corner,
                                   float distance,
                                   ImGuiCond condition /* = 0 */) {
  ImVec2 window_pos =
      ImVec2((corner & 1) ? ImGui::GetIO().DisplaySize.x - distance : distance,
             (corner & 2) ? ImGui::GetIO().DisplaySize.y - distance : distance);
  ImVec2 window_pos_pivot =
      ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
  if (corner != -1) {
    ImGui::SetNextWindowPos(window_pos, condition, window_pos_pivot);
  }
}

void ImGuiBridge::doWindowPosPopup(bool& is_parent_open, WindowPos& pos) {
  pushFont(Font::kPretty);
  if (ImGui::BeginPopupContextWindow()) {
    if (ImGui::MenuItem("Custom", NULL, pos == WindowPos::kCustom)) {
      pos = WindowPos::kCustom;
    }
    if (ImGui::MenuItem("Top-left", NULL, pos == WindowPos::kTopLeft)) {
      pos = WindowPos::kTopLeft;
    }
    if (ImGui::MenuItem("Top-right", NULL, pos == WindowPos::kTopRight)) {
      pos = WindowPos::kTopRight;
    }
    if (ImGui::MenuItem("Bottom-left", NULL, pos == WindowPos::kBottomLeft)) {
      pos = WindowPos::kBottomLeft;
    }
    if (ImGui::MenuItem("Bottom-right", NULL, pos == WindowPos::kBottomRight)) {
      pos = WindowPos::kBottomRight;
    }
    if (is_parent_open && ImGui::MenuItem("Close")) {
      is_parent_open = false;
    }
    ImGui::EndPopup();
  }
  popFont();
}

void ImGuiBridge::HelpMarker(const std::string& text, bool on_same_line, int spacing) {
  if (on_same_line) {
    ImGui::SameLine(0, spacing);
  }
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * kHelpMarkerWidth);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

ImGuiBridge::Overlay::Overlay(ImGuiBridge& bridge,
                              const std::string& name,
                              Font font,
                              WindowPos initial_position,
                              bool& is_open,
                              Callback callback)
    : bridge_{bridge}
    , name_{name}
    , font_{font}
    , position_{initial_position}
    , is_open_{is_open}
    , callback_{callback} {}

void ImGuiBridge::Overlay::draw() {
  static constexpr ImGuiWindowFlags kOverlayWindowFlags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
      ImGuiWindowFlags_NoNav;

  bridge_.setNextWindowPos(position_, 10.0f, ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(0.7f);
  if (ImGui::Begin(name_.c_str(), &is_open_, kOverlayWindowFlags)) {
    bridge_.pushFont(font_);
    callback_();
    bridge_.doWindowPosPopup(is_open_, position_);
    bridge_.popFont();
  }
  ImGui::End();
}

void draw_input_help_table(const HelpInputTableGroup& group,
                           const std::string& table_name,
                           const std::string& input_class,
                           float input_column_width,
                           bool do_header,
                           bool do_mods_column) {
  int num_columns = do_mods_column ? 3 : 2;
  ImGui::BeginTable(table_name.c_str(), num_columns, ImGuiTableFlags_BordersInnerV);
  if (do_mods_column) {
    ImGui::TableSetupColumn("Mods", ImGuiTableColumnFlags_WidthFixed, 80.f);
  }
  ImGui::TableSetupColumn(
      input_class.c_str(), ImGuiTableColumnFlags_WidthFixed, input_column_width);
  ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);
  if (do_header) {
    ImGui::TableHeadersRow();
  }
  for (const auto& [mods, input, action] : group) {
    int column = 0;
    ImGui::TableNextRow();
    if (do_mods_column) {
      ImGui::TableSetColumnIndex(column++);
      ImGui::Text("%s", mods.c_str());
    }
    ImGui::TableSetColumnIndex(column++);
    ImGui::Text("%s", input.c_str());
    ImGui::TableSetColumnIndex(column++);
    ImGui::Text("%s", action.c_str());
  }
  ImGui::EndTable();
}

void draw_modkey_help_text() {
  static std::string modkey_help_text = "Mod keys: Sh=Shift C=Ctrl A=Alt Sp=Super";
  ImGui::Text("%s", modkey_help_text.c_str());
}

void draw_hotkey_help_table(const HelpInputTableGroup& hotkeys,
                            bool do_header,
                            bool do_mods_column) {
  draw_input_help_table(hotkeys, "Hotkeys", "Key", 40.f, do_header, do_mods_column);
}

void draw_mouse_action_help_table(const HelpInputTableGroup& actions,
                                  bool do_header,
                                  bool do_mods_column) {
  draw_input_help_table(
      actions, "Mouse Actions", "Mouse Action", 160.f, do_header, do_mods_column);
}

void draw_hotkey_help(const HelpInputTable& table, bool do_header, bool do_mods_column) {
  bool is_first_group = true;
  for (auto itr = table.begin(); itr != table.end(); ++itr) {
    draw_input_help_table(
        *itr, "Hotkeys", "Key", 40.f, do_header && is_first_group, do_mods_column);
    if (itr + 1 != table.end()) {
      ImGui::Spacing();
    }
    is_first_group = false;
  }
}

}  // namespace gfx
