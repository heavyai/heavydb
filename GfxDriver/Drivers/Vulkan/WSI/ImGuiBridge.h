/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>

#include <imgui/imgui.h>
#include <boost/program_options.hpp>

#include "GfxDriver/Resources/Framebuffer.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Types.h"

// Custom backend uses GfxDriver API to handle drawing (Experimental)
// Default backend uses ImGui's Vulkan backend to handle drawing
#define USE_CUSTOM_IMGUI_BACKEND false

#if USE_CUSTOM_IMGUI_BACKEND
#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge_CustomBackend.h"
#else
#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge_DefaultBackend.h"
#endif

namespace gfx {

// ImGuiOptions class
// Handles all program options for the bridge
class ImGuiOptions {
 public:
  ImGuiOptions();

  void parseCommandLine(int argc, char const* const* argv);
  boost::program_options::options_description const& getOptions() const;

  bool use_font_scale{false};
  float font_scale{1.0f};

 private:
  std::unique_ptr<boost::program_options::options_description> options_;
};

//
// ImGuiBridge class
//
class ImGuiBridge {
 public:
  static constexpr float kFontGlyphSize = 15.0f;    // glyph size in pixels
  static constexpr float kHelpMarkerWidth = 35.0f;  // width of hover text for help `?`

  // Font selection for pushFont
  enum Font { kPretty, kFixedWidth, kTinySans, kTinyRegular, kCOUNT };

  // Window positions for setNextWindowPos and DoWindowPosPopup
  enum WindowPos { kCustom = -1, kTopLeft = 0, kTopRight, kBottomLeft, kBottomRight };

  // ini_file should include the full path or it will be written to cwd (not recommended)
  // Pass an empty string to disable ini file support entirely
  explicit ImGuiBridge(const std::string& ini_file,
                       const DeviceContext& device,
                       const WindowSystemIntegration& wsi,
                       ImGuiOptions& options);
  ~ImGuiBridge();

  // Initialize the backend, either ImGui's Vulkan backend or GfxDriver (custom)
  // framebuffer_layout required for RenderPass creation
  void initBackendAndResources(const Framebuffer::Layout& framebuffer_layout);
  void destroyBackendAndResources();

  void setDisplaySize(uint32_t width, uint32_t height);

  void beginRecording();
  void endRecordingAndDraw(Framebuffer& framebuffer);

  // Facades for fonts
  // Automatically handles missing font files
  // DO NOT use ImGui API directly
  void pushFont(ImGuiBridge::Font font);
  void popFont();

  // Set the location of the next window using corner enum and distance in pixels
  void setNextWindowPos(WindowPos corner, float distance, ImGuiCond condition = 0);

  // Show right-click window pos popup window (typically used for overlays)
  void doWindowPosPopup(bool& is_parent_open, WindowPos& pos);

  // Show a question mark (?) with tooltip
  // on_same_line will place the marker to the right of the previous widget
  // spacing adds additional padding when on_same_line = true (units = pixels)
  void HelpMarker(const std::string& text, bool on_same_line = true, int spacing = 0);

  // Display a combo box populated with strings
  // T must be convertable to int (typically an enum)
  // selected is the program state variable for the selected item
  template <typename T>
  static bool doComboBox(const std::string& label,
                         const std::vector<std::string>& strings,
                         T& selected);

  // Scope guard style helper to automatically disable subsequent widgets within the scope
  // if b is false
  class ScopedDisable {
    bool b_;

   public:
    explicit ScopedDisable(bool b) : b_{b} {
      if (!b_) {
        ImGui::BeginDisabled();
      }
    }
    ~ScopedDisable() {
      if (!b_) {
        ImGui::EndDisabled();
      }
    }
  };

  // Overlay helper class
  class Overlay {
   public:
    using Callback = std::function<void()>;
    explicit Overlay(ImGuiBridge& bridge,
                     const std::string& name,
                     Font font,
                     WindowPos initial_pos,
                     bool& is_open,
                     Callback callback);
    void draw();

   private:
    ImGuiBridge& bridge_;
    std::string name_;
    Font font_;
    WindowPos position_;
    bool& is_open_;
    Callback callback_;
  };

 private:
  bool is_initialized_;
  std::string ini_filename_;
  const WindowSystemIntegration& wsi_;
  const DeviceContext& device_;
  std::array<ImFont*, Font::kCOUNT> fonts_;
  resource_ptr<RenderPass> render_pass_;
  float font_glyph_size_;
  bool fonts_loaded_;
  bool is_recording_;

#if USE_CUSTOM_IMGUI_BACKEND
  std::unique_ptr<ImGuiBridge_CustomBackend> bridge_backend_;
#else
  std::unique_ptr<ImGuiBridge_DefaultBackend> bridge_backend_;
#endif
};

template <typename T>
bool ImGuiBridge::doComboBox(const std::string& label,
                             const std::vector<std::string>& strings,
                             T& selected) {
  bool rtn = false;
  size_t item_current = static_cast<size_t>(selected);
  if (ImGui::BeginCombo(label.c_str(), strings[item_current].c_str())) {
    for (size_t n = 0; n < strings.size(); ++n) {
      const bool is_selected = (item_current == n);
      if (ImGui::Selectable(strings[n].c_str())) {
        selected = static_cast<T>(n);
        rtn = true;
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  return rtn;
}

struct HelpInputTableItem {
  std::string mods;
  std::string action;
  std::string description;
};

using HelpInputTableGroup = std::vector<HelpInputTableItem>;
using HelpInputTable = std::vector<HelpInputTableGroup>;

void draw_modkey_help_text();
void draw_hotkey_help_table(const HelpInputTableGroup& hotkeys,
                            bool do_header,
                            bool do_mods_column);
void draw_mouse_action_help_table(const HelpInputTableGroup& actions,
                                  bool do_header,
                                  bool do_mods_column);

void draw_hotkey_help(const HelpInputTable& table, bool do_header, bool do_mods_column);

}  // namespace gfx
