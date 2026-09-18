/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/WSI/VulkanWSI_GLFW.h"

#include <sstream>
#include <unordered_map>

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/Drivers/Vulkan/WSI/VulkanSwapchain.h"
#include "GfxDriver/Utils/ImageUtils.h"
#include "GfxDriver/WSI/Enums.h"
#include "GfxDriver/WSI/Icons/HeavyAIIcon.h"

namespace gfx {

static WSIMouseButton glfw_mouse_button_to_wsi(int32_t glfw_mouse_button) {
  switch (glfw_mouse_button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      return WSIMouseButton::kLeft;
    case GLFW_MOUSE_BUTTON_RIGHT:
      return WSIMouseButton::kRight;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      return WSIMouseButton::kMiddle;
    default:
      return WSIMouseButton::kLeft;
  }
}

static WSIMouseAction glfw_mouse_action_to_wsi(int32_t glfw_mouse_action) {
  switch (glfw_mouse_action) {
    case GLFW_PRESS:
      return WSIMouseAction::kPress;
    case GLFW_RELEASE:
      return WSIMouseAction::kRelease;
    default:
      return WSIMouseAction::kPress;
  }
}

static WSIKeyboardAction glfw_keyboard_action_to_wsi(int32_t glfw_key_action) {
  switch (glfw_key_action) {
    case GLFW_PRESS:
      return WSIKeyboardAction::kPress;
    case GLFW_REPEAT:
      return WSIKeyboardAction::kRepeat;
    case GLFW_RELEASE:
      return WSIKeyboardAction::kRelease;
  }
  UNREACHABLE();
  return WSIKeyboardAction::kPress;
}

static WSIKeyboardModBits glfw_keyboard_mod_bits_to_wsi(int32_t glfw_mod_bits) {
  WSIKeyboardModBits bits{0};
  if (glfw_mod_bits & GLFW_MOD_SHIFT) {
    bits |= WSIKeyboardModBits::kShift;
  }
  if (glfw_mod_bits & GLFW_MOD_CONTROL) {
    bits |= WSIKeyboardModBits::kControl;
  }
  if (glfw_mod_bits & GLFW_MOD_ALT) {
    bits |= WSIKeyboardModBits::kAlt;
  }
  if (glfw_mod_bits & GLFW_MOD_SUPER) {
    bits |= WSIKeyboardModBits::kSuper;
  }
  if (glfw_mod_bits & GLFW_MOD_CAPS_LOCK) {
    bits |= WSIKeyboardModBits::kCapsLock;
  }
  if (glfw_mod_bits & GLFW_MOD_NUM_LOCK) {
    bits |= WSIKeyboardModBits::kNumLock;
  }
  return bits;
}

static WSIKeyboardKey glfw_keyboard_key_to_wsi(int32_t glfw_key) {
  using Key = WSIKeyboardKey;

  static std::unordered_map<int32_t, WSIKeyboardKey> glfw_key_to_wsi_map = {
      {GLFW_KEY_SPACE, Key::kSpace},
      {GLFW_KEY_APOSTROPHE, Key::kApostrophe},
      {GLFW_KEY_COMMA, Key::kComma},
      {GLFW_KEY_MINUS, Key::kMinus},
      {GLFW_KEY_PERIOD, Key::kPeriod},
      {GLFW_KEY_SLASH, Key::kSlash},
      {GLFW_KEY_0, Key::k0},
      {GLFW_KEY_1, Key::k1},
      {GLFW_KEY_2, Key::k2},
      {GLFW_KEY_3, Key::k3},
      {GLFW_KEY_4, Key::k4},
      {GLFW_KEY_5, Key::k5},
      {GLFW_KEY_6, Key::k6},
      {GLFW_KEY_7, Key::k7},
      {GLFW_KEY_8, Key::k8},
      {GLFW_KEY_9, Key::k9},
      {GLFW_KEY_SEMICOLON, Key::kSemicolon},
      {GLFW_KEY_EQUAL, Key::kEqual},
      {GLFW_KEY_A, Key::kA},
      {GLFW_KEY_B, Key::kB},
      {GLFW_KEY_C, Key::kC},
      {GLFW_KEY_D, Key::kD},
      {GLFW_KEY_E, Key::kE},
      {GLFW_KEY_F, Key::kF},
      {GLFW_KEY_G, Key::kG},
      {GLFW_KEY_H, Key::kH},
      {GLFW_KEY_I, Key::kI},
      {GLFW_KEY_J, Key::kJ},
      {GLFW_KEY_K, Key::kK},
      {GLFW_KEY_L, Key::kL},
      {GLFW_KEY_M, Key::kM},
      {GLFW_KEY_N, Key::kN},
      {GLFW_KEY_O, Key::kO},
      {GLFW_KEY_P, Key::kP},
      {GLFW_KEY_Q, Key::kQ},
      {GLFW_KEY_R, Key::kR},
      {GLFW_KEY_S, Key::kS},
      {GLFW_KEY_T, Key::kT},
      {GLFW_KEY_U, Key::kU},
      {GLFW_KEY_V, Key::kV},
      {GLFW_KEY_W, Key::kW},
      {GLFW_KEY_X, Key::kX},
      {GLFW_KEY_Y, Key::kY},
      {GLFW_KEY_Z, Key::kZ},
      {GLFW_KEY_LEFT_BRACKET, Key::kLeftBracket},
      {GLFW_KEY_BACKSLASH, Key::kBackslash},
      {GLFW_KEY_RIGHT_BRACKET, Key::kRightBracket},
      {GLFW_KEY_GRAVE_ACCENT, Key::kGraveAccent},
      {GLFW_KEY_ESCAPE, Key::kEscape},
      {GLFW_KEY_ENTER, Key::kEnter},
      {GLFW_KEY_TAB, Key::kTab},
      {GLFW_KEY_BACKSPACE, Key::kBackspace},
      {GLFW_KEY_INSERT, Key::kInsert},
      {GLFW_KEY_DELETE, Key::kDelete},
      {GLFW_KEY_RIGHT, Key::kRight},
      {GLFW_KEY_LEFT, Key::kLeft},
      {GLFW_KEY_UP, Key::kUp},
      {GLFW_KEY_DOWN, Key::kDown},
      {GLFW_KEY_PAGE_UP, Key::kPageUp},
      {GLFW_KEY_PAGE_DOWN, Key::kPageDown},
      {GLFW_KEY_HOME, Key::kHome},
      {GLFW_KEY_END, Key::kEnd},
      {GLFW_KEY_CAPS_LOCK, Key::kCapsLock},
      {GLFW_KEY_SCROLL_LOCK, Key::kScrollLock},
      {GLFW_KEY_NUM_LOCK, Key::kNumLock},
      {GLFW_KEY_PRINT_SCREEN, Key::kPrintScreen},
      {GLFW_KEY_PAUSE, Key::kPause},
      {GLFW_KEY_F1, Key::kF1},
      {GLFW_KEY_F2, Key::kF2},
      {GLFW_KEY_F3, Key::kF3},
      {GLFW_KEY_F4, Key::kF4},
      {GLFW_KEY_F5, Key::kF5},
      {GLFW_KEY_F6, Key::kF6},
      {GLFW_KEY_F7, Key::kF7},
      {GLFW_KEY_F8, Key::kF8},
      {GLFW_KEY_F9, Key::kF9},
      {GLFW_KEY_F10, Key::kF10},
      {GLFW_KEY_F11, Key::kF11},
      {GLFW_KEY_F12, Key::kF12},
      {GLFW_KEY_F13, Key::kF13},
      {GLFW_KEY_F14, Key::kF14},
      {GLFW_KEY_F15, Key::kF15},
      {GLFW_KEY_F16, Key::kF16},
      {GLFW_KEY_F17, Key::kF17},
      {GLFW_KEY_F18, Key::kF18},
      {GLFW_KEY_F19, Key::kF19},
      {GLFW_KEY_F20, Key::kF20},
      {GLFW_KEY_F21, Key::kF21},
      {GLFW_KEY_F22, Key::kF22},
      {GLFW_KEY_F23, Key::kF23},
      {GLFW_KEY_F24, Key::kF24},
      {GLFW_KEY_F25, Key::kF25},
      {GLFW_KEY_KP_0, Key::kKP0},
      {GLFW_KEY_KP_1, Key::kKP1},
      {GLFW_KEY_KP_2, Key::kKP2},
      {GLFW_KEY_KP_3, Key::kKP3},
      {GLFW_KEY_KP_4, Key::kKP4},
      {GLFW_KEY_KP_5, Key::kKP5},
      {GLFW_KEY_KP_6, Key::kKP6},
      {GLFW_KEY_KP_7, Key::kKP7},
      {GLFW_KEY_KP_8, Key::kKP8},
      {GLFW_KEY_KP_9, Key::kKP9},
      {GLFW_KEY_KP_DECIMAL, Key::kKPDecimal},
      {GLFW_KEY_KP_DIVIDE, Key::kKPDivide},
      {GLFW_KEY_KP_MULTIPLY, Key::kKPMultiply},
      {GLFW_KEY_KP_SUBTRACT, Key::kKPSubtract},
      {GLFW_KEY_KP_ADD, Key::kKPAdd},
      {GLFW_KEY_KP_ENTER, Key::kKPEnter},
      {GLFW_KEY_KP_EQUAL, Key::kKPEqual},
      {GLFW_KEY_LEFT_SHIFT, Key::kLeftShift},
      {GLFW_KEY_LEFT_CONTROL, Key::kLeftControl},
      {GLFW_KEY_LEFT_ALT, Key::kLeftAlt},
      {GLFW_KEY_LEFT_SUPER, Key::kLeftSuper},
      {GLFW_KEY_RIGHT_SHIFT, Key::kRightShift},
      {GLFW_KEY_RIGHT_CONTROL, Key::kRightControl},
      {GLFW_KEY_RIGHT_ALT, Key::kRightAlt},
      {GLFW_KEY_RIGHT_SUPER, Key::kRightSuper}};

  if (!glfw_key_to_wsi_map.count(glfw_key)) {
    return Key::kUnknown;
  } else {
    return glfw_key_to_wsi_map.at(glfw_key);
  }
}

static void glfw_error_callback(int error, const char* description) {
  LOG(ERROR) << "Error: " << description;
}

inline std::string make_window_title(const std::string& name,
                                     uint32_t width,
                                     uint32_t height) {
  std::stringstream ss;
  ss << name << " [" << width << " x " << height << "]";
  return ss.str();
}

VulkanWSI_GLFW::VulkanWSI_GLFW(const WindowSystemCreateInfo& wsi_ci)
    : glfw_window_{nullptr}
    , vk_surface_{VK_NULL_HANDLE}
    , name_{wsi_ci.window_name}
    , show_resolution_in_title_{wsi_ci.show_resolution_in_title} {
  CHECK_GT(wsi_ci.width, 0u);
  CHECK_GT(wsi_ci.height, 0u);

  glfwSetErrorCallback(glfw_error_callback);

  if (!glfwInit()) {
    exit(EXIT_FAILURE);
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  // Use undecorated name here as it will be the app name in the Ubuntu desktop menu bar
  glfw_window_ =
      glfwCreateWindow(wsi_ci.width, wsi_ci.height, name_.c_str(), nullptr, nullptr);

  // Set window icon
  auto [icon_png_data, icon_png_data_len] = heavyai::get_heavy_ai_icon();
  auto image_info = read_png_image_from_memory(icon_png_data, icon_png_data_len);
  if (!image_info.pixels.empty()) {
    GLFWimage glfw_image;
    glfw_image.width = image_info.width;
    glfw_image.height = image_info.height;
    glfw_image.pixels = image_info.pixels.data();
    glfwSetWindowIcon(glfw_window_, 1, &glfw_image);
  }

  // Set window title including resolution. This will show up on the window itself and
  // in the desktop application menu
  if (show_resolution_in_title_) {
    glfwSetWindowTitle(glfw_window_,
                       make_window_title(name_, wsi_ci.width, wsi_ci.height).c_str());
  }

  glfwSetWindowUserPointer(glfw_window_, this);
  registerGLFWCallbacks();
}

VulkanWSI_GLFW::~VulkanWSI_GLFW() {
  glfwDestroyWindow(glfw_window_);
  glfw_window_ = nullptr;
  destroySurface();
  glfwTerminate();
}

std::vector<const char*> VulkanWSI_GLFW::getRequiredInstanceExtensions() const {
  uint32_t glfw_extension_count = 0;
  const char** glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  return std::vector<const char*>(glfw_extensions,
                                  glfw_extensions + glfw_extension_count);
}

VkSurfaceKHR VulkanWSI_GLFW::createSurface(VkInstance instance) {
  vk_instance_ = instance;
  if (vk_surface_ != VK_NULL_HANDLE) {
    // TODO: exception? CHECK?
    LOG(WARNING) << "createSurface called when surface already exists.";
  } else {
    CHECK_VKRESULT(glfwCreateWindowSurface(instance, glfw_window_, nullptr, &vk_surface_),
                   "creating glfw window surface");
  }
  return vk_surface_;
}

void VulkanWSI_GLFW::destroySurface() {
  if (vk_surface_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(vk_instance_, vk_surface_, nullptr);
    vk_surface_ = VK_NULL_HANDLE;
  }
}

VkSurfaceKHR VulkanWSI_GLFW::getSurface() const {
  return vk_surface_;
}

std::pair<uint32_t, uint32_t> VulkanWSI_GLFW::getWindowSize() const {
  int width, height;
  glfwGetFramebufferSize(glfw_window_, &width, &height);
  return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

std::pair<float, float> VulkanWSI_GLFW::getWindowContentScale() const {
  float xscale, yscale;
  glfwGetWindowContentScale(glfw_window_, &xscale, &yscale);
  return {xscale, yscale};
}

void VulkanWSI_GLFW::setWindowSize(uint32_t width, uint32_t height) {
  glfwSetWindowSize(glfw_window_, width, height);
}

void VulkanWSI_GLFW::setWindowVisibility(bool is_visible) const {
  if (is_visible) {
    glfwShowWindow(glfw_window_);
  } else {
    glfwHideWindow(glfw_window_);
  }
}

void VulkanWSI_GLFW::setWindowTitle(const std::string& title) {
  glfwSetWindowTitle(glfw_window_, title.c_str());
}

bool VulkanWSI_GLFW::windowShouldClose() const {
  return glfwWindowShouldClose(glfw_window_);
}

void VulkanWSI_GLFW::pollEvents() const {
  glfwPollEvents();
}

void VulkanWSI_GLFW::waitEvents(float timeout_seconds) const {
  glfwWaitEventsTimeout(timeout_seconds);
}

void VulkanWSI_GLFW::registerGLFWCallbacks() {
  CHECK(glfw_window_);
  glfwSetFramebufferSizeCallback(glfw_window_, glfw_onResizeCallback);
  glfwSetKeyCallback(glfw_window_, glfw_onKeyPressCallback);
  glfwSetMouseButtonCallback(glfw_window_, glfw_onMouseButtonCallback);
  glfwSetCursorPosCallback(glfw_window_, glfw_onCursorPosCallback);
  glfwSetScrollCallback(glfw_window_, glfw_onScrollCallback);
}

void VulkanWSI_GLFW::glfw_onResizeCallback(GLFWwindow* glfw_window,
                                           int width,
                                           int height) {
  VulkanWSI_GLFW* this_wsi =
      static_cast<VulkanWSI_GLFW*>(glfwGetWindowUserPointer(glfw_window));
  CHECK(this_wsi);
  CHECK(this_wsi->present_device_);
  this_wsi->present_device_->getSwapchain()->setInvalid();
  if (this_wsi->hasEventHandlers()) {
    WSIWindowResizeEvent event(width, height);
    this_wsi->notifyEvent(event);
  }
  if (this_wsi->show_resolution_in_title_) {
    glfwSetWindowTitle(glfw_window,
                       make_window_title(this_wsi->name_, width, height).c_str());
  }
}

void VulkanWSI_GLFW::glfw_onKeyPressCallback(GLFWwindow* glfw_window,
                                             int key,
                                             int scancode,
                                             int action,
                                             int modifiers) {
  VulkanWSI_GLFW* this_wsi =
      static_cast<VulkanWSI_GLFW*>(glfwGetWindowUserPointer(glfw_window));
  CHECK(this_wsi);
  if (this_wsi->hasEventHandlers()) {
    WSIKeyboardEvent event(glfw_keyboard_key_to_wsi(key),
                           glfwGetKeyName(key, scancode),
                           glfw_keyboard_action_to_wsi(action),
                           glfw_keyboard_mod_bits_to_wsi(modifiers));
    this_wsi->notifyEvent(event);
  }
}

void VulkanWSI_GLFW::glfw_onMouseButtonCallback(GLFWwindow* glfw_window,
                                                int button,
                                                int action,
                                                int mods) {
  VulkanWSI_GLFW* this_wsi =
      static_cast<VulkanWSI_GLFW*>(glfwGetWindowUserPointer(glfw_window));
  CHECK(this_wsi);
  if (this_wsi->hasEventHandlers()) {
    WSIMouseButtonEvent event(glfw_mouse_button_to_wsi(button),
                              glfw_mouse_action_to_wsi(action),
                              glfw_keyboard_mod_bits_to_wsi(mods));
    this_wsi->notifyEvent(event);
  }
}

void VulkanWSI_GLFW::glfw_onCursorPosCallback(GLFWwindow* glfw_window,
                                              double x,
                                              double y) {
  VulkanWSI_GLFW* this_wsi =
      static_cast<VulkanWSI_GLFW*>(glfwGetWindowUserPointer(glfw_window));
  CHECK(this_wsi);
  if (this_wsi->hasEventHandlers()) {
    WSIMouseCursorEvent event(x, y);
    this_wsi->notifyEvent(event);
  }
}

void VulkanWSI_GLFW::glfw_onScrollCallback(GLFWwindow* glfw_window, double x, double y) {
  VulkanWSI_GLFW* this_wsi =
      static_cast<VulkanWSI_GLFW*>(glfwGetWindowUserPointer(glfw_window));
  CHECK(this_wsi);
  if (this_wsi->hasEventHandlers()) {
    WSIScrollEvent event(x, y);
    this_wsi->notifyEvent(event);
  }
}

}  // namespace gfx
