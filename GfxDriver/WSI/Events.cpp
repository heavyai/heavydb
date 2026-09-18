/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/WSI/Events.h"

namespace gfx {

WSIEvent::WSIEvent(Type type) : type_{type} {}
const WSIEvent::Type WSIEvent::type() const {
  return type_;
}

// Window resize
WSIWindowResizeEvent::WSIWindowResizeEvent(uint32_t width, uint32_t height)
    : WSIEvent(Type::kWindowResize), width_{width}, height_{height} {}

uint32_t WSIWindowResizeEvent::width() const {
  return width_;
}
uint32_t WSIWindowResizeEvent::height() const {
  return height_;
}

// Keyboard
WSIKeyboardEvent::WSIKeyboardEvent(WSIKeyboardKey key,
                                   const char* key_name,
                                   WSIKeyboardAction action,
                                   WSIKeyboardModBits mod_bits)
    : WSIEvent(Type::kKeyboard)
    , key_{key}
    , key_name_{key_name}
    , action_{action}
    , mod_bits_{mod_bits} {}

WSIKeyboardKey WSIKeyboardEvent::key() const {
  return key_;
}
const char* WSIKeyboardEvent::keyName() const {
  return key_name_;
}
WSIKeyboardAction WSIKeyboardEvent::action() const {
  return action_;
};
WSIKeyboardModBits WSIKeyboardEvent::modBits() const {
  return mod_bits_;
}

// Mouse button
WSIMouseButtonEvent::WSIMouseButtonEvent(WSIMouseButton button,
                                         WSIMouseAction action,
                                         WSIKeyboardModBits mod_bits)
    : WSIEvent(Type::kMouseButton)
    , button_{button}
    , action_{action}
    , mod_bits_{mod_bits} {}

WSIMouseButton WSIMouseButtonEvent::button() const {
  return button_;
}
WSIMouseAction WSIMouseButtonEvent::action() const {
  return action_;
}
WSIKeyboardModBits WSIMouseButtonEvent::modBits() const {
  return mod_bits_;
}

// Mouse cursor
WSIMouseCursorEvent::WSIMouseCursorEvent(double x, double y)
    : WSIEvent(Type::kCursor), x_{x}, y_{y} {}

double WSIMouseCursorEvent::x() const {
  return x_;
}
double WSIMouseCursorEvent::y() const {
  return y_;
}

// Scroll
WSIScrollEvent::WSIScrollEvent(double x, double y)
    : WSIEvent(Type::kScroll), x_{x}, y_{y} {}

double WSIScrollEvent::x() const {
  return x_;
}
double WSIScrollEvent::y() const {
  return y_;
}

}  // namespace gfx
