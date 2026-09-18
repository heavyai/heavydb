/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>

#include "GfxDriver/WSI/Enums.h"

namespace gfx {

//
// WSIEvent
//
// Base class for events. These provide details about window system
// interactions such as window sizing, keyboard actions, etc.
// A callback handler can be registered by the application using the global
// WindowSystemIntegration class
//
class WSIEvent {
 public:
  enum class Type { kWindowResize, kKeyboard, kMouseButton, kCursor, kScroll };
  explicit WSIEvent(Type type);
  virtual ~WSIEvent() = default;
  const Type type() const;

 private:
  Type type_;
};

using WSIEventHandler = std::function<void(const WSIEvent&)>;
using WSIEventHandlerID = uint32_t;

//
// Standard WSIEvent types
//

// Window resize
class WSIWindowResizeEvent : public WSIEvent {
 public:
  explicit WSIWindowResizeEvent(uint32_t width, uint32_t height);
  ~WSIWindowResizeEvent() override = default;

  uint32_t width() const;
  uint32_t height() const;

 private:
  uint32_t width_;
  uint32_t height_;
};

// Keyboard
class WSIKeyboardEvent : public WSIEvent {
 public:
  explicit WSIKeyboardEvent(WSIKeyboardKey key,
                            const char* key_name,
                            WSIKeyboardAction action,
                            WSIKeyboardModBits mod_bits);
  ~WSIKeyboardEvent() override = default;

  WSIKeyboardKey key() const;
  const char* keyName() const;  // may return null
  WSIKeyboardAction action() const;
  WSIKeyboardModBits modBits() const;

 private:
  WSIKeyboardKey key_;
  const char* key_name_;
  WSIKeyboardAction action_;
  WSIKeyboardModBits mod_bits_;
};

// Mouse button
class WSIMouseButtonEvent : public WSIEvent {
 public:
  explicit WSIMouseButtonEvent(WSIMouseButton button,
                               WSIMouseAction action,
                               WSIKeyboardModBits mod_bits);
  ~WSIMouseButtonEvent() override = default;

  WSIMouseButton button() const;
  WSIMouseAction action() const;
  WSIKeyboardModBits modBits() const;

 private:
  WSIMouseButton button_;
  WSIMouseAction action_;
  WSIKeyboardModBits mod_bits_;
};

// Mouse cursor
class WSIMouseCursorEvent : public WSIEvent {
 public:
  explicit WSIMouseCursorEvent(double x, double y);
  ~WSIMouseCursorEvent() override = default;

  double x() const;
  double y() const;

 private:
  double x_;
  double y_;
};

// Scroll
class WSIScrollEvent : public WSIEvent {
 public:
  explicit WSIScrollEvent(double x, double y);
  ~WSIScrollEvent() override = default;

  double x() const;
  double y() const;

 private:
  double x_;
  double y_;
};

}  // namespace gfx
