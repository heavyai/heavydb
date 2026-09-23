/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/WSI/Enums.h"

namespace gfx {

std::ostream& operator<<(std::ostream& os, const WSIMouseButton& value) {
  switch (value) {
    case WSIMouseButton::kLeft:
      os << "Left";
      break;
    case WSIMouseButton::kRight:
      os << "Right";
      break;
    case WSIMouseButton::kMiddle:
      os << "Middle";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const WSIMouseAction& value) {
  switch (value) {
    case WSIMouseAction::kPress:
      os << "Press";
      break;
    case WSIMouseAction::kRelease:
      os << "Release";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const WSIKeyboardAction& value) {
  switch (value) {
    case WSIKeyboardAction::kPress:
      os << "Press";
      break;
    case WSIKeyboardAction::kRepeat:
      os << "Repeat";
      break;
    case WSIKeyboardAction::kRelease:
      os << "Release";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const WSIKeyboardModBits& value) {
  using bits = WSIKeyboardModBits;
  if (any_bits_set(value & bits::kShift)) {
    os << "[Shift]";
  }
  if (any_bits_set(value & bits::kControl)) {
    os << "[Control]";
  }
  if (any_bits_set(value & bits::kAlt)) {
    os << "[Alt]";
  }
  if (any_bits_set(value & bits::kSuper)) {
    os << "[Super]";
  }
  if (any_bits_set(value & bits::kCapsLock)) {
    os << "[CapsLock]";
  }
  if (any_bits_set(value & bits::kNumLock)) {
    os << "[NumLock]";
  }
  return os;
}

}  // namespace gfx
