/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>

#include "Shared/EnumBitmaskOps.h"

namespace gfx {

enum class WSIMouseButton { kLeft, kRight, kMiddle };

enum class WSIMouseAction { kPress, kRelease };

enum class WSIKeyboardAction { kPress, kRepeat, kRelease };

enum class WSIKeyboardModBits {
  kShift = 1 << 0,
  kControl = 1 << 1,
  kAlt = 1 << 2,
  kSuper = 1 << 3,
  kCapsLock = 1 << 4,
  kNumLock = 1 << 5
};

enum class WSIKeyboardKey {
  kUnknown = -1,
  kSpace = 0,
  kApostrophe,
  kComma,
  kMinus,
  kPeriod,
  kSlash,
  k0,
  k1,
  k2,
  k3,
  k4,
  k5,
  k6,
  k7,
  k8,
  k9,
  kSemicolon,
  kEqual,
  kA,
  kB,
  kC,
  kD,
  kE,
  kF,
  kG,
  kH,
  kI,
  kJ,
  kK,
  kL,
  kM,
  kN,
  kO,
  kP,
  kQ,
  kR,
  kS,
  kT,
  kU,
  kV,
  kW,
  kX,
  kY,
  kZ,
  kLeftBracket,
  kBackslash,
  kRightBracket,
  kGraveAccent,
  kEscape,
  kEnter,
  kTab,
  kBackspace,
  kInsert,
  kDelete,
  kRight,
  kLeft,
  kUp,
  kDown,
  kPageUp,
  kPageDown,
  kHome,
  kEnd,
  kCapsLock,
  kScrollLock,
  kNumLock,
  kPrintScreen,
  kPause,
  // Function keys
  kF1,
  kF2,
  kF3,
  kF4,
  kF5,
  kF6,
  kF7,
  kF8,
  kF9,
  kF10,
  kF11,
  kF12,
  kF13,
  kF14,
  kF15,
  kF16,
  kF17,
  kF18,
  kF19,
  kF20,
  kF21,
  kF22,
  kF23,
  kF24,
  kF25,
  // Keypad keys
  kKP0,
  kKP1,
  kKP2,
  kKP3,
  kKP4,
  kKP5,
  kKP6,
  kKP7,
  kKP8,
  kKP9,
  kKPDecimal,
  kKPDivide,
  kKPMultiply,
  kKPSubtract,
  kKPAdd,
  kKPEnter,
  kKPEqual,
  // Modifier keys
  kLeftShift,
  kLeftControl,
  kLeftAlt,
  kLeftSuper,
  kRightShift,
  kRightControl,
  kRightAlt,
  kRightSuper
};

std::ostream& operator<<(std::ostream& os, const WSIMouseButton& value);
std::ostream& operator<<(std::ostream& os, const WSIMouseAction& value);
std::ostream& operator<<(std::ostream& os, const WSIKeyboardAction& value);
std::ostream& operator<<(std::ostream& os, const WSIKeyboardModBits& value);

}  // namespace gfx

ENABLE_BITMASK_OPS(::gfx::WSIKeyboardModBits)
