/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/Types.h"

namespace gfx {

std::string to_string(const UniqueResourceId& value) {
  return std::string("Device ") + std::to_string(value.first) +
         std::string(", Resource ") + std::to_string(value.second);
}

}  // end namespace gfx

std::ostream& operator<<(std::ostream& os, const gfx::UniqueResourceId& value) {
  os << gfx::to_string(value);
  return os;
}
