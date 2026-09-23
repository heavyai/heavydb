/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <glm/vec4.hpp>

namespace gfx {

// Color conversion using glm::vec4
glm::vec4 HLSAtoRGBA(float h, float l, float s, float a);

}  // namespace gfx
