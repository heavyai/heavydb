/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

testing::AssertionResult validate_shader_caches(const ShaderCacheShPtrVector& caches);

}  // namespace gfx
