/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <glslang/Public/ShaderLang.h>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

void rebind_tshader_function_calls(glslang::TShader& shader,
                                   const SubroutineMap& call_rebind_map);

}  // namespace gfx
