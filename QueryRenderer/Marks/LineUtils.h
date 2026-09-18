/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/ShaderCompiler/GlslStructBuilder.h"

namespace QueryRenderer {
void generate_line_interface_blocks(gfx::GlslStructBuilder& geometry_inputs,
                                    gfx::GlslStructBuilder& fragment_inputs,
                                    bool has_accumulator);
};  // namespace QueryRenderer
