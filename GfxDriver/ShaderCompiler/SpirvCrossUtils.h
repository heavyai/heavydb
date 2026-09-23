/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>

#include <spirv_cross/spirv_cross.hpp>

namespace gfx {

void write_spirv_reflection(std::ostream&, spirv_cross::Compiler&);

}