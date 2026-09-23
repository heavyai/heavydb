/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace gfx {

class PipelineDescriptor;
using PipelineDescriptorUqPtr = std::unique_ptr<PipelineDescriptor>;

struct PrimitiveAssemblyAttrInfo;
struct PushConstantRange;
class PushConstantRanges;

class ShaderBindingTable;
using ShaderBindingTableUqPtr = std::unique_ptr<ShaderBindingTable>;

}  // namespace gfx
