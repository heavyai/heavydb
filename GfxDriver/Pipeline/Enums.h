/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Shared/EnumBitmaskOps.h"

namespace gfx {
// PipelineStageBits values match their Vulkan counterparts
enum class PipelineStageBits {
  kNone = 0,
  kTopOfPipeBit = 1 << 0,
  kDrawIndirectBit = 1 << 1,
  kVertexInputBit = 1 << 2,
  kVertexShaderBit = 1 << 3,
  kTessControlShaderBit = 1 << 4,
  kTessEvalShaderBit = 1 << 5,
  kGeometryShaderBit = 1 << 6,
  kFragmentShaderBit = 1 << 7,
  kEarlyFragmentTestsBit = 1 << 8,
  kLateFragmentTestsBit = 1 << 9,
  kColorAttachmentOutputBit = 1 << 10,
  kComputeShaderBit = 1 << 11,
  kTransferBit = 1 << 12,
  kBottomOfPipeBit = 1 << 13,
  kHostBit = 1 << 14,
  kAllGraphicsBit = 1 << 15,
  kAllCommandsBit = 1 << 16,
  kTaskShaderBit = 1 << 19,
  kMeshShaderBit = 1 << 20,
  kRayTracingShaderBit = 1 << 21,
  kAccelStructureBuildBit = 1 << 25,
};

}  // namespace gfx

ENABLE_BITMASK_OPS(gfx::PipelineStageBits);
