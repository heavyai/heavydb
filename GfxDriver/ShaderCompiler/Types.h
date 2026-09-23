/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Shared/EnumBitmaskOps.h"

namespace gfx {

using StrVector = std::vector<std::string>;
using spirv_t = std::vector<uint32_t>;

enum class ShaderStage : uint8_t {
  kVertex,
  kFragment,
  kGeometry,
  kTessControl,
  kTessEval,
  kCompute,
  kRayGen,
  kAnyHit,
  kClosestHit,
  kMiss,
  kIntersection,
  kCallable,
  kMesh,
  kTask
};

// must be kept in sync with enum
static constexpr int kShaderStageCount = 8;

// Stage bits - add stages as needed
enum class ShaderStageBits {
  kVertex = 1 << 0,
  kFragment = 1 << 1,
  kGeometry = 1 << 2,
  kCompute = 1 << 3,
  kRayGen = 1 << 4,
  kAnyHit = 1 << 5,
  kClosestHit = 1 << 6,
  kMiss = 1 << 7,
  kIntersection = 1 << 8,
  kCallable = 1 << 9,
  kMesh = 1 << 10,
  kTask = 1 << 11
};

enum ShaderArtifactTypeBits {
  kNone = 0x00,
  kBuilder = 0x01,     // builder serialization
  kGlsl = 0x02,        // input glsl from builder
  kSpvBin = 0x04,      // spirv binary blob
  kSpvDis = 0x08,      // spirv disassembly
  kSpvGlsl = 0x10,     // glsl output from spirv-cross
  kSpvReflect = 0x20,  // reflection information from spir-cross
  kAll = 0xFFFF
};

// <key, <target, is_required>>
using SubroutineMap = std::unordered_map<std::string, std::pair<std::string, bool>>;

class Library;
using LibraryUqPtr = std::unique_ptr<Library>;

class ShaderManager;
using ShaderManagerUqPtr = std::unique_ptr<ShaderManager>;

class GlslangWrapper;
using GlslangWrapperUqPtr = std::unique_ptr<GlslangWrapper>;

class ShaderReflection;
using ShaderReflectionUqPtr = std::unique_ptr<ShaderReflection>;

class ShaderCache;
using ShaderCacheShPtr = std::shared_ptr<ShaderCache>;
using ShaderCacheShPtrVector = std::vector<std::shared_ptr<ShaderCache>>;

// conversions to string
std::string to_string(const ShaderStage id);

}  // namespace gfx

// streaming
std::ostream& operator<<(std::ostream& os, const gfx::ShaderStage value);

ENABLE_BITMASK_OPS(::gfx::ShaderStageBits);
