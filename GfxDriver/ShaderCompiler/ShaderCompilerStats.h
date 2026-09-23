/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

// Class to accumulate shader compilation statistics and generate a formatted
// report
class ShaderCompilerStatsReporter {
 public:
  ShaderCompilerStatsReporter();
  ~ShaderCompilerStatsReporter();

  // Spirv generation stats (convert Builder to spirv)
  // operator_time: time to generate valid GLSL from a Builder
  // glsl_to_spirv_time: time in GlslangWrapper and spirv-tools
  void addBuildSpirvStats(ShaderStage stage,
                          const std::string& template_name,
                          uint64_t operator_time,
                          uint64_t glsl_time);

  void addSubBuilderStats(const std::string& template_name, uint64_t operator_time);

  // ShaderCache creation time
  // Accumulates the time required to build spirv for all shaders in a Material
  void addBuildCacheStats(const std::string& name, uint64_t create_cache_time);

  // Write a formatted report to ostream
  void generateReport(std::ostream& os);

 private:
  // Default MinN = 3 to filter singleton compiles
  // (for be-render-tests, which does a clear_gpu, so singletons run twice)
  static constexpr uint32_t kReportMinN = 3;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gfx
