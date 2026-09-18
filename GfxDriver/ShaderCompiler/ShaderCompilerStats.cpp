/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/ShaderCompilerStats.h"

#include "GfxDriver/Utils/StatsUtils.h"
#include "Logger/Logger.h"

namespace gfx {

//
// BuildSpirvStats
//
// Custom StatsType for storing stats required by ShaderManager::buildSpirv
//
struct BuildSpirvStats {
  uint64_t operator_time{0};
  uint64_t glsl_to_spirv_time{0};

  BuildSpirvStats() = default;
  explicit BuildSpirvStats(uint64_t operator_time, uint64_t glsl_time)
      : operator_time{operator_time}, glsl_to_spirv_time{glsl_time} {}

  void operator+=(const BuildSpirvStats& other) {
    operator_time += other.operator_time;
    glsl_to_spirv_time += other.glsl_to_spirv_time;
  }

  bool operator>(const BuildSpirvStats& other) {
    return (operator_time + glsl_to_spirv_time) >
           (other.operator_time + other.glsl_to_spirv_time);
  }

  void operator/=(uint32_t N) {
    CHECK_GT(N, 0u);
    operator_time /= N;
    glsl_to_spirv_time /= N;
  }
};

std::ostream& operator<<(std::ostream& os, BuildSpirvStats& stats) {
  os << std::left << std::setw(15)
     << ("  total=" + std::to_string(stats.operator_time + stats.glsl_to_spirv_time));
  os << std::left << std::setw(18)
     << ("  operators=" + std::to_string(stats.operator_time));
  os << std::left << std::setw(14)
     << ("  glsl=" + std::to_string(stats.glsl_to_spirv_time));
  return os;
}

//
// ShaderCompilerStatsReporter::Impl
//
class ShaderCompilerStatsReporter::Impl {
 public:
  void addBuildSpirvStats(ShaderStage stage,
                          const std::string& template_name,
                          uint64_t operator_time,
                          uint64_t glsl_to_spirv_time);
  void addSubBuilderStats(const std::string& template_name, uint64_t operator_time);
  void addBuildCacheStats(const std::string& name, uint64_t create_cache_time);
  void generateReport(std::ostream& os, uint32_t min_N);

 private:
  StatsMap<BuildSpirvStats> spirv_stage_map_;     // Averages accumulated by shader stage
  StatsMap<BuildSpirvStats> spirv_template_map_;  // Averages
  StatsMap<uint64_t> sub_builder_map_;
  StatsMap<uint64_t> cache_map_;
};

void ShaderCompilerStatsReporter::Impl::addBuildSpirvStats(
    ShaderStage stage,
    const std::string& template_name,
    uint64_t operator_time,
    uint64_t glsl_time) {
  BuildSpirvStats new_stat{operator_time, glsl_time};
  spirv_stage_map_.accumulate(to_string(stage), new_stat);
  spirv_template_map_.accumulate(template_name, new_stat);
}

void ShaderCompilerStatsReporter::Impl::addSubBuilderStats(
    const std::string& template_name,
    uint64_t operator_time) {
  sub_builder_map_.accumulate(template_name, operator_time);
}

void ShaderCompilerStatsReporter::Impl::addBuildCacheStats(const std::string& name,
                                                           uint64_t create_cache_time) {
  cache_map_.accumulate(name, create_cache_time);
}

void ShaderCompilerStatsReporter::Impl::generateReport(std::ostream& os, uint32_t min_N) {
  os << "\n==============================\n";
  os << "Shader compiler stats\n";
  os << "==============================\n";
  auto map_report = [&](auto const& stats_map, auto const& label) {
    if (!stats_map.is_empty()) {
      stats_map.generateReport(os, label, min_N);
      os << "\n";
    }
  };
  map_report(spirv_stage_map_, "Build spirv stage averages");
  map_report(spirv_template_map_, "Build spirv template averages");
  map_report(sub_builder_map_, "Process sub-builder averages");
  map_report(cache_map_, "Cache build averages");
}

//
// ShaderCompilerStatsReporter
//
ShaderCompilerStatsReporter::ShaderCompilerStatsReporter()
    : impl_{std::make_unique<ShaderCompilerStatsReporter::Impl>()} {}

ShaderCompilerStatsReporter::~ShaderCompilerStatsReporter() {}

void ShaderCompilerStatsReporter::addBuildSpirvStats(ShaderStage stage,
                                                     const std::string& template_name,
                                                     uint64_t operator_time,
                                                     uint64_t glsl_to_spirv_time) {
  impl_->addBuildSpirvStats(stage, template_name, operator_time, glsl_to_spirv_time);
}
void ShaderCompilerStatsReporter::addSubBuilderStats(const std::string& template_name,
                                                     uint64_t operator_time) {
  impl_->addSubBuilderStats(template_name, operator_time);
}
void ShaderCompilerStatsReporter::addBuildCacheStats(const std::string& name,
                                                     uint64_t create_cache_time) {
  impl_->addBuildCacheStats(name, create_cache_time);
}

void ShaderCompilerStatsReporter::generateReport(std::ostream& os) {
  impl_->generateReport(os, kReportMinN);
}

}  // namespace gfx
