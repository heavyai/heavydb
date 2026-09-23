/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Render/GeoCountResources.h"

#include "GfxDriver/DriverInstance.h"
#include "GfxDriver/Enums.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Resources/ResourceManager.h"

#define PROFILE_COMPUTE_PASSES 0
#if PROFILE_COMPUTE_PASSES
#include <iostream>
#include "Shared/measure.h"
#endif

namespace gfx {

GeoCountResources::GeoCountResources(const GfxContext& gfx_context,
                                     ResourceManager& rsrc_mgr)
    : gfx_context_{gfx_context}, rsrc_mgr_{rsrc_mgr} {
  createResources();
}

void GeoCountResources::createResources() {
  // get subgroup size
  auto const& driver = gfx_context_.getPrimaryDriver();
  subgroup_size_ = driver.getLimits().subgroup_size;
  CHECK_GT(subgroup_size_, 0u);
  auto const subgroup_size_bits = uint32_t(log2(subgroup_size_));

  // get max workgroup count
  max_workgroup_count_ = driver.getLimits().max_mesh_workgroup_count[0];

  // create shaders and caches
  auto& shader_mgr = gfx_context_.getShaderManager();
  auto count_builder =
      shader_mgr.createBuilderVector({{"Rendering/geoCount_count.comp"}});
  auto build_builder =
      shader_mgr.createBuilderVector({{"Rendering/geoCount_build.comp"}});
  count_builder[0]->replaceFirstTag("workgroupSize", std::to_string(subgroup_size_));
  build_builder[0]->replaceFirstTag("workgroupSize", std::to_string(subgroup_size_));
  build_builder[0]->replaceFirstTag("workgroupSizeBits",
                                    std::to_string(subgroup_size_bits));
  static constexpr gfx::DeviceCapabilityBits subgroups_bits =
      gfx::DeviceCapabilityBits::kSubgroupVote |
      gfx::DeviceCapabilityBits::kSubgroupArithmetic;
  const bool use_subgroups = driver.queryCapabilities(subgroups_bits);
  count_builder[0]->replaceFirstTag("useSubgroups", std::to_string(use_subgroups));
  build_builder[0]->replaceFirstTag("useSubgroups", std::to_string(use_subgroups));
  auto count_cache = shader_mgr.createCacheVector(std::move(count_builder));
  CHECK(!count_cache.empty());
  auto build_cache = shader_mgr.createCacheVector(std::move(build_builder));
  CHECK(!build_cache.empty());

  // create materials
  count_material_ = rsrc_mgr_.createMaterial("GeoCount Count", count_cache);
  CHECK(count_material_);
  build_material_ = rsrc_mgr_.createMaterial("GeoCount Build", build_cache);
  CHECK(build_material_);

  // create pipelines
  count_pipeline_ = rsrc_mgr_.createComputePipeline("GeoCount Count", *count_material_);
  CHECK(count_pipeline_);
  count_pipeline_->create();

  build_pipeline_ = rsrc_mgr_.createComputePipeline("GeoCount Build", *build_material_);
  CHECK(build_pipeline_);
  build_pipeline_->create();

  // create atomic buffer
  atomic_buffer_ = rsrc_mgr_.createBuffer("GeoCount Atomic",
                                          {gfx::BufferType::kUnspecified,
                                           sizeof(uint32_t),
                                           gfx::BufferUsageBits::kStorageBufferBit});
  CHECK(atomic_buffer_);
}

void GeoCountResources::destroyResources() {
  if (count_pipeline_) {
    rsrc_mgr_.destroyPipeline(std::move(count_pipeline_));
  }
  if (build_pipeline_) {
    rsrc_mgr_.destroyPipeline(std::move(build_pipeline_));
  }

  count_material_ = nullptr;
  build_material_ = nullptr;

  if (atomic_buffer_) {
    rsrc_mgr_.destroyBuffer(std::move(atomic_buffer_));
  }
  if (work_units_buffer_) {
    rsrc_mgr_.destroyBuffer(std::move(work_units_buffer_));
  }
}

uint32_t GeoCountResources::createWorkUnits(CommandList& cmd_list,
                                            const uint64_t base_address,
                                            const uint32_t num_rows,
                                            const uint32_t num_columns,
                                            const uint32_t column_index,
                                            const uint32_t value_byte_size) {
  // run count pass
  auto const num_work_units = runCountPass(
      cmd_list, base_address, num_rows, num_columns, column_index, value_byte_size);

  // any work units?
  if (num_work_units == 0U) {
    return 0U;
  }

  // create work units buffer
  CHECK(!work_units_buffer_);
  work_units_buffer_ = rsrc_mgr_.createBuffer("GeoCount Work Units",
                                              {gfx::BufferType::kUnspecified,
                                               sizeof(WorkUnit) * num_work_units,
                                               gfx::BufferUsageBits::kStorageBufferBit});
  CHECK(work_units_buffer_);

  // run build pass
  runBuildPass(
      cmd_list, base_address, num_rows, num_columns, column_index, value_byte_size);

  // return count
  return num_work_units;
}

uint32_t GeoCountResources::runCountPass(CommandList& cmd_list,
                                         const uint64_t base_address,
                                         const uint32_t num_rows,
                                         const uint32_t num_columns,
                                         const uint32_t column_index,
                                         const uint32_t value_byte_size) {
  const uint32_t group_count_x = (num_rows + (subgroup_size_ - 1)) / subgroup_size_;
  CHECK_LE(group_count_x, max_workgroup_count_);

  count_material_->setUniformAttribute("base_address", base_address);
  count_material_->setUniformAttribute("num_rows", num_rows);
  count_material_->setUniformAttribute("num_columns", num_columns);
  count_material_->setUniformAttribute("column_index", column_index);
  count_material_->setUniformAttribute("value_byte_size", value_byte_size);

  count_material_->bindShaderStorageBufferToBlock("WORKGROUP_COUNT_SSBO",
                                                  *atomic_buffer_);

  count_material_->updateDescriptorSets();

  uint32_t workgroup_count = 0u;
  atomic_buffer_->updateSubData(&workgroup_count, sizeof(uint32_t), 0ull);

#if PROFILE_COMPUTE_PASSES
  auto count_timer = timer_start();
#endif

  cmd_list.pushLabel("GeoCount Count")
      .dispatchCompute(*count_pipeline_, 0u, group_count_x, 1u, 1u)
      .popLabel()
      .flush("GeoCount Count", gfx::CommandList::SubmitType::kWaitComplete);

#if PROFILE_COMPUTE_PASSES
  auto count_us =
      timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
          count_timer);
  std::cout << "DEBUG: GeoCountResources: count pass " << count_us;
#endif

  atomic_buffer_->getData(&workgroup_count, sizeof(uint32_t));
  return workgroup_count;
}

void GeoCountResources::runBuildPass(CommandList& cmd_list,
                                     const uint64_t base_address,
                                     const uint32_t num_rows,
                                     const uint32_t num_columns,
                                     const uint32_t column_index,
                                     const uint32_t value_byte_size) {
  const uint32_t group_count_x = (num_rows + (subgroup_size_ - 1)) / subgroup_size_;
  CHECK_LE(group_count_x, max_workgroup_count_);

  build_material_->setUniformAttribute("base_address", base_address);
  build_material_->setUniformAttribute("num_rows", num_rows);
  build_material_->setUniformAttribute("num_columns", num_columns);
  build_material_->setUniformAttribute("column_index", column_index);
  build_material_->setUniformAttribute("value_byte_size", value_byte_size);

  build_material_->bindShaderStorageBufferToBlock("WORK_UNIT_INDEX_SSBO",
                                                  *atomic_buffer_);
  build_material_->bindShaderStorageBufferToBlock("WORK_UNITS_SSBO", *work_units_buffer_);

  build_material_->updateDescriptorSets();

  uint32_t work_unit_index = 0u;
  atomic_buffer_->updateSubData(&work_unit_index, sizeof(uint32_t), 0ull);

#if PROFILE_COMPUTE_PASSES
  auto build_timer = timer_start();
#endif

  cmd_list.pushLabel("GeoCount Build")
      .dispatchCompute(*build_pipeline_, 0u, group_count_x, 1u, 1u)
      .popLabel()
#if PROFILE_COMPUTE_PASSES
      .flush("GeoCount Build", gfx::CommandList::SubmitType::kWaitComplete);
#else
      .bufferMemoryBarrier(work_units_buffer_->getBuffer(),
                           gfx::BufferMemoryBarrierType::kComputeToMeshShader);
#endif

#if PROFILE_COMPUTE_PASSES
  auto build_us =
      timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
          build_timer);
  std::cout << "us, build pass " << build_us << "us" << std::endl;
#endif
}

void GeoCountResources::destroyWorkUnitsBuffer() {
  CHECK(work_units_buffer_);
  rsrc_mgr_.destroyBuffer(std::move(work_units_buffer_));
}

}  // namespace gfx
