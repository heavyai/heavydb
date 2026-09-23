/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/GfxContext.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

class GeoCountResources {
 public:
  explicit GeoCountResources(const GfxContext& gfx_context, ResourceManager& rsrc_mgr);
  GeoCountResources() = delete;
  ~GeoCountResources() = default;

  void destroyResources();

  uint32_t createWorkUnits(CommandList& cmd_list,
                           const uint64_t base_address,
                           const uint32_t num_rows,
                           const uint32_t num_columns,
                           const uint32_t column_index,
                           const uint32_t value_byte_size);

  const gfx::BufferWrapper* getWorkUnitsBuffer() {
    CHECK(work_units_buffer_);
    return work_units_buffer_.get();
  }

  void destroyWorkUnitsBuffer();

 private:
  void createResources();

  uint32_t runCountPass(CommandList& cmd_list,
                        const uint64_t base_address,
                        const uint32_t num_rows,
                        const uint32_t num_columns,
                        const uint32_t column_index,
                        const uint32_t value_byte_size);

  void runBuildPass(CommandList& cmd_list,
                    const uint64_t base_address,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const uint32_t column_index,
                    const uint32_t value_byte_size);

  struct WorkUnit {
    // row index into QOB
    uint32_t row_index;
    // packed value index base and count
    // assuming a WORKGROUP_SIZE of 32 this is:
    //   bottom 5 bits are count (0 means 32)
    //   upper 27 bits are base
    uint32_t value_index_base_and_count;
  };

  const GfxContext& gfx_context_;
  ResourceManager& rsrc_mgr_;
  uint32_t subgroup_size_;
  uint32_t max_workgroup_count_;

  gfx::MaterialUqPtr count_material_;
  gfx::MaterialUqPtr build_material_;

  gfx::resource_ptr<gfx::ComputePipeline> count_pipeline_;
  gfx::resource_ptr<gfx::ComputePipeline> build_pipeline_;

  gfx::BufferWrapperUqPtr atomic_buffer_;
  gfx::BufferWrapperUqPtr work_units_buffer_;
};

}  // namespace gfx
