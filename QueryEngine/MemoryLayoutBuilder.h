/*
 * Copyright 2022 Intel Corporation.
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "DataMgr/GpuMgr.h"
#include "QueryEngine/Descriptors/QueryMemoryDescriptor.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/RelAlgExecutionUnit.h"

class Executor;

/**
 * @brief Determines memory layout for a given RelAlgExecutionUnit and builds
 * QueryMemoryDescriptor, which conveys memory layout information.
 *
 */
class MemoryLayoutBuilder {
 public:
  MemoryLayoutBuilder(const RelAlgExecutionUnit& ra_exe_unit);

  std::unique_ptr<QueryMemoryDescriptor> build(
      const std::vector<InputTableInfo>& query_infos,
      const bool allow_multifrag,
      const size_t max_groups_buffer_entry_count,
      const int8_t crt_min_byte_width,
      const bool output_columnar_hint,
      const bool just_explain,
      std::optional<int64_t> group_cardinality_estimation,
      Executor* executor,
      const ExecutorDeviceType device_type);

  size_t gpuSharedMemorySize(QueryMemoryDescriptor* query_mem_desc,
                             const GpuMgr* gpu_mgr,
                             Executor* executor,
                             const ExecutorDeviceType device_type) const;

 private:
  const RelAlgExecutionUnit& ra_exe_unit_;
};

inline size_t get_count_distinct_sub_bitmap_count(const size_t bitmap_sz_bits,
                                                  const RelAlgExecutionUnit& ra_exe_unit,
                                                  const ExecutorDeviceType device_type) {
  // For count distinct on a column with a very small number of distinct values
  // contention can be very high, especially for non-grouped queries. We'll split
  // the bitmap into multiple sub-bitmaps which are unified to get the full result.
  // The threshold value for bitmap_sz_bits works well on Kepler.
  return bitmap_sz_bits < 50000 && ra_exe_unit.groupby_exprs.empty() &&
                 device_type == ExecutorDeviceType::GPU
             ? 64  // NB: must be a power of 2 to keep runtime offset computations cheap
             : 1;
}
