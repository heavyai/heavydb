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

#include "QueryEngine/Descriptors/QueryMemoryDescriptor.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/RelAlgExecutionUnit.h"

class Executor;

class MemoryLayoutBuilder {
 public:
  MemoryLayoutBuilder(const RelAlgExecutionUnit& ra_exe_unit);

  std::unique_ptr<QueryMemoryDescriptor> build(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<InputTableInfo>& query_infos,
      const bool allow_multifrag,
      const size_t max_groups_buffer_entry_count,
      const int8_t crt_min_byte_width,
      const bool output_columnar_hint,
      const bool just_explain,
      std::optional<int64_t> group_cardinality_estimation,
      Executor* executor,
      const ExecutorDeviceType device_type);

  size_t cudaSharedMemorySize(const RelAlgExecutionUnit& ra_exe_unit,
                              QueryMemoryDescriptor* query_mem_desc,
                              const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                              Executor* executor,
                              const ExecutorDeviceType device_type) const;
};