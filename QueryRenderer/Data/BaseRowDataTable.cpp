/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/BaseRowDataTable.h"
#include "QueryRenderer/Data/RowDataTableGpuResources.h"
#include "QueryRenderer/Data/Types.h"

namespace QueryRenderer {

BaseRowDataTable::BaseRowDataTable(DataInputFormat input_format)
    : BaseDataTable(input_format, DataOutputFormat::kRows)
    , gpu_resources_{std::make_unique<RowDataTableGpuResources>(input_format)} {}

RowDataTableGpuResources& BaseRowDataTable::getGpuResources() const {
  return *gpu_resources_;
}

}  // namespace QueryRenderer
