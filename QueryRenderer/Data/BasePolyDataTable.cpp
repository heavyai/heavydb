/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/BasePolyDataTable.h"

namespace QueryRenderer {

std::string BasePolyDataTable::x_coord_name = "x";
std::string BasePolyDataTable::y_coord_name = "y";

BasePolyDataTable::BasePolyDataTable(DataInputFormat input_format)
    : BaseDataTable(input_format, DataOutputFormat::kPolys)
    , gpu_resources_{std::make_unique<PolyDataTableGpuResources>(input_format)} {}

std::vector<GpuId> BasePolyDataTable::getUsedGpuIds() const {
  return gpu_resources_->getGpuDataMap().getGpuIds();
}

PolyDataTableGpuResources& BasePolyDataTable::getGpuResources() const {
  return *gpu_resources_;
}

}  // namespace QueryRenderer
