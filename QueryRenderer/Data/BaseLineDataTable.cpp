/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/BaseLineDataTable.h"

namespace QueryRenderer {

const std::string BaseLineDataTable::x_coord_name = "x";
const std::string BaseLineDataTable::y_coord_name = "y";

BaseLineDataTable::BaseLineDataTable(DataInputFormat input_format)
    : BaseDataTable(input_format, DataOutputFormat::kLines)
    , gpu_resources_{std::make_unique<LineDataTableGpuResources>(input_format)} {}

LineDataTableGpuResources& BaseLineDataTable::getGpuResources() const {
  return *gpu_resources_;
}

}  // namespace QueryRenderer
