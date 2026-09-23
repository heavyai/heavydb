/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/LineDataTableGpuResources.h"

namespace QueryRenderer {

//
// class BaseLineDataTable
//
// Base class for SqlLineDataTableJSON (query based) and
// EmbeddedLineDataTable (embedded data)
//
class BaseLineDataTable : public BaseDataTable {
 public:
  static const std::string x_coord_name;
  static const std::string y_coord_name;

  BaseLineDataTable(DataInputFormat input_format);
  ~BaseLineDataTable() override = default;

  LineDataTableGpuResources& getGpuResources() const;

 protected:
  std::unique_ptr<LineDataTableGpuResources> gpu_resources_;
};

}  // namespace QueryRenderer
