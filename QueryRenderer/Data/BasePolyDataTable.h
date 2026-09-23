/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <memory>

#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/PolyDataTableGpuResources.h"

namespace QueryRenderer {

//
// class BasePolyDataTable
//
// Base class for SqlPolyDataTableJSON (query based) and
// EmbeddedPolyDataTable (inline data)
//
class BasePolyDataTable : public BaseDataTable {
 public:
  static std::string x_coord_name;
  static std::string y_coord_name;

  BasePolyDataTable(DataInputFormat input_format);
  ~BasePolyDataTable() override = default;

  std::vector<GpuId> getUsedGpuIds() const override;

  PolyDataTableGpuResources& getGpuResources() const;

 protected:
  std::unique_ptr<PolyDataTableGpuResources> gpu_resources_;
};

}  // namespace QueryRenderer
