/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/RowDataTableGpuResources.h"
#include "QueryRenderer/Data/Types.h"

namespace QueryRenderer {

//
// class BaseRowDataTable
//
// Base class for unspecialized row based DataTable types
//
class BaseRowDataTable : public BaseDataTable {
 public:
  BaseRowDataTable(DataInputFormat input_format);
  ~BaseRowDataTable() override = default;

  RowDataTableGpuResources& getGpuResources() const;

 protected:
  std::unique_ptr<RowDataTableGpuResources> gpu_resources_;
};

}  // namespace QueryRenderer
