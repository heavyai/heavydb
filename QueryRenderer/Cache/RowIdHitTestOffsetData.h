/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include "QueryRenderer/Cache/HitTestCacheResults.h"
#include "QueryRenderer/Interface/RenderQueryExecuteData.h"
#include "QueryRenderer/Interface/RenderQueryInterfaceDeclarations.h"

namespace QueryRenderer {

struct RowIdHitTestOffsetData {
  RowIdHitTestOffsetData(const std::string& in_sql_str,
                         const RenderQueryOutput& render_query_output);

  RowIdHitTestOffsetData() = delete;
  RowIdHitTestOffsetData(const RowIdHitTestOffsetData&) = delete;

  const std::vector<const Analyzer::TargetEntry*> rowid_cols;
  const std::vector<const SelectedTableInfo*> rowid_tables;
  const std::vector<uint32_t> rowid_offsets;
  const std::string& sql_str;

  HitTestTableContainer unpackRowId(const int64_t rowid_to_unpack,
                                    const PhysicalTableInfoContainer& used_tables) const;
};

}  // namespace QueryRenderer
