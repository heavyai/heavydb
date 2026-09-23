/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/BaseEmbeddedDataTable.h"

#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

BaseEmbeddedDataTable::BaseEmbeddedDataTable(
    QueryRendererContext& ctx,
    const std::string& name,
    const JSONLocation& json_loc,
    const RenderQuerySpecialtyType render_query_type)
    : JSONRefObject(ctx, RefType::kData, name, json_loc.getPathRef()) {}

}  // namespace QueryRenderer
