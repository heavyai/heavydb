/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Types.h"
#include "Shared/sqltypes.h"

namespace QueryRenderer {

SQLTypeInfo get_float_equivalent_type(const SQLTypeInfo& type);
QueryDataType get_float_equivalent_type(const QueryDataType data_type);

// query_sql_type_to_render_type is a utility function for
// converting SQLTypeInfo objects to BufferAttrTypes for rendering.
// This function should be specifically used by all data coming from via
// a query result set. Data in a result set (or data from in-situ data)
// is currently always placed into 64-bit chunks, so this function handles
// that conversion.
// So, bool -> int64, int -> int64, dict-encoded str -> int64, float->double, etc.
gfx::BufferAttrType query_sql_type_to_render_type(const std::string& attr,
                                                  const SQLTypeInfo& ti);

// this utility conversion function does a default 1-to-1 conversion from SQLTypeInfo
// to BufferAttrType
// So, bool->bool, int->int, bigint->int64, float->float, etc.
gfx::BufferAttrType sql_type_to_render_type(const std::string& attr,
                                            const SQLTypeInfo& ti);
SQLTypeInfo render_type_to_sql_type(const gfx::BufferAttrType buffer_attr_type);

}  // namespace QueryRenderer
