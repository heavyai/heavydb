/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace QueryRenderer {

using TableId = int32_t;  // should be the same as TableDescriptor::tableId in
                          // <Catalog/TableDescriptor.h>

using ColumnId = int32_t;  // should be the same as ColumnDescriptor::columnId in
                           // <Catalog/ColumnDescriptor.h>

}  // namespace QueryRenderer
