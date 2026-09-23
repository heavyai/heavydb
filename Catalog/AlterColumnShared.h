/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <list>

#include "Catalog/ColumnDescriptor.h"

namespace alter_column_shared {
using TypePairs = std::list<std::pair<const ColumnDescriptor*, ColumnDescriptor*>>;
}
