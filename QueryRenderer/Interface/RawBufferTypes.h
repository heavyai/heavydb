/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace QueryRenderer {

using RowIdVector = std::vector<uint32_t>;
using TableIdVector = std::vector<uint32_t>;
using AccumDataVector = std::vector<uint32_t>;
using IdBufferTuple = std::tuple<RowIdVector, RowIdVector, TableIdVector>;

}  // namespace QueryRenderer
