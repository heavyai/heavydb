/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>
#include <string>

namespace QueryRenderer {

enum class QueryBufferType { kVertex, kIndex, kStorage, kIndirectVertex, kIndirectIndex };

std::string to_string(QueryBufferType type);

std::ostream& operator<<(std::ostream& os, const QueryBufferType& type);

}  // namespace QueryRenderer
