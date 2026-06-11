/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>

using IndexPair = std::pair<int64_t, int64_t>;

template <typename T>
struct SumAndCountPair {
  T sum;
  size_t count;
};
