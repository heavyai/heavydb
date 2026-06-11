/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace foreign_storage {

struct ColumnType {};

struct FragmentType {};

template <typename T>
struct Interval {
  int start, end;
};

}  // namespace foreign_storage
