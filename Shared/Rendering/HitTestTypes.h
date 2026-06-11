/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <regex>

struct HitTestTypes {
  static constexpr int max_num_rowids() { return 3; }
  static std::regex get_rowid_regex() {
    static_assert(max_num_rowids() < 10, "rowid regex only supports < 10 rowids");
    return std::regex("^rowid([0-" + std::to_string(max_num_rowids() - 2) + "])?$");
  }
};
