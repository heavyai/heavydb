/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace shared {

constexpr bool isPowOfTwo(size_t const n) {
  return n != 0u && (n & (n - 1u)) == 0u;
}

unsigned getExpOfTwo(unsigned n);

}  // namespace shared
