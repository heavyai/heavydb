/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

namespace shared {

unsigned getExpOfTwo(unsigned n) {
  unsigned i = 0;

  while ((n = n >> 1)) {
    ++i;
  }

  return i;
}

}  // namespace shared
