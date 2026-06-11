/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TestRuntimeLib.h"

template <typename T>
T _test_runtime_add(T x, T y) {
  return x + y;
}

template <typename T>
T _test_runtime_sub(T x, T y) {
  return x - y;
}

template int64_t _test_runtime_add(int64_t, int64_t);
template double _test_runtime_add(double, double);
template int64_t _test_runtime_sub(int64_t, int64_t);
template double _test_runtime_sub(double, double);