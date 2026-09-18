/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* This library is built as a stand-alone shared library to test our
runtime dynamically loaded library support. The library is loaded at
runtime using boost::shared_library (which is essentially a wrapper
around dlopen()). Then, we register table functions that use functions
defined in this library at runtime, allowing for table functions to
use code from optional "plugin" libraries. */

#include <cstdint>

template <typename T>
T _test_runtime_add(T x, T y);
template <typename T>
T _test_runtime_sub(T x, T y);