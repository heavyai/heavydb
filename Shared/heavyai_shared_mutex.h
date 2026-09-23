/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>
#include <shared_mutex>

namespace heavyai {
using shared_mutex = std::shared_timed_mutex;
template <typename T>
using lock_guard = std::lock_guard<T>;
template <typename T>
using unique_lock = std::unique_lock<T>;
template <typename T>
using shared_lock = std::shared_lock<T>;
}  // namespace heavyai
