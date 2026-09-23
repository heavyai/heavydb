/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/Allocators/CpuMgrArenaAllocator.h"

namespace import_export {
// Use a vector that that utilizes managed memory
template <typename T>
using vector = managed_memory::vector<T>;
}  // namespace import_export
