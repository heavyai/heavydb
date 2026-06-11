/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

struct ApproxQuantileDescriptor {
  size_t buffer_size;     // number of elements in TDigest buffer
  size_t centroids_size;  // number of elements in TDigest centroids
};

using ApproxQuantileDescriptors = std::vector<ApproxQuantileDescriptor>;
