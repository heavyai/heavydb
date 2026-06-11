/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "distributed.h"
#include <cstdint>

namespace dist {
bool is_distributed() {
  return g_cluster;
}

bool is_first_leaf() {
  return (is_distributed() && g_distributed_leaf_idx == 0);
}

bool is_leaf_node() {
  return (is_distributed() && g_distributed_leaf_idx >= 0);
}

bool is_aggregator() {
  return (is_distributed() && g_distributed_leaf_idx == -1);
}
}  // namespace dist
