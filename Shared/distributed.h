/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

extern int32_t g_distributed_leaf_idx;
extern int32_t g_distributed_num_leaves;
extern bool g_cluster;

namespace dist {
bool is_distributed();
bool is_first_leaf();
bool is_leaf_node();
bool is_aggregator();
}  // namespace dist
