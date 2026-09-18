/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <vector>

class RelAlgNode;
class RelLeftDeepInnerJoin;
class RexScalar;

// Gets the start node of a left-deep pattern starting at the node itself or its child.
std::shared_ptr<const RelAlgNode> get_left_deep_join_root(
    const std::shared_ptr<RelAlgNode>& node);

void create_left_deep_join(std::vector<std::shared_ptr<RelAlgNode>>& nodes);

void rebind_inputs_from_left_deep_join(const RexScalar* rex,
                                       const RelLeftDeepInnerJoin* left_deep_join);
