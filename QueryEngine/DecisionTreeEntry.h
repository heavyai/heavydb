/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

struct DecisionTreeEntry {
  double value;
  int64_t feature_index;
  int64_t left_child_row_idx;
  int64_t right_child_row_idx;

  // Constructor for a split node. Note that right_child_row_idx won't
  // be known until later in the depth-first traversal that constructs
  // the table of DecisionTreeEntrys, at which point that value will
  // be populated
  DecisionTreeEntry(const double value,
                    const int64_t feature_index,
                    const int64_t left_child_row_idx)
      : value(value)
      , feature_index(feature_index)
      , left_child_row_idx(left_child_row_idx) {}

  // Constructor for a terminal/non-split node
  DecisionTreeEntry(const double value) : value(value), feature_index(-1) {}

  inline bool isSplitNode() const { return feature_index >= 0; }
};
