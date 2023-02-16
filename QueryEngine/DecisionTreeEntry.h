/*
 * Copyright 2023 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

struct DecisionTreeEntry {
  double value;
  int64_t feature_index;
  int64_t left_child_row_idx;
  int64_t right_child_row_idx;

  DecisionTreeEntry(const double value,
                    const int64_t feature_index,
                    const int64_t left_child_row_idx)
      : value(value)
      , feature_index(feature_index)
      , left_child_row_idx(left_child_row_idx) {}
  DecisionTreeEntry(const double value) : value(value), feature_index(-1) {}

  inline bool isSplitNode() const { return feature_index >= 0; }
};
