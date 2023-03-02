/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#ifndef __CUDACC__
#include <sstream>
#endif

#include "SegmentTreeUtils.h"
#include "Shared/checked_alloc.h"
#include "Shared/sqldefs.h"
#include "Shared/sqltypes.h"

// A generic segment tree class that builds a segment tree of a given input_col_buf
// with a fan_out
// depending on what aggregation operator we use, it constructs internal node differently
// i.e., for sum aggregation, a parent node becomes a sum of "all" child elements
template <typename INPUT_TYPE, typename AGG_TYPE>
class SegmentTree {
 public:
  SegmentTree(std::vector<const int8_t*> const& input_col_bufs,
              SQLTypeInfo const& input_col_ti,
              int32_t const* original_input_col_idx_buf,
              int64_t const* ordered_input_col_idx_buf,
              int64_t num_elems,
              SqlWindowFunctionKind agg_type,
              size_t fan_out)
      : input_col_buf_(!input_col_bufs.empty()
                           ? reinterpret_cast<const INPUT_TYPE*>(input_col_bufs.front())
                           : nullptr)
      , cond_col_buf_(input_col_bufs.size() == 2 ? input_col_bufs[1] : nullptr)
      , input_col_ti_(input_col_ti)
      , original_input_col_idx_buf_(original_input_col_idx_buf)
      , ordered_input_col_idx_buf_(ordered_input_col_idx_buf)
      , num_elems_(num_elems)
      , fan_out_(fan_out)
      , agg_type_(agg_type)
      , is_conditional_agg_(agg_type == SqlWindowFunctionKind::SUM_IF)
      , input_type_null_val_(inline_null_value<INPUT_TYPE>())
      , null_val_(inline_null_value<AGG_TYPE>())
      , invalid_val_(agg_type_ == SqlWindowFunctionKind::MIN
                         ? std::numeric_limits<INPUT_TYPE>::max()
                     : agg_type_ == SqlWindowFunctionKind::MAX
                         ? std::numeric_limits<INPUT_TYPE>::min()
                         : 0) {
    CHECK_GT(num_elems_, (int64_t)0);
    auto max_tree_height_and_leaf_range = findMaxTreeHeight(num_elems_, fan_out_);
    leaf_depth_ = max_tree_height_and_leaf_range.first;
    leaf_range_ = max_tree_height_and_leaf_range.second;
    // the index of the last elem of the leaf level is the same as the tree's size
    tree_size_ = leaf_range_.second;
    leaf_size_ = leaf_range_.second - leaf_range_.first;
    // for derived aggregate, we maintain both "sum" aggregates and element counts
    // to compute the value correctly
    if (agg_type_ == SqlWindowFunctionKind::AVG) {
      derived_aggregated_ = reinterpret_cast<SumAndCountPair<AGG_TYPE>*>(
          checked_malloc(tree_size_ * sizeof(SumAndCountPair<AGG_TYPE>)));
      buildForDerivedAggregate(0, 0);
    } else {
      // we can use an array as a segment tree for the rest of aggregates
      aggregated_values_ =
          reinterpret_cast<AGG_TYPE*>(checked_malloc(tree_size_ * sizeof(AGG_TYPE)));
      switch (agg_type_) {
        case SqlWindowFunctionKind::COUNT:
          buildForCount(0, 0);
          break;
        case SqlWindowFunctionKind::CONDITIONAL_CHANGE_EVENT:
          buildForConditionalChangeEvent(0, 0);
          break;
        default: {
          if (is_conditional_agg_) {
            buildForCondAgg(0, 0);
          } else {
            build(0, 0);
          }
        }
      }
    }
  }

  ~SegmentTree() {
    if (num_elems_ > 0) {
      if (agg_type_ == SqlWindowFunctionKind::AVG) {
        free(derived_aggregated_);
      } else {
        free(aggregated_values_);
      }
    }
  }

  // try to aggregate values of the given query range
  AGG_TYPE query(const IndexPair& query_range) const {
    if (query_range.first > query_range.second || query_range.first < 0 ||
        query_range.second > leaf_size_) {
      return null_val_;
    }
    // todo (yoonmin) : support more derived aggregate functions
    if (agg_type_ == SqlWindowFunctionKind::AVG) {
      SumAndCountPair<AGG_TYPE> sum_and_count_pair =
          searchForDerivedAggregate(query_range, 0, 0, 0, leaf_size_);
      if (sum_and_count_pair.sum == null_val_) {
        return null_val_;
      } else if (sum_and_count_pair.count == 0) {
        return 0;
      } else {
        if (input_col_ti_.is_decimal()) {
          return (static_cast<double>(sum_and_count_pair.sum) /
                  pow(10, input_col_ti_.get_scale())) /
                 sum_and_count_pair.count;
        } else {
          return sum_and_count_pair.sum / sum_and_count_pair.count;
        }
      }
    } else {
      const auto res = search(query_range, 0, 0, 0, leaf_size_);
      if (res == null_val_) {
        switch (agg_type_) {
          case SqlWindowFunctionKind::MIN:
          case SqlWindowFunctionKind::MAX:
            return input_type_null_val_;
          default:
            return null_val_;
        }
      }
      return res;
    }
  }

  AGG_TYPE* getAggregatedValues() const { return aggregated_values_; }

  SumAndCountPair<AGG_TYPE>* getDerivedAggregatedValues() const {
    return derived_aggregated_;
  }

  size_t getLeafSize() const { return leaf_size_; }

  size_t getTreeSize() const { return tree_size_; }

  size_t getNumElems() const { return num_elems_; }

  size_t getLeafDepth() const { return leaf_depth_; }

  size_t getTreeFanout() const { return fan_out_; }

  IndexPair getLeafRange() const { return leaf_range_; }

 private:
  // build a segment tree for a given aggregate function recursively
  inline AGG_TYPE fetch_col_for_count(size_t const idx) {
    return input_col_buf_ && input_col_buf_[idx] == input_type_null_val_ ? null_val_ : 1;
  }

  inline AGG_TYPE fetch_col_for_non_cond_agg(size_t const idx) {
    auto const col_val = input_col_buf_[idx];
    return col_val != input_type_null_val_ ? col_val : null_val_;
  }

  inline AGG_TYPE fetch_col_for_cond_agg(size_t const idx) {
    AGG_TYPE res{null_val_};
    if (cond_col_buf_[idx] == 1) {
      auto const col_val = input_col_buf_[idx];
      res = col_val != input_type_null_val_ ? col_val : null_val_;
    }
    return res;
  }

  AGG_TYPE build(int64_t cur_node_idx, size_t cur_node_depth) {
    if (cur_node_idx >= leaf_range_.first && cur_node_idx < leaf_range_.second) {
      // arrive at leaf level, so try to put a corresponding input column value
      int64_t input_col_idx = cur_node_idx - leaf_range_.first;
      if (input_col_idx >= num_elems_) {
        // fill an invalid value
        aggregated_values_[cur_node_idx] = invalid_val_;
        return invalid_val_;
      }
      // try to get the current row's column value
      const auto input_col_idx_ordered = ordered_input_col_idx_buf_[input_col_idx];
      const auto refined_input_col_idx =
          original_input_col_idx_buf_[input_col_idx_ordered];
      aggregated_values_[cur_node_idx] =
          fetch_col_for_non_cond_agg(refined_input_col_idx);
      // return the current value to fill a parent node
      return aggregated_values_[cur_node_idx];
    }

    // when reaching here, we need to take care of a node having child nodes,
    // and we compute an aggregated value from its child nodes
    std::vector<AGG_TYPE> child_vals =
        prepareChildValuesforAggregation(cur_node_idx, cur_node_depth);

    // compute the new aggregated value
    aggregated_values_[cur_node_idx] = aggregateValue(child_vals);

    // return the value for the upper-level aggregation
    return aggregated_values_[cur_node_idx];
  }

  AGG_TYPE buildForCondAgg(int64_t cur_node_idx, size_t cur_node_depth) {
    if (cur_node_idx >= leaf_range_.first && cur_node_idx < leaf_range_.second) {
      // arrive at leaf level, so try to put a corresponding input column value
      int64_t input_col_idx = cur_node_idx - leaf_range_.first;
      if (input_col_idx >= num_elems_) {
        // fill an invalid value
        aggregated_values_[cur_node_idx] = invalid_val_;
        return invalid_val_;
      }
      // try to get the current row's column value
      const auto input_col_idx_ordered = ordered_input_col_idx_buf_[input_col_idx];
      const auto refined_input_col_idx =
          original_input_col_idx_buf_[input_col_idx_ordered];
      aggregated_values_[cur_node_idx] = fetch_col_for_cond_agg(refined_input_col_idx);
      // return the current value to fill a parent node
      return aggregated_values_[cur_node_idx];
    }

    // when reaching here, we need to take care of a node having child nodes,
    // and we compute an aggregated value from its child nodes
    std::vector<AGG_TYPE> child_vals =
        prepareChildValuesforConditionalAggregation(cur_node_idx, cur_node_depth);

    // compute the new aggregated value
    aggregated_values_[cur_node_idx] = aggregateValue(child_vals);

    // return the value for the upper-level aggregation
    return aggregated_values_[cur_node_idx];
  }

  AGG_TYPE buildForCount(int64_t cur_node_idx, size_t cur_node_depth) {
    if (cur_node_idx >= leaf_range_.first && cur_node_idx <= leaf_range_.second) {
      // arrive at leafs, so try to put a corresponding input column value
      int64_t input_col_idx = cur_node_idx - leaf_range_.first;
      if (input_col_idx >= num_elems_) {
        // fill an invalid value
        aggregated_values_[cur_node_idx] = invalid_val_;
        return invalid_val_;
      }
      // try to get the current row's column value
      const auto input_col_idx_ordered = ordered_input_col_idx_buf_[input_col_idx];
      const auto refined_input_col_idx =
          original_input_col_idx_buf_[input_col_idx_ordered];
      aggregated_values_[cur_node_idx] = fetch_col_for_count(refined_input_col_idx);
      // return the current value to fill a parent node
      return aggregated_values_[cur_node_idx];
    }

    // when reaching here, we need to take care of a node having child nodes,
    // and we compute an aggregated value from its child nodes
    std::vector<AGG_TYPE> child_vals =
        prepareChildValuesforAggregationForCount(cur_node_idx, cur_node_depth);

    // compute the new aggregated value
    aggregated_values_[cur_node_idx] = aggregateValue(child_vals);

    // return the value for the upper-level aggregation
    return aggregated_values_[cur_node_idx];
  }

  AGG_TYPE buildForConditionalChangeEvent(int64_t cur_node_idx, size_t cur_node_depth) {
    if (cur_node_idx >= leaf_range_.first && cur_node_idx <= leaf_range_.second) {
      int64_t input_col_idx = cur_node_idx - leaf_range_.first;
      if (input_col_idx >= num_elems_) {
        aggregated_values_[cur_node_idx] = invalid_val_;
        return invalid_val_;
      }
      auto cur_row_idx =
          original_input_col_idx_buf_[ordered_input_col_idx_buf_[input_col_idx]];
      if (input_col_idx == 0) {
        aggregated_values_[cur_node_idx] = 0;
      } else {
        auto prev_row_idx =
            original_input_col_idx_buf_[ordered_input_col_idx_buf_[input_col_idx - 1]];
        auto const col_val = input_col_buf_[cur_row_idx];
        auto const prev_col_val = input_col_buf_[prev_row_idx];
        aggregated_values_[cur_node_idx] =
            col_val != input_type_null_val_ && prev_col_val != input_type_null_val_
                ? col_val != prev_col_val
                : 0;
      }
      return aggregated_values_[cur_node_idx];
    }

    // when reaching here, we need to take care of a node having child nodes,
    // and we compute an aggregated value from its child nodes
    std::vector<AGG_TYPE> child_vals =
        prepareChildValuesforAggregationForConditionalChangeEvent(cur_node_idx,
                                                                  cur_node_depth);

    // compute the new aggregated value
    aggregated_values_[cur_node_idx] = aggregateValue(child_vals);

    // return the value for the upper-level aggregation
    return aggregated_values_[cur_node_idx];
  }

  // the logic is exactly the same as normal aggregate case, but has a different
  // node type: SumAndCountPair
  SumAndCountPair<AGG_TYPE> buildForDerivedAggregate(int64_t cur_node_idx,
                                                     size_t cur_node_depth) {
    if (cur_node_idx >= leaf_range_.first && cur_node_idx <= leaf_range_.second) {
      int64_t input_col_idx = cur_node_idx - leaf_range_.first;
      if (input_col_idx >= num_elems_) {
        derived_aggregated_[cur_node_idx] = {invalid_val_, 0};
      } else {
        const auto input_col_idx_ordered = ordered_input_col_idx_buf_[input_col_idx];
        const auto refined_input_col_idx =
            original_input_col_idx_buf_[input_col_idx_ordered];
        auto col_val = input_col_buf_[refined_input_col_idx];
        if (col_val != input_type_null_val_) {
          derived_aggregated_[cur_node_idx] = {col_val, 1};
        } else {
          derived_aggregated_[cur_node_idx] = {null_val_, 0};
        }
      }
      return derived_aggregated_[cur_node_idx];
    }

    std::vector<SumAndCountPair<AGG_TYPE>> child_vals =
        prepareChildValuesforDerivedAggregate(cur_node_idx, cur_node_depth);

    derived_aggregated_[cur_node_idx] = aggregateValueForDerivedAggregate(child_vals);
    return derived_aggregated_[cur_node_idx];
  }

  // gather aggregated values of each child node
  std::vector<AGG_TYPE> prepareChildValuesforAggregation(int64_t parent_idx,
                                                         size_t cur_node_depth) {
    std::vector<AGG_TYPE> child_vals(fan_out_);
    int64_t const offset = cur_node_depth ? parent_idx * fan_out_ + 1 : 1;
    size_t const next_node_depth = cur_node_depth + 1;
    for (size_t i = 0; i < fan_out_; ++i) {
      child_vals[i] = build(offset + i, next_node_depth);
    }
    return child_vals;
  }

  // gather aggregated values of each child node
  std::vector<AGG_TYPE> prepareChildValuesforConditionalAggregation(
      int64_t parent_idx,
      size_t cur_node_depth) {
    std::vector<AGG_TYPE> child_vals(fan_out_);
    int64_t const offset = cur_node_depth ? parent_idx * fan_out_ + 1 : 1;
    size_t const next_node_depth = cur_node_depth + 1;
    for (size_t i = 0; i < fan_out_; ++i) {
      child_vals[i] = buildForCondAgg(offset + i, next_node_depth);
    }
    return child_vals;
  }

  // gather aggregated values of each child node
  std::vector<AGG_TYPE> prepareChildValuesforAggregationForCount(int64_t parent_idx,
                                                                 size_t cur_node_depth) {
    std::vector<AGG_TYPE> child_vals(fan_out_);
    int64_t const offset = cur_node_depth ? parent_idx * fan_out_ + 1 : 1;
    size_t const next_node_depth = cur_node_depth + 1;
    for (size_t i = 0; i < fan_out_; ++i) {
      child_vals[i] = buildForCount(offset + i, next_node_depth);
    }
    return child_vals;
  }

  // gather aggregated values of each child node
  std::vector<AGG_TYPE> prepareChildValuesforAggregationForConditionalChangeEvent(
      int64_t const parent_idx,
      size_t const cur_node_depth) {
    std::vector<AGG_TYPE> child_vals(fan_out_);
    int64_t const offset = cur_node_depth ? parent_idx * fan_out_ + 1 : 1;
    size_t const next_node_depth = cur_node_depth + 1;
    for (size_t i = 0; i < fan_out_; ++i) {
      child_vals[i] = buildForConditionalChangeEvent(offset + i, next_node_depth);
    }
    return child_vals;
  }

  std::vector<SumAndCountPair<AGG_TYPE>> prepareChildValuesforDerivedAggregate(
      int64_t parent_idx,
      size_t cur_node_depth) {
    std::vector<SumAndCountPair<AGG_TYPE>> child_vals(fan_out_);
    int64_t const offset = cur_node_depth ? parent_idx * fan_out_ + 1 : 1;
    size_t const next_node_depth = cur_node_depth + 1;
    for (size_t i = 0; i < fan_out_; ++i) {
      child_vals[i] = buildForDerivedAggregate(offset + i, next_node_depth);
    }
    return child_vals;
  }

  // compute aggregated value of the given values depending on the aggregate function
  AGG_TYPE aggregateValue(const std::vector<AGG_TYPE>& vals) const {
    // todo (yoonmin) : optimize logic for a non-null column
    bool all_nulls = true;
    if (agg_type_ == SqlWindowFunctionKind::MIN) {
      AGG_TYPE min_val = std::numeric_limits<AGG_TYPE>::max();
      std::for_each(
          vals.begin(), vals.end(), [&min_val, &all_nulls, this](const AGG_TYPE val) {
            if (val != null_val_ && val != invalid_val_) {
              all_nulls = false;
              min_val = std::min(min_val, val);
            }
          });
      if (all_nulls) {
        min_val = null_val_;
      }
      return min_val;
    } else if (agg_type_ == SqlWindowFunctionKind::MAX) {
      AGG_TYPE max_val = std::numeric_limits<AGG_TYPE>::min();
      std::for_each(
          vals.begin(), vals.end(), [&max_val, &all_nulls, this](const AGG_TYPE val) {
            if (val != null_val_ && val != invalid_val_) {
              all_nulls = false;
              max_val = std::max(max_val, val);
            }
          });
      if (all_nulls) {
        max_val = null_val_;
      }
      return max_val;
    } else {
      AGG_TYPE agg_val = 0;
      std::for_each(
          vals.begin(), vals.end(), [&agg_val, &all_nulls, this](const AGG_TYPE val) {
            if (val != null_val_) {
              agg_val += val;
              all_nulls = false;
            }
          });
      if (all_nulls) {
        agg_val = null_val_;
      }
      return agg_val;
    }
  }

  AGG_TYPE aggregateValueViaColumnAccess(int64_t cur_col_idx, size_t num_visits) const {
    // todo (yoonmin) : optimize logic for a non-null column
    bool all_nulls = true;
    auto end_idx = cur_col_idx + num_visits;
    if (agg_type_ == SqlWindowFunctionKind::MIN) {
      AGG_TYPE min_val = std::numeric_limits<AGG_TYPE>::max();
      for (size_t i = cur_col_idx; i < end_idx; ++i) {
        AGG_TYPE val = aggregated_values_[i];
        if (val != null_val_ && val != invalid_val_) {
          all_nulls = false;
          min_val = std::min(min_val, val);
        }
      }
      if (all_nulls) {
        min_val = null_val_;
      }
      return min_val;
    } else if (agg_type_ == SqlWindowFunctionKind::MAX) {
      AGG_TYPE max_val = std::numeric_limits<AGG_TYPE>::min();
      for (size_t i = cur_col_idx; i < end_idx; ++i) {
        AGG_TYPE val = aggregated_values_[i];
        if (val != null_val_ && val != invalid_val_) {
          all_nulls = false;
          max_val = std::max(max_val, val);
        }
      }
      if (all_nulls) {
        max_val = null_val_;
      }
      return max_val;
    } else {
      AGG_TYPE agg_val = 0;
      for (size_t i = cur_col_idx; i < end_idx; ++i) {
        AGG_TYPE val = aggregated_values_[i];
        if (val != null_val_) {
          agg_val += val;
          all_nulls = false;
        }
      }
      if (all_nulls) {
        agg_val = null_val_;
      }
      return agg_val;
    }
  }

  SumAndCountPair<AGG_TYPE> aggregateValueForDerivedAggregate(
      const std::vector<SumAndCountPair<AGG_TYPE>>& vals) const {
    SumAndCountPair<AGG_TYPE> res{0, 0};
    bool all_nulls = true;
    std::for_each(vals.begin(),
                  vals.end(),
                  [&res, &all_nulls, this](const SumAndCountPair<AGG_TYPE>& pair) {
                    if (pair.sum != null_val_ && pair.sum != invalid_val_) {
                      res.sum += pair.sum;
                      res.count += pair.count;
                      all_nulls = false;
                    }
                  });
    if (all_nulls) {
      return {null_val_, 0};
    }
    return res;
  }

  SumAndCountPair<AGG_TYPE> aggregateValueForDerivedAggregateViaColumnAccess(
      int64_t cur_col_idx,
      size_t num_visits) const {
    SumAndCountPair<AGG_TYPE> res{0, 0};
    bool all_nulls = true;
    auto end_idx = cur_col_idx + num_visits;
    for (size_t i = cur_col_idx; i < end_idx; ++i) {
      const SumAndCountPair<AGG_TYPE> cur_pair_val = aggregated_values_[i];
      if (cur_pair_val.sum != null_val_) {
        res.sum += cur_pair_val.sum;
        res.count += cur_pair_val.count;
        all_nulls = false;
      }
    }
    if (all_nulls) {
      return {null_val_, 0};
    }
    return res;
  }

  // search an aggregated value of the given query range
  // by visiting necessary nodes of the segment tree including leafs
  AGG_TYPE search(const IndexPair& query_range,
                  int64_t cur_node_idx,
                  size_t cur_node_depth,
                  int64_t search_range_start_idx,
                  int64_t search_range_end_idx) const {
    if (num_elems_ <= 0) {
      return null_val_;
    }
    if (search_range_end_idx < query_range.first ||
        query_range.second < search_range_start_idx) {
      // completely out-of-bound
      return invalid_val_;
    } else if (query_range.first <= search_range_start_idx &&
               search_range_end_idx <= query_range.second) {
      // perfectly fitted in a range of the current node
      return aggregated_values_[cur_node_idx];
    } else {
      // this node is partially within left and right indexes
      if (cur_node_depth == leaf_depth_) {
        // we already reach the leaf level, so do not need to proceed to the next level
        // and so aggregate the value in the range by using a simple loop
        size_t num_visits = query_range.second - search_range_start_idx + 1;
        return aggregateValueViaColumnAccess(
            cur_node_idx, num_visits, null_val_, invalid_val_);
      } else {
        // find aggregated value from child nodes
        int64_t pivot_idx = search_range_start_idx +
                            ((search_range_end_idx - search_range_start_idx) / fan_out_);
        int64_t child_search_start_idx = search_range_start_idx;
        int64_t child_search_end_idx = pivot_idx;
        std::vector<size_t> child_indexes(fan_out_);
        computeChildIndexes(child_indexes, cur_node_idx, cur_node_depth);
        std::vector<AGG_TYPE> child_vals(fan_out_);
        for (size_t i = 0; i < child_indexes.size(); ++i) {
          child_vals[i] = search(query_range,
                                 child_indexes[i],
                                 cur_node_depth + 1,
                                 child_search_start_idx,
                                 child_search_end_idx);
          child_search_start_idx = child_search_end_idx + 1;
          child_search_end_idx = child_search_start_idx + pivot_idx;
          if (child_search_end_idx > search_range_end_idx) {
            child_search_end_idx = search_range_end_idx;
          }
        }
        return aggregateValue(child_vals);
      }
    }
  }

  SumAndCountPair<AGG_TYPE> searchForDerivedAggregate(
      const IndexPair& query_range,
      int64_t cur_node_idx,
      size_t cur_node_depth,
      int64_t search_range_start_idx,
      int64_t search_range_end_idx) const {
    if (search_range_end_idx < query_range.first ||
        query_range.second < search_range_start_idx) {
      return {invalid_val_, 0};
    } else if (query_range.first <= search_range_start_idx &&
               search_range_end_idx <= query_range.second) {
      return derived_aggregated_[cur_node_idx];
    } else {
      if (cur_node_depth == leaf_depth_) {
        size_t num_visits = query_range.second - search_range_start_idx + 1;
        return aggregateValueForDerivedAggregateViaColumnAccess(
            cur_node_idx, num_visits, null_val_, invalid_val_);
      } else {
        // find aggregated value from child nodes
        std::vector<int64_t> child_indexes(fan_out_);
        computeChildIndexes(child_indexes, cur_node_idx, cur_node_depth);
        std::vector<SumAndCountPair<AGG_TYPE>> child_vals(fan_out_, {invalid_val_, 0});
        int64_t pivot_idx = search_range_start_idx +
                            ((search_range_end_idx - search_range_start_idx) / fan_out_);
        int64_t child_search_start_idx = search_range_start_idx;
        int64_t child_search_end_idx = pivot_idx;
        for (size_t i = 0; i < child_indexes.size(); ++i) {
          child_vals[i] = searchForDerivedAggregate(query_range,
                                                    child_indexes[i],
                                                    cur_node_depth + 1,
                                                    child_search_start_idx,
                                                    child_search_end_idx);
          child_search_start_idx = child_search_end_idx + 1;
          child_search_end_idx = child_search_start_idx + pivot_idx;
          if (child_search_end_idx > search_range_end_idx) {
            child_search_end_idx = search_range_end_idx;
          }
        }
        return aggregateValueForDerivedAggregate(child_vals);
      }
    }
  }

  // for prepareChildValuesForAggregate function includes the logic of the
  // `computeChildIndexes` function internally to reduce # creation of the temporary
  // vector while building a segment tree, but keep it to use it for `search` function
  std::vector<int64_t> computeChildIndexes(std::vector<int64_t>& child_indexes,
                                           int64_t parent_idx,
                                           size_t parent_tree_depth) const {
    if (parent_tree_depth == 0) {
      for (size_t i = 0; i < fan_out_; ++i) {
        child_indexes[i] = i + 1;
      }
    } else {
      int64_t cur_depth_start_offset = parent_idx * fan_out_ + 1;
      for (size_t i = 0; i < fan_out_; ++i) {
        child_indexes[i] = cur_depth_start_offset + i;
      }
    }
    return child_indexes;
  }

  // a utility function that computes a length and the start node index of a segment tree
  // to contain 'num_elem' with a given 'fan_out'
  std::pair<size_t, IndexPair> findMaxTreeHeight(int64_t num_elem, int fan_out) {
    if (num_elem <= 0) {
      return std::make_pair(0, std::make_pair(0, 0));
    }
    int64_t cur_level_start_offset = 0;
    size_t depth = 0;
    IndexPair index_pair = std::make_pair(0, 0);
    while (true) {
      size_t maximum_node_at_next_level = pow(fan_out, depth);
      if (num_elem < maximum_node_at_next_level) {
        index_pair = std::make_pair(cur_level_start_offset,
                                    cur_level_start_offset + maximum_node_at_next_level);
        return std::make_pair(depth, index_pair);
      }
      depth++;
      cur_level_start_offset += maximum_node_at_next_level;
    }
  }

  // agg input column buffer and its type info
  const INPUT_TYPE* const input_col_buf_;
  // to deal with conditional aggregate function
  const int8_t* const cond_col_buf_;
  SQLTypeInfo const input_col_ti_;
  // the following two idx buffers are for accessing the sorted input column
  // based on our current window function context (i.e., indirect column access)
  // i-th row -> get indirect idx 'i_idx' from `ordered_input_col_idx_buf_`
  // and use `i_idx` to get the true row index `t_idx` from `original_input_col_idx_buf_`
  // and use `t_idx` to get i-th row of the sorted column
  // otherwise, we can access the column when it keeps sorted values
  // original index (i.e., row_id) to access input_col_buf
  const int32_t* const original_input_col_idx_buf_;
  // ordered index to access sorted input_col_buf
  const int64_t* const ordered_input_col_idx_buf_;
  // # input elements
  int64_t const num_elems_;
  // tree fanout
  size_t const fan_out_;
  // a type of aggregate function
  SqlWindowFunctionKind const agg_type_;
  bool const is_conditional_agg_;
  INPUT_TYPE const input_type_null_val_;
  AGG_TYPE const null_val_;
  // invalid_val is required to fill an empty node for a correct aggregation
  INPUT_TYPE const invalid_val_;
  // # nodes in the leaf level
  size_t leaf_size_;
  // level of the leaf
  size_t leaf_depth_;
  // start ~ end indexes of the leaf level
  IndexPair leaf_range_;
  // total number of nodes in the tree
  size_t tree_size_;
  // depending on aggregate function, we use different aggregation logic
  // 1) a segment tree for computing derived aggregates, i.e., avg and stddev
  SumAndCountPair<AGG_TYPE>* derived_aggregated_;
  // 2) rest of aggregate functions can use a vector of elements
  AGG_TYPE* aggregated_values_;
};
