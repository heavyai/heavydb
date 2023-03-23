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

#ifndef __CUDACC__

#include "OneHotEncoder.h"
#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsCommon.hpp"
#include "Shared/ThreadInfo.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

namespace TableFunctions_Namespace {

namespace OneHotEncoder_Namespace {

/**
 * @brief This function calculates the top k most frequent keys (categories) in the
 * provided column based on a given minimum percentage of the column total per key. It
 * returns the top k keys along with a boolean value indicating whether there are other
 * keys beyond the top k keys.
 *
 * @param text_col - dictionary-encoded text column to extract top-k keys from
 * @param top_k - integer representing the top-k most common keys to return
 * @param min_perc_col_total_per_key - Enforces that any top-k key must represent at least
 * this minimum percentage of the total number of values in the column
 * @return std::pair<std::vector<int32_t>, bool> - A vector of the top k keys and a
 * boolean indicating whether there are other keys beyond the top k keys
 */
NEVER_INLINE HOST std::pair<std::vector<int32_t>, bool> get_top_k_keys(
    const Column<TextEncodingDict>& text_col,
    const int32_t top_k,
    const double min_perc_col_total_per_key) {
  auto timer = DEBUG_TIMER(__func__);
  const auto col_min_max = get_column_min_max(text_col);
  const int32_t col_size = text_col.size();
  const int32_t col_range = col_min_max.second - col_min_max.first + 1;

  // Todo(todd): Calculate the counts in parallel when `col_range` is
  // relatively small, and better still, vary the number of threads
  // based on that range.
  // There is a tradeoff between parallelism and the memory pressure and performance
  // overhead of allocating separate count buffers per thread. When `col_range`
  // is relatively small, it will be advantageous to maximize parallelism,
  // but in cases when `col_range` is large, it will likely be better to use
  // a single or only a few threads

  // Note that when trying to parallelize this method with
  // the use of TBB::concurrent_vector, the runtime went up 10X,
  // from roughly 100ms for a 120M record column to 1000ms
  std::vector<int32_t> key_counts(col_range, 0);
  for (int32_t idx = 0; idx < col_size; ++idx) {
    if (!text_col.isNull(idx)) {
      const int32_t key_idx = text_col[idx] - col_min_max.first;
      key_counts[key_idx]++;
    }
  }

  std::vector<int32_t> permutation_idxs(col_range);

  tbb::parallel_for(tbb::blocked_range<int32_t>(0, col_range),
                    [&](const tbb::blocked_range<int32_t>& r) {
                      const int32_t r_end = r.end();
                      for (int32_t p = r.begin(); p < r_end; ++p) {
                        permutation_idxs[p] = p;
                      }
                    });

  // Sort key_counts in descending order
  tbb::parallel_sort(
      permutation_idxs.begin(),
      permutation_idxs.begin() + col_range,
      [&](const int32_t& a, const int32_t& b) { return key_counts[a] > key_counts[b]; });
  int32_t actual_top_k = std::min(col_range, top_k);
  std::vector<int32_t> top_k_keys;
  top_k_keys.reserve(actual_top_k);
  const float col_size_fp = static_cast<float>(col_size);
  int32_t k = 0;
  // Todo(todd): Optimize the below code to do a binary search to find
  // the first key (if it exists) where key_count_perc < min_perc_col_total_per_key,
  // and then do a parallel for pass to write to the `top_k_keys` vec
  for (; k < actual_top_k; ++k) {
    const int32_t key_counts_idx = permutation_idxs[k];
    const int32_t key_count = key_counts[key_counts_idx];
    const float key_count_perc = key_count / col_size_fp;
    if (key_count_perc < min_perc_col_total_per_key) {
      break;
    }
    top_k_keys.emplace_back(key_counts_idx + col_min_max.first);
  }
  const bool has_other_keys = k < col_range && key_counts[permutation_idxs[k]] > 0;
  return std::make_pair(top_k_keys, has_other_keys);
}

/**
 * @brief Allocates memory for the one-hot encoded columns and initializes them to zero.
 * It takes the number of one-hot columns and the column size as input and returns a
 * vector of one-hot encoded columns.
 *
 * @tparam F
 * @param num_one_hot_cols - number of one-hot encoded columns to allocate
 * @param col_size - Size of each column in number of values/rows
 * @return std::vector<std::vector<F>> - a vector of vectors, with each inner vector
 * representing a one-hot encoding for a value in a column
 */
template <typename F>
NEVER_INLINE HOST std::vector<std::vector<F>> allocate_one_hot_cols(
    const int64_t num_one_hot_cols,
    const int64_t col_size) {
  std::vector<std::vector<F>> one_hot_allocated_buffers(num_one_hot_cols);
  const int64_t target_num_col_allocations_per_thread =
      std::ceil(100000.0 / (col_size + 1));
  const ThreadInfo thread_info(std::thread::hardware_concurrency(),
                               num_one_hot_cols,
                               target_num_col_allocations_per_thread);
  CHECK_GE(thread_info.num_threads, 1L);
  CHECK_GE(thread_info.num_elems_per_thread, 1L);
  std::vector<std::future<void>> allocator_threads;
  for (int64_t col_idx = 0; col_idx < num_one_hot_cols;
       col_idx += thread_info.num_elems_per_thread) {
    allocator_threads.emplace_back(std::async(
        std::launch::async,
        [&one_hot_allocated_buffers, num_one_hot_cols, col_size, &thread_info](
            const int64_t start_col_idx) {
          const int64_t end_col_idx = std::min(
              start_col_idx + thread_info.num_elems_per_thread, num_one_hot_cols);
          for (int64_t alloc_col_idx = start_col_idx; alloc_col_idx < end_col_idx;
               ++alloc_col_idx) {
            one_hot_allocated_buffers[alloc_col_idx].resize(col_size, 0);
          }
        },
        col_idx));
  }
  return one_hot_allocated_buffers;
}

template NEVER_INLINE HOST std::vector<std::vector<float>> allocate_one_hot_cols(
    const int64_t num_one_hot_cols,
    const int64_t col_size);
template NEVER_INLINE HOST std::vector<std::vector<double>> allocate_one_hot_cols(
    const int64_t num_one_hot_cols,
    const int64_t col_size);

/**
 * @brief Finds the minimum and maximum keys in a given vector of keys and returns them as
 * a pair
 *
 * @param top_k_keys - The top-k keys for a column
 * @return std::pair<int32_t, int32_t> - A pair representing the minimum and maximum key
 */
std::pair<int32_t, int32_t> get_min_max_keys(const std::vector<int32_t>& top_k_keys) {
  int32_t min_key = std::numeric_limits<int32_t>::max();
  int32_t max_key = std::numeric_limits<int32_t>::lowest();
  for (const auto& key : top_k_keys) {
    if (key == StringDictionary::INVALID_STR_ID) {
      continue;
    }
    if (key < min_key) {
      min_key = key;
    }
    if (key > max_key) {
      max_key = key;
    }
  }
  return std::make_pair(min_key, max_key);
}

constexpr int16_t INVALID_COL_IDX{-1};

/**
 * @struct KeyToOneHotColBytemap
 * @brief  A struct that creates a bytemap to map each key to its corresponding one-hot
 * column index.
 */

struct KeyToOneHotColBytemap {
  KeyToOneHotColBytemap(const std::vector<int32_t>& top_k_keys,
                        const int32_t min_key,
                        const int32_t max_key,
                        const bool has_other_key)
      : min_key_(min_key)
      , max_key_(max_key)
      , has_other_key_(has_other_key)
      , other_key_(top_k_keys.size())
      , bytemap_(init_bytemap(top_k_keys, min_key, max_key, has_other_key)) {}

  static std::vector<int16_t> init_bytemap(const std::vector<int32_t>& top_k_keys,
                                           const int32_t min_key,
                                           const int32_t max_key,
                                           const bool has_other_key) {
    // The bytemap can be quite large if the dictionary-encoded key range is large, so for
    // efficiency we store the offsets as int16_t Since we use `top_k_keys.size()` as the
    // sentinel for the OTHER key, we check to see if the top_k_keys.size() is smaller
    // than the maximum allowable value for int16_t
    if (static_cast<int64_t>(top_k_keys.size()) >= std::numeric_limits<int16_t>::max()) {
      std::ostringstream error_oss;
      error_oss << "Error: More than " << std::numeric_limits<int16_t>::max() - 1
                << " top k categorical keys not allowed.";
      throw std::runtime_error(error_oss.str());
    }
    std::vector<int16_t> bytemap(max_key - min_key + 1,
                                 has_other_key ? top_k_keys.size() : INVALID_COL_IDX);
    int16_t offset = 0;
    for (const auto& key : top_k_keys) {
      bytemap[key - min_key] = offset++;
    }
    return bytemap;
  }

  inline int16_t get_col_idx_for_key(const int32_t key) const {
    if (key < min_key_ || key > max_key_) {
      return has_other_key_ ? other_key_ : INVALID_COL_IDX;
    }
    return bytemap_[key - min_key_];
  }

  const int32_t min_key_;
  const int32_t max_key_;
  const bool has_other_key_;
  const int32_t other_key_;
  const std::vector<int16_t> bytemap_;
};

template <typename F>
NEVER_INLINE HOST OneHotEncodedCol<F> one_hot_encode(
    const Column<TextEncodingDict>& text_col,
    const TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo&
        one_hot_encoding_info) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(one_hot_encoding_info.is_one_hot_encoded);
  OneHotEncodedCol<F> one_hot_encoded_col;
  bool include_others_key = false;
  std::vector<int> top_k_keys;
  if (one_hot_encoding_info.cat_features.empty()) {
    CHECK_GT(one_hot_encoding_info.top_k_attrs, 1);
    CHECK_GE(one_hot_encoding_info.min_attr_proportion, 0);

    const auto [top_k_keys_temp, has_other_keys] =
        get_top_k_keys(text_col,
                       one_hot_encoding_info.top_k_attrs,
                       one_hot_encoding_info.min_attr_proportion);
    top_k_keys = top_k_keys_temp;
    include_others_key = one_hot_encoding_info.include_others_attr && has_other_keys;
    // If top k keys comprises all keys (i.e. k == n) and no other key is requested,
    // then remove the least common key as otherwise we overdetermine the degrees
    // of freedom and can get strange regression coefficients/results
    // Note we do not remove a key if there is only one key
    if (!has_other_keys && !one_hot_encoding_info.include_others_attr &&
        top_k_keys.size() > 1) {
      top_k_keys.pop_back();
    }
    for (const auto top_k_key : top_k_keys) {
      one_hot_encoded_col.cat_features.emplace_back(
          text_col.string_dict_proxy_->getString(top_k_key));
    }
  } else {
    one_hot_encoded_col.cat_features = one_hot_encoding_info.cat_features;
    for (const auto& cat_feature : one_hot_encoded_col.cat_features) {
      top_k_keys.emplace_back(text_col.string_dict_proxy_->getIdOfString(cat_feature));
    }
  }

  const int64_t num_one_hot_cols = top_k_keys.size() + (include_others_key ? 1 : 0);
  const int64_t col_size = text_col.size();
  one_hot_encoded_col.encoded_buffers =
      allocate_one_hot_cols<F>(num_one_hot_cols, col_size);
  constexpr int64_t max_bytemap_size = 10000000L;

  const auto [min_key, max_key] = get_min_max_keys(top_k_keys);
  const int64_t key_range = max_key - min_key + 1;
  if (key_range > max_bytemap_size) {
    throw std::runtime_error(
        "One-hot vectors currently can only be generated on string columns with less "
        "that " +
        std::to_string(max_bytemap_size) + " unique entries.");
  }
  const KeyToOneHotColBytemap key_to_one_hot_bytemap(
      top_k_keys, min_key, max_key, include_others_key);

  tbb::parallel_for(tbb::blocked_range<int64_t>(0, col_size),
                    [&](const tbb::blocked_range<int64_t>& r) {
                      const int64_t r_end = r.end();
                      for (int64_t row_idx = r.begin(); row_idx < r_end; ++row_idx) {
                        const int32_t key = text_col[row_idx];
                        const auto col_idx =
                            key_to_one_hot_bytemap.get_col_idx_for_key(key);
                        if (col_idx != INVALID_COL_IDX) {
                          one_hot_encoded_col.encoded_buffers[col_idx][row_idx] = 1.;
                        }
                      }
                    });

  return one_hot_encoded_col;
}

template NEVER_INLINE HOST OneHotEncodedCol<float> one_hot_encode(
    const Column<TextEncodingDict>& text_col,
    const TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo&
        one_hot_encoding_info);

template NEVER_INLINE HOST OneHotEncodedCol<double> one_hot_encode(
    const Column<TextEncodingDict>& text_col,
    const TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo&
        one_hot_encoding_info);

template <typename F>
NEVER_INLINE HOST std::vector<OneHotEncodedCol<F>> one_hot_encode(
    const ColumnList<TextEncodingDict>& text_cols,
    const std::vector<
        TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo>&
        one_hot_encoding_infos) {
  const int64_t num_input_cols = text_cols.numCols();
  // std::vector<std::vector<std::vector<F>>> one_hot_buffers;
  std::vector<OneHotEncodedCol<F>> one_hot_encoded_cols;
  one_hot_encoded_cols.reserve(num_input_cols);
  for (int64_t input_col_idx = 0; input_col_idx < num_input_cols; ++input_col_idx) {
    Column<TextEncodingDict> dummy_text_col(
        reinterpret_cast<TextEncodingDict*>(text_cols.ptrs_[input_col_idx]),
        text_cols.num_rows_,
        text_cols.string_dict_proxies_[input_col_idx]);
    one_hot_encoded_cols.emplace_back(
        one_hot_encode<F>(dummy_text_col, one_hot_encoding_infos[input_col_idx]));
  }
  return one_hot_encoded_cols;
}

template NEVER_INLINE HOST std::vector<OneHotEncodedCol<float>> one_hot_encode(
    const ColumnList<TextEncodingDict>& text_cols,
    const std::vector<
        TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo>&
        one_hot_encoding_infos);

template NEVER_INLINE HOST std::vector<OneHotEncodedCol<double>> one_hot_encode(
    const ColumnList<TextEncodingDict>& text_cols,
    const std::vector<
        TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo>&
        one_hot_encoding_infos);

}  // namespace OneHotEncoder_Namespace

}  // namespace TableFunctions_Namespace

#endif  // #ifndef __CUDACC__
