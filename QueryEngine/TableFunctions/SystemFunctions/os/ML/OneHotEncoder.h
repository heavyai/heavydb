/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef __CUDACC__

#include "QueryEngine/heavydbTypes.h"
#include "Shared/funcannotations.h"

#include <vector>

namespace TableFunctions_Namespace {

namespace OneHotEncoder_Namespace {

struct OneHotEncodingInfo {
  bool is_one_hot_encoded;
  int32_t top_k_attrs;
  float min_attr_proportion;
  bool include_others_attr;
  std::vector<std::string> cat_features;

  OneHotEncodingInfo() : is_one_hot_encoded(false) {}

  OneHotEncodingInfo(const int32_t top_k_attrs,
                     const float min_attr_proportion,
                     const bool include_others_attr)
      : is_one_hot_encoded(true)
      , top_k_attrs(top_k_attrs)
      , min_attr_proportion(min_attr_proportion)
      , include_others_attr(include_others_attr) {}

  OneHotEncodingInfo(const std::vector<std::string>& cat_features)
      : is_one_hot_encoded(true), cat_features(cat_features) {}
};

template <typename F>
struct OneHotEncodedCol {
  std::vector<std::vector<F>> encoded_buffers;
  std::vector<std::string> cat_features;
};

/**
 * @brief Takes a column of text-encoded data and one-hot encoding information as input.
 * It performs the one-hot encoding process and returns an object containing the one-hot
 * encoded columns and their corresponding categorical features.
 *
 * @tparam F
 * @param text_col - input TextEncodingDict column
 * @param one_hot_encoding_info - struct of parameters specifying how to encode the column
 * @return OneHotEncodedCol<F> - A transformed column with multiple one-hot encoded
 * sub-columns, one for each of the top-k keys
 */

template <typename F>
NEVER_INLINE HOST OneHotEncodedCol<F> one_hot_encode(
    const Column<TextEncodingDict>& text_col,
    const TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo&
        one_hot_encoding_info);

/**
 * @brief One-hot encode multiple columns of text-encoded data in a column list,
 * given a vector of one-hot encoding information for each column.
 *
 * @tparam F
 * @param text_cols - Vector of input TextEncodingDict columns
 * @param one_hot_encoding_infos - structs of parameters for each column specifying how to
 * encode the column
 * @return std::vector<OneHotEncodedCol<F>> - A vector of transformed columns, each with
 * multiple one-hot encoded sub-columns, one for each of the top-k keys for that column
 */

template <typename F>
NEVER_INLINE HOST std::vector<OneHotEncodedCol<F>> one_hot_encode(
    const ColumnList<TextEncodingDict>& text_cols,
    const std::vector<
        TableFunctions_Namespace::OneHotEncoder_Namespace::OneHotEncodingInfo>&
        one_hot_encoding_infos);

}  // namespace OneHotEncoder_Namespace

}  // namespace TableFunctions_Namespace

#endif  // #ifndef __CUDACC__
