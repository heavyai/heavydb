/*
 * Copyright 2022 HEAVY.AI, Inc., Inc.
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

#ifndef __CUDACC__
#ifdef HAVE_TBB

#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsMatrix.hpp"

using namespace TableFunctions_Namespace;

template <typename U, typename K, typename S>
int64_t write_cos_sim(const std::vector<S>& similarity_vector,
                      const std::vector<U>& key_map,
                      const ColumnMetadata& primary_key_metadata,
                      Column<K>& output_key_col,
                      Column<S>& output_similarity) {
  const uint64_t num_rows = key_map.size();
  set_output_row_size(num_rows);

  for (U c = 0; c != key_map.size(); ++c) {
    output_key_col[c] = primary_key_metadata.map_to_uncompressed_range(key_map[c]);
    output_similarity[c] = similarity_vector[c];
  }
  return num_rows;
}

template <typename U, typename K, typename S>
int64_t write_cos_sim(const DenseMatrix<S>& similarity_matrix,
                      const std::vector<U>& key_map,
                      const ColumnMetadata& primary_key_metadata,
                      Column<K>& output_key_col_1,
                      Column<K>& output_key_col_2,
                      Column<S>& output_similarity) {
  const uint64_t num_rows =
      similarity_matrix.num_cols * (similarity_matrix.num_cols + 1) / 2;  // rows of a

  set_output_row_size(num_rows);

  uint64_t output_idx = 0;
  for (U c = 0; c != similarity_matrix.num_cols; ++c) {
    const U uncompressed_col_key =
        primary_key_metadata.map_to_uncompressed_range(key_map[c]);
    const U max_row = c + 1;
    for (U r = 0; r != max_row; ++r) {
      const U uncompressed_row_key =
          primary_key_metadata.map_to_uncompressed_range(key_map[r]);
      output_key_col_1[output_idx] = uncompressed_row_key;
      output_key_col_2[output_idx] = uncompressed_col_key;
      output_similarity[output_idx] = similarity_matrix.get(r, c);
      output_idx++;
    }
  }
  return num_rows;
}

template <typename K, typename F, typename M, typename U, typename S>
int64_t similarity_vector_impl(const Column<K>& matrix_primary_key,
                               const ColumnList<F>& matrix_pivot_features,
                               const Column<M>& metric,
                               const CompositeKeyMetadata& matrix_primary_key_metadata,
                               const CompositeKeyMetadata& matrix_pivot_features_metadata,
                               const ColumnList<F>& vector_pivot_features,
                               const Column<M>& vector_metric,
                               const CompositeKeyMetadata& vector_pivot_features_metadata,
                               Column<K>& output_primary_key,
                               Column<S>& output_similarity,
                               const bool normalize_by_idf) {
  CompositeKeyMetadata unioned_pivot_features_metadata = unionCompositeKeyMetadata(
      matrix_pivot_features_metadata, vector_pivot_features_metadata);

  // Need to override unioned metadata with the null sentinel for each the matrix and
  // vector pivot columns, as those are input dependant
  copyCompositeKeyMetadataNulls(unioned_pivot_features_metadata,
                                matrix_pivot_features_metadata);

  SparseMatrixCsc<U, S> sparse_matrix_csc =
      pivot_table_to_sparse_csc_matrix<K, F, M, U, S>(matrix_primary_key,
                                                      matrix_pivot_features,
                                                      metric,
                                                      matrix_primary_key_metadata,
                                                      unioned_pivot_features_metadata);
  copyCompositeKeyMetadataNulls(unioned_pivot_features_metadata,
                                vector_pivot_features_metadata);

  SparseVector<U, S> sparse_vector = pivot_table_to_sparse_vector<F, M, U, S>(
      vector_pivot_features, vector_metric, unioned_pivot_features_metadata);

  if (normalize_by_idf) {
    const std::vector<double> idf_vec = idf_normalize(
        sparse_matrix_csc, static_cast<U>(unioned_pivot_features_metadata.num_keys));
    const size_t sparse_vec_size = sparse_vector.data.size();
    for (size_t r = 0; r < sparse_vec_size; ++r) {
      sparse_vector.data[r] *= idf_vec[sparse_vector.row_indices[r]];
    }
  }

  const std::vector<S> similarity_vector =
      multiply_matrix_by_vector(sparse_matrix_csc, sparse_vector, true);

  const int64_t num_rows =
      write_cos_sim(similarity_vector,
                    sparse_matrix_csc.col_values,
                    matrix_primary_key_metadata.keys_metadata[0].column_metadata,
                    output_primary_key,
                    output_similarity);

  return num_rows;
}

template <typename K, typename F, typename M, typename U, typename S>
int64_t similarity_impl(const Column<K>& primary_key,
                        const ColumnList<F>& pivot_features,
                        const Column<M>& metric,
                        const CompositeKeyMetadata& primary_key_metadata,
                        const CompositeKeyMetadata& pivot_features_metadata,
                        Column<K>& output_primary_key_1,
                        Column<K>& output_primary_key_2,
                        Column<S>& output_similarity,
                        const bool normalize_by_idf) {
  SparseMatrixCsc<U, S> sparse_matrix_csc =
      pivot_table_to_sparse_csc_matrix<K, F, M, U, S>(primary_key,
                                                      pivot_features,
                                                      metric,
                                                      primary_key_metadata,
                                                      pivot_features_metadata);

  if (normalize_by_idf) {
    idf_normalize(sparse_matrix_csc, static_cast<U>(pivot_features_metadata.num_keys));
  }

  const DenseMatrix<S> similarity_matrix =
      multiply_matrix_by_transpose(sparse_matrix_csc, true);

  const int64_t num_rows =
      write_cos_sim(similarity_matrix,
                    sparse_matrix_csc.col_values,
                    primary_key_metadata.keys_metadata[0].column_metadata,
                    output_primary_key_1,
                    output_primary_key_2,
                    output_similarity);

  return num_rows;
}

// clang-format off
/*
  UDTF: tf_feature_similarity__cpu_template(Cursor<Column<K> primary_key,
  ColumnList<F> pivot_features, Column<M> metric> primary_features,
  Cursor<ColumnList<F> comparison_pivot_features, Column<M> comparison_metric> comparison_features,
  bool use_tf_idf | default=false) -> Column<K> class | input_id=args<0>, Column<float> similarity_score, K=[int64_t, TextEncodingDict], F=[int64_t], M=[int64_t, double]
*/
// clang-format on

template <typename K, typename F, typename M>
int64_t tf_feature_similarity__cpu_template(
    const Column<K>& primary_key,
    const ColumnList<F>& pivot_features,
    const Column<M>& metric,
    const ColumnList<F>& comparison_pivot_features,
    const Column<M>& comparison_metric,
    const bool use_tf_idf,
    Column<K>& output_primary_key,
    Column<float>& output_similarity) {
  if (pivot_features.numCols() != comparison_pivot_features.numCols()) {
    std::cout << "Error: Pivot features must have the same number of keys." << std::endl;
    set_output_row_size(0);
    return 0;
  }

  const auto primary_key_metadata = getCompositeKeyMetadata(primary_key);
  const auto pivot_features_metadata = getCompositeKeyMetadata(pivot_features);
  const auto comparison_pivot_features_metadata =
      getCompositeKeyMetadata(comparison_pivot_features);

  // todo: should extend by comparison_pivot_features
  const uint64_t max_dimension_range =
      std::max(primary_key_metadata.num_keys, pivot_features_metadata.num_keys);

  if (max_dimension_range > std::numeric_limits<uint32_t>::max()) {
    return similarity_vector_impl<K, F, M, uint64_t, float>(
        primary_key,
        pivot_features,
        metric,
        primary_key_metadata,
        pivot_features_metadata,
        comparison_pivot_features,
        comparison_metric,
        comparison_pivot_features_metadata,
        output_primary_key,
        output_similarity,
        use_tf_idf);

  } else {
    return similarity_vector_impl<K, F, M, uint32_t, float>(
        primary_key,
        pivot_features,
        metric,
        primary_key_metadata,
        pivot_features_metadata,
        comparison_pivot_features,
        comparison_metric,
        comparison_pivot_features_metadata,
        output_primary_key,
        output_similarity,
        use_tf_idf);
  }
}

// clang-format off
/*
  UDTF: tf_feature_similarity__cpu_template(Cursor<Column<K> primary_key, ColumnList<TextEncodingDict> pivot_features,
  Column<M> metric> primary_features, Cursor<ColumnList<TextEncodingDict> comparison_pivot_features, 
  Column<M> comparison_metric> comparison_features, bool use_tf_idf | default=false) ->
  Column<K> class | input_id=args<0>, Column<float> similarity_score, K=[int64_t, TextEncodingDict], M=[int64_t, double]
*/
// clang-format on

template <typename K, typename M>
int64_t tf_feature_similarity__cpu_template(
    const Column<K>& primary_key,
    const ColumnList<TextEncodingDict>& pivot_features,
    const Column<M>& metric,
    const ColumnList<TextEncodingDict>& comparison_pivot_features,
    const Column<M>& comparison_metric,
    const bool use_tf_idf,
    Column<K>& output_primary_key,
    Column<float>& output_similarity) {
  if (pivot_features.numCols() != comparison_pivot_features.numCols()) {
    std::cout << "Error: Pivot features must have the same number of keys." << std::endl;
    set_output_row_size(0);
    return 0;
  }

  const int64_t num_feature_cols = pivot_features.numCols();
  const int64_t num_comparison_rows = comparison_pivot_features.size();
  std::vector<int8_t*> new_col_ptrs;
  std::vector<StringDictionaryProxy*> new_sdp_ptrs;
  std::vector<std::vector<int32_t>> translated_col_ids(num_feature_cols);
  for (int64_t col_idx = 0; col_idx < num_feature_cols; ++col_idx) {
    const auto primary_sdp = pivot_features.string_dict_proxies_[col_idx];
    const auto& primary_sdp_string_dict_id = primary_sdp->getDictKey();
    const auto comparison_sdp = comparison_pivot_features.string_dict_proxies_[col_idx];
    const auto& comparison_string_dict_id = comparison_sdp->getDictKey();
    if (primary_sdp_string_dict_id != comparison_string_dict_id) {
      const auto translation_map =
          comparison_sdp->buildIntersectionTranslationMapToOtherProxy(primary_sdp, {});
      translated_col_ids[col_idx].resize(num_comparison_rows);
      int32_t* translated_ids = translated_col_ids[col_idx].data();
      const auto source_col_ptr =
          reinterpret_cast<const int32_t*>(comparison_pivot_features.ptrs_[col_idx]);
      for (int64_t row_idx = 0; row_idx < num_comparison_rows; ++row_idx) {
        const auto source_id = source_col_ptr[row_idx];
        const auto translated_id =
            source_id != inline_null_value<int32_t>() ? translation_map[source_id] : -1;
        translated_ids[row_idx] =
            translated_id == -1 ? inline_null_value<int32_t>() : translated_id;
      }
      new_col_ptrs.emplace_back(reinterpret_cast<int8_t*>(translated_ids));
      new_sdp_ptrs.emplace_back(primary_sdp);
    } else {
      new_col_ptrs.emplace_back(comparison_pivot_features.ptrs_[col_idx]);
      new_sdp_ptrs.emplace_back(comparison_sdp);
    }
  }
  ColumnList<TextEncodingDict> translated_comparison_pivot_features(
      new_col_ptrs.data(), num_feature_cols, num_comparison_rows, new_sdp_ptrs.data());

  const auto primary_key_metadata = getCompositeKeyMetadata(primary_key);
  const auto pivot_features_metadata = getCompositeKeyMetadata(pivot_features);
  const auto comparison_pivot_features_metadata =
      getCompositeKeyMetadata(translated_comparison_pivot_features);

  // todo: should extend by comparison_pivot_features
  const uint64_t max_dimension_range =
      std::max(primary_key_metadata.num_keys, pivot_features_metadata.num_keys);

  if (max_dimension_range > std::numeric_limits<uint32_t>::max()) {
    return similarity_vector_impl<K, TextEncodingDict, M, uint64_t, float>(
        primary_key,
        pivot_features,
        metric,
        primary_key_metadata,
        pivot_features_metadata,
        translated_comparison_pivot_features,
        comparison_metric,
        comparison_pivot_features_metadata,
        output_primary_key,
        output_similarity,
        use_tf_idf);

  } else {
    return similarity_vector_impl<K, TextEncodingDict, M, uint32_t, float>(
        primary_key,
        pivot_features,
        metric,
        primary_key_metadata,
        pivot_features_metadata,
        translated_comparison_pivot_features,
        comparison_metric,
        comparison_pivot_features_metadata,
        output_primary_key,
        output_similarity,
        use_tf_idf);
  }
}

// clang-format off
/*
  UDTF: tf_feature_self_similarity__cpu_template(Cursor<Column<K> primary_key, ColumnList<F> pivot_features,
  Column<M> metric> primary_features, bool use_tf_idf | default=false) -> Column<K> class1 | input_id=args<0>,
  Column<K> class2 | input_id=args<0>, Column<float> similarity_score,
  K=[int64_t, TextEncodingDict], F=[int64_t, TextEncodingDict], M=[int64_t, double]
*/
// clang-format on

template <typename K, typename F, typename M>
int64_t tf_feature_self_similarity__cpu_template(const Column<K>& primary_key,
                                                 const ColumnList<F>& pivot_features,
                                                 const Column<M>& metric,
                                                 const bool use_tf_idf,
                                                 Column<K>& output_primary_key_1,
                                                 Column<K>& output_primary_key_2,
                                                 Column<float>& output_similarity) {
  const auto primary_key_metadata = getCompositeKeyMetadata(primary_key);
  const auto pivot_features_metadata = getCompositeKeyMetadata(pivot_features);

  const uint64_t max_dimension_range =
      std::max(primary_key_metadata.num_keys, pivot_features_metadata.num_keys);
  if (max_dimension_range > std::numeric_limits<uint32_t>::max()) {
    return similarity_impl<K, F, M, uint64_t, float>(primary_key,
                                                     pivot_features,
                                                     metric,
                                                     primary_key_metadata,
                                                     pivot_features_metadata,
                                                     output_primary_key_1,
                                                     output_primary_key_2,
                                                     output_similarity,
                                                     use_tf_idf);

  } else {
    return similarity_impl<K, F, M, uint32_t, float>(primary_key,
                                                     pivot_features,
                                                     metric,
                                                     primary_key_metadata,
                                                     pivot_features_metadata,
                                                     output_primary_key_1,
                                                     output_primary_key_2,
                                                     output_similarity,
                                                     use_tf_idf);
  }
}

#endif  // #ifdef HAVE_TBB
#endif  // #ifndef __CUDACC__
