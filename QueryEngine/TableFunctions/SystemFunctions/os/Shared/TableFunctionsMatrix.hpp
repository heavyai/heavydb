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

#include <iostream>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsCommon.h"

namespace TableFunctions_Namespace {

struct ColumnMetadata {
  int64_t min;
  int64_t max;
  bool has_nulls;
  int64_t null_sentinel;
  ColumnMetadata()
      : min(std::numeric_limits<int64_t>::max())
      , max(std::numeric_limits<int64_t>::min())
      , has_nulls(false) {}
  inline uint64_t map_to_compressed_range(const int64_t val) const {
    if (has_nulls && val == null_sentinel) {
      return max + 1 - min;
    }
    return val - min;
  }

  inline int64_t map_to_uncompressed_range(const int64_t val) const {
    if (has_nulls && val == (max - min + 1)) {
      return null_sentinel;
    }
    return val + min;
  }
};
struct KeyMetadata {
  ColumnMetadata column_metadata;
  const int64_t range;
  const int64_t composite_prefix_range;
  KeyMetadata(const ColumnMetadata& column_metadata_in,
              const int64_t composite_prefix_range_in)
      : column_metadata(column_metadata_in)
      , range(column_metadata.max - column_metadata.min + 1 +
              (column_metadata.has_nulls ? 1 : 0))
      , composite_prefix_range(composite_prefix_range_in) {}
};

template <typename T>
inline ColumnMetadata get_integer_column_metadata(const Column<T>& col) {
  ColumnMetadata column_metadata;
  auto [col_min, col_max, has_nulls] = get_column_metadata(col);
  column_metadata.min = col_min;
  column_metadata.max = col_max;
  column_metadata.has_nulls = has_nulls;
  column_metadata.null_sentinel = inline_null_value<T>();
  return column_metadata;
}

struct CompositeKeyMetadata {
  std::vector<KeyMetadata> keys_metadata;
  int64_t num_keys;
};

template <typename K>
inline uint64_t map_to_compressed_range(
    const ColumnList<K>& keys,
    const CompositeKeyMetadata& composite_key_metadata,
    const int64_t idx) {
  uint64_t val = 0;
  const uint64_t num_keys = keys.numCols();
  for (uint64_t k = 0; k < num_keys; ++k) {
    val +=
        composite_key_metadata.keys_metadata[k].column_metadata.map_to_compressed_range(
            keys[k][idx]) *
        composite_key_metadata.keys_metadata[k].composite_prefix_range;
  }
  return val;
}

template <typename K>
inline uint64_t map_to_compressed_range_separate_nulls(
    const ColumnList<K>& keys,
    const CompositeKeyMetadata& composite_key_metadata,
    const int64_t idx,
    const uint64_t separated_null_val) {
  uint64_t val = 0;
  const uint64_t num_keys = keys.numCols();
  for (uint64_t k = 0; k < num_keys; ++k) {
    const K key = keys[k][idx];
    if (key == composite_key_metadata.keys_metadata[k].column_metadata.null_sentinel) {
      return separated_null_val;
    }
    val +=
        composite_key_metadata.keys_metadata[k].column_metadata.map_to_compressed_range(
            keys[k][idx]) *
        composite_key_metadata.keys_metadata[k].composite_prefix_range;
  }
  return val;
}

template <typename K>
inline uint64_t map_to_compressed_range(
    const Column<K>& keys,
    const CompositeKeyMetadata& composite_key_metadata,
    const int64_t idx) {
  return composite_key_metadata.keys_metadata[0].column_metadata.map_to_compressed_range(
      keys[idx]);
}

template <typename K>
inline uint64_t map_to_compressed_range(
    const K key,
    const CompositeKeyMetadata& composite_key_metadata) {
  return composite_key_metadata.keys_metadata[0].column_metadata.map_to_compressed_range(
      key);
}

template <typename K>
inline CompositeKeyMetadata getCompositeKeyMetadata(const ColumnList<K>& keys) {
  CompositeKeyMetadata composite_key_metadata;
  const size_t num_keys = keys.numCols();
  composite_key_metadata.num_keys = 1;
  for (size_t k = 0; k != num_keys; ++k) {
    composite_key_metadata.keys_metadata.emplace_back(
        get_integer_column_metadata(keys[k]), composite_key_metadata.num_keys);
    composite_key_metadata.num_keys *= composite_key_metadata.keys_metadata[k].range;
  }
  return composite_key_metadata;
}

template <typename K>
inline CompositeKeyMetadata getCompositeKeyMetadata(const Column<K>& key) {
  CompositeKeyMetadata composite_key_metadata;
  composite_key_metadata.keys_metadata.emplace_back(get_integer_column_metadata(key), 1);
  composite_key_metadata.num_keys = composite_key_metadata.keys_metadata[0].range;
  return composite_key_metadata;
}

inline CompositeKeyMetadata unionCompositeKeyMetadata(
    const CompositeKeyMetadata& composite_key_metadata1,
    const CompositeKeyMetadata& composite_key_metadata2) {
  // Note: Unequal key sizes should be caught and handled/thrown at a higher level
  CHECK_EQ(composite_key_metadata1.keys_metadata.size(),
           composite_key_metadata2.keys_metadata.size());
  const size_t num_keys = composite_key_metadata1.keys_metadata.size();
  CompositeKeyMetadata unioned_composite_key_metadata;
  unioned_composite_key_metadata.num_keys = 1;
  for (size_t k = 0; k != num_keys; ++k) {
    const KeyMetadata& key_metadata1 = composite_key_metadata1.keys_metadata[k];
    const KeyMetadata& key_metadata2 = composite_key_metadata2.keys_metadata[k];
    ColumnMetadata unioned_column_metadata;
    unioned_column_metadata.min =
        std::min(key_metadata1.column_metadata.min, key_metadata2.column_metadata.min);
    unioned_column_metadata.max =
        std::max(key_metadata1.column_metadata.max, key_metadata2.column_metadata.max);
    unioned_column_metadata.has_nulls = key_metadata1.column_metadata.has_nulls ||
                                        key_metadata2.column_metadata.has_nulls;
    // We're not going to handle the null sentinel here as that still needs to be handled
    // per column (as they may be of different types)
    unioned_composite_key_metadata.keys_metadata.emplace_back(
        unioned_column_metadata, unioned_composite_key_metadata.num_keys);
    unioned_composite_key_metadata.num_keys *=
        unioned_composite_key_metadata.keys_metadata[k].range;
  }
  return unioned_composite_key_metadata;
}

void copyCompositeKeyMetadataNulls(CompositeKeyMetadata& dest_metadata,
                                   const CompositeKeyMetadata& src_metadata) {
  CHECK_EQ(dest_metadata.keys_metadata.size(), src_metadata.keys_metadata.size());
  for (size_t k = 0; k != dest_metadata.keys_metadata.size(); ++k) {
    dest_metadata.keys_metadata[k].column_metadata.null_sentinel =
        src_metadata.keys_metadata[k].column_metadata.null_sentinel;
  }
}

template <typename U, typename S>
struct SparseVector {
  std::vector<U> row_indices;
  std::vector<S> data;
  SparseVector(const size_t size) : row_indices(size), data(size) {}
};

template <typename U, typename S>
struct SparseMatrixCsc {
  std::vector<uint64_t> col_offsets;
  std::vector<U> col_values;
  std::vector<U> row_indices;
  std::vector<S> data;
  SparseMatrixCsc(const size_t size) : row_indices(size), data(size) {}
};

template <typename M>
struct DenseMatrix {
  const uint64_t num_rows;
  const uint64_t num_cols;
  std::vector<M> data;

  DenseMatrix(const uint64_t n_rows, const uint64_t n_cols)
      : num_rows(n_rows), num_cols(n_cols), data(num_rows * num_cols, 0) {}

  inline M get(const uint64_t r, const uint64_t c) const {
    return data[c * num_cols + r];
  }
  inline void set(const uint64_t r, const uint64_t c, const M val) {
    data[c * num_cols + r] = val;
  }

  template <typename W>
  std::string to_string_with_precision(const W field, const size_t field_width) const {
    std::ostringstream out;
    out.precision(field_width);
    out << std::fixed << field;
    return out.str();
  }

  template <typename W>
  void print_field(const W field, const size_t field_width) const {
    // const std::string string_field {std::to_string(static_cast<uint64_t>(field))};
    const std::string string_field =
        to_string_with_precision(field, field_width >= 9 ? field_width - 4 : 5);
    std::cout << std::string(
                     std::max(field_width - string_field.size(), static_cast<size_t>(0)),
                     ' ')
              << string_field;
  }

  void print(const uint64_t min_r,
             const uint64_t max_r,
             const uint64_t min_c,
             const uint64_t max_c) const {
    const uint64_t safe_max_r = std::max(std::min(max_r, num_rows - 1), min_r);
    const uint64_t safe_max_c = std::max(std::min(max_c, num_cols - 1), min_r);
    const size_t field_width{10};

    std::cout << std::endl << std::endl << std::string(field_width, ' ');
    for (uint64_t c = min_c; c <= safe_max_c; c++) {
      print_field(c, field_width);
    }
    for (uint64_t r = min_r; r <= safe_max_r; r++) {
      std::cout << std::endl;
      print_field(r, field_width);
      for (uint64_t c = min_c; c <= safe_max_c; c++) {
        const M field = get(r, c);
        print_field(field, field_width);
      }
    }
    std::cout << std::endl << std::endl;
  }
};

template <typename F, typename M, typename U, typename S>
SparseVector<U, S> pivot_table_to_sparse_vector(
    const ColumnList<F>& key_col,
    const Column<M>& metric_col,
    const CompositeKeyMetadata& key_metadata) {
  const uint64_t num_rows = key_col.size();
  std::vector<U> compressed_keys(num_rows);
  std::vector<uint64_t> sort_permutation(num_rows);

  const size_t loop_grain_size{4096};
  const U separated_null_val = std::numeric_limits<U>::max();
  tbb::parallel_for(tbb::blocked_range<uint64_t>(0, num_rows, loop_grain_size),
                    [&](const tbb::blocked_range<uint64_t>& i) {
                      for (uint64_t r = i.begin(); r != i.end(); ++r) {
                        compressed_keys[r] = map_to_compressed_range_separate_nulls(
                            key_col, key_metadata, r, separated_null_val);
                        sort_permutation[r] = r;
                      }
                    });

  tbb::parallel_sort(sort_permutation.begin(),
                     sort_permutation.end(),
                     [&](const uint64_t& a, const uint64_t& b) {
                       return (compressed_keys[a] < compressed_keys[b]);
                     });

  // Nulls will always be at end of permuted keys since they were mapped to max uint64_t
  uint64_t num_nulls = 0;
  for (; num_nulls < num_rows; num_nulls++) {
    if (compressed_keys[sort_permutation[num_rows - num_nulls - 1]] !=
        separated_null_val) {
      break;
    }
  }

  const uint64_t num_non_null_rows = num_rows - num_nulls;

  SparseVector<U, S> sparse_vector(num_non_null_rows);
  tbb::parallel_for(tbb::blocked_range<uint64_t>(0, num_non_null_rows, loop_grain_size),
                    [&](const tbb::blocked_range<uint64_t>& i) {
                      const uint64_t i_end{i.end()};
                      for (uint64_t r = i.begin(); r != i_end; ++r) {
                        const uint64_t permute_idx = sort_permutation[r];
                        sparse_vector.row_indices[r] = compressed_keys[permute_idx];
                        sparse_vector.data[r] = metric_col[permute_idx];
                      }
                    });

  return sparse_vector;
}

// template <typename K, typename F, typename M, typename U, typename S>
// SparseMatrixCsc<U, S> pivot_table_to_sparse_csc_matrix(
//    const Column<K>& primary_key_col,
//    const ColumnList<F>& secondary_key_cols,
//    const Column<M>& metric_col,
//    const CompositeKeyMetadata& primary_key_metadata,
//    const CompositeKeyMetadata& secondary_key_metadata) {
//  ColumnList<K> primary_key_cols;
//  std::vector<int8_t*> ptrs (1);
//  ptrs[0] = reinterpret_cast<int8_t*>(primary_key_col.ptr_);
//  primary_key_cols.ptrs_ = ptrs.data();
//
//  primary_key_cols.num_cols_ = 1;
//  primary_key_cols.size_ = primary_key_col.size_;
//  return pivot_table_to_sparse_csc_matrix(primary_key_cols, secondary_key_cols,
//  metric_col, primary_key_metadata, secondary_key_metadata);
//}

template <typename K, typename F, typename M, typename U, typename S>
SparseMatrixCsc<U, S> pivot_table_to_sparse_csc_matrix(
    const Column<K>& primary_key_col,
    const ColumnList<F>& secondary_key_cols,
    const Column<M>& metric_col,
    const CompositeKeyMetadata& primary_key_metadata,
    const CompositeKeyMetadata& secondary_key_metadata) {
  const uint64_t num_rows = primary_key_col.size();
  std::vector<U> compressed_primary_keys(num_rows);
  std::vector<U> compressed_secondary_keys(num_rows);
  std::vector<uint64_t> sort_permutation(num_rows);

  const size_t loop_grain_size{4096};
  tbb::parallel_for(tbb::blocked_range<uint64_t>(0, num_rows, loop_grain_size),
                    [&](const tbb::blocked_range<uint64_t>& i) {
                      for (uint64_t r = i.begin(); r != i.end(); ++r) {
                        compressed_primary_keys[r] = map_to_compressed_range(
                            primary_key_col, primary_key_metadata, r);
                        compressed_secondary_keys[r] = map_to_compressed_range(
                            secondary_key_cols, secondary_key_metadata, r);
                        sort_permutation[r] = r;
                      }
                    });

  tbb::parallel_sort(
      sort_permutation.begin(),
      sort_permutation.end(),
      [&](const uint64_t& a, const uint64_t& b) {
        if (compressed_primary_keys[a] < compressed_primary_keys[b]) {
          return true;
        } else if (compressed_primary_keys[a] > compressed_primary_keys[b]) {
          return false;
        }
        return (compressed_secondary_keys[a] < compressed_secondary_keys[b]);
      });

  SparseMatrixCsc<U, S> sparse_matrix_csc(num_rows);
  tbb::parallel_for(tbb::blocked_range<uint64_t>(0, num_rows, loop_grain_size),
                    [&](const tbb::blocked_range<uint64_t>& i) {
                      const uint64_t i_end{i.end()};
                      for (uint64_t r = i.begin(); r != i_end; ++r) {
                        const uint64_t permute_idx = sort_permutation[r];
                        sparse_matrix_csc.row_indices[r] =
                            compressed_secondary_keys[permute_idx];
                        sparse_matrix_csc.data[r] = metric_col[permute_idx];
                      }
                    });

  U last_primary_key_compressed = std::numeric_limits<U>::max();
  for (uint64_t r = 0; r < num_rows; ++r) {
    const U primary_key_compressed = compressed_primary_keys[sort_permutation[r]];
    if (primary_key_compressed != last_primary_key_compressed) {
      sparse_matrix_csc.col_offsets.emplace_back(r);
      sparse_matrix_csc.col_values.emplace_back(primary_key_compressed);
      last_primary_key_compressed = primary_key_compressed;
    }
  }
  sparse_matrix_csc.col_offsets.emplace_back(num_rows);  // End sentinel

  return sparse_matrix_csc;
}

template <typename U, typename S>
std::vector<double> idf_normalize(SparseMatrixCsc<U, S>& sparse_matrix,
                                  const U num_secondary_keys) {
  const U num_primary_keys = sparse_matrix.col_values.size();
  const uint64_t num_rows = sparse_matrix.data.size();
  std::vector<double> secondary_key_idf(num_secondary_keys, 0.);

  for (uint64_t r = 0; r < num_rows; ++r) {
    secondary_key_idf[sparse_matrix.row_indices[r]] +=
        (sparse_matrix.data[r] > 0.001 ? 1 : 0);
  }
  const size_t loop_grain_size{1000};

  tbb::parallel_for(tbb::blocked_range<U>(0, num_secondary_keys, loop_grain_size),
                    [&](const tbb::blocked_range<U>& i) {
                      for (U k = i.begin(); k != i.end(); ++k) {
                        secondary_key_idf[k] =
                            log((num_primary_keys + 1.0) / secondary_key_idf[k]) + 1.;
                      }
                    });

  tbb::parallel_for(tbb::blocked_range<uint64_t>(0, num_rows, loop_grain_size),
                    [&](const tbb::blocked_range<uint64_t>& i) {
                      for (uint64_t r = i.begin(); r != i.end(); ++r) {
                        sparse_matrix.data[r] *=
                            secondary_key_idf[sparse_matrix.row_indices[r]];
                      }
                    });
  return secondary_key_idf;
}

template <typename U, typename S>
std::vector<S> get_matrix_col_mags(const SparseMatrixCsc<U, S>& sparse_matrix) {
  const U num_cols = sparse_matrix.col_values.size();
  std::vector<S> column_mags(num_cols);
  tbb::parallel_for(
      tbb::blocked_range<U>(0, num_cols), [&](const tbb::blocked_range<U>& c) {
        const U c_end = c.end();
        for (U col_idx = c.begin(); col_idx != c_end; ++col_idx) {
          const U end_col_offset = sparse_matrix.col_offsets[col_idx + 1];
          S column_mag_sq = 0;
          for (U col_pos = sparse_matrix.col_offsets[col_idx]; col_pos < end_col_offset;
               ++col_pos) {
            column_mag_sq += sparse_matrix.data[col_pos] * sparse_matrix.data[col_pos];
          }
          column_mags[col_idx] = sqrt(column_mag_sq);
        }
      });
  return column_mags;
}

template <typename U, typename S>
S get_vector_mag(const SparseVector<U, S>& sparse_vector) {
  const U num_vals = sparse_vector.row_indices.size();
  S vec_mag_sq = 0;
  for (U row_idx = 0; row_idx != num_vals; ++row_idx) {
    vec_mag_sq += sparse_vector.data[row_idx] * sparse_vector.data[row_idx];
  }
  return sqrt(vec_mag_sq);
}

template <typename U, typename S>
std::vector<S> multiply_matrix_by_vector(const SparseMatrixCsc<U, S>& sparse_matrix,
                                         const SparseVector<U, S>& sparse_vector,
                                         const bool unit_normalize) {
  const U num_cols = sparse_matrix.col_values.size();
  std::vector<S> dot_product_vec(num_cols);
  const uint64_t vec_length = sparse_vector.row_indices.size();
  tbb::parallel_for(
      tbb::blocked_range<U>(0, num_cols), [&](const tbb::blocked_range<U>& c) {
        const uint64_t c_end = c.end();
        for (U col_idx = c.begin(); col_idx != c_end; ++col_idx) {
          const U matrix_start_col_offset = sparse_matrix.col_offsets[col_idx];
          const U matrix_end_col_offset = sparse_matrix.col_offsets[col_idx + 1];
          S dot_product = 0;
          U m_pos = matrix_start_col_offset;
          U v_pos = 0;
          while (m_pos < matrix_end_col_offset && v_pos < vec_length) {
            const U m_row_index = sparse_matrix.row_indices[m_pos];
            const U v_row_index = sparse_vector.row_indices[v_pos];
            if (m_row_index < v_row_index) {
              ++m_pos;
            } else if (v_row_index < m_row_index) {
              ++v_pos;
            } else {
              dot_product += sparse_matrix.data[m_pos++] * sparse_vector.data[v_pos++];
            }
          }
          dot_product_vec[col_idx] = dot_product;
        }
      });
  if (unit_normalize) {
    const std::vector<S> matrix_mags = get_matrix_col_mags(sparse_matrix);
    const S vector_mag = get_vector_mag(sparse_vector);
    for (U c = 0; c != num_cols; ++c) {
      const S co_mag = matrix_mags[c] * vector_mag;
      if (co_mag > 0) {
        dot_product_vec[c] /= co_mag;
      }
    }
  }
  return dot_product_vec;
}

template <typename U, typename S>
DenseMatrix<S> multiply_matrix_by_transpose(const SparseMatrixCsc<U, S>& sparse_matrix,
                                            const bool unit_normalize) {
  const U num_cols = sparse_matrix.col_values.size();
  const U num_non_zero_entries = sparse_matrix.data.size();
  const U avg_non_zero_entries_per_col =
      std::max(num_non_zero_entries / num_cols, static_cast<U>(1));
  const U cache_size = 1 << 21;
  const double bytes_per_entry =
      sizeof(U) + sizeof(S) +
      (sizeof(U) + sizeof(uint64_t)) *
          (sparse_matrix.col_offsets.size() / sparse_matrix.row_indices.size());
  const U cols_per_partition =
      cache_size / (avg_non_zero_entries_per_col * bytes_per_entry) + 1;
  const U num_partitions = (num_cols + cols_per_partition - 1) / cols_per_partition;
  const U num_mn_partitions = num_partitions * num_partitions;

  DenseMatrix<S> similarity_matrix(num_cols, num_cols);
  tbb::parallel_for(
      tbb::blocked_range<U>(0, num_mn_partitions), [&](const tbb::blocked_range<U>& p) {
        for (U mn_part_idx = p.begin(); mn_part_idx != p.end(); ++mn_part_idx) {
          const U m_p = mn_part_idx / num_partitions;
          const U n_p = mn_part_idx % num_partitions;
          if (m_p > n_p) {
            continue;
          }
          const U n_start = n_p * cols_per_partition;
          const U n_end = std::min((n_p + 1) * cols_per_partition, num_cols);
          const U m_start = m_p * cols_per_partition;
          const U m_block_end = std::min((m_p + 1) * cols_per_partition, num_cols);
          for (U n = n_start; n < n_end; ++n) {
            const U m_end = std::min(m_block_end, n + 1);
            for (U m = m_start; m < m_end; ++m) {
              S dot_product = 0;
              const U n_pos_end = sparse_matrix.col_offsets[n + 1];
              const U m_pos_end = sparse_matrix.col_offsets[m + 1];
              U m_pos = sparse_matrix.col_offsets[m];
              U n_pos = sparse_matrix.col_offsets[n];
              while (m_pos < m_pos_end && n_pos < n_pos_end) {
                const U m_row_index = sparse_matrix.row_indices[m_pos];
                const U n_row_index = sparse_matrix.row_indices[n_pos];
                if (m_row_index < n_row_index) {
                  ++m_pos;
                } else if (n_row_index < m_row_index) {
                  ++n_pos;
                } else {
                  dot_product +=
                      sparse_matrix.data[m_pos++] * sparse_matrix.data[n_pos++];
                }
              }
              similarity_matrix.set(m, n, dot_product);
            }
          }
        }
      });

  if (unit_normalize) {
    std::vector<S> inv_norms(similarity_matrix.num_cols);
    for (U c = 0; c != similarity_matrix.num_cols; ++c) {
      const S col_length = similarity_matrix.get(c, c);
      const S col_length_sqrt = sqrt(col_length);
      inv_norms[c] = col_length_sqrt > 0 ? 1.0 / col_length_sqrt : 0;
    }

    for (U c = 0; c != similarity_matrix.num_cols; ++c) {
      const S inv_norm_c = inv_norms[c];
      const U max_row = c + 1;
      for (U r = 0; r != max_row; ++r) {
        similarity_matrix.set(
            r, c, similarity_matrix.get(r, c) * inv_norms[r] * inv_norm_c);
      }
    }
  }

  return similarity_matrix;
}

}  // namespace TableFunctions_Namespace

#endif  // #ifdef HAVE_TBB
#endif  // #ifndef __CUDACC__
