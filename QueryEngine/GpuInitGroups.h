/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    GpuInitGroups.h
 * @brief
 *
 */

#ifndef GPUINITGROUPS_H
#define GPUINITGROUPS_H
#include <cstddef>
#include <cstdint>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

struct DeviceColumnFragmentStats {
  int64_t int_min{0};
  int64_t int_max{0};
  double fp_min{0.0};
  double fp_max{0.0};
  bool has_nulls{false};
  bool has_values{false};
};

struct DeviceResultSetEntryComparison {
  uint64_t target_offset{0};
  int64_t null_bits{0};
  int64_t int_literal{0};
  double fp_literal{0.0};
  int32_t op{0};
  uint8_t target_width{0};
  bool is_fp{false};
  bool is_float{false};
  bool nullable{false};
};

struct DeviceBaselineHashReductionSlot {
  enum Op : uint8_t { Sum = 1, Min = 2, Max = 3 };

  uint32_t offset{0};
  int64_t init_val{0};
  uint8_t width{0};
  uint8_t op{0};
  bool skip_null_val{false};
  bool is_fp{false};
};

void init_group_by_buffer_on_device(int64_t* groups_buffer,
                                    const int64_t* init_vals,
                                    const size_t groups_buffer_entry_count,
                                    const uint32_t key_count,
                                    const uint32_t key_width,
                                    const uint32_t agg_col_count,
                                    const bool keyless,
                                    const int8_t warp_size,
                                    const size_t block_size_x,
                                    const size_t grid_size_x,
                                    CUstream cuda_stream);

void init_columnar_group_by_buffer_on_device(int64_t* groups_buffer,
                                             const int64_t* init_vals,
                                             const uint32_t groups_buffer_entry_count,
                                             const uint32_t key_count,
                                             const uint32_t agg_col_count,
                                             const int8_t* col_sizes,
                                             const bool need_padding,
                                             const bool keyless,
                                             const int8_t key_size,
                                             const size_t block_size_x,
                                             const size_t grid_size_x,
                                             CUstream cuda_stream);

size_t count_non_empty_baseline_hash_rows_on_device(const int8_t* groups_buffer,
                                                    const size_t entry_count,
                                                    const size_t row_size,
                                                    const size_t key_width,
                                                    uint64_t* row_count,
                                                    const int device_id,
                                                    CUstream cuda_stream = 0);

void compact_baseline_hash_rows_on_device(const int8_t* groups_buffer,
                                          int8_t* compacted_buffer,
                                          uint64_t* compacted_row_count,
                                          const size_t entry_count,
                                          const size_t row_size,
                                          const size_t key_width,
                                          const int device_id,
                                          CUstream cuda_stream = 0);

void count_baseline_hash_partition_rows_on_device(const int8_t* groups_buffer,
                                                  uint64_t* partition_counts,
                                                  const size_t entry_count,
                                                  const size_t row_size,
                                                  const size_t key_width,
                                                  const size_t key_count,
                                                  const size_t partition_count,
                                                  const int device_id,
                                                  CUstream cuda_stream = 0);

void partition_baseline_hash_rows_on_device(const int8_t* groups_buffer,
                                            int8_t* partitioned_buffer,
                                            uint64_t* partition_write_counts,
                                            const uint64_t* partition_offsets,
                                            const size_t entry_count,
                                            const size_t row_size,
                                            const size_t key_width,
                                            const size_t key_count,
                                            const size_t partition_count,
                                            const int device_id,
                                            CUstream cuda_stream = 0);

size_t count_non_empty_keyless_hash_rows_on_device(const int8_t* groups_buffer,
                                                   const size_t entry_count,
                                                   const size_t row_size,
                                                   const size_t key_slot_offset,
                                                   const size_t key_slot_width,
                                                   const int64_t key_init_val,
                                                   uint64_t* row_count,
                                                   const int device_id,
                                                   CUstream cuda_stream = 0);

void compact_keyless_hash_rows_on_device(const int8_t* groups_buffer,
                                         int8_t* compacted_buffer,
                                         uint64_t* compacted_row_count,
                                         uint64_t* compacted_entry_indices,
                                         const size_t entry_count,
                                         const size_t row_size,
                                         const size_t key_slot_offset,
                                         const size_t key_slot_width,
                                         const int64_t key_init_val,
                                         const int device_id,
                                         CUstream cuda_stream = 0);

size_t count_matching_baseline_hash_rows_on_device(
    const int8_t* groups_buffer,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count,
    const int64_t* preserved_keys,
    const size_t preserved_key_count,
    uint64_t* row_count,
    const int device_id,
    CUstream cuda_stream = 0);

void compact_matching_baseline_hash_rows_on_device(
    const int8_t* groups_buffer,
    int8_t* compacted_buffer,
    uint64_t* compacted_row_count,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count,
    const int64_t* preserved_keys,
    const size_t preserved_key_count,
    const int device_id,
    CUstream cuda_stream = 0);

void compact_matching_keyless_hash_rows_on_device(
    int8_t* groups_buffer,
    int8_t* compacted_buffer,
    uint64_t* compacted_row_count,
    uint64_t* compacted_entry_indices,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_slot_offset,
    const size_t key_slot_width,
    const int64_t key_init_val,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count,
    const int device_id,
    CUstream cuda_stream = 0);

void synthesize_perfect_hash_group_key_column_on_device(const uint64_t* entry_indices,
                                                        int8_t* columnar_buffer,
                                                        const size_t entry_count,
                                                        const size_t output_width,
                                                        const int64_t min_val,
                                                        const int64_t bucket,
                                                        const int64_t source_null_val,
                                                        const int64_t normalized_null_val,
                                                        const int device_id,
                                                        CUstream cuda_stream = 0);

size_t count_baseline_hash_rows_excluding_keys_on_device(const int8_t* groups_buffer,
                                                         const size_t entry_count,
                                                         const size_t row_size,
                                                         const size_t key_width,
                                                         const int64_t* excluded_keys,
                                                         const size_t excluded_key_count,
                                                         uint64_t* row_count,
                                                         const int device_id,
                                                         CUstream cuda_stream = 0);

void compact_baseline_hash_rows_excluding_keys_on_device(const int8_t* groups_buffer,
                                                         int8_t* compacted_buffer,
                                                         uint64_t* compacted_row_count,
                                                         const size_t entry_count,
                                                         const size_t row_size,
                                                         const size_t key_width,
                                                         const int64_t* excluded_keys,
                                                         const size_t excluded_key_count,
                                                         const int device_id,
                                                         CUstream cuda_stream = 0);

void compact_baseline_hash_rows_matching_keys_on_device(int8_t* groups_buffer,
                                                        int8_t* compacted_buffer,
                                                        uint64_t* compacted_row_count,
                                                        const size_t entry_count,
                                                        const size_t row_size,
                                                        const size_t key_width,
                                                        const int64_t* matching_keys,
                                                        const size_t matching_key_count,
                                                        const bool clear_matching_keys,
                                                        const int device_id,
                                                        CUstream cuda_stream = 0);

void extract_fixed_width_column_from_rows_on_device(const int8_t* rowwise_buffer,
                                                    int8_t* columnar_buffer,
                                                    const size_t entry_count,
                                                    const size_t row_size,
                                                    const size_t source_offset,
                                                    const size_t source_width,
                                                    const size_t output_width,
                                                    const int64_t dict_entry_count,
                                                    const int64_t source_null_val,
                                                    const int64_t normalized_null_val,
                                                    const int device_id,
                                                    CUstream cuda_stream = 0);

bool reduce_baseline_hash_rows_on_device(int8_t* destination_buffer,
                                         const size_t destination_entry_count,
                                         const int8_t* source_buffer,
                                         const size_t source_entry_count,
                                         const size_t row_size,
                                         const size_t key_width,
                                         const size_t key_count,
                                         const DeviceBaselineHashReductionSlot* slots,
                                         const size_t slot_count,
                                         int* error_code,
                                         const int device_id,
                                         CUstream cuda_stream = 0);

bool reduce_baseline_hash_buffers_on_device(int8_t* destination_buffer,
                                            const size_t destination_entry_count,
                                            const int8_t* source_buffers,
                                            const size_t source_entry_count,
                                            const size_t source_buffer_stride,
                                            const size_t source_buffer_count,
                                            const size_t row_size,
                                            const size_t key_width,
                                            const size_t key_count,
                                            const DeviceBaselineHashReductionSlot* slots,
                                            const size_t slot_count,
                                            int* error_code,
                                            const int device_id,
                                            CUstream cuda_stream = 0);

bool reduce_perfect_hash_rows_on_device(int8_t* destination_buffer,
                                        const int8_t* source_buffer,
                                        const size_t entry_count,
                                        const size_t row_size,
                                        const size_t key_width,
                                        const size_t key_count,
                                        const bool keyless,
                                        const size_t key_slot_offset,
                                        const size_t key_slot_width,
                                        const int64_t key_init_val,
                                        const DeviceBaselineHashReductionSlot* slots,
                                        const size_t slot_count,
                                        int* error_code,
                                        const int device_id,
                                        CUstream cuda_stream = 0);

bool compute_columnar_fragment_int_stats_on_device(const int8_t* column_buffer,
                                                   const size_t entry_count,
                                                   const size_t elem_size,
                                                   const int64_t null_val,
                                                   const int device_id,
                                                   DeviceColumnFragmentStats& stats,
                                                   CUstream cuda_stream = 0);

bool compute_columnar_fragment_fp_stats_on_device(const int8_t* column_buffer,
                                                  const size_t entry_count,
                                                  const size_t elem_size,
                                                  const double null_val,
                                                  const int device_id,
                                                  DeviceColumnFragmentStats& stats,
                                                  CUstream cuda_stream = 0);

#endif  // GPUINITGROUPS_H
