/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef HAVE_RF_PROP_TFS
#ifndef __CUDACC__

#include "RFPropTableFunctions.hpp"

void PartitionInfo::permute_by_partition() {
  std::sort(permuted_idxs.begin(),
            permuted_idxs.end(),
            [&](const int32_t& a, const int32_t& b) {
              return (idx_partition_map[a] < idx_partition_map[b]);
            });
}

void PartitionInfo::fill_partition_offsets_and_sizes() {
  int32_t current_partition_idx = 0;
  int32_t current_partition_start = 0;
  for (int32_t e = 0; e < num_elems; ++e) {
    const int32_t elem_partition_idx = idx_partition_map[permuted_idxs[e]];
    assert((elem_partition_idx >= 0 && elem_partition_idx < num_partitions) ||
           elem_partition_idx == PartitionInfo::invalid_partition_idx);
    if (elem_partition_idx == PartitionInfo::invalid_partition_idx) {
      first_valid_element = e + 1;
      continue;
    }

    if (elem_partition_idx != current_partition_idx) {
      const int32_t current_partition_num_elems = e - current_partition_start;
      if (current_partition_num_elems > 0) {
        partition_offsets_and_sizes[current_partition_idx] =
            std::make_pair(current_partition_start, current_partition_num_elems);
        num_non_sparse_partitions++;
      }
      current_partition_idx = elem_partition_idx;
      current_partition_start = e;
    }
  }
  const int32_t current_partition_num_elems = num_elems - current_partition_start;
  if (current_partition_num_elems > 0) {
    partition_offsets_and_sizes[current_partition_idx] =
        std::make_pair(current_partition_start, current_partition_num_elems);
    num_non_sparse_partitions++;
  }
}

double get_min_distance_for_power_frequency(const double receiver_power_threshold_dbm,
                                            const double source_power_dbm,
                                            const double signal_frequency_mhz) {
  // Use Free Space Power Loss equation to solve for max distance (meters)
  // to achieve at least receiver_power_threshold (db), given source power (db)
  return pow(10.0,
             (receiver_power_threshold_dbm - source_power_dbm +
              20.0 * log10(signal_frequency_mhz) - 27.55) *
                 -0.05);
}

double get_min_distance_for_power_frequency_antenna_max_gain(
    const double receiver_power_threshold_dbm,
    const double source_power_dbm,
    const double signal_frequency_mhz,
    const double antenna_max_gain_dbm) {
  // Use Free Space Power Loss equation to solve for max distance (meters)
  // to achieve at least receiver_power_threshold (dBm), given source power (dBm)

  return pow(10.0,
             (receiver_power_threshold_dbm - source_power_dbm - antenna_max_gain_dbm +
              20.0 * log10(signal_frequency_mhz) - 27.55) *
                 -0.05);
}

#endif  // __CUDACC__
#endif  // HAVE_TF_RF_PROP_TFS
