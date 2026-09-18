/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef __CUDACC__

#include <cmath>
#include <iostream>
#include <vector>

#include "ThirdParty/robin_hood/robin_hood.h"

inline int log_a_to_base_b(int a, int b) {
  return log2(a) / log2(b);
}

template <typename T1>
struct AngleAttenuationsMap {
  const T1 min_angle;
  const T1 max_angle;
  const int32_t num_segments;
  const int32_t divisions_per_level;
  const int32_t num_levels;
  const T1 angle_ratio;
  const T1 angle_segment_ratio;
  robin_hood::unordered_flat_map<int32_t, T1> attenuation_map;

  void write_angle_attenuation(const int32_t level_idx,
                               const int32_t angle_division,
                               const T1 attenuation_dbm) {
    const int32_t key = level_idx * num_segments + angle_division;
    auto key_itr = attenuation_map.find(key);
    if (key_itr == attenuation_map.end()) {
      attenuation_map[key] = attenuation_dbm;
    } else {
      key_itr->second += attenuation_dbm;
    }
  }

  T1 get_angle_attenuation(const T1 angle) const {
    T1 this_angle_ratio = (angle - min_angle) * angle_ratio;
    T1 angle_attenuation_dbm = 0.0;
    int32_t divisions_for_level = divisions_per_level;
    for (int32_t level_idx = 0; level_idx < num_levels; ++level_idx) {
      const int32_t level_angle_division = this_angle_ratio * divisions_for_level;
      const int32_t key = level_idx * num_segments + level_angle_division;
      auto key_itr = attenuation_map.find(key);
      if (key_itr != attenuation_map.end()) {
        angle_attenuation_dbm += key_itr->second;
      }
      divisions_for_level *= divisions_per_level;
    }
    return angle_attenuation_dbm;
  }

  void add_partial_attenuations(const int32_t start_angle_segment,
                                const int32_t end_angle_segment,
                                const T1 attenuation_dbm,
                                const int32_t level_idx,
                                const int32_t current_level_divisions) {
    const int32_t segments_per_division = num_segments / current_level_divisions;
    const int32_t fill_start_angle_division =
        (start_angle_segment + segments_per_division - 1) / segments_per_division;
    const int32_t fill_end_angle_division =
        (end_angle_segment - segments_per_division + 1) / segments_per_division;
    for (int32_t fill_angle_division = fill_start_angle_division;
         fill_angle_division <= fill_end_angle_division;
         ++fill_angle_division) {
      write_angle_attenuation(level_idx, fill_angle_division, attenuation_dbm);
    }
    if (current_level_divisions < num_segments) {
      const int32_t next_level_divisions = current_level_divisions * divisions_per_level;
      const int32_t fill_start_angle_segment =
          fill_start_angle_division * segments_per_division;
      if (start_angle_segment < fill_start_angle_segment) {
        add_partial_attenuations(start_angle_segment,
                                 fill_start_angle_segment - 1,
                                 attenuation_dbm,
                                 level_idx + 1,
                                 next_level_divisions);
      }
      const int32_t fill_end_angle_segment =
          (fill_end_angle_division + 1) * segments_per_division - 1;
      if (end_angle_segment > fill_end_angle_segment) {
        add_partial_attenuations(fill_end_angle_segment + 1,
                                 end_angle_segment,
                                 attenuation_dbm,
                                 level_idx + 1,
                                 next_level_divisions);
      }
    }
  }

  void add_attenuations(const T1 start_angle,
                        const T1 end_angle,
                        const T1 attenuation_dbm) {
    const int32_t start_angle_segment = (start_angle - min_angle) * angle_segment_ratio;
    const int32_t end_angle_segment = (end_angle - min_angle) * angle_segment_ratio;
    add_partial_attenuations(
        start_angle_segment, end_angle_segment, attenuation_dbm, 0, divisions_per_level);
  }

  AngleAttenuationsMap(const T1 min_angle,
                       const T1 max_angle,
                       const int32_t num_segments,
                       const int32_t divisions_per_level)
      : min_angle(min_angle)
      , max_angle(max_angle)
      , num_segments(num_segments)
      , divisions_per_level(divisions_per_level)
      , num_levels(log_a_to_base_b(num_segments, divisions_per_level))
      , angle_ratio(1.0 / (max_angle - min_angle))
      , angle_segment_ratio(num_segments / (max_angle - min_angle)) {}
};

template <typename T1>
struct AngleAttenuationsFlatMap {
  const T1 min_angle;
  const T1 max_angle;
  const int32_t num_segments;
  const T1 angle_segment_ratio;
  std::vector<T1> attenuations;

  inline T1 get_angle_attenuation(const T1 angle) const {
    const int32_t angle_segment = (angle - min_angle) * angle_segment_ratio;
    return attenuations[angle_segment];
  }

  void add_attenuations(const T1 start_angle,
                        const T1 end_angle,
                        const T1 attenuation_dbm) {
    const int32_t start_angle_segment = (start_angle - min_angle) * angle_segment_ratio;
    const int32_t end_angle_segment = (end_angle - min_angle) * angle_segment_ratio;
    for (int32_t angle_segment = start_angle_segment; angle_segment < end_angle_segment;
         ++angle_segment) {
      attenuations[angle_segment] += attenuation_dbm;
    }
  }

  AngleAttenuationsFlatMap(const T1 min_angle,
                           const T1 max_angle,
                           const int32_t num_segments)
      : min_angle(min_angle)
      , max_angle(max_angle)
      , num_segments(num_segments)
      , angle_segment_ratio(num_segments / (max_angle - min_angle))
      , attenuations(num_segments, 0.0) {}
};

#endif  // __CUDACC__
