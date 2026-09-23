/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef __CUDACC__

#include <vector>
#include "QueryEngine/heavydbTypes.h"

enum class AntennaAxis { HORIZONTAL, VERTICAL };

struct AntennaPattern {
  bool is_isotropic;
  double gain;
  std::vector<double> horizontal_pattern;
  std::vector<double> vertical_pattern;
  double max_horizontal_gain;
  double max_vertical_gain;

  AntennaPattern(const std::string& ant_file_path);

  template <typename T>
  AntennaPattern(const T antenna_gain,
                 const Array<T>& antenna_horizontal_degrees,
                 const Array<T>& antenna_horizontal_attenuations,
                 const Array<T>& antenna_vertical_degrees,
                 const Array<T>& antenna_vertical_attenuations);

  AntennaPattern()
      : is_isotropic(true)
      , gain(0.0)
      , horizontal_pattern(360, 0.0)
      , vertical_pattern(360, 0.0)
      , max_horizontal_gain(0.0)
      , max_vertical_gain(0.0) {}

  template <typename T>
  void parse_2d_antenna_pattern(const Array<T>& antenna_degrees,
                                const Array<T>& antenna_attenuations,
                                std::vector<double>& antenna_pattern,
                                double& max_gain);

  template <typename T>
  T get_directional_tx_power_component(const T antenna_azimuth_degrees,
                                       const T bearing_degrees,
                                       const AntennaAxis axis) const;
};

struct AntennaPatternMap {
  std::vector<AntennaPattern> antenna_patterns_storage;
  std::vector<int64_t> antenna_pattern_map;

  const AntennaPattern& operator[](int64_t index) const {
    return antenna_patterns_storage[antenna_pattern_map[index]];
  }
};

template <typename T>
AntennaPatternMap generate_antenna_pattern_map(
    const Column<TextEncodingDict>& rf_source_antenna_type,
    const Column<TextEncodingDict>& antenna_types,
    const Column<T>& antenna_gains,
    const Column<Array<T>>& antenna_horizontal_degrees,
    const Column<Array<T>>& antenna_horizontal_attenuations,
    const Column<Array<T>>& antenna_vertical_degrees,
    const Column<Array<T>>& antenna_vertical_attenuations);

#endif  // __CUDACC__
