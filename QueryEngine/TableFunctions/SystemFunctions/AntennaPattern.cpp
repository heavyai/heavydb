/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __CUDACC__

#include <fstream>

#include "AntennaPattern.h"

template <typename T>
void AntennaPattern::parse_2d_antenna_pattern(const Array<T>& antenna_degrees,
                                              const Array<T>& antenna_attenuations,
                                              std::vector<double>& antenna_pattern,
                                              double& max_gain) {
  const auto num_entries = antenna_degrees.size();
  if (num_entries != antenna_attenuations.size()) {
    throw std::runtime_error(
        "Number of entries in antenna degrees and attenuations array do not match.");
  }
  if (num_entries == 0) {
    throw std::runtime_error("Antenna pattern cannot be empty");
  }
  antenna_pattern.resize(360, std::numeric_limits<double>::lowest());
  max_gain = std::numeric_limits<double>::lowest();
  for (size_t entry_idx = 0; entry_idx < num_entries; ++entry_idx) {
    int64_t rounded_degree =
        static_cast<int64_t>(std::round(std::fmod(antenna_degrees[entry_idx], 360.0)));
    if (rounded_degree < 0) {
      rounded_degree += 360;
    }
    const auto gain = -antenna_attenuations[entry_idx];
    antenna_pattern[rounded_degree] = gain;
    if (gain > max_gain) {
      max_gain = gain;
    }
  }
  int64_t last_valid_degree = -1;
  double last_gain_value = std::numeric_limits<double>::lowest();
  int64_t first_valid_degree = -1;
  for (int64_t degree = 0; degree < 360; ++degree) {
    const auto gain_value = antenna_pattern[degree];
    if (gain_value != std::numeric_limits<double>::lowest()) {
      if (last_gain_value != std::numeric_limits<double>::lowest()) {
        const int64_t degrees_diff = (degree - last_valid_degree);
        const double degrees_diff_denom = 1.0 / degrees_diff;
        for (int64_t fill_degree = last_valid_degree + 1; fill_degree < degree;
             ++fill_degree) {
          // Take weighted average between last valid gain value and this one,
          // interpolated by degree
          antenna_pattern[fill_degree] =
              (fill_degree - last_valid_degree) * degrees_diff_denom * gain_value +
              (degree - fill_degree) * degrees_diff_denom * last_gain_value;
        }
      } else {
        first_valid_degree = degree;
      }
      last_valid_degree = degree;
      last_gain_value = gain_value;
    }
  }
  int64_t first_gain_value = antenna_pattern[first_valid_degree];
  const int64_t degrees_diff = (360 + first_valid_degree - last_valid_degree);
  const double degrees_diff_denom = 1.0 / degrees_diff;
  for (int64_t raw_degree = last_valid_degree + 1; raw_degree < 360 + first_valid_degree;
       ++raw_degree) {
    const int64_t degree = raw_degree % 360;
    antenna_pattern[degree] =
        (raw_degree - last_valid_degree) * degrees_diff_denom * first_gain_value +
        (first_valid_degree + 360 - raw_degree) * degrees_diff_denom * last_gain_value;
  }
}

template void AntennaPattern::parse_2d_antenna_pattern(
    const Array<float>& antenna_degrees,
    const Array<float>& antenna_attenuations,
    std::vector<double>& antenna_pattern,
    double& max_gain);

template void AntennaPattern::parse_2d_antenna_pattern(
    const Array<double>& antenna_degrees,
    const Array<double>& antenna_attenuations,
    std::vector<double>& antenna_pattern,
    double& max_gain);

AntennaPattern::AntennaPattern(const std::string& ant_file_path) {
  std::ifstream ant_file(ant_file_path);
  if (!ant_file.is_open()) {
    std::string error_str = "Could not open ANT file (" + ant_file_path + ").";
    throw std::runtime_error(error_str);
  }
  horizontal_pattern.reserve(360);
  vertical_pattern.reserve(360);
  std::string line;
  size_t line_idx = 0;
  max_horizontal_gain = std::numeric_limits<double>::lowest();
  while (std::getline(ant_file, line) && line_idx++ < 360) {
    const double horizontal_gain = std::stod(line);
    horizontal_pattern.emplace_back(horizontal_gain);
    if (horizontal_gain > max_horizontal_gain) {
      max_horizontal_gain = horizontal_gain;
    }
  }
  max_vertical_gain = std::numeric_limits<double>::lowest();
  while (std::getline(ant_file, line) && line_idx++ < 720) {
    const double vertical_gain = std::stod(line);
    vertical_pattern.emplace_back(vertical_gain);
    if (vertical_gain > max_horizontal_gain) {
      max_vertical_gain = vertical_gain;
    }
  }
  ant_file.close();
  if (line_idx != 720) {
    std::string error_str =
        "ANT file (" + ant_file_path + ") does not have required 720 lines.";
    throw std::runtime_error(error_str);
  }
}

template <typename T>
AntennaPattern::AntennaPattern(const T antenna_gain,
                               const Array<T>& antenna_horizontal_degrees,
                               const Array<T>& antenna_horizontal_attenuations,
                               const Array<T>& antenna_vertical_degrees,
                               const Array<T>& antenna_vertical_attenuations)
    : is_isotropic(false), gain(antenna_gain) {
  parse_2d_antenna_pattern(antenna_horizontal_degrees,
                           antenna_horizontal_attenuations,
                           horizontal_pattern,
                           max_horizontal_gain);
  parse_2d_antenna_pattern(antenna_vertical_degrees,
                           antenna_vertical_attenuations,
                           vertical_pattern,
                           max_vertical_gain);
}

template AntennaPattern::AntennaPattern(
    const float antenna_gain,
    const Array<float>& antenna_horizontal_degrees,
    const Array<float>& antenna_horizontal_attenuations,
    const Array<float>& antenna_vertical_degrees,
    const Array<float>& antenna_vertical_attenuations);

template AntennaPattern::AntennaPattern(
    const double antenna_gain,
    const Array<double>& antenna_horizontal_degrees,
    const Array<double>& antenna_horizontal_attenuations,
    const Array<double>& antenna_vertical_degrees,
    const Array<double>& antenna_vertical_attenuations);

template <typename T>
T AntennaPattern::get_directional_tx_power_component(const T antenna_azimuth_degrees,
                                                     const T bearing_degrees,
                                                     const AntennaAxis axis) const {
  if (is_isotropic) {
    return 0.0;
  }
  double relative_bearing_degrees =
      std::fmod(bearing_degrees - antenna_azimuth_degrees, 360.0);
  if (relative_bearing_degrees < 0) {
    relative_bearing_degrees = 360 + relative_bearing_degrees;
  }
  const int64_t floor_degree = std::floor(relative_bearing_degrees);
  int64_t unnormalized_ceiling_degree = std::ceil(relative_bearing_degrees);
  const int64_t ceiling_degree =
      unnormalized_ceiling_degree == 360 ? 0 : unnormalized_ceiling_degree;
  const double ceiling_weight = std::fmod(relative_bearing_degrees, 1.0);
  const double floor_weight = 1.0 - ceiling_weight;
  switch (axis) {
    case AntennaAxis::HORIZONTAL: {
      return static_cast<T>(horizontal_pattern[floor_degree] * floor_weight +
                            horizontal_pattern[ceiling_degree] * ceiling_weight);
    }
    case AntennaAxis::VERTICAL: {
      return static_cast<T>(vertical_pattern[floor_degree] * floor_weight +
                            vertical_pattern[ceiling_degree] * ceiling_weight);
    }
    default: {
      // Make compiler happy
      UNREACHABLE();
      return 0;
    }
  }
}

template float AntennaPattern::get_directional_tx_power_component(
    const float antenna_azimuth_degrees,
    const float bearing_degrees,
    const AntennaAxis axis) const;

template double AntennaPattern::get_directional_tx_power_component(
    const double antenna_azimuth_degrees,
    const double bearing_degrees,
    const AntennaAxis axis) const;

template <typename T>
AntennaPatternMap generate_antenna_pattern_map(
    const Column<TextEncodingDict>& rf_source_antenna_type,
    const Column<TextEncodingDict>& antenna_types,
    const Column<T>& antenna_gain,
    const Column<Array<T>>& antenna_horizontal_degrees,
    const Column<Array<T>>& antenna_horizontal_attenuations,
    const Column<Array<T>>& antenna_vertical_degrees,
    const Column<Array<T>>& antenna_vertical_attenuations) {
  const int64_t num_rf_sources = rf_source_antenna_type.size();
  AntennaPatternMap antenna_pattern_map;
  std::map<std::string, int64_t> antenna_type_to_pattern_idx_map;
  const int64_t num_antenna_patterns = static_cast<int64_t>(antenna_types.size());
  for (int64_t antenna_type_idx = 0; antenna_type_idx < num_antenna_patterns;
       ++antenna_type_idx) {
    const std::string lowercase_antenna_type =
        boost::algorithm::to_lower_copy(antenna_types.getString(antenna_type_idx));
    antenna_type_to_pattern_idx_map.insert(
        std::make_pair(lowercase_antenna_type, antenna_type_idx));
  }
  std::map<int64_t, int64_t> ant_pattern_idx_to_antenna_pattern_map;
  int64_t default_ant_pattern_idx = -1;
  const auto default_itr = antenna_type_to_pattern_idx_map.find("default");
  if (default_itr != antenna_type_to_pattern_idx_map.end()) {
    default_ant_pattern_idx = default_itr->second;
    antenna_pattern_map.antenna_patterns_storage.emplace_back(
        antenna_gain[default_ant_pattern_idx],
        antenna_horizontal_degrees(default_ant_pattern_idx),
        antenna_horizontal_attenuations(default_ant_pattern_idx),
        antenna_vertical_degrees(default_ant_pattern_idx),
        antenna_vertical_attenuations(default_ant_pattern_idx));
    // ant_pattern_idx_to_antenna_pattern_map.insert(std::make_pair(
    //    default_ant_pattern_idx, &antenna_pattern_map.antenna_patterns.back()));
  } else {
    default_ant_pattern_idx = num_antenna_patterns;
    antenna_type_to_pattern_idx_map.insert(
        std::make_pair("default", default_ant_pattern_idx));
    antenna_pattern_map.antenna_patterns_storage.emplace_back();
  }
  ant_pattern_idx_to_antenna_pattern_map.insert(std::make_pair(
      default_ant_pattern_idx,
      static_cast<int64_t>(antenna_pattern_map.antenna_patterns_storage.size() - 1)));

  for (int64_t tower_idx = 0; tower_idx < num_rf_sources; ++tower_idx) {
    if (rf_source_antenna_type.isNull(tower_idx)) {
      antenna_pattern_map.antenna_pattern_map.emplace_back(
          ant_pattern_idx_to_antenna_pattern_map[default_ant_pattern_idx]);
      continue;
    }
    const std::string tower_type_str =
        boost::algorithm::to_lower_copy(rf_source_antenna_type.getString(tower_idx));
    const auto ant_itr = antenna_type_to_pattern_idx_map.find(tower_type_str);
    if (ant_itr == antenna_type_to_pattern_idx_map.end()) {
      antenna_pattern_map.antenna_pattern_map.emplace_back(
          ant_pattern_idx_to_antenna_pattern_map[default_ant_pattern_idx]);
      continue;
    }
    // If here antenna type was found
    const auto pattern_idx = ant_itr->second;
    const auto ant_pattern_itr = ant_pattern_idx_to_antenna_pattern_map.find(pattern_idx);
    if (ant_pattern_itr == ant_pattern_idx_to_antenna_pattern_map.end()) {
      // If here we haven't read in the pattern yet
      antenna_pattern_map.antenna_patterns_storage.emplace_back(
          antenna_gain[pattern_idx],
          antenna_horizontal_degrees(pattern_idx),
          antenna_horizontal_attenuations(pattern_idx),
          antenna_vertical_degrees(pattern_idx),
          antenna_vertical_attenuations(pattern_idx));
      antenna_pattern_map.antenna_pattern_map.emplace_back(
          static_cast<int64_t>(antenna_pattern_map.antenna_patterns_storage.size() - 1));
      ant_pattern_idx_to_antenna_pattern_map.insert(std::make_pair(
          default_ant_pattern_idx,
          static_cast<int64_t>(antenna_pattern_map.antenna_patterns_storage.size() - 1)));
    } else {
      antenna_pattern_map.antenna_pattern_map.emplace_back(ant_pattern_itr->second);
    }
  }
  return antenna_pattern_map;
}

template AntennaPatternMap generate_antenna_pattern_map(
    const Column<TextEncodingDict>& rf_source_antenna_type,
    const Column<TextEncodingDict>& antenna_types,
    const Column<float>& antenna_gain,
    const Column<Array<float>>& antenna_horizontal_degrees,
    const Column<Array<float>>& antenna_horizontal_attenuations,
    const Column<Array<float>>& antenna_vertical_degrees,
    const Column<Array<float>>& antenna_vertical_attenuations);

template AntennaPatternMap generate_antenna_pattern_map(
    const Column<TextEncodingDict>& rf_source_antenna_type,
    const Column<TextEncodingDict>& antenna_types,
    const Column<double>& antenna_gain,
    const Column<Array<double>>& antenna_horizontal_degrees,
    const Column<Array<double>>& antenna_horizontal_attenuations,
    const Column<Array<double>>& antenna_vertical_degrees,
    const Column<Array<double>>& antenna_vertical_attenuations);

#endif  // __CUDACC__
