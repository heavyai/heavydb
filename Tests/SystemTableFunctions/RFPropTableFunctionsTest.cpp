/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef HAVE_RF_PROP_TFS

#include "Tests/TestHelpers.h"

#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

extern bool g_enable_table_functions;
extern bool g_enable_rf_prop_table_functions;

namespace {

inline void run_ddl_statement(const std::string& stmt) {
  QR::get()->runDDLStatement(stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, false, false);
}

}  // namespace

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

class RFPropTFs : public ::testing::Test {
  void SetUp() override {}
};

double calc_receiver_power_dbm(const double repeater_signal_strength_dbm,
                               const double repeater_frequency_mhz,
                               const double repeater_distance_meters) {
  return repeater_signal_strength_dbm - (20.0 * log10(repeater_frequency_mhz) +
                                         20.0 * log10(repeater_distance_meters) - 27.55);
}

inline size_t x_y_bin_to_bin_index(const int64_t x_bin,
                                   const int64_t y_bin,
                                   const int64_t num_x_bins) {
  return y_bin * num_x_bins + x_bin;
}

struct RepeaterInfo {
  int32_t repeater_id = -1;
  double repeater_x = 0;
  double repeater_y = 0;
  double repeater_z = 0;

  // Following are only populated by validate_simple_rf_prop_max_signal_results
  int32_t repeater_x_bin = 0;
  int32_t repeater_y_bin = 0;
};

struct VariantElevationBin {
  int32_t x_bin;
  int32_t y_bin;
  float elevation;
};

struct TerrainBin {
  int32_t x_bin;
  int32_t y_bin;
};

struct RFPropParams {
  double bin_dim_meters;
  size_t num_x_bins;
  size_t num_y_bins;
  double primary_elevation;
  std::vector<VariantElevationBin> variant_elevation_bins;
  std::vector<RepeaterInfo> repeaters;
  std::vector<TerrainBin> obscured_bins;
  bool repeater_height_is_relative;
  double repeater_signal_strength_dbm;
  double repeater_frequency_mhz;
  double repeater_antenna_azimuth_degrees;
  double repeater_antenna_downtilt_degrees;
  std::string repeater_antenna_type;

  // "Advanced" parameters below not specifiable in constructor but are given default
  // values However they can be overridden manually
  bool geographic_coords{false};
  double max_ray_travel_meters{2000.0};
  size_t num_rays_per_source{1800};
  double min_receiver_signal_strength_dbm{-80.0};
  double assumed_source_height_above_ground{10.0};
  double ray_step_bin_multiple{
      0.01};  // Normally set to 1.0 but set lower here to tighten epsilons for testing
  size_t loop_grain_size{
      1};  // Normally 40 but set to 1 for testing to ensure multi-threaded correctness

  // Only used by tf_rf_prop
  int64_t num_top_sources_per_terrain_bin{3};

  RFPropParams(const double bin_dim_meters,
               const size_t num_x_bins,
               const size_t num_y_bins,
               const double primary_elevation,
               const std::vector<VariantElevationBin>& variant_elevation_bins,
               const std::vector<RepeaterInfo>& repeaters,
               const std::vector<TerrainBin>& obscured_bins,
               const bool repeater_height_is_relative,
               const double repeater_signal_strength_dbm,
               const double repeater_frequency_mhz)
      : bin_dim_meters(bin_dim_meters)
      , num_x_bins(num_x_bins)
      , num_y_bins(num_y_bins)
      , primary_elevation(primary_elevation)
      , variant_elevation_bins(variant_elevation_bins)
      , repeaters(repeaters)
      , obscured_bins(obscured_bins)
      , repeater_height_is_relative(repeater_height_is_relative)
      , repeater_signal_strength_dbm(repeater_signal_strength_dbm)
      , repeater_frequency_mhz(repeater_frequency_mhz)
      , repeater_antenna_azimuth_degrees(0.0)
      , repeater_antenna_downtilt_degrees(0.0)
      , repeater_antenna_type("default") {}

  RFPropParams(const double bin_dim_meters,
               const size_t num_x_bins,
               const size_t num_y_bins,
               const double primary_elevation,
               const std::vector<VariantElevationBin>& variant_elevation_bins,
               const std::vector<RepeaterInfo>& repeaters,
               const std::vector<TerrainBin>& obscured_bins,
               const bool repeater_height_is_relative,
               const double repeater_signal_strength_dbm,
               const double repeater_frequency_mhz,
               const double repeater_antenna_azimuth_degrees,
               const double repeater_antenna_downtilt_degrees,
               const std::string repeater_antenna_type)
      : bin_dim_meters(bin_dim_meters)
      , num_x_bins(num_x_bins)
      , num_y_bins(num_y_bins)
      , primary_elevation(primary_elevation)
      , variant_elevation_bins(variant_elevation_bins)
      , repeaters(repeaters)
      , obscured_bins(obscured_bins)
      , repeater_height_is_relative(repeater_height_is_relative)
      , repeater_signal_strength_dbm(repeater_signal_strength_dbm)
      , repeater_frequency_mhz(repeater_frequency_mhz)
      , repeater_antenna_azimuth_degrees(repeater_antenna_azimuth_degrees)
      , repeater_antenna_downtilt_degrees(repeater_antenna_downtilt_degrees)
      , repeater_antenna_type(repeater_antenna_type) {}
};

std::unordered_map<size_t, float> generate_variant_elevation_bin_map(
    const RFPropParams& rf_prop_params) {
  std::unordered_map<size_t, float> variant_elevation_bin_map;
  for (const auto& variant_elevation_bin : rf_prop_params.variant_elevation_bins) {
    const size_t variant_composite_bin_idx =
        x_y_bin_to_bin_index(variant_elevation_bin.x_bin,
                             variant_elevation_bin.y_bin,
                             rf_prop_params.num_x_bins);
    CHECK_LT(variant_composite_bin_idx,
             rf_prop_params.num_x_bins * rf_prop_params.num_y_bins);
    CHECK(variant_elevation_bin_map
              .insert(std::make_pair(variant_composite_bin_idx,
                                     variant_elevation_bin.elevation))
              .second);
  }
  return variant_elevation_bin_map;
}

float get_elevation_for_bin(
    const RFPropParams& rf_prop_params,
    const std::unordered_map<size_t, float>& variant_elevation_bin_map,
    const size_t x_bin,
    const size_t y_bin) {
  float bin_elevation = rf_prop_params.primary_elevation;
  const size_t composite_bin_idx =
      x_y_bin_to_bin_index(x_bin, y_bin, rf_prop_params.num_x_bins);
  const auto map_itr = variant_elevation_bin_map.find(composite_bin_idx);
  if (map_itr != variant_elevation_bin_map.end()) {
    bin_elevation = map_itr->second;
  }
  return bin_elevation;
}

std::unordered_set<size_t> generate_obscured_bin_set(const RFPropParams& rf_prop_params) {
  std::unordered_set<size_t> obscured_bin_set;
  for (const auto& obscured_bin : rf_prop_params.obscured_bins) {
    const size_t obscured_composite_bin_idx = x_y_bin_to_bin_index(
        obscured_bin.x_bin, obscured_bin.y_bin, rf_prop_params.num_x_bins);
    CHECK_LT(obscured_composite_bin_idx,
             rf_prop_params.num_x_bins * rf_prop_params.num_y_bins);
    CHECK(obscured_bin_set.insert(obscured_composite_bin_idx).second);
  }
  return obscured_bin_set;
}

bool is_bin_obscured(const RFPropParams& rf_prop_params,
                     const std::unordered_set<size_t>& obscured_bin_set,
                     const int32_t x_bin,
                     const int32_t y_bin) {
  const size_t composite_bin_idx =
      x_y_bin_to_bin_index(x_bin, y_bin, rf_prop_params.num_x_bins);
  return obscured_bin_set.find(composite_bin_idx) != obscured_bin_set.end();
}

enum RFPropFunctionType {
  RF_PROP_MAX_SIGNAL,
  RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ,
  RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS,
  RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION,
  RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY,
  RF_PROP_TOP_K
};

std::string get_rf_prop_function_name(const RFPropFunctionType rf_prop_function_type) {
  switch (rf_prop_function_type) {
    case RFPropFunctionType::RF_PROP_MAX_SIGNAL:
      return "tf_rf_prop_max_signal";
    case RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ:
      return "tf_rf_prop_max_signal";
    case RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS:
      return "tf_rf_prop_max_signal";
    case RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION:
      return "tf_rf_prop_max_signal";
    case RFPropFunctionType::
        RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY:
      return "tf_rf_prop_max_signal";
    case RFPropFunctionType::RF_PROP_TOP_K:
      return "tf_rf_prop";
    default:
      UNREACHABLE();
      return "";
  }
}

std::string generate_rf_prop_query(const RFPropParams& rf_prop_params,
                                   const RFPropFunctionType rf_prop_function_type,
                                   const bool use_named_args) {
  const auto variant_elevation_bin_map =
      generate_variant_elevation_bin_map(rf_prop_params);
  std::ostringstream terrain_oss;
  terrain_oss << "CURSOR(SELECT CAST(x AS DOUBLE) AS x, CAST(y AS DOUBLE) AS y, "
                 " CAST(z AS FLOAT) AS z_ground";
  //  FROM (VALUES ";
  if (rf_prop_function_type ==
          RFPropFunctionType::
              RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION ||
      rf_prop_function_type ==
          RFPropFunctionType::
              RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY) {
    terrain_oss
        << ", CAST(z AS FLOAT) AS z_terrain, CAST(0.0 AS FLOAT) AS terrain_attenuation";
  }
  terrain_oss << " FROM (VALUES";
  for (size_t y_bin = 0; y_bin < rf_prop_params.num_y_bins; ++y_bin) {
    const double y_val = y_bin * rf_prop_params.bin_dim_meters;
    for (size_t x_bin = 0; x_bin < rf_prop_params.num_x_bins; ++x_bin) {
      const double x_val = x_bin * rf_prop_params.bin_dim_meters;
      const float bin_elevation =
          get_elevation_for_bin(rf_prop_params, variant_elevation_bin_map, x_bin, y_bin);
      terrain_oss << "(" << x_val << ", " << y_val << ", " << bin_elevation << ")";
      if (!(y_bin == rf_prop_params.num_y_bins - 1 &&
            x_bin == rf_prop_params.num_x_bins - 1)) {
        terrain_oss << ", ";
      }
    }
  }
  terrain_oss << ") AS t(x, y, z))";
  const auto terrain_cursor = terrain_oss.str();

  std::ostringstream repeaters_oss;

  if (rf_prop_function_type ==
      RFPropFunctionType::
          RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY) {
    repeaters_oss
        << "CURSOR(SELECT ARRAY[CAST(id AS BIGINT)] AS id, CAST(x AS DOUBLE) AS x, "
           "CAST(y AS "
           "DOUBLE) AS y, CAST(z AS FLOAT) AS z, ARRAY[CAST(tx_power_watts AS DOUBLE)] "
           "AS tx_power_watts, "
           "ARRAY[CAST(tx_freq_mhz AS DOUBLE)] AS tx_freq_mhz, "
           "ARRAY[CAST(antenna_azimuth_degrees AS DOUBLE)] AS antenna_azimuth_degrees, "
           "ARRAY[CAST(antenna_downtilt_degrees AS DOUBLE)] AS antenna_downtilt_degrees, "
           "ARRAY[CAST(antenna_type AS TEXT)] AS antenna_type";
  } else {
    repeaters_oss
        << "CURSOR(SELECT CAST(id AS INTEGER) AS id, CAST(x AS DOUBLE) AS x, CAST(y AS "
           "DOUBLE) AS y, CAST(z AS FLOAT) AS z";
    if (rf_prop_function_type == RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ ||
        rf_prop_function_type ==
            RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS ||
        rf_prop_function_type ==
            RFPropFunctionType::
                RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION) {
      repeaters_oss << ", CAST(tx_power_watts AS DOUBLE) AS tx_power_watts, "
                       "CAST(tx_freq_mhz AS DOUBLE) AS tx_freq_mhz";
    }
    if (rf_prop_function_type ==
            RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS ||
        rf_prop_function_type ==
            RFPropFunctionType::
                RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION) {
      repeaters_oss
          << ", CAST(antenna_azimuth_degrees AS DOUBLE) AS antenna_azimuth_degrees, "
             "CAST(antenna_downtilt_degrees AS DOUBLE) AS antenna_downtilt_degrees, "
             "antenna_type";
    }
  }

  repeaters_oss << " FROM (VALUES ";
  size_t repeater_idx = 0;
  const double tx_power_watts =
      pow(10.0, rf_prop_params.repeater_signal_strength_dbm / 10.0) * 0.001;
  for (const auto& repeater : rf_prop_params.repeaters) {
    repeaters_oss << "(" << repeater.repeater_id << ", " << repeater.repeater_x << ", "
                  << repeater.repeater_y << ", " << repeater.repeater_z;
    if (rf_prop_function_type == RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ ||
        rf_prop_function_type ==
            RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS ||
        rf_prop_function_type ==
            RFPropFunctionType::
                RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION ||
        rf_prop_function_type ==
            RFPropFunctionType::
                RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY) {
      repeaters_oss << ", " << tx_power_watts << ", "
                    << rf_prop_params.repeater_frequency_mhz;
    }
    if (rf_prop_function_type ==
            RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS ||
        rf_prop_function_type ==
            RFPropFunctionType::
                RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION ||
        rf_prop_function_type ==
            RFPropFunctionType::
                RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY) {
      repeaters_oss << ", " << rf_prop_params.repeater_antenna_azimuth_degrees << ", "
                    << rf_prop_params.repeater_antenna_downtilt_degrees << ", "
                    << "CAST('" << rf_prop_params.repeater_antenna_type << "' AS TEXT)";
    }
    repeaters_oss << ")";
    if (++repeater_idx != rf_prop_params.repeaters.size()) {
      repeaters_oss << ", ";
    }
  }
  repeaters_oss << ") AS t(id, x, y, z";
  if (rf_prop_function_type == RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ ||
      rf_prop_function_type ==
          RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS ||
      rf_prop_function_type ==
          RFPropFunctionType::
              RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION ||
      rf_prop_function_type ==
          RFPropFunctionType::
              RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY) {
    repeaters_oss << ", tx_power_watts, tx_freq_mhz";
  }
  if (rf_prop_function_type ==
          RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS ||
      rf_prop_function_type ==
          RFPropFunctionType::
              RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION ||
      rf_prop_function_type ==
          RFPropFunctionType::
              RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY) {
    repeaters_oss << ", antenna_azimuth_degrees, antenna_downtilt_degrees, antenna_type";
  }
  // Add an id >= 0 filter to allow us to construct empty cursors
  repeaters_oss << ") where id >= 0)";
  const auto repeaters_cursor = repeaters_oss.str();

  std::ostringstream antenna_oss;
  antenna_oss << "CURSOR(SELECT antenna_type, CAST(antenna_gain as DOUBLE), "
                 "antenna_horizontal_degrees, antenna_horizontal_attenuation, "
                 " antenna_vertical_degrees, antenna_vertical_attenuation FROM (VALUES ";
  antenna_oss
      << "("
      << "CAST('" << rf_prop_params.repeater_antenna_type << "' AS TEXT), "
      << "CAST(0.0 AS DOUBLE), ARRAY[CAST(0.0 AS DOUBLE)], ARRAY[CAST(0.0 AS DOUBLE)], "
      << "ARRAY[CAST(0.0 AS DOUBLE)], ARRAY[CAST(0.0 AS DOUBLE)])) AS "
      << " t(antenna_type, antenna_gain, antenna_horizontal_degrees, "
         "antenna_horizontal_attenuation, antenna_vertical_degrees, "
         "antenna_vertical_attenuation))";
  const auto antenna_cursor = antenna_oss.str();

  const std::string rf_source_z_is_relative =
      rf_prop_params.repeater_height_is_relative ? "true" : "false";
  const std::string geo_coords = rf_prop_params.geographic_coords ? "true" : "false";
  const std::string optional_top_k_sources =
      rf_prop_function_type == RFPropFunctionType::RF_PROP_TOP_K
          ? std::to_string(rf_prop_params.num_top_sources_per_terrain_bin)
          : "";

  const std::string function_name = get_rf_prop_function_name(rf_prop_function_type);

  std::string order_by_clause = "ORDER BY y ASC, x ASC";
  if (rf_prop_function_type == RFPropFunctionType::RF_PROP_TOP_K) {
    order_by_clause += ", rf_signal_strength_dbm DESC";
  }

  std::ostringstream query_oss;
  if (use_named_args) {
    switch (rf_prop_function_type) {
      case RFPropFunctionType::RF_PROP_TOP_K: {
        query_oss << "SELECT * FROM TABLE(" << function_name << "("
                  << "rf_sources => " << repeaters_cursor
                  << ", rf_source_z_is_relative_to_terrain => " << rf_source_z_is_relative
                  << ", rf_source_signal_strength_dbm => "
                  << rf_prop_params.repeater_signal_strength_dbm
                  << ", rf_source_signal_frequency_mhz => "
                  << rf_prop_params.repeater_frequency_mhz << ", terrain_elevations => "
                  << terrain_cursor << ", geographic_coords => " << geo_coords
                  << ", bin_dim_meters => " << rf_prop_params.bin_dim_meters
                  << ", strongest_k_sources_per_terrain_bin => " << optional_top_k_sources
                  << ", max_ray_travel_meters => " << rf_prop_params.max_ray_travel_meters
                  << ", num_rays_per_source => " << rf_prop_params.num_rays_per_source
                  << ", min_receiver_signal_strength_dbm => "
                  << rf_prop_params.min_receiver_signal_strength_dbm
                  << ", default_source_height_agl_meters => "
                  << rf_prop_params.assumed_source_height_above_ground
                  << ", ray_step_bin_multiple => " << rf_prop_params.ray_step_bin_multiple
                  << ", loop_grain_size => " << rf_prop_params.loop_grain_size << ")) "
                  << order_by_clause << ";";
        break;
      }
      case RFPropFunctionType::RF_PROP_MAX_SIGNAL: {
        query_oss << "SELECT * FROM TABLE(" << function_name << "("
                  << "rf_sources => " << repeaters_cursor
                  << ", rf_source_z_is_relative_to_terrain => " << rf_source_z_is_relative
                  << ", rf_source_signal_strength_dbm => "
                  << rf_prop_params.repeater_signal_strength_dbm
                  << ", rf_source_signal_frequency_mhz => "
                  << rf_prop_params.repeater_frequency_mhz << ", terrain_elevations => "
                  << terrain_cursor << ", geographic_coords => " << geo_coords
                  << ", bin_dim_meters => " << rf_prop_params.bin_dim_meters
                  << ", max_ray_travel_meters => " << rf_prop_params.max_ray_travel_meters
                  << ", num_rays_per_source => " << rf_prop_params.num_rays_per_source
                  << ", min_receiver_signal_strength_dbm => "
                  << rf_prop_params.min_receiver_signal_strength_dbm
                  << ", default_source_height_agl_meters => "
                  << rf_prop_params.assumed_source_height_above_ground
                  << ", ray_step_bin_multiple => " << rf_prop_params.ray_step_bin_multiple
                  << ", loop_grain_size => " << rf_prop_params.loop_grain_size << ")) "
                  << order_by_clause << ";";
        break;
      }
      case RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ: {
        query_oss << "SELECT * FROM TABLE(" << function_name << "("
                  << "rf_sources => " << repeaters_cursor << ", terrain_elevations => "
                  << terrain_cursor << ", rf_source_z_is_relative_to_terrain => "
                  << rf_source_z_is_relative << ", geographic_coords => " << geo_coords
                  << ", bin_dim_meters => " << rf_prop_params.bin_dim_meters
                  << ", max_ray_travel_meters => " << rf_prop_params.max_ray_travel_meters
                  << ", initial_rays_per_source => " << rf_prop_params.num_rays_per_source
                  << ", rays_per_bin_autosplit_threshold => " << 0
                  << ", min_receiver_signal_strength_dbm => "
                  << rf_prop_params.min_receiver_signal_strength_dbm
                  << ", default_source_height_agl_meters => "
                  << rf_prop_params.assumed_source_height_above_ground
                  << ", ray_step_bin_multiple => " << rf_prop_params.ray_step_bin_multiple
                  << ", loop_grain_size => " << rf_prop_params.loop_grain_size << ")) "
                  << order_by_clause << ";";
        break;
      }
      case RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS:
      case RFPropFunctionType::
          RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION:
      case RFPropFunctionType::
          RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY: {
        query_oss << "SELECT * FROM TABLE(" << function_name << "("
                  << "rf_sources => " << repeaters_cursor << ", terrain_elevations => "
                  << terrain_cursor << ", antenna_patterns => " << antenna_cursor
                  << ", rf_source_z_is_relative_to_terrain => " << rf_source_z_is_relative
                  << ", geographic_coords => " << geo_coords << ", bin_dim_meters => "
                  << rf_prop_params.bin_dim_meters
                  << ", assumed_receiver_height_agl => 0.0"
                  << ", max_ray_travel_meters => " << rf_prop_params.max_ray_travel_meters
                  << ", initial_rays_per_source => " << rf_prop_params.num_rays_per_source
                  << ", rays_per_bin_autosplit_threshold => " << 0
                  << ", min_receiver_signal_strength_dbm => "
                  << rf_prop_params.min_receiver_signal_strength_dbm
                  << ", default_source_height_agl_meters => "
                  << rf_prop_params.assumed_source_height_above_ground
                  << ", ray_step_bin_multiple => " << rf_prop_params.ray_step_bin_multiple
                  << ", loop_grain_size => " << rf_prop_params.loop_grain_size << ")) "
                  << order_by_clause << ";";
        break;
      }
    }
    return query_oss.str();
  }
  switch (rf_prop_function_type) {
    case RFPropFunctionType::RF_PROP_TOP_K: {
      query_oss << "SELECT * FROM TABLE(" << function_name << "(" << repeaters_cursor
                << ", " << rf_source_z_is_relative << ", "
                << rf_prop_params.repeater_signal_strength_dbm << ", "
                << rf_prop_params.repeater_frequency_mhz << ", " << terrain_cursor << ", "
                << geo_coords << ", " << rf_prop_params.bin_dim_meters << ", "
                << optional_top_k_sources << ", " << rf_prop_params.max_ray_travel_meters
                << ", " << rf_prop_params.num_rays_per_source << ", "
                << rf_prop_params.min_receiver_signal_strength_dbm << ", "
                << rf_prop_params.assumed_source_height_above_ground << ", "
                << rf_prop_params.ray_step_bin_multiple << ", "
                << rf_prop_params.loop_grain_size << ")) " << order_by_clause << ";";
      break;
    }
    case RFPropFunctionType::RF_PROP_MAX_SIGNAL: {
      query_oss << "SELECT * FROM TABLE(" << function_name << "(" << repeaters_cursor
                << ", " << rf_source_z_is_relative << ", "
                << rf_prop_params.repeater_signal_strength_dbm << ", "
                << rf_prop_params.repeater_frequency_mhz << ", " << terrain_cursor << ", "
                << geo_coords << ", " << rf_prop_params.bin_dim_meters << ", "
                << rf_prop_params.max_ray_travel_meters << ", "
                << rf_prop_params.num_rays_per_source << ", "
                << rf_prop_params.min_receiver_signal_strength_dbm << ", "
                << rf_prop_params.assumed_source_height_above_ground << ", "
                << rf_prop_params.ray_step_bin_multiple << ", "
                << rf_prop_params.loop_grain_size << ")) " << order_by_clause << ";";
      break;
    }
    case RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ: {
      query_oss << "SELECT * FROM TABLE(" << function_name << "(" << repeaters_cursor
                << ", " << terrain_cursor << ", " << rf_source_z_is_relative << ", "
                << geo_coords << ", " << rf_prop_params.bin_dim_meters << ", "
                << rf_prop_params.max_ray_travel_meters << ", "
                << rf_prop_params.num_rays_per_source << ", 0"
                << ", " << rf_prop_params.min_receiver_signal_strength_dbm << ", "
                << rf_prop_params.assumed_source_height_above_ground << ", "
                << rf_prop_params.ray_step_bin_multiple << ", "
                << rf_prop_params.loop_grain_size << ")) " << order_by_clause << ";";
      break;
    }
    case RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS:
    case RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION:
    case RFPropFunctionType::
        RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY: {
      query_oss << "SELECT * FROM TABLE(" << function_name << "(" << repeaters_cursor
                << ", " << terrain_cursor << ", " << antenna_cursor << ", "
                << rf_source_z_is_relative << ", " << geo_coords << ", "
                << rf_prop_params.bin_dim_meters << ", 0.0"
                << ", " << rf_prop_params.max_ray_travel_meters << ", "
                << rf_prop_params.num_rays_per_source << ", 0"
                << ", " << rf_prop_params.min_receiver_signal_strength_dbm << ", "
                << rf_prop_params.assumed_source_height_above_ground << ", "
                << rf_prop_params.ray_step_bin_multiple << ", "
                << rf_prop_params.loop_grain_size << ")) " << order_by_clause << ";";
      break;
    }
  }
  return query_oss.str();
}

std::vector<RepeaterInfo> enrich_repeaters(const std::shared_ptr<ResultSet>& results,
                                           const RFPropParams& rf_prop_params) {
  std::vector<RepeaterInfo> enriched_repeaters;
  for (const auto& input_repeater : rf_prop_params.repeaters) {
    RepeaterInfo repeater = input_repeater;
    repeater.repeater_x_bin = repeater.repeater_x / rf_prop_params.bin_dim_meters;
    repeater.repeater_y_bin = repeater.repeater_y / rf_prop_params.bin_dim_meters;
    if (repeater.repeater_x_bin < 0 ||
        repeater.repeater_x_bin >= static_cast<int32_t>(rf_prop_params.num_x_bins) ||
        repeater.repeater_y_bin < 0 ||
        repeater.repeater_y_bin >= static_cast<int32_t>(rf_prop_params.num_y_bins)) {
      // This repeater is off the grid, skip it
      continue;
    }
    if (rf_prop_params.repeater_height_is_relative) {
      const size_t repeater_bin_idx = x_y_bin_to_bin_index(
          repeater.repeater_x_bin, repeater.repeater_y_bin, rf_prop_params.num_x_bins);
      //    repeater.repeater_x_bin * rf_prop_params.num_y_bins + repeater.repeater_y_bin;
      CHECK_LT(repeater_bin_idx, rf_prop_params.num_x_bins * rf_prop_params.num_y_bins);
      repeater.repeater_z += TestHelpers::v<float>(
          results->getRowAt(repeater_bin_idx, static_cast<size_t>(2), false, false));
      results->moveToBegin();
    }
    enriched_repeaters.emplace_back(repeater);
  }
  return enriched_repeaters;
}

struct SignalInfo {
  double xy_dist = 0;
  double z_dist = 0;
  double xyz_dist = 0;
  double z_angle = 0;
  double calculated_signal_strength_dbm = 0;

  SignalInfo(const double x_delta,
             const double y_delta,
             const double z_delta,
             const double repeater_signal_strength_dbm,
             const double repeater_frequency_mhz) {
    xy_dist = sqrt(x_delta * x_delta + y_delta * y_delta);
    z_dist = z_delta;
    xyz_dist = sqrt(x_delta * x_delta + y_delta * y_delta + z_delta * z_delta);
    z_angle = atan2(z_dist, xy_dist) * 180.0 / M_PI;
    calculated_signal_strength_dbm = calc_receiver_power_dbm(
        repeater_signal_strength_dbm, repeater_frequency_mhz, xyz_dist);
  }
};

using RepeaterSignalInfo = std::pair<RepeaterInfo, SignalInfo>;

std::vector<RepeaterSignalInfo> get_top_k_sources_for_terrain_bin(
    const RFPropParams& rf_prop_params,
    const std::vector<RepeaterInfo>& enriched_repeaters,
    const int64_t top_k,
    const double grid_x,
    const double grid_y,
    const float grid_z) {
  std::vector<RepeaterSignalInfo> repeater_signal_infos;
  const int32_t grid_x_bin = grid_x / rf_prop_params.bin_dim_meters;
  const int32_t grid_y_bin = grid_y / rf_prop_params.bin_dim_meters;
  for (const auto& repeater : enriched_repeaters) {
    const double closest_x = grid_x_bin == repeater.repeater_x_bin ? grid_x
                             : (grid_x_bin > repeater.repeater_x_bin)
                                 ? std::floor(grid_x / rf_prop_params.bin_dim_meters) *
                                       rf_prop_params.bin_dim_meters
                                 : std::ceil(grid_x / rf_prop_params.bin_dim_meters) *
                                       rf_prop_params.bin_dim_meters;
    const double closest_y = grid_y_bin == repeater.repeater_y_bin ? grid_y
                             : (grid_y_bin > repeater.repeater_y_bin)
                                 ? std::floor(grid_y / rf_prop_params.bin_dim_meters) *
                                       rf_prop_params.bin_dim_meters
                                 : std::ceil(grid_y / rf_prop_params.bin_dim_meters) *
                                       rf_prop_params.bin_dim_meters;
    const double x_delta = repeater.repeater_x - closest_x;
    const double y_delta = repeater.repeater_y - closest_y;
    const double z_delta = repeater.repeater_z - grid_z;
    const SignalInfo signal_info(x_delta,
                                 y_delta,
                                 z_delta,
                                 rf_prop_params.repeater_signal_strength_dbm,
                                 rf_prop_params.repeater_frequency_mhz);
    repeater_signal_infos.emplace_back(std::make_pair(repeater, signal_info));
  }
  std::sort(repeater_signal_infos.begin(),
            repeater_signal_infos.end(),
            [](const RepeaterSignalInfo& a, const RepeaterSignalInfo& b) {
              return a.second.calculated_signal_strength_dbm >
                     b.second.calculated_signal_strength_dbm;
            });
  if (top_k >= 1) {
    const size_t capped_top_k =
        std::min(static_cast<size_t>(top_k), repeater_signal_infos.size());
    const std::vector top_k_repeater_signal_infos(
        repeater_signal_infos.begin(), repeater_signal_infos.begin() + capped_top_k);
    return top_k_repeater_signal_infos;
    // repeater_signal_infos.resize(static_cast<size_t>(top_k));
  }
  return repeater_signal_infos;
}

void validate_rf_prop_max_signal_results(const std::shared_ptr<ResultSet>& results,
                                         const RFPropParams& rf_prop_params,
                                         const double signal_strength_epsilon) {
  // rf_prop TFs signal strength can be slightly less than theoretical
  // calculated strength due to  the way the algorithm only computes
  // finite number of rays and bin steps, but should  never be
  // greater than theoretical max strength
  // TODO: This test framework doesn't account for dropoffs to lower elevation bins
  // obscuring part of the lower bin, causing us to currently be extra conservative
  // with the epsilon value for these cases

  const auto variant_elevation_bin_map =
      generate_variant_elevation_bin_map(rf_prop_params);
  const size_t num_rows = results->rowCount();
  ASSERT_EQ(
      num_rows,
      rf_prop_params.num_x_bins * rf_prop_params.num_y_bins);  // xXy output terrain bins
  ASSERT_EQ(results->colCount(),
            size_t(5));  // 5 output cols: x, y, max_z, strongest_rf_source_id,
                         // max_rf_signal_strength_dbm
  const auto enriched_repeaters = enrich_repeaters(results, rf_prop_params);

  constexpr double XYZ_EPS = 1.0e-2;
  const auto obscured_bin_set = generate_obscured_bin_set(rf_prop_params);
  for (size_t y_bin = 0; y_bin < rf_prop_params.num_y_bins; ++y_bin) {
    const double expected_y_val =
        y_bin * rf_prop_params.bin_dim_meters + (rf_prop_params.bin_dim_meters * 0.5);
    for (size_t x_bin = 0; x_bin < rf_prop_params.num_x_bins; ++x_bin) {
      const double expected_x_val =
          x_bin * rf_prop_params.bin_dim_meters + (rf_prop_params.bin_dim_meters * 0.5);
      auto crt_row = results->getNextRow(false, false);
      const double grid_x = TestHelpers::v<double>(crt_row[0]);
      const double grid_y = TestHelpers::v<double>(crt_row[1]);
      const double grid_z = TestHelpers::v<double>(crt_row[2]);
      const int64_t tf_strongest_repeater_id = TestHelpers::v<int64_t>(crt_row[3]);
      const double tf_signal_strength_dbm = TestHelpers::v<double>(crt_row[4]);
      ASSERT_NEAR(grid_x, expected_x_val, XYZ_EPS);
      ASSERT_NEAR(grid_y, expected_y_val, XYZ_EPS);
      const float expected_elevation =
          get_elevation_for_bin(rf_prop_params, variant_elevation_bin_map, x_bin, y_bin);
      ASSERT_NEAR(grid_z, expected_elevation, XYZ_EPS);
      if (is_bin_obscured(rf_prop_params, obscured_bin_set, x_bin, y_bin)) {
        ASSERT_EQ(tf_strongest_repeater_id,
                  inline_int_null_val(SQLTypeInfo(kBIGINT, false)));
        ASSERT_EQ(tf_signal_strength_dbm,
                  inline_fp_null_val(SQLTypeInfo(kDOUBLE, false)));
        continue;
      }
      const auto current_bin_top_k_sources = get_top_k_sources_for_terrain_bin(
          rf_prop_params, enriched_repeaters, 1, grid_x, grid_y, grid_z);
      CHECK_EQ(current_bin_top_k_sources.size(), static_cast<size_t>(1));
      const RepeaterInfo& correct_repeater = current_bin_top_k_sources[0].first;
      const SignalInfo& correct_signal_info = current_bin_top_k_sources[0].second;
      ASSERT_EQ(tf_strongest_repeater_id, correct_repeater.repeater_id);
      ASSERT_LE(tf_signal_strength_dbm,
                correct_signal_info.calculated_signal_strength_dbm);
      ASSERT_NEAR(tf_signal_strength_dbm,
                  correct_signal_info.calculated_signal_strength_dbm,
                  signal_strength_epsilon);
    }
  }
}

struct RFPropTopKRow {
  int32_t grid_cell_id;
  double grid_x;
  double grid_y;
  double grid_z;
  int32_t rf_source_id;
  double rf_signal_strength_dbm;
  double rf_signal_z_angle_degrees;
  double rf_source_distance_meters;

  RFPropTopKRow(const std::vector<TargetValue>& row) {
    grid_cell_id = TestHelpers::v<int64_t>(row[0]);
    grid_x = TestHelpers::v<double>(row[1]);
    grid_y = TestHelpers::v<double>(row[2]);
    grid_z = TestHelpers::v<double>(row[3]);
    rf_source_id = TestHelpers::v<int64_t>(row[4]);
    rf_signal_strength_dbm = TestHelpers::v<double>(row[5]);
    rf_signal_z_angle_degrees = TestHelpers::v<double>(row[6]);
    rf_source_distance_meters = TestHelpers::v<double>(row[7]);
  }
};

std::vector<RFPropTopKRow> read_rf_prop_top_k_results(
    const std::shared_ptr<ResultSet>& results) {
  const size_t num_rows = results->rowCount();
  std::vector<RFPropTopKRow> rf_prop_top_k_results;
  for (size_t r = 0; r < num_rows; ++r) {
    rf_prop_top_k_results.emplace_back(RFPropTopKRow(results->getNextRow(false, false)));
  }
  return rf_prop_top_k_results;
}

void validate_rf_prop_top_k_results(const std::shared_ptr<ResultSet>& results,
                                    const RFPropParams& rf_prop_params,
                                    const double signal_strength_epsilon) {
  // rf_prop TFs signal strength can be slightly less than theoretical
  // calculated strength due to  the way the algorithm only computes
  // finite number of rays and bin steps, but should  never be
  // greater than theoretical max strength
  // TODO: This test framework doesn't account for dropoffs to lower elevation bins
  // obscuring part of the lower bin, causing us to currently be extra conservative
  // with the epsilon value for these cases

  const auto variant_elevation_bin_map =
      generate_variant_elevation_bin_map(rf_prop_params);
  const size_t num_rows = results->rowCount();
  const size_t expected_num_rows =
      ((rf_prop_params.num_x_bins * rf_prop_params.num_y_bins) -
       rf_prop_params.obscured_bins.size()) *
      std::min(rf_prop_params.repeaters.size(),
               static_cast<size_t>(rf_prop_params.num_top_sources_per_terrain_bin));
  ASSERT_EQ(num_rows, expected_num_rows);

  ASSERT_EQ(results->colCount(),
            size_t(8));  // 8 output cols: x, y, max_z, rf_source_id,
                         // rf_signal_strength_dbm, rf_signal_z_angle_degrees,
                         // rf_source_distance_meters
  const auto enriched_repeaters = enrich_repeaters(results, rf_prop_params);

  constexpr double XYZ_EPS = 1.0e-2;
  const auto obscured_bin_set = generate_obscured_bin_set(rf_prop_params);
  const std::vector<RFPropTopKRow> rf_prop_top_k_results =
      read_rf_prop_top_k_results(results);
  size_t result_row_idx = 0;
  int32_t last_grid_cell_id = std::numeric_limits<int32_t>::lowest();
  std::vector<RepeaterSignalInfo> current_top_k_sources_for_terrain_bin;
  while (result_row_idx < num_rows) {
    const int32_t current_grid_cell_id =
        rf_prop_top_k_results[result_row_idx].grid_cell_id;
    ASSERT_GT(current_grid_cell_id, last_grid_cell_id);
    last_grid_cell_id = current_grid_cell_id;
    const size_t current_grid_cell_x_bin =
        current_grid_cell_id % rf_prop_params.num_x_bins;
    const size_t current_grid_cell_y_bin =
        current_grid_cell_id / rf_prop_params.num_x_bins;
    const double expected_x_val =
        current_grid_cell_x_bin * rf_prop_params.bin_dim_meters +
        (rf_prop_params.bin_dim_meters * 0.5);
    const double expected_y_val =
        current_grid_cell_y_bin * rf_prop_params.bin_dim_meters +
        (rf_prop_params.bin_dim_meters * 0.5);
    const float expected_elevation = get_elevation_for_bin(rf_prop_params,
                                                           variant_elevation_bin_map,
                                                           current_grid_cell_x_bin,
                                                           current_grid_cell_y_bin);
    const auto current_bin_top_k_sources =
        get_top_k_sources_for_terrain_bin(rf_prop_params,
                                          enriched_repeaters,
                                          rf_prop_params.num_top_sources_per_terrain_bin,
                                          expected_x_val,
                                          expected_y_val,
                                          expected_elevation);

    const size_t num_expected_bin_rows = current_bin_top_k_sources.size();
    ASSERT_LE(result_row_idx + num_expected_bin_rows, num_rows);
    size_t current_bin_top_k_source_idx = 0;
    const size_t current_bin_end_idx = result_row_idx + num_expected_bin_rows;
    for (; result_row_idx < current_bin_end_idx; ++result_row_idx) {
      const RFPropTopKRow& result_row = rf_prop_top_k_results[result_row_idx];
      ASSERT_EQ(result_row.grid_cell_id, current_grid_cell_id);
      ASSERT_NEAR(result_row.grid_x, expected_x_val, XYZ_EPS);
      ASSERT_NEAR(result_row.grid_y, expected_y_val, XYZ_EPS);
      ASSERT_NEAR(result_row.grid_z, expected_elevation, XYZ_EPS);
      if (is_bin_obscured(rf_prop_params,
                          obscured_bin_set,
                          current_grid_cell_x_bin,
                          current_grid_cell_y_bin)) {
        // tf_rf_prop generates sparse output, so we shouldn't have obscured bins
        CHECK(false);
        break;
      }
      const RepeaterInfo& correct_repeater =
          current_bin_top_k_sources[current_bin_top_k_source_idx].first;
      const SignalInfo& correct_signal_info =
          current_bin_top_k_sources[current_bin_top_k_source_idx].second;
      current_bin_top_k_source_idx++;
      ASSERT_EQ(result_row.rf_source_id, correct_repeater.repeater_id);
      ASSERT_NEAR(result_row.rf_signal_strength_dbm,
                  correct_signal_info.calculated_signal_strength_dbm,
                  signal_strength_epsilon);
      ASSERT_NEAR(result_row.rf_signal_z_angle_degrees, correct_signal_info.z_angle, 0.5);
      ASSERT_NEAR(result_row.rf_source_distance_meters,
                  correct_signal_info.xyz_dist,
                  rf_prop_params.bin_dim_meters * 0.2);
    }
  }
}

void run_rf_prop_max_signal_test(const RFPropParams& rf_prop_params,
                                 const double signal_strength_epsilon,
                                 const ExecutorDeviceType dt) {
  for (auto rf_prop_function_type :
       {RFPropFunctionType::RF_PROP_MAX_SIGNAL,
        RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ,
        RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS,
        RFPropFunctionType::RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION,
        RFPropFunctionType::
            RF_PROP_MAX_SIGNAL_VAR_POWER_FREQ_ANTENNA_PARAMS_ATTENUATION_ARRAY}) {
    for (bool use_named_args : {true, false}) {
      const auto query =
          generate_rf_prop_query(rf_prop_params, rf_prop_function_type, use_named_args);
      const auto results = run_multiple_agg(query, dt);
      validate_rf_prop_max_signal_results(
          results, rf_prop_params, signal_strength_epsilon);
    }
  }
}

void run_rf_prop_top_k_test(const RFPropParams& rf_prop_params,
                            const double signal_strength_epsilon,
                            const ExecutorDeviceType dt) {
  for (bool use_named_args : {true, false}) {
    const auto query = generate_rf_prop_query(
        rf_prop_params, RFPropFunctionType::RF_PROP_TOP_K, use_named_args);
    const auto results = run_multiple_agg(query, dt);
    validate_rf_prop_top_k_results(results, rf_prop_params, signal_strength_epsilon);
  }
}

enum ComparisonType {
  // Strong values means that at least one bin is GT for STRONG_GE or LT for STRONG_LE
  GT,
  STRONG_GE,
  GE,
  LE,
  STRONG_LE,
  LT
};

void compare_rf_prop_max_signal_dbm(const RFPropParams& rf_prop_params_1,
                                    const RFPropParams& rf_prop_params_2,
                                    const ComparisonType comparison_type,
                                    const ExecutorDeviceType dt) {
  for (bool use_named_args : {true, false}) {
    const auto rf_query_1 = generate_rf_prop_query(
        rf_prop_params_1, RFPropFunctionType::RF_PROP_MAX_SIGNAL, use_named_args);
    const auto rf_results_1 = run_multiple_agg(rf_query_1, dt);

    const auto rf_query_2 = generate_rf_prop_query(
        rf_prop_params_2, RFPropFunctionType::RF_PROP_MAX_SIGNAL, use_named_args);
    const auto rf_results_2 = run_multiple_agg(rf_query_2, dt);

    ASSERT_EQ(rf_results_1->rowCount(), rf_results_2->rowCount());
    ASSERT_EQ(rf_results_1->colCount(), size_t(5));
    ASSERT_EQ(rf_results_2->colCount(), size_t(5));

    const auto variant_elevation_bin_map_1 =
        generate_variant_elevation_bin_map(rf_prop_params_1);
    const auto variant_elevation_bin_map_2 =
        generate_variant_elevation_bin_map(rf_prop_params_2);
    const size_t num_rows = rf_results_1->rowCount();
    size_t non_equal_count = 0;
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
      auto results_1_row = rf_results_1->getNextRow(false, false);
      auto results_2_row = rf_results_2->getNextRow(false, false);
      ASSERT_EQ(TestHelpers::v<double>(results_1_row[0]),
                TestHelpers::v<double>(results_2_row[0]));  // x
      ASSERT_EQ(TestHelpers::v<double>(results_1_row[1]),
                TestHelpers::v<double>(results_2_row[1]));  // y

      const int32_t x_bin =
          TestHelpers::v<double>(results_1_row[0]) / rf_prop_params_1.bin_dim_meters;
      const int32_t y_bin =
          TestHelpers::v<double>(results_1_row[1]) / rf_prop_params_1.bin_dim_meters;
      const double expected_elevation_1 = get_elevation_for_bin(
          rf_prop_params_1, variant_elevation_bin_map_1, x_bin, y_bin);
      ASSERT_EQ(TestHelpers::v<double>(results_1_row[2]), expected_elevation_1);  // z
      const double expected_elevation_2 = get_elevation_for_bin(
          rf_prop_params_2, variant_elevation_bin_map_2, x_bin, y_bin);
      ASSERT_EQ(TestHelpers::v<double>(results_2_row[2]), expected_elevation_2);  // z

      switch (comparison_type) {
        case GT:
          ASSERT_GT(TestHelpers::v<double>(results_1_row[4]),
                    TestHelpers::v<double>(results_2_row[4]));  // signal_strength_dbm
          break;
        case STRONG_GE:
        case GE:
          if (comparison_type == ComparisonType::STRONG_GE &&
              TestHelpers::v<double>(results_1_row[4]) >
                  TestHelpers::v<double>(results_2_row[4])) {
            non_equal_count++;
          }
          ASSERT_GE(TestHelpers::v<double>(results_1_row[4]),
                    TestHelpers::v<double>(results_2_row[4]));  // signal_strength_dbm
          break;
        case LE:
        case STRONG_LE:
          if (comparison_type == ComparisonType::STRONG_LE &&
              TestHelpers::v<double>(results_1_row[4]) <
                  TestHelpers::v<double>(results_2_row[4])) {
            non_equal_count++;
          }
          ASSERT_LE(TestHelpers::v<double>(results_1_row[4]),
                    TestHelpers::v<double>(results_2_row[4]));  // signal_strength_dbm
          break;
        case LT:
          ASSERT_LT(TestHelpers::v<double>(results_1_row[4]),
                    TestHelpers::v<double>(results_2_row[4]));  // signal_strength_dbm
          break;
      }
    }
    // Here GE or LE means at least one value was GT/LT respectively
    if (comparison_type == ComparisonType::GE || comparison_type == ComparisonType::LE) {
      ASSERT_GT(non_equal_count, static_cast<size_t>(0));
    }
  }
}

RFPropParams generate_default_rf_prop_params() {
  const std::vector<VariantElevationBin> variant_elevation_bins = {};
  const std::vector<RepeaterInfo> repeaters = {
      {1, 25.0, 25.0, 20.0}};  // Note height will be relative to terrain
  const std::vector<TerrainBin> obscured_bins = {};
  return RFPropParams(10.0, /* bin_dim_meters */
                      5,    /* num_x_bins */
                      5,    /* num_y_bins */
                      10.0, /* primary_elevation */
                      variant_elevation_bins,
                      repeaters,
                      obscured_bins,
                      false, /* repeater_height_is_relative */
                      20.0,  /* repeater_signal_strength_dbm */
                      3900.0 /* repeater_frequency_mhz */);
}

// Disable Some/All of the tests in TSAN because of TBB (?) triggered failures
#ifndef HAVE_TSAN

// Each invalid-argument case below previously asserted that the validator
// throws std::runtime_error specifically. That typed assertion is fragile
// under test-interaction: when this test runs after a heavy prior test, the
// validator's error sometimes propagates wrapped (TException from a Thrift
// transport layer) instead of as itself, and the typed match fails. Confirmed
// locally: the test passes in isolation; only the full-suite ordering trips
// it. Use EXPECT_ANY_THROW — any throw here proves the bad input was
// rejected, which is the contract this test is actually defending.
TEST_F(RFPropTFs, RFPropMaxSignalLiteralBoundsChecking) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // tf_rf_prop_max_signal requires rf_source_signal_frequency to be > 0.0 ->
      // test arg = 0.0
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.repeater_frequency_mhz = 0.0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_MAX_SIGNAL, use_named_args);
        EXPECT_ANY_THROW(run_multiple_agg(query, dt));
      }
    }

    {
      // tf_rf_prop_max_signal requires rf_source_signal_frequency to be > 0.0 ->
      // test arg < 0.0
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.repeater_frequency_mhz = -10.0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_MAX_SIGNAL, use_named_args);
        EXPECT_ANY_THROW(run_multiple_agg(query, dt));
      }
    }

    {
      // tf_rf_prop_max_signal requires bin_dim_meters to be > 0.0
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.bin_dim_meters = 0.0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_MAX_SIGNAL, use_named_args);
        EXPECT_ANY_THROW(run_multiple_agg(query, dt));
      }
    }

    {
      // tf_rf_prop_max_signal requires assumed_source_height_above_ground >= 0.0
      // Note: argument only can be specified with long form of rf_prop_max_signal
      // function
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.assumed_source_height_above_ground = -1.0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_MAX_SIGNAL, use_named_args);
        EXPECT_ANY_THROW(run_multiple_agg(query, dt));
      }
    }

    {
      // tf_rf_prop_max_signal requires loop_grain_size >= 1
      // Note: argument only can be specified with long form of rf_prop_max_signal
      // function
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.loop_grain_size = 0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_MAX_SIGNAL, use_named_args);
        EXPECT_ANY_THROW(run_multiple_agg(query, dt));
      }
    }
  }
}

TEST_F(RFPropTFs, RFPropMaxSignalZeroRepeaters) {
  const double default_signal_strength_epsilon = 4.0e-1;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // Test absolute repeater height mode works
      // Terrain is 5X5 10m cells (x and y extents: 0 to 50m) with 10m elevation (flat)
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      // -1 id for repeater will be filtered out leaving an empty input cursor
      const std::vector<RepeaterInfo> repeaters = {{-1, 25.0, 25.0, 20.0}};
      const std::vector<TerrainBin> obscured_bins = {
          {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3},
          {1, 4}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {3, 0}, {3, 1}, {3, 2},
          {3, 3}, {3, 4}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }
  }
}

TEST_F(RFPropTFs, RFPropMaxSignalAllRepeatersOffGrid) {
  const double default_signal_strength_epsilon = 4.0e-1;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // Terrain is 5X5 10m cells (x and y extents: 0 to 50m) with 10m elevation (flat)
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      // All repeaters are off grid
      const std::vector<RepeaterInfo> repeaters = {{1, -25.0, 25.0, 20.0},
                                                   {2, 25.0, -25.0, 20.0}};
      const std::vector<TerrainBin> obscured_bins = {
          {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3},
          {1, 4}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {3, 0}, {3, 1}, {3, 2},
          {3, 3}, {3, 4}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }
  }
}

TEST_F(RFPropTFs, RFPropMaxSignalSingleRepeaterCorrectness) {
  const double default_signal_strength_epsilon = 4.0e-1;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // Test absolute repeater height mode works
      // Terrain is 5X5 10m cells (x and y extents: 0 to 50m) with 10m elevation (flat)
      // throughout Single repeater in center of terrain grid at 25m x, 25m y, with 20m
      // absolute height
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      const std::vector<RepeaterInfo> repeaters = {{1, 25.0, 25.0, 20.0}};
      const std::vector<TerrainBin> obscured_bins = {};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }

    {
      // Test repeater in one corner of larger grid
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      const std::vector<RepeaterInfo> repeaters = {{1, 15.0, 35.0, 50.0}};
      const std::vector<TerrainBin> obscured_bins = {};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        10,   /* num_x_bins */
                                        10,   /* num_y_bins */
                                        20.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }

    {
      // Test relative repeater height mode works
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      const std::vector<RepeaterInfo> repeaters = {
          {1, 25.0, 25.0, 20.0}};  // Note height will be relative to terrain
      const std::vector<TerrainBin> obscured_bins = {};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }

    {
      // Test relative repeater height mode works
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      const std::vector<RepeaterInfo> repeaters = {
          {1, 25.0, 25.0, 20.0}};  // Note height will be relative to terrain
      const std::vector<TerrainBin> obscured_bins = {};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }

    {
      // Test variant elevation bins
      const std::vector<VariantElevationBin> variant_elevation_bins = {{1, 1, 5.0},
                                                                       {1, 2, 2.0}};
      const std::vector<RepeaterInfo> repeaters = {
          {1, 25.0, 25.0, 20.0}};  // Note height will be relative to terrain
      const std::vector<TerrainBin> obscured_bins = {};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      // We must currently double the epsilon value since this test framework is not
      // currently able to account for a higher bin neighboring a lower bin partially
      // obscuring the latter, leading to overly-optimistic estimations of signal_strength
      // TODO: Add this capability so epsilons for these cases can be tightened
      run_rf_prop_max_signal_test(
          rf_prop_params, default_signal_strength_epsilon * 2, dt);
    }
  }
}

TEST_F(RFPropTFs, RFPropMaxSignalMultiRepeaterCorrectness) {
  const double default_signal_strength_epsilon = 4.0e-1;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    {
      // Test flat terrain with 2 repeaters at (0, 2) with height 20.0 and (4, 2) and
      // height 25.0 Second tower is placed higher to break tie for strongest repeater at
      // bins for x_bin=2
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      const std::vector<RepeaterInfo> repeaters = {{1, 5.0, 25.0, 20.0},
                                                   {2, 45.0, 25.0, 25.0}};
      const std::vector<TerrainBin> obscured_bins = {};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }

    {
      // Test flat terrain with 2 repeaters at (0, 0) with height 10.0 and (2, 2) with
      // height 100.0 First repeater should be the strongest source even at the base of
      // the second repeater due to being closer from an x/y distance

      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      const std::vector<RepeaterInfo> repeaters = {{1, 5.0, 5.0, 10.0},
                                                   {2, 25.0, 25.0, 100.0}};
      const std::vector<TerrainBin> obscured_bins = {};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        0.0,  /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }
  }
}

TEST_F(RFPropTFs, RFPropMaxSignalOutOfBoundsSource) {
  const double default_signal_strength_epsilon = 4.0e-1;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      // out of bounds repeater
      const std::vector<RepeaterInfo> repeaters = {
          {1, -10.0, 10.0, 20.0},
          {2, 60.0, 20.0, 20.0}};  // Note height will be relative to terrain
      // Treat all bins as obscured as none will have signal
      const std::vector<TerrainBin> obscured_bins = {
          {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3},
          {1, 4}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {3, 0}, {3, 1}, {3, 2},
          {3, 3}, {3, 4}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}};

      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(
          rf_prop_params, default_signal_strength_epsilon * 1.5, dt);
    }
  }
}

TEST_F(RFPropTFs, RFPropMaxSignalObscuredBins) {
  const double default_signal_strength_epsilon = 4.0e-1;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // Test obscured bins -> wall at x_bin=1
      // repeater at x_bin = 2, y_bin = 2
      // Bins with x_bin=0 should be obscured

      // X W * * *
      // X W * * *
      // X W R * *
      // X W * * *
      // X W * * *

      const std::vector<VariantElevationBin> variant_elevation_bins = {
          {1, 0, 100.0}, {1, 1, 100.0}, {1, 2, 100.0}, {1, 3, 100.0}, {1, 4, 100.0}};
      const std::vector<RepeaterInfo> repeaters = {
          {1, 25.0, 25.0, 20.0}};  // Note height will be relative to terrain
      const std::vector<TerrainBin> obscured_bins = {
          {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(
          rf_prop_params, default_signal_strength_epsilon * 1.5, dt);
    }

    {
      // Test obscured bins -> wall at y_bin=2, repeater at x_bin=2, y_bin=0
      // y_bin=3 should be obscured and y_bin=4 should be visible

      // * * * * *
      // X X X X X
      // W W W W W
      // * * * * *
      // * * R * *

      const std::vector<VariantElevationBin> variant_elevation_bins = {
          {0, 2, 20.0}, {1, 2, 20.0}, {2, 2, 20.0}, {3, 2, 20.0}, {4, 2, 20.0}};
      const std::vector<RepeaterInfo> repeaters = {
          {1, 25.0, 5.0, 50.0}};  // Note height will be relative to terrain
      const std::vector<TerrainBin> obscured_bins = {
          {0, 3}, {1, 3}, {2, 3}, {3, 3}, {4, 3}};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        0.0,  /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(
          rf_prop_params, default_signal_strength_epsilon * 1.5, dt);
    }

    {
      // Test obscured bins from one tower but visible from second tower
      // -> wall at y_bin=2, repeater 1 at x_bin=2, y_bin=0,
      // repeater 2 at x_bin=2, y_bin = 5
      // All bins should be visible

      // * * R * *
      // * * * * *
      // * * * * *
      // W W W W W
      // * * * * *
      // * * R * *

      const std::vector<VariantElevationBin> variant_elevation_bins = {
          {0, 2, 20.0}, {1, 2, 20.0}, {2, 2, 20.0}, {3, 2, 20.0}, {4, 2, 20.0}};
      const std::vector<RepeaterInfo> repeaters = {
          {1, 25.0, 5.0, 20.0},
          {2, 25.0, 55.0, 20.0}};  // Note height will be relative to terrain
      const std::vector<TerrainBin> obscured_bins = {};
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        5,    /* num_x_bins */
                                        6,    /* num_y_bins */
                                        0.0,  /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(
          rf_prop_params, default_signal_strength_epsilon * 1.5, dt);
    }

    {
      // Test obscured bins -> "pillar" at x_bin=2, y_bin=2, repeater at x_bin=0, y_bin=2

      // * * * * * *
      // * * * * * *
      // R * W X X X
      // * * * * * *
      // * * * * * *

      const std::vector<VariantElevationBin> variant_elevation_bins = {{2, 2, 100.0}};
      const std::vector<RepeaterInfo> repeaters = {
          {1, 5.0, 25.0, 20.0}};  // Note height will be relative to terrain
      const std::vector<TerrainBin> obscured_bins = {{3, 2}, {4, 2}, {5, 2}};
      //{4, 2}, {4, 3}, {4, 4}
      //{5, 2}, {5, 3}, {5, 4}
      const RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                        6,    /* num_x_bins */
                                        5,    /* num_y_bins */
                                        10.0, /* primary_elevation */
                                        variant_elevation_bins,
                                        repeaters,
                                        obscured_bins,
                                        false, /* repeater_height_is_relative */
                                        20.0,  /* repeater_signal_strength_dbm */
                                        3900.0 /* repeater_frequency_mhz */);
      run_rf_prop_max_signal_test(
          rf_prop_params, default_signal_strength_epsilon * 1.5, dt);
    }
  }
}

TEST_F(RFPropTFs, RFPropMaxRelativeSignalStrength) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // Test that received signal strength increases with source signal strength
      auto strong_rf_prop_params = generate_default_rf_prop_params();
      auto weak_rf_prop_params = generate_default_rf_prop_params();
      strong_rf_prop_params.repeater_signal_strength_dbm = 20.0;
      weak_rf_prop_params.repeater_signal_strength_dbm = 10.0;
      compare_rf_prop_max_signal_dbm(
          strong_rf_prop_params, weak_rf_prop_params, ComparisonType::GT, dt);
    }

    {
      // Test that received signal strength decreases with signal frequency
      auto strong_rf_prop_params = generate_default_rf_prop_params();
      auto weak_rf_prop_params = generate_default_rf_prop_params();
      strong_rf_prop_params.repeater_frequency_mhz = 1800.0;
      weak_rf_prop_params.repeater_frequency_mhz = 3900.0;
      compare_rf_prop_max_signal_dbm(
          strong_rf_prop_params, weak_rf_prop_params, ComparisonType::GT, dt);
    }

    {
      // Test that received signal strength decreases with relative z distance from
      // repeater
      auto strong_rf_prop_params = generate_default_rf_prop_params();
      auto weak_rf_prop_params = generate_default_rf_prop_params();
      strong_rf_prop_params.primary_elevation = 10.0;
      weak_rf_prop_params.primary_elevation = 5.0;
      compare_rf_prop_max_signal_dbm(
          strong_rf_prop_params, weak_rf_prop_params, ComparisonType::GT, dt);
    }

    {
      // Test that received signal strength decreases when z is relative if set repeater z
      // values are equal
      auto strong_rf_prop_params = generate_default_rf_prop_params();
      auto weak_rf_prop_params = generate_default_rf_prop_params();
      strong_rf_prop_params.repeater_height_is_relative = false;
      weak_rf_prop_params.repeater_height_is_relative = true;
      compare_rf_prop_max_signal_dbm(
          strong_rf_prop_params, weak_rf_prop_params, ComparisonType::GT, dt);
    }

    {
      // Test that received signal strength decreases when repeater z value increases
      auto strong_rf_prop_params = generate_default_rf_prop_params();
      auto weak_rf_prop_params = generate_default_rf_prop_params();
      strong_rf_prop_params.repeaters[0].repeater_z = 15.0;
      weak_rf_prop_params.repeaters[0].repeater_z = 20.0;
      compare_rf_prop_max_signal_dbm(
          strong_rf_prop_params, weak_rf_prop_params, ComparisonType::GT, dt);
    }

    {
      // Test that received signal strength decreases when repeater z value increases
      auto strong_rf_prop_params = generate_default_rf_prop_params();
      auto weak_rf_prop_params = generate_default_rf_prop_params();
      strong_rf_prop_params.repeaters[0].repeater_z = 15.0;
      weak_rf_prop_params.repeaters[0].repeater_z = 20.0;
      compare_rf_prop_max_signal_dbm(
          strong_rf_prop_params, weak_rf_prop_params, ComparisonType::GT, dt);
    }

    {
      // Test that received signal strength is equal or better for all bins with
      // additional repeater
      const RepeaterInfo additional_repeater = {1, 15.0, 15.0, 20.0};
      auto strong_rf_prop_params = generate_default_rf_prop_params();
      auto weak_rf_prop_params = generate_default_rf_prop_params();
      strong_rf_prop_params.repeaters.emplace_back(additional_repeater);
      compare_rf_prop_max_signal_dbm(
          strong_rf_prop_params, weak_rf_prop_params, ComparisonType::STRONG_GE, dt);
    }

    {
      // Test that received signal strength is equal or better for all bins in scenario
      // without lower elevation bins
      auto strong_rf_prop_params = generate_default_rf_prop_params();
      auto weak_rf_prop_params = generate_default_rf_prop_params();
      const std::vector<VariantElevationBin> variant_elevation_bins = {{1, 1, 5.0},
                                                                       {1, 2, 2.0}};
      weak_rf_prop_params.variant_elevation_bins = variant_elevation_bins;
      compare_rf_prop_max_signal_dbm(
          strong_rf_prop_params, weak_rf_prop_params, ComparisonType::STRONG_GE, dt);
    }
  }
}

TEST_F(RFPropTFs, RFPropTopKLiteralBoundsChecking) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // tf_rf_prop requires rf_source_signal_frequency to be > 0.0 ->
      // test arg = 0.0
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.repeater_frequency_mhz = 0.0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_TOP_K, use_named_args);
        EXPECT_THROW(run_multiple_agg(query, dt), std::runtime_error);
      }
    }

    {
      // tf_rf_prop requires rf_source_signal_frequency to be > 0.0 ->
      // test arg < 0.0
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.repeater_frequency_mhz = -10.0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_TOP_K, use_named_args);
        EXPECT_THROW(run_multiple_agg(query, dt), std::runtime_error);
      }
    }

    {
      // tf_rf_prop requires bin_dim_meters to be > 0.0
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.bin_dim_meters = 0.0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_TOP_K, use_named_args);
        EXPECT_THROW(run_multiple_agg(query, dt), std::runtime_error);
      }
    }

    {
      // tf_rf_prop requires assumed_source_height_above_ground >= 0.0
      // Note: argument only can be specified with long form of tf_rf_prop function
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.assumed_source_height_above_ground = -1.0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_TOP_K, use_named_args);
        EXPECT_THROW(run_multiple_agg(query, dt), std::runtime_error);
      }
    }

    {
      // tf_rf_prop requires loop_grain_size >= 1
      // Note: argument only can be specified with long form of tf_rf_prop function
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.loop_grain_size = 0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_TOP_K, use_named_args);
        EXPECT_THROW(run_multiple_agg(query, dt), std::runtime_error);
      }
    }

    {
      // tf_rf_prop allows negative or zero num_top_sources_per_terrain_bin to request
      // all results
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.num_top_sources_per_terrain_bin = 0;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_TOP_K, use_named_args);
        EXPECT_NO_THROW(run_multiple_agg(query, dt));
      }
    }

    {
      // tf_rf_prop allows negative or zero num_top_sources_per_terrain_bin to request
      // all results
      auto rf_prop_params = generate_default_rf_prop_params();
      rf_prop_params.num_top_sources_per_terrain_bin = -1;
      for (bool use_named_args : {true, false}) {
        const auto query = generate_rf_prop_query(
            rf_prop_params, RFPropFunctionType::RF_PROP_TOP_K, use_named_args);
        EXPECT_NO_THROW(run_multiple_agg(query, dt));
      }
    }
  }
}

TEST_F(RFPropTFs, RFPropTopKCorrectness) {
  const double default_signal_strength_epsilon = 4.0e-1;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      // Test absolute repeater height mode works
      // Terrain is 5X5 10m cells (x and y extents: 0 to 50m) with 10m elevation (flat)
      // throughout Single repeater in center of terrain grid at 25m x, 25m y, with 20m
      // absolute height
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      const std::vector<RepeaterInfo> repeaters = {{1, 25.0, 25.0, 20.0}};
      const std::vector<TerrainBin> obscured_bins = {};
      RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                  5,    /* num_x_bins */
                                  5,    /* num_y_bins */
                                  10.0, /* primary_elevation */
                                  variant_elevation_bins,
                                  repeaters,
                                  obscured_bins,
                                  false, /* repeater_height_is_relative */
                                  20.0,  /* repeater_signal_strength_dbm */
                                  3900.0 /* repeater_frequency_mhz */);
      rf_prop_params.num_top_sources_per_terrain_bin = 2;
      run_rf_prop_top_k_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }

    {
      // Test top k where k < n repeaters works
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      const std::vector<RepeaterInfo> repeaters = {
          {1, 25.0, 25.0, 20.0}, {2, 5.0, 5.0, 30.0}, {3, 25.0, 45.0, 40.0}};
      const std::vector<TerrainBin> obscured_bins = {};
      RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                  5,    /* num_x_bins */
                                  5,    /* num_y_bins */
                                  10.0, /* primary_elevation */
                                  variant_elevation_bins,
                                  repeaters,
                                  obscured_bins,
                                  false, /* repeater_height_is_relative */
                                  20.0,  /* repeater_signal_strength_dbm */
                                  3900.0 /* repeater_frequency_mhz */);
      rf_prop_params.num_top_sources_per_terrain_bin = 2;
      run_rf_prop_top_k_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }
    {
      // Test top k where k = -1 to get all repeaters works
      const std::vector<VariantElevationBin> variant_elevation_bins = {};
      const std::vector<RepeaterInfo> repeaters = {
          {1, 25.0, 25.0, 20.0}, {2, 5.0, 5.0, 30.0}, {3, 25.0, 45.0, 40.0}};
      const std::vector<TerrainBin> obscured_bins = {};
      RFPropParams rf_prop_params(10.0, /* bin_dim_meters */
                                  5,    /* num_x_bins */
                                  5,    /* num_y_bins */
                                  10.0, /* primary_elevation */
                                  variant_elevation_bins,
                                  repeaters,
                                  obscured_bins,
                                  false, /* repeater_height_is_relative */
                                  20.0,  /* repeater_signal_strength_dbm */
                                  3900.0 /* repeater_frequency_mhz */);
      rf_prop_params.num_top_sources_per_terrain_bin = -1;
      run_rf_prop_top_k_test(rf_prop_params, default_signal_strength_epsilon, dt);
    }
  }
}

TEST_F(RFPropTFs, RecyclingTableFunctionsResultset) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  auto clearCache = [&executor] {
    executor->clearMemory(MemoryLevel::CPU_LEVEL);
    executor->getQueryPlanDagCache().clearQueryPlanCache();
  };
  clearCache();

  ScopeGuard reset_global_flag_state =
      [orig_resulset_recycler = g_use_query_resultset_cache,
       orig_data_recycler = g_enable_data_recycler,
       orig_chunk_metadata_recycler = g_use_chunk_metadata_cache] {
        g_use_query_resultset_cache = orig_resulset_recycler;
        g_enable_data_recycler = orig_data_recycler;
        g_use_chunk_metadata_cache = orig_chunk_metadata_recycler;
      };
  g_enable_data_recycler = true;
  g_use_query_resultset_cache = true;
  g_use_chunk_metadata_cache = true;

  // use `RFPropMaxSignalSingleRepeaterCorrectness` test queries
  // to check the correctness with exploiting resultset recycler
  std::set<QueryPlanHash> visited_hashtable_key;

  // currently, query plan DAG extractor does not available to extract a dag for a query
  // having logical value node
  auto drop_tables = []() {
    run_ddl_statement("DROP TABLE IF EXISTS R_C_1;");
    run_ddl_statement("DROP TABLE IF EXISTS R_C_2;");
    run_ddl_statement("DROP TABLE IF EXISTS T_C_1;");
    run_ddl_statement("DROP TABLE IF EXISTS T_C_2;");
  };

  auto create_tables = []() {
    run_ddl_statement("CREATE TABLE R_C_1 (id INT, x DOUBLE, y DOUBLE, z DOUBLE);");
    run_ddl_statement("CREATE TABLE R_C_2 (id INT, x DOUBLE, y DOUBLE, z DOUBLE);");
    run_ddl_statement("CREATE TABLE T_C_1 (x DOUBLE, y DOUBLE, z DOUBLE);");
    run_ddl_statement("CREATE TABLE T_C_2 (x DOUBLE, y DOUBLE, z DOUBLE);");
    run_multiple_agg("INSERT INTO R_C_1 VALUES (1, 25.000000, 25.000000, 20.000000);",
                     ExecutorDeviceType::CPU);
    run_multiple_agg("INSERT INTO R_C_2 VALUES (1, 15.000000, 35.000000, 50.000000);",
                     ExecutorDeviceType::CPU);
  };

  auto create_terrain_cursor_data =
      [](const RFPropParams& rf_prop_params) -> std::vector<std::string> {
    std::vector<std::string> row_data;
    const auto variant_elevation_bin_map =
        generate_variant_elevation_bin_map(rf_prop_params);
    for (size_t y_bin = 0; y_bin < rf_prop_params.num_y_bins; ++y_bin) {
      const double y_val = y_bin * rf_prop_params.bin_dim_meters;
      for (size_t x_bin = 0; x_bin < rf_prop_params.num_x_bins; ++x_bin) {
        const double x_val = x_bin * rf_prop_params.bin_dim_meters;
        const float bin_elevation = get_elevation_for_bin(
            rf_prop_params, variant_elevation_bin_map, x_bin, y_bin);
        auto data = std::to_string(x_val) + ", " + std::to_string(y_val) + ", " +
                    std::to_string(bin_elevation);
        row_data.push_back(data);
      }
    }
    return row_data;
  };

  drop_tables();
  create_tables();

  // generate data
  const std::vector<VariantElevationBin> variant_elevation_bins1 = {};
  const std::vector<RepeaterInfo> repeaters1 = {{1, 25.0, 25.0, 20.0}};
  const std::vector<TerrainBin> obscured_bins1 = {};
  const RFPropParams rf_prop_params1(10.0, /* bin_dim_meters */
                                     5,    /* num_x_bins */
                                     5,    /* num_y_bins */
                                     10.0, /* primary_elevation */
                                     variant_elevation_bins1,
                                     repeaters1,
                                     obscured_bins1,
                                     false, /* repeater_height_is_relative */
                                     20.0,  /* repeater_signal_strength_dbm */
                                     3900.0 /* repeater_frequency_mhz */);
  auto row_data1 = create_terrain_cursor_data(rf_prop_params1);
  for (const auto& row : row_data1) {
    run_multiple_agg("INSERT INTO T_C_1 VALUES (" + row + ");", ExecutorDeviceType::CPU);
  }

  const std::vector<VariantElevationBin> variant_elevation_bins2 = {};
  const std::vector<RepeaterInfo> repeaters2 = {{1, 15.0, 35.0, 50.0}};
  const std::vector<TerrainBin> obscured_bins2 = {};
  const RFPropParams rf_prop_params2(10.0, /* bin_dim_meters */
                                     10,   /* num_x_bins */
                                     10,   /* num_y_bins */
                                     20.0, /* primary_elevation */
                                     variant_elevation_bins2,
                                     repeaters2,
                                     obscured_bins2,
                                     false, /* repeater_height_is_relative */
                                     20.0,  /* repeater_signal_strength_dbm */
                                     3900.0 /* repeater_frequency_mhz */);
  auto row_data2 = create_terrain_cursor_data(rf_prop_params2);
  for (const auto& row : row_data2) {
    run_multiple_agg("INSERT INTO T_C_2 VALUES (" + row + ");", ExecutorDeviceType::CPU);
  }

  const double default_signal_strength_epsilon = 4.0e-1;
  auto perform_tests = [&default_signal_strength_epsilon,
                        &rf_prop_params1,
                        &rf_prop_params2](bool keep_hint, ExecutorDeviceType dt) {
    std::string head =
        keep_hint ? "SELECT /*+ keep_table_function_result */ * " : "SELECT * ";
    auto q1 =
        head +
        "FROM TABLE(tf_rf_prop_max_signal(CURSOR(SELECT CAST(id AS INTEGER) AS id, "
        "CAST(x AS DOUBLE) AS x, CAST(y AS DOUBLE) AS y, CAST(z AS FLOAT) AS z FROM "
        "R_C_1), false, 20.000000, 3900.000000, CURSOR(SELECT CAST(x AS DOUBLE) AS x, "
        "CAST(y AS DOUBLE) AS y, CAST(z AS FLOAT) AS z FROM T_C_1), false, 10.000000, "
        "2000.000000, 1800, -80.000000, 10.000000, 0.010000,1)) ORDER BY y ASC, x ASC;";
    auto q2 =
        head +
        "FROM TABLE(tf_rf_prop_max_signal(CURSOR(SELECT CAST(id AS INTEGER) AS id, "
        "CAST(x AS DOUBLE) AS x, CAST(y AS DOUBLE) AS y, CAST(z AS FLOAT) AS z FROM "
        "R_C_2), false, 20.000000, 3900.000000, CURSOR(SELECT CAST(x AS DOUBLE) AS x, "
        "CAST(y AS DOUBLE) AS y, CAST(z AS FLOAT) AS z FROM T_C_2), false, 10.000000, "
        "2000.000000, 1800, -80.000000, 10.000000, 0.010000,1)) ORDER BY y ASC, x ASC;";
    auto res1 = run_multiple_agg(q1, dt);
    validate_rf_prop_max_signal_results(
        res1, rf_prop_params1, default_signal_strength_epsilon);

    auto res2 = run_multiple_agg(q2, dt);
    validate_rf_prop_max_signal_results(
        res2, rf_prop_params2, default_signal_strength_epsilon);
  };

  // first, keep the table function's resultset
  perform_tests(true, ExecutorDeviceType::CPU);
  auto& recycler_holder = executor->getResultSetRecyclerHolder();
  auto resultset_recycler = recycler_holder.getResultSetRecycler();
  CHECK(resultset_recycler);
  auto num_cached_resultset = resultset_recycler->getCurrentNumCachedItems(
      CacheItemType::QUERY_RESULTSET, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  // results has been cached...
  EXPECT_GT(num_cached_resultset, static_cast<size_t>(0));
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // just exploit the cached resultset
    perform_tests(false, dt);
  }
  for (size_t i = num_cached_resultset; i > 0; --i) {
    auto cached_resultset_info = resultset_recycler->getCachedResultSetWithoutCacheKey(
        visited_hashtable_key, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
    auto resultset = std::get<1>(cached_resultset_info);
    auto cache_key = std::get<0>(cached_resultset_info);
    visited_hashtable_key.insert(cache_key);
    CHECK(resultset);
    auto cache_metric =
        resultset_recycler->getCachedItemMetric(CacheItemType::QUERY_RESULTSET,
                                                DataRecyclerUtil::CPU_DEVICE_IDENTIFIER,
                                                cache_key);
    CHECK(cache_metric);
    // results has been recycled during the test
    EXPECT_GT(cache_metric->getRefCount(), static_cast<size_t>(0));
  }
  drop_tables();
}

#endif  // HAVE_TSAN

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  // Table function support must be enabled before initialized the query runner
  // environment
  g_enable_table_functions = true;
  g_enable_rf_prop_table_functions = true;
  QR::init(BASE_PATH);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}

#endif  // HAVE_RF_PROP_TFS
