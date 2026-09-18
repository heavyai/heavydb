/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef __CUDACC__

#include <QueryEngine/heavydbTypes.h>
#include <vector>

template <typename T1, typename T2>
void set_antenna_array(const std::vector<T1>& data,
                       Column<Array<T2>>& output_arr,
                       const int64_t index) {
  const int64_t arr_size = static_cast<int64_t>(data.size());
  auto arr = output_arr.getItem(index, arr_size);
  for (int64_t arr_idx = 0; arr_idx < arr_size; ++arr_idx) {
    arr[arr_idx] = data[arr_idx];
  }
}

// clang-format off
/*
  UDTF: tf_transmitter_coalesce__cpu_template(TableFunctionManager,
  Cursor<Column<S> site, Column<S> id, Column<T> x, Column<T> y, Column<Z> z, Column<T> tx_power,
  Column<T> tx_freq, Column<T> azimuth, Column<T> downtilt, Column<TextEncodingDict> antenna_type> transmitters) ->
  Column<S> site | input_id=args<0>, Column<Array<S>> id | input_id=args<1>, Column<T> x, Column<T> y, Column<Z> z, Column<Array<T>> tx_power,
  Column<Array<T>> tx_freq, Column<Array<T>> azimuth, Column<Array<T>> downtilt,
  Column<Array<TextEncodingDict>> antenna_type | input_id=args<9>, S=[int64_t, TextEncodingDict], 
  T=[double], Z=[double]
*/
// clang-format on

template <typename S, typename T, typename Z>
TEMPLATE_NOINLINE int32_t tf_transmitter_coalesce__cpu_template(
    TableFunctionManager& mgr,
    const Column<S>& input_site,
    const Column<S>& input_id,
    const Column<T>& input_x,
    const Column<T>& input_y,
    const Column<Z>& input_z,
    const Column<T>& input_power,
    const Column<T>& input_freq,
    const Column<T>& input_azimuth,
    const Column<T>& input_downtilt,
    const Column<TextEncodingDict>& input_antenna_type,
    Column<S>& output_site,
    Column<Array<S>>& output_id,
    Column<T>& output_x,
    Column<T>& output_y,
    Column<Z>& output_z,
    Column<Array<T>>& output_power,
    Column<Array<T>>& output_freq,
    Column<Array<T>>& output_azimuth,
    Column<Array<T>>& output_downtilt,
    Column<Array<TextEncodingDict>>& output_antenna_type) {
  struct AntennaParams {
    std::vector<S> ids;
    double x;
    double y;
    double z;
    std::vector<double> powers;
    std::vector<double> freqs;
    std::vector<double> azimuths;
    std::vector<double> downtilts;
    std::vector<TextEncodingDict> antenna_types;
    int64_t num_transmitters;

    AntennaParams(const S id,
                  const double x,
                  const double y,
                  const double z,
                  const double power,
                  const double freq,
                  const double azimuth,
                  const double downtilt,
                  const TextEncodingDict antenna_type)
        : x(x), y(y), z(z), num_transmitters(1) {
      ids.emplace_back(id);
      powers.emplace_back(power);
      freqs.emplace_back(freq);
      azimuths.emplace_back(azimuth);
      downtilts.emplace_back(downtilt);
      antenna_types.emplace_back(antenna_type);
    }
  };

  std::map<S, AntennaParams> transmitter_map;

  const int64_t num_inputs = input_id.size();
  int64_t num_outputs = 0;
  std::vector<int64_t> permuted_idxs(num_inputs);
  for (int64_t idx = 0; idx < num_inputs; ++idx) {
    permuted_idxs[idx] = idx;
  }
  std::sort(permuted_idxs.begin(),
            permuted_idxs.begin() + num_inputs,
            [&](const int64_t& a, const int64_t& b) {
              return input_azimuth[a] < input_azimuth[b];
            });

  for (int64_t idx = 0; idx < num_inputs; ++idx) {
    const auto input_idx = permuted_idxs[idx];

    const auto site = input_site[input_idx];
    const auto map_itr = transmitter_map.find(site);
    if (map_itr != transmitter_map.end()) {
      map_itr->second.ids.emplace_back(input_id[input_idx]);
      map_itr->second.x += input_x[input_idx];
      map_itr->second.y += input_y[input_idx];
      map_itr->second.z += input_z[input_idx];
      map_itr->second.powers.emplace_back(input_power[input_idx]);
      map_itr->second.freqs.emplace_back(input_freq[input_idx]);
      map_itr->second.azimuths.emplace_back(input_azimuth[input_idx]);
      map_itr->second.downtilts.emplace_back(input_downtilt[input_idx]);
      map_itr->second.antenna_types.emplace_back(input_antenna_type[input_idx]);
      map_itr->second.num_transmitters++;
    } else {
      ++num_outputs;

      transmitter_map.emplace(
          site,
          AntennaParams(input_id[input_idx],
                        static_cast<double>(input_x[input_idx]),
                        static_cast<double>(input_y[input_idx]),
                        static_cast<double>(input_z[input_idx]),
                        static_cast<double>(input_power[input_idx]),
                        static_cast<double>(input_freq[input_idx]),
                        static_cast<double>(input_azimuth[input_idx]),
                        static_cast<double>(input_downtilt[input_idx]),
                        input_antenna_type[input_idx]));
    }
  }
  mgr.set_output_array_values_total_number(1, num_inputs);
  mgr.set_output_array_values_total_number(5, num_inputs);
  mgr.set_output_array_values_total_number(6, num_inputs);
  mgr.set_output_array_values_total_number(7, num_inputs);
  mgr.set_output_array_values_total_number(8, num_inputs);
  mgr.set_output_array_values_total_number(9, num_inputs);
  mgr.set_output_row_size(num_outputs);

  int64_t output_idx = 0;

  for (const auto& rf_site : transmitter_map) {
    output_site[output_idx] = rf_site.first;
    set_antenna_array(rf_site.second.ids, output_id, output_idx);
    output_x[output_idx] = rf_site.second.x / rf_site.second.num_transmitters;
    output_y[output_idx] = rf_site.second.y / rf_site.second.num_transmitters;
    output_z[output_idx] = rf_site.second.z / rf_site.second.num_transmitters;
    set_antenna_array(rf_site.second.powers, output_power, output_idx);
    set_antenna_array(rf_site.second.freqs, output_freq, output_idx);
    set_antenna_array(rf_site.second.azimuths, output_azimuth, output_idx);
    set_antenna_array(rf_site.second.downtilts, output_downtilt, output_idx);
    set_antenna_array(rf_site.second.antenna_types, output_antenna_type, output_idx);
    ++output_idx;
  }

  return num_outputs;
}

#endif  // __CUDACC__