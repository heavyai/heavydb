/*
 * Copyright 2017 MapD Technologies, Inc.
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

/*
 * File:   parameters.h
 * Author: michael
 *
 * Created on January 19, 2017, 4:31 PM
 */

#ifndef MAPDPARAMETERS_H
#define MAPDPARAMETERS_H

#include <string>

struct MapDParameters {
  size_t cuda_block_size = 0;       // block size for the kernel execution
  size_t cuda_grid_size = 0;        // grid size for the kernel execution
  size_t calcite_max_mem = 1024;    // max memory for calcite jvm in MB
  int mapd_server_port = 9091;      // default port mapd_server runs on
  int calcite_port = 9093;          // default port for calcite server to run on
  std::string ha_group_id;          // name of the HA group this server is in
  std::string ha_unique_server_id;  // name of the HA unique id for this server
  std::string ha_brokers;           // name of the HA broker
  std::string ha_shared_data;       // name of shared data directory base
  bool is_decr_start_epoch;         // are we doing a start epoch decrement?

  MapDParameters() : cuda_block_size(0), cuda_grid_size(0), calcite_max_mem(1024) {}
};

#endif /* MAPDPARAMETERS_H */
