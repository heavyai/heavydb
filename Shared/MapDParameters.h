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
  std::string ha_group_id;          // name of the HA group this server is in
  std::string ha_unique_server_id;  // name of the HA unique id for this server
  std::string ha_brokers;           // name of the HA broker
  std::string ha_shared_data;       // name of shared data directory base

  MapDParameters() : cuda_block_size(0), cuda_grid_size(0), calcite_max_mem(1024) {}
};

#endif /* MAPDPARAMETERS_H */
