/*
 *  Some cool MapD License
 */

/*
 * File:   parameters.h
 * Author: michael
 *
 * Created on January 19, 2017, 4:31 PM
 */

#ifndef MAPDPARAMETERS_H
#define MAPDPARAMETERS_H

struct MapDParameters {
  size_t cuda_block_size = 0;     // block size for the kernel execution
  size_t cuda_grid_size = 0;      // grid size for the kernel execution
  size_t calcite_max_mem = 1024;  // max memory for calcite jvm in MB

  MapDParameters() : cuda_block_size(0), cuda_grid_size(0), calcite_max_mem(1024) {}
};

#endif /* MAPDPARAMETERS_H */
