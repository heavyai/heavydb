/*
 * @file    QueryPhysicalInputsCollector.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Find out all the physical inputs (columns) a query is using.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_QUERYPHYSICALINPUTSCOLLECTOR_H
#define QUERYENGINE_QUERYPHYSICALINPUTSCOLLECTOR_H

#include <unordered_set>

class RelAlgNode;

struct PhysicalInput {
  int col_id;
  int table_id;

  bool operator==(const PhysicalInput& that) const { return col_id == that.col_id && table_id == that.table_id; }
};

namespace std {

template <>
struct hash<PhysicalInput> {
  size_t operator()(const PhysicalInput& phys_input) const { return phys_input.col_id ^ phys_input.table_id; }
};

}  // std

std::unordered_set<PhysicalInput> get_physical_inputs(const RelAlgNode*);
std::unordered_set<int> get_physical_table_inputs(const RelAlgNode*);

#endif  // QUERYENGINE_QUERYPHYSICALINPUTSCOLLECTOR_H
