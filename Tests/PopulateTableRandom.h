/**
 * @file		PopulateTableRandom.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Populate a table with random data
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef POPULATE_TABLE_RANDOM_H
#define POPULATE_TABLE_RANDOM_H

#include <string>
#include <vector>
#include <cstdlib>
#include "../Catalog/Catalog.h"

std::vector<size_t> populate_table_random(const std::string& table_name,
                                          const size_t num_rows,
                                          const Catalog_Namespace::Catalog& cat);

#endif  // POPULATE_TABLE_RANDOM_H
