/**
 * @file		ScanTable.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Scan through each column of a table via Chunk iterators
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef SCAN_TABLE_H
#define SCAN_TABLE_H

#include <cstdlib>
#include <string>
#include <vector>
#include "../Catalog/Catalog.h"

std::vector<size_t>
scan_table_return_hash(const std::string &table_name, const Catalog_Namespace::Catalog &cat);
std::vector<size_t>
scan_table_return_hash_non_iter(const std::string &table_name, const Catalog_Namespace::Catalog &cat);

#endif // SCAN_TABLE_H
