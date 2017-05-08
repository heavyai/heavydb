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

std::vector<size_t> scan_table_return_hash(const std::string& table_name, const Catalog_Namespace::Catalog& cat);
std::vector<size_t> scan_table_return_hash_non_iter(const std::string& table_name,
                                                    const Catalog_Namespace::Catalog& cat);

#endif  // SCAN_TABLE_H
