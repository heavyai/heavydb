/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @file		PopulateTableRandom.h
 * @brief		Populate a table with random data
 *
 */

#ifndef POPULATE_TABLE_RANDOM_H
#define POPULATE_TABLE_RANDOM_H

#include <cstdlib>
#include <string>
#include <vector>
#include "../Catalog/Catalog.h"

std::vector<size_t> populate_table_random(const std::string& table_name,
                                          const size_t num_rows,
                                          const Catalog_Namespace::Catalog& cat);

#endif  // POPULATE_TABLE_RANDOM_H
