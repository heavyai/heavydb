/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
