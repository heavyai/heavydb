/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TableGenerations.h"
#include "Logger/Logger.h"

void TableGenerations::setGeneration(const shared::TableKey& table_key,
                                     const TableGeneration& generation) {
  const auto it_ok = table_key_to_generation_.emplace(table_key, generation);
  CHECK(it_ok.second);
}

const TableGeneration& TableGenerations::getGeneration(
    const shared::TableKey& table_key) const {
  const auto it = table_key_to_generation_.find(table_key);
  CHECK(it != table_key_to_generation_.end());
  return it->second;
}

const std::unordered_map<shared::TableKey, TableGeneration>& TableGenerations::asMap()
    const {
  return table_key_to_generation_;
}

void TableGenerations::clear() {
  decltype(table_key_to_generation_)().swap(table_key_to_generation_);
}
