/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <unordered_map>

#include "Shared/DbObjectKeys.h"

struct TableGeneration {
  int64_t tuple_count;
  int64_t start_rowid;
};

class TableGenerations {
 public:
  void setGeneration(const shared::TableKey& table_key,
                     const TableGeneration& generation);

  const TableGeneration& getGeneration(const shared::TableKey& table_key) const;

  const std::unordered_map<shared::TableKey, TableGeneration>& asMap() const;

  void clear();

 private:
  std::unordered_map<shared::TableKey, TableGeneration> table_key_to_generation_;
};
