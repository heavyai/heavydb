/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_map>

#include "Shared/DbObjectKeys.h"

class StringDictionaryGenerations {
 public:
  StringDictionaryGenerations(){};

  void setGeneration(const shared::StringDictKey& dict_key, const uint64_t generation);

  void updateGeneration(const shared::StringDictKey& dict_key, const uint64_t generation);

  int64_t getGeneration(const shared::StringDictKey& dict_key) const;

  const std::unordered_map<shared::StringDictKey, uint64_t>& asMap() const;

  void clear();

 private:
  std::unordered_map<shared::StringDictKey, uint64_t> dict_key_to_generation_;
};
