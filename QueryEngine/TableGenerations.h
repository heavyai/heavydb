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

#ifndef QUERYENGINE_TABLEGENERATIONS_H
#define QUERYENGINE_TABLEGENERATIONS_H

#include <unordered_map>

struct TableGeneration {
  size_t tuple_count;
  size_t start_rowid;
};

class TableGenerations {
 public:
  void setGeneration(const uint32_t id, const TableGeneration& generation);

  const TableGeneration& getGeneration(const uint32_t id) const;

  const std::unordered_map<uint32_t, TableGeneration>& asMap() const;

  void clear();

 private:
  std::unordered_map<uint32_t, TableGeneration> id_to_generation_;
};

#endif  // QUERYENGINE_TABLEGENERATIONS_H
