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
 * @file    Types.h
 * @brief   Catch-all for publicly accessible types utilized in various Query Engine
 * Descriptors
 *
 */

#pragma once

#include <ostream>
#include <sstream>

enum class QueryDescriptionType {
  GroupByPerfectHash,
  GroupByBaselineHash,
  Projection,
  TableFunction,
  NonGroupedAggregate,
  Estimator
};

inline std::ostream& operator<<(std::ostream& os, const QueryDescriptionType& type) {
  switch (type) {
    case QueryDescriptionType::GroupByPerfectHash:
      os << "GroupByPerfectHash";
      break;
    case QueryDescriptionType::GroupByBaselineHash:
      os << "GroupByBaselineHash";
      break;
    case QueryDescriptionType::Projection:
      os << "Projection";
      break;
    case QueryDescriptionType::TableFunction:
      os << "TableFunction";
      break;
    case QueryDescriptionType::NonGroupedAggregate:
      os << "NonGroupedAggregate";
      break;
    case QueryDescriptionType::Estimator:
      os << "Estimator";
      break;
    default:
      os << "Unknown QueryDescriptionType";
  }
  return os;
}

inline std::string toString(const QueryDescriptionType& type) {
  std::ostringstream ss;
  ss << type;
  return ss.str();
}
