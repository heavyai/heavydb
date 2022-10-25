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

#include "InputDescriptors.h"

#include <boost/functional/hash.hpp>

InputSourceType InputDescriptor::getSourceType() const {
  return table_key_.table_id > 0 ? InputSourceType::TABLE : InputSourceType::RESULT;
}

size_t InputDescriptor::hash() const {
  auto hash = table_key_.hash();
  boost::hash_combine(hash, nest_level_);
  return hash;
}

std::string InputDescriptor::toString() const {
  return ::typeName(this) + "(db_id=" + std::to_string(table_key_.db_id) +
         ", table_id=" + std::to_string(table_key_.table_id) +
         ", nest_level=" + std::to_string(nest_level_) + ")";
}
