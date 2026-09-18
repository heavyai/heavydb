/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
