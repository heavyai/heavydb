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
 * @file		StringNoneEncoder.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		For unencoded strings
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include "StringNoneEncoder.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include "MemoryLevel.h"

using Data_Namespace::AbstractBuffer;

size_t StringNoneEncoder::getNumElemsForBytesInsertData(
    const std::vector<std::string>* srcData,
    const int start_idx,
    const size_t numAppendElems,
    const size_t byteLimit,
    const bool replicating) {
  size_t dataSize = 0;
  size_t n = start_idx;
  for (; n < start_idx + numAppendElems; n++) {
    size_t len = (*srcData)[replicating ? 0 : n].length();
    if (dataSize + len > byteLimit) {
      break;
    }
    dataSize += len;
  }
  return n - start_idx;
}

void StringNoneEncoder::updateStats(const std::vector<std::string>* const src_data,
                                    const size_t start_idx,
                                    const size_t num_elements) {
  for (size_t n = start_idx; n < start_idx + num_elements; n++) {
    update_elem_stats((*src_data)[n]);
    if (has_nulls) {
      break;
    }
  }
}

void StringNoneEncoder::update_elem_stats(const std::string& elem) {
  if (!has_nulls && elem.empty()) {
    has_nulls = true;
  }
}
