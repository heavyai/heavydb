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
 * @file    InPlaceSort.h
 * @brief
 *
 */

#ifndef INPLACESORT_H
#define INPLACESORT_H

#include "GpuMemUtils.h"

#include <cstdint>
#include <list>

namespace Analyzer {
struct OrderEntry;
}
class QueryMemoryDescriptor;
namespace Data_Namespace {
class DataMgr;
}

void inplace_sort_gpu(const std::list<Analyzer::OrderEntry>&,
                      const QueryMemoryDescriptor&,
                      const GpuGroupByBuffers&,
                      Data_Namespace::DataMgr*,
                      const int device_id);

void sort_groups_cpu(int64_t* val_buff,
                     int32_t* key_buff,
                     const uint64_t entry_count,
                     const bool desc,
                     const uint32_t chosen_bytes);

void apply_permutation_cpu(int64_t* val_buff,
                           int32_t* idx_buff,
                           const uint64_t entry_count,
                           int64_t* tmp_buff,
                           const uint32_t chosen_bytes);

#endif  // INPLACESORT_H
