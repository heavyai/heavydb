/* copyright 2017 MapD Technologies, Inc.
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
#ifndef QUERYENGINE_TOPKSORT_H
#define QUERYENGINE_TOPKSORT_H

#ifdef HAVE_CUDA
#include "ResultSetSortImpl.h"
#include "../Shared/sqltypes.h"
#include <cuda.h>

#include <vector>

namespace Data_Namespace {

class DataMgr;

}  // Data_Namespace

class ThrustAllocator;

std::vector<int8_t> pop_n_rows_from_merged_heaps_gpu(Data_Namespace::DataMgr* data_mgr,
                                                     const int64_t* dev_heaps,
                                                     const size_t heaps_size,
                                                     const size_t n,
                                                     const PodOrderEntry& oe,
                                                     const GroupByBufferLayoutInfo& layout,
                                                     const size_t group_key_bytes,
                                                     const size_t thread_count,
                                                     const int device_id);

#endif  // HAVE_CUDA

#endif  // QUERYENGINE_TOPKSORT_H
