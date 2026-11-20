/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Copyright 2026 HEAVY.AI, Inc.
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

#pragma once

#include "Shared/sqltypes.h"

#include <cstdint>

#if HAVE_CUDA
#include <cuda.h>
#else
#include "Shared/nocuda.h"
#endif

class ThrustAllocator;

enum class GpuWindowRankKind { Rank, DenseRank };
enum class GpuWindowExtremaKind { Min, Max };

bool compute_window_rank_on_gpu(const int32_t* payload,
                                const int32_t* offsets,
                                const int32_t* counts,
                                const int64_t elem_count,
                                const int32_t partition_count,
                                const int32_t max_partition_count,
                                const int8_t* order_col,
                                const SQLTypes order_type,
                                const int order_type_size,
                                const bool order_col_nullable,
                                const int64_t order_null_pattern,
                                const bool desc,
                                const bool nulls_first,
                                const GpuWindowRankKind rank_kind,
                                int64_t* output,
                                ThrustAllocator& allocator,
                                CUstream cuda_stream);

bool compute_window_partition_extrema_on_gpu(const int32_t* payload,
                                             const int32_t* offsets,
                                             const int32_t* counts,
                                             const int64_t elem_count,
                                             const int32_t partition_count,
                                             const int8_t* input_col,
                                             const SQLTypes input_type,
                                             const int input_type_size,
                                             const bool input_col_nullable,
                                             const int64_t input_null_pattern,
                                             const GpuWindowExtremaKind extrema_kind,
                                             int8_t* value_output,
                                             int64_t* row_position_output,
                                             CUstream cuda_stream);
