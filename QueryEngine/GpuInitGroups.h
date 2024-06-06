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
 * @file    GpuInitGroups.h
 * @brief
 *
 */

#ifndef GPUINITGROUPS_H
#define GPUINITGROUPS_H
#include <cstdint>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

void init_group_by_buffer_on_device(int64_t* groups_buffer,
                                    const int64_t* init_vals,
                                    const uint32_t groups_buffer_entry_count,
                                    const uint32_t key_count,
                                    const uint32_t key_width,
                                    const uint32_t agg_col_count,
                                    const bool keyless,
                                    const int8_t warp_size,
                                    const size_t block_size_x,
                                    const size_t grid_size_x,
                                    CUstream cuda_stream);

void init_columnar_group_by_buffer_on_device(int64_t* groups_buffer,
                                             const int64_t* init_vals,
                                             const uint32_t groups_buffer_entry_count,
                                             const uint32_t key_count,
                                             const uint32_t agg_col_count,
                                             const int8_t* col_sizes,
                                             const bool need_padding,
                                             const bool keyless,
                                             const int8_t key_size,
                                             const size_t block_size_x,
                                             const size_t grid_size_x,
                                             CUstream cuda_stream);

#endif  // GPUINITGROUPS_H
