/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
