/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    InPlaceSortImpl.h
 * @brief
 *
 */

#ifndef INPLACESORTIMPL_H
#define INPLACESORTIMPL_H

#include <cstdint>

#if HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

class ThrustAllocator;

void sort_on_gpu(int64_t* val_buff,
                 int32_t* key_buff,
                 const uint64_t entry_count,
                 const bool desc,
                 const uint32_t chosen_bytes,
                 ThrustAllocator& alloc,
                 CUstream cuda_stream);

void sort_on_cpu(int64_t* val_buff,
                 int32_t* key_buff,
                 const uint64_t entry_count,
                 const bool desc,
                 const uint32_t chosen_bytes);

void apply_permutation_on_gpu(int64_t* val_buff,
                              int32_t* idx_buff,
                              const uint64_t entry_count,
                              const uint32_t chosen_bytes,
                              ThrustAllocator& alloc,
                              CUstream cuda_stream);

void apply_permutation_on_cpu(int64_t* val_buff,
                              int32_t* idx_buff,
                              const uint64_t entry_count,
                              int64_t* tmp_buff,
                              const uint32_t chosen_bytes);

#endif  // INPLACESORTIMPL_H
