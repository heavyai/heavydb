/*
 * @file    InPlaceSortImpl.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef INPLACESORTIMPL_H
#define INPLACESORTIMPL_H

#include <stdint.h>

class ThrustAllocator;

void sort_on_gpu(int64_t* val_buff,
                 int32_t* key_buff,
                 const uint64_t entry_count,
                 const bool desc,
                 const uint32_t chosen_bytes,
                 ThrustAllocator& alloc);

void sort_on_cpu(int64_t* val_buff,
                 int32_t* key_buff,
                 const uint64_t entry_count,
                 const bool desc,
                 const uint32_t chosen_bytes);

void apply_permutation_on_gpu(int64_t* val_buff,
                              int32_t* idx_buff,
                              const uint64_t entry_count,
                              const uint32_t chosen_bytes,
                              ThrustAllocator& alloc);

void apply_permutation_on_cpu(int64_t* val_buff,
                              int32_t* idx_buff,
                              const uint64_t entry_count,
                              int64_t* tmp_buff,
                              const uint32_t chosen_bytes);

#endif  // INPLACESORTIMPL_H
