/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    BufferEntryUtils.h
 * @brief   Utility functions for group by buffer entries
 *
 */

#ifndef QUERYENGINE_BUFFERENTRYUTILS_H
#define QUERYENGINE_BUFFERENTRYUTILS_H

#include "../Shared/funcannotations.h"
#include "GpuRtConstants.h"

#ifdef FORCE_CPU_VERSION
#undef DEVICE
#define DEVICE
#undef INLINE
#define INLINE inline
#else
#define INLINE
#endif

namespace {

template <class K>
INLINE DEVICE bool is_empty_entry(const size_t entry_idx,
                                  const int8_t* groupby_buffer,
                                  const size_t key_stride);

template <>
INLINE DEVICE bool is_empty_entry<int32_t>(const size_t entry_idx,
                                           const int8_t* groupby_buffer,
                                           const size_t key_stride) {
  const auto key_ptr = groupby_buffer + entry_idx * key_stride;
  return (*reinterpret_cast<const int32_t*>(key_ptr) == EMPTY_KEY_32);
}

template <>
INLINE DEVICE bool is_empty_entry<int64_t>(const size_t entry_idx,
                                           const int8_t* groupby_buffer,
                                           const size_t key_stride) {
  const auto key_ptr = groupby_buffer + entry_idx * key_stride;
  return (*reinterpret_cast<const int64_t*>(key_ptr) == EMPTY_KEY_64);
}

}  // namespace

#undef INLINE
#undef DEVICE

#endif  // QUERYENGINE_BUFFERENTRYUTILS_H
