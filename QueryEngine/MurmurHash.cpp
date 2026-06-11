/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/MurmurHash.h"
#include "QueryEngine/MurmurHash1Inl.h"
#include "QueryEngine/MurmurHash3Inl.h"

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE uint32_t MurmurHash1(const void* key,
                                                                   int len,
                                                                   const uint32_t seed) {
  return MurmurHash1Impl(key, len, seed);
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE uint64_t MurmurHash64A(const void* key,
                                                                     int len,
                                                                     uint64_t seed) {
  return MurmurHash64AImpl(key, len, seed);
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE uint32_t MurmurHash3(const void* key,
                                                                   int len,
                                                                   const uint32_t seed) {
  return MurmurHash3Impl(key, len, seed);
}
