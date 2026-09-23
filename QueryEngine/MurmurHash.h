/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_MURMURHASH_H
#define QUERYENGINE_MURMURHASH_H

#include <cstdint>
#include "../Shared/funcannotations.h"

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE RUNTIME_EXPORT uint32_t
MurmurHash1(const void* key, int len, const uint32_t seed);

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE RUNTIME_EXPORT uint64_t
MurmurHash64A(const void* key, int len, uint64_t seed);

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE RUNTIME_EXPORT uint32_t
MurmurHash3(const void* key, int len, const uint32_t seed);

#endif  // QUERYENGINE_MURMURHASH_H
