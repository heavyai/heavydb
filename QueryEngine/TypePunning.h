/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_TYPEPUNNING_H
#define QUERYENGINE_TYPEPUNNING_H

#include "../Shared/funcannotations.h"

// Mark ptr as safe for type-punning operations. We need it whenever we want to
// interpret a sequence of bytes as float / double through a reinterpret_cast.

#ifdef _MSC_VER
template <class T>
FORCE_INLINE T* may_alias_ptr(T* ptr) {
  return ptr;
}
#else
template <class T>
    FORCE_INLINE T __attribute__((__may_alias__)) * may_alias_ptr(T* ptr) {
  return ptr;
}
#endif

#endif  // QUERYENGINE_TYPEPUNNING_H
