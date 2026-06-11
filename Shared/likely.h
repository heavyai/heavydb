/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SHARED_LIKELY
#define SHARED_LIKELY

#ifdef _MSC_VER
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#else
#define LIKELY(x) (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))
#endif
#endif  // SHARED_LIKELY
