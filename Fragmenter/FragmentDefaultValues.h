/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#define DEFAULT_FRAGMENT_ROWS 32000000ULL     // in tuples
#define DEFAULT_PAGE_SIZE 2097152ULL          // in bytes
#define DEFAULT_METADATA_PAGE_SIZE 4096ULL    // in bytes
#define DEFAULT_MAX_ROWS ((1LL) << 62)        // in rows
#define DEFAULT_MAX_CHUNK_SIZE 2147483648ULL  // in bytes 2G
