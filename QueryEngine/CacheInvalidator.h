/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CACHEINVALIDATOR_H
#define CACHEINVALIDATOR_H

template <typename... CACHE_HOLDING_TYPES>
class CacheInvalidator {
 public:
  static void invalidateCaches() { (..., CACHE_HOLDING_TYPES::invalidateCache()); }
  static void invalidateCachesByTable(size_t table_key) {
    // input: a hashed table chunk key: {db_id, table_id}
    (..., CACHE_HOLDING_TYPES::markCachedItemAsDirty(table_key));
  }

 private:
  CacheInvalidator() = delete;
  ~CacheInvalidator() = delete;
};

#endif
