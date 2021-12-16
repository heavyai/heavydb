/*
 * Copyright 2019 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
