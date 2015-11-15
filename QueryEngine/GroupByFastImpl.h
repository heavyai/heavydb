/*
 * @file    GroupByFastImpl.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_GROUPBYFASTIMPL_H
#define QUERYENGINE_GROUPBYFASTIMPL_H

#include "../Shared/funcannotations.h"
#include <stdint.h>

extern "C" ALWAYS_INLINE DEVICE int64_t* SUFFIX(get_group_value_fast_impl)(int64_t* groups_buffer,
                                                                           const int64_t key,
                                                                           const int64_t min_key,
                                                                           const uint32_t agg_col_count,
                                                                           const bool read_only) {
  int64_t off = (key - min_key) * (1 + agg_col_count);
  if (!read_only && groups_buffer[off] == EMPTY_KEY) {
    groups_buffer[off] = key;
  }
  return groups_buffer + off + 1;
}

extern "C" ALWAYS_INLINE DEVICE int64_t* SUFFIX(get_group_value_fast)(int64_t* groups_buffer,
                                                                      const int64_t key,
                                                                      const int64_t min_key,
                                                                      const uint32_t agg_col_count) {
  return SUFFIX(get_group_value_fast_impl)(groups_buffer, key, min_key, agg_col_count, false);
}

extern "C" ALWAYS_INLINE DEVICE int64_t* SUFFIX(get_hash_slot)(int64_t* groups_buffer,
                                                               const int64_t key,
                                                               const int64_t min_key,
                                                               const uint32_t agg_col_count) {
  return SUFFIX(get_group_value_fast_impl)(groups_buffer, key, min_key, agg_col_count, true);
}

#endif  // QUERYENGINE_GROUPBYFASTIMPL_H
