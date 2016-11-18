/*
 * @file    JoinHashImpl.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_GROUPBYFASTIMPL_H
#define QUERYENGINE_GROUPBYFASTIMPL_H

#include "../Shared/funcannotations.h"
#include <stdint.h>

extern "C" ALWAYS_INLINE DEVICE int32_t* SUFFIX(get_hash_slot)(int32_t* buff,
                                                               const int64_t key,
                                                               const int64_t min_key) {
  return buff + (key - min_key);
}

#endif  // QUERYENGINE_GROUPBYFASTIMPL_H
