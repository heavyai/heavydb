/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef UPDATECACHEINVALIDATORS_H
#define UPDATECACHEINVALIDATORS_H

#include "CacheInvalidator.h"

/**
 * External cache invalidators clear caches not managed by the Buffer Manager (i.e.
 * external to the buffer manager).
 */

// Classes that are involved in needing a cache invalidated
#include "JoinHashTable/BaselineJoinHashTable.h"
#include "JoinHashTable/BoundingBoxIntersectJoinHashTable.h"
#include "JoinHashTable/PerfectJoinHashTable.h"
#include "ResultSetRecyclerHolder.h"

using UpdateTriggeredCacheInvalidator =
    CacheInvalidator<BoundingBoxIntersectJoinHashTable,
                     BaselineJoinHashTable,
                     PerfectJoinHashTable>;
using DeleteTriggeredCacheInvalidator = UpdateTriggeredCacheInvalidator;

// Note that this is functionally the same as the above two invalidators. The
// JoinHashTableCacheInvalidator is a generic invalidator used during `clear_cpu` calls.
// The above cache invalidators are specific invalidators called during update/delete and
// will likely be extended in the future.
using JoinHashTableCacheInvalidator = CacheInvalidator<BoundingBoxIntersectJoinHashTable,
                                                       BaselineJoinHashTable,
                                                       PerfectJoinHashTable>;
using ResultSetCacheInvalidator = CacheInvalidator<ResultSetRecyclerHolder>;

#endif
