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

#ifndef UPDATECACHEINVALIDATORS_H
#define UPDATECACHEINVALIDATORS_H

#include "CacheInvalidator.h"

/**
 * External cache invalidators clear caches not managed by the Buffer Manager (i.e.
 * external to the buffer manager).
 */

// Classes that are involved in needing a cache invalidated
#include "JoinHashTable/BaselineJoinHashTable.h"
#include "JoinHashTable/JoinHashTable.h"
#include "JoinHashTable/OverlapsJoinHashTable.h"

using UpdateTriggeredCacheInvalidator =
    CacheInvalidator<OverlapsJoinHashTable, BaselineJoinHashTable, JoinHashTable>;
using DeleteTriggeredCacheInvalidator = UpdateTriggeredCacheInvalidator;

// Note that this is functionally the same as the above two invalidators. The
// JoinHashTableCacheInvalidator is a generic invalidator used during `clear_cpu` calls.
// The above cache invalidators are specific invalidators called during update/delete and
// will likely be extended in the future.
using JoinHashTableCacheInvalidator =
    CacheInvalidator<OverlapsJoinHashTable, BaselineJoinHashTable, JoinHashTable>;

#endif
