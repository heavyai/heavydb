#ifndef UPDATECACHEINVALIDATORS_H
#define UPDATECACHEINVALIDATORS_H

#include "CacheInvalidator.h"

// Classes that are involved in needing a cache invalidated when there is an update
#include "BaselineJoinHashTable.h"
#include "JoinHashTable.h"

using UpdateTriggeredCacheInvalidator =
    CacheInvalidator<BaselineJoinHashTable, JoinHashTable>;
using DeleteTriggeredCacheInvalidator = UpdateTriggeredCacheInvalidator;

#endif
