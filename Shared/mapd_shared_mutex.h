/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef MAPD_SHARED_MUTEX
#define MAPD_SHARED_MUTEX

#include <shared_mutex>

#ifdef HAVE_FOLLY
#include <folly/SharedMutex.h>
using mapd_shared_mutex = folly::SharedMutex;
// Folly includes windows.h and pollutes global namespace with macroses
#include "cleanup_global_namespace.h"
#else
using mapd_shared_mutex = std::shared_timed_mutex;
#endif  // HAVE_FOLLY

#define mapd_lock_guard std::lock_guard
#define mapd_unique_lock std::unique_lock
#define mapd_shared_lock std::shared_lock

#endif  // MAPD_SHARED_MUTEX
