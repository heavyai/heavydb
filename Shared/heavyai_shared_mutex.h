/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include <mutex>
#include <shared_mutex>

#ifdef HAVE_FOLLY
#include <folly/SharedMutex.h>
namespace heavyai {
using shared_mutex = folly::SharedMutex;
}  // namespace heavyai
// Folly includes windows.h and pollutes global namespace with macroses
#include "cleanup_global_namespace.h"
#else
namespace heavyai {
using shared_mutex = std::shared_timed_mutex;
}  // namespace heavyai
#endif  // HAVE_FOLLY

namespace heavyai {
template <typename T>
using lock_guard = std::lock_guard<T>;
template <typename T>
using unique_lock = std::unique_lock<T>;
template <typename T>
using shared_lock = std::shared_lock<T>;
}  // namespace heavyai
