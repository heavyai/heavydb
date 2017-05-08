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

#ifndef SHARED_UNREACHABLE_H
#define SHARED_UNREACHABLE_H

#ifndef __CUDACC__
#include <glog/logging.h>

#define UNREACHABLE() CHECK(false)
#else
#define UNREACHABLE() abort()
#endif  // __CUDACC__

#endif  // SHARED_UNREACHABLE_H
