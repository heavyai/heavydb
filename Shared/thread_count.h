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

#ifndef THREAD_COUNT_H
#define THREAD_COUNT_H

#include <unistd.h>
#include <algorithm>

extern std::size_t g_cpu_threads_override;

inline int cpu_threads() {
  // could use std::thread::hardware_concurrency(), but some
  // slightly out-of-date compilers (gcc 4.7) implement it as always 0.
  // Play it POSIX.1 safe instead.
  auto ov = g_cpu_threads_override;
  return (ov <= 0) ? std::max(2 * sysconf(_SC_NPROCESSORS_CONF), 1L) : ov;
}

#endif  // THREAD_COUNT_H
