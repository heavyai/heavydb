/*
 * Copyright 2022 OmniSci, Inc.
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

#include <algorithm>

struct ThreadInfo {
  int64_t num_threads{0};
  int64_t num_elems_per_thread;

  ThreadInfo(const int64_t max_thread_count,
             const int64_t num_elems,
             const int64_t target_elems_per_thread) {
    num_threads =
        std::min(std::max(max_thread_count, int64_t(1)),
                 ((num_elems + target_elems_per_thread - 1) / target_elems_per_thread));
    num_elems_per_thread =
        std::max((num_elems + num_threads - 1) / num_threads, int64_t(1));
  }
};
