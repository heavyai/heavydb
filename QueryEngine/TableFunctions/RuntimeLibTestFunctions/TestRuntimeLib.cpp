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

#include "TestRuntimeLib.h"

template <typename T>
T _test_runtime_add(T x, T y) {
  return x + y;
}

template <typename T>
T _test_runtime_sub(T x, T y) {
  return x - y;
}

template int64_t _test_runtime_add(int64_t, int64_t);
template double _test_runtime_add(double, double);
template int64_t _test_runtime_sub(int64_t, int64_t);
template double _test_runtime_sub(double, double);