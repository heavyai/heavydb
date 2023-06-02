/*
 * Copyright 2023 HEAVY.AI, Inc.
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

/* This library is built as a stand-alone shared library to test our
runtime dynamically loaded library support. The library is loaded at
runtime using boost::shared_library (which is essentially a wrapper
around dlopen()). Then, we register table functions that use functions
defined in this library at runtime, allowing for table functions to
use code from optional "plugin" libraries. */

#include <cstdint>

template <typename T>
T _test_runtime_add(T x, T y);
template <typename T>
T _test_runtime_sub(T x, T y);