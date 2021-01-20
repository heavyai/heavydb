/*
 * Copyright 2020 OmniSci, Inc.
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

#ifndef BOOST_STACKTRACE_H
#define BOOST_STACKTRACE_H

#ifdef _WIN32
#include <windows.h>

// boost includes dbgeng.h which uses IN and OUT macroses
// in structure declaration. For some reason those macroses
// can be undefined (not clear if it is SDK or our bug, rpcdce.h defines them only),
// so we define these macroses here to fix the problem.
#ifndef IN
#define IN
#define OUT
#define UNDEF_IN_OUT
#endif

#endif  // _WIN32

#include <boost/stacktrace.hpp>

#ifdef _WIN32

#ifdef UNDEF_IN_OUT
#undef IN
#undef OUT
#endif  // UNDEF_IN_OUT

#include "cleanup_global_namespace.h"
#endif  // _WIN32

#endif  // BOOST_STACKTRACE_H
