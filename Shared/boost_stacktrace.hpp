/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
