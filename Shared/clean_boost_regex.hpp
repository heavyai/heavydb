/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// If you need to include boost/regex.hpp, please include this file instead
// so that the windows.h it includes is cleaned-up after. Otherwise macros
// like GetObject are defined that interfere with rapidjson, etc.
#include <boost/regex.hpp>
#include "cleanup_global_namespace.h"
