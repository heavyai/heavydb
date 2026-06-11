/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef __CUDACC__
#include <ostream>
#endif

enum class ExecutorDeviceType { CPU, GPU };

#ifndef __CUDACC__
std::ostream& operator<<(std::ostream& os, const ExecutorDeviceType device_type);
std::string toString(const ExecutorDeviceType device_type);
#endif
