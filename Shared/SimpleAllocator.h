/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

class SimpleAllocator {
 public:
  virtual ~SimpleAllocator() = default;
  virtual int8_t* allocate(const size_t num_bytes) = 0;
};
