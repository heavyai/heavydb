/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "L0Exception.h"

#define L0_SAFE_CALL(call)           \
  {                                  \
    auto status = (call);            \
    if (status) {                    \
      throw l0::L0Exception(status); \
    }                                \
  }
