/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thrift/TConfiguration.h>

#include <limits>
#include <memory>

namespace shared {

inline std::shared_ptr<apache::thrift::TConfiguration> default_tconfig() {
  return std::make_shared<apache::thrift::TConfiguration>(
      std::numeric_limits<int32_t>::max());
}

}  // namespace shared
