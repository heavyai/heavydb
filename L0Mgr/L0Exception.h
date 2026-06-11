/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <exception>

namespace l0 {
using L0result = int;

class L0Exception : public std::exception {
 public:
  L0Exception(L0result status);

  const char* what() const noexcept override;

 private:
  L0result const status_;
};
}  // namespace l0