/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "L0Exception.h"

namespace l0 {
L0Exception::L0Exception(L0result status) : status_(status) {}

const char* L0Exception::what() const noexcept {
  // avoid clang unused private member warning
  // marking status_ directly triggers a gcc attribute warning
  [[maybe_unused]] int foo = status_;
  return "L0 is not enabled";
}
}  // namespace l0