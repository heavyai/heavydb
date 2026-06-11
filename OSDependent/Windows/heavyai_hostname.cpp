/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OSDependent/heavyai_hostname.h"

#include "Shared/clean_windows.h"

namespace heavyai {
std::string get_hostname() {
  static constexpr DWORD kSize = MAX_COMPUTERNAME_LENGTH + 1;
  DWORD buffer_size = kSize;
  char hostname[MAX_COMPUTERNAME_LENGTH + 1];
  if (GetComputerNameA(hostname, &buffer_size)) {
    return {hostname};
  } else {
    return {};
  }
}
}  // namespace heavyai
