/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/heavyai_hostname.h"

#include <unistd.h>
#include <climits>

namespace heavyai {
std::string get_hostname() {
  char hostname[_POSIX_HOST_NAME_MAX];
  gethostname(hostname, _POSIX_HOST_NAME_MAX);
  return {hostname};
}
}  // namespace heavyai
