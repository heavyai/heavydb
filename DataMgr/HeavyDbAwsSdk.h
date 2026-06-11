/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace heavydb_aws_sdk {
struct SslConfig {
  std::string ca_path;
  std::string ca_file;
};

void init_sdk();
void shutdown_sdk();
SslConfig get_ssl_config();
};  // namespace heavydb_aws_sdk
