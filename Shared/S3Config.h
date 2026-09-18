/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file S3Config.h
 * @brief S3Config struct
 *
 */

#pragma once

#include <string>

namespace shared {
struct S3Config {
  std::string access_key;  // per-query credentials to override the
  std::string secret_key;  // settings in ~/.aws/credentials or environment
  std::string session_token;
  std::string region;
  std::string endpoint;
  bool use_virtual_addressing = true;
  int32_t max_concurrent_downloads =
      8;  // maximum number of concurrent file downloads from S3
};
}  // namespace shared
