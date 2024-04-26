/*
 * Copyright 2024 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
