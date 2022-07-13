/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#if defined(HAVE_AWS_S3)
#include <map>
#include <optional>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <aws/core/Globals.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>

extern bool g_allow_s3_server_privileges;

namespace foreign_storage {
class ParquetS3DetectFileSystem {
 public:
  struct ParquetS3DetectFileSystemConfiguration {
    std::optional<std::string> s3_access_key = std::nullopt;
    std::optional<std::string> s3_secret_key = std::nullopt;
    std::optional<std::string> s3_session_token = std::nullopt;
    std::string s3_region;
  };

  static std::shared_ptr<arrow::fs::FileSystem> create(
      const ParquetS3DetectFileSystemConfiguration& config) {
    auto s3_options = arrow::fs::S3Options::Anonymous();
    if (config.s3_access_key.has_value() || config.s3_secret_key.has_value() ||
        config.s3_session_token.has_value()) {
      if (config.s3_access_key.has_value() && config.s3_secret_key.has_value()) {
        if (config.s3_session_token.has_value()) {
          s3_options =
              arrow::fs::S3Options::FromAccessKey(config.s3_access_key.value(),
                                                  config.s3_secret_key.value(),
                                                  config.s3_session_token.value());
        } else {
          s3_options = arrow::fs::S3Options::FromAccessKey(config.s3_access_key.value(),
                                                           config.s3_secret_key.value());
        }
      }
    } else if (g_allow_s3_server_privileges) {
      Aws::Auth::DefaultAWSCredentialsProviderChain default_provider;
      if ((default_provider.GetAWSCredentials().GetAWSAccessKeyId().size() > 0) &&
          (default_provider.GetAWSCredentials().GetAWSSecretKey().size() > 0)) {
        if (default_provider.GetAWSCredentials().GetSessionToken().size() > 0) {
          s3_options = arrow::fs::S3Options::FromAccessKey(
              default_provider.GetAWSCredentials().GetAWSAccessKeyId(),
              default_provider.GetAWSCredentials().GetAWSSecretKey(),
              default_provider.GetAWSCredentials().GetSessionToken());
        } else {
          s3_options = arrow::fs::S3Options::FromAccessKey(
              default_provider.GetAWSCredentials().GetAWSAccessKeyId(),
              default_provider.GetAWSCredentials().GetAWSSecretKey());
        }
      }
    }
    s3_options.region = config.s3_region;
    auto fs_result = arrow::fs::S3FileSystem::Make(s3_options);
    if (!fs_result.ok()) {
      throw std::runtime_error{"An error occurred when setting up S3 connection. " +
                               fs_result.status().message()};
    }
    return fs_result.ValueOrDie();
  }
};
}  // namespace foreign_storage
#endif  //  defined(HAVE_AWS_S3)
