/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(HAVE_AWS_S3)
#include <map>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <aws/core/Globals.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>

#include "Catalog/ForeignServer.h"
#include "Catalog/UserMapping.h"

extern bool g_allow_s3_server_privileges;

namespace foreign_storage {
class ParquetS3FileSystem {
 public:
  static std::shared_ptr<arrow::fs::FileSystem> create(
      const ForeignServer* foreign_server,
      const UserMapping* user_mapping) {
    auto s3_options = arrow::fs::S3Options::Anonymous();
    if (user_mapping) {
      const auto options = user_mapping->getUnencryptedOptions();
      if (options.find(AbstractFileStorageDataWrapper::S3_ACCESS_KEY) != options.end() &&
          options.find(AbstractFileStorageDataWrapper::S3_SECRET_KEY) != options.end()) {
        if (options.find(AbstractFileStorageDataWrapper::S3_SESSION_TOKEN) !=
            options.end()) {
          s3_options = arrow::fs::S3Options::FromAccessKey(
              options.find(AbstractFileStorageDataWrapper::S3_ACCESS_KEY)->second,
              options.find(AbstractFileStorageDataWrapper::S3_SECRET_KEY)->second,
              options.find(AbstractFileStorageDataWrapper::S3_SESSION_TOKEN)->second);
        } else {
          s3_options = arrow::fs::S3Options::FromAccessKey(
              options.find(AbstractFileStorageDataWrapper::S3_ACCESS_KEY)->second,
              options.find(AbstractFileStorageDataWrapper::S3_SECRET_KEY)->second);
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
    s3_options.region =
        foreign_server->options.find(AbstractFileStorageDataWrapper::AWS_REGION_KEY)
            ->second;
    if (foreign_server->options.find(AbstractFileStorageDataWrapper::S3_ENDPOINT) !=
        foreign_server->options.end()) {
      s3_options.endpoint_override =
          foreign_server->options.find(AbstractFileStorageDataWrapper::S3_ENDPOINT)
              ->second;
    }

#ifdef HAVE_TSAN
    // TSAN requires more retry attempts to succeed
    s3_options.retry_strategy =
        arrow::fs::S3RetryStrategy::GetAwsStandardRetryStrategy(20);
#endif

    auto fs_result = arrow::fs::S3FileSystem::Make(s3_options);
    if (!fs_result.ok()) {
      throw std::runtime_error{"An error occurred when setting up S3 connection. " +
                               fs_result.status().message()};
    }
    return fs_result.ValueOrDie();
  }

  template <typename T>
  static auto arrowGetFileInfo(const std::shared_ptr<arrow::fs::FileSystem>& file_system,
                               const T& files) {
    // TODO: switch to using auto parameter type for C++20
    auto timer = DEBUG_TIMER(__func__);
    return file_system->GetFileInfo(files);
  }
};
}  // namespace foreign_storage
#endif  //  defined(HAVE_AWS_S3)
