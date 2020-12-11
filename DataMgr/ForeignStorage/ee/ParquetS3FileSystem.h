/*
 * Copyright 2020 OmniSci, Inc.
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

#if defined(HAVE_AWS_S3) && defined(ENABLE_S3_FSI)
#include <map>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <aws/core/Globals.h>

#include "Catalog/ForeignServer.h"
#include "Catalog/ee/UserMapping.h"

namespace foreign_storage {
class ParquetS3FileSystem {
 public:
  static std::shared_ptr<arrow::fs::FileSystem> create(
      const ForeignServer* foreign_server,
      const UserMapping* user_mapping) {
    static bool is_initialized = false;
    if (!is_initialized) {
      // Arrow requires its own S3 initialization (which does an AWS SDK initialization)
      // before any S3 related call can be made. However, S3Archive also does an
      // AWS SDK initialization. Both classes do not work if their own initialization
      // is not done. When both initializations are done, AWS SDK does not correctly
      // clean up before the second initialization. The following ensures that is done.
      // TODO: Figure out a way to have a single AWS SDK initialization that works for
      // both Arrow and S3Archive.
      Aws::CleanupEnumOverflowContainer();
      is_initialized = true;
    }

    arrow::fs::EnsureS3Initialized();
    auto s3_options = arrow::fs::S3Options::Anonymous();
    if (user_mapping) {
      const auto options = user_mapping->getUnencryptedOptions();
      if (options.find(UserMapping::S3_ACCESS_KEY) != options.end() &&
          options.find(UserMapping::S3_SECRET_KEY) != options.end()) {
        s3_options = arrow::fs::S3Options::FromAccessKey(
            options.find(UserMapping::S3_ACCESS_KEY)->second,
            options.find(UserMapping::S3_SECRET_KEY)->second);
      }
    }
    s3_options.region =
        foreign_server->options.find(ForeignServer::AWS_REGION_KEY)->second;
    auto fs_result = arrow::fs::S3FileSystem::Make(s3_options);
    if (!fs_result.ok()) {
      throw std::runtime_error{"An error occurred when setting up S3 connection. " +
                               fs_result.status().message()};
    }
    return fs_result.ValueOrDie();
  }
};
}  // namespace foreign_storage
#endif  //  defined(HAVE_AWS_S3) && defined(ENABLE_S3_FSI)
