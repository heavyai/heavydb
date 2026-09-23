/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <aws/core/Aws.h>
#include <aws/core/VersionConfig.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace shared {
struct FilePathOptions;
}

namespace foreign_storage {

struct ForeignServer;
struct UserMapping;
struct ForeignTable;

Aws::S3::S3ClientConfiguration get_s3_client_config(const ForeignServer* server);
std::shared_ptr<Aws::S3::S3EndpointProviderBase> get_endpoint_provider(
    const Aws::S3::S3ClientConfiguration& s3_client_config);

std::unique_ptr<Aws::S3::S3Client> create_s3_client(
    const ForeignServer* foreign_server,
    std::shared_ptr<Aws::Auth::AWSCredentialsProvider> aws_credentials);

Aws::S3::Model::GetObjectRequest create_request(const std::string& bucket_name,
                                                const std::string& obj_name,
                                                size_t start = 0,
                                                size_t end = 0);

inline std::string get_file_path(const std::string& bucket,
                                 const std::string& object_name) {
  return bucket + "/" + object_name;
}

inline std::string get_error_message(const std::string& exception_name,
                                     const std::string& message) {
  return exception_name + ": " + message;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> get_credentials(
    const UserMapping* user_mapping);

using S3FileInfo = std::pair<std::string, size_t>;
std::vector<S3FileInfo> list_files_s3(const std::unique_ptr<Aws::S3::S3Client>& s3_client,
                                      const std::string& prefix_name,
                                      const std::string& bucket_name,
                                      const shared::FilePathOptions& options);

}  // namespace foreign_storage
