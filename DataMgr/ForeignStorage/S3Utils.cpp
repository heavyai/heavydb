/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DataMgr/ForeignStorage/S3Utils.h"

#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/s3/S3ClientConfiguration.h>
#include <aws/s3/S3EndpointProvider.h>
#include <aws/s3/model/ListObjectsV2Request.h>

#include <boost/filesystem.hpp>

#include "Catalog/ForeignServer.h"
#include "Catalog/ForeignTable.h"
#include "Catalog/UserMapping.h"
#include "DataMgr/ForeignStorage/AbstractFileStorageDataWrapper.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/S3FilePathUtil.h"
#include "DataMgr/HeavyDbAwsSdk.h"

extern bool g_allow_s3_server_privileges;

namespace foreign_storage {

Aws::S3::S3ClientConfiguration get_s3_client_config(const ForeignServer* foreign_server) {
  Aws::S3::S3ClientConfiguration s3_config;
  s3_config.region =
      foreign_server->options.find(AbstractFileStorageDataWrapper::AWS_REGION_KEY)
          ->second;
  auto const ssl_config = heavydb_aws_sdk::get_ssl_config();
  s3_config.caPath = ssl_config.ca_path;
  s3_config.caFile = ssl_config.ca_file;
  if (foreign_server->options.find(AbstractFileStorageDataWrapper::S3_ENDPOINT) !=
      foreign_server->options.end()) {
    s3_config.endpointOverride =
        foreign_server->options.find(AbstractFileStorageDataWrapper::S3_ENDPOINT)->second;
  }
  s3_config.useVirtualAddressing = foreign_server->getOptionAsBool(
      AbstractFileStorageDataWrapper::S3_USE_VIRTUAL_ADDRESSING_KEY,
      /*default_value=*/true);
  return s3_config;
}

std::shared_ptr<Aws::S3::S3EndpointProviderBase> get_endpoint_provider(
    const Aws::S3::S3ClientConfiguration& s3_client_config) {
  // @TODO se/mg 9/18/24
  // does this need to be config-specific, or can we just return
  // a Aws::MakeShared<S3EndpointProvider>(ALLOCATION_TAG) like
  // the default parameter in the S3Client constructors
  auto endpoint_provider = std::make_shared<Aws::S3::S3EndpointProvider>();
  endpoint_provider->InitBuiltInParameters(s3_client_config);
  return endpoint_provider;
}

std::unique_ptr<Aws::S3::S3Client> create_s3_client(
    const ForeignServer* foreign_server,
    std::shared_ptr<Aws::Auth::AWSCredentialsProvider> aws_credentials) {
  auto s3_client_config = get_s3_client_config(foreign_server);
  auto endpoint_provider = get_endpoint_provider(s3_client_config);
  return std::make_unique<Aws::S3::S3Client>(
      aws_credentials, std::move(endpoint_provider), s3_client_config);
}

Aws::S3::Model::GetObjectRequest create_request(const std::string& bucket_name,
                                                const std::string& obj_name,
                                                size_t start,
                                                size_t end) {
  CHECK(start <= end);
  Aws::S3::Model::GetObjectRequest object_request;
  object_request.WithBucket(bucket_name).WithKey(obj_name);
  if (end > 0) {
    object_request.SetRange(std::string("bytes=") + std::to_string(start) + "-" +
                            std::to_string(end));
  }
  return object_request;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> get_credentials(
    const UserMapping* user_mapping) {
  if (user_mapping) {
    const auto options = user_mapping->getUnencryptedOptions();
    if (options.find(AbstractFileStorageDataWrapper::S3_ACCESS_KEY) != options.end() &&
        options.find(AbstractFileStorageDataWrapper::S3_SECRET_KEY) != options.end()) {
      if (options.find(AbstractFileStorageDataWrapper::S3_SESSION_TOKEN) !=
          options.end()) {
        return std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>(
            options.find(AbstractFileStorageDataWrapper::S3_ACCESS_KEY)->second,
            options.find(AbstractFileStorageDataWrapper::S3_SECRET_KEY)->second,
            options.find(AbstractFileStorageDataWrapper::S3_SESSION_TOKEN)->second);
      }
      return std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>(
          options.find(AbstractFileStorageDataWrapper::S3_ACCESS_KEY)->second,
          options.find(AbstractFileStorageDataWrapper::S3_SECRET_KEY)->second);
    }
  } else if (g_allow_s3_server_privileges) {
    return std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
  }
  return std::make_shared<Aws::Auth::AnonymousAWSCredentialsProvider>();
}

namespace {
Aws::S3::Model::ListObjectsV2Outcome s3_list_objects_v2(
    const std::unique_ptr<Aws::S3::S3Client>& s3_client,
    const Aws::S3::Model::ListObjectsV2Request& objects_request) {
  auto timer = DEBUG_TIMER(__func__);
  return s3_client->ListObjectsV2(objects_request);
}
}  // namespace

std::vector<S3FileInfo> list_files_s3(const std::unique_ptr<Aws::S3::S3Client>& s3_client,
                                      const std::string& prefix_name,
                                      const std::string& bucket_name,
                                      const shared::FilePathOptions& options) {
  Aws::S3::Model::ListObjectsV2Request objects_request;
  std::vector<S3FileInfo> file_infos;
  objects_request.WithBucket(bucket_name);
  objects_request.WithPrefix(prefix_name);
  auto list_objects_outcome = s3_list_objects_v2(s3_client, objects_request);
  if (list_objects_outcome.IsSuccess()) {
    auto object_list = list_objects_outcome.GetResult().GetContents();
    if (0 == object_list.size()) {
      throw_file_not_found_error(get_file_path(bucket_name, prefix_name));
    }
    object_list = s3_objects_filter_sort_files(object_list, options);

    // Instantiate FileReaderS3 for each valid object
    for (auto const& obj : object_list) {
      std::string objkey = obj.GetKey().c_str();

      // skip keys with trailing / (directories) or basename with heading '.'
      // (hidden/system files)
      boost::filesystem::path path{objkey};
      if (0 == obj.GetSize()) {
        continue;
      }
      if ('/' == objkey.back()) {
        continue;
      }
      if ('.' == path.filename().string().front()) {
        continue;
      }
      file_infos.emplace_back(S3FileInfo(objkey, obj.GetSize()));
    }
  } else {
    if (list_objects_outcome.GetError().GetResponseCode() ==
        Aws::Http::HttpResponseCode::NOT_FOUND) {
      throw_file_not_found_error(get_file_path(bucket_name, prefix_name));
    } else {
      throw_file_access_error(
          get_file_path(bucket_name, prefix_name),
          get_error_message(list_objects_outcome.GetError().GetExceptionName(),
                            list_objects_outcome.GetError().GetMessage()));
    }
  }
  return file_infos;
}

}  // namespace foreign_storage
