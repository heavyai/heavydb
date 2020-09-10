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
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/Object.h>

#include "Catalog/ForeignServer.h"
#include "DataMgr/ForeignStorage/ee/CsvReaderS3.h"

namespace foreign_storage {

namespace {

Aws::Client::ClientConfiguration get_s3_config(const ForeignServer* server_options) {
  Aws::Client::ClientConfiguration s3_config;
  s3_config.region = server_options->options.find(ForeignServer::AWS_REGION_KEY)->second;

  // Find SSL certificate trust store to connect to S3
  std::list<std::string> v_known_ca_paths({
      "/etc/ssl/certs/ca-certificates.crt",
      "/etc/pki/tls/certs/ca-bundle.crt",
      "/usr/share/ssl/certs/ca-bundle.crt",
      "/usr/local/share/certs/ca-root.crt",
      "/etc/ssl/cert.pem",
      "/etc/ssl/ca-bundle.pem",
  });
  char* env;
  if (nullptr != (env = getenv("SSL_CERT_DIR"))) {
    s3_config.caPath = env;
  }
  if (nullptr != (env = getenv("SSL_CERT_FILE"))) {
    v_known_ca_paths.push_front(env);
  }
  for (const auto& known_ca_path : v_known_ca_paths) {
    if (boost::filesystem::exists(known_ca_path)) {
      s3_config.caFile = known_ca_path;
      break;
    }
  }
  return s3_config;
}

Aws::S3::Model::GetObjectRequest create_request(const std::string& bucket_name,
                                                const std::string& obj_name,
                                                size_t start = 0,
                                                size_t end = 0) {
  CHECK(start <= end);
  Aws::S3::Model::GetObjectRequest object_request;
  object_request.WithBucket(bucket_name).WithKey(obj_name);
  if (end > 0) {
    object_request.SetRange(std::string("bytes=") + std::to_string(start) + "-" +
                            std::to_string(end));
  }
  return object_request;
}

std::string get_s3_url(std::string bucket, std::string prefix) {
  return "s3://" + bucket + "/" + prefix;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> get_credentials() {
  // TODO: get user config mapping
  return std::make_shared<Aws::Auth::AnonymousAWSCredentialsProvider>();
}

}  // namespace

CsvReaderS3::CsvReaderS3(const std::string& obj_key,
                         size_t file_size,
                         const import_export::CopyParams& copy_params,
                         const ForeignServer* server_options)
    : CsvReader(obj_key, copy_params)
    , file_size_(file_size)
    , scan_finished_(false)
    , obj_key_(obj_key)
    , copy_params_(copy_params)
    , current_offset_(0)
    , header_offset_(0) {
  bucket_name_ = server_options->options.find(ForeignServer::S3_BUCKET_KEY)->second;
  s3_client_.reset(
      new Aws::S3::S3Client(get_credentials(), get_s3_config(server_options)));
  skipHeader();
  if (header_offset_ >= file_size_) {
    scan_finished_ = true;
  }
  file_size_ = file_size_ - header_offset_;
}

size_t CsvReaderS3::read(void* buffer, size_t max_size) {
  size_t byte_start = header_offset_ + current_offset_;
  size_t byte_end = byte_start + max_size;
  auto object_request = create_request(bucket_name_, obj_key_, byte_start, byte_end);
  auto get_object_outcome = s3_client_->GetObject(object_request);

  if (!get_object_outcome.IsSuccess()) {
    throw std::runtime_error("Failed to get object '" + obj_key_ + "' of s3 url '" +
                             get_s3_url(bucket_name_, obj_key_) +
                             "': " + get_object_outcome.GetError().GetExceptionName() +
                             ": " + get_object_outcome.GetError().GetMessage());
  }
  get_object_outcome.GetResult().GetBody().read(static_cast<char*>(buffer), max_size);

  size_t read_bytes = get_object_outcome.GetResult().GetBody().gcount();
  current_offset_ += read_bytes;
  if (current_offset_ + header_offset_ >= file_size_) {
    scan_finished_ = true;
  }
  return read_bytes;
}

void CsvReaderS3::skipHeader() {
  if (copy_params_.has_header != import_export::ImportHeaderRow::NO_HEADER) {
    size_t header_size = 1024;
    bool header_found = false;
    while (header_found == false) {
      auto object_request = create_request(bucket_name_, obj_key_, 0, header_size);

      std::unique_ptr<char[]> header_buff = std::make_unique<char[]>(header_size);
      auto get_object_outcome = s3_client_->GetObject(object_request);

      if (!get_object_outcome.IsSuccess()) {
        throw std::runtime_error("Failed to get object '" + obj_key_ + "' of s3 url '" +
                                 get_s3_url(bucket_name_, obj_key_) + "': " +
                                 get_object_outcome.GetError().GetExceptionName() + ": " +
                                 get_object_outcome.GetError().GetMessage());
      }

      get_object_outcome.GetResult().GetBody().getline((header_buff.get()), header_size);
      if (get_object_outcome.GetResult().GetBody().fail()) {
        // We didnt get a full line
        if (header_size == file_size_) {
          // File only contains one header line
          header_offset_ = file_size_;
          break;
        }
        header_size *= 2;
        if (header_size > file_size_) {
          header_size = file_size_;
        }
      } else {
        header_offset_ = get_object_outcome.GetResult().GetBody().gcount();
        header_found = true;
      }
    }
  }
}

MultiS3Reader::MultiS3Reader(const std::string& prefix_name,
                             const import_export::CopyParams& copy_params,
                             const ForeignServer* server_options)
    : MultiFileReader(prefix_name, copy_params) {
  Aws::S3::Model::ListObjectsV2Request objects_request;
  auto bucket_name = server_options->options.find(ForeignServer::S3_BUCKET_KEY)->second;
  objects_request.WithBucket(bucket_name);
  objects_request.WithPrefix(prefix_name);

  std::unique_ptr<Aws::S3::S3Client> s3_client;
  auto credentials = get_credentials();
  auto config = get_s3_config(server_options);
  s3_client.reset(new Aws::S3::S3Client(credentials, config));
  auto list_objects_outcome = s3_client->ListObjectsV2(objects_request);
  if (list_objects_outcome.IsSuccess()) {
    auto object_list = list_objects_outcome.GetResult().GetContents();
    if (0 == object_list.size()) {
      throw std::runtime_error("No object was found with s3 url '" +
                               get_s3_url(bucket_name, prefix_name) + "'");
    }
    // Instantiate CsvReaderS3 for each valid object
    for (auto const& obj : object_list) {
      std::string objkey = obj.GetKey().c_str();

      // skip keys with trailing / or basename with heading '.'
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
      // TODO: remove filename restriction on txt when new S3 test datasests are added
      if (boost::filesystem::extension(path) != ".csv" &&
          boost::filesystem::extension(path) != ".tsv") {
        continue;
      }
      files_.emplace_back(std::make_unique<CsvReaderS3>(
          objkey, obj.GetSize(), copy_params, server_options));
    }
  } else {
    throw std::runtime_error("Could not list objects for s3 url '" +
                             get_s3_url(bucket_name, prefix_name) + "'");
  }
}
}  // namespace foreign_storage
