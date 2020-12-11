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
#if defined(HAVE_AWS_S3) && defined(ENABLE_S3_FSI)
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/Object.h>

#include "Catalog/ForeignServer.h"
#include "DataMgr/ForeignStorage/FsiJsonUtils.h"
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

std::string get_access_error_message(const std::string& bucket,
                                     const std::string& object_name,
                                     const std::string& exception_name,
                                     const std::string& message) {
  return "Unable to access s3 file: " + bucket + "/" + object_name + ". " +
         exception_name + ": " + message;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> get_credentials(
    const UserMapping* user_mapping) {
  if (user_mapping) {
    const auto options = user_mapping->getUnencryptedOptions();
    if (options.find(UserMapping::S3_ACCESS_KEY) != options.end() &&
        options.find(UserMapping::S3_SECRET_KEY) != options.end()) {
      return std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>(
          options.find(UserMapping::S3_ACCESS_KEY)->second,
          options.find(UserMapping::S3_SECRET_KEY)->second);
    }
  }
  return std::make_shared<Aws::Auth::AnonymousAWSCredentialsProvider>();
}

}  // namespace

CsvReaderS3::CsvReaderS3(const std::string& obj_key,
                         size_t file_size,
                         const import_export::CopyParams& copy_params,
                         const ForeignServer* server_options,
                         const UserMapping* user_mapping)
    : CsvReader(obj_key, copy_params)
    , file_size_(file_size)
    , scan_finished_(false)
    , obj_key_(obj_key)
    , copy_params_(copy_params)
    , current_offset_(0)
    , header_offset_(0) {
  bucket_name_ = server_options->options.find(ForeignServer::S3_BUCKET_KEY)->second;
  s3_client_.reset(new Aws::S3::S3Client(get_credentials(user_mapping),
                                         get_s3_config(server_options)));
  skipHeader();
  if (header_offset_ >= file_size_) {
    scan_finished_ = true;
  }
  file_size_ = file_size_ - header_offset_;
}

CsvReaderS3::CsvReaderS3(const std::string& obj_key,
                         const import_export::CopyParams& copy_params,
                         const ForeignServer* server_options,
                         const UserMapping* user_mapping,
                         const rapidjson::Value& value)
    : CsvReader(obj_key, copy_params)
    , scan_finished_(false)
    , obj_key_(obj_key)
    , copy_params_(copy_params)
    , current_offset_(0)
    , header_offset_(0) {
  bucket_name_ = server_options->options.find(ForeignServer::S3_BUCKET_KEY)->second;
  s3_client_.reset(new Aws::S3::S3Client(get_credentials(user_mapping),
                                         get_s3_config(server_options)));
  scan_finished_ = true;
  json_utils::get_value_from_object(value, header_offset_, "header_offset");
  json_utils::get_value_from_object(value, file_size_, "file_size");
}

void CsvReaderS3::serialize(rapidjson::Value& value,
                            rapidjson::Document::AllocatorType& allocator) const {
  CHECK(scan_finished_);
  json_utils::add_value_to_object(value, header_offset_, "header_offset", allocator);
  json_utils::add_value_to_object(value, file_size_, "file_size", allocator);
};

size_t CsvReaderS3::read(void* buffer, size_t max_size) {
  size_t byte_start = header_offset_ + current_offset_;
  size_t byte_end = byte_start + max_size;
  auto object_request = create_request(bucket_name_, obj_key_, byte_start, byte_end);
  auto get_object_outcome = s3_client_->GetObject(object_request);

  if (!get_object_outcome.IsSuccess()) {
    throw std::runtime_error{
        get_access_error_message(bucket_name_,
                                 obj_key_,
                                 get_object_outcome.GetError().GetExceptionName(),
                                 get_object_outcome.GetError().GetMessage())};
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
        throw std::runtime_error{
            get_access_error_message(bucket_name_,
                                     obj_key_,
                                     get_object_outcome.GetError().GetExceptionName(),
                                     get_object_outcome.GetError().GetMessage())};
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

void CsvReaderS3::increaseFileSize(size_t new_size) {
  CHECK(scan_finished_);
  CHECK_GT(new_size, file_size_);
  current_offset_ = file_size_;
  file_size_ = new_size;
  scan_finished_ = false;
}

namespace {

using S3FileInfo = std::pair<std::string, size_t>;
void list_files_s3(std::unique_ptr<Aws::S3::S3Client>& s3_client,
                   const std::string& prefix_name,
                   const std::string& bucket_name,
                   std::set<S3FileInfo>& file_info_set) {
  Aws::S3::Model::ListObjectsV2Request objects_request;
  objects_request.WithBucket(bucket_name);
  objects_request.WithPrefix(prefix_name);
  auto list_objects_outcome = s3_client->ListObjectsV2(objects_request);
  if (list_objects_outcome.IsSuccess()) {
    auto object_list = list_objects_outcome.GetResult().GetContents();
    if (0 == object_list.size()) {
      throw std::runtime_error{get_access_error_message(
          bucket_name, prefix_name, "Error", "No object was found at the given path.")};
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
      file_info_set.insert(S3FileInfo(objkey, obj.GetSize()));
    }
  } else {
    throw std::runtime_error{
        get_access_error_message(bucket_name,
                                 prefix_name,
                                 list_objects_outcome.GetError().GetExceptionName(),
                                 list_objects_outcome.GetError().GetMessage())};
  }
}
}  // namespace

MultiS3Reader::MultiS3Reader(const std::string& prefix_name,
                             const import_export::CopyParams& copy_params,
                             const ForeignServer* foreign_server,
                             const UserMapping* user_mapping)
    : MultiFileReader(prefix_name, copy_params) {
  auto credentials = get_credentials(user_mapping);
  auto config = get_s3_config(foreign_server);
  s3_client_.reset(new Aws::S3::S3Client(credentials, config));
  bucket_name_ = foreign_server->options.find(ForeignServer::S3_BUCKET_KEY)->second;
  std::set<S3FileInfo> file_info_set;
  list_files_s3(s3_client_, prefix_name, bucket_name_, file_info_set);
  for (const auto& file_info : file_info_set) {
    files_.emplace_back(std::make_unique<CsvReaderS3>(
        file_info.first, file_info.second, copy_params, foreign_server, user_mapping));
    file_locations_.push_back(file_info.first);
    file_sizes_.push_back(file_info.second);
  }
}

MultiS3Reader::MultiS3Reader(const std::string& file_path,
                             const import_export::CopyParams& copy_params,
                             const ForeignServer* foreign_server,
                             const UserMapping* user_mapping,
                             const rapidjson::Value& value)
    : MultiFileReader(file_path, copy_params, value) {
  auto credentials = get_credentials(user_mapping);
  auto config = get_s3_config(foreign_server);
  s3_client_.reset(new Aws::S3::S3Client(credentials, config));
  bucket_name_ = foreign_server->options.find(ForeignServer::S3_BUCKET_KEY)->second;
  // reconstruct files from metadata
  CHECK(value.HasMember("files_metadata"));
  for (size_t index = 0; index < file_locations_.size(); index++) {
    files_.emplace_back(
        std::make_unique<CsvReaderS3>(file_locations_[index],
                                      copy_params,
                                      foreign_server,
                                      user_mapping,
                                      value["files_metadata"].GetArray()[index]));
  }
  json_utils::get_value_from_object(value, file_sizes_, "file_sizes");
}

void MultiS3Reader::serialize(rapidjson::Value& value,
                              rapidjson::Document::AllocatorType& allocator) const {
  json_utils::add_value_to_object(value, file_sizes_, "file_sizes", allocator);
  MultiFileReader::serialize(value, allocator);
};

void MultiS3Reader::checkForMoreRows(size_t file_offset,
                                     const ForeignServer* foreign_server,
                                     const UserMapping* user_mapping) {
  CHECK(isScanFinished());
  CHECK(file_offset == current_offset_);
  CHECK(foreign_server != nullptr);

  // Look for new files
  std::set<S3FileInfo> file_info_set;
  list_files_s3(s3_client_, file_path_, bucket_name_, file_info_set);
  int new_files = 0;
  for (const auto& file_info : file_info_set) {
    if (std::find(file_locations_.begin(), file_locations_.end(), file_info.first) ==
        file_locations_.end()) {
      files_.emplace_back(std::make_unique<CsvReaderS3>(
          file_info.first, file_info.second, copy_params_, foreign_server, user_mapping));
      file_locations_.push_back(file_info.first);
      new_files++;
    }
  }
  // If no new files added and only one file in archive, check for new rows
  if (new_files == 0 && files_.size() == 1) {
    if (file_info_set.size() < 1 ||
        find(file_locations_.begin(),
             file_locations_.end(),
             file_info_set.begin()->first) == file_locations_.end()) {
      throw std::runtime_error{
          "Foreign table refreshed with APPEND mode missing entry \"" +
          file_locations_[0] + "\"."};
    }
    if (file_info_set.begin()->second < file_sizes_[0]) {
      throw std::runtime_error{
          "Refresh of foreign table created with APPEND update mode failed as remote "
          "file "
          "reduced in size: \"" +
          file_locations_[0] + "\"."};
    }

    if (file_info_set.begin()->second > file_sizes_[0]) {
      CsvReaderS3* s3_reader = dynamic_cast<CsvReaderS3*>(files_[0].get());
      CHECK(s3_reader != nullptr);
      s3_reader->increaseFileSize(file_info_set.begin()->second);
      file_sizes_[0] = file_info_set.begin()->second;
      current_index_ = 0;
      cumulative_sizes_ = {};
    }
  }
}

}  // namespace foreign_storage
#endif  //  defined(HAVE_AWS_S3) && defined(ENABLE_S3_FSI)
