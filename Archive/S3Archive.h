/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef ARCHIVE_S3ARCHIVE_H_
#define ARCHIVE_S3ARCHIVE_H_

#include <cstdio>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include "Archive.h"

#include <openssl/evp.h>

#ifdef HAVE_AWS_S3
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#endif  // HAVE_AWS_S3

// this is the based archive class for files hosted on AWS S3.
// known variants:
//  . parquet files
//  . compressed files
// no mixed of above is supported yet
class S3Archive : public Archive {
 public:
  S3Archive(const std::string& url, const bool plain_text) : Archive(url, plain_text) {
    // these envs are on server side so are global settings
    // which make few senses in case of private s3 resources
    char* env;
    if (0 != (env = getenv("AWS_REGION"))) {
      s3_region = env;
    }
    if (0 != (env = getenv("AWS_ACCESS_KEY_ID"))) {
      s3_access_key = env;
    }
    if (0 != (env = getenv("AWS_SECRET_ACCESS_KEY"))) {
      s3_secret_key = env;
    }
    if (0 != (env = getenv("AWS_SESSION_TOKEN"))) {
      s3_session_token = env;
    }

    if (0 != (env = getenv("AWS_ENDPOINT"))) {
      s3_endpoint = env;
    }
  }

  S3Archive(const std::string& url,
            const std::string& s3_access_key,
            const std::string& s3_secret_key,
            const std::string& s3_session_token,
            const std::string& s3_region,
            const std::string& s3_endpoint,
            const bool plain_text,
            const std::optional<std::string>& regex_path_filter,
            const std::optional<std::string>& file_sort_order_by,
            const std::optional<std::string>& file_sort_regex)
      : S3Archive(url, plain_text) {
    this->s3_access_key = s3_access_key;
    this->s3_secret_key = s3_secret_key;
    this->s3_session_token = s3_session_token;
    this->s3_region = s3_region;
    this->s3_endpoint = s3_endpoint;
    this->regex_path_filter = regex_path_filter;
    this->file_sort_order_by = file_sort_order_by;
    this->file_sort_regex = file_sort_regex;

    // this must be local to omnisci_server not client
    // or posix dir path accessible to omnisci_server
    auto env_s3_temp_dir = getenv("TMPDIR");
    s3_temp_dir = env_s3_temp_dir ? env_s3_temp_dir : "/tmp";
  }

  ~S3Archive() override {
#ifdef HAVE_AWS_S3
    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
#endif  // HAVE_AWS_S3
  }

#ifdef HAVE_AWS_S3
  void init_for_read() override;
#else
  void init_for_read() override {
    throw std::runtime_error("AWS S3 support not available");
  }
#endif
  const std::vector<std::string>& get_objkeys() { return objkeys; }
#ifdef HAVE_AWS_S3
  const std::string land(const std::string& objkey,
                         std::exception_ptr& teptr,
                         const bool for_detection);
  void vacuum(const std::string& objkey);
#else
  const std::string land(const std::string& objkey,
                         std::exception_ptr& teptr,
                         const bool for_detection) {
    throw std::runtime_error("AWS S3 support not available");
  }
  void vacuum(const std::string& objkey) {
    throw std::runtime_error("AWS S3 support not available");
  }
#endif  // HAVE_AWS_S3
  size_t get_total_file_size() const { return total_file_size; }

 private:
#ifdef HAVE_AWS_S3
  static int awsapi_count;
  static std::mutex awsapi_mtx;
  static Aws::SDKOptions awsapi_options;

  std::unique_ptr<Aws::S3::S3Client> s3_client;
  std::vector<std::thread> threads;
#endif                        // HAVE_AWS_S3
  std::string s3_access_key;  // per-query credentials to override the
  std::string s3_secret_key;  // settings in ~/.aws/credentials or environment
  std::string s3_session_token;
  std::string s3_region;
  std::string s3_endpoint;
  std::string s3_temp_dir;

  std::string bucket_name;
  std::string prefix_name;
  std::optional<std::string> regex_path_filter;
  std::optional<std::string> file_sort_order_by;
  std::optional<std::string> file_sort_regex;
  std::vector<std::string> objkeys;
  std::map<const std::string, const std::string> file_paths;
  size_t total_file_size{0};
};

class S3ParquetArchive : public S3Archive {
 public:
  S3ParquetArchive(const std::string& url,
                   const std::string& s3_access_key,
                   const std::string& s3_secret_key,
                   const std::string& s3_session_token,
                   const std::string& s3_region,
                   const std::string& s3_endpoint,
                   const bool plain_text,
                   const std::optional<std::string>& regex_path_filter,
                   const std::optional<std::string>& file_sort_order_by,
                   const std::optional<std::string>& file_sort_regex)
      : S3Archive(url,
                  s3_access_key,
                  s3_secret_key,
                  s3_session_token,
                  s3_region,
                  s3_endpoint,
                  plain_text,
                  regex_path_filter,
                  file_sort_order_by,
                  file_sort_regex) {}
};

#endif /* ARCHIVE_S3ARCHIVE_H_ */
