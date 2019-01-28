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

#include <stdio.h>
#include <exception>
#include <map>
#include <thread>
#include "Archive.h"

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
// init aws api should be singleton because because
// it's bad to call Aws::InitAPI and Aws::ShutdownAPI
// multiple times.
#ifdef HAVE_AWS_S3
    {
      std::unique_lock<std::mutex> lck(awsapi_mtx);
      if (0 == awsapi_count++)
        Aws::InitAPI(awsapi_options);
    }
#endif  // HAVE_AWS_S3

    // these envs are on server side so are global settings
    // which make few senses in case of private s3 resources
    char* env;
    if (0 != (env = getenv("AWS_REGION")))
      s3_region = env;
    if (0 != (env = getenv("AWS_ACCESS_KEY_ID")))
      s3_access_key = env;
    if (0 != (env = getenv("AWS_SECRET_ACCESS_KEY")))
      s3_secret_key = env;
  }

  S3Archive(const std::string& url,
            const std::string& s3_access_key,
            const std::string& s3_secret_key,
            const std::string& s3_region,
            const bool plain_text)
      : S3Archive(url, plain_text) {
    this->s3_access_key = s3_access_key;
    this->s3_secret_key = s3_secret_key;
    this->s3_region = s3_region;

    // this must be local to omnisci_server not client
    // or posix dir path accessible to omnisci_server
    auto env_s3_temp_dir = getenv("TMPDIR");
    s3_temp_dir = env_s3_temp_dir ? env_s3_temp_dir : "/tmp";
  }

  virtual ~S3Archive() {
#ifdef HAVE_AWS_S3
    for (auto& thread : threads)
      if (thread.joinable())
        thread.join();
    std::unique_lock<std::mutex> lck(awsapi_mtx);
    if (0 == --awsapi_count)
      Aws::ShutdownAPI(awsapi_options);
#endif  // HAVE_AWS_S3
  }

  virtual void init_for_read();
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
  std::string s3_region;
  std::string s3_temp_dir;

  std::string bucket_name;
  std::string prefix_name;
  std::vector<std::string> objkeys;
  std::map<const std::string, const std::string> file_paths;
  size_t total_file_size{0};
};

class S3ParquetArchive : public S3Archive {
 public:
  S3ParquetArchive(const std::string& url,
                   const std::string& s3_access_key,
                   const std::string& s3_secret_key,
                   const std::string& s3_region,
                   const bool plain_text)
      : S3Archive(url, s3_access_key, s3_secret_key, s3_region, plain_text) {}

 private:
};

#endif /* ARCHIVE_S3ARCHIVE_H_ */
