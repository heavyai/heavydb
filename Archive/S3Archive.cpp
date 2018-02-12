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
#include <atomic>
#include <glog/logging.h>
#include <boost/filesystem.hpp>
#include "S3Archive.h"

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/model/Object.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <fstream>

int S3Archive::awsapi_count;
std::mutex S3Archive::awsapi_mtx;
Aws::SDKOptions S3Archive::awsapi_options;

void S3Archive::init_for_read() {
  boost::filesystem::create_directories(s3_temp_dir);
  if (!boost::filesystem::is_directory(s3_temp_dir))
    throw std::runtime_error("failed to create s3_temp_dir directory '" + s3_temp_dir + "'");

  try {
    bucket_name = url_part(4);
    prefix_name = url_part(5);

    // a prefix '/obj/' should become 'obj/'
    // a prefix '/obj'  should become 'obj'
    if (prefix_name.size() && '/' == prefix_name.front())
      prefix_name = prefix_name.substr(1);

    Aws::S3::Model::ListObjectsRequest objects_request;
    objects_request.WithBucket(bucket_name);
    objects_request.WithPrefix(prefix_name);
    objects_request.SetMaxKeys(1 << 16);

    // for a daemon like mapd_server it seems improper to set s3 credentials
    // via AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env's because that way
    // credentials are configured *globally* while different users with private
    // s3 resources may need separate credentials to access.in that case, use
    // WITH s3_access_key/s3_secret_key parameters.
    Aws::Client::ClientConfiguration s3_config;
    s3_config.region = s3_region.size() ? s3_region : Aws::Region::US_EAST_1;

    if (!s3_access_key.empty() && !s3_secret_key.empty())
      s3_client.reset(new Aws::S3::S3Client(Aws::Auth::AWSCredentials(s3_access_key, s3_secret_key), s3_config));
    else
      s3_client.reset(new Aws::S3::S3Client(s3_config));

    auto list_objects_outcome = s3_client->ListObjects(objects_request);
    if (!list_objects_outcome.IsSuccess())
      throw std::runtime_error("failed to list objects of s3 url '" + url + "': " +
                               list_objects_outcome.GetError().GetExceptionName() + ": " +
                               list_objects_outcome.GetError().GetMessage());

    // pass only object keys to next stage, which may be Importer::import_parquet,
    // Importer::import_compressed or else, depending on copy_params (eg. .is_parquet)
    auto object_list = list_objects_outcome.GetResult().GetContents();
    if (0 == object_list.size())
      throw std::runtime_error("no object was found with s3 url '" + url + "'");

    LOG(INFO) << "Found " << object_list.size() << " objects with url '" + url + "':";
    for (auto const& obj : object_list) {
      std::string objkey = obj.GetKey().c_str();
      LOG(INFO) << "\t" << objkey << " (size = " << obj.GetSize() << " bytes)";
      // skip _SUCCESS and keys with trailing / or basename with heading '.'
      boost::filesystem::path path{objkey};
      if (0 == obj.GetSize())
        continue;
      if ('/' == objkey.back())
        continue;
      if ('.' == path.filename().string().front())
        continue;
      objkeys.push_back(objkey);
    }
  } catch (...) {
    throw;
  }
}

// a bit complicated with S3 archive of parquet files is that these files
// use parquet api (not libarchive) and the files must be landed locally
// to be imported. besides, since parquet archives are often big in size
// to avoid extra EBS cost to customers, generally we don't want to land
// them at once but one by one.
//
// likely in the future there will be other file types that need to
// land entirely to be imported... (avro?)
const std::string S3Archive::land(const std::string& objkey, std::exception_ptr& teptr) {
  // 7z file needs entire landing; other file types use a named pipe
  static std::atomic<int64_t> seqno(((int64_t)getpid() << 32) | time(0));
  // need a dummy ext b/c no-ext now indicate plain_text
  std::string file_path = s3_temp_dir + "/s3tmp_" + std::to_string(++seqno) + ".s3";
  boost::filesystem::remove(file_path);

  auto ext = strrchr(objkey.c_str(), '.');
  auto use_pipe = (0 == ext || 0 != strcmp(ext, ".7z"));
  if (use_pipe)
    if (mkfifo(file_path.c_str(), 0660) < 0)
      throw std::runtime_error("failed to create named pipe '" + file_path + "': " + strerror(errno));

  // streaming means asynch
  auto th_writer = std::thread([=, &teptr]() {
    LOG(INFO) << "download s3://" << bucket_name << "/" << objkey << " to " << (use_pipe ? "pipe " : "file ")
              << file_path;
    try {
      Aws::S3::Model::GetObjectRequest object_request;
      object_request.WithBucket(bucket_name).WithKey(objkey);
      auto get_object_outcome = s3_client->GetObject(object_request);
      if (!get_object_outcome.IsSuccess())
        throw std::runtime_error("failed to get object '" + objkey + "' of s3 url '" + url + "': " +
                                 get_object_outcome.GetError().GetExceptionName() + ": " +
                                 get_object_outcome.GetError().GetMessage());

      Aws::OFStream local_file;
      local_file.open(file_path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
      local_file << get_object_outcome.GetResult().GetBody().rdbuf();
    } catch (...) {
      // need this way to capture any exception occurring when
      // this thread runs as a disjoint asynchronous thread
      if (use_pipe)
        teptr = std::current_exception();
      else
        throw;
    }
  });

  if (use_pipe)
    th_writer.detach();
  else
    th_writer.join();

  file_paths.insert(std::pair<const std::string, const std::string>(objkey, file_path));
  return file_path;
}

void S3Archive::vacuum(const std::string& objkey) {
  auto it = file_paths.find(objkey);
  if (file_paths.end() == it)
    return;
  boost::filesystem::remove(it->second);
  file_paths.erase(it);
}
