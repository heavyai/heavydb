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
#include "S3Archive.h"
#include <glog/logging.h>
#include <atomic>
#include <boost/filesystem.hpp>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <aws/s3/model/Object.h>
#include <fstream>
#include <memory>

int S3Archive::awsapi_count;
std::mutex S3Archive::awsapi_mtx;
Aws::SDKOptions S3Archive::awsapi_options;

void S3Archive::init_for_read() {
  boost::filesystem::create_directories(s3_temp_dir);
  if (!boost::filesystem::is_directory(s3_temp_dir)) {
    throw std::runtime_error("failed to create s3_temp_dir directory '" + s3_temp_dir +
                             "'");
  }

  try {
    bucket_name = url_part(4);
    prefix_name = url_part(5);

    // a prefix '/obj/' should become 'obj/'
    // a prefix '/obj'  should become 'obj'
    if (prefix_name.size() && '/' == prefix_name.front()) {
      prefix_name = prefix_name.substr(1);
    }

    Aws::S3::Model::ListObjectsV2Request objects_request;
    objects_request.WithBucket(bucket_name);
    objects_request.WithPrefix(prefix_name);
    objects_request.SetMaxKeys(1 << 20);

    // for a daemon like omnisci_server it seems improper to set s3 credentials
    // via AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env's because that way
    // credentials are configured *globally* while different users with private
    // s3 resources may need separate credentials to access.in that case, use
    // WITH s3_access_key/s3_secret_key parameters.
    Aws::Client::ClientConfiguration s3_config;
    s3_config.region = s3_region.size() ? s3_region : Aws::Region::US_EAST_1;

    /*
       Fix a wrong ca path established at building libcurl on Centos being carried to
       Ubuntu. To fix the issue, this is this sequence of locating ca file: 1) if
       `SSL_CERT_DIR` or `SSL_CERT_FILE` is set, set it to S3 ClientConfiguration. 2) if
       none ^ is set, omnisci core searches a list of known ca file paths. 3) if 2) finds
       nothing, it is users' call to set correct SSL_CERT_DIR or SSL_CERT_FILE. S3 c++
       sdk: "we only want to override the default path if someone has explicitly told us
       to."
     */
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

    if (!s3_access_key.empty() && !s3_secret_key.empty()) {
      s3_client.reset(new Aws::S3::S3Client(
          Aws::Auth::AWSCredentials(s3_access_key, s3_secret_key), s3_config));
    } else {
      s3_client.reset(new Aws::S3::S3Client(
          std::make_shared<Aws::Auth::AnonymousAWSCredentialsProvider>(), s3_config));
    }
    while (true) {
      auto list_objects_outcome = s3_client->ListObjectsV2(objects_request);
      if (list_objects_outcome.IsSuccess()) {
        // pass only object keys to next stage, which may be Importer::import_parquet,
        // Importer::import_compressed or else, depending on copy_params (eg. .is_parquet)
        auto object_list = list_objects_outcome.GetResult().GetContents();
        if (0 == object_list.size()) {
          if (objkeys.empty()) {
            throw std::runtime_error("no object was found with s3 url '" + url + "'");
          }
        }

        LOG(INFO) << "Found " << (objkeys.empty() ? "" : "another ") << object_list.size()
                  << " objects with url '" + url + "':";
        for (auto const& obj : object_list) {
          std::string objkey = obj.GetKey().c_str();
          LOG(INFO) << "\t" << objkey << " (size = " << obj.GetSize() << " bytes)";
          total_file_size += obj.GetSize();
          // skip _SUCCESS and keys with trailing / or basename with heading '.'
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
          objkeys.push_back(objkey);
        }
      } else {
        // could not ListObject
        // could be the object is there but we do not have listObject Privilege
        // We can treat it as a specific object, so should try to parse it and pass to
        // getObject as a singleton
        if (objkeys.empty()) {
          objkeys.push_back(prefix_name);
        }
      }
      // continue to read next 1000 files
      if (list_objects_outcome.GetResult().GetIsTruncated()) {
        objects_request.SetContinuationToken(
            list_objects_outcome.GetResult().GetNextContinuationToken());
      } else {
        break;
      }
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
const std::string S3Archive::land(const std::string& objkey,
                                  std::exception_ptr& teptr,
                                  const bool for_detection) {
  // 7z file needs entire landing; other file types use a named pipe
  static std::atomic<int64_t> seqno(((int64_t)getpid() << 32) | time(0));
  // need a dummy ext b/c no-ext now indicate plain_text
  std::string file_path = s3_temp_dir + "/s3tmp_" + std::to_string(++seqno) + ".s3";
  boost::filesystem::remove(file_path);

  auto ext = strrchr(objkey.c_str(), '.');
  auto use_pipe = (0 == ext || 0 != strcmp(ext, ".7z"));
  if (use_pipe) {
    if (mkfifo(file_path.c_str(), 0660) < 0) {
      throw std::runtime_error("failed to create named pipe '" + file_path +
                               "': " + strerror(errno));
    }
  }

  /*
    Here is the background info that makes the thread interaction here a bit subtle:
    1) We need two threading modes for the `th_writer` thread below:
       a) synchronous mode to land .7z files or any file that must land fully as a local
    file before it can be processed by libarchive. b) asynchronous mode to land a file
    that can be processed by libarchive as a stream. With this mode, the file is streamed
    into a temporary named pipe. 2) Cooperating with the `th_writer` thread is the
    `th_pipe_writer` thread in Importer.cpp. For mode b), th_pipe_writer thread reads data
    from the named pipe written by th_writer. Before it reads, it needs to open the pipe.
    It will be blocked indefinitely (hang) if th_writer exits from any error before
    th_pipe_writer opens the pipe. 3) AWS S3 client s3_client->GetObject returns an
    'object' rather than a pointer to the object. That makes it hard to use smart pointer
    for RAII. Calling s3_client->GetObject in th_writer body appears to the immediate
    approach. 4) If s3_client->GetObject were called inside th_writer and th_writer is in
    async mode, a tragic scenario is that th_writer receives an error (eg. bad
    credentials) from AWS S3 server then quits when th_pipe_writer has proceeded to open
    the named pipe and get blocked (hangs).

    So a viable approach is to move s3_client->GetObject out of th_writer body but *move*
    the `object outcome` into th_writer body. This way we can better assure any error of
    s3_client->GetObject will be thrown immediately to upstream (ie. th_pipe_writer) and
    `object outcome` will be released later after the object is landed.
   */
  Aws::S3::Model::GetObjectRequest object_request;
  object_request.WithBucket(bucket_name).WithKey(objkey);

  // set a download byte range (max 10mb) to avoid getting stuck on detecting big s3 files
  if (use_pipe && for_detection) {
    object_request.SetRange("bytes=0-10000000");
  }

  auto get_object_outcome = s3_client->GetObject(object_request);
  if (!get_object_outcome.IsSuccess()) {
    throw std::runtime_error("failed to get object '" + objkey + "' of s3 url '" + url +
                             "': " + get_object_outcome.GetError().GetExceptionName() +
                             ": " + get_object_outcome.GetError().GetMessage());
  }

  // streaming means asynch
  std::atomic<bool> is_get_object_outcome_moved(false);
  // fix a race between S3Archive::land and S3Archive::~S3Archive on S3Archive itself
  auto& bucket_name = this->bucket_name;
  auto th_writer =
      std::thread([=, &teptr, &get_object_outcome, &is_get_object_outcome_moved]() {
        try {
          // this static mutex protect the static google::last_tm_time_for_raw_log from
          // concurrent LOG(INFO)s that call RawLog__SetLastTime to write the variable!
          static std::mutex mutex_glog;
#define MAPD_S3_LOG(x)                             \
  {                                                \
    std::unique_lock<std::mutex> lock(mutex_glog); \
    x;                                             \
  }
          MAPD_S3_LOG(LOG(INFO)
                      << "downloading s3://" << bucket_name << "/" << objkey << " to "
                      << (use_pipe ? "pipe " : "file ") << file_path << "...")
          auto get_object_outcome_moved =
              decltype(get_object_outcome)(std::move(get_object_outcome));
          is_get_object_outcome_moved = true;
          Aws::OFStream local_file;
          local_file.open(file_path.c_str(),
                          std::ios::out | std::ios::binary | std::ios::trunc);
          local_file << get_object_outcome_moved.GetResult().GetBody().rdbuf();
          MAPD_S3_LOG(LOG(INFO)
                      << "downloaded s3://" << bucket_name << "/" << objkey << " to "
                      << (use_pipe ? "pipe " : "file ") << file_path << ".")
        } catch (...) {
          // need this way to capture any exception occurring when
          // this thread runs as a disjoint asynchronous thread
          if (use_pipe) {
            teptr = std::current_exception();
          } else {
            throw;
          }
        }
      });

  if (use_pipe) {
    // in async (pipe) case, this function needs to wait for get_object_outcome
    // to be moved before it can exits; otherwise, the move() above will boom!!
    while (!is_get_object_outcome_moved) {
      std::this_thread::yield();
    }

    // no more detach this thread b/c detach thread is not possible to terminate
    // safely. when sanity test exits and glog is destructed too soon, the LOG(INFO)
    // above may still be holding glog rwlock while glog dtor tries to destruct the lock,
    // this causing a race, though unlikely this race would happen in production env.
    threads.push_back(std::move(th_writer));
    // join is delayed to ~S3Archive; any exception happening to rdbuf()
    // is passed to the upstream Importer th_pipe_writer thread via teptr.
  } else {
    try {
      th_writer.join();
    } catch (...) {
      throw;
    }
  }

  file_paths.insert(std::pair<const std::string, const std::string>(objkey, file_path));
  return file_path;
}

void S3Archive::vacuum(const std::string& objkey) {
  auto it = file_paths.find(objkey);
  if (file_paths.end() == it) {
    return;
  }
  boost::filesystem::remove(it->second);
  file_paths.erase(it);
}
