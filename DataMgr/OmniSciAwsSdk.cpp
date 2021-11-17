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

#include "OmniSciAwsSdk.h"

#include <arrow/filesystem/s3fs.h>
#include <arrow/status.h>
#include <boost/filesystem.hpp>

#ifdef ARROW_HAS_PRIVATE_AWS_SDK
#include <aws/core/Aws.h>
#endif

#include "Logger/Logger.h"

#ifdef ARROW_HAS_PRIVATE_AWS_SDK
static Aws::SDKOptions awsapi_options;
#endif

void omnisci_aws_sdk::init_sdk() {
  auto ssl_config = omnisci_aws_sdk::get_ssl_config();
  arrow::fs::FileSystemGlobalOptions global_options;
  global_options.tls_ca_dir_path = ssl_config.ca_path;
  global_options.tls_ca_file_path = ssl_config.ca_file;
  arrow::fs::Initialize(global_options);
  arrow::fs::S3GlobalOptions s3_global_options;
  auto status = arrow::fs::InitializeS3(s3_global_options);
  CHECK(status.ok()) << "InitializeS3 resulted in an error: " << status.message();
#ifdef ARROW_HAS_PRIVATE_AWS_SDK
  // Directly initialize the AWS SDK, if Arrow uses a private version of the SDK
  Aws::InitAPI(awsapi_options);
#endif
}

void omnisci_aws_sdk::shutdown_sdk() {
  auto status = arrow::fs::FinalizeS3();
  CHECK(status.ok()) << "FinalizeS3 resulted in an error: " << status.message();
#ifdef ARROW_HAS_PRIVATE_AWS_SDK
  // Directly shutdown the AWS SDK, if Arrow uses a private version of the SDK
  Aws::ShutdownAPI(awsapi_options);
#endif
}

using omnisci_aws_sdk::SslConfig;
SslConfig omnisci_aws_sdk::get_ssl_config() {
  SslConfig ssl_config;
  /*
    Fix a wrong ca path established at building libcurl on Centos being carried to
    Ubuntu. To fix the issue, this is this sequence of locating ca file: 1) if
    `SSL_CERT_DIR` or `SSL_CERT_FILE` is set, set it to S3 ClientConfiguration. 2) if
    none ^ is set, omnisci_server searches a list of known ca file paths. 3) if 2)
    finds nothing, it is users' call to set correct SSL_CERT_DIR or SSL_CERT_FILE. S3
    c++ sdk: "we only want to override the default path if someone has explicitly told
    us to."
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
    ssl_config.ca_path = env;
  }
  if (nullptr != (env = getenv("SSL_CERT_FILE"))) {
    v_known_ca_paths.push_front(env);
  }
  for (const auto& known_ca_path : v_known_ca_paths) {
    if (boost::filesystem::exists(known_ca_path)) {
      ssl_config.ca_file = known_ca_path;
      break;
    }
  }
  return ssl_config;
}
