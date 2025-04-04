/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/sts/STSClient.h>
#include <aws/sts/model/Credentials.h>
#include <aws/sts/model/GetSessionTokenRequest.h>
#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "Catalog/ForeignServer.h"

#ifdef HEAVYAI_S3_HAS_S3CLIENT_CONFIGURATION
#include <aws/s3/S3ClientConfiguration.h>
#include <aws/s3/S3EndpointProvider.h>
#endif

bool is_valid_aws_key(std::pair<std::string, std::string> key) {
  return (key.first.size() > 0) && (key.second.size() > 0);
}

bool is_valid_aws_key(const std::map<std::string, std::string>& env_vars) {
  return env_vars.find("AWS_ACCESS_KEY_ID") != env_vars.end() &&
         env_vars.find("AWS_ACCESS_KEY_ID")->second.size() > 0 &&
         env_vars.find("AWS_SECRET_ACCESS_KEY") != env_vars.end() &&
         env_vars.find("AWS_SECRET_ACCESS_KEY")->second.size() > 0;
}

bool is_valid_aws_role() {
  Aws::Auth::InstanceProfileCredentialsProvider instance_provider;
  return (instance_provider.GetAWSCredentials().GetAWSAccessKeyId().size() > 0) &&
         (instance_provider.GetAWSCredentials().GetAWSSecretKey().size() > 0) &&
         (instance_provider.GetAWSCredentials().GetSessionToken().size() > 0);
}

// Extract AWS key from environment (either environment variables or user profile file)
std::pair<std::string, std::string> get_aws_keys_from_env() {
  std::string user_key;
  std::string secret_key;
  Aws::Auth::EnvironmentAWSCredentialsProvider env_provider;

  user_key = env_provider.GetAWSCredentials().GetAWSAccessKeyId();
  secret_key = env_provider.GetAWSCredentials().GetAWSSecretKey();

  if (!is_valid_aws_key(std::make_pair(user_key, secret_key))) {
    auto file_provider =
        Aws::Auth::ProfileConfigFileAWSCredentialsProvider("omnisci_test");
    user_key = file_provider.GetAWSCredentials().GetAWSAccessKeyId();
    secret_key = file_provider.GetAWSCredentials().GetAWSSecretKey();
  }

  return {user_key, secret_key};
}

Aws::STS::Model::Credentials generate_sts_credentials(
    const std::pair<std::string, std::string>& aws_keys,
    const Aws::Client::ClientConfiguration& client_config,
    int32_t session_token_duration_seconds = 900) {
  const auto credentials =
      Aws::Auth::SimpleAWSCredentialsProvider(aws_keys.first, aws_keys.second)
          .GetAWSCredentials();
  CHECK(!credentials.IsExpiredOrEmpty())
      << "Failed to get AWSCredentials which are expired or empty";

  /* Session Tokens created by IAM users range from 900s to 129,600s.
   * Session Tokens created by AWS account owners range from 900s to 3,600s
   * If the duration_seconds is > 3600s for an AWS account owner, then
   * duration_seconds is defaulted to 3,600s */
  Aws::STS::Model::GetSessionTokenRequest session_token_request;
  session_token_request.SetDurationSeconds(
      std::min(129600, std::max(900, session_token_duration_seconds)));
  return Aws::STS::STSClient(credentials, client_config)
      .GetSessionToken(session_token_request)
      .GetResult()
      .GetCredentials();
}

namespace {
const std::set<std::string> AWS_ENV_KEYS_AND_PROFILE = {"AWS_ACCESS_KEY_ID",
                                                        "AWS_SECRET_ACCESS_KEY",
                                                        "AWS_SESSION_TOKEN",
                                                        "AWS_SHARED_CREDENTIALS_FILE",
                                                        "AWS_PROFILE"};

const std::set<std::string> AWS_ENV_KEYS = {"AWS_ACCESS_KEY_ID",
                                            "AWS_SECRET_ACCESS_KEY",
                                            "AWS_SESSION_TOKEN"};

std::map<std::string, std::string> unset_env_set(
    const std::set<std::string>& env_var_names) {
  std::map<std::string, std::string> env_vars;
  for (const auto& name : env_var_names) {
    env_vars.emplace(name, getenv(name.c_str()) ? std::string(getenv(name.c_str())) : "");
    unsetenv(name.c_str());
  }
  return env_vars;
}

void restore_pair(const std::pair<std::string, std::string>& pair) {
  if (pair.second == "") {
    unsetenv(pair.first.c_str());
  } else {
    setenv(pair.first.c_str(), pair.second.c_str(), 1);
  }
}

void restore_env_vars(const std::map<std::string, std::string>& env_vars,
                      const std::set<std::string>& whitelist = std::set<std::string>()) {
  for (const auto& pair : env_vars) {
    if (whitelist.empty()) {
      restore_pair(pair);
    } else if (!whitelist.empty() && whitelist.find(pair.first) != whitelist.end()) {
      restore_pair(pair);
    }
  }
}

Aws::S3::S3Client create_s3_client_for_region(const std::string& aws_region) {
  const auto [access_key, secret_key] = get_aws_keys_from_env();
#ifdef HEAVYAI_S3_HAS_S3CLIENT_CONFIGURATION
  Aws::S3::S3ClientConfiguration s3_client_config;
  s3_client_config.region = aws_region;
  auto endpoint_provider = foreign_storage::get_endpoint_provider(s3_client_config);
  Aws::S3::S3Client s3_client(Aws::Auth::AWSCredentials(access_key, secret_key),
                              std::move(endpoint_provider),
                              s3_client_config);
#else
  Aws::Client::ClientConfiguration client_config;
  client_config.region = aws_region;
  Aws::S3::S3Client s3_client(Aws::Auth::AWSCredentials(access_key, secret_key),
                              client_config,
                              Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
                              true);
#endif
  return s3_client;
}
}  // namespace

std::map<std::string, std::string> unset_aws_env() {
  return unset_env_set(AWS_ENV_KEYS_AND_PROFILE);
}
std::map<std::string, std::string> unset_aws_keys() {
  return unset_env_set(AWS_ENV_KEYS);
}
void restore_aws_env(const std::map<std::string, std::string>& env_vars) {
  restore_env_vars(env_vars);
}
void restore_aws_keys(const std::map<std::string, std::string>& env_vars) {
  restore_env_vars(env_vars, AWS_ENV_KEYS);
}

void create_stub_aws_profile(const std::string& aws_credentials_dir) {
  // If a valid profile is not required, a dummy profile is still required so that AWS
  // profiles along the default paths are not included
  setenv(
      "AWS_SHARED_CREDENTIALS_FILE", (aws_credentials_dir + "/credentials").c_str(), 1);
  setenv("AWS_PROFILE", "omnisci_test", 1);
  boost::filesystem::create_directory(aws_credentials_dir);
  std::ofstream credentials_file(aws_credentials_dir + "/credentials");
  credentials_file << "[omnisci_test]\n";
  credentials_file.close();
}

void set_aws_profile(const std::string& aws_credentials_dir,
                     const bool use_valid_profile,
                     const std::map<std::string, std::string>& env_vars =
                         std::map<std::string, std::string>()) {
  std::ofstream credentials_file(aws_credentials_dir + "/credentials");
  credentials_file << "[omnisci_test]\n";
  if (use_valid_profile) {
    CHECK(is_valid_aws_key(env_vars))
        << "Sufficent private credentials required for creating an authorized AWS "
           "profile";
    std::string aws_access_key_id("");
    std::string aws_secret_access_key("");
    std::string aws_session_token("");
    if (env_vars.find("AWS_ACCESS_KEY_ID") != env_vars.end()) {
      aws_access_key_id = env_vars.find("AWS_ACCESS_KEY_ID")->second;
    }
    if (env_vars.find("AWS_SECRET_ACCESS_KEY") != env_vars.end()) {
      aws_secret_access_key = env_vars.find("AWS_SECRET_ACCESS_KEY")->second;
    }
    if (env_vars.find("AWS_SESSION_TOKEN") != env_vars.end()) {
      aws_session_token = env_vars.find("AWS_SESSION_TOKEN")->second;
    }
    credentials_file << "aws_access_key_id = " << aws_access_key_id << "\n";
    credentials_file << "aws_secret_access_key = " << aws_secret_access_key << "\n";
    credentials_file << "aws_session_token = " << aws_session_token << "\n";
  }
  credentials_file.close();
}

void upload_file_to_s3(const std::string& s3_bucket_name,
                       const std::string& local_file_path,
                       const std::string& s3_object_key,
                       const std::string& aws_region) {
  Aws::S3::Model::PutObjectRequest request;
  request.SetBucket(s3_bucket_name);
  request.SetKey(s3_object_key);

  auto file_data = Aws::MakeShared<Aws::FStream>(
      "S3UploadFile", local_file_path.c_str(), std::ios_base::in | std::ios_base::binary);
  if (!file_data || !(*file_data)) {
    throw std::runtime_error{"An error occurred when attempting to read file: " +
                             local_file_path};
  }
  request.SetBody(file_data);

  auto s3_client = create_s3_client_for_region(aws_region);
  auto outcome = s3_client.PutObject(request);
  if (!outcome.IsSuccess()) {
    throw std::runtime_error{
        "An error occurred when attempting to upload file to S3. Error message: " +
        outcome.GetError().GetMessage() + ", S3 bucket name: " + s3_bucket_name +
        ", local file path: " + local_file_path + ", S3 object key: " + s3_object_key +
        ", aws region: " + aws_region};
  }
}

void delete_object_keys_with_prefix(const std::string& s3_bucket_name,
                                    const std::string& s3_object_keys_prefix,
                                    const std::string& aws_region) {
  auto s3_client = create_s3_client_for_region(aws_region);

  Aws::S3::Model::ListObjectsRequest list_objects_request;
  list_objects_request.WithBucket(s3_bucket_name);
  list_objects_request.WithPrefix(s3_object_keys_prefix);

  std::vector<std::string> object_keys;
  auto list_outcome = s3_client.ListObjects(list_objects_request);
  if (list_outcome.IsSuccess()) {
    for (const auto& object : list_outcome.GetResult().GetContents()) {
      object_keys.emplace_back(object.GetKey());
    }
  } else {
    throw std::runtime_error{
        "An error occurred when attempting to list S3 objects. Error message: " +
        list_outcome.GetError().GetMessage() + ", S3 bucket name: " + s3_bucket_name +
        ", S3 object key prefix: " + s3_object_keys_prefix +
        ", aws region: " + aws_region};
  }

  Aws::S3::Model::DeleteObjectRequest delete_object_request;
  delete_object_request.WithBucket(s3_bucket_name);
  std::vector<std::pair<std::string, std::string>> object_keys_with_delete_errors;
  for (const auto& object_key : object_keys) {
    delete_object_request.WithKey(object_key);
    auto delete_outcome = s3_client.DeleteObject(delete_object_request);
    if (!delete_outcome.IsSuccess()) {
      object_keys_with_delete_errors.emplace_back(object_key,
                                                  delete_outcome.GetError().GetMessage());
    }
  }

  if (!object_keys_with_delete_errors.empty()) {
    std::string error_message{
        "An error occurred when attempting to delete the following object keys:\n"};
    for (const auto& [object_key, error] : object_keys_with_delete_errors) {
      error_message += "{ object key: " + object_key + ", error: " + error + "}\n";
    }
    error_message +=
        ", S3 bucket name: " + s3_bucket_name + ", aws region: " + aws_region;
    throw std::runtime_error{error_message};
  }
}
