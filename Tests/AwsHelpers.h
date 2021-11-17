/*
 * Copyright 2021 OmniSci, Inc.
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
#include <aws/sts/STSClient.h>
#include <aws/sts/model/Credentials.h>
#include <aws/sts/model/GetSessionTokenRequest.h>
#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "Catalog/ForeignServer.h"

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
