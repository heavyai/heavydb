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

#ifdef ARROW_HAS_PRIVATE_AWS_SDK
#include <aws/core/Aws.h>
#endif

#include "Logger/Logger.h"

#ifdef ARROW_HAS_PRIVATE_AWS_SDK
static Aws::SDKOptions awsapi_options;
#endif

void omnisci_aws_sdk::init_sdk() {
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
