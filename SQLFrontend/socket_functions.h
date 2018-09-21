/*
 * Copyright 2018 OmniSci, Inc.
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

#ifndef SOCKET_FNC_H
#define SOCKET_FNC_H
#include <thrift/transport/TBufferTransports.h>
#include <string>
#include "Shared/mapd_shared_ptr.h"

mapd::shared_ptr<::apache::thrift::transport::TTransport> openBufferedClientTransport(
    const std::string& server_host,
    const int port,
    const std::string& ca_cert_name);

mapd::shared_ptr<::apache::thrift::transport::TTransport> openHttpClientTransport(
    const std::string& server_host,
    const int port,
    const std::string& trust_cert_file,
    const std::string& trust_cert_dir,
    bool use_https,
    bool skip_verify);

#endif
