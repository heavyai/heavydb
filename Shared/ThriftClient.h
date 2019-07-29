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

#ifndef THRIFTCLIENT_H
#define THRIFTCLIENT_H
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/THttpClient.h>
#include <thrift/transport/TSSLSocket.h>
#include <thrift/transport/TSocket.h>
#include <string>
#include "Shared/mapd_shared_ptr.h"

using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

enum class ThriftConnectionType { HTTPS, HTTP, BINARY, BINARY_SSL };

class ThriftClientConnection {
 public:
  ThriftClientConnection(const std::string& server_host,
                         const int port,
                         const ThriftConnectionType conn_type,
                         bool skip_host_verify,
                         const std::string& ca_cert_name,
                         const std::string& trust_cert_file)
      : server_host_(server_host)
      , port_(port)
      , conn_type_(conn_type)
      , skip_host_verify_(skip_host_verify)
      , ca_cert_name_(ca_cert_name)
      , trust_cert_file_(trust_cert_file){};

  ThriftClientConnection(const std::string& ca_cert_name);
  ThriftClientConnection(){};

  mapd::shared_ptr<TTransport> open_buffered_client_transport(
      const std::string& server_host,
      const int port,
      const std::string& ca_cert_name,
      const bool with_timeout = false,
      const unsigned connect_timeout = 0,
      const unsigned recv_timeount = 0,
      const unsigned send_timeout = 0);

  mapd::shared_ptr<TTransport> open_http_client_transport(
      const std::string& server_host,
      const int port,
      const std::string& trust_cert_file_,
      bool use_https,
      bool skip_verify);

  mapd::shared_ptr<TProtocol> get_protocol();

 private:
  std::string server_host_;
  int port_;
  ThriftConnectionType conn_type_;
  bool skip_host_verify_;
  std::string ca_cert_name_;
  std::string trust_cert_file_;

  mapd::shared_ptr<TSSLSocketFactory> factory_;
};

#endif  // THRIFTCLIENT_H
