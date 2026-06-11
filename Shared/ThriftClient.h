/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef THRIFTCLIENT_H
#define THRIFTCLIENT_H
#include <thrift/protocol/TProtocol.h>
#include <thrift/transport/TSSLSocket.h>
#include <string>
// TProtocol.h > winsock2.h > windows.h
#include "cleanup_global_namespace.h"

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

  ThriftClientConnection(const std::string& server_host,
                         const int port,
                         const ThriftConnectionType conn_type,
                         bool skip_host_verify,
                         std::shared_ptr<TSSLSocketFactory> factory);

  ThriftClientConnection(){};

  std::shared_ptr<TTransport> open_buffered_client_transport(
      const std::string& server_host,
      const int port,
      const std::string& ca_cert_name,
      const bool with_timeout = false,
      const bool with_keepalive = true,
      const unsigned connect_timeout = 0,
      const unsigned recv_timeount = 0,
      const unsigned send_timeout = 0);

  std::shared_ptr<TTransport> open_http_client_transport(
      const std::string& server_host,
      const int port,
      const std::string& trust_cert_file_,
      bool use_https,
      bool skip_verify);

  std::shared_ptr<TProtocol> get_protocol();
  virtual ~ThriftClientConnection();

 private:
  std::string server_host_;
  int port_;
  ThriftConnectionType conn_type_;
  bool skip_host_verify_;
  std::string ca_cert_name_;
  std::string trust_cert_file_;
  bool using_X509_store_ = false;
  std::shared_ptr<TSSLSocketFactory> factory_;
};

#endif  // THRIFTCLIENT_H
