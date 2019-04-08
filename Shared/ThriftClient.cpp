/*
 * Copyright 2018, OmniSci, Inc.
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

#include "ThriftClient.h"
#include <thrift/transport/THttpClient.h>
#include <thrift/transport/TSSLSocket.h>
#include <thrift/transport/TSocket.h>
#include <boost/filesystem.hpp>

using namespace ::apache::thrift::transport;
using Decision = AccessManager::Decision;

class InsecureAccessManager : public AccessManager {
 public:
  Decision verify(const sockaddr_storage& sa) throw() { return ALLOW; };
  Decision verify(const std::string& host, const char* name, int size) throw() {
    return ALLOW;
  };
  Decision verify(const sockaddr_storage& sa, const char* data, int size) throw() {
    return ALLOW;
  };
};

mapd::shared_ptr<TTransport> openBufferedClientTransport(
    const std::string& server_host,
    const int port,
    const std::string& ca_cert_name) {
  mapd::shared_ptr<TTransport> transport;
  if (ca_cert_name.empty()) {
    transport = mapd::shared_ptr<TTransport>(new TBufferedTransport(
        mapd::shared_ptr<TTransport>(new TSocket(server_host, port))));
  } else {
    // Thrift issue 4164 https://jira.apache.org/jira/browse/THRIFT-4164 reports a problem
    // if TSSLSocketFactory is destroyed before any sockets it creates are destroyed.
    // Making the factory static should ensure a safe destruction order.
    static mapd::shared_ptr<TSSLSocketFactory> factory =
        mapd::shared_ptr<TSSLSocketFactory>(new TSSLSocketFactory(SSLProtocol::SSLTLS));
    factory->ciphers("ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");
    factory->loadTrustedCertificates(ca_cert_name.c_str());
    factory->authenticate(false);
    factory->access(mapd::shared_ptr<InsecureAccessManager>(new InsecureAccessManager()));
    mapd::shared_ptr<TSocket> secure_socket = factory->createSocket(server_host, port);
    transport = mapd::shared_ptr<TTransport>(new TBufferedTransport(secure_socket));
  }
  return transport;
}

mapd::shared_ptr<TTransport> openHttpClientTransport(const std::string& server_host,
                                                     const int port,
                                                     const std::string& trust_cert_file_,
                                                     bool use_https,
                                                     bool skip_verify) {
  std::string trust_cert_file{trust_cert_file_};
  if (trust_cert_file_.empty()) {
    static std::list<std::string> v_known_ca_paths({
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/usr/share/ssl/certs/ca-bundle.crt",
        "/usr/local/share/certs/ca-root.crt",
        "/etc/ssl/cert.pem",
        "/etc/ssl/ca-bundle.pem",
    });
    for (const auto& known_ca_path : v_known_ca_paths) {
      if (boost::filesystem::exists(known_ca_path)) {
        trust_cert_file = known_ca_path;
        break;
      }
    }
  }

  mapd::shared_ptr<TTransport> transport;
  mapd::shared_ptr<TTransport> socket;
  if (use_https) {
    // Thrift issue 4164 https://jira.apache.org/jira/browse/THRIFT-4164 reports a problem
    // if TSSLSocketFactory is destroyed before any sockets it creates are destroyed.
    // Making the factory static should ensure a safe destruction order.
    static mapd::shared_ptr<TSSLSocketFactory> sslSocketFactory =
        mapd::shared_ptr<TSSLSocketFactory>(new TSSLSocketFactory());
    if (skip_verify) {
      sslSocketFactory->authenticate(false);
      sslSocketFactory->access(
          mapd::shared_ptr<InsecureAccessManager>(new InsecureAccessManager()));
    }
    sslSocketFactory->loadTrustedCertificates(trust_cert_file.c_str());
    socket = sslSocketFactory->createSocket(server_host, port);
    transport = mapd::shared_ptr<TTransport>(new THttpClient(socket, server_host, "/"));
  } else {
    transport = mapd::shared_ptr<TTransport>(new THttpClient(server_host, port, "/"));
  }
  return transport;
}
