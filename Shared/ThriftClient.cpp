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
#include "Shared/ThriftClient.h"
#include <boost/algorithm/string.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>

using namespace ::apache::thrift::transport;
using Decision = AccessManager::Decision;

class InsecureAccessManager : public AccessManager {
 public:
  Decision verify(const sockaddr_storage& sa) throw() override {
    boost::ignore_unused(sa);
    return ALLOW;
  };
  Decision verify(const std::string& host, const char* name, int size) throw() override {
    boost::ignore_unused(host);
    boost::ignore_unused(name);
    boost::ignore_unused(size);
    return ALLOW;
  };
  Decision verify(const sockaddr_storage& sa,
                  const char* data,
                  int size) throw() override {
    boost::ignore_unused(sa);
    boost::ignore_unused(data);
    boost::ignore_unused(size);
    return ALLOW;
  };
};

/*
 * The Http client that comes with Thrift constructs a very simple set of HTTP
 * headers, ignoring cookies.  This class simply inherits from THttpClient to
 * override the two methods - parseHeader (where it collects any cookies) and
 * flush where it inserts the cookies into the http header.
 *
 * The methods that are over ridden here are virtual in the parent class, as is
 * the parents class's destructor.
 *
 */
class ProxyTHttpClient : public THttpClient {
 public:
  // mimic and call the super constructors.
  ProxyTHttpClient(mapd::shared_ptr<TTransport> transport,
                   std::string host,
                   std::string path)
      : THttpClient(transport, host, path) {}

  ProxyTHttpClient(std::string host, int port, std::string path)
      : THttpClient(host, port, path) {}

  ~ProxyTHttpClient() override {}
  // thrift parseHeader d and call the super constructor.
  void parseHeader(char* header) override {
    //  note boost::istarts_with is case insensitive
    if (boost::istarts_with(header, "set-cookie:")) {
      std::string tmp(header);
      std::string cookie = tmp.substr(tmp.find(":") + 1, std::string::npos);
      cookies_.push_back(cookie);
    }
    THttpClient::parseHeader(header);
  }

  void flush() override {
    /*
     * Unfortunately the decision to write the header and the body in the same
     * method precludes using the parent class's flush method here; in what is
     * effectively a copy of 'flush' in THttpClient with the addition of
     * cookies, a better error report for a header that is too large and
     * 'Connection: keep-alive'.
     */
    uint8_t* buf;
    uint32_t len;
    writeBuffer_.getBuffer(&buf, &len);

    std::ostringstream h;
    h << "POST " << path_ << " HTTP/1.1" << THttpClient::CRLF << "Host: " << host_
      << THttpClient::CRLF << "Content-Type: application/x-thrift" << THttpClient::CRLF
      << "Content-Length: " << len << THttpClient::CRLF << "Accept: application/x-thrift"
      << THttpClient::CRLF << "User-Agent: Thrift/" << THRIFT_PACKAGE_VERSION
      << " (C++/THttpClient)" << THttpClient::CRLF << "Connection: keep-alive"
      << THttpClient::CRLF;
    if (!cookies_.empty()) {
      std::string cookie = "Cookie:" + boost::algorithm::join(cookies_, ";");
      h << cookie << THttpClient::CRLF;
    }
    h << THttpClient::CRLF;

    cookies_.clear();
    std::string header = h.str();
    if (header.size() > (std::numeric_limits<uint32_t>::max)()) {
      throw TTransportException(
          "Header too big [" + std::to_string(header.size()) +
          "]. Max = " + std::to_string((std::numeric_limits<uint32_t>::max)()));
    }
    // Write the header, then the data, then flush
    transport_->write((const uint8_t*)header.c_str(),
                      static_cast<uint32_t>(header.size()));
    transport_->write(buf, len);
    transport_->flush();

    // Reset the buffer and header variables
    writeBuffer_.resetBuffer();
    readHeaders_ = true;
  }

  std::vector<std::string> cookies_;
};

ThriftClientConnection::ThriftClientConnection(const std::string& ca_cert_name) {
  if (!ca_cert_name.empty()) {
    factory_ =
        mapd::shared_ptr<TSSLSocketFactory>(new TSSLSocketFactory(SSLProtocol::SSLTLS));
    factory_->ciphers("ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");

    factory_->loadTrustedCertificates(ca_cert_name.c_str());
    factory_->authenticate(false);
    factory_->access(
        mapd::shared_ptr<InsecureAccessManager>(new InsecureAccessManager()));
  }
}

mapd::shared_ptr<TProtocol> ThriftClientConnection::get_protocol() {
  mapd::shared_ptr<apache::thrift::transport::TTransport> mytransport;
  if (conn_type_ == ThriftConnectionType::HTTP ||
      conn_type_ == ThriftConnectionType::HTTPS) {
    mytransport = open_http_client_transport(server_host_,
                                             port_,
                                             ca_cert_name_,
                                             conn_type_ == ThriftConnectionType::HTTPS,
                                             skip_host_verify_);

  } else {
    mytransport = open_buffered_client_transport(server_host_, port_, ca_cert_name_);
  }

  try {
    mytransport->open();
  } catch (const apache::thrift::TException& e) {
    throw apache::thrift::TException(std::string(e.what()) + ": host " + server_host_ +
                                     ", port " + std::to_string(port_));
  }
  if (conn_type_ == ThriftConnectionType::HTTP ||
      conn_type_ == ThriftConnectionType::HTTPS) {
    return mapd::shared_ptr<TProtocol>(new TJSONProtocol(mytransport));
  } else {
    return mapd::shared_ptr<TProtocol>(new TBinaryProtocol(mytransport));
  }
};

mapd::shared_ptr<TTransport> ThriftClientConnection::open_buffered_client_transport(
    const std::string& server_host,
    const int port,
    const std::string& ca_cert_name,
    bool with_timeout,
    unsigned connect_timeout,
    unsigned recv_timeout,
    unsigned send_timeout) {
  mapd::shared_ptr<TTransport> transport;

  if (!factory_ && !ca_cert_name.empty()) {
    // need to build a factory once for ssl conection
    factory_ =
        mapd::shared_ptr<TSSLSocketFactory>(new TSSLSocketFactory(SSLProtocol::SSLTLS));
    factory_->ciphers("ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");
    factory_->loadTrustedCertificates(ca_cert_name.c_str());
    factory_->authenticate(false);
    factory_->access(
        mapd::shared_ptr<InsecureAccessManager>(new InsecureAccessManager()));
  }
  if (ca_cert_name.empty()) {
    const auto socket = mapd::make_shared<TSocket>(server_host, port);
    if (with_timeout) {
      socket->setConnTimeout(connect_timeout);
      socket->setRecvTimeout(recv_timeout);
      socket->setSendTimeout(send_timeout);
    }
    transport = mapd::make_shared<TBufferedTransport>(socket);
  } else {
    mapd::shared_ptr<TSocket> secure_socket = factory_->createSocket(server_host, port);
    if (with_timeout) {
      secure_socket->setConnTimeout(connect_timeout);
      secure_socket->setRecvTimeout(recv_timeout);
      secure_socket->setSendTimeout(send_timeout);
    }
    transport = mapd::shared_ptr<TTransport>(new TBufferedTransport(secure_socket));
  }
  return transport;
}

mapd::shared_ptr<TTransport> ThriftClientConnection::open_http_client_transport(
    const std::string& server_host,
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

  if (!factory_) {
    factory_ = mapd::shared_ptr<TSSLSocketFactory>(new TSSLSocketFactory());
  }
  mapd::shared_ptr<TTransport> transport;
  mapd::shared_ptr<TTransport> socket;
  if (use_https) {
    if (skip_verify) {
      factory_->authenticate(false);
      factory_->access(
          mapd::shared_ptr<InsecureAccessManager>(new InsecureAccessManager()));
    }
    factory_->loadTrustedCertificates(trust_cert_file.c_str());
    socket = factory_->createSocket(server_host, port);
    // transport = mapd::shared_ptr<TTransport>(new THttpClient(socket,
    // server_host,
    // "/"));
    transport =
        mapd::shared_ptr<TTransport>(new ProxyTHttpClient(socket, server_host, "/"));
  } else {
    transport =
        mapd::shared_ptr<TTransport>(new ProxyTHttpClient(server_host, port, "/"));
  }
  return transport;
}
