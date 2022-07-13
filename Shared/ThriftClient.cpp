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

#include "Shared/ThriftClient.h"
#ifdef HAVE_THRIFT_MESSAGE_LIMIT
#include "Shared/ThriftConfig.h"
#endif

#include <iostream>
#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/filesystem.hpp>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/THttpClient.h>
#include <thrift/transport/TSocket.h>
#include "Shared/ThriftJSONProtocolInclude.h"

using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::protocol;
using Decision = AccessManager::Decision;

void check_standard_ca(std::string& ca_cert_file) {
  if (ca_cert_file.empty()) {
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
        ca_cert_file = known_ca_path;
        break;
      }
    }
  }
}

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
  ProxyTHttpClient(std::shared_ptr<TTransport> transport,
                   std::string host,
                   std::string path)
#ifdef HAVE_THRIFT_MESSAGE_LIMIT
      : THttpClient(transport, host, path, shared::default_tconfig()) {
  }
#else
      : THttpClient(transport, host, path) {
  }
#endif

  ProxyTHttpClient(std::string host, int port, std::string path)
#ifdef HAVE_THRIFT_MESSAGE_LIMIT
      : THttpClient(host, port, path, shared::default_tconfig()) {
  }
#else
      : THttpClient(host, port, path) {
  }
#endif

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

    constexpr static const char* CRLF = "\r\n";

    std::ostringstream h;
    h << "POST " << path_ << " HTTP/1.1" << CRLF << "Host: " << host_ << CRLF
      << "Content-Type: application/x-thrift" << CRLF << "Content-Length: " << len << CRLF
      << "Accept: application/x-thrift" << CRLF << "User-Agent: Thrift/"
      << THRIFT_PACKAGE_VERSION << " (C++/THttpClient)" << CRLF
      << "Connection: keep-alive" << CRLF;
    if (!cookies_.empty()) {
      std::string cookie = "Cookie:" + boost::algorithm::join(cookies_, ";");
      h << cookie << CRLF;
    }
    h << CRLF;

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
ThriftClientConnection::~ThriftClientConnection() {}
ThriftClientConnection::ThriftClientConnection(const std::string& server_host,
                                               const int port,
                                               const ThriftConnectionType conn_type,
                                               bool skip_host_verify,
                                               std::shared_ptr<TSSLSocketFactory> factory)
    : server_host_(server_host)
    , port_(port)
    , conn_type_(conn_type)
    , skip_host_verify_(skip_host_verify)
    , trust_cert_file_("") {
  if (factory && (conn_type_ == ThriftConnectionType::BINARY_SSL ||
                  conn_type_ == ThriftConnectionType::HTTPS)) {
    using_X509_store_ = true;
    factory_ = factory;
    factory_->ciphers("ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");
    if (skip_host_verify_) {
      factory_->access(
          std::shared_ptr<InsecureAccessManager>(new InsecureAccessManager()));
    }
  }
}

std::shared_ptr<TProtocol> ThriftClientConnection::get_protocol() {
  std::shared_ptr<apache::thrift::transport::TTransport> mytransport;
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
    return std::shared_ptr<TProtocol>(new TJSONProtocol(mytransport));
  } else {
    return std::shared_ptr<TProtocol>(new TBinaryProtocol(mytransport));
  }
}

std::shared_ptr<TTransport> ThriftClientConnection::open_buffered_client_transport(
    const std::string& server_host,
    const int port,
    const std::string& ca_cert_name,
    bool with_timeout,
    bool with_keepalive,
    unsigned connect_timeout,
    unsigned recv_timeout,
    unsigned send_timeout) {
  std::shared_ptr<TTransport> transport;

  if (!factory_ && !ca_cert_name.empty()) {
    // need to build a factory once for ssl conection
    factory_ =
        std::shared_ptr<TSSLSocketFactory>(new TSSLSocketFactory(SSLProtocol::SSLTLS));
    factory_->ciphers("ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");
    factory_->loadTrustedCertificates(ca_cert_name.c_str());
    factory_->authenticate(false);
    factory_->access(std::shared_ptr<InsecureAccessManager>(new InsecureAccessManager()));
  }
  if (!using_X509_store_ && ca_cert_name.empty()) {
#ifdef HAVE_THRIFT_MESSAGE_LIMIT
    const auto socket =
        std::make_shared<TSocket>(server_host, port, shared::default_tconfig());
#else
    const auto socket = std::make_shared<TSocket>(server_host, port);
#endif
    if (with_timeout) {
      socket->setKeepAlive(with_keepalive);
      socket->setConnTimeout(connect_timeout);
      socket->setRecvTimeout(recv_timeout);
      socket->setSendTimeout(send_timeout);
    }
#ifdef HAVE_THRIFT_MESSAGE_LIMIT
    transport = std::make_shared<TBufferedTransport>(socket, shared::default_tconfig());
#else
    transport = std::make_shared<TBufferedTransport>(socket);
#endif
  } else {
    std::shared_ptr<TSocket> secure_socket = factory_->createSocket(server_host, port);
    if (with_timeout) {
      secure_socket->setKeepAlive(with_keepalive);
      secure_socket->setConnTimeout(connect_timeout);
      secure_socket->setRecvTimeout(recv_timeout);
      secure_socket->setSendTimeout(send_timeout);
    }
#ifdef HAVE_THRIFT_MESSAGE_LIMIT
    transport = std::shared_ptr<TTransport>(
        new TBufferedTransport(secure_socket, shared::default_tconfig()));
#else
    transport = std::shared_ptr<TTransport>(new TBufferedTransport(secure_socket));
#endif
  }

  return transport;
}

std::shared_ptr<TTransport> ThriftClientConnection::open_http_client_transport(
    const std::string& server_host,
    const int port,
    const std::string& trust_cert_fileX,
    bool use_https,
    bool skip_verify) {
  trust_cert_file_ = trust_cert_fileX;
  check_standard_ca(trust_cert_file_);

  if (!factory_) {
    factory_ =
        std::shared_ptr<TSSLSocketFactory>(new TSSLSocketFactory(SSLProtocol::SSLTLS));
  }
  std::shared_ptr<TTransport> transport;
  std::shared_ptr<TTransport> socket;
  if (use_https) {
    if (skip_verify) {
      factory_->authenticate(false);
      factory_->access(
          std::shared_ptr<InsecureAccessManager>(new InsecureAccessManager()));
    }
    if (!using_X509_store_) {
      factory_->loadTrustedCertificates(trust_cert_file_.c_str());
    }
    socket = factory_->createSocket(server_host, port);
    // transport = std::shared_ptr<TTransport>(new THttpClient(socket,
    // server_host,
    // "/"));
    transport =
        std::shared_ptr<TTransport>(new ProxyTHttpClient(socket, server_host, "/"));
  } else {
    transport = std::shared_ptr<TTransport>(new ProxyTHttpClient(server_host, port, "/"));
  }
  return transport;
}
