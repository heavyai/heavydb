/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file    RemoteHostInfo.h
 * @brief   Remote host connection info for HA master failover.
 *
 */

#ifndef REMOTEHOSTINFO_H
#define REMOTEHOSTINFO_H

#include <cstdint>
#include <string>

class RemoteHostInfo {
 public:
  static constexpr bool default_with_keepalive = true;
  static constexpr unsigned default_connect_timeout = 20000;
  static constexpr unsigned default_recv_timeout = 300000;
  static constexpr unsigned default_send_timeout = 300000;

  explicit RemoteHostInfo(const std::string& host,
                          const uint16_t port,
                          const bool with_keepalive = default_with_keepalive,
                          const unsigned connect_timeout = default_connect_timeout,
                          const unsigned recv_timeout = default_recv_timeout,
                          const unsigned send_timeout = default_send_timeout,
                          const std::string& ca_cert = "")
      : host_(host)
      , port_(port)
      , with_keepalive_(with_keepalive)
      , connect_timeout_(connect_timeout)
      , recv_timeout_(recv_timeout)
      , send_timeout_(send_timeout)
      , ca_ref_cert_(ca_cert) {}

  const std::string& getHost() const { return host_; }

  uint16_t getPort() const { return port_; }

  const bool getWithKeepAlive() const { return with_keepalive_; }
  const unsigned getConnectTimeout() const { return connect_timeout_; }
  const unsigned getRecvTimeout() const { return recv_timeout_; }
  const unsigned getSendTimeout() const { return send_timeout_; }
  const std::string& getCACertFile() const { return ca_ref_cert_; }

 private:
  std::string host_;
  uint16_t port_;
  bool with_keepalive_;
  unsigned connect_timeout_;
  unsigned recv_timeout_;
  unsigned send_timeout_;
  std::string ca_ref_cert_;
};

#endif  // REMOTEHOSTINFO_H
