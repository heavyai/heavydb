/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file    LeafHostInfo.h
 * @brief   Information about leaf nodes and utilities to parse a cluster configuration
 * file.
 *
 */

#ifndef LEAFHOSTINFO_H
#define LEAFHOSTINFO_H

#include <string>
#include <vector>

enum class NodeRole { NA };

class LeafHostInfo {
 public:
  LeafHostInfo(const std::string& host, const uint16_t port, const NodeRole role) {}

  const std::string& getHost() const { return host_; }

  uint16_t getPort() const { return port_; }

  NodeRole getRole() const { return role_; }

  unsigned getConnectTimeout(unsigned connect_timeout) { return 0; }
  unsigned getRecvTimeout(unsigned recv_timeout) { return 0; }
  unsigned getSendTimeout() { return 0; }
  std::string& getSSLCertFile() { return host_; }
  static std::vector<LeafHostInfo> parseClusterConfig(const std::string& file_path,
                                                      const unsigned connect_timeout,
                                                      const unsigned recv_timeout,
                                                      const unsigned send_timeout,
                                                      const std::string& ca_cert) {
    return std::vector<LeafHostInfo>{};
  };

 private:
  std::string host_;
  uint16_t port_;
  NodeRole role_;
};

#endif  // LEAFHOSTINFO_H
