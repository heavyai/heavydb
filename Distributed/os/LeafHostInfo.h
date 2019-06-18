/*
 * Copyright 2017 MapD Technologies, Inc.
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

/*
 * @file    LeafHostInfo.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Information about leaf nodes and utilities to parse a cluster configuration
 * file.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
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
