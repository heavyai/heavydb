/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "Logger/Logger.h"

enum class ClientProtocol { TCP, HTTP, Other };

struct ConnectionInfo {
  std::string address;
  ClientProtocol protocol;

  std::string toString() const {
    switch (protocol) {
      case ClientProtocol::TCP:
        return "tcp:" + address;
      case ClientProtocol::HTTP:
        return "http:" + address;
      case ClientProtocol::Other:
        return "Other";
    }
    UNREACHABLE();
    return "";
  }
};
