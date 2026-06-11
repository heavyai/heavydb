/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SAMLSERVER_H
#define SAMLSERVER_H

#include "Catalog/AuthMetadata.h"

#include <string>

class SamlServer {
 public:
  SamlServer() {}
  SamlServer(const AuthMetadata& authMetadata) {}
  bool authenticate_user(const std::string& userName, const std::string& assertion) {
    return false;
  }
  bool inUse() { return false; }
};

#endif /* SAMLSERVER_H */
