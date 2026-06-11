/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * File:   PkiServer.h
 *
 */

#pragma once

#include "Catalog/AuthMetadata.h"

#include <string>

class PkiServer {
 public:
  PkiServer() {}
  PkiServer(const AuthMetadata& authMetadata) {}
  bool validate_certificate(const std::string& pki_cert, std::string& common_name) {
    return false;
  }
  bool encrypt_session(const std::string& pki_cert, std::string& session) {
    return false;
  }
  bool inUse() { return false; }
};
