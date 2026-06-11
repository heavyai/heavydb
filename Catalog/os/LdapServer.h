/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * File:   LdapServer.h
 *
 */

#ifndef LDAPSERVER_H
#define LDAPSERVER_H

#include "Catalog/AuthMetadata.h"

#include <string>

class LdapServer {
 public:
  LdapServer() {}
  LdapServer(const AuthMetadata& authMetadata) {}
  bool authenticate_user(const std::string& userName, const std::string& passwd) {
    return false;
  }
  bool inUse() { return false; }
};

#endif /* LDAPSERVER_H */
