/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LDAPSERVER_H
#define LDAPSERVER_H

#include "Catalog/AuthMetadata.h"

#include <ldap.h>
#include <string>
#include <vector>

class LdapServer {
 public:
  LdapServer();
  LdapServer(const AuthMetadata& authMetadata);
  bool authenticate_user(const std::string& userName,
                         const std::string& passwd,
                         std::vector<std::string>& ldap_user_roles);
  void login(const std::string& username, const std::string& password);
  bool inUse() const;
  bool isRoleSyncInUse() const;
  const std::string& get_superuser_rolename() const;

 private:
  const AuthMetadata* authMetadata_;
  bool ldapInUse;
};

#endif /* LDAPSERVER_H */
