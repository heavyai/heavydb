/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SAMLSERVER_H
#define SAMLSERVER_H

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <xercesc/util/Xerces_autoconf_config.hpp>
#include "Catalog/AuthMetadata.h"
#include "Shared/Restriction.h"
#include "Shared/heavyai_shared_mutex.h"

namespace opensaml {

namespace saml2md {
class MetadataProvider;
class EntityDescriptor;
}  // namespace saml2md

class SecurityPolicy;
class SecurityPolicyRule;
}  // namespace opensaml

namespace xmltooling {
class Credential;
}

class SamlServer {
 public:
  SamlServer();
  SamlServer(const AuthMetadata& authMetadata);
  ~SamlServer();
  void login(std::string& username,
             const std::string& saml_response_base64,
             Restrictions& restrictions);
  bool authenticate_user(std::string& user_name,
                         const std::string& saml_response_base64,
                         std::vector<std::string>& saml_roles,
                         Restrictions& restrictions,
                         std::optional<std::string>& default_db);
  bool inUse() const;
  bool isRoleSyncInUse() const;

 private:
  const AuthMetadata* authMetadata_;
  bool samlInUse_;
  std::unique_ptr<opensaml::saml2md::MetadataProvider> idp_metadata_;
  const opensaml::saml2md::EntityDescriptor* descriptor_ = nullptr;
  std::unique_ptr<opensaml::SecurityPolicy> policy_;
  mutable heavyai::shared_mutex policy_mutex_;
  std::vector<opensaml::SecurityPolicyRule*> rules_;
  const xmltooling::Credential* credential_ = nullptr;
  XMLCh* sp_url_ = nullptr;
};

#endif /* SAMLSERVER_H */
