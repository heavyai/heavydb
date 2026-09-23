/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AUTHMETADATA_H
#define AUTHMETADATA_H

#include <string>
struct AuthMetadata {
  AuthMetadata() {}
  int32_t port;
  std::string uri;
  std::string distinguishedName;
  std::string ldapQueryUrl;
  std::string ldapRoleRegex;
  std::string ldapSuperUserRole;
  std::string domainComp;
#ifdef HAVE_SAML
  std::string samlIdpMetadataFile;
  std::string samlSpTargetUrl;
  bool samlSyncRoles;
  bool samlSignedResponse;
  bool samlSignedAssertion;
#endif /* HAVE_SAML */
  bool allowLocalAuthFallback;
};

#endif /* AUTHMETADATA_H */
