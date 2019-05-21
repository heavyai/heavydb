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
  std::string restUrl;
  std::string restToken;
  bool allowLocalAuthFallback;
};

#endif /* AUTHMETADATA_H */
