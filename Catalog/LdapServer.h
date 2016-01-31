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
 * File:   LdapServer.h
 * Author: michael
 *
 * Created on January 26, 2016, 11:50 PM
 */

#ifndef LDAPSERVER_H
#define LDAPSERVER_H

#include <string>
#include <glog/logging.h>

/*
 * @type LdapMetadata
 * @brief ldap data for using ldap server for authentication
 */
struct LdapMetadata {
  LdapMetadata(const std::string& uri, const std::string& dn) : uri(uri), distinguishedName(dn) {}
  LdapMetadata() {}
  int32_t port;
  std::string uri;
  std::string distinguishedName;
  std::string domainComp;
};

class LdapServer {
 public:
  LdapServer(){};
  LdapServer(const LdapMetadata& ldapMetadata){};
  bool authenticate_user(const std::string& userName, const std::string& passwd) { return false; };
  bool inUse() { return false; };
};

#endif /* LDAPSERVER_H */
