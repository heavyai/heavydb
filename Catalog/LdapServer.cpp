/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "LdapServer.h"

#include <lber.h>

#include <boost/regex.hpp>
#include "Catalog/Catalog.h"
#include "Logger/Logger.h"

namespace Catalog_Namespace {
extern bool g_log_user_id;
}

LdapServer::LdapServer() : authMetadata_(nullptr) {
  LOG(INFO) << "No LDAP server defined, will not attempt to authenticate via ldap";
  ldapInUse = false;
}

LdapServer::LdapServer(const AuthMetadata& authMetadata)
    : authMetadata_(&authMetadata), ldapInUse(false) {
  if (authMetadata.uri.empty()) {
    ldapInUse = false;
  } else {
    LOG(INFO) << "LDAP being used for Authentication, uri: " << authMetadata.uri
              << " DN: " << authMetadata.distinguishedName;
    ldapInUse = true;

    // Show a warning in case we can not initialize or connect to LDAP
    // (with invalid credentials)
    LDAP* ldp = nullptr;
    int rc;
    rc = ldap_initialize(&ldp, authMetadata_->uri.c_str());
    if (rc != LDAP_SUCCESS) {
      LOG(ERROR) << "ldap_initialize failed: " << ldap_err2string(rc);
    } else {
      char empty_string[]{};
      berval creds{0, empty_string};
      rc =
          ldap_sasl_bind_s(ldp, empty_string, nullptr, &creds, nullptr, nullptr, nullptr);
      if (rc != LDAP_INVALID_CREDENTIALS && rc != LDAP_SUCCESS) {
        LOG(WARNING) << "Failed to connect to LDAP server on startup: "
                     << ldap_err2string(rc);
      }
    }
    if (ldp) {
      ldap_unbind_ext_s(ldp, nullptr, nullptr);
    }
  }
}

bool LdapServer::inUse() const {
  return ldapInUse;
}

bool LdapServer::isRoleSyncInUse() const {
  return inUse() && !authMetadata_->ldapQueryUrl.empty();
}

const std::string& LdapServer::get_superuser_rolename() const {
  return authMetadata_->ldapSuperUserRole;
}

void LdapServer::login(const std::string& username, const std::string& password) {
  auto& syscat = Catalog_Namespace::SysCatalog::instance();
  std::vector<std::string> ldap_roles;
  if (!authenticate_user(username, password, ldap_roles)) {
    throw std::runtime_error("Invalid credentials.");
  }
  if (isRoleSyncInUse()) {
    Catalog_Namespace::UserAlterations alts;
    alts.is_super = false;
    auto su_role_iter =
        std::find(ldap_roles.begin(), ldap_roles.end(), get_superuser_rolename());
    if (su_role_iter != ldap_roles.end()) {
      alts.is_super = true;
      ldap_roles.erase(su_role_iter);
    }
    syscat.syncUserWithRemoteProvider(username, ldap_roles, alts);
  }
}

bool LdapServer::authenticate_user(const std::string& userName,
                                   const std::string& passwd,
                                   std::vector<std::string>& ldap_user_roles) {
  // do not allow bind attenpts with empty string
  // issue with AD Unauthenticated Authentication
  if (passwd.empty()) {
    throw std::runtime_error("Invalid credentials.");
  }

  LDAP* ldp;
  int rc, version;
  berval creds;

  boost::regex usernameRE("\\$USERNAME");
  std::string bind_dn =
      boost::regex_replace(authMetadata_->distinguishedName, usernameRE, userName);

  if (Catalog_Namespace::g_log_user_id) {
    LOG(INFO) << "User [username omitted by log-user-id] connecting";
  } else {
    LOG(INFO) << "User " << userName << " connecting as " << bind_dn;
  }

  /* Open LDAP Connection */
  /* Get a handle to an LDAP connection. */
  rc = ldap_initialize(&ldp, authMetadata_->uri.c_str());
  if (rc != LDAP_SUCCESS) {
    LOG(ERROR) << "ldap_initialize failed " << ldap_err2string(rc);
    throw std::runtime_error(ldap_err2string(rc));
  }
  version = LDAP_VERSION3;

  ldap_set_option(ldp, LDAP_OPT_PROTOCOL_VERSION, &version);
  /* User authentication (bind) */
  std::vector<char> writable_passwd(passwd.begin(), passwd.end());
  writable_passwd.push_back('\0');
  creds.bv_val = &writable_passwd[0];
  creds.bv_len = passwd.length();
  rc = ldap_sasl_bind_s(ldp, bind_dn.c_str(), nullptr, &creds, nullptr, nullptr, nullptr);
  if (rc != LDAP_SUCCESS) {
    ldap_unbind_ext_s(ldp, nullptr, nullptr);
    if (rc == LDAP_INVALID_CREDENTIALS) {
      return false;
    } else {
      throw std::runtime_error(ldap_err2string(rc));
    }
  }
  std::string const loggable =
      Catalog_Namespace::g_log_user_id ? std::string("") : userName + ' ';
  LOG(INFO) << " User " << loggable << "successfully logged in with LDAP authentication";

  if (isRoleSyncInUse()) {
    // code evolved from https://gist.github.com/syzdek/1470233

    /// @brief stores data for interactive SASL authentication
    using LDAPAuth = struct ldapexample_auth;
    struct ldapexample_auth {
      const char* dn;        ///< DN to use for simple bind
      const char* saslmech;  ///< SASL mechanism to use for authentication
      const char* authuser;  ///< user to authenticate
      const char* user;      ///< pre-authenticated user
      const char* realm;     ///< SASL realm used for authentication
      BerValue cred;         ///< the credentials of "user" (i.e. password)
    };

    /// @brief stores data for interactive SASL authentication

    using LDAPConfig = struct ldapexample_config;
    struct ldapexample_config {
      int verbose;
      const char* ldap_ca;
      int ldap_tls;
      int ldap_version;
      int search_limit;
      struct timeval search_timeout;
      struct timeval tcp_timeout;
      LDAPURLDesc* ludp;
      LDAPAuth auth;
    };

    LDAPConfig config;

    struct timeval* timeoutp;
    int msgid;
    int msgtype;
    int msgcount;
    int err;
    // char * errmsg;
    LDAPMessage* res;
    const char* attribute;
    BerElement* ber;
    BerValue** vals;
    int pos;

    // reset config data
    memset(&config, 0, sizeof(LDAPConfig));

    // add the search portion to the URI
    std::string fullURI =
        boost::regex_replace(authMetadata_->ldapQueryUrl, usernameRE, userName);

    config.ldap_version = LDAP_VERSION3;
    timeoutp = nullptr;

    if ((ldap_url_parse(fullURI.c_str(), &config.ludp))) {
      ldap_unbind_ext_s(ldp, nullptr, nullptr);
      throw std::runtime_error("Invalid LDAP query URL: " + fullURI);
    };

    ldap_search_ext(ldp,                      // LDAP            * ld
                    config.ludp->lud_dn,      // char            * base
                    config.ludp->lud_scope,   // int               scope
                    config.ludp->lud_filter,  // char            * filter
                    config.ludp->lud_attrs,   // char            * attrs[]
                    0,                        // int               attrsonly
                    nullptr,                  // LDAPControl    ** serverctrls
                    nullptr,                  // LDAPControl    ** clientctrls
                    timeoutp,                 // struct timeval  * timeout
                    config.search_limit,      // int               sizelimit
                    &msgid                    // int             * msgidp
    );

    // loops through results from search
    msgtype = LDAP_RES_SEARCH_ENTRY;
    for (msgcount = 0; msgtype != LDAP_RES_SEARCH_RESULT; msgcount++) {
      // retrieves result
      err = ldap_result(ldp, msgid, 0, timeoutp, &res);
      switch (err) {
        case -1:
          ldap_get_option(ldp, LDAP_OPT_RESULT_CODE, &err);
          ldap_unbind_ext_s(ldp, nullptr, nullptr);
          throw std::runtime_error(ldap_err2string(err));
        case 0:
          ldap_abandon_ext(ldp, msgid, nullptr, nullptr);
          ldap_unbind_ext_s(ldp, nullptr, nullptr);
          throw std::runtime_error("Timout expired for LDAP request");
        default:
          break;
      };

      // determines result type
      msgtype = ldap_msgtype(res);
      if (msgtype != LDAP_RES_SEARCH_ENTRY) {
        continue;
      }

      // loops through attributes and values
      attribute = ldap_first_attribute(ldp, res, &ber);
      while ((attribute)) {
        vals = ldap_get_values_len(ldp, res, attribute);
        for (pos = 0; pos < ldap_count_values_len(vals); pos++) {
          std::string s = vals[pos]->bv_val;
          boost::regex roleRegex(authMetadata_->ldapRoleRegex);
          boost::smatch result;
          if (boost::regex_search(s, result, roleRegex) && result.size() > 1) {
            std::string submatch(result[1].first, result[1].second);
            ldap_user_roles.push_back(submatch);
          }
        }

        ldap_value_free_len(vals);
        attribute = ldap_next_attribute(ldp, res, ber);
      };
      ber_free(ber, 0);

      // frees result
      ldap_memfree(res);
    };

    // parses search result
    ldap_parse_result(ldp, res, &err, nullptr, nullptr, nullptr, nullptr, 0);
    ldap_memfree(res);

    //
    //  ends connection and frees resources
    //
    ldap_free_urldesc(config.ludp);
  }

  ldap_unbind_ext_s(ldp, nullptr, nullptr);
  return true;
}
