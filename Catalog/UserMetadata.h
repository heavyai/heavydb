/*
 * Copyright 2022 OmniSci, Inc.
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

#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "Shared/Restriction.h"

namespace Catalog_Namespace {

/*
 * @type UserMetadata
 * @brief metadata for a db user
 */
struct UserMetadata {
  UserMetadata(int32_t u,
               const std::string& n,
               const std::string& p,
               bool s,
               int32_t d,
               bool l,
               bool t)
      : userId(u)
      , userName(n)
      , passwd_hash(p)
      , isSuper(s)
      , defaultDbId(d)
      , can_login(l)
      , is_temporary(t) {}
  UserMetadata() {}
  UserMetadata(UserMetadata const& user_meta)
      : UserMetadata(user_meta.userId,
                     user_meta.userName,
                     user_meta.passwd_hash,
                     user_meta.isSuper.load(),
                     user_meta.defaultDbId,
                     user_meta.can_login,
                     user_meta.is_temporary) {
    restriction = user_meta.restriction;
  }
  UserMetadata& operator=(UserMetadata const& user_meta) {
    if (this != &user_meta) {
      userId = user_meta.userId;
      userName = user_meta.userName;
      passwd_hash = user_meta.passwd_hash;
      isSuper.store(user_meta.isSuper.load());
      defaultDbId = user_meta.defaultDbId;
      can_login = user_meta.can_login;
      is_temporary = user_meta.is_temporary;
      restriction = user_meta.restriction;
    }
    return *this;
  }
  int32_t userId;
  std::string userName;
  std::string passwd_hash;
  std::atomic<bool> isSuper{false};
  int32_t defaultDbId;
  bool can_login{true};
  bool is_temporary{false};
  Restriction restriction;

  // Return a string that is safe to log for the username based on --log-user-id.
  std::string userLoggable() const;

  void setRestriction(Restriction in_restriction) { restriction = in_restriction; }
};

}  // namespace Catalog_Namespace