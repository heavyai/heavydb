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
