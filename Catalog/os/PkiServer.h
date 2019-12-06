/*
 * Copyright 2019 OmniSci, Inc.
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
