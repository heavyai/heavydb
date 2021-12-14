/*
 * Copyright 2021 OmniSci, Inc.
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

/**
 * @file    file_type.cpp
 * @author  andrew.do@omnisci.com>
 * @brief   shared utility for mime-types
 *
 */

#include <regex>
#include <vector>

#include <boost/filesystem.hpp>

#include "Shared/file_type.h"

#include "Shared/misc.h"

namespace shared {

bool is_compressed_mime_type(const std::string& mime_type) {
  // strips possible vender (vnd.), personal (prs.) & unregistered (x.) prefixes from the
  // mime-type
  static const std::regex mime_prefix("\\/(vnd|prs|x)(\\.|-)");
  const auto mime_type_no_prefix = std::regex_replace(mime_type, mime_prefix, "/");
  static const std::vector<std::string> compressed_types = {"application/7z-compressed",
                                                            "application/bzip",
                                                            "application/bzip2",
                                                            "application/gzip",
                                                            "application/rar",
                                                            "application/tar",
                                                            "application/zip"};
  return shared::contains(compressed_types, mime_type_no_prefix);
}

bool is_compressed_file_extension(const std::string& location) {
  static const std::vector<std::string> compressed_exts = {
      ".7z", ".bz", ".bz2", ".gz", ".rar", ".tar", ".zip", ".tgz"};
  return shared::contains(compressed_exts, boost::filesystem::extension(location));
}

}  // namespace shared