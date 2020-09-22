/*
 * Copyright 2020 OmniSci, Inc.
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

#include "ForeignServer.h"

#include <boost/algorithm/string/predicate.hpp>
#include "OptionsContainer.h"
#include "TableDescriptor.h"

namespace foreign_storage {
struct ForeignTable : public TableDescriptor, public OptionsContainer {
  const ForeignServer* foreign_server;
  static constexpr std::array<char const*, 2> supported_options{"FRAGMENT_SIZE",
                                                                "REFRESH_UPDATE_TYPE"};

  static bool isValidOption(const std::pair<std::string, std::string>& entry) {
    if (std::find(supported_options.begin(), supported_options.end(), entry.first) ==
        supported_options.end()) {
      // Not a valid option
      return false;
    }
    if (boost::iequals(entry.first, "REFRESH_UPDATE_TYPE") &&
        !(boost::iequals(entry.second, "ALL") ||
          boost::iequals(entry.second, "APPEND"))) {
      std::string error_message = "Invalid value \"" + entry.second +
                                  "\" for REFRESH_UPDATE_TYPE option." +
                                  " Value must be \"APPEND\" or \"ALL\".";
      throw std::runtime_error{error_message};
    }
    return true;
  }

  bool isAppendMode() const {
    auto update_mode = options.find("REFRESH_UPDATE_TYPE");
    return (update_mode != options.end()) &&
           boost::iequals(update_mode->second, "APPEND");
  }
};
}  // namespace foreign_storage
