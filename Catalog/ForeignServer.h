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

#include <string>
#include <unordered_map>

#include "Catalog/OptionsContainer.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapperFactory.h"

namespace foreign_storage {
struct ForeignServer : public OptionsContainer {
  int32_t id;
  std::string name;
  std::string data_wrapper_type;
  int32_t user_id;
  time_t creation_time;

  ForeignServer() {}

  ForeignServer(const int32_t server_id,
                const std::string& server_name,
                const std::string& data_wrapper_type,
                const std::string& options_str,
                const int32_t user_id,
                const time_t creation_time)
      : OptionsContainer(options_str)
      , id(server_id)
      , name(server_name)
      , data_wrapper_type(data_wrapper_type)
      , user_id(user_id)
      , creation_time(creation_time) {}

  ForeignServer(const std::string& server_name,
                const std::string& data_wrapper_type,
                const std::map<std::string, std::string, std::less<>>& options,
                const int32_t user_id)
      : OptionsContainer(options)
      , name(server_name)
      , data_wrapper_type(data_wrapper_type)
      , user_id(user_id) {}

  void validate() {
    ForeignDataWrapperFactory::validateDataWrapperType(data_wrapper_type);
    validateStorageParameters();
  }

 private:
  void validateStorageParameters() {
    ForeignDataWrapperFactory::createForValidation(data_wrapper_type)
        .validateServerOptions(this);
  }
};
}  // namespace foreign_storage
