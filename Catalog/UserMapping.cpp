/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "UserMapping.h"

#include "DataMgr/ForeignStorage/ForeignDataWrapperFactory.h"
#include "Shared/Encryption.h"
#include "Shared/JsonUtils.h"
#include "Shared/misc.h"

namespace foreign_storage {
UserMapping::UserMapping(const int32_t id,
                         const int32_t user_id,
                         const int32_t foreign_server_id,
                         const std::string type,
                         const std::string options)
    : id(id)
    , user_id(user_id)
    , foreign_server_id(foreign_server_id)
    , type(type)
    , options(options) {}

OptionsMap UserMapping::getUnencryptedOptions() const {
  const auto& plain_text = PkiEncryptor::privateKeyDecrypt(options);
  OptionsContainer options_container{};
  options_container.populateOptionsMap(plain_text);
  return options_container.options;
}

void UserMapping::validate(const ForeignServer* foreign_server) {
  const auto* data_wrapper =
      ForeignDataWrapperFactory::createForValidation(foreign_server->data_wrapper_type);
  const auto options = getUnencryptedOptions();
  const auto& supported_options = data_wrapper->getSupportedUserMappingOptions();
  for (const auto& entry : options) {
    if (!shared::contains(supported_options, entry.first)) {
      throw std::runtime_error{"Invalid user mapping option \"" + entry.first +
                               "\". Option must be one of the following: " +
                               join(supported_options, ", ") + "."};
    }
  }
  data_wrapper->validateUserMappingOptions(this, foreign_server);
}

void UserMapping::setOptions(const OptionsMap& options) {
  rapidjson::Document d;
  d.SetObject();

  for (const auto& [key, value] : options) {
    json_utils::add_value_to_object(d, value, key, d.GetAllocator());
  }
  auto options_str = json_utils::write_to_string(d);
  this->options = PkiEncryptor::publicKeyEncrypt(options_str);
}
}  // namespace foreign_storage
