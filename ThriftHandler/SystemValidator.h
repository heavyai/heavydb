/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Shared/measure.h>
#include "Catalog/Catalog.h"
#include "Logger/Logger.h"

#include "QueryState.h"

class DBHandler;

namespace system_validator {
std::string validate_table_epochs(
    const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs,
    const std::string& table_name);

bool should_validate_epoch(const TableDescriptor* table_descriptor);

/**
 * @brief Driver for running validation on a single node.
 */
class SingleNodeValidator {
 public:
  SingleNodeValidator(const std::string& type, Catalog_Namespace::Catalog& catalog)
      : catalog_(catalog) {
    if (!type.empty()) {
      throw std::runtime_error{
          "Unexpected validation type specified. Only the \"VALIDATE;\" command is "
          "currently supported."};
    }
  }

  std::string validate() const;

 private:
  Catalog_Namespace::Catalog& catalog_;
};
}  // namespace system_validator
