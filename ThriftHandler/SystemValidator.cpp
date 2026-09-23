/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SystemValidator.h"

namespace system_validator {
std::string SingleNodeValidator::validate() const {
  std::ostringstream validation_result;
  const auto tables = catalog_.getAllTableMetadata();
  for (const auto& table : tables) {
    if (should_validate_epoch(table)) {
      const auto table_epochs =
          catalog_.getTableEpochs(catalog_.getDatabaseId(), table->tableId);
      validation_result << validate_table_epochs(table_epochs, table->tableName);
    }
  }

  if (validation_result.str().length() > 0) {
    return validation_result.str();
  } else {
    return "Instance OK";
  }
}

bool should_validate_epoch(const TableDescriptor* table_descriptor) {
  // Epoch validation only applies to persisted local tables. Validation uses the logical
  // table id to validate epoch consistency across shards.
  return (table_descriptor->shard == -1 && !table_descriptor->isForeignTable() &&
          !table_descriptor->isTemporaryTable() && !table_descriptor->isView);
}

std::string validate_table_epochs(
    const std::vector<Catalog_Namespace::TableEpochInfo>& table_epochs,
    const std::string& table_name) {
  std::ostringstream validation_result;
  CHECK(!table_epochs.empty());
  bool epochs_are_inconsistent{false};
  auto first_epoch = table_epochs[0].table_epoch;
  for (const auto& table_epoch : table_epochs) {
    if (first_epoch != table_epoch.table_epoch) {
      epochs_are_inconsistent = true;
      break;
    }
  }

  if (epochs_are_inconsistent) {
    validation_result << "\nEpoch values for table \"" << table_name
                      << "\" are inconsistent:\n"
                      << std::left << std::setw(10) << "Table Id" << std::setw(10)
                      << "Epoch"
                      << "\n========= =========\n";
    for (const auto& table_epoch : table_epochs) {
      validation_result << "\n";
      validation_result << std::setw(10) << table_epoch.table_id << std::setw(10)
                        << table_epoch.table_epoch;
    }
    validation_result << "\n";
  } else if (first_epoch < 0) {
    validation_result << "\nNegative epoch value found for table \"" << table_name
                      << "\". Epoch: " << first_epoch << ".";
  }

  return validation_result.str();
}
}  // namespace system_validator
