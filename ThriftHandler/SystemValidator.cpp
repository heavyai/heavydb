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

#include "SystemValidator.h"

namespace system_validator {
std::string SingleNodeValidator::validate() const {
  std::ostringstream validation_result;
  const auto tables = catalog_.getAllTableMetadata();
  for (const auto& table : tables) {
    if (should_validate_epoch(table)) {
      const auto table_epochs =
          catalog_.getTableEpochs(catalog_.getDatabaseId(), table->tableId);
      validation_result << validate_table_epochs(table_epochs, table->tableName, false);
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
    const std::string& table_name,
    const bool is_cluster_validation) {
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
                      << std::left;
    // Only add "Node" header for cluster validation
    if (is_cluster_validation) {
      validation_result << std::setw(10) << "Node";
    }
    validation_result << std::setw(10) << "Table Id" << std::setw(10) << "Epoch"
                      << "\n========= ========= ";
    // Add separator for "Node" header if this is a cluster validation
    if (is_cluster_validation) {
      validation_result << "========= ";
    }
    for (const auto& table_epoch : table_epochs) {
      validation_result << "\n";
      // Only add leaf index for cluster validation
      if (is_cluster_validation) {
        validation_result << std::setw(10)
                          << ("Leaf " + std::to_string(table_epoch.leaf_index));
      }
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
