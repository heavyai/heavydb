/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Catalog/TableDescriptor.h"

namespace foreign_storage {
void inline validate_non_foreign_table_write(const TableDescriptor* table_descriptor) {
  if (table_descriptor && table_descriptor->storageType == StorageType::FOREIGN_TABLE) {
    throw std::runtime_error{
        "DELETE, INSERT, TRUNCATE, OR UPDATE commands are not supported for foreign "
        "tables."};
  }
}
}  // namespace foreign_storage
