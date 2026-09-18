/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "Catalog/CatalogFwd.h"

namespace foreign_storage {
void refresh_foreign_table_unlocked(Catalog_Namespace::Catalog& catalog,
                                    const ForeignTable& foreign_table,
                                    const bool evict_cached_entries);

void refresh_foreign_table(Catalog_Namespace::Catalog& catalog,
                           const std::string& table_name,
                           const bool evict_cached_entries);
}  // namespace foreign_storage
