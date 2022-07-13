/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "Catalog/CatalogFwd.h"

namespace foreign_storage {
void refresh_foreign_table_unlocked(Catalog_Namespace::Catalog& catalog,
                                    const ForeignTable& foreign_table,
                                    const bool evict_cached_entries);

void refresh_foreign_table(Catalog_Namespace::Catalog& catalog,
                           const std::string& table_name,
                           const bool evict_cached_entries);
}  // namespace foreign_storage
