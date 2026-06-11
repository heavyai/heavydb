/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <list>

#include "Catalog/DashboardDescriptor.h"
#include "Catalog/SysCatalog.h"
#include "Catalog/TableDescriptor.h"
#include "InternalSystemDataWrapper.h"

namespace foreign_storage {

class InternalCatalogDataWrapper : public InternalSystemDataWrapper {
 public:
  InternalCatalogDataWrapper();

  InternalCatalogDataWrapper(const int db_id, const ForeignTable* foreign_table);

 private:
  void initializeObjectsForTable(const std::string& table_name) override;

  void populateChunkBuffersForTable(
      const std::string& table_name,
      std::map<std::string, import_export::UnmanagedTypedImportBuffer*>& import_buffers)
      override;

  std::list<Catalog_Namespace::UserMetadata> users_;
  std::map<int32_t, std::vector<TableDescriptor>> tables_by_database_;
  std::map<int32_t, std::vector<DashboardDescriptor>> dashboards_by_database_;
  std::vector<ColumnDescriptor> columns_;
  std::vector<ObjectRoleDescriptor> object_permissions_;
  std::list<Catalog_Namespace::DBMetadata> databases_;
  std::set<std::string> roles_;
  std::map<std::string, std::vector<std::string>> user_names_by_role_;
};
}  // namespace foreign_storage
