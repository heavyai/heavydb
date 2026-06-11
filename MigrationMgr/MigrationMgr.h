/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "Catalog/Types.h"
#include "OSDependent/heavyai_locks.h"
#include "SqliteConnector/SqliteConnector.h"

namespace Catalog_Namespace {
class Catalog;
}

namespace migrations {

class MigrationMgr {
 public:
  static void migrateDateInDaysMetadata(
      const Catalog_Namespace::TableDescriptorMapById& table_descriptors_by_id,
      const int database_id,
      Catalog_Namespace::Catalog* cat,
      SqliteConnector& sqlite);

  static bool dropRenderGroupColumns(
      const Catalog_Namespace::TableDescriptorMapById& table_descriptors_by_id,
      Catalog_Namespace::Catalog* cat);

  static void executeRebrandMigration(const std::string& base_path);

  static void takeMigrationLock(const std::string& base_path);
  static void relaxMigrationLock();
  static bool migrationEnabled() { return migration_enabled_; }

  static void destroy() {
    if (migration_mutex_) {
      migration_mutex_->unlock();
      migration_mutex_.reset();
    }
  }

 private:
  static inline std::unique_ptr<heavyai::DistributedSharedMutex> migration_mutex_;
  static inline bool migration_enabled_{false};
};

}  // namespace migrations
