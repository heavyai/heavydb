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
