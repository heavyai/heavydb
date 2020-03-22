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

#pragma once

#include "LockMgr/LegacyLockMgr.h"
#include "LockMgr/LockMgrImpl.h"

#include <map>
#include <memory>
#include <string>

#include "Catalog/Catalog.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/types.h"

namespace lockmgr {

/**
 * @brief Locks protecting a physical table object returned from the catalog.
 * Table Metadata Locks prevent incompatible concurrent operations on table objects.
 * For example, before dropping or altering a table, a metadata write lock must be
 * acquired. This prevents concurrent read + drop, concurrent drops, etc.
 */
class TableSchemaLockMgr : public TableLockMgrImpl<TableSchemaLockMgr> {
 public:
  static TableSchemaLockMgr& instance() {
    static TableSchemaLockMgr table_lock_mgr;
    return table_lock_mgr;
  }

 private:
  TableSchemaLockMgr() {}
};

/**
 * @brief Prevents simultaneous inserts into the same table.
 * To allow concurrent Insert/Select queries, Insert queries only obtain a write lock on
 * table data when checkpointing (flushing chunks to disk). Inserts/Data load will take an
 * exclusive (write) lock to ensure only one insert proceeds on each table at a time.
 */
class InsertDataLockMgr : public TableLockMgrImpl<InsertDataLockMgr> {
 public:
  static InsertDataLockMgr& instance() {
    static InsertDataLockMgr insert_data_lock_mgr;
    return insert_data_lock_mgr;
  }

 protected:
  InsertDataLockMgr() {}
};

/**
 * @brief Locks protecting table data.
 * Read queries take a read lock, while write queries (update, delete) obtain a write
 * lock. Note that insert queries do not currently take a write lock (to allow concurrent
 * inserts). Instead, insert queries obtain a write lock on the table metadata to allow
 * existing read queries to finish (and block new ones) before flushing the inserted data
 * to disk.
 */
class TableDataLockMgr : public TableLockMgrImpl<TableDataLockMgr> {
 public:
  static TableDataLockMgr& instance() {
    static TableDataLockMgr data_lock_mgr;
    return data_lock_mgr;
  }

 protected:
  TableDataLockMgr() {}
};

class TableLockContainerImpl {
  std::string getTableName() const { return table_name_; }

 protected:
  TableLockContainerImpl(const std::string& table_name) : table_name_(table_name) {}

  std::string table_name_;
};

template <typename LOCK_TYPE>
class TableSchemaLockContainer
    : public LockContainerImpl<const TableDescriptor*, LOCK_TYPE>,
      public TableLockContainerImpl {
  static_assert(std::is_same<LOCK_TYPE, ReadLock>::value ||
                std::is_same<LOCK_TYPE, WriteLock>::value);

 public:
  TableSchemaLockContainer(const TableSchemaLockContainer&) = delete;  // non-copyable
};

template <>
class TableSchemaLockContainer<ReadLock>
    : public LockContainerImpl<const TableDescriptor*, ReadLock>,
      public TableLockContainerImpl {
 public:
  static auto acquireTableDescriptor(const Catalog_Namespace::Catalog& cat,
                                     const std::string& table_name,
                                     const bool populate_fragmenter = true) {
    return TableSchemaLockContainer<ReadLock>(
        cat.getMetadataForTable(table_name, populate_fragmenter),
        TableSchemaLockMgr::getReadLockForTable(cat, table_name));
  }

  static auto acquireTableDescriptor(const Catalog_Namespace::Catalog& cat,
                                     const int table_id) {
    const auto td = cat.getMetadataForTable(table_id);
    if (!td) {
      throw std::runtime_error("Table/View ID " + std::to_string(table_id) +
                               " does not exist.");
    }
    return TableSchemaLockContainer<ReadLock>(
        td, TableSchemaLockMgr::getReadLockForTable(cat, td->tableName));
  }

 private:
  TableSchemaLockContainer<ReadLock>(const TableDescriptor* obj, ReadLock&& lock)
      : LockContainerImpl<const TableDescriptor*, ReadLock>(obj, std::move(lock))
      , TableLockContainerImpl(obj ? obj->tableName : "") {}
};

template <>
class TableSchemaLockContainer<WriteLock>
    : public LockContainerImpl<const TableDescriptor*, WriteLock>,
      public TableLockContainerImpl {
 public:
  static auto acquireTableDescriptor(const Catalog_Namespace::Catalog& cat,
                                     const std::string& table_name,
                                     const bool populate_fragmenter = true) {
    return TableSchemaLockContainer<WriteLock>(
        cat.getMetadataForTable(table_name, populate_fragmenter),
        TableSchemaLockMgr::getWriteLockForTable(cat, table_name));
  }

  static auto acquireTableDescriptor(const Catalog_Namespace::Catalog& cat,
                                     const int table_id) {
    const auto td = cat.getMetadataForTable(table_id);
    if (!td) {
      throw std::runtime_error("Table/View ID " + std::to_string(table_id) +
                               " does not exist.");
    }
    return TableSchemaLockContainer<WriteLock>(
        td, TableSchemaLockMgr::getWriteLockForTable(cat, td->tableName));
  }

 private:
  TableSchemaLockContainer<WriteLock>(const TableDescriptor* obj, WriteLock&& lock)
      : LockContainerImpl<const TableDescriptor*, WriteLock>(obj, std::move(lock))
      , TableLockContainerImpl(obj ? obj->tableName : "") {}
};

template <typename LOCK_TYPE>
class TableDataLockContainer
    : public LockContainerImpl<const TableDescriptor*, LOCK_TYPE>,
      public TableLockContainerImpl {
  static_assert(std::is_same<LOCK_TYPE, ReadLock>::value ||
                std::is_same<LOCK_TYPE, WriteLock>::value);

 public:
  TableDataLockContainer(const TableDataLockContainer&) = delete;  // non-copyable
};
template <>
class TableDataLockContainer<WriteLock>
    : public LockContainerImpl<const TableDescriptor*, WriteLock>,
      public TableLockContainerImpl {
 public:
  static auto acquire(const int db_id, const TableDescriptor* td) {
    CHECK(td);
    ChunkKey chunk_key{db_id, td->tableId};
    return TableDataLockContainer<WriteLock>(
        td, TableDataLockMgr::getWriteLockForTable(chunk_key));
  }

 private:
  TableDataLockContainer<WriteLock>(const TableDescriptor* obj, WriteLock&& lock)
      : LockContainerImpl<const TableDescriptor*, WriteLock>(obj, std::move(lock))
      , TableLockContainerImpl(obj->tableName) {}
};

template <>
class TableDataLockContainer<ReadLock>
    : public LockContainerImpl<const TableDescriptor*, ReadLock>,
      public TableLockContainerImpl {
 public:
  static auto acquire(const int db_id, const TableDescriptor* td) {
    CHECK(td);
    ChunkKey chunk_key{db_id, td->tableId};
    return TableDataLockContainer<ReadLock>(
        td, TableDataLockMgr::getReadLockForTable(chunk_key));
  }

 private:
  TableDataLockContainer<ReadLock>(const TableDescriptor* obj, ReadLock&& lock)
      : LockContainerImpl<const TableDescriptor*, ReadLock>(obj, std::move(lock))
      , TableLockContainerImpl(obj->tableName) {}
};

using LockedTableDescriptors =
    std::vector<std::unique_ptr<lockmgr::AbstractLockContainer<const TableDescriptor*>>>;

}  // namespace lockmgr
