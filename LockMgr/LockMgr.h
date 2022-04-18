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
#include "Shared/heavyai_shared_mutex.h"
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

template <typename LOCK_TYPE>
class TableSchemaLockContainer
    : public LockContainerImpl<const TableDescriptor*, LOCK_TYPE> {
  static_assert(std::is_same<LOCK_TYPE, ReadLock>::value ||
                std::is_same<LOCK_TYPE, WriteLock>::value);

 public:
  TableSchemaLockContainer(const TableSchemaLockContainer&) = delete;  // non-copyable
};

inline void validate_table_descriptor_after_lock(const TableDescriptor* td_prelock,
                                                 const Catalog_Namespace::Catalog& cat,
                                                 const std::string& table_name,
                                                 const bool populate_fragmenter) {
  auto td_postlock = cat.getMetadataForTable(table_name, populate_fragmenter);
  if (td_prelock != td_postlock) {
    if (td_postlock == nullptr) {
      throw std::runtime_error("Table/View ID " + table_name + " for catalog " +
                               cat.getCurrentDB().dbName + " does not exist");
    } else {
      // This should be very unusual case where a table has moved
      // read DROP, CREATE kind of pattern
      // but kept same name
      // it is not safe to proceed here as the locking was based on the old
      // chunk attributes of the table, which could belong to a different table now
      throw std::runtime_error("Table/View ID " + table_name + " for catalog " +
                               cat.getCurrentDB().dbName +
                               " changed whilst attempting to acquire table lock");
    }
  }
}

template <>
class TableSchemaLockContainer<ReadLock>
    : public LockContainerImpl<const TableDescriptor*, ReadLock> {
 public:
  static auto acquireTableDescriptor(const Catalog_Namespace::Catalog& cat,
                                     const std::string& table_name,
                                     const bool populate_fragmenter = true) {
    VLOG(1) << "Acquiring Table Schema Read Lock for table: " << table_name;
    auto ret = TableSchemaLockContainer<ReadLock>(
        cat.getMetadataForTable(table_name, populate_fragmenter),
        TableSchemaLockMgr::getReadLockForTable(cat, table_name));
    validate_table_descriptor_after_lock(ret(), cat, table_name, populate_fragmenter);
    return ret;
  }

  static auto acquireTableDescriptor(const Catalog_Namespace::Catalog& cat,
                                     const int table_id) {
    const auto table_name = cat.getTableName(table_id);
    if (!table_name.has_value()) {
      throw std::runtime_error("Table/View ID " + std::to_string(table_id) +
                               " for catalog " + cat.getCurrentDB().dbName +
                               " does not exist. Cannot aquire read lock");
    }
    return acquireTableDescriptor(cat, table_name.value());
  }

 private:
  TableSchemaLockContainer<ReadLock>(const TableDescriptor* obj, ReadLock&& lock)
      : LockContainerImpl<const TableDescriptor*, ReadLock>(obj, std::move(lock)) {}
};

template <>
class TableSchemaLockContainer<WriteLock>
    : public LockContainerImpl<const TableDescriptor*, WriteLock> {
 public:
  static auto acquireTableDescriptor(const Catalog_Namespace::Catalog& cat,
                                     const std::string& table_name,
                                     const bool populate_fragmenter = true) {
    VLOG(1) << "Acquiring Table Schema Write Lock for table: " << table_name;
    auto ret = TableSchemaLockContainer<WriteLock>(
        cat.getMetadataForTable(table_name, populate_fragmenter),
        TableSchemaLockMgr::getWriteLockForTable(cat, table_name));
    validate_table_descriptor_after_lock(ret(), cat, table_name, populate_fragmenter);
    return ret;
  }

  static auto acquireTableDescriptor(const Catalog_Namespace::Catalog& cat,
                                     const int table_id) {
    const auto table_name = cat.getTableName(table_id);
    if (!table_name.has_value()) {
      throw std::runtime_error("Table/View ID " + std::to_string(table_id) +
                               " for catalog " + cat.getCurrentDB().dbName +
                               " does not exist. Cannot aquire write lock");
    }
    return acquireTableDescriptor(cat, table_name.value());
  }

 private:
  TableSchemaLockContainer<WriteLock>(const TableDescriptor* obj, WriteLock&& lock)
      : LockContainerImpl<const TableDescriptor*, WriteLock>(obj, std::move(lock)) {}
};

template <typename LOCK_TYPE>
class TableDataLockContainer
    : public LockContainerImpl<const TableDescriptor*, LOCK_TYPE> {
  static_assert(std::is_same<LOCK_TYPE, ReadLock>::value ||
                std::is_same<LOCK_TYPE, WriteLock>::value);

 public:
  TableDataLockContainer(const TableDataLockContainer&) = delete;  // non-copyable
};

template <>
class TableDataLockContainer<WriteLock>
    : public LockContainerImpl<const TableDescriptor*, WriteLock> {
 public:
  static auto acquire(const int db_id, const TableDescriptor* td) {
    CHECK(td);
    ChunkKey chunk_key{db_id, td->tableId};
    VLOG(1) << "Acquiring Table Data Write Lock for table: " << td->tableName;
    return TableDataLockContainer<WriteLock>(
        td, TableDataLockMgr::getWriteLockForTable(chunk_key));
  }

 private:
  TableDataLockContainer<WriteLock>(const TableDescriptor* obj, WriteLock&& lock)
      : LockContainerImpl<const TableDescriptor*, WriteLock>(obj, std::move(lock)) {}
};

template <>
class TableDataLockContainer<ReadLock>
    : public LockContainerImpl<const TableDescriptor*, ReadLock> {
 public:
  static auto acquire(const int db_id, const TableDescriptor* td) {
    CHECK(td);
    ChunkKey chunk_key{db_id, td->tableId};
    VLOG(1) << "Acquiring Table Data Read Lock for table: " << td->tableName;
    return TableDataLockContainer<ReadLock>(
        td, TableDataLockMgr::getReadLockForTable(chunk_key));
  }

 private:
  TableDataLockContainer<ReadLock>(const TableDescriptor* obj, ReadLock&& lock)
      : LockContainerImpl<const TableDescriptor*, ReadLock>(obj, std::move(lock)) {}
};

template <typename LOCK_TYPE>
class TableInsertLockContainer
    : public LockContainerImpl<const TableDescriptor*, LOCK_TYPE> {
  static_assert(std::is_same<LOCK_TYPE, ReadLock>::value ||
                std::is_same<LOCK_TYPE, WriteLock>::value);

 public:
  TableInsertLockContainer(const TableInsertLockContainer&) = delete;  // non-copyable
};

template <>
class TableInsertLockContainer<WriteLock>
    : public LockContainerImpl<const TableDescriptor*, WriteLock> {
 public:
  static auto acquire(const int db_id, const TableDescriptor* td) {
    CHECK(td);
    ChunkKey chunk_key{db_id, td->tableId};
    VLOG(1) << "Acquiring Table Insert Write Lock for table: " << td->tableName;
    return TableInsertLockContainer<WriteLock>(
        td, InsertDataLockMgr::getWriteLockForTable(chunk_key));
  }

 private:
  TableInsertLockContainer<WriteLock>(const TableDescriptor* obj, WriteLock&& lock)
      : LockContainerImpl<const TableDescriptor*, WriteLock>(obj, std::move(lock)) {}
};

template <>
class TableInsertLockContainer<ReadLock>
    : public LockContainerImpl<const TableDescriptor*, ReadLock> {
 public:
  static auto acquire(const int db_id, const TableDescriptor* td) {
    CHECK(td);
    ChunkKey chunk_key{db_id, td->tableId};
    VLOG(1) << "Acquiring Table Insert Read Lock for table: " << td->tableName;
    return TableInsertLockContainer<ReadLock>(
        td, InsertDataLockMgr::getReadLockForTable(chunk_key));
  }

 private:
  TableInsertLockContainer<ReadLock>(const TableDescriptor* obj, ReadLock&& lock)
      : LockContainerImpl<const TableDescriptor*, ReadLock>(obj, std::move(lock)) {}
};

using LockedTableDescriptors =
    std::vector<std::unique_ptr<lockmgr::AbstractLockContainer<const TableDescriptor*>>>;

}  // namespace lockmgr
