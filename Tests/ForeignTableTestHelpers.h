/*
 * Copyright 2025 HEAVY.AI, Inc.
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

#include <gtest/gtest.h>
#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "Catalog/SysCatalog.h"
#include "DataMgr/ForeignStorage/AbstractFileStorageDataWrapper.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapper.h"
#include "Geospatial/Compression.h"
#include "Shared/misc.h"

namespace foreign_storage {

struct FragmentBuffers {
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> buffer_containers;
  ChunkToBufferMap buffers;
  std::unique_ptr<ForeignStorageBuffer> delete_buffer;

  FragmentBuffers(const ChunkMetadataVector& meta_vec) {
    for (const auto& [key, meta] : meta_vec) {
      buffer_containers[key] = std::make_unique<ForeignStorageBuffer>();
      buffers[key] = buffer_containers[key].get();
      if (key.size() > 4 && key[4] == 1) {
        ChunkKey index_key = key;
        index_key[4] = 2;
        buffer_containers[index_key] = std::make_unique<ForeignStorageBuffer>();
        buffers[index_key] = buffer_containers[index_key].get();
      }
    }
  }

  FragmentBuffers(const std::vector<ChunkKey>& keys) {
    for (const auto& key : keys) {
      buffer_containers[key] = std::make_unique<ForeignStorageBuffer>();
      buffers[key] = buffer_containers[key].get();
    }
  }

  template <class T>
  std::tuple<size_t, T, T> getSizeFirstLast(const ChunkKey& key) {
    auto& buffer = at(key);
    auto typed_buffer = reinterpret_cast<T*>(buffer.getMemoryPtr());
    return {
        buffer.size(), typed_buffer[0], typed_buffer[(buffer.size() / sizeof(T)) - 1]};
  }

  ForeignStorageBuffer& at(const ChunkKey& key) {
    return *(shared::get_from_map(buffer_containers, key));
  }

  bool operator==(const FragmentBuffers& other) const {
    if (buffer_containers.size() != other.buffer_containers.size()) {
      return false;
    }
    for (const auto& [key, ptr] : buffer_containers) {
      if (auto it = other.buffer_containers.find(key);
          it == other.buffer_containers.end()) {
        return false;
      } else {
        if (ptr->size() != it->second->size()) {
          return false;
        } else {
          for (auto i = 0U; i < ptr->size(); ++i) {
            if (ptr->getMemoryPtr()[i] != it->second->getMemoryPtr()[i]) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }

  template <class T>
  void printBuffer(const ChunkKey& key) {
    auto& buffer = *(buffer_containers.at(key));
    auto buffer_ptr = (T*)(buffer.getMemoryPtr());
    for (size_t i = 0; i < buffer.size() / sizeof(T); ++i) {
      std::cerr << buffer_ptr[i] << ", ";
    }
    std::cerr << "\n";
  }

  template <class T>
  void printKey(const ChunkKey& key, size_t num_elems) {
    T x_val;
    const auto& buffer = buffers[key];
    std::cerr << "buffer size = " << buffer->size() << "\n";
    for (size_t offset = 0; offset < num_elems * sizeof(x_val); offset += sizeof(x_val)) {
      buffer->read((int8_t*)&x_val, sizeof(x_val), offset);
      std::cerr << x_val << ", ";
    }
    std::cerr << "\n";
  }

  void printAllKeys(size_t num_elems) {
    for (const auto& [key, buffer] : buffers) {
      const auto& type_info = buffer->getSqlType();
      const auto type = type_info.get_type();
      if (type == kINT) {
        printKey<int32_t>(key, num_elems);
      } else if (type == kDOUBLE) {
        printKey<double>(key, num_elems);
      } else {
        UNREACHABLE() << "Unknown type: " << toString(type);
      }
    }
  }

  std::vector<double> getDecompressedCoordsAt(const SQLTypeInfo& type,
                                              const ChunkKey& key) {
    auto& buffer = at(key);
    return *(Geospatial::decompress_coords<double, SQLTypeInfo>(
        type, buffer.getMemoryPtr(), buffer.size()));
  }

  ChunkMetadata getMetadata(const ChunkKey key) {
    return at(key).getEncoder()->getMetadata();
  }

  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> getMetadata() const {
    std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> meta_map;
    for (const auto& [key, ptr] : buffer_containers) {
      if (ptr->hasEncoder()) {
        // We skip keys that have no encoder (like index keys).
        meta_map[key] = std::make_shared<ChunkMetadata>(ptr->getEncoder()->getMetadata());
      }
    }
    return meta_map;
  }
};

std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> create_meta_map(
    const ChunkMetadataVector& meta_vec) {
  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> map;
  for (const auto& [key, meta] : meta_vec) {
    map[key] = meta;
  }
  return map;
}

ChunkMetadataVector map_to_vec(
    const std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& meta_map) {
  ChunkMetadataVector meta_vec;
  for (const auto& [key, meta] : meta_map) {
    meta_vec.emplace_back(key, meta);
  }
  return meta_vec;
}

std::unique_ptr<ForeignTable> create_foreign_table() {
  auto ft = std::make_unique<ForeignTable>();
  ft->tableId = -1;
  ft->shard = -1;
  ft->tableName = "temp_test";
  ft->userId = -1;
  ft->nColumns = -1;
  ft->isView = false;
  ft->viewSQL = "";
  ft->fragments = "";
  ft->fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
  ft->maxFragRows = 32000000;
  ft->maxChunkSize = -1;
  ft->fragPageSize = -1;
  ft->maxRows = -1;
  ft->partitions = "";
  ft->keyMetainfo = "";
  ft->fragmenter = nullptr;
  ft->nShards = -1;
  ft->shardedColumnId = -1;
  ft->sortedColumnId = -1;
  ft->persistenceLevel = Data_Namespace::MemoryLevel::CPU_LEVEL;
  ft->hasDeletedCol = true;
  ft->columnIdBySpi_ = {};
  ft->storageType = StorageType::FOREIGN_TABLE;
  ft->maxRollbackEpochs = DEFAULT_MAX_ROLLBACK_EPOCHS;
  ft->is_system_table = false;
  ft->is_in_memory_system_table = false;
  ft->mutex_ = std::make_shared<std::mutex>();
  return ft;
}

std::string json_from_map(const OptionsMap& map) {
  std::stringstream ss;
  ss << "{";
  std::string separator;
  for (const auto& [key, value] : map) {
    ss << separator << "\"" << key << "\": \"" << value << "\"";
    separator = ",";
  }
  ss << "}";
  return ss.str();
}

using ADW = AbstractFileStorageDataWrapper;
using FT = ForeignTable;

class ForeignDataWrapperUnitTest : public ::testing::Test {
 public:
  inline static const std::string db_name{"fdw_test_db"};

  inline static int32_t db_id_;
  inline static std::shared_ptr<Catalog_Namespace::Catalog> cat_ptr_;
  inline static Catalog_Namespace::SysCatalog* sys_cat_ptr_;

  inline static const SQLTypeInfo point_t =
      SQLTypeInfo(kPOINT, 0, 0, false, kENCODING_GEOINT, 32, kNULLT);

  static void SetUpTestSuite() {
    sys_cat_ptr_ = &Catalog_Namespace::SysCatalog::instance();
    TearDownTestSuite();
    sys_cat_ptr_->createDatabase(db_name, shared::kRootUserId);
    cat_ptr_ = sys_cat_ptr_->getCatalog(db_name);
    db_id_ = cat_ptr_->getDatabaseId();
  }

  static void TearDownTestSuite() {
    Catalog_Namespace::DBMetadata db;
    if (sys_cat_ptr_->getMetadataForDB(db_name, db)) {
      sys_cat_ptr_->dropDatabase(db);
    }
  }

  void TearDown() override {
    if (foreign_table_) {
      cat_ptr_->dropTable(foreign_table_.get());
    }
    wrapper_ = nullptr;
    foreign_table_ = nullptr;
    user_mapping_ = nullptr;
  }

  // These column descriptors need to be dynamically constructed
  // because the db_id_ is not known at compile time.
  static std::list<ColumnDescriptor> createPointIndexSchema() {
    return {ColumnDescriptor(0, 0, "index", kINT, db_id_),
            ColumnDescriptor(0, 0, "p", point_t, db_id_)};
  }

  static std::list<ColumnDescriptor> createPointSchema() {
    return {ColumnDescriptor(0, 0, "p", point_t, db_id_)};
  }

  static std::list<ColumnDescriptor> createPointIndexExtraSchema() {
    return {ColumnDescriptor(0, 0, "index", kINT, db_id_),
            ColumnDescriptor(0, 0, "p", point_t, db_id_),
            ColumnDescriptor(0, 0, "extra", kINT, db_id_)};
  }

  static std::list<ColumnDescriptor> createDoubleSchema() {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "index", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "lon", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "lat", kDOUBLE, db_id_));
    return columns;
  }

  static std::list<ColumnDescriptor> createIdxPointSpacePointSchema() {
    return {ColumnDescriptor(0, 0, "index", kINT, db_id_),
            ColumnDescriptor(0, 0, "p1", point_t, db_id_),
            ColumnDescriptor(0, 0, "space", kINT, db_id_),
            ColumnDescriptor(0, 0, "p2", point_t, db_id_)};
  }

  virtual std::string getServerName() const = 0;

  virtual std::unique_ptr<foreign_storage::ForeignDataWrapper>
  createWrapperPtr(int32_t db_id, ForeignTable* ft, UserMapping* um) const = 0;

  void createWrapper(const std::string& file_name,
                     const std::list<ColumnDescriptor>& columns,
                     const OptionsMap& extra_options = {{}}) {
    OptionsMap options;
    options[ADW::FILE_PATH_KEY] = file_name;
    // Refresh options are not used in unit testing, but are required for a valid foreign
    // table.
    options[FT::REFRESH_TIMING_TYPE_KEY] = FT::MANUAL_REFRESH_TIMING_TYPE;
    options[FT::REFRESH_UPDATE_TYPE_KEY] = FT::ALL_REFRESH_UPDATE_TYPE;
    options["THREADS"] = "1";

    for (auto& [key, val] : extra_options) {
      options[key] = val;
    }

    foreign_table_ = create_foreign_table();
    foreign_table_->populateOptionsMap(json_from_map(options));
    foreign_table_->foreign_server = cat_ptr_->getForeignServer(getServerName());

    cat_ptr_->createTable(*foreign_table_, columns, {}, true);

    user_mapping_ = nullptr;
    wrapper_ = createWrapperPtr(db_id_, foreign_table_.get(), user_mapping_.get());

    // These validation steps would usually happen in the catalog during table creation.
    wrapper_->validateServerOptions(foreign_table_->foreign_server);
    wrapper_->validateTableOptions(foreign_table_.get());
    wrapper_->validateSchema(columns, foreign_table_.get());
  }

  ChunkMetadataVector populateChunkMetadata() {
    ChunkMetadataVector meta_vec;
    wrapper_->populateChunkMetadata(meta_vec);
    return meta_vec;
  }

  std::unique_ptr<ForeignTable> foreign_table_;
  std::unique_ptr<UserMapping> user_mapping_;
  std::unique_ptr<ForeignDataWrapper> wrapper_;
};
}  // namespace foreign_storage
