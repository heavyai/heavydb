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

#include "DBEngine.h"
#include <boost/filesystem.hpp>
#include <boost/variant.hpp>
#include <iostream>
#include "Catalog/Catalog.h"
#include "Logger/Logger.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/sqltypes.h"

namespace EmbeddedDatabase {

enum class ColumnType : uint32_t { Unknown, Integer, Double, Float, String, Array };

/**
 * Cursor internal implementation
 */
class CursorImpl : public Cursor {
 public:
  CursorImpl(std::shared_ptr<ResultSet> result_set,
             std::shared_ptr<Data_Namespace::DataMgr> data_mgr)
      : result_set_(result_set), data_mgr_(data_mgr) {}

  size_t getColCount() { return result_set_->colCount(); }

  size_t getRowCount() { return result_set_->rowCount(); }

  Row getNextRow() {
    auto row = result_set_->getNextRow(true, false);
    if (row.empty()) {
      return Row();
    }
    return Row(row);
  }

  ColumnType getColType(uint32_t col_num) {
    if (col_num < getColCount()) {
      SQLTypeInfo type_info = result_set_->getColType(col_num);
      switch (type_info.get_type()) {
        case kNUMERIC:
        case kDECIMAL:
        case kINT:
        case kSMALLINT:
        case kBIGINT:
        case kTINYINT:
          return ColumnType::Integer;

        case kDOUBLE:
          return ColumnType::Double;

        case kFLOAT:
          return ColumnType::Float;

        case kCHAR:
        case kVARCHAR:
        case kTEXT:
          return ColumnType::String;

        default:
          return ColumnType::Unknown;
      }
    }
    return ColumnType::Unknown;
  }

 private:
  std::shared_ptr<ResultSet> result_set_;
  std::weak_ptr<Data_Namespace::DataMgr> data_mgr_;
};

/**
 * DBEngine internal implementation
 */
class DBEngineImpl : public DBEngine {
 public:
  // TODO: Remove all that hardcoded settings
  const int CALCITEPORT = 3279;
  const std::string OMNISCI_DEFAULT_DB = "omnisci";
  const std::string OMNISCI_ROOT_USER = "admin";
  const std::string OMNISCI_DATA_PATH = "//mapd_data";

  void reset() {
    // TODO: Destroy all cursors in the cursors_
    if (query_runner_ != nullptr) {
      query_runner_->reset();
    }
  }

  void executeDDL(const std::string& query) {
    if (query_runner_ != nullptr) {
      query_runner_->runDDLStatement(query);
    }
  }

  Cursor* executeDML(const std::string& query) {
    if (query_runner_ != nullptr) {
      auto rs = query_runner_->runSQL(query, ExecutorDeviceType::CPU);
      cursors_.emplace_back(new CursorImpl(rs, data_mgr_));
      return cursors_.back();
    }
    return nullptr;
  }

  DBEngineImpl(const std::string& base_path)
      : base_path_(base_path), query_runner_(nullptr) {
    if (!boost::filesystem::exists(base_path_)) {
      std::cerr << "Catalog basepath " + base_path_ + " does not exist.\n";
      // TODO: Create database if it does not exist
    } else {
      SystemParameters system_parameters;
      std::string data_path = base_path_ + OMNISCI_DATA_PATH;
      data_mgr_ = std::make_shared<Data_Namespace::DataMgr>(
          data_path, system_parameters, false, 0);
      auto calcite = std::make_shared<Calcite>(-1, CALCITEPORT, base_path_, 1024, 5000);
      auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
      sys_cat.init(base_path_, data_mgr_, {}, calcite, false, false, {});
      if (!sys_cat.getSqliteConnector()) {
        std::cerr << "SqliteConnector is null " << std::endl;
      } else {
        sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, database_);  // TODO: Check
        auto catalog = Catalog_Namespace::Catalog::get(base_path_,
                                                       database_,
                                                       data_mgr_,
                                                       std::vector<LeafHostInfo>(),
                                                       calcite,
                                                       false);
        sys_cat.getMetadataForUser(OMNISCI_ROOT_USER, user_);
        auto session = std::make_unique<Catalog_Namespace::SessionInfo>(
            catalog, user_, ExecutorDeviceType::CPU, "");
        query_runner_ = QueryRunner::QueryRunner::init(session);
      }
    }
  }

 private:
  std::string base_path_;
  std::shared_ptr<Data_Namespace::DataMgr> data_mgr_;
  Catalog_Namespace::DBMetadata database_;
  Catalog_Namespace::UserMetadata user_;
  QueryRunner::QueryRunner* query_runner_;
  std::vector<CursorImpl*> cursors_;
};

/********************************************* DBEngine external methods*/

/**
 * Creates DBEngine instance
 *
 * @param sPath Path to the existing database
 */
DBEngine* DBEngine::create(std::string path) {
  return new DBEngineImpl(path);
}

/** DBEngine downcasting methods */
inline DBEngineImpl* getImpl(DBEngine* ptr) {
  return (DBEngineImpl*)ptr;
}
inline const DBEngineImpl* getImpl(const DBEngine* ptr) {
  return (const DBEngineImpl*)ptr;
}

void DBEngine::reset() {
  // TODO: Make sure that dbengine does not released twice
  DBEngineImpl* engine = getImpl(this);
  engine->reset();
}

void DBEngine::executeDDL(std::string query) {
  DBEngineImpl* engine = getImpl(this);
  engine->executeDDL(query);
}

Cursor* DBEngine::executeDML(std::string query) {
  DBEngineImpl* engine = getImpl(this);
  return engine->executeDML(query);
}

/********************************************* Row methods */

Row::Row() {}

Row::Row(std::vector<TargetValue>& row) : row_(std::move(row)) {}

int64_t Row::getInt(size_t col_num) {
  if (col_num < row_.size()) {
    const auto scalar_value = boost::get<ScalarTargetValue>(&row_[col_num]);
    const auto value = boost::get<int64_t>(scalar_value);
    return *value;
  }
  return 0;
}

double Row::getDouble(size_t col_num) {
  if (col_num < row_.size()) {
    const auto scalar_value = boost::get<ScalarTargetValue>(&row_[col_num]);
    const auto value = boost::get<double>(scalar_value);
    return *value;
  }
  return 0.;
}

std::string Row::getStr(size_t col_num) {
  if (col_num < row_.size()) {
    const auto scalar_value = boost::get<ScalarTargetValue>(&row_[col_num]);
    auto value = boost::get<NullableString>(scalar_value);
    bool is_null = !value || boost::get<void*>(value);
    if (is_null) {
      return "Empty";
    } else {
      auto value_notnull = boost::get<std::string>(value);
      return *value_notnull;
    }
  }
  return "Out of range";
}

/********************************************* Cursor external methods*/

/** Cursor downcasting methods */
inline CursorImpl* getImpl(Cursor* ptr) {
  return (CursorImpl*)ptr;
}
inline const CursorImpl* getImpl(const Cursor* ptr) {
  return (const CursorImpl*)ptr;
}

size_t Cursor::getColCount() {
  CursorImpl* cursor = getImpl(this);
  return cursor->getColCount();
}

size_t Cursor::getRowCount() {
  CursorImpl* cursor = getImpl(this);
  return cursor->getRowCount();
}

Row Cursor::getNextRow() {
  CursorImpl* cursor = getImpl(this);
  return cursor->getNextRow();
}

int Cursor::getColType(uint32_t col_num) {
  CursorImpl* cursor = getImpl(this);
  return (int)cursor->getColType(col_num);
}
}  // namespace EmbeddedDatabase
