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
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/Logger.h"
#include "Shared/mapdpath.h"
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
      : m_result_set(result_set), m_data_mgr(data_mgr) {}

  size_t getColCount() { return m_result_set->colCount(); }

  size_t getRowCount() { return m_result_set->rowCount(); }

  Row getNextRow() {
    auto row = m_result_set->getNextRow(true, false);
    if (row.empty()) {
      return Row();
    }
    return Row(row);
  }

  ColumnType getColType(uint32_t col_num) {
    if (col_num < getColCount()) {
      SQLTypeInfo type_info = m_result_set->getColType(col_num);
      switch (type_info.get_type()) {
        case kNUMERIC:
        case kDECIMAL:
        case kINT:
        case kSMALLINT:
        case kBIGINT:
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
  std::shared_ptr<ResultSet> m_result_set;
  std::weak_ptr<Data_Namespace::DataMgr> m_data_mgr;
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
    // TODO: Destroy all cursors in the m_Cursors
    if (m_query_runner != nullptr) {
      m_query_runner->reset();
    }
  }

  void executeDDL(const std::string& query) {
    if (m_query_runner != nullptr) {
      m_query_runner->runDDLStatement(query);
    }
  }

  Cursor* executeDML(const std::string& query) {
    if (m_query_runner != nullptr) {
      auto rs = m_query_runner->runSQL(query, ExecutorDeviceType::CPU);
      m_cursors.emplace_back(new CursorImpl(rs, m_data_mgr));
      return m_cursors.back();
    }
    return nullptr;
  }

  DBEngineImpl(const std::string& base_path)
      : m_base_path(base_path), m_query_runner(nullptr) {
    if (!boost::filesystem::exists(m_base_path)) {
      std::cerr << "Catalog basepath " + m_base_path + " does not exist.\n";
      // TODO: Create database if it does not exist
    } else {
      MapDParameters mapd_parms;
      std::string data_path = m_base_path + OMNISCI_DATA_PATH;
      m_data_mgr =
          std::make_shared<Data_Namespace::DataMgr>(data_path, mapd_parms, false, 0);
      auto calcite = std::make_shared<Calcite>(-1, CALCITEPORT, m_base_path, 1024, 5000);
      auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
      sys_cat.init(m_base_path, m_data_mgr, {}, calcite, false, false, {});
      if (!sys_cat.getSqliteConnector()) {
        std::cerr << "SqliteConnector is null " << std::endl;
      } else {
        sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, m_database);  // TODO: Check
        auto catalog = Catalog_Namespace::Catalog::get(m_base_path,
                                                       m_database,
                                                       m_data_mgr,
                                                       std::vector<LeafHostInfo>(),
                                                       calcite,
                                                       false);
        sys_cat.getMetadataForUser(OMNISCI_ROOT_USER, m_user);
        auto session = std::make_unique<Catalog_Namespace::SessionInfo>(
            catalog, m_user, ExecutorDeviceType::CPU, "");
        m_query_runner = QueryRunner::QueryRunner::init(session);
      }
    }
  }

 private:
  std::string m_base_path;
  std::shared_ptr<Data_Namespace::DataMgr> m_data_mgr;
  Catalog_Namespace::DBMetadata m_database;
  Catalog_Namespace::UserMetadata m_user;
  QueryRunner::QueryRunner* m_query_runner;
  std::vector<CursorImpl*> m_cursors;
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

Row::Row(std::vector<TargetValue>& row) : m_row(std::move(row)) {}

int64_t Row::getInt(size_t col_num) {
  if (col_num < m_row.size()) {
    const auto scalar_value = boost::get<ScalarTargetValue>(&m_row[col_num]);
    const auto value = boost::get<int64_t>(scalar_value);
    return *value;
  }
  return 0;
}

double Row::getDouble(size_t col_num) {
  if (col_num < m_row.size()) {
    const auto scalar_value = boost::get<ScalarTargetValue>(&m_row[col_num]);
    const auto value = boost::get<double>(scalar_value);
    return *value;
  }
  return 0.;
}

std::string Row::getStr(size_t col_num) {
  if (col_num < m_row.size()) {
    const auto scalar_value = boost::get<ScalarTargetValue>(&m_row[col_num]);
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
