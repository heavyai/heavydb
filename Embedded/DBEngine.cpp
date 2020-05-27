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
#include "QueryEngine/Execute.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/Logger.h"
#include "Shared/mapdpath.h"
#include "Shared/sqltypes.h"
#include <thread>
#include <chrono>



namespace EmbeddedDatabase {

inline ColumnType sqlToColumnType(const SQLTypes& type) {
  switch (type) {
    case kBOOLEAN:
      return ColumnType::BOOL;
    case kTINYINT:
      return ColumnType::TINYINT;
    case kSMALLINT:
      return ColumnType::SMALLINT;
    case kINT:
      return ColumnType::INT;
    case kBIGINT:
      return ColumnType::BIGINT;
    case kFLOAT:
      return ColumnType::FLOAT;
    case kNUMERIC:
    case kDECIMAL:
      return ColumnType::DECIMAL;
    case kDOUBLE:
      return ColumnType::DOUBLE;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return ColumnType::STR;
    case kTIME:
      return ColumnType::TIME;
    case kTIMESTAMP:
      return ColumnType::TIMESTAMP;
    case kDATE:
      return ColumnType::DATE;
    case kINTERVAL_DAY_TIME:
      return ColumnType::INTERVAL_DAY_TIME;
    case kINTERVAL_YEAR_MONTH:
      return ColumnType::INTERVAL_YEAR_MONTH;
    case kPOINT:
      return ColumnType::POINT;
    case kLINESTRING:
      return ColumnType::LINESTRING;
    case kPOLYGON:
      return ColumnType::POLYGON;
    case kMULTIPOLYGON:
      return ColumnType::MULTIPOLYGON;
    case kGEOMETRY:
      return ColumnType::GEOMETRY;
    case kGEOGRAPHY:
      return ColumnType::GEOGRAPHY;
    default:
      return ColumnType::UNKNOWN;
  }
  return ColumnType::UNKNOWN;
}

inline ColumnEncoding sqlToColumnEncoding(const EncodingType& type) {
  switch (type) {
    case kENCODING_NONE:
      return ColumnEncoding::NONE;
    case kENCODING_FIXED:
      return ColumnEncoding::FIXED;
    case kENCODING_RL:
      return ColumnEncoding::RL;
    case kENCODING_DIFF:
      return ColumnEncoding::DIFF;
    case kENCODING_DICT:
      return ColumnEncoding::DICT;
    case kENCODING_SPARSE:
      return ColumnEncoding::SPARSE;
    case kENCODING_GEOINT:
      return ColumnEncoding::GEOINT;
    case kENCODING_DATE_IN_DAYS:
      return ColumnEncoding::DATE_IN_DAYS;
    default:
      return ColumnEncoding::NONE;
  }
  return ColumnEncoding::NONE;
}
//enum class ColumnType : uint32_t { Unknown, Integer, Double, Float, String, Array };

 ColumnDetails::ColumnDetails()
  : col_type(ColumnType::UNKNOWN)
  , encoding(ColumnEncoding::NONE)
  , nullable(false)
  , is_array(false)
  , precision(0)
  , scale(0)
  , comp_param(0)
  {}

 ColumnDetails::ColumnDetails(const std::string& _col_name,
                ColumnType _col_type,
                ColumnEncoding _encoding,
                bool _nullable,
                bool _is_array,
                int _precision,
                int _scale,
                int _comp_param)
  : col_name(_col_name)
  , col_type(_col_type)
  , encoding(_encoding)
  , nullable(_nullable)
  , is_array(_is_array)
  , precision(_precision)
  , scale(_scale)
  , comp_param(_comp_param)
  {}

/**
 * Cursor internal implementation
 */
class CursorImpl : public Cursor {
 public:
  CursorImpl(std::shared_ptr<ResultSet> result_set,
             std::vector<std::string> col_names,
             std::shared_ptr<Data_Namespace::DataMgr> data_mgr)
      : result_set_(result_set), col_names_(col_names), data_mgr_(data_mgr) {}

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
          return ColumnType::INT;

        case kDOUBLE:
          return ColumnType::DOUBLE;

        case kFLOAT:
          return ColumnType::FLOAT;

        case kCHAR:
        case kVARCHAR:
        case kTEXT:
          return ColumnType::STR;

        default:
          return ColumnType::UNKNOWN;
      }
    }
    return ColumnType::UNKNOWN;
  }

  std::shared_ptr<arrow::RecordBatch> getArrowRecordBatch() {
    auto col_count = getColCount();
    if (col_count > 0) {
        auto row_count = getRowCount();;
        if (row_count > 0) {
            if (auto data_mgr = data_mgr_.lock()) {
//                const auto & 
                    converter_ = std::make_unique<ArrowResultSetConverter>(
                    result_set_,
                    data_mgr,
                    ExecutorDeviceType::CPU,
                    0,
                    col_names_,
                    row_count);
                record_batch_ = converter_->convertToArrow();
                return record_batch_;
            }
        }
    }
    return nullptr;
  }

 private:
  std::shared_ptr<ResultSet> result_set_;
  std::vector<std::string> col_names_;
  std::weak_ptr<Data_Namespace::DataMgr> data_mgr_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
  std::unique_ptr<ArrowResultSetConverter> converter_;
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
    std::cout << "DBE RESET !!!!!!!!!!!!!!!!!!!!" << std::endl;
    if (query_runner_ != nullptr) {
      query_runner_->reset();
    }
  }

  void executeDDL(const std::string& query) {
    std::cout << "DBE:CPP:executeDDL: " << query << std::endl;
    if (query_runner_ != nullptr) {

      try {
        query_runner_->runDDLStatement(query);
      } catch(std::exception const& e) {
        std::cout << "DBE:CPP:executeDDL:Exception: " << e.what() << std::endl;
      }
      std::cout << "DBE:CPP:executeDDL: OK" << std::endl;
    } else {
      std::cout << "DBE:CPP:executeDDL: query_runner is NULL" << std::endl;
    }
  }

  Cursor* executeDML(const std::string& query) {
    std::cout << "g_enable_columnar_output = " << g_enable_columnar_output << std::endl;
    g_enable_columnar_output = true;
    //g_enable_debug_timer = true;
    std::cout << "DBE:CPP:executeDML: " << query << std::endl;
    if (query_runner_ != nullptr) {
      try {
        const auto execution_result =
          query_runner_->runSelectQuery(query, ExecutorDeviceType::CPU, true, true);
        auto targets = execution_result.getTargetsMeta();
        std::vector<std::string> col_names;
        for (const auto target : targets) {
          col_names.push_back(target.get_resname());
        }
        auto rs = execution_result.getRows();
        cursors_.emplace_back(new CursorImpl(rs, col_names, data_mgr_));
        return cursors_.back();
      } catch(std::exception const& e) {
        std::cout << "DBE:CPP:executeDML:Exception: " << e.what() << std::endl;
      }
      std::cout << "DBE:CPP:executeDML: OK" << std::endl;
    } else {
      std::cout << "DBE:CPP:executeDML: query_runner is NULL" << std::endl;
    }
    return nullptr;
  }


  std::vector<std::string> getTables() {
    std::cout << "DBE:CPP:getTables" << std::endl;
    std::vector<std::string> table_names;
    if (query_runner_) {
      auto catalog = query_runner_->getCatalog();
      if (catalog) {
        try {
//  auto const session_ptr = stdlog.getConstSessionInfo();
//  auto const& cat = session_ptr->getCatalog();
          const auto tables = catalog->getAllTableMetadata();
          for (const auto td : tables) {
            if (td->shard >= 0) {
              // skip shards, they're not standalone tables
              continue;
            }
            std::cout << "DBE:CPP:getTables: " << td->tableName << std::endl;
            table_names.push_back(td->tableName);
          }
        } catch(std::exception const& e) {
          std::cout << "DBE:CPP:getTables:Exception: " << e.what() << std::endl;
        }
        std::cout << "DBE:CPP:getTables: OK" << std::endl;
      } else {
        std::cout << "DBE:CPP:executeDML: catalog is NULL" << std::endl;
      }
    } else {
      std::cout << "DBE:CPP:executeDML: query_runner is NULL" << std::endl;
    }
    return table_names;
  }

  DBEngineImpl(const std::string& base_path, int calcite_port)
      : base_path_(base_path), query_runner_(nullptr) {
    std::cout << "DBE:CPP:DBEngine: " << base_path << ". port = " << calcite_port << std::endl;
    if (!boost::filesystem::exists(base_path_)) {
      std::cerr << "Catalog basepath " + base_path_ + " does not exist.\n";
      // TODO: Create database if it does not exist
    } else {
      SystemParameters mapd_parms;
      std::string data_path = base_path_ + OMNISCI_DATA_PATH;
        std::cout << "data_path =" << data_path << std::endl;
      data_mgr_ =
          std::make_shared<Data_Namespace::DataMgr>(data_path, mapd_parms, false, 0);
      auto calcite = std::make_shared<Calcite>(-1, calcite_port, base_path_, 1024, 5000);
      //std::this_thread::sleep_for(5s);
      auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
      sys_cat.init(base_path_, data_mgr_, {}, calcite, false, false, {});
      //std::this_thread::sleep_for(5s);
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
        std::cout << "DBE:CPP:DBEngine: OK" << std::endl;
      }
    }
  }

  std::vector<ColumnDetails> getTableDetails(const std::string& table_name) {
    std::cout << "DBE:CPP:getTableDetails: " << table_name << std::endl;
    std::vector<ColumnDetails> result;
    if (query_runner_) {
      auto catalog = query_runner_->getCatalog();
      if (catalog) {
         auto metadata = catalog->getMetadataForTable(table_name, false);
        if (metadata) {
          const auto col_descriptors =
            catalog->getAllColumnMetadataForTable(metadata->tableId, false, true, false);
          const auto deleted_cd = catalog->getDeletedColumn(metadata);
          for (const auto cd : col_descriptors) {
            if (cd == deleted_cd) {
              continue;
            }
            ColumnDetails col_details;
            col_details.col_name = cd->columnName;
            auto ct = cd->columnType;
            SQLTypes sql_type = ct.get_type();
            EncodingType sql_enc = ct.get_compression();
            col_details.col_type = sqlToColumnType(sql_type);
            col_details.encoding = sqlToColumnEncoding(sql_enc);
            col_details.nullable = !ct.get_notnull();
            col_details.is_array = (sql_type == kARRAY);
            if (IS_GEO(sql_type)) {
              col_details.precision = static_cast<int>(ct.get_subtype());
              col_details.scale = ct.get_output_srid();
            } else {
              col_details.precision = ct.get_precision();
              col_details.scale = ct.get_scale();
            }
            if (col_details.encoding == ColumnEncoding::DICT) {
              // have to get the actual size of the encoding from the dictionary definition
              const int dict_id = ct.get_comp_param();
              auto dd = catalog->getMetadataForDict(dict_id, false);
              if (dd) {
                col_details.comp_param = dd->dictNBits;
              } else {
                std::cout << "Dictionary doesn't exist" << std::endl;
              }
            } else {
              col_details.comp_param = ct.get_comp_param();
              if (ct.is_date_in_days() && col_details.comp_param == 0) {
                col_details.comp_param = 32;
              }
            }
            result.push_back(col_details);
          }
          std::cout << "DBE:CPP:getTableDetails: OK" << std::endl;
        }
      }
    }
    return result;
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
DBEngine* DBEngine::create(std::string path, int calcite_port) {
  return new DBEngineImpl(path, calcite_port);
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

std::vector<ColumnDetails> DBEngine::getTableDetails(const std::string& table_name) {
  DBEngineImpl* engine = getImpl(this);
  return engine->getTableDetails(table_name);
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

ColumnType Cursor::getColType(uint32_t col_num) {
  CursorImpl* cursor = getImpl(this);
  return cursor->getColType(col_num);
}

std::shared_ptr<arrow::RecordBatch> Cursor::getArrowRecordBatch() {
  CursorImpl* cursor = getImpl(this);
  return cursor->getArrowRecordBatch();
}
}  // namespace EmbeddedDatabase
