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
#include "QueryEngine/ArrowResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "QueryEngine/Execute.h"

namespace EmbeddedDatabase {

class DBEngineImpl;

/**
 * Cursor internal implementation
 */
class CursorImpl : public Cursor {
 public:
  CursorImpl(std::shared_ptr<ResultSet> result_set,
             std::vector<std::string> col_names,
             std::shared_ptr<Data_Namespace::DataMgr> data_mgr)
      : result_set_(result_set), col_names_(col_names), data_mgr_(data_mgr) {}

  ~CursorImpl() {
    col_names_.clear();
    record_batch_.reset();
    result_set_.reset();
    converter_.reset();
  }

  size_t getColCount() { return result_set_ ? result_set_->colCount() : 0; }

  size_t getRowCount() { return result_set_ ? result_set_->rowCount() : 0; }

  Row getNextRow() {
    if (result_set_) {
      auto row = result_set_->getNextRow(true, false);
      return row.empty() ? Row() : Row(row);
    }
    return Row();
  }

  ColumnType getColType(uint32_t col_num) {
    if (col_num < getColCount()) {
      SQLTypeInfo type_info = result_set_->getColType(col_num);
      return sqlToColumnType(type_info.get_type());
    }
    return ColumnType::UNKNOWN;
  }

  std::shared_ptr<arrow::RecordBatch> getArrowRecordBatch() {
    if (record_batch_) {
      return record_batch_;
    }
    auto col_count = getColCount();
    if (col_count > 0) {
      auto row_count = getRowCount();
      if (row_count > 0) {
        if (auto data_mgr = data_mgr_.lock()) {
          if (!converter_) {
            converter_ = std::make_unique<ArrowResultSetConverter>(
              result_set_, data_mgr, ExecutorDeviceType::CPU, 0, col_names_, row_count);
          }
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
  // TODO: Remove all that hardcoded settings
  const int CALCITEPORT = 3279;
  const std::string OMNISCI_DEFAULT_DB = "omnisci";
  const std::string OMNISCI_ROOT_USER = "admin";
  const std::string OMNISCI_DATA_PATH = "//mapd_data";

 public:
  DBEngineImpl(const std::string& base_path, int calcite_port) : query_runner_(nullptr) {
    if (init(base_path, calcite_port)) {
      std::cout << "DBEngine initialization succeed" << std::endl;
      if (g_enable_columnar_output) {
        std::cout << "Columnar format enabled" << std::endl;
      }
    } else {
      std::cerr << "DBEngine initialization failed" << std::endl;
    }
  }

  bool init(std::string base_path, int calcite_port) {
    std::cout << "DBE:init(" << base_path << ", " << calcite_port << ")" << std::endl;
    if (query_runner_) {
      std::cout << "DBE:init: Alreary initialized at " << base_path_ << std::endl;
      return base_path == base_path_;
    } else if (!boost::filesystem::exists(base_path)) {
      std::cerr << "DBE:init: Catalog basepath " + base_path_ + " does not exist.\n"
                << std::endl;
      // TODO: Create database if it does not exist
      return false;
    }
    SystemParameters mapd_parms;
    std::string data_path = base_path + OMNISCI_DATA_PATH;
    try {
      logger::LogOptions log_options("DBE");
      log_options.set_base_path(base_path);
      logger::init(log_options);
      data_mgr_ =
          std::make_shared<Data_Namespace::DataMgr>(data_path, mapd_parms, false, 0);
      auto calcite = std::make_shared<Calcite>(-1, calcite_port, base_path, 1024, 5000);
      auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
      sys_cat.init(base_path, data_mgr_, {}, calcite, false, false, {});
      if (!sys_cat.getSqliteConnector()) {
        std::cerr << "DBE:init: SqliteConnector is null" << std::endl;
        return false;
      }
      sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, database_);  // TODO: Check
      auto catalog = Catalog_Namespace::Catalog::get(
          base_path, database_, data_mgr_, std::vector<LeafHostInfo>(), calcite, false);
      sys_cat.getMetadataForUser(OMNISCI_ROOT_USER, user_);
      auto session = std::make_unique<Catalog_Namespace::SessionInfo>(
          catalog, user_, ExecutorDeviceType::CPU, "");
      query_runner_ = QueryRunner::QueryRunner::init(session);
      base_path_ = base_path;
      return true;
    } catch (std::exception const& e) {
      std::cerr << "DBE:init: " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "DBE:init: Unknown exception" << std::endl;
    }
    return false;
  }

  void reset() {
    std::cout << "DBE:reset" << std::endl;
    cursors_.clear();
    if (query_runner_) {
      query_runner_->reset();
      query_runner_ = nullptr;
    }
    data_mgr_.reset();
    base_path_.clear();
  }

  void executeDDL(const std::string& query) {
    if (query_runner_) {
      try {
        query_runner_->runDDLStatement(query);
      } catch (std::exception const& e) {
        std::cerr << "DBE:executeDDL: " << e.what() << std::endl;
      } catch (...) {
        std::cerr << "DBE:executeDDL: Unknown exception" << std::endl;
      }
    } else {
      std::cerr << "DBE:executeDDL: query_runner is NULL" << std::endl;
    }
  }

  Cursor* executeDML(const std::string& query) {
    if (query_runner_) {
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
      } catch (std::exception const& e) {
        std::cerr << "DBE:executeDML: " << e.what() << std::endl;
      } catch (...) {
        std::cerr << "DBE:executeDML: Unknown exception" << std::endl;
      }
    } else {
      std::cerr << "DBE:executeDML: query_runner is NULL" << std::endl;
    }
    return nullptr;
  }

  std::vector<std::string> getTables() {
    std::vector<std::string> table_names;
    if (query_runner_) {
      auto catalog = query_runner_->getCatalog();
      if (catalog) {
        try {
          const auto tables = catalog->getAllTableMetadata();
          for (const auto td : tables) {
            if (td->shard >= 0) {
              // skip shards, they're not standalone tables
              continue;
            }
            table_names.push_back(td->tableName);
          }
        } catch (std::exception const& e) {
          std::cerr << "DBE:getTables: " << e.what() << std::endl;
        }
      } else {
        std::cerr << "DBE:getTables: catalog is NULL" << std::endl;
      }
    } else {
      std::cerr << "DBE:getTables: query_runner is NULL" << std::endl;
    }
    return table_names;
  }

  std::vector<ColumnDetails> getTableDetails(const std::string& table_name) {
    std::vector<ColumnDetails> result;
    if (query_runner_) {
      auto catalog = query_runner_->getCatalog();
      if (catalog) {
        auto metadata = catalog->getMetadataForTable(table_name, false);
        if (metadata) {
          const auto col_descriptors = catalog->getAllColumnMetadataForTable(
              metadata->tableId, false, true, false);
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
              // have to get the actual size of the encoding from the dictionary
              // definition
              const int dict_id = ct.get_comp_param();
              auto dd = catalog->getMetadataForDict(dict_id, false);
              if (dd) {
                col_details.comp_param = dd->dictNBits;
              } else {
                std::cerr << "DBE:getTableDetails: Dictionary doesn't exist" << std::endl;
              }
            } else {
              col_details.comp_param = ct.get_comp_param();
              if (ct.is_date_in_days() && col_details.comp_param == 0) {
                col_details.comp_param = 32;
              }
            }
            result.push_back(col_details);
          }
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

/**
 * Creates DBEngine instance
 *
 * @param sPath Path to the existing database
 */
DBEngine* DBEngine::create(std::string path, int calcite_port, bool enable_columnar_output) {
  g_enable_columnar_output = enable_columnar_output;
  return new DBEngineImpl(path, calcite_port);
}

/** DBEngine downcasting methods */

inline DBEngineImpl* getImpl(DBEngine* ptr) {
  return (DBEngineImpl*)ptr;
}

inline const DBEngineImpl* getImpl(const DBEngine* ptr) {
  return (const DBEngineImpl*)ptr;
}

/** DBEngine external methods */

void DBEngine::reset() {
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

/** Cursor downcasting methods */

inline CursorImpl* getImpl(Cursor* ptr) {
  return (CursorImpl*)ptr;
}

inline const CursorImpl* getImpl(const Cursor* ptr) {
  return (const CursorImpl*)ptr;
}

/** Cursor external methods */

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
