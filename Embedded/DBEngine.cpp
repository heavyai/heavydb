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
#include "Parser/ParserWrapper.h"
#include "Parser/parser.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "QueryRunner/QueryRunner.h"

#define CALCITEPORT 3279

using QR = QueryRunner::QueryRunner;

namespace EmbeddedDatabase {

class DBEngineImpl;

/**
 * Cursor internal implementation
 */
class CursorImpl : public Cursor {
 public:
  CursorImpl(std::shared_ptr<ResultSet> result_set,
             std::vector<std::string> col_names,
             Data_Namespace::DataMgr& data_mgr)
      : result_set_(result_set), col_names_(col_names), data_mgr_(&data_mgr) {}

  ~CursorImpl() {
    col_names_.clear();
    record_batch_.reset();
    result_set_.reset();
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
        const auto& converter =
            std::make_unique<ArrowResultSetConverter>(result_set_, col_names_, -1);
        record_batch_ = converter->convertToArrow();
        return record_batch_;
      }
    }
    return nullptr;
  }

 private:
  std::shared_ptr<ResultSet> result_set_;
  std::vector<std::string> col_names_;
  Data_Namespace::DataMgr* data_mgr_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
};

/**
 * DBEngine internal implementation
 */
class DBEngineImpl : public DBEngine {
 public:
  DBEngineImpl(const std::string& base_path) {
    if (init(base_path)) {
      std::cout << "DBEngine initialization succeed" << std::endl;
    } else {
      std::cerr << "DBEngine initialization failed" << std::endl;
    }
  }

  bool init(const std::string& base_path) {
    SystemParameters mapd_parms;
    std::string db_path = base_path.empty() ? DEFAULT_BASE_PATH : base_path;
    std::cout << "DBE:init(" << db_path << ")" << std::endl;
    std::string data_path = db_path + +"/mapd_data";

    try {
      auto is_new_db = !catalogExists(db_path);
      if (is_new_db) {
        cleanCatalog(db_path);
        createCatalog(db_path);
      }
      data_mgr_ =
          std::make_shared<Data_Namespace::DataMgr>(data_path, mapd_parms, false, 0);
      calcite_ = std::make_shared<Calcite>(-1, CALCITEPORT, db_path, 1024, 5000);

      ExtensionFunctionsWhitelist::add(calcite_->getExtensionFunctionWhitelist());
      // TODO: add UDFs with engine parameters handling
      // if (!udf_filename.empty()) {
      //  ExtensionFunctionsWhitelist::addUdfs(calcite_->getUserDefinedFunctionWhitelist());
      //}
      table_functions::TableFunctionsFactory::init();

      auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
      sys_cat.init(db_path, data_mgr_, {}, calcite_, is_new_db, false, {});

      logger::LogOptions log_options("DBE");
      log_options.set_base_path(db_path);
      logger::init(log_options);

      if (!sys_cat.getSqliteConnector()) {
        std::cerr << "DBE:init: SqliteConnector is null" << std::endl;
        return false;
      }

      sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, database_);
      auto catalog = Catalog_Namespace::Catalog::get(
          db_path, database_, data_mgr_, std::vector<LeafHostInfo>(), calcite_, false);
      sys_cat.getMetadataForUser(OMNISCI_ROOT_USER, user_);
      auto session = std::make_unique<Catalog_Namespace::SessionInfo>(
          catalog, user_, ExecutorDeviceType::CPU, "");
      QR::init(session);

      base_path_ = db_path;
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
    QR::reset();
    base_path_.clear();
  }

  void executeDDL(const std::string& query) {
    try {
      QR::get()->runDDLStatement(query);
    } catch (std::exception const& e) {
      std::cerr << "DBE:executeDDL: " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "DBE:executeDDL: Unknown exception" << std::endl;
    }
  }

  Cursor* executeDML(const std::string& query) {
    try {
      ParserWrapper pw{query};
      if (pw.isCalcitePathPermissable()) {
        const auto execution_result =
            QR::get()->runSelectQuery(query, ExecutorDeviceType::CPU, true, true);
        auto targets = execution_result->getTargetsMeta();
        std::vector<std::string> col_names;
        for (const auto target : targets) {
          col_names.push_back(target.get_resname());
        }
        auto rs = execution_result->getRows();
        cursors_.emplace_back(
            new CursorImpl(rs, col_names, QR::get()->getCatalog()->getDataMgr()));
        return cursors_.back();
      }

      auto session_info = QR::get()->getSession();
      auto query_state = QR::create_query_state(session_info, query);
      auto stdlog = STDLOG(query_state);

      SQLParser parser;
      std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
      std::string last_parsed;
      CHECK_EQ(parser.parse(query, parse_trees, last_parsed), 0) << query;
      CHECK_EQ(parse_trees.size(), size_t(1));
      auto stmt = parse_trees.front().get();
      auto insert_values_stmt = dynamic_cast<InsertValuesStmt*>(stmt);
      CHECK(insert_values_stmt);
      insert_values_stmt->execute(*session_info);
      return nullptr;
    } catch (std::exception const& e) {
      std::cerr << "DBE:executeDML: " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "DBE:executeDML: Unknown exception" << std::endl;
    }
    return nullptr;
  }

  std::vector<std::string> getTables() {
    std::vector<std::string> table_names;
    auto catalog = QR::get()->getCatalog();
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
    return table_names;
  }

  std::vector<ColumnDetails> getTableDetails(const std::string& table_name) {
    std::vector<ColumnDetails> result;
    auto catalog = QR::get()->getCatalog();
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
    return result;
  }

  void createUser(const std::string& user_name, const std::string& password) {
    Catalog_Namespace::UserMetadata user;
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    if (!sys_cat.getMetadataForUser(user_name, user)) {
      sys_cat.createUser(user_name, password, false, "", true);
    }
  }

  void dropUser(const std::string& user_name) {
    Catalog_Namespace::UserMetadata user;
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    if (!sys_cat.getMetadataForUser(user_name, user)) {
      sys_cat.dropUser(user_name);
    }
  }

  void createDatabase(const std::string& db_name) {
    Catalog_Namespace::DBMetadata db;
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    if (!sys_cat.getMetadataForDB(db_name, db)) {
      sys_cat.createDatabase(db_name, user_.userId);
    }
  }

  void dropDatabase(const std::string& db_name) {
    Catalog_Namespace::DBMetadata db;
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    if (sys_cat.getMetadataForDB(db_name, db)) {
      sys_cat.dropDatabase(db);
    }
  }

  bool setDatabase(std::string& db_name) {
    try {
      auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
      auto catalog = sys_cat.switchDatabase(db_name, user_.userName);
      updateSession(catalog);
      sys_cat.getMetadataForDB(db_name, database_);
      return true;
    } catch (std::exception const& e) {
      std::cerr << "DBE:setDatabase: " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "DBE:setDatabase: Unknown exception" << std::endl;
    }
    return false;
  }

  bool login(std::string& db_name, std::string& user_name, const std::string& password) {
    Catalog_Namespace::UserMetadata user_meta;
    try {
      auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
      auto catalog = sys_cat.login(db_name, user_name, password, user_meta, true);
      updateSession(catalog);
      sys_cat.getMetadataForDB(db_name, database_);
      sys_cat.getMetadataForUser(user_name, user_);
      return true;
    } catch (std::exception const& e) {
      std::cerr << "DBE:login: " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "DBE:login: Unknown exception" << std::endl;
    }
    return false;
  }

 protected:
  void updateSession(std::shared_ptr<Catalog_Namespace::Catalog> catalog) {
    auto session = std::make_unique<Catalog_Namespace::SessionInfo>(
        catalog, user_, ExecutorDeviceType::CPU, "");
    cursors_.clear();
    QR::reset();
    QR::init(session);
  }

  bool catalogExists(const std::string& base_path) {
    if (!boost::filesystem::exists(base_path)) {
      return false;
    }
    for (auto& subdir : system_folders_) {
      std::string path = base_path + "/" + subdir;
      if (!boost::filesystem::exists(path)) {
        return false;
      }
    }
    return true;
  }

  void cleanCatalog(const std::string& base_path) {
    if (boost::filesystem::exists(base_path)) {
      for (auto& subdir : system_folders_) {
        std::string path = base_path + "/" + subdir;
        if (boost::filesystem::exists(path)) {
          boost::filesystem::remove_all(path);
        }
      }
    }
  }

  void createCatalog(const std::string& base_path) {
    if (!boost::filesystem::exists(base_path)) {
      if (!boost::filesystem::create_directory(base_path)) {
        std::cerr << "Cannot create database directory: " << base_path << std::endl;
        return;
      }
    }
    for (auto& subdir : system_folders_) {
      std::string path = base_path + "/" + subdir;
      if (!boost::filesystem::exists(path)) {
        if (!boost::filesystem::create_directory(path)) {
          std::cerr << "Cannot create database subdirectory: " << path << std::endl;
          return;
        }
      }
    }
  }

 private:
  std::string base_path_;
  std::shared_ptr<Data_Namespace::DataMgr> data_mgr_;
  std::shared_ptr<Calcite> calcite_;
  Catalog_Namespace::DBMetadata database_;
  Catalog_Namespace::UserMetadata user_;
  std::vector<CursorImpl*> cursors_;

  std::string system_folders_[3] = {"mapd_catalogs", "mapd_data", "mapd_export"};
};

/**
 * Creates DBEngine instance
 *
 * @param sPath Path to the existing database
 */
DBEngine* DBEngine::create(const std::string& path) {
  return new DBEngineImpl(path);
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

void DBEngine::executeDDL(const std::string& query) {
  DBEngineImpl* engine = getImpl(this);
  engine->executeDDL(query);
}

Cursor* DBEngine::executeDML(const std::string& query) {
  DBEngineImpl* engine = getImpl(this);
  return engine->executeDML(query);
}

std::vector<std::string> DBEngine::getTables() {
  DBEngineImpl* engine = getImpl(this);
  return engine->getTables();
}

std::vector<ColumnDetails> DBEngine::getTableDetails(const std::string& table_name) {
  DBEngineImpl* engine = getImpl(this);
  return engine->getTableDetails(table_name);
}

void DBEngine::createUser(const std::string& user_name, const std::string& password) {
  DBEngineImpl* engine = getImpl(this);
  engine->createUser(user_name, password);
}

void DBEngine::dropUser(const std::string& user_name) {
  DBEngineImpl* engine = getImpl(this);
  engine->dropUser(user_name);
}

void DBEngine::createDatabase(const std::string& db_name) {
  DBEngineImpl* engine = getImpl(this);
  engine->createDatabase(db_name);
}

void DBEngine::dropDatabase(const std::string& db_name) {
  DBEngineImpl* engine = getImpl(this);
  engine->dropDatabase(db_name);
}

bool DBEngine::setDatabase(std::string& db_name) {
  DBEngineImpl* engine = getImpl(this);
  return engine->setDatabase(db_name);
}

bool DBEngine::login(std::string& db_name,
                     std::string& user_name,
                     const std::string& password) {
  DBEngineImpl* engine = getImpl(this);
  return engine->login(db_name, user_name, password);
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
