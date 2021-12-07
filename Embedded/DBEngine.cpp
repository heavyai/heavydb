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
#include <stdexcept>
#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"
#include "Fragmenter/FragmentDefaultValues.h"
#include "Parser/ParserWrapper.h"
#include "Parser/parser.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "ThriftHandler/CommandLineOptions.h"
#include "ThriftHandler/DBHandler.h"

extern bool g_enable_thrift_logs;
extern bool g_enable_union;
extern bool g_serialize_temp_tables;

namespace EmbeddedDatabase {

class DBEngineImpl;

/**
 * Cursor internal implementation
 */
class CursorImpl : public Cursor {
 public:
  CursorImpl(std::shared_ptr<ResultSet> result_set, std::vector<std::string> col_names)
      : result_set_(result_set), col_names_(col_names) {}

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
    auto converter =
        std::make_unique<ArrowResultSetConverter>(result_set_, col_names_, -1);
    record_batch_ = converter->convertToArrow();
    return record_batch_;
  }

  std::shared_ptr<arrow::Table> getArrowTable() {
    if (table_) {
      return table_;
    }
    auto converter =
        std::make_unique<ArrowResultSetConverter>(result_set_, col_names_, -1);
    table_ = converter->convertToArrowTable();
    return table_;
  }

 private:
  std::shared_ptr<ResultSet> result_set_;
  std::vector<std::string> col_names_;
  std::shared_ptr<arrow::RecordBatch> record_batch_;
  std::shared_ptr<arrow::Table> table_;
};

/**
 * DBEngine internal implementation
 */
class DBEngineImpl : public DBEngine {
  static constexpr const char* log_option = "omnisci_dbe";

 public:
  DBEngineImpl() : is_temp_db_(false), prog_config_opts_(log_option) {}

  ~DBEngineImpl() { reset(); }

  bool init(const std::string& cmd_line) {
    static bool initialized{false};
    if (initialized) {
      throw std::runtime_error("Database engine already initialized");
    }

    g_serialize_temp_tables = true;

    // Split the command line into parameters
    std::vector<std::string> parameters;
    if (!cmd_line.empty()) {
      parameters = boost::program_options::split_unix(cmd_line);
    }

    // Generate command line to initialize CommandLineOptions for DBHandler
    std::vector<const char*> cstrings;
    cstrings.push_back(log_option);
    for (auto& param : parameters) {
      cstrings.push_back(param.c_str());
    }
    int argc = cstrings.size();
    const char** argv = cstrings.data();

    if (prog_config_opts_.parse_command_line(argc, argv, false)) {
      throw std::runtime_error("DBE paramerameters parsing failed");
    }

    if (!g_enable_thrift_logs) {
      apache::thrift::GlobalOutput.setOutputFunction([](const char* msg) {});
    }

    auto base_path = prog_config_opts_.base_path;

    // Check path to the database
    bool is_new_db = base_path.empty() || !catalogExists(base_path);
    if (is_new_db) {
      base_path = createCatalog(base_path);
      if (base_path.empty()) {
        throw std::runtime_error("Database directory could not be created");
      }
    }
    prog_config_opts_.base_path = base_path;
    prog_config_opts_.init_logging();

    prog_config_opts_.system_parameters.omnisci_server_port = -1;
    prog_config_opts_.system_parameters.calcite_keepalive = true;

    try {
      db_handler_ =
          std::make_shared<DBHandler>(prog_config_opts_.db_leaves,
                                      prog_config_opts_.string_leaves,
                                      prog_config_opts_.base_path,
                                      prog_config_opts_.allow_multifrag,
                                      prog_config_opts_.jit_debug,
                                      prog_config_opts_.intel_jit_profile,
                                      prog_config_opts_.read_only,
                                      prog_config_opts_.allow_loop_joins,
                                      prog_config_opts_.enable_rendering,
                                      prog_config_opts_.renderer_use_vulkan_driver,
                                      prog_config_opts_.enable_auto_clear_render_mem,
                                      prog_config_opts_.render_oom_retry_threshold,
                                      prog_config_opts_.render_mem_bytes,
                                      prog_config_opts_.max_concurrent_render_sessions,
                                      prog_config_opts_.reserved_gpu_mem,
                                      prog_config_opts_.render_compositor_use_last_gpu,
                                      prog_config_opts_.num_reader_threads,
                                      prog_config_opts_.authMetadata,
                                      prog_config_opts_.system_parameters,
                                      prog_config_opts_.enable_legacy_syntax,
                                      prog_config_opts_.idle_session_duration,
                                      prog_config_opts_.max_session_duration,
                                      prog_config_opts_.enable_runtime_udf,
                                      prog_config_opts_.udf_file_name,
                                      prog_config_opts_.udf_compiler_path,
                                      prog_config_opts_.udf_compiler_options,
#ifdef ENABLE_GEOS
                                      prog_config_opts_.libgeos_so_filename,
#endif
                                      prog_config_opts_.disk_cache_config,
                                      is_new_db);
    } catch (const std::exception& e) {
      LOG(FATAL) << "Failed to initialize database handler: " << e.what();
    }
    db_handler_->connect(
        session_id_, OMNISCI_ROOT_USER, OMNISCI_ROOT_PASSWD_DEFAULT, OMNISCI_DEFAULT_DB);
    base_path_ = base_path;
    initialized = true;
    return true;
  }

  std::shared_ptr<CursorImpl> sql_execute_dbe(const TSessionId& session_id,
                                              const std::string& query_str,
                                              const bool column_format,
                                              const int32_t first_n,
                                              const int32_t at_most_n) {
    ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                       ExecutorDeviceType::CPU,
                                                       QueryMemoryDescriptor(),
                                                       nullptr,
                                                       nullptr,
                                                       0,
                                                       0),
                           {}};
    db_handler_->sql_execute(
        result, session_id, query_str, column_format, first_n, at_most_n);
    auto& targets = result.getTargetsMeta();
    std::vector<std::string> col_names;
    for (const auto target : targets) {
      col_names.push_back(target.get_resname());
    }
    return std::make_shared<CursorImpl>(result.empty() ? nullptr : result.getRows(),
                                        col_names);
  }

  void executeDDL(const std::string& query) {
    auto res = sql_execute_dbe(session_id_, query, false, -1, -1);
  }

  void importArrowTable(const std::string& name,
                        std::shared_ptr<arrow::Table>& table,
                        uint64_t fragment_size) {
    setArrowTable(name, table);
    try {
      auto session = db_handler_->get_session_copy(session_id_);
      TableDescriptor td;
      td.tableName = name;
      td.userId = session.get_currentUser().userId;
      td.storageType = "ARROW:" + name;
      td.persistenceLevel = Data_Namespace::MemoryLevel::CPU_LEVEL;
      td.isView = false;
      td.fragmenter = nullptr;
      td.fragType = Fragmenter_Namespace::FragmenterType::INSERT_ORDER;
      td.maxFragRows = fragment_size > 0 ? fragment_size : DEFAULT_FRAGMENT_ROWS;
      td.maxChunkSize = DEFAULT_MAX_CHUNK_SIZE;
      td.fragPageSize = DEFAULT_PAGE_SIZE;
      td.maxRows = DEFAULT_MAX_ROWS;
      td.keyMetainfo = "[]";

      std::list<ColumnDescriptor> cols;
      std::vector<Parser::SharedDictionaryDef> dictionaries;
      auto catalog = session.get_catalog_ptr();
      // nColumns
      catalog->createTable(td, cols, dictionaries, false);
      Catalog_Namespace::SysCatalog::instance().createDBObject(
          session.get_currentUser(), td.tableName, TableDBObjectType, *catalog);
    } catch (...) {
      releaseArrowTable(name);
      throw;
    }
    releaseArrowTable(name);
  }

  std::shared_ptr<CursorImpl> executeDML(const std::string& query) {
    return sql_execute_dbe(session_id_, query, false, -1, -1);
  }

  std::shared_ptr<CursorImpl> executeRA(const std::string& query) {
    return sql_execute_dbe(session_id_, query, false, -1, -1);
  }

  std::vector<std::string> getTables() {
    std::vector<std::string> table_names;
    auto& catalog = db_handler_->get_session_copy(session_id_).getCatalog();
    const auto tables = catalog.getAllTableMetadata();
    for (const auto td : tables) {
      if (td->shard >= 0) {
        // skip shards, they're not standalone tables
        continue;
      }
      table_names.push_back(td->tableName);
    }
    return table_names;
  }

  std::vector<ColumnDetails> getTableDetails(const std::string& table_name) {
    std::vector<ColumnDetails> result;
    auto& catalog = db_handler_->get_session_copy(session_id_).getCatalog();
    auto metadata = catalog.getMetadataForTable(table_name, false);
    if (metadata) {
      const auto col_descriptors =
          catalog.getAllColumnMetadataForTable(metadata->tableId, false, true, false);
      const auto deleted_cd = catalog.getDeletedColumn(metadata);
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
          auto dd = catalog.getMetadataForDict(dict_id, false);
          if (dd) {
            col_details.comp_param = dd->dictNBits;
          } else {
            throw std::runtime_error("Dictionary definition for column doesn't exist");
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
    return result;
  }

  bool setDatabase(std::string& db_name) {
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    auto& user = db_handler_->get_session_copy(session_id_).get_currentUser();
    sys_cat.switchDatabase(db_name, user.userName);
    return true;
  }

  bool login(std::string& db_name, std::string& user_name, const std::string& password) {
    db_handler_->disconnect(session_id_);
    db_handler_->connect(session_id_, user_name, password, db_name);
    return true;
  }

 protected:
  void reset() {
    if (db_handler_) {
      try {
        db_handler_->disconnect(session_id_);
      } catch (const std::exception& e) {
        // skip exception in case of invalid session
      }
      db_handler_->shutdown();
    }
    Catalog_Namespace::SysCatalog::destroy();
    db_handler_.reset();
    logger::shutdown();
    if (is_temp_db_) {
      boost::filesystem::remove_all(base_path_);
    }
    base_path_.clear();
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

  std::string createCatalog(const std::string& base_path) {
    std::string root_dir = base_path;
    if (base_path.empty()) {
      boost::system::error_code error;
      auto tmp_path = boost::filesystem::temp_directory_path(error);
      if (boost::system::errc::success != error.value()) {
        std::cerr << error.message() << std::endl;
        return "";
      }
      tmp_path /= "omnidbe_%%%%-%%%%-%%%%";
      auto uniq_path = boost::filesystem::unique_path(tmp_path, error);
      if (boost::system::errc::success != error.value()) {
        std::cerr << error.message() << std::endl;
        return "";
      }
      root_dir = uniq_path.string();
      is_temp_db_ = true;
    }
    if (!boost::filesystem::exists(root_dir)) {
      if (!boost::filesystem::create_directory(root_dir)) {
        std::cerr << "Cannot create database directory: " << root_dir << std::endl;
        return "";
      }
    }
    size_t absent_count = 0;
    for (auto& sub_dir : system_folders_) {
      std::string path = root_dir + "/" + sub_dir;
      if (!boost::filesystem::exists(path)) {
        if (!boost::filesystem::create_directory(path)) {
          std::cerr << "Cannot create database subdirectory: " << path << std::endl;
          return "";
        }
        ++absent_count;
      }
    }
    if ((absent_count > 0) && (absent_count < system_folders_.size())) {
      std::cerr << "Database directory structure is broken: " << root_dir << std::endl;
      return "";
    }
    return root_dir;
  }

 private:
  std::string base_path_;
  std::string session_id_;
  std::shared_ptr<DBHandler> db_handler_;
  bool is_temp_db_;
  std::string udf_filename_;
  CommandLineOptions prog_config_opts_;

  std::vector<std::string> system_folders_ = {"mapd_catalogs",
                                              "mapd_data",
                                              "mapd_export"};
};

namespace {
std::mutex engine_create_mutex;
}

std::shared_ptr<DBEngine> DBEngine::create(const std::string& cmd_line) {
  const std::lock_guard<std::mutex> lock(engine_create_mutex);
  auto engine = std::make_shared<DBEngineImpl>();
  if (!engine->init(cmd_line)) {
    throw std::runtime_error("DBE initialization failed");
  }
  return engine;
}

/** DBEngine downcasting methods */

inline DBEngineImpl* getImpl(DBEngine* ptr) {
  return (DBEngineImpl*)ptr;
}
inline const DBEngineImpl* getImpl(const DBEngine* ptr) {
  return (const DBEngineImpl*)ptr;
}

/** DBEngine external methods */

void DBEngine::executeDDL(const std::string& query) {
  DBEngineImpl* engine = getImpl(this);
  engine->executeDDL(query);
}

std::shared_ptr<Cursor> DBEngine::executeDML(const std::string& query) {
  DBEngineImpl* engine = getImpl(this);
  return engine->executeDML(query);
}

std::shared_ptr<Cursor> DBEngine::executeRA(const std::string& query) {
  DBEngineImpl* engine = getImpl(this);
  return engine->executeRA(query);
}

void DBEngine::importArrowTable(const std::string& name,
                                std::shared_ptr<arrow::Table>& table,
                                uint64_t fragment_size) {
  DBEngineImpl* engine = getImpl(this);
  engine->importArrowTable(name, table, fragment_size);
}

std::vector<std::string> DBEngine::getTables() {
  DBEngineImpl* engine = getImpl(this);
  return engine->getTables();
}

std::vector<ColumnDetails> DBEngine::getTableDetails(const std::string& table_name) {
  DBEngineImpl* engine = getImpl(this);
  return engine->getTableDetails(table_name);
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
  CHECK(cursor);
  return cursor->getColCount();
}

size_t Cursor::getRowCount() {
  CursorImpl* cursor = getImpl(this);
  CHECK(cursor);
  return cursor->getRowCount();
}

Row Cursor::getNextRow() {
  CursorImpl* cursor = getImpl(this);
  CHECK(cursor);
  return cursor->getNextRow();
}

ColumnType Cursor::getColType(uint32_t col_num) {
  CursorImpl* cursor = getImpl(this);
  CHECK(cursor);
  return cursor->getColType(col_num);
}

std::shared_ptr<arrow::RecordBatch> Cursor::getArrowRecordBatch() {
  CursorImpl* cursor = getImpl(this);
  CHECK(cursor);
  return cursor->getArrowRecordBatch();
}

std::shared_ptr<arrow::Table> Cursor::getArrowTable() {
  CursorImpl* cursor = getImpl(this);
  return cursor->getArrowTable();
}

}  // namespace EmbeddedDatabase
