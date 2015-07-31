#include "MapDServer.h"
#include "gen-cpp/MapD.h"
#include <thrift/protocol/TBinaryProtocol.h>
//#include <thrift/server/TSimpleServer.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

#include "Catalog/Catalog.h"
#include "QueryEngine/Execute.h"
#include "Parser/parser.h"
#include "Planner/Planner.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/measure.h"
#include "Import/Importer.h"
#include "MapDRelease.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/bimap.hpp>
#include <memory>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <random>
#include <map>
#include <cmath>
#include <typeinfo>
#include <glog/logging.h>
#include <signal.h>
#include <unistd.h>


using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;


using boost::shared_ptr;

#define INVALID_SESSION_ID  -1

class MapDHandler : virtual public MapDIf {
public:
  MapDHandler(const std::string& base_data_path,
              const std::string& executor_device,
              const bool allow_multifrag,
              const bool jit_debug)
  : base_data_path_(base_data_path)
  , random_gen_(std::random_device{}())
  , session_id_dist_(0, INT32_MAX)
  , jit_debug_(jit_debug)
  , allow_multifrag_(allow_multifrag) {
        LOG(INFO) << "MapD Server " << MapDRelease;
    if (executor_device == "gpu") {
        executor_device_type_ = ExecutorDeviceType::GPU;
        LOG(INFO) << "Started in GPU Mode" << std::endl; 
        cpu_mode_only_ = false;
    }
    else if (executor_device == "hybrid") {
        executor_device_type_ = ExecutorDeviceType::Hybrid;
        LOG(INFO) << "Started in Hybrid Mode" << std::endl; 
        cpu_mode_only_ = false;
    }
    else {
        executor_device_type_ = ExecutorDeviceType::CPU;
        LOG(INFO) << "Started in CPU Mode" << std::endl; 
        cpu_mode_only_ = true;
    }
    const auto data_path = boost::filesystem::path(base_data_path_) / "mapd_data";
    data_mgr_ = new Data_Namespace::DataMgr(data_path.string(), !cpu_mode_only_); // second param is whether to initialize GPU buffer pool
    sys_cat_.reset(new Catalog_Namespace::SysCatalog(base_data_path_, data_mgr_));
  }
  ~MapDHandler() {
    LOG(INFO) << "mapd_server exits." << std::endl;
  }

  TSessionId connect(const std::string &user, const std::string &passwd, const std::string &dbname) {
    mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
    Catalog_Namespace::UserMetadata user_meta;
    if (!sys_cat_->getMetadataForUser(user, user_meta)) {
      TMapDException ex;
      ex.error_msg = std::string("User ") + user + " does not exist.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    if (user_meta.passwd != passwd) {
      TMapDException ex;
      ex.error_msg = std::string("Password for User ") + user + " is incorrect.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    Catalog_Namespace::DBMetadata db_meta;
    if (!sys_cat_->getMetadataForDB(dbname, db_meta)) {
      TMapDException ex;
      ex.error_msg = std::string("Database ") + dbname + " does not exist.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    if (!user_meta.isSuper && user_meta.userId != db_meta.dbOwner) {
      TMapDException ex;
      ex.error_msg = std::string("User ") + user + " is not authorized to access database " + dbname;
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    TSessionId session = INVALID_SESSION_ID;
    while (true) {
      session = session_id_dist_(random_gen_);
      auto session_it = sessions_.find(session);
      if (session_it == sessions_.end())
        break;
    }
    auto cat_it = cat_map_.find(dbname);
    if (cat_it == cat_map_.end()) {
      Catalog_Namespace::Catalog *cat = new Catalog_Namespace::Catalog(base_data_path_, db_meta, data_mgr_);
      cat_map_[dbname].reset(cat);
      sessions_[session].reset(new Catalog_Namespace::SessionInfo(cat_map_[dbname], user_meta, executor_device_type_, session));
    } else
      sessions_[session].reset(new Catalog_Namespace::SessionInfo(cat_it->second, user_meta, executor_device_type_, session));
    LOG(INFO) << "User " << user << " connected to database " << dbname << std::endl;
    return session;
  }

  void disconnect(const TSessionId session) {
    mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
    auto session_it = get_session_it(session);
    const auto dbname = session_it->second->get_catalog().get_currentDB().dbName;
    LOG(INFO) << "User " << session_it->second->get_currentUser().userName << " disconnected from database " << dbname << std::endl;
    sessions_.erase(session_it);
  }

  TDatum value_to_thrift(const TargetValue& tv, const SQLTypeInfo& ti) {
    TDatum datum;
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    if (!scalar_tv) {
      const auto list_tv = boost::get<std::vector<ScalarTargetValue>>(&tv);
      CHECK(list_tv);
      CHECK(ti.is_array());
      for (const auto& elem_tv : *list_tv) {
        const auto scalar_col_val = value_to_thrift(elem_tv, ti.get_elem_type());
        datum.val.arr_val.push_back(scalar_col_val);
      }
      datum.is_null = datum.val.arr_val.empty();
      return datum;
    }
    if (boost::get<int64_t>(scalar_tv)) {
      datum.val.int_val = *(boost::get<int64_t>(scalar_tv));
      switch (ti.get_type()) {
        case kBOOLEAN:
          datum.is_null = (datum.val.int_val == NULL_BOOLEAN);
          break;
        case kSMALLINT:
          datum.is_null = (datum.val.int_val == NULL_SMALLINT);
          break;
        case kINT:
          datum.is_null = (datum.val.int_val == NULL_INT);
          break;
        case kBIGINT:
          datum.is_null = (datum.val.int_val == NULL_BIGINT);
          break;
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          if (sizeof(time_t) == 4)
            datum.is_null = (datum.val.int_val == NULL_INT);
          else
            datum.is_null = (datum.val.int_val == NULL_BIGINT);
          break;
        default:
          datum.is_null = false;
      }
    } else if (boost::get<double>(scalar_tv)) {
      datum.val.real_val = *(boost::get<double>(scalar_tv));
      if (ti.get_type() == kFLOAT) {
        datum.is_null = (datum.val.real_val == NULL_FLOAT);
      } else {
        datum.is_null = (datum.val.real_val == NULL_DOUBLE);
      }
    } else if (boost::get<NullableString>(scalar_tv)) {
      auto s_n = boost::get<NullableString>(scalar_tv);
      auto s = boost::get<std::string>(s_n);
      if (s) {
        datum.val.str_val = *s;
      } else {
        auto null_p = boost::get<void*>(s_n);
        CHECK(null_p && !*null_p);
      }
      datum.is_null = !s;
    } else {
      CHECK(false);
    }
    return datum;
  }

  void sql_execute(TQueryResult& _return, const TSessionId session, const std::string& query_str) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    auto executor_device_type = session_info.get_executor_device_type();
    LOG(INFO) << query_str;
    auto execute_time = measure<>::execution([]() {});
    auto total_time = measure<>::execution([&]() {
    SQLParser parser;
    std::list<Parser::Stmt*> parse_trees;
    std::string last_parsed;
    int num_parse_errors = 0;
    try {
      num_parse_errors = parser.parse(query_str, parse_trees, last_parsed);
    }
    catch (std::exception &e) {
        TMapDException ex;
        ex.error_msg = std::string("Exception: ") + e.what();
        LOG(ERROR) << ex.error_msg;
        throw ex;
    }
    if (num_parse_errors > 0) {
      TMapDException ex;
      ex.error_msg = "Syntax error at: " + last_parsed;
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    execute_time = 0;
    for (auto stmt : parse_trees) {
      try {
      std::unique_ptr<Parser::Stmt> stmt_ptr(stmt);
      Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
      Parser::ExplainStmt *explain_stmt = nullptr;
      if (ddl != nullptr)
        explain_stmt = dynamic_cast<Parser::ExplainStmt*>(ddl);
      if (ddl != nullptr && explain_stmt == nullptr) {
        execute_time += measure<>::execution([&]() {
          ddl->execute(session_info);
        });
      } else {
        const Parser::DMLStmt *dml; 
        if (explain_stmt != nullptr)
          dml = explain_stmt->get_stmt();
        else
          dml = dynamic_cast<Parser::DMLStmt*>(stmt);
        Analyzer::Query query;
        dml->analyze(cat, query);
        Planner::Optimizer optimizer(query, cat);
        auto root_plan = optimizer.optimize();
        std::unique_ptr<Planner::RootPlan> plan_ptr(root_plan);  // make sure it's deleted
        if (explain_stmt != nullptr) {
          root_plan->set_plan_dest(Planner::RootPlan::Dest::kEXPLAIN);
        }
        auto executor = Executor::getExecutor(root_plan->get_catalog().get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "");
        ResultRows results({}, nullptr, nullptr);
        execute_time += measure<>::execution([&]() {
          results = executor->execute(root_plan, true, executor_device_type, ExecutorOptLevel::Default, allow_multifrag_);
        });
        if (explain_stmt) {
          CHECK_EQ(size_t(1), results.size());
          TColumnType proj_info;
          proj_info.col_name = "Explanation";
          proj_info.col_type.type = TDatumType::STR;
          proj_info.col_type.nullable = false;
          proj_info.col_type.is_array = false;
          _return.row_set.row_desc.push_back(proj_info);
          TRow trow;
          TDatum explanation;
          const auto tv = results.get(0, 0, false);
          const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
          CHECK(scalar_tv);
          const auto s_n = boost::get<NullableString>(scalar_tv);
          CHECK(s_n);
          const auto s = boost::get<std::string>(s_n);
          CHECK(s);
          explanation.val.str_val = *s;
          explanation.is_null = false;
          trow.cols.push_back(explanation);
          _return.row_set.rows.push_back(trow);
          return;
        }
        const auto plan = root_plan->get_plan();
        const auto& targets = plan->get_targetlist();
        {
          CHECK(plan);
          TColumnType proj_info;
          size_t i = 0;
          for (const auto target : targets) {
            proj_info.col_name = target->get_resname();
            if (proj_info.col_name.empty()) {
              proj_info.col_name = "result_" + std::to_string(i + 1);
            }
            const auto& target_ti = target->get_expr()->get_type_info();
            proj_info.col_type.type = type_to_thrift(target_ti);
            proj_info.col_type.encoding = encoding_to_thrift(target_ti);
            proj_info.col_type.nullable = !target_ti.get_notnull();
            proj_info.col_type.is_array = target_ti.get_type() == kARRAY;
            _return.row_set.row_desc.push_back(proj_info);
            ++i;
          }
        }
        for (size_t row_idx = 0; row_idx < results.size(); ++row_idx) {
          TRow trow;
          for (size_t i = 0; i < results.colCount(); ++i) {
            const auto agg_result = results.get(row_idx, i, true);
            trow.cols.push_back(value_to_thrift(agg_result, targets[i]->get_expr()->get_type_info()));
          }
          _return.row_set.rows.push_back(trow);
        }
      }
    }
    catch (std::exception &e) {
        TMapDException ex;
        ex.error_msg = std::string("Exception: ") + e.what();
        LOG(ERROR) << ex.error_msg;
        throw ex;
    }
    }
    });
    _return.execution_time_ms = execute_time;
    LOG(INFO) << "Total: " << total_time << " (ms), Execution: " << execute_time<< " (ms)";
  }

  void get_table_descriptor(TTableDescriptor& _return, const TSessionId session, const std::string& table_name) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    auto td = cat.getMetadataForTable(table_name);
    if (!td) {
      TMapDException ex;
      ex.error_msg = "Table doesn't exist";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    const auto col_descriptors = cat.getAllColumnMetadataForTable(td->tableId);
    for (const auto cd : col_descriptors) {
      TColumnType col_type;
      col_type.col_type.type = type_to_thrift(cd->columnType);
      col_type.col_type.encoding = encoding_to_thrift(cd->columnType);
      col_type.col_type.nullable = !cd->columnType.get_notnull();
      col_type.col_type.is_array = cd->columnType.get_type() == kARRAY;
      _return.insert(std::make_pair(cd->columnName, col_type));
    }
  }

  void get_row_descriptor(TRowDescriptor& _return, const TSessionId session, const std::string& table_name) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    auto td = cat.getMetadataForTable(table_name);
    if (!td) {
      TMapDException ex;
      ex.error_msg = "Table doesn't exist";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    const auto col_descriptors = cat.getAllColumnMetadataForTable(td->tableId);
    for (const auto cd : col_descriptors) {
      TColumnType col_type;
      col_type.col_name = cd->columnName;
      col_type.col_type.type = type_to_thrift(cd->columnType);
      col_type.col_type.encoding = encoding_to_thrift(cd->columnType);
      col_type.col_type.nullable = !cd->columnType.get_notnull();
      col_type.col_type.is_array = cd->columnType.get_type() == kARRAY;
      _return.push_back(col_type);
    }
  }

  void get_frontend_view(std::string& _return, const TSessionId session, const std::string& view_name) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    auto vd = cat.getMetadataForFrontendView(view_name);
    if (!vd) {
      TMapDException ex;
      ex.error_msg = "View " + view_name + " doesn't exist";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    _return.append(vd->viewState);
  }

  void get_tables(std::vector<std::string> & table_names, const TSessionId session) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    const auto tables = cat.getAllTableMetadata();
    for (const auto td : tables) {
      table_names.push_back(td->tableName);
    }
  }

  void get_users(std::vector<std::string> &user_names) {
    std::list<Catalog_Namespace::UserMetadata> user_list = sys_cat_->getAllUserMetadata();
    for (auto u : user_list) {
      user_names.push_back(u.userName);
    }
  }

  void get_version(std::string &version) {
    version =  MapDRelease;
  }

  void get_databases(std::vector<TDBInfo> &dbinfos) {
    std::list<Catalog_Namespace::DBMetadata> db_list = sys_cat_->getAllDBMetadata();
    std::list<Catalog_Namespace::UserMetadata> user_list = sys_cat_->getAllUserMetadata();
    for (auto d : db_list) {
      TDBInfo dbinfo;
      dbinfo.db_name = d.dbName;
      for (auto u : user_list) {
        if (d.dbOwner == u.userId) {
          dbinfo.db_owner = u.userName;
          break;
        }
      }
      dbinfos.push_back(dbinfo);
    }
  }

  void get_frontend_views(std::vector<std::string> & view_names, const TSessionId session) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    const auto views = cat.getAllFrontendViewMetadata();
    for (const auto vd : views) {
      view_names.push_back(vd->viewName);
    }
  }

  void set_execution_mode(const TSessionId session, const TExecuteMode::type mode) {
    mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
    auto session_it = get_session_it(session);
    const std::string &user_name = session_it->second->get_currentUser().userName;
    switch (mode) {
      case TExecuteMode::GPU:
        if (cpu_mode_only_) {
          TMapDException e;
          e.error_msg = "Cannot switch to GPU mode in a server started in CPU-only mode.";
          throw e;
        }
        session_it->second->set_executor_device_type(ExecutorDeviceType::GPU);
        LOG(INFO) << "User " << user_name << " sets GPU mode.";
        break;
      case TExecuteMode::CPU:
        session_it->second->set_executor_device_type(ExecutorDeviceType::CPU);
        LOG(INFO) << "User " << user_name << " sets CPU mode.";
        break;
      case TExecuteMode::HYBRID:
        if (cpu_mode_only_) {
          TMapDException e;
          e.error_msg = "Cannot switch to Hybrid mode in a server started in CPU-only mode.";
          throw e;
        }
        session_it->second->set_executor_device_type(ExecutorDeviceType::Hybrid);
        LOG(INFO) << "User " << user_name << " sets HYBRID mode.";
        break;
    }
  }

  void load_table_binary(const TSessionId session, const std::string &table_name, const std::vector<TRow> &rows) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    const TableDescriptor *td = cat.getMetadataForTable(table_name);
    if (td == nullptr) {
      TMapDException ex;
      ex.error_msg = "Table " + table_name + " does not exist.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    Importer_NS::Loader loader(cat, td);
    if (rows.front().cols.size() != static_cast<size_t>(td->nColumns)) {
      TMapDException ex;
      ex.error_msg = "Wrong number of columns to load into Table " + table_name;
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    auto col_descs = loader.get_column_descs();
    std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
    for (auto cd : col_descs) {
      import_buffers.push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(new Importer_NS::TypedImportBuffer(cd, loader.get_string_dict(cd))));
    }
    for (auto row : rows) {
      try {
        int col_idx = 0;
        for (auto cd : col_descs) {
          import_buffers[col_idx]->add_value(cd, row.cols[col_idx], row.cols[col_idx].is_null);
          col_idx++;
        }
      } catch (const std::exception &e) {
        LOG(WARNING) << "load_table exception thrown: " << e.what() << ". Row discarded.";
      }
    }
    if (loader.load(import_buffers, rows.size()))
      loader.checkpoint();
  }

  void load_table(const TSessionId session, const std::string &table_name, const std::vector<TStringRow> &rows) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    const TableDescriptor *td = cat.getMetadataForTable(table_name);
    if (td == nullptr) {
      TMapDException ex;
      ex.error_msg = "Table " + table_name + " does not exist.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    Importer_NS::Loader loader(cat, td);
    Importer_NS::CopyParams copy_params;
    if (rows.front().cols.size() != static_cast<size_t>(td->nColumns)) {
      TMapDException ex;
      ex.error_msg = "Wrong number of columns to load into Table " + table_name + " (" + std::to_string(rows.size()) + " vs " + std::to_string(td->nColumns) + ")";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    auto col_descs = loader.get_column_descs();
    std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
    for (auto cd : col_descs) {
      import_buffers.push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(new Importer_NS::TypedImportBuffer(cd, loader.get_string_dict(cd))));
    }
    for (auto row : rows) {
      try {
        int col_idx = 0;
        for (auto cd : col_descs) {
          import_buffers[col_idx]->add_value(cd, row.cols[col_idx].str_val, row.cols[col_idx].is_null, copy_params);
          col_idx++;
        }
      } catch (const std::exception &e) {
        LOG(WARNING) << "load_table exception thrown: " << e.what() << ". Row discarded.";
      }
    }
    if (loader.load(import_buffers, rows.size()))
      loader.checkpoint();
  }

  void detect_column_types(TRowSet& _return,
                           const TSessionId session,
                           const std::string& file_name,
                           const std::string& delimiter) {
    get_session(session);

    boost::filesystem::path file_path = file_name;  // FIXME
    if (!boost::filesystem::exists(file_path)) {
      TMapDException ex;
      ex.error_msg = "File does not exist.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }

    Importer_NS::Detector detector(file_path);
    std::vector<SQLTypes> best_types = detector.best_sqltypes;
    std::vector<std::string> headers = detector.get_headers();

    _return.row_desc.resize(best_types.size());
    TColumnType col;
    for (size_t col_idx = 0; col_idx < best_types.size(); col_idx++) {
      SQLTypes t = best_types[col_idx];
      SQLTypeInfo* ti = new SQLTypeInfo(t, true);
      col.col_type.type = type_to_thrift(*ti);
      col.col_name = headers[col_idx];
      _return.row_desc[col_idx] = col;
    }

    size_t num_samples = 10;
    auto sample_data = detector.get_sample_rows(num_samples);

    TRow sample_row;
    for (auto row : sample_data) {
      {
        std::vector<TDatum> empty;
        sample_row.cols.swap(empty);
      }
      for (const auto &s : row) {
        TDatum td;
        td.val.str_val = s;
        td.is_null = s.empty();
        sample_row.cols.push_back(td);
      }
      _return.rows.push_back(sample_row);
    }

  }

  void render(std::string& _return, const TSessionId session, const std::string &query, const std::string &render_type, const TRenderPropertyMap &render_properties, const TColumnRenderMap &col_render_properties) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    LOG(INFO) << "Render: " << query;
    SQLParser parser;
    std::list<Parser::Stmt*> parse_trees;
    std::string last_parsed;
    int num_parse_errors = 0;
    try {
      num_parse_errors = parser.parse(query, parse_trees, last_parsed);
    }
    catch (std::exception &e) {
        TMapDException ex;
        ex.error_msg = std::string("Exception: ") + e.what();
        LOG(ERROR) << ex.error_msg;
        throw ex;
    }
    if (num_parse_errors > 0) {
      TMapDException ex;
      ex.error_msg = "Syntax error at: " + last_parsed;
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    if (parse_trees.size() != 1) {
      TMapDException ex;
      ex.error_msg = "Can only render a single query at a time.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    Parser::Stmt *stmt = parse_trees.front();
    try {
      std::unique_ptr<Parser::Stmt> stmt_ptr(stmt);
      Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
      if (ddl != nullptr) {
        TMapDException ex;
        ex.error_msg = "Can only render SELECT statements.";
        LOG(ERROR) << ex.error_msg;
        throw ex;
      } else {
        auto dml = dynamic_cast<Parser::DMLStmt*>(stmt);
        Analyzer::Query query;
        dml->analyze(cat, query);
        if (query.get_stmt_type() != kSELECT) {
          TMapDException ex;
          ex.error_msg = "Can only render SELECT statements.";
          LOG(ERROR) << ex.error_msg;
          throw ex;
        }
        Planner::Optimizer optimizer(query, cat);
        auto root_plan = optimizer.optimize();
        std::unique_ptr<Planner::RootPlan> plan_ptr(root_plan);  // make sure it's deleted
        root_plan->set_render_type(render_type);
        root_plan->set_render_properties(&render_properties);
        root_plan->set_column_render_properties(&col_render_properties);
        root_plan->set_plan_dest(Planner::RootPlan::Dest::kRENDER);
        // @TODO(alex) execute query, render and fill _return
      }
    }
    catch (std::exception &e) {
        TMapDException ex;
        ex.error_msg = std::string("Exception: ") + e.what();
        LOG(ERROR) << ex.error_msg;
        throw ex;
    }
  }
  void create_frontend_view(const TSessionId session, const std::string &view_name, const std::string &view_state) {
    const auto session_info = get_session(session);
    auto &cat = session_info.get_catalog();
    FrontendViewDescriptor vd;
    vd.viewName = view_name;
    vd.viewState = view_state;

    cat.createFrontendView(vd);
  }

private:
  typedef std::map<TSessionId, std::shared_ptr<Catalog_Namespace::SessionInfo>> SessionMap;

  SessionMap::iterator get_session_it(const TSessionId session) {
    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end()) {
      TMapDException ex;
      ex.error_msg = "Session not valid.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    session_it->second->update_time();
    return session_it;
  }

  Catalog_Namespace::SessionInfo get_session(const TSessionId session) {
    mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
    return *get_session_it(session)->second;
  }

  std::unique_ptr<Catalog_Namespace::SysCatalog> sys_cat_;
  Data_Namespace::DataMgr* data_mgr_;
  std::map<TSessionId, std::shared_ptr<Catalog_Namespace::SessionInfo>> sessions_;
  std::map<std::string, std::shared_ptr<Catalog_Namespace::Catalog>> cat_map_;

  const std::string base_data_path_;
  ExecutorDeviceType executor_device_type_;
  std::default_random_engine random_gen_;
  std::uniform_int_distribution<int64_t> session_id_dist_;
  const bool jit_debug_;
  const bool allow_multifrag_;
  bool cpu_mode_only_;
  mapd_shared_mutex rw_mutex_;
};

int main(int argc, char **argv) {
  int port = 9091;
  std::string base_path;
  std::string device("gpu");
  bool flush_log = false;
  bool jit_debug = false;
  bool allow_multifrag = false;

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
		("path", po::value<std::string>(&base_path)->required(), "Directory path to Mapd catalogs")
    ("flush-log", "Force aggressive log file flushes.  Use when trouble-shooting.")
    ("jit-debug", "Enable debugger support for the JIT. The generated code can be found at /tmp/mapdquery")
    ("allow-multifrag", "Allow execution over multiple fragments in a single round-trip to GPU")
    ("cpu", "Run on CPU only")
    ("gpu", "Run on GPUs (Default)")
    ("hybrid", "Run on both CPU and GPUs")
    ("version,v", "Print Release Version Number")
    ("port,p", po::value<int>(&port), "Port number (default 9091)");

	po::positional_options_description positionalOptions;
	positionalOptions.add("path", 1);

	po::variables_map vm;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
		if (vm.count("help")) {
			std::cout << "Usage: mapd_server <catalog path> [<database name>] [--cpu|--gpu|--hybrid] [-p <port number>][--flush-log][--version|-v]\n";
			return 0;
		}
		if (vm.count("version")) {
			std::cout << "MapD Version: " << MapDRelease << std::endl;
      return 0;
    }
		if (vm.count("cpu"))
			device = "cpu";
    if (vm.count("gpu"))
      device = "gpu";
    if (vm.count("hybrid"))
      device = "hybrid";
    if (vm.count("flush-log"))
      flush_log = true;
    if (vm.count("jit-debug"))
      jit_debug = true;
    if (vm.count("allow-multifrag"))
      allow_multifrag = true;

		po::notify(vm);
	}
	catch (boost::program_options::error &e)
	{
		std::cerr << "Usage Error: " << e.what() << std::endl;
		return 1;
	}

  if (!boost::filesystem::exists(base_path)) {
    std::cerr << "Path " << base_path << " does not exist." << std::endl;
    return 1;
  }
  const auto system_db_file = boost::filesystem::path(base_path) / "mapd_catalogs" / "mapd";
  if (!boost::filesystem::exists(system_db_file)) {
    std::cerr << "MapD system catalogs does not exist at " << system_db_file << ". Run initdb" << std::endl;
    return 1;
  }
  const auto data_path = boost::filesystem::path(base_path) / "mapd_data";
  if (!boost::filesystem::exists(data_path)) {
    std::cerr << "MapD data directory does not exist at " << base_path << ". Run initdb" << std::endl;
    return 1;
  }
  const auto db_file = boost::filesystem::path(base_path) / "mapd_catalogs" / MAPD_SYSTEM_DB;
  if (!boost::filesystem::exists(db_file)) {
    std::cerr << "MapD database " << MAPD_SYSTEM_DB << " does not exist." << std::endl;
    return 1;
  }

  const auto lock_file = boost::filesystem::path(base_path) / "mapd_server_pid.lck";
  if (boost::filesystem::exists(lock_file)) {
    std::ifstream lockf;
    lockf.open(lock_file.c_str());
    pid_t pid;
    lockf >> pid;
    lockf.close();
    if (kill(pid, 0) == 0) {
      std::cerr << "Another MapD Server is running on the same MapD directory." << std::endl;
      return 1;
    }
  }
  std::ofstream lockf;
  lockf.open(lock_file.c_str(), std::ios::out | std::ios::trunc);
  lockf << getpid();
  lockf.close();

  while (true) {
    auto pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
      break;
    }
    for (auto fd = sysconf(_SC_OPEN_MAX); fd > 0; --fd) {
      close(fd);
    }
    int status { 0 };
    CHECK_NE(-1, waitpid(pid, &status, 0));
    LOG(ERROR) << "Server exit code: " << status;
  }

  const auto log_path = boost::filesystem::path(base_path) / "mapd_log";
  (void)boost::filesystem::create_directory(log_path);
  FLAGS_log_dir = log_path.c_str();
  if (flush_log)
    FLAGS_logbuflevel=-1;
  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);
  shared_ptr<MapDHandler> handler(new MapDHandler(base_path, device, allow_multifrag, jit_debug));
  shared_ptr<TProcessor> processor(new MapDProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
  TThreadedServer server (processor, serverTransport, transportFactory, protocolFactory);

  try {
    server.serve();
  } catch (std::exception &e) {
    LOG(ERROR) << "Exception: " << e.what() << std::endl;
  }
  return 0;
}
