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
#include "Shared/measure.h"
#include "MapDRelease.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <memory>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <random>
#include <map>
#include <glog/logging.h>


using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;


using boost::shared_ptr;

namespace {

TDatumType::type type_to_thrift(const SQLTypeInfo& type_info) {
  switch (type_info.get_type()) {
    case kBOOLEAN:
      return TDatumType::BOOL;
    case kSMALLINT:
    case kINT:
    case kBIGINT:
      return TDatumType::INT;
    case kNUMERIC:
    case kDECIMAL:
      CHECK(false);
    case kFLOAT:
    case kDOUBLE:
      return TDatumType::REAL;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return TDatumType::STR;
    case kTIME:
      return TDatumType::TIME;
    case kTIMESTAMP:
      return TDatumType::TIMESTAMP;
    case kDATE:
      return TDatumType::DATE;
    default:
      break;
  }
  CHECK(false);
}

}

#define INVALID_SESSION_ID  -1

class MapDHandler : virtual public MapDIf {
public:
  MapDHandler(const std::string& base_data_path, const std::string& executor_device, const bool jit_debug) : base_data_path_(base_data_path), random_gen_(std::random_device{}()), session_id_dist_(0, INT32_MAX), jit_debug_(jit_debug) {
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
    const auto system_db_file = boost::filesystem::path(base_data_path_) / "mapd_catalogs" / "mapd";
    const auto data_path = boost::filesystem::path(base_data_path_) / "mapd_data";
    data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string(), !cpu_mode_only_)); // second param is whether to initialize GPU buffer pool
    sys_cat_.reset(new Catalog_Namespace::SysCatalog(base_data_path_, *data_mgr_));
  }
  ~MapDHandler() {
    LOG(INFO) << "mapd_server exits." << std::endl;
  }

  TSessionId connect(const std::string &user, const std::string &passwd, const std::string &dbname) {
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
      Catalog_Namespace::Catalog *cat = new Catalog_Namespace::Catalog(base_data_path_, db_meta, *data_mgr_);
      cat_map_[dbname].reset(cat);
      sessions_[session].reset(new Catalog_Namespace::SessionInfo(cat_map_[dbname], user_meta));;
    } else
      sessions_[session].reset(new Catalog_Namespace::SessionInfo(cat_it->second, user_meta));
    LOG(INFO) << "User " << user << " connected to database " << dbname << std::endl;
    return session;
  }

  void disconnect(const TSessionId session) {
    auto session_it = sessions_.find(session);
    std::string dbname;
    if (session_it == sessions_.end()) {
      TMapDException ex;
      ex.error_msg = "Session not valid.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    dbname = session_it->second->get_catalog().get_currentDB().dbName;
    sessions_.erase(session_it);
    LOG(INFO) << "User " << session_it->second->get_currentUser().userName << " disconnected from database " << dbname << std::endl;
  }

  void sql_execute(TQueryResult& _return, const TSessionId session, const std::string& query_str) {

    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end()) {
      TMapDException ex;
      ex.error_msg = "Session not valid.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    auto &cat = session_it->second->get_catalog();
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
      if (ddl != nullptr) {
        execute_time += measure<>::execution([&]() {
          ddl->execute(*session_it->second);
        });
      } else {
        auto dml = dynamic_cast<Parser::DMLStmt*>(stmt);
        Analyzer::Query query;
        dml->analyze(cat, query);
        Planner::Optimizer optimizer(query, cat);
        auto root_plan = optimizer.optimize();
        std::unique_ptr<Planner::RootPlan> plan_ptr(root_plan);  // make sure it's deleted
        auto executor = Executor::getExecutor(root_plan->get_catalog().get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "");
        ResultRows results({}, nullptr, nullptr);
        execute_time += measure<>::execution([&]() {
          results = executor->execute(root_plan,true,executor_device_type_);
        });
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
            proj_info.col_type.nullable = !target_ti.get_notnull();
            _return.row_set.row_desc.push_back(proj_info);
            ++i;
          }
        }
        for (size_t row_idx = 0; row_idx < results.size(); ++row_idx) {
          TRow trow;
          for (size_t i = 0; i < results.colCount(); ++i) {
            TColumnValue col_val;
            const auto agg_result = results.get(row_idx, i, true);
            if (boost::get<int64_t>(&agg_result)) {
              col_val.datum.int_val = *(boost::get<int64_t>(&agg_result));
              switch (targets[i]->get_expr()->get_type_info().get_type()) {
                case kBOOLEAN:
                  col_val.is_null = (col_val.datum.int_val == NULL_BOOLEAN);
                  break;
                case kSMALLINT:
                  col_val.is_null = (col_val.datum.int_val == NULL_SMALLINT);
                  break;
                case kINT:
                  col_val.is_null = (col_val.datum.int_val == NULL_INT);
                  break;
                case kBIGINT:
                  col_val.is_null = (col_val.datum.int_val == NULL_BIGINT);
                  break;
                case kTIME:
                case kTIMESTAMP:
                case kDATE:
                  if (sizeof(time_t) == 4)
                    col_val.is_null = (col_val.datum.int_val == NULL_INT);
                  else
                    col_val.is_null = (col_val.datum.int_val == NULL_BIGINT);
                  break;
                default:
                  col_val.is_null = false;
              }
            } else if (boost::get<double>(&agg_result)) {
              col_val.datum.real_val = *(boost::get<double>(&agg_result));
              if (targets[i]->get_expr()->get_type_info().get_type() == kFLOAT) {
                col_val.is_null = (col_val.datum.real_val == NULL_FLOAT);
              } else {
                col_val.is_null = (col_val.datum.real_val == NULL_DOUBLE);
              }
            } else {
              auto s = boost::get<std::string>(&agg_result);
              if (s) {
                col_val.datum.str_val = *s;
              } else {
                auto null_p = boost::get<void*>(&agg_result);
                CHECK(null_p && !*null_p);
              }
              col_val.is_null = !s;
            }
            trow.cols.push_back(col_val);
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
    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end()) {
      TMapDException ex;
      ex.error_msg = "Session not valid.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    auto &cat = session_it->second->get_catalog(); 
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
      col_type.col_type.nullable = !cd->columnType.get_notnull();
      _return.insert(std::make_pair(cd->columnName, col_type));
    }
  }

  void get_tables(std::vector<std::string> & table_names, const TSessionId session) {
    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end()) {
      TMapDException ex;
      ex.error_msg = "Session not valid.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    auto &cat = session_it->second->get_catalog(); 
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

  void set_execution_mode(const TExecuteMode::type mode) {
    switch (mode) {
      case TExecuteMode::GPU:
        if (cpu_mode_only_) {
          TMapDException e;
          e.error_msg = "Cannot switch to GPU mode in a server started in CPU-only mode.";
          throw e;
        }
        executor_device_type_ = ExecutorDeviceType::GPU;
        LOG(INFO) << "Client sets GPU mode.";
        break;
      case TExecuteMode::CPU:
        LOG(INFO) << "Client sets CPU mode.";
        executor_device_type_ = ExecutorDeviceType::CPU;
        break;
      case TExecuteMode::HYBRID:
        if (cpu_mode_only_) {
          TMapDException e;
          e.error_msg = "Cannot switch to Hybrid mode in a server started in CPU-only mode.";
          throw e;
        }
        LOG(INFO) << "Client sets Hybrid mode.";
        executor_device_type_ = ExecutorDeviceType::Hybrid;
        break;
    }
  }

  TLoadId start_load(const TSessionId session, const std::string &table_name) {
    return 0;
  }

  void load_table(const TSessionId session, const TLoadId load, const TRowSet &rows) {
  }

  void end_load(const TSessionId session, const TLoadId load) {
  }

private:
  std::unique_ptr<Catalog_Namespace::SysCatalog> sys_cat_;
  std::unique_ptr<Data_Namespace::DataMgr> data_mgr_;
  std::map<TSessionId, std::shared_ptr<Catalog_Namespace::SessionInfo>> sessions_;
  std::map<std::string, std::shared_ptr<Catalog_Namespace::Catalog>> cat_map_;

  const std::string base_data_path_;
  ExecutorDeviceType executor_device_type_;
  std::default_random_engine random_gen_;
  std::uniform_int_distribution<int64_t> session_id_dist_;
  bool jit_debug_;
  bool cpu_mode_only_;
};

int main(int argc, char **argv) {
  int port = 9091;
  std::string base_path;
  std::string device("gpu");
  bool flush_log = false;
  bool jit_debug = false;

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
		("path", po::value<std::string>(&base_path)->required(), "Directory path to Mapd catalogs")
    ("flush-log", "Force aggressive log file flushes.  Use when trouble-shooting.")
    ("jit-debug", "Enable debugger support for the JIT. The generated code can be found at /tmp/mapdquery")
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
			std::cout << "Usage: mapd_server <catalog path> [<database name>] [--cpu|--gpu] [-p <port number>][--flush-log][--version|-v]\n";
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
  shared_ptr<MapDHandler> handler(new MapDHandler(base_path, device, jit_debug));
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
