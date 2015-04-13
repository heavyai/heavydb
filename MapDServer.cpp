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

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <memory>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <random>
#include <map>


using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;


using boost::shared_ptr;

namespace {

TDatumType::type type_to_thrift(const SQLTypeInfo& type_info) {
  switch (type_info.get_type()) {
    case kBOOLEAN:
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

double dtime() {
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime,(struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return (tseconds);
}

#define INVALID_SESSION_ID  -1

class MapDHandler : virtual public MapDIf {
public:
  MapDHandler(const std::string& base_data_path, const std::string& executor_device) : base_data_path_(base_data_path), random_gen_(std::random_device{}()), session_id_dist_(0, INT64_MAX) {
    if (executor_device == "gpu") {
        executor_device_type_ = ExecutorDeviceType::GPU;
        std::cout << "GPU Mode" << std::endl; 
    }
    else if (executor_device == "auto") {
        executor_device_type_ = ExecutorDeviceType::Auto;
        std::cout << "Auto Mode" << std::endl; 
    }
    else {
        executor_device_type_ = ExecutorDeviceType::CPU;
        std::cout << "CPU Mode" << std::endl; 
    }
    const auto system_db_file = boost::filesystem::path(base_data_path_) / "mapd_catalogs" / "mapd";
    const auto data_path = boost::filesystem::path(base_data_path_) / "mapd_data";
    data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string(), executor_device_type_ == ExecutorDeviceType::GPU || executor_device_type_ == ExecutorDeviceType::Auto)); // second param is whether to initialize GPU buffer pool
    sys_cat_.reset(new Catalog_Namespace::SysCatalog(base_data_path_, *data_mgr_));
    logFile.open("mapd_log.txt", std::ios::out | std::ios::app);
  }

  SessionId connect(const std::string &user, const std::string &passwd, const std::string &dbname) {
    Catalog_Namespace::UserMetadata user_meta;
    if (!sys_cat_->getMetadataForUser(user, user_meta)) {
      MapDException ex;
      ex.error_msg = std::string("User ") + user + " does not exist.";
      throw ex;
    }
    if (user_meta.passwd != passwd) {
      MapDException ex;
      ex.error_msg = std::string("Password for User ") + user + " is incorrect.";
      throw ex;
    }
    Catalog_Namespace::DBMetadata db_meta;
    if (!sys_cat_->getMetadataForDB(dbname, db_meta)) {
      MapDException ex;
      ex.error_msg = std::string("Database ") + dbname + " does not exist.";
      throw ex;
    }
    if (!user_meta.isSuper && user_meta.userId != db_meta.dbOwner) {
      MapDException ex;
      ex.error_msg = std::string("User ") + user + " is not authorized to access database " + dbname;
      throw ex;
    }
    SessionId session = INVALID_SESSION_ID;
    while (true) {
      session = session_id_dist_(random_gen_);
      auto session_it = sessions_.find(session);
      if (session_it == sessions_.end())
        break;
    }
    auto cat_it = cat_map_.find(dbname);
    if (cat_it == cat_map_.end()) {
      Catalog_Namespace::Catalog *cat = new Catalog_Namespace::Catalog(base_data_path_, user_meta, db_meta, *data_mgr_);
      cat_map_[dbname].reset(cat);
      sessions_[session] = cat_map_[dbname];
    } else
      sessions_[session] = cat_it->second;
    return session;
  }

  void disconnect(const SessionId session) {
    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end()) {
      MapDException ex;
      ex.error_msg = "Session not valid.";
      throw ex;
    }
    sessions_.erase(session_it);
  }

  void select(QueryResult& _return, const SessionId session, const std::string& query_str) {

    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end()) {
      MapDException ex;
      ex.error_msg = "Session not valid.";
      throw ex;
    }
    auto cat = session_it->second.get(); 
    CHECK(cat);
    logFile << query_str << '\t';
    double tStart = dtime();
    SQLParser parser;
    std::list<Parser::Stmt*> parse_trees;
    std::string last_parsed;
    int num_errors = parser.parse(query_str, parse_trees, last_parsed);
    if (num_errors > 0) {
      MapDException ex;
      ex.error_msg = "Syntax error at: " + last_parsed;
      throw ex;
    }
    for (auto stmt : parse_trees) {
      try {
      std::unique_ptr<Parser::Stmt> stmt_ptr(stmt);
      Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
      if (ddl != nullptr) {
        ddl->execute(*cat);
      } else {
        auto dml = dynamic_cast<Parser::DMLStmt*>(stmt);
        Analyzer::Query query;
        dml->analyze(*cat, query);
        Planner::Optimizer optimizer(query, *cat);
        auto root_plan = optimizer.optimize();
        std::unique_ptr<Planner::RootPlan> plan_ptr(root_plan);  // make sure it's deleted
        auto executor = Executor::getExecutor(root_plan->get_catalog().get_currentDB().dbId);
        const auto results = executor->execute(root_plan,true,executor_device_type_);
        const auto plan = root_plan->get_plan();
        const auto& targets = plan->get_targetlist();
        {
          CHECK(plan);
          ProjInfo proj_info;
          size_t i = 0;
          for (const auto target : targets) {
            proj_info.proj_name = target->get_resname();
            if (proj_info.proj_name.empty()) {
              proj_info.proj_name = std::to_string(i);
            }
            const auto& target_ti = target->get_expr()->get_type_info();
            proj_info.proj_type.type = type_to_thrift(target_ti);
            proj_info.proj_type.nullable = !target_ti.get_notnull();
            _return.proj_info.push_back(proj_info);
            ++i;
          }
        }
        for (const auto& row : results) {
          TResultRow trow;
          for (size_t i = 0; i < row.size(); ++i) {
            ColumnValue col_val;
            const auto agg_result = row.agg_result(i);
            if (boost::get<int64_t>(&agg_result)) {
              col_val.type = type_to_thrift(row.agg_type(i));
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
              col_val.type = TDatumType::REAL;
              col_val.datum.real_val = *(boost::get<double>(&agg_result));
              if (targets[i]->get_expr()->get_type_info().get_type() == kFLOAT) {
                col_val.is_null = (col_val.datum.real_val == NULL_FLOAT);
              } else {
                col_val.is_null = (col_val.datum.real_val == NULL_DOUBLE);
              }
            } else {
              auto s = boost::get<std::string>(&agg_result);
              CHECK(s);
              col_val.type = TDatumType::STR;
              col_val.datum.str_val = *s;
              col_val.is_null = s->empty();
            }
            trow.cols.push_back(col_val);
          }
          _return.rows.push_back(trow);
        }
      }
    }
    catch (std::exception &e) {
        MapDException ex;
        ex.error_msg = std::string("Exception: ") + e.what();
        throw ex;
    }
    }
    double tStop = dtime();
    double tElapsed = (tStop - tStart) * 1000;
    logFile << tElapsed << "\n";
    
  }

  void getColumnTypes(ColumnTypes& _return, const SessionId session, const std::string& table_name) {
    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end()) {
      MapDException ex;
      ex.error_msg = "Session not valid.";
      throw ex;
    }
    auto cat = session_it->second.get(); 
    CHECK(cat);
    auto td = cat->getMetadataForTable(table_name);
    if (!td) {
      MapDException ex;
      ex.error_msg = "Table doesn't exist";
      throw ex;
    }
    const auto col_descriptors = cat->getAllColumnMetadataForTable(td->tableId);
    for (const auto cd : col_descriptors) {
      ColumnType col_type;
      col_type.type = type_to_thrift(cd->columnType);
      col_type.nullable = !cd->columnType.get_notnull();
      _return.insert(std::make_pair(cd->columnName, col_type));
    }
  }

  void getTables(std::vector<std::string> & table_names, const SessionId session) {
    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end()) {
      MapDException ex;
      ex.error_msg = "Session not valid.";
      throw ex;
    }
    auto cat = session_it->second.get(); 
    CHECK(cat);
    const auto tables = cat->getAllTableMetadata();
    for (const auto td : tables) {
      table_names.push_back(td->tableName);
    }
  }

  void getUsers(std::vector<std::string> &user_names) {
    std::list<Catalog_Namespace::UserMetadata> user_list = sys_cat_->getAllUserMetadata();
    for (auto u : user_list) {
      user_names.push_back(u.userName);
    }
  }

  void getDatabases(std::vector<DBInfo> &dbinfos) {
    std::list<Catalog_Namespace::DBMetadata> db_list = sys_cat_->getAllDBMetadata();
    std::list<Catalog_Namespace::UserMetadata> user_list = sys_cat_->getAllUserMetadata();
    for (auto d : db_list) {
      DBInfo dbinfo;
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

private:
  std::unique_ptr<Catalog_Namespace::SysCatalog> sys_cat_;
  std::unique_ptr<Data_Namespace::DataMgr> data_mgr_;
  std::map<SessionId, std::shared_ptr<Catalog_Namespace::Catalog>> sessions_;
  std::map<std::string, std::shared_ptr<Catalog_Namespace::Catalog>> cat_map_;
  std::ofstream logFile;

  const std::string base_data_path_;
  ExecutorDeviceType executor_device_type_;
  std::default_random_engine random_gen_;
  std::uniform_int_distribution<int64_t> session_id_dist_;
};

int main(int argc, char **argv) {
  int port = 9091;
  std::string base_path;
  std::string db_name(MAPD_SYSTEM_DB);
  std::string device("auto");

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
		("path", po::value<std::string>(&base_path)->required(), "Directory path to Mapd catalogs")
		("db", po::value<std::string>(&db_name), "Database name")
    ("cpu", "Run on CPU only")
    ("gpu", "Run on GPUs")
    ("port,p", po::value<int>(&port), "Port number (default 9091)");

	po::positional_options_description positionalOptions;
	positionalOptions.add("path", 1);
	positionalOptions.add("db", 1);

	po::variables_map vm;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
		if (vm.count("help")) {
			std::cout << "Usage: mapd_server <catalog path> [<database name>] [--cpu|--gpu] [-p <port number>]\n";
			return 0;
		}
		if (vm.count("cpu"))
			device = "cpu";
    if (vm.count("gpu"))
      device = "gpu";

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
  const auto db_file = boost::filesystem::path(base_path) / "mapd_catalogs" / db_name;
  if (!boost::filesystem::exists(db_file)) {
    std::cerr << "MapD database " << db_name << " does not exist." << std::endl;
    return 1;
  }
  shared_ptr<MapDHandler> handler(new MapDHandler(base_path, device));
  shared_ptr<TProcessor> processor(new MapDProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
  TThreadedServer server (processor, serverTransport, transportFactory, protocolFactory);

  try {
    server.serve();
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }
  return 0;
}
