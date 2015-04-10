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

class MapDHandler : virtual public MapDIf {
public:
  MapDHandler(const std::string& base_data_path, const std::string &db_name, const std::string& executor_device) : base_data_path_(base_data_path), db_name_(db_name) {
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
    std::cout << "MapDHandler initialized" << std::endl; 
    const auto system_db_file = boost::filesystem::path(base_data_path_) / "mapd_catalogs" / "mapd";
    const auto data_path = boost::filesystem::path(base_data_path_) / "mapd_data";
    data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string()));
    Catalog_Namespace::SysCatalog sys_cat(base_data_path_, *data_mgr_);
    Catalog_Namespace::UserMetadata user_meta;
    CHECK(sys_cat.getMetadataForUser(user_, user_meta));
    CHECK_EQ(user_meta.passwd, pass_);
    Catalog_Namespace::DBMetadata db_meta;
    CHECK(sys_cat.getMetadataForDB(db_name_, db_meta));
    CHECK(user_meta.isSuper || user_meta.userId == db_meta.dbOwner);
    cat_.reset(new Catalog_Namespace::Catalog(base_data_path_, user_meta, db_meta, *data_mgr_));
    logFile.open("mapd_log.txt", std::ios::out | std::ios::app);
  }

  void select(QueryResult& _return, const std::string& query_str) {

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
        ddl->execute(*cat_);
      } else {
        auto dml = dynamic_cast<Parser::DMLStmt*>(stmt);
        Analyzer::Query query;
        dml->analyze(*cat_, query);
        Planner::Optimizer optimizer(query, *cat_);
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

  void getColumnTypes(ColumnTypes& _return, const std::string& table_name) {
    auto td = cat_->getMetadataForTable(table_name);
    if (!td) {
      MapDException ex;
      ex.error_msg = "Table doesn't exist";
      throw ex;
    }
    const auto col_descriptors = cat_->getAllColumnMetadataForTable(td->tableId);
    for (const auto cd : col_descriptors) {
      ColumnType col_type;
      col_type.type = type_to_thrift(cd->columnType);
      col_type.nullable = !cd->columnType.get_notnull();
      _return.insert(std::make_pair(cd->columnName, col_type));
    }
  }

  void getTables(std::vector<std::string> & table_names) {
    const auto tables = cat_->getAllTableMetadata();
    for (const auto td : tables) {
      table_names.push_back(td->tableName);
    }
  }

private:
  std::unique_ptr<Catalog_Namespace::Catalog> cat_;
  std::unique_ptr<Data_Namespace::DataMgr> data_mgr_;
  std::ofstream logFile;


  const std::string base_data_path_;
  const std::string db_name_;
  const std::string user_ { MAPD_ROOT_USER };
  const std::string pass_ { MAPD_ROOT_PASSWD_DEFAULT };
  ExecutorDeviceType executor_device_type_;
  //const std::string executor_device_type_;
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
  shared_ptr<MapDHandler> handler(new MapDHandler(base_path, db_name, device));
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
