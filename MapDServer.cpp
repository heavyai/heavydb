#include "gen-cpp/MapD.h"
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

#include "Catalog/Catalog.h"
#include "QueryEngine/Execute.h"
#include "Parser/parser.h"
#include "Planner/Planner.h"

#include <boost/filesystem.hpp>
#include <memory>
#include <string>


using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;

using boost::shared_ptr;

class MapDHandler : virtual public MapDIf {
public:
  MapDHandler() {
    CHECK(boost::filesystem::exists(base_path_));
    const auto system_db_file = boost::filesystem::path(base_path_) / "mapd_catalogs" / "mapd";
    CHECK(boost::filesystem::exists(system_db_file));
    const auto data_path = boost::filesystem::path(base_path_) / "mapd_data";
    data_mgr_.reset(new Data_Namespace::DataMgr(2, data_path.string()));
    Catalog_Namespace::SysCatalog sys_cat(base_path_, *data_mgr_);
    Catalog_Namespace::UserMetadata user_meta;
    CHECK(sys_cat.getMetadataForUser(user_, user_meta));
    CHECK_EQ(user_meta.passwd, pass_);
    Catalog_Namespace::DBMetadata db_meta;
    CHECK(sys_cat.getMetadataForDB(db_name_, db_meta));
    CHECK(user_meta.isSuper || user_meta.userId == db_meta.dbOwner);
    cat_.reset(new Catalog_Namespace::Catalog(base_path_, user_meta, db_meta, *data_mgr_));
  }

  void select(QueryResult& _return, const std::string& query_str) {
    SQLParser parser;
    std::list<Parser::Stmt*> parse_trees;
    std::string last_parsed;
    int num_errors = parser.parse(query_str, parse_trees, last_parsed);
    if (num_errors > 0) {
      throw std::runtime_error("Syntax error at: " + last_parsed);
    }
    for (auto stmt : parse_trees) {
      std::unique_ptr<Parser::Stmt> stmt_ptr(stmt);
      Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
      if (ddl != nullptr) {
        throw std::runtime_error("Query not allowed");
      } else {
        auto dml = dynamic_cast<Parser::DMLStmt*>(stmt);
        Analyzer::Query query;
        dml->analyze(*cat_, query);
        Planner::Optimizer optimizer(query, *cat_);
        auto plan = optimizer.optimize();
        std::unique_ptr<Planner::RootPlan> plan_ptr(plan);  // make sure it's deleted
        Executor executor(plan);
        const auto results = executor.execute();
        if (!results.empty()) {
          for (size_t i = 0; i < results.front().size(); ++i) {
            _return.proj_names.push_back(std::to_string(i));
          }
        }
        for (const auto& row : results) {
          TResultRow trow;
          for (size_t i = 0; i < row.size(); ++i) {
            TDatum datum;
            const auto agg_result = row.agg_result(i);
            if (boost::get<int64_t>(&agg_result)) {
              datum.type = TDatumType::INT;
              datum.int_val = *(boost::get<int64_t>(&agg_result));
            } else {
              const auto p = boost::get<double>(&agg_result);
              CHECK(p);
              datum.type = TDatumType::REAL;
              datum.real_val = *p;
            }
            trow.push_back(datum);
          }
          _return.rows.push_back(trow);
        }
      }
    }
  }
private:
  std::unique_ptr<Catalog_Namespace::Catalog> cat_;
  std::unique_ptr<Data_Namespace::DataMgr> data_mgr_;

  const std::string db_name_ { MAPD_SYSTEM_DB };
  const std::string user_ { MAPD_ROOT_USER };
  const std::string pass_ { MAPD_ROOT_PASSWD_DEFAULT };
  const std::string base_path_ { "/tmp" };
};

int main(int argc, char **argv) {
  int port = 9090;
  shared_ptr<MapDHandler> handler(new MapDHandler());
  shared_ptr<TProcessor> processor(new MapDProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TSimpleServer server(processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return 0;
}
