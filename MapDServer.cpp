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

namespace {

TDatumType::type type_to_thrift(const SQLTypes type) {
  if (IS_INTEGER(type)) {
    return TDatumType::INT;
  }
  if (IS_STRING(type)) {
    return TDatumType::STR;
  }
  if (type == kFLOAT || type == kDOUBLE) {
    return TDatumType::REAL;
  }
  if (IS_TIME(type)) {
    return TDatumType::TIME;
  }
  CHECK(false);
}

}
class MapDHandler : virtual public MapDIf {
public:
  MapDHandler() {
    CHECK(boost::filesystem::exists(base_path_));
    const auto system_db_file = boost::filesystem::path(base_path_) / "mapd_catalogs" / "mapd";
    CHECK(boost::filesystem::exists(system_db_file));
    const auto data_path = boost::filesystem::path(base_path_) / "mapd_data";
    data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string()));
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
      MapDException ex;
      ex.error_msg = "Syntax error at: " + last_parsed;
      throw ex;
    }
    for (auto stmt : parse_trees) {
      std::unique_ptr<Parser::Stmt> stmt_ptr(stmt);
      Parser::DDLStmt *ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
      if (ddl != nullptr) {
        MapDException ex;
        ex.error_msg = "Query not allowed";
        throw ex;
      } else {
        auto dml = dynamic_cast<Parser::DMLStmt*>(stmt);
        Analyzer::Query query;
        dml->analyze(*cat_, query);
        Planner::Optimizer optimizer(query, *cat_);
        auto root_plan = optimizer.optimize();
        std::unique_ptr<Planner::RootPlan> plan_ptr(root_plan);  // make sure it's deleted
        auto executor = Executor::getExecutor(root_plan->get_catalog().get_currentDB().dbId);
        const auto results = executor->execute(root_plan);
        {
          const auto plan = root_plan->get_plan();
          CHECK(plan);
          const auto& targets = plan->get_targetlist();
          ProjInfo proj_info;
          size_t i = 0;
          for (const auto target : targets) {
            proj_info.proj_name = target->get_resname();
            if (proj_info.proj_name.empty()) {
              proj_info.proj_name = std::to_string(i);
            }
            const auto& target_type = target->get_expr()->get_type_info();
            proj_info.proj_type.type = type_to_thrift(target_type.type);
            proj_info.proj_type.nullable = !target_type.notnull;
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
            } else if (boost::get<double>(&agg_result)) {
              col_val.type = TDatumType::REAL;
              col_val.datum.real_val = *(boost::get<double>(&agg_result));
            } else {
              auto s = boost::get<std::string>(&agg_result);
              CHECK(s);
              col_val.type = TDatumType::STR;
              col_val.datum.str_val = *s;
            }
            trow.push_back(col_val);
          }
          _return.rows.push_back(trow);
        }
      }
    }
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
      col_type.type = type_to_thrift(cd->columnType.type);
      col_type.nullable = !cd->columnType.notnull;
      _return.insert(std::make_pair(cd->columnName, col_type));
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
