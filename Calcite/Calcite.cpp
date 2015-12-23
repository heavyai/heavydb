/*
 *  Some cool MapD License
 */

/*
 * File:   Calcite.cpp
 * Author: michael
 *
 * Created on November 23, 2015, 9:33 AM
 */

#include "Calcite.h"
#include "Shared/measure.h"

#include <glog/logging.h>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/CalciteServer.h"

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

// Calcite::Calcite(){
// LOG(INFO) << "Creating Calcite Class"  << std::endl;
//}

Calcite::Calcite(int port) : server_available(true) {
  LOG(INFO) << "Creating Calcite Class with port " << port << std::endl;
  boost::shared_ptr<TTransport> socket(new TSocket("localhost", port));
  boost::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  client.reset(new CalciteServerClient(protocol));

  try {
    transport->open();

    auto ms = measure<>::execution([&]() { client->ping(); });

    LOG(INFO) << "ping took " << ms << " ms " << endl;

  } catch (TException& tx) {
    LOG(ERROR) << tx.what() << endl;
    server_available = false;
  }
}

string Calcite::process(string user, string passwd, string catalog, string sql_string) {
  if (server_available) {
    LOG(INFO) << "User " << user << " catalog " << catalog << " sql " << sql_string << endl;
    TPlanResult ret;
    auto ms = measure<>::execution([&]() { client->process(ret, user, passwd, catalog, sql_string); });
    LOG(INFO) << ret.plan_result << endl;
    LOG(INFO) << "Time in Thrift " << (ms > ret.execution_time_ms ? ms - ret.execution_time_ms : 0)
              << " (ms), Time in Java Calcite server " << ret.execution_time_ms << " (ms)" << endl;
    return ret.plan_result;
  } else {
    LOG(INFO) << "Not routing to Calcite server, server is not up" << endl;
    return "";
  }
}

Calcite::~Calcite() {
  LOG(INFO) << "Destroy Calcite Class" << std::endl;
}
