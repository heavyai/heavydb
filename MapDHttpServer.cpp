#include "gen-cpp/MapD.h"
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/THttpServer.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include <boost/program_options.hpp>
#include <string>
#include <iostream>


using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;


using boost::shared_ptr;

class MapDHandler : virtual public MapDIf {
public:
  MapDHandler(MapDClient &client) : client_(client) {}

  void select(QueryResult& _return, const std::string& query_str) {
    try {
    client_.select(_return, query_str);
    }
    catch (std::exception &e) {
      MapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void getColumnTypes(ColumnTypes& _return, const std::string& table_name) {
    try {
    client_.getColumnTypes(_return, table_name);
    }
    catch (std::exception &e) {
      MapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void getTables(std::vector<std::string> & _return)
  {
    try {
    client_.getTables(_return);
    }
    catch (std::exception &e) {
      MapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

private:
  MapDClient &client_;
};

int main(int argc, char **argv) {
  int port = 9091;
  int server_port = 9090;

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
    ("server,s", po::value<int>(&server_port), "Server port number (default 9090)")
    ("port,p", po::value<int>(&port), "Port number (default 9091)");

	po::variables_map vm;

	try {
		if (vm.count("help")) {
			std::cout << "Usage: mapd_http_server [{-p|--port} <port number>] [{-s|--server} <port number>]\n";
			return 0;
		}

		po::notify(vm);
	}
	catch (boost::program_options::error &e)
	{
		std::cerr << "Usage Error: " << e.what() << std::endl;
		return 1;
	}

  shared_ptr<TTransport> socket(new TSocket("localhost", server_port));
  shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  MapDClient client(protocol);
  shared_ptr<MapDHandler> handler(new MapDHandler(client));
  shared_ptr<TProcessor> processor(new MapDProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new THttpServerTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TJSONProtocolFactory());
  TThreadedServer server (processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return 0;
}
