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
  MapDHandler(TTransport &transport, MapDClient &client) : client_(client), transport_(transport) {}

  SessionId connect(const std::string &user, const std::string &passwd, const std::string &dbname) {
    SessionId session = -1;
    try {
      session = client_.connect(user, passwd, dbname);
    }
    catch (TException &te) {
      try {
        transport_.open();
        session = client_.connect(user, passwd, dbname);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
    catch (MapDException &e) {
      throw e;
    }
    catch (std::exception &e) {
      std::cerr << "connect caught exception: " << e.what() << std::endl;
      MapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
    return session;
  }

  void disconnect(SessionId session) {
    try {
      client_.disconnect(session);
    }
    catch (TException &te) {
      std::cerr << "Thrift exception: " << te.what() << std::endl;
    }
    catch (std::exception &e) {
      std::cerr << "disconnect caught exception: " << e.what() << std::endl;
    }
  }

  void sql_execute(QueryResult& _return, const SessionId session, const std::string& query_str) {
    try {
      client_.sql_execute(_return, session, query_str);
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.sql_execute(_return, session, query_str);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
      catch (MapDException &e) {
        throw e;
      }
      catch (std::exception &e) {
        std::cerr << "select caught exception: " << e.what() << std::endl;
        MapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    }
    catch (MapDException &e) {
      throw e;
    }
    catch (std::exception &e) {
      std::cerr << "select caught exception: " << e.what() << std::endl;
      MapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void getColumnTypes(ColumnTypes& _return, const SessionId session, const std::string& table_name) {
    try {
      client_.getColumnTypes(_return, session, table_name);
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.getColumnTypes(_return, session, table_name);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
      catch (MapDException &e) {
        throw e;
      }
      catch (std::exception &e) {
        std::cerr << "getColumnTypes caught exception: " << e.what() << std::endl;
        MapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    }
    catch (MapDException &e) {
      throw e;
    }
    catch (std::exception &e) {
      std::cerr << "getColumnTypes caught exception: " << e.what() << std::endl;
      MapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void getTables(std::vector<std::string> & _return, const SessionId session)
  {
    try {
      client_.getTables(_return, session);
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.getTables(_return, session);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
      catch (MapDException &e) {
        throw e;
      }
      catch (std::exception &e) {
        std::cerr << "getTables caught exception: " << e.what() << std::endl;
        MapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    }
    catch (MapDException &e) {
      throw e;
    }
    catch (std::exception &e) {
      std::cerr << "getTables caught exception: " << e.what() << std::endl;
      MapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void getUsers(std::vector<std::string> & _return)
  {
    try {
      client_.getUsers(_return);
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.getUsers(_return);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
  }

  void getDatabases(std::vector<DBInfo> & _return)
  {
    try {
      client_.getDatabases(_return);
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.getDatabases(_return);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
  }

  void set_execution_mode(const TExecuteMode::type mode) {
    try {
      client_.set_execution_mode(mode);
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.set_execution_mode(mode);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
  }

private:
  MapDClient &client_;
  TTransport &transport_;
};

int main(int argc, char **argv) {
  int port = 9090;
  int server_port = 9091;

	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
		("help,h", "Print help messages ")
    ("server,s", po::value<int>(&server_port), "MapD Server port number (default 9091)")
    ("port,p", po::value<int>(&port), "Port number (default 9090)");

	po::variables_map vm;
	po::positional_options_description positionalOptions;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
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
  transport->open();
  shared_ptr<MapDHandler> handler(new MapDHandler(*transport, client));
  shared_ptr<TProcessor> processor(new MapDProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new THttpServerTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TJSONProtocolFactory());
  TThreadedServer server (processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  transport->close();
  return 0;
}
