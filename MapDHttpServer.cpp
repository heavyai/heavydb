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

#include "MapDRelease.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;


using boost::shared_ptr;

class MapDHandler : virtual public MapDIf {
public:
  MapDHandler(TTransport &transport, MapDClient &client) : client_(client), transport_(transport) {}

  TSessionId connect(const std::string &user, const std::string &passwd, const std::string &dbname) {
    TSessionId session = -1;
    try {
      session = client_.connect(user, passwd, dbname);
    }
    catch (TMapDException &e) {
      throw e;
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
    catch (std::exception &e) {
      std::cerr << "connect caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
    return session;
  }

  void disconnect(TSessionId session) {
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

  void sql_execute(TQueryResult& _return, const TSessionId session, const std::string& query_str) {
    try {
      client_.sql_execute(_return, session, query_str);
    }
    catch (TMapDException &e) {
      throw e;
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.sql_execute(_return, session, query_str);
      }
      catch (TMapDException &e) {
        throw e;
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
      catch (std::exception &e) {
        std::cerr << "select caught exception: " << e.what() << std::endl;
        TMapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    }
    catch (std::exception &e) {
      std::cerr << "select caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void get_table_descriptor(TTableDescriptor& _return, const TSessionId session, const std::string& table_name) {
    try {
      client_.get_table_descriptor(_return, session, table_name);
    }
    catch (TMapDException &e) {
      throw e;
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.get_table_descriptor(_return, session, table_name);
      }
      catch (TMapDException &e) {
        throw e;
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
      catch (std::exception &e) {
        std::cerr << "get_table_descriptor caught exception: " << e.what() << std::endl;
        TMapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    }
    catch (std::exception &e) {
      std::cerr << "get_table_descriptor caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void get_tables(std::vector<std::string> & _return, const TSessionId session)
  {
    try {
      client_.get_tables(_return, session);
    }
    catch (TMapDException &e) {
      throw e;
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.get_tables(_return, session);
      }
      catch (TMapDException &e) {
        throw e;
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
      catch (std::exception &e) {
        std::cerr << "get_tables caught exception: " << e.what() << std::endl;
        TMapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    }
    catch (std::exception &e) {
      std::cerr << "get_tables caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void get_users(std::vector<std::string> & _return)
  {
    try {
      client_.get_users(_return);
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.get_users(_return);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
  }

  void get_databases(std::vector<TDBInfo> & _return)
  {
    try {
      client_.get_databases(_return);
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.get_databases(_return);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
  }

  void get_version(std::string &version) {
    try {
      client_.get_version(version);
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.get_version(version);
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

  void load_table_binary(const TSessionId session, const std::string &table_name, const std::vector<TRow> &rows) {
    try {
      client_.load_table_binary(session, table_name, rows);
    }
    catch (TMapDException &e) {
      throw e;
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.load_table_binary(session, table_name, rows);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
    catch (std::exception &e) {
      std::cerr << "load_table_binary caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void load_table_string(const TSessionId session, const std::string &table_name, const std::vector<TStringRow> &rows) {
    try {
      client_.load_table_string(session, table_name, rows);
    }
    catch (TMapDException &e) {
      throw e;
    }
    catch (TException &te) {
      try {
        transport_.open();
        client_.load_table_string(session, table_name, rows);
      }
      catch (TException &te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
    catch (std::exception &e) {
      std::cerr << "load_table_string caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
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
    ("version,v", "Print MapD Release Version")
    ("server,s", po::value<int>(&server_port), "MapD Server port number (default 9091)")
    ("port,p", po::value<int>(&port), "Port number (default 9090)");

	po::variables_map vm;
	po::positional_options_description positionalOptions;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);
		if (vm.count("help")) {
			std::cout << "Usage: mapd_http_server [{-p|--port} <port number>] [{-s|--server} <port number>][--version|-v]\n";
			return 0;
		}
		if (vm.count("version")) {
			std::cout << "MapD Version: " << MapDRelease << std::endl;
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
