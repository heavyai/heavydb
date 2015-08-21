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
#include <mutex>

#include "MapDRelease.h"

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;

using boost::shared_ptr;

class MapDClientHandle {
 public:
  MapDClientHandle(const int server_port) {
    socket_.reset(new TSocket("localhost", server_port));
    transport_.reset(new TBufferedTransport(socket_));
    protocol_.reset(new TBinaryProtocol(transport_));
    client_.reset(new MapDClient(protocol_));
    transport_->open();
  }

  MapDClient* operator->() const { return client_.get(); }

  void reopen() { transport_->open(); }

 private:
  shared_ptr<TSocket> socket_;
  shared_ptr<TTransport> transport_;
  shared_ptr<TProtocol> protocol_;
  std::unique_ptr<MapDClient> client_;
};

class MapDHandler : virtual public MapDIf {
 public:
  MapDHandler(const int server_port) : server_port_(server_port) {}

  TSessionId connect(const std::string& user, const std::string& passwd, const std::string& dbname) {
    instantiateClient();
    TSessionId session = -1;
    try {
      session = (*client_)->connect(user, passwd, dbname);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        session = (*client_)->connect(user, passwd, dbname);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    } catch (std::exception& e) {
      std::cerr << "connect caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
    return session;
  }

  void disconnect(TSessionId session) {
    instantiateClient();
    try {
      (*client_)->disconnect(session);
    } catch (TException& te) {
      std::cerr << "Thrift exception: " << te.what() << std::endl;
    } catch (std::exception& e) {
      std::cerr << "disconnect caught exception: " << e.what() << std::endl;
    }
  }

  void sql_execute(TQueryResult& _return, const TSessionId session, const std::string& query_str, const bool column_format) {
    instantiateClient();
    try {
      (*client_)->sql_execute(_return, session, query_str, column_format);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->sql_execute(_return, session, query_str, column_format);
      } catch (TMapDException& e) {
        throw e;
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      } catch (std::exception& e) {
        std::cerr << "select caught exception: " << e.what() << std::endl;
        TMapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    } catch (std::exception& e) {
      std::cerr << "select caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void get_table_descriptor(TTableDescriptor& _return, const TSessionId session, const std::string& table_name) {
    instantiateClient();
    try {
      (*client_)->get_table_descriptor(_return, session, table_name);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->get_table_descriptor(_return, session, table_name);
      } catch (TMapDException& e) {
        throw e;
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      } catch (std::exception& e) {
        std::cerr << "get_table_descriptor caught exception: " << e.what() << std::endl;
        TMapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    } catch (std::exception& e) {
      std::cerr << "get_table_descriptor caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void get_row_descriptor(TRowDescriptor& _return, const TSessionId session, const std::string& table_name) {
    instantiateClient();
    try {
      (*client_)->get_row_descriptor(_return, session, table_name);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->get_row_descriptor(_return, session, table_name);
      } catch (TMapDException& e) {
        throw e;
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      } catch (std::exception& e) {
        std::cerr << "get_row_descriptor caught exception: " << e.what() << std::endl;
        TMapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    } catch (std::exception& e) {
      std::cerr << "get_row_descriptor caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void get_frontend_view(std::string& _return, const TSessionId session, const std::string& view_name) {
    instantiateClient();
    try {
      (*client_)->get_frontend_view(_return, session, view_name);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->get_frontend_view(_return, session, view_name);
      } catch (TMapDException& e) {
        throw e;
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      } catch (std::exception& e) {
        std::cerr << "get_view_descriptor caught exception: " << e.what() << std::endl;
        TMapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    } catch (std::exception& e) {
      std::cerr << "get_view_descriptor caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void get_tables(std::vector<std::string>& _return, const TSessionId session) {
    instantiateClient();
    try {
      (*client_)->get_tables(_return, session);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->get_tables(_return, session);
      } catch (TMapDException& e) {
        throw e;
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      } catch (std::exception& e) {
        std::cerr << "get_tables caught exception: " << e.what() << std::endl;
        TMapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    } catch (std::exception& e) {
      std::cerr << "get_tables caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void get_frontend_views(std::vector<std::string>& _return, const TSessionId session) {
    instantiateClient();
    try {
      (*client_)->get_frontend_views(_return, session);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->get_frontend_views(_return, session);
      } catch (TMapDException& e) {
        throw e;
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      } catch (std::exception& e) {
        std::cerr << "get_frontend_views caught exception: " << e.what() << std::endl;
        TMapDException ex;
        ex.error_msg = e.what();
        throw ex;
      }
    } catch (std::exception& e) {
      std::cerr << "get_frontend_views caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void get_users(std::vector<std::string>& _return) {
    instantiateClient();
    try {
      (*client_)->get_users(_return);
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->get_users(_return);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
  }

  void get_databases(std::vector<TDBInfo>& _return) {
    instantiateClient();
    try {
      (*client_)->get_databases(_return);
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->get_databases(_return);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
  }

  void get_version(std::string& version) {
    instantiateClient();
    try {
      (*client_)->get_version(version);
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->get_version(version);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
  }

  void set_execution_mode(const TSessionId session, const TExecuteMode::type mode) {
    instantiateClient();
    try {
      (*client_)->set_execution_mode(session, mode);
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->set_execution_mode(session, mode);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    }
  }

  void load_table_binary(const TSessionId session, const std::string& table_name, const std::vector<TRow>& rows) {
    instantiateClient();
    try {
      (*client_)->load_table_binary(session, table_name, rows);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->load_table_binary(session, table_name, rows);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    } catch (std::exception& e) {
      std::cerr << "load_table_binary caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void load_table(const TSessionId session, const std::string& table_name, const std::vector<TStringRow>& rows) {
    instantiateClient();
    try {
      (*client_)->load_table(session, table_name, rows);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->load_table(session, table_name, rows);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    } catch (std::exception& e) {
      std::cerr << "load_table caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void detect_column_types(TDetectResult& _return,
                           const TSessionId session,
                           const std::string& file_name,
                           const TCopyParams& copy_params) {
    try {
      (*client_)->detect_column_types(_return, session, file_name, copy_params);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->detect_column_types(_return, session, file_name, copy_params);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    } catch (std::exception& e) {
      std::cerr << "detect_column_types caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void render(std::string& _return,
              const TSessionId session,
              const std::string& query,
              const std::string& render_type,
              const TRenderPropertyMap& render_properties,
              const TColumnRenderMap& col_render_properties) {
    instantiateClient();
    try {
      (*client_)->render(_return, session, query, render_type, render_properties, col_render_properties);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->render(_return, session, query, render_type, render_properties, col_render_properties);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    } catch (std::exception& e) {
      std::cerr << "render caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void create_frontend_view(const TSessionId session, const std::string& view_name, const std::string& view) {
    instantiateClient();
    try {
      (*client_)->create_frontend_view(session, view_name, view);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->create_frontend_view(session, view_name, view);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    } catch (std::exception& e) {
      std::cerr << "create_frontend_view caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void create_table(const TSessionId session, const std::string& table_name, const TRowDescriptor& row_desc) {
    try {
      (*client_)->create_table(session, table_name, row_desc);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->create_table(session, table_name, row_desc);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    } catch (std::exception& e) {
      std::cerr << "create_table caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

  void import_table(const TSessionId session,
                    const std::string& table_name,
                    const std::string& file_name,
                    const TCopyParams& copy_params) {
    try {
      (*client_)->import_table(session, table_name, file_name, copy_params);
    } catch (TMapDException& e) {
      throw e;
    } catch (TException& te) {
      try {
        client_->reopen();
        (*client_)->import_table(session, table_name, file_name, copy_params);
      } catch (TException& te1) {
        std::cerr << "Thrift exception: " << te1.what() << std::endl;
        ThriftException thrift_exception;
        thrift_exception.error_msg = te1.what();
        throw thrift_exception;
      }
    } catch (std::exception& e) {
      std::cerr << "import_table caught exception: " << e.what() << std::endl;
      TMapDException ex;
      ex.error_msg = e.what();
      throw ex;
    }
  }

 private:
  void instantiateClient() {
    if (!client_) {
      client_ = new MapDClientHandle(server_port_);
    }
  }
  const int server_port_;
  static __thread MapDClientHandle* client_;
};

__thread MapDClientHandle* MapDHandler::client_{nullptr};

int main(int argc, char** argv) {
  int port = 9090;
  int server_port = 9091;

  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages ")("version,v", "Print MapD Release Version")(
      "server,s", po::value<int>(&server_port), "MapD Server port number (default 9091)")(
      "port,p", po::value<int>(&port), "Port number (default 9090)");

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
  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  shared_ptr<MapDHandler> handler(new MapDHandler(server_port));
  shared_ptr<TProcessor> processor(new MapDProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new THttpServerTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TJSONProtocolFactory());
  TThreadedServer server(processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
  return 0;
}
